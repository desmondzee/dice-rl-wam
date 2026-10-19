import argparse
import json
import os
import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval_config import TASK_IDS, EvalConfig, episode_plan
from script.lingbot_rl_buffer import ChunkReplay, host_bf16
from script.lingbot_rl_config import RLConfig, config_from_dict
from script.lingbot_rl_data import featurize_experts, load_manifest_episodes
from script.lingbot_rl_label import LabelPool
from script.lingbot_rl_model import (
    ACTION_DIM, ACTOR_EVERY, BATCH, GAMMA, GRADIENT_STEPS, HORIZON, DiceResidualModel, apply_residual, mlp_float,
)
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, load_residual_policy, slice_env_actions,
)
from script.lingbot_rl_pose import Curriculum, PoseOverride
from script.lingbot_rl_progress import calibrate, progress, shaping_reward
from script.lingbot_rl_sharpen import bin_entropy, log_std
from script.lingbot_rl_standardize import expert_standardization
from script.lingbot_rl_tokens import TokenStore

LiberoEnv = None
SHARPEN_EVERY_STEPS = 5_000
FIDELITY_CHUNKS = 20
FIDELITY_POSITION_M = 1e-3
FIDELITY_ANGLE_RAD = 0.01745
SHADOW_MATCH_M = 1e-2
RESUME_KEYS = {
    "actor", "critic", "target_critic", "plan_critic", "layout", "actor_opt", "critic_opt", "plan_opt",
    "actor_lr", "critic_lr", "plan_lr", "replay", "rng", "env_steps", "chunks", "recipe", "checkpointed", "wandb_id",
    "episodes", "select_critic", "select_opt", "select_lr", "warmup_end_steps", "curriculum",
}


def _env_class():
    global LiberoEnv
    if LiberoEnv is None:
        from lerobot.envs.libero import LiberoEnv as cls
        LiberoEnv = cls
    return LiberoEnv


def make_env(task_id, suite=None, init_states=True, plus=False, proprio=False):
    return _env_class()(
        task_suite=suite, task_id=task_id, task_suite_name="libero_10",
        episode_length=520, observation_height=128, observation_width=128,
        obs_type="pixels_agent_pos" if proprio else "pixels",
        init_states=init_states, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative",
        hard_reset=True, is_libero_plus=plus,
    )


def _plus_suite(base):
    from lerobot.envs.libero import _get_suite
    from libero.libero.benchmark import Task
    suite = _get_suite("libero_10")
    suite.tasks.append(Task(name=base["name"], language=base["instruction"], problem="Libero", problem_folder="libero_10",
                            bddl_file=f"{base['name']}.bddl", init_states_file=f"{base['name']}.pruned_init"))
    suite.n_tasks = len(suite.tasks)
    return suite, suite.n_tasks - 1


RESET_KINDS = ("standard", "robot_init", "layout")


def draw_reset(config, episode, cap=None):
    if not config.plus_train:
        return 0, None, 0.0, None
    rng = np.random.default_rng([config.seed, episode])
    kind = int(rng.integers(3))
    if kind == 0:
        return 0, None, float(rng.random()), None
    variants = config.plus_train[("robot_init", "layout")[kind - 1]]
    variant = int(variants[rng.integers(len(variants))])
    draw = float(rng.random())
    procedural = config.procedural() if kind == 1 else None
    if procedural is None:
        return kind, variant, draw, None
    from script.lingbot_rl_pose import procedural_offset
    return kind, None, draw, procedural_offset(rng, procedural["magnitude_lo"], procedural["cap_start"] if cap is None else cap)


def proprio_from_observation(observation):
    state = observation["robot_state"]
    quat = np.asarray(state["eef"]["quat"], dtype=np.float64).reshape(4)
    w = min(max(float(quat[3]), -1.0), 1.0)
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    axis_angle = np.zeros(3) if den <= 1e-10 else quat[:3] * (2.0 * np.arccos(w) / den)
    return np.concatenate([np.asarray(state["eef"]["pos"], dtype=np.float64).reshape(3), axis_angle,
                           np.asarray(state["gripper"]["qpos"], dtype=np.float64).reshape(2)]).astype(np.float32)


def rl_observation_batch(observation, instruction, device, proprio):
    from script.lingbot_eval import observation_batch
    batch = observation_batch(observation, instruction, device)
    if proprio:
        batch["proprio"] = torch.from_numpy(proprio_from_observation(observation))[None].to(device)
    return batch


def run_rl_episode(policy, env, entry, instruction, norm, device, proprio, video_writer=None):
    from script.lingbot_eval import decode_action, run_episode, seed_all, video_frame
    if not proprio:
        return run_episode(policy, env, entry, instruction, norm, device, video_writer=video_writer)
    started = time.perf_counter()
    seed_all(entry["seed"])
    env.init_state_id = entry["init_state_id"]
    observation, _ = env.reset(seed=entry["seed"])
    policy.reset()
    seed_all(entry["seed"])
    if video_writer is not None:
        video_writer.append_data(video_frame(observation))
    success = terminated = truncated = False
    with torch.inference_mode():
        for step in range(1, 521):
            normalized = policy.select_action(rl_observation_batch(observation, instruction, device, True))
            observation, _, terminated, truncated, info = env.step(decode_action(normalized, norm))
            success = env_success(info)
            if video_writer is not None:
                video_writer.append_data(video_frame(observation))
            if success or terminated or truncated:
                break
    truncated = bool(truncated or (step == 520 and not success and not terminated))
    return {**entry, "instruction": instruction, "success": success, "policy_steps": step, "terminated": bool(terminated),
            "truncated": truncated, "seconds": time.perf_counter() - started}


def object_layout(inner):
    model = inner.sim.model
    bodies = sorted(inner.obj_body_id)
    joints = []
    for name in bodies:
        root = inner.obj_body_id[name]
        for joint in range(model.njnt):
            if model.jnt_type[joint] not in (2, 3):
                continue
            body = model.jnt_bodyid[joint]
            while body > 0 and body != root:
                body = model.body_parentid[body]
            if body == root:
                joints.append(model.joint_id2name(joint))
    return {"bodies": bodies, "joints": joints, "body_quaternion": "wxyz"}


def object_state(inner, layout):
    data, model = inner.sim.data, inner.sim.model
    parts = [np.concatenate([data.body_xpos[inner.obj_body_id[name]], data.body_xquat[inner.obj_body_id[name]]])
             for name in layout["bodies"]]
    parts.append(np.asarray([data.qpos[model.get_joint_qpos_addr(name)] for name in layout["joints"]], dtype=np.float64))
    return np.concatenate(parts).astype(np.float32)


def make_shadow(task_id, suite, init_states, plus, seed):
    shadow = make_env(task_id, suite, init_states=init_states, plus=plus)
    rng = capture_rng()
    try:
        shadow.reset(seed=seed)
    finally:
        restore_rng(rng)
    inner = shadow._env.env
    inner.ignore_done = True
    for observable in inner._observables.values():
        if observable.modality == "image":
            observable.set_enabled(False)
    return shadow


def capture_live(live):
    return {"state": live.sim.get_state(), "warmstart": np.array(live.sim.data.qacc_warmstart),
            "robots": [(np.array(robot.controller.goal_pos), np.array(robot.controller.goal_ori),
                        np.array(robot.controller.initial_joint), np.array(robot.gripper.current_action)) for robot in live.robots]}


def sync_shadow(snapshot, twin):
    twin.sim.set_state(snapshot["state"])
    twin.sim.data.qacc_warmstart[:] = snapshot["warmstart"]
    twin.sim.forward()
    for robot, (goal_pos, goal_ori, initial_joint, gripper_action) in zip(twin.robots, snapshot["robots"]):
        robot.controller.update(force=True)
        robot.controller.goal_pos = np.array(goal_pos)
        robot.controller.goal_ori = np.array(goal_ori)
        robot.controller.initial_joint = np.array(initial_joint)
        robot.gripper.current_action = np.array(gripper_action)


def label_plans(shadow, snapshot, before, candidates, first_chunk, norm, layout, calibration):
    from script.lingbot_eval import decode_action
    twin = shadow._env.env
    plans, k = candidates.shape[:2]
    env_actions = slice_env_actions(candidates.reshape(plans * k, HORIZON, ACTION_DIM), first_chunk)
    labels = np.zeros(plans * k, np.float32)
    bodies = len(layout["bodies"])
    positional = np.r_[np.arange(7 * bodies).reshape(bodies, 7)[:, :3].reshape(-1), np.arange(7 * bodies, before.shape[0])]
    for index in range(plans * k):
        sync_shadow(snapshot, twin)
        if index == 0:
            gap = float(np.abs(before[positional] - object_state(twin, layout)[positional]).max())
            if gap > SHADOW_MATCH_M:
                raise RuntimeError(f"shadow state mismatch {gap:.4f}")
        success = False
        for step in range(env_actions.shape[1]):
            _, _, done, _ = twin.step(decode_action(env_actions[index, step].reshape(1, 7), norm))
            success = bool(twin._check_success())
            if success or done:
                break
        labels[index] = progress(calibration, layout, before, object_state(twin, layout), success)
    return labels.reshape(plans, k)


def shadow_fidelity(env, shadow, layout, act, observation, chunks=FIDELITY_CHUNKS):
    live, twin = env._env.env, shadow._env.env
    bodies = 7 * len(layout["bodies"])
    worst = np.zeros(2)
    for chunk in range(chunks):
        sync_shadow(capture_live(live), twin)
        for _ in range(env_action_count(chunk == 0)):
            action = act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            twin.step(action)
        live_state, twin_state = object_state(live, layout), object_state(twin, layout)
        poses = np.stack([live_state[:bodies].reshape(-1, 7), twin_state[:bodies].reshape(-1, 7)]).astype(np.float64)
        position = np.abs(poses[0, :, :3] - poses[1, :, :3]).max()
        angle = (2.0 * np.arccos(np.minimum(1.0, np.abs((poses[0, :, 3:] * poses[1, :, 3:]).sum(axis=1))))).max()
        joint = np.abs(live_state[bodies:] - twin_state[bodies:]).max(initial=0.0)
        worst = np.maximum(worst, [position, max(angle, joint)])
        if env_success(info) or terminated or truncated:
            break
    result = {"position_m": float(worst[0]), "angle_rad": float(worst[1]), "chunks": chunk + 1}
    if worst[0] >= FIDELITY_POSITION_M or worst[1] >= FIDELITY_ANGLE_RAD:
        raise RuntimeError(f"shadow fidelity check failed: {result}")
    return result


def sharpening_metrics(model, state, a_base, action):
    if a_base.ndim == 4:
        batch, candidates, horizon, dim = a_base.shape
        state = state.unsqueeze(1).expand(batch, candidates, -1).reshape(batch * candidates, -1)
        a_base = a_base.reshape(batch * candidates, horizon, dim)
        action = action.reshape(batch * candidates, horizon, dim)
    with torch.no_grad():
        delta_v = float((model.select_critic(state, action) - model.select_critic(state, a_base)).mean())
        delta_h = histogram_entropy(a_base) - histogram_entropy(action)
    return {"delta_h": delta_h, "delta_v": delta_v}


def replay_sharpening(model, sample):
    online = mlp_float(sample["is_expert"]).reshape(-1) == 0
    if int(online.sum()) < 2:
        return {}
    state = model.state_input(sample["s"], sample.get("state_struct"), sample.get("proprio"))[online]
    z_all, base_all = mlp_float(sample["z_all"])[online], mlp_float(sample["a_base_all"])[online]
    pool = model.plan_input(sample.get("plan_pool"), sample.get("plan_struct"))
    batch, k = z_all.shape[:2]
    state_k = state.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, -1)
    pool_k = None if pool is None else pool[online].unsqueeze(1).expand(batch, k, -1).reshape(batch * k, -1)
    critic_k = model.critic_state(state_k, pool_k)
    base_flat = base_all.reshape(batch * k, HORIZON, ACTION_DIM)
    with torch.no_grad():
        action = apply_residual(base_flat, model.actor(state_k, z_all.reshape(batch * k, HORIZON, ACTION_DIM), base_flat))
        delta_v = float((model.select_critic(critic_k, action) - model.select_critic(critic_k, base_flat)).mean())
        before, after = base_all.cpu().numpy(), action.reshape(batch, k, HORIZON, ACTION_DIM).cpu().numpy()
        metrics = {"replay_delta_v": delta_v, "replay_delta_h": float((bin_entropy(before) - bin_entropy(after)).mean()),
                   "replay_delta_log_std": float((log_std(after) - log_std(before)).mean())}
        if model.plan_critic is not None:
            pools_all = model.plan_input(sample["plan_pools_all"], sample.get("plan_struct_all"))[online]
            plans = pools_all.shape[1]
            v_plan = model.plan_critic(state.unsqueeze(1).expand(batch, plans, -1).reshape(batch * plans, -1),
                                       pools_all.reshape(batch * plans, -1)).reshape(batch, plans)
            metrics["replay_plan_value_spread"] = float((v_plan.max(dim=1).values - v_plan.min(dim=1).values).mean())
            index = sample["plan_index"].reshape(-1)[online]
            executed = pools_all[torch.arange(batch, device=pools_all.device), index]
            metrics["replay_plan_delta_v"] = float((v_plan.gather(1, index.reshape(-1, 1)).squeeze(1) - v_plan.mean(dim=1)).mean())
            metrics["replay_plan_delta_h"] = float((torch.log(pools_all.reshape(batch * plans, -1).std(dim=0) + 1e-6)
                                                    - torch.log(executed.std(dim=0) + 1e-6)).mean())
    return metrics


def recipe_of(config):
    return config.protocol()


def due_points(env_steps, schedule, done):
    return [point for point in schedule if env_steps >= point and point not in done]


def _ingest_experts(buffer, rows, task_ids):
    for row in rows:
        if int(row["task_id"]) not in task_ids:
            continue
        buffer.add_expert(row)
        if float(row["done"]) == 1.0:
            buffer.finalize_episode()


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_inference(path, model):
    payload = model.inference_state_dict()
    if set(payload) - {"actor", "critic", "target_critic", "plan_critic", "select_critic", "layout"}:
        raise ValueError("Inference dump may only contain actor and critic weights")
    _atomic_torch(path, payload)


def checkpoint_dir(output_dir, env_steps):
    return Path(output_dir) / "checkpoints" / f"step_{int(env_steps):06d}"


def save_inference_checkpoints(output_dir, env_steps, model):
    save_inference(Path(output_dir) / "residual.pt", model)
    save_inference(checkpoint_dir(output_dir, env_steps) / "residual.pt", model)


def _cpu_byte_rng(state):
    if isinstance(state, (list, tuple)):
        return [_cpu_byte_rng(item) for item in state]
    if torch.is_tensor(state):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(state, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(state, dtype=np.uint8))
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def capture_rng():
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": [],
    }
    if torch.cuda.is_available():
        rng["cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return rng


def restore_rng(rng):
    torch.set_rng_state(_cpu_byte_rng(rng["torch"]))
    np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(_cpu_byte_rng(rng["cuda"]))


def save_resume(path, model, buffer, env_steps, chunks, recipe, checkpointed=(), wandb_id=None, episodes=0, warmup_end_steps=None,
                curriculum=None):
    payload = {
        **model.resume_state_dict(),
        "replay": buffer.state_dict(),
        "rng": capture_rng(),
        "env_steps": int(env_steps),
        "chunks": int(chunks),
        "recipe": recipe,
        "checkpointed": sorted(int(point) for point in checkpointed),
        "wandb_id": wandb_id,
        "episodes": int(episodes),
        "warmup_end_steps": None if warmup_end_steps is None else int(warmup_end_steps),
        "curriculum": None if curriculum is None else curriculum.state_dict(),
    }
    if "transformer" in payload or set(payload) - RESUME_KEYS:
        raise ValueError("Resume payload includes disallowed keys")
    _atomic_torch(path, payload)


def load_resume(path, model, buffer, recipe, curriculum=None):
    path = Path(path)
    try:
        payload = torch.load(path, map_location=model.device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load resume checkpoint {path}: {exc}") from exc
    if payload.get("recipe") != recipe:
        raise ValueError("Resume recipe fingerprint mismatch")
    if "transformer" in payload:
        raise ValueError("Resume file contains frozen transformer weights")
    model.load_resume_state_dict(payload)
    buffer.load_state_dict(payload["replay"])
    if payload.get("rng"):
        restore_rng(payload["rng"])
    checkpointed = {int(point) for point in payload.get("checkpointed", ())}
    if curriculum is not None and payload.get("curriculum"):
        curriculum.load_state_dict(payload["curriculum"])
    return (int(payload["env_steps"]), int(payload.get("chunks", 0)), checkpointed, payload.get("wandb_id"), int(payload.get("episodes", 0)),
            payload.get("warmup_end_steps"))


def _read_prepared(path):
    path = Path(path) if path is not None else None
    if path is None or not path.is_file():
        return {
            "tasks": [
                {"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50}
                for task_id in TASK_IDS
            ],
            "normalization": {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23},
            "checkpoint": None,
            "model_path": None,
            "architecture": {},
            "assets_path": None,
            "source_run": "libero30-sft",
            "checkpoint_step": 600,
        }
    from script.lingbot_eval import read_json
    return read_json(path)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _host_array(tensor):
    return tensor.detach().float().contiguous().cpu().numpy()


def _timed(fn, *args):
    started = time.perf_counter()
    return fn(*args), time.perf_counter() - started


def env_success(info):
    if "is_success" not in info or not isinstance(info["is_success"], (bool, np.bool_)):
        raise ValueError("Environment must report a boolean is_success explicitly")
    return bool(info["is_success"])


def _to_model_device(sample, device):
    moved = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.float()
        moved[key] = value
    return moved


def _sample_batch(buffer, expert_ratio, device, *extra):
    return _to_model_device(buffer.sample(min(BATCH, len(buffer)), expert_ratio, *extra), device)


def _update_from_buffer(model, buffer, expert_ratio, device, filter_active, actor_active=True):
    info = {}
    extra = (("object_state", "object_state_n"),) if model.privileged else ()
    for step in range(GRADIENT_STEPS):
        sample = _sample_batch(buffer, expert_ratio, device, *extra)
        state = model.state_input(sample["s"], sample["state_struct"], sample["proprio"])
        next_state = model.state_input(sample["s_next"], sample["state_struct_next"], sample["proprio_next"])
        plan_in = model.plan_input(sample["plan_pool"], sample["plan_struct"])
        plan_in_next = model.plan_input(sample["plan_pool_next"], sample["plan_struct_next"])
        critic_state = model.critic_state(state, plan_in, sample.get("object_state"))
        next_critic_state = model.critic_state(next_state, plan_in_next, sample.get("object_state_n"))
        select = () if model.select_critic is model.critic else (model.critic_state(state, plan_in),)
        target = model.n_step_target(
            sample["reward"], sample["done"], next_critic_state,
            sample["z_next_all"], sample["a_base_next_all"], sample["n_steps"])
        if actor_active and (step + 1) % ACTOR_EVERY == 0:
            info.update(model.update_actor(
                state, critic_state, sample["a"], sample["z_all"], sample["a_base_all"], sample["is_expert"],
                sample["mc_return"], filter_active))
        info.update(model.update_critic(critic_state, sample["a"], target, *select))
        if model.plan_critic is not None:
            info.update(model.update_plan_critic(
                state, model.plan_input(sample["plan_pools_all"], sample["plan_struct_all"]), sample["z_all"], sample["plan_a_base_all"],
                sample["plan_labels"], sample["is_expert"]))
        if (step + 1) % ACTOR_EVERY == 0:
            model.polyak_update()
    return info


def _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes, proprio=False):
    plan_cfg = EvalConfig(stage="heldout", seed=42)
    rows = []
    for task in tasks:
        plan = episode_plan(plan_cfg, task["task_id"], plan_cfg.episodes_per_task)[:episodes]
        env = make_env(task["env_task"], suite, init_states=False, proprio=proprio)
        try:
            for entry in plan:
                row = run_rl_episode(policy, env, entry, task["instruction"], norm, device, proprio)
                rows.append(row)
                _write_json(
                    checkpoint_dir(output_dir, env_steps) / "train_eval" / f"task_{task['task_id']:02d}_{entry['episode_index']:02d}.json",
                    row,
                )
        finally:
            env.close()
    per_task = {
        str(task["task_id"]): float(np.mean([row["success"] for row in rows if row["task_id"] == task["task_id"]]))
        for task in tasks
    }
    return {"env_steps": env_steps, "per_task_success": per_task,
            "macro_success_rate": float(np.mean([row["success"] for row in rows]))}


def _maybe_sharpen(model, policy, batch, device):
    if policy.eval_plans > 1:
        decoded = policy.select_plan(batch, policy.eval_plans, 8, rank_fn=model.rank_plans)
    else:
        decoded = policy.decode_candidates(batch, k=8)
    state = model.state_input(decoded["s"].to(device), decoded.get("state_struct"), batch.get("proprio"))
    pool = model.plan_input(decoded["plan_pool"].to(device), decoded.get("plan_struct")) if policy.eval_plans > 1 else None
    a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
    noise = decoded["z"].to(device=device, dtype=torch.float32)
    pooled = state
    if state.shape[0] == 1 and a_base.shape[0] > 1:
        state = state.expand(a_base.shape[0], -1)
    with torch.no_grad():
        action = apply_residual(a_base, model.actor(state, noise, a_base))
    if a_base.ndim == 3:
        a_base = a_base.unsqueeze(0)
        action = action.unsqueeze(0)
    else:
        pooled = state
    critic_state = model.critic_state(pooled, None if pool is None else pool.expand(pooled.shape[0], -1))
    return sharpening_metrics(model, critic_state, a_base, action)


def train(config=None, prepared_path=None, output_dir=None, run_name=None, resume=False,
          commit=None, max_env_steps=None, dataset_root=None, demos_root=None):
    import wandb
    from script.lingbot_eval import decode_action, observation_batch, seed_all

    config = (config or RLConfig()).validate()
    if (config.privileged_critic or config.shaping) and demos_root is None:
        raise ValueError("demos_root is required for privileged critic or shaping")
    recipe = recipe_of(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _read_prepared(prepared_path)
    if prepared.get("source_run") not in (None, config.source_run) or prepared.get("checkpoint_step") not in (None, config.checkpoint_step):
        raise ValueError("Prepared checkpoint does not match the pinned RL recipe")
    device = _device()
    runtime = {"cuda": torch.version.cuda, "machine": platform.machine(),
               "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()) if device == "cuda" else None}
    if device == "cuda" and runtime["machine"] == "aarch64" and not str(runtime["cuda"]).startswith("13"):
        raise RuntimeError("GB10 hosts need a CUDA 13 torch build (DICE_TORCH_INDEX)")
    if device == "cuda":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    seed_all(config.seed)
    architecture = prepared.get("architecture") or {}
    if prepared.get("checkpoint"):
        from script.lingbot_eval import read_checkpoint_metadata, read_json
        metadata = read_checkpoint_metadata(prepared["checkpoint"])
        architecture = metadata["architecture"]
        norm = metadata["normalization"]
    else:
        norm = prepared.get("normalization") or {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}
    policy = load_residual_policy(prepared.get("checkpoint"), prepared.get("model_path"), architecture, config.video_exec_step,
                                  config.video_steps, config.action_steps)
    model_kwargs = {"device": device, "residual_input": config.residual_input, "epsilon": config.epsilon,
                    "plan_candidates": config.plan_candidates, "state_features": config.state_features, "proprio": config.proprio,
                    "plan_target": config.plan_target, "residual_bound": config.residual_bound, "bc_weight": config.bc_weight,
                    "privileged_critic": config.privileged_critic}
    model = None if config.standardize_features else DiceResidualModel(**model_kwargs)
    policy.residual_model = model
    policy.eval_candidates = config.k_candidates
    policy.eval_plans = config.plan_candidates
    buffer = ChunkReplay(plans=config.plan_candidates)
    tasks = prepared["tasks"]
    expert_rows, expert_episodes = [], []
    if prepared.get("checkpoint") and (not resume or config.standardize_features):
        cache_path = output_dir / "expert_features.pt"
        manifest = read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json")
        if dataset_root:
            task_to_id = {task["instruction"]: task["task_id"] for task in tasks}
            expert_episodes = load_manifest_episodes(dataset_root, manifest, task_to_id, norm)
        if expert_episodes or cache_path.is_file():
            expert_rows = [row for row in featurize_experts(policy, expert_episodes, manifest, norm, cache_path)
                           if int(row["task_id"]) in config.task_ids]
    suite = None
    if prepared.get("assets_path") and not config.plus_train:
        from script.lingbot_eval import describe_suite
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = suite_tasks
    tasks = [{**task, "env_task": task["task_id"]} for task in tasks if task["task_id"] in config.task_ids]
    if config.plus_train:
        suite, tasks[0]["env_task"] = _plus_suite(tasks[0])
    layout = calibration = fidelity = None
    if config.proprio:
        rng = capture_rng()
        env = make_env(tasks[0]["env_task"], suite, init_states=False, proprio=True)
        try:
            observation, _ = env.reset(seed=config.seed)
            layout = object_layout(env._env.env)
            if config.plan_target == "rollout" or config.shaping or config.diagnostic_label_every > 0:
                calibration = calibrate(tasks[0]["task_id"], env._env.env, layout, object_state(env._env.env, layout))
            if config.plan_target == "rollout":
                act = lambda current: decode_action(
                    policy.select_action(rl_observation_batch(current, tasks[0]["instruction"], device, True)), norm)
                shadow = make_shadow(tasks[0]["env_task"], suite, False, False, config.seed)
                try:
                    policy.reset()
                    fidelity = {"base": shadow_fidelity(env, shadow, layout, act, observation), "plus": None}
                finally:
                    shadow.close()
                    policy.reset()
        finally:
            env.close()
        if fidelity is not None and config.plus_train:
            variant = int(config.plus_train["layout"][0])
            env, shadow = make_env(variant, suite, init_states=True, plus=True, proprio=True), make_shadow(variant, suite, True, True, config.seed)
            try:
                env.init_state_id = 0
                observation, _ = env.reset(seed=config.seed)
                policy.reset()
                fidelity["plus"] = {"variant_index": variant, **shadow_fidelity(env, shadow, layout, act, observation, chunks=3)}
            finally:
                env.close()
                shadow.close()
                policy.reset()
        restore_rng(rng)
    expert_states = None
    if expert_rows and (config.privileged_critic or config.shaping):
        from script.lingbot_rl_expert_state import attach_object_states, expert_object_states
        rng = capture_rng()
        states = expert_object_states(demos_root, tasks[0]["name"], suite, tasks[0]["env_task"],
                                      [episode for episode in expert_episodes if episode["task_id"] in config.task_ids], norm,
                                      output_dir / "expert_object_states.pt", layout)
        restore_rng(rng)
        attach_object_states(expert_rows, states)
        expert_states = {"demos": int(sum(float(row["done"]) == 1.0 for row in expert_rows))}
        if config.shaping:
            for row in expert_rows:
                row["reward"] = np.float32(float(row["reward"]) + shaping_reward(
                    calibration, layout, row["object_state"], row["object_state_next"], GAMMA))
    if model is None:
        standardize = expert_standardization(expert_rows) if expert_rows else None
        model = DiceResidualModel(**model_kwargs, standardize=standardize)
        policy.residual_model = model
    if not resume:
        _ingest_experts(buffer, expert_rows, config.task_ids)
    procedural = config.procedural()
    pose_override = curriculum = None
    if procedural is not None:
        pose_override = PoseOverride()
        curriculum = Curriculum(procedural["cap_start"], procedural["cap_end"], procedural["trigger_success"], procedural["trigger_window"])
    env_steps = 0
    chunks = 0
    episodes = 0
    warmup_end_steps = None
    checkpointed = set()
    wandb_id = None
    resume_path = output_dir / "resume" / "latest.pt"
    if resume:
        env_steps, chunks, checkpointed, wandb_id, episodes, warmup_end_steps = load_resume(resume_path, model, buffer, recipe, curriculum)
    tokens = TokenStore(output_dir / "tokens", chunks) if config.store_tokens else None
    if max_env_steps is None and env_steps == 0:
        if not any(float(row["is_expert"]) == 1.0 for row in buffer.rows()):
            raise RuntimeError("Expert features missing; RLPD requires the SFT demos")
    _write_json(output_dir / "settings.json", {"config": config.to_dict(), "recipe": recipe, "runtime": runtime, "object_layout": layout,
                                               "progress_calibration": calibration, "shadow_fidelity": fidelity,
                                               "standardize": model.standardize_layout(), "expert_object_states": expert_states,
                                               "procedural_robot_init": procedural})
    wandb_kwargs = {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "name": run_name or config.default_run_name,
        "config": config.to_dict(),
    }
    if resume and wandb_id:
        wandb_kwargs["id"] = wandb_id
        wandb_kwargs["resume"] = "must"
    elif resume:
        wandb_kwargs["resume"] = "allow"
    run = wandb.init(**wandb_kwargs)
    wandb_id = getattr(run, "id", None) or wandb_id
    smoke = max_env_steps is not None
    budget = max_env_steps if smoke else config.online_env_steps
    min_online_rows = 1 if smoke else BATCH
    next_sharpen = 0
    chunk_seconds, chunk_actions, plan_info, chunk_info = 0.0, 0, {}, {}
    eval_points = set(config.train_eval_schedule())

    def maybe_checkpoint():
        due = due_points(env_steps, config.checkpoint_schedule(), checkpointed)
        if not due or smoke:
            return
        checkpointed.update(due)
        point = max(due)
        save_inference_checkpoints(output_dir, point, model)
        if point in eval_points:
            policy.eval_candidates = 1 if env_steps < config.selection_warmup_steps else config.k_candidates
            rng = capture_rng()
            summary = _train_eval(policy, tasks, norm, device, suite, point, output_dir, config.train_eval_episodes, config.proprio)
            restore_rng(rng)
            logged = {"train_eval/macro_success_rate": summary["macro_success_rate"], "env_steps": env_steps}
            for task_id, success in summary["per_task_success"].items():
                logged[f"train_eval/task_{task_id}_success"] = success
            env = make_env(tasks[0]["env_task"], suite, init_states=False, proprio=config.proprio)
            try:
                observation, _ = env.reset(seed=0)
                policy.reset()
                logged.update(_maybe_sharpen(
                    model, policy, rl_observation_batch(observation, tasks[0]["instruction"], device, config.proprio), device))
            finally:
                env.close()
                policy.reset()
            run.log(logged)
        save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id, episodes, warmup_end_steps, curriculum)
        if commit is not None:
            commit(point)

    maybe_checkpoint()
    features = config.proprio or bool(config.plus_train) or config.state_features == "struct"
    labeller = ThreadPoolExecutor(max_workers=1) if tokens is not None else None
    pool = None
    if config.plan_target == "rollout" or config.diagnostic_label_every > 0:
        spec = None if suite is None else {"plus": bool(config.plus_train), "base": tasks[0], "assets_path": prepared.get("assets_path")}
        pool = LabelPool(config.label_workers, spec, norm, layout, calibration)
    try:
        while env_steps < budget:
            task = tasks[int(np.random.randint(len(tasks)))]
            task_id = task["task_id"]
            kind, variant, draw, offset = draw_reset(config, episodes, cap=curriculum.cap if curriculum else None)
            env_task = task_id if variant is None else variant
            stored = bool(config.plus_train) and kind != 1
            sim_task = task["env_task"] if variant is None else variant
            residual_active = episodes >= config.actor_warmup_episodes
            if residual_active and warmup_end_steps is None:
                warmup_end_steps = env_steps
            if offset is not None:
                pose_override.set(offset)
            env = make_env(sim_task, suite, init_states=stored, plus=bool(config.plus_train), proprio=config.proprio)
            if stored:
                env.init_state_id = int(draw * len(env._init_states))
            policy.reset()
            episode_seed = config.seed + env_steps
            observation, _ = env.reset(seed=episode_seed)
            future = token_future = None
            episode_return = 0.0
            episode_success = 0.0
            episode_length = 0
            saw_success = False
            try:
                while env_steps < budget and episode_length < 520:
                    batch = rl_observation_batch(observation, task["instruction"], device, config.proprio)
                    sim_state = object_state(env._env.env, layout) if layout is not None else None
                    started = time.perf_counter()
                    plans = config.plan_candidates
                    if plans > 1:
                        random_plan = env_steps < config.selection_warmup_steps or np.random.random() < config.plan_random_prob
                        decoded = policy.select_plan(batch, plans, config.k_candidates, rank_fn=model.rank_plans,
                                                     random_plan=random_plan, decode_all=True, return_tokens=tokens is not None)
                    else:
                        decoded = policy.decode_candidates(batch, k=config.k_candidates)
                    if config.plan_target == "rollout":
                        future = pool.submit(capture_live(env._env.env), sim_state, decoded["plan_a_base_all"].detach().float().cpu().numpy(),
                                             decoded["first_chunk"], sim_task, stored, bool(config.plus_train), episode_seed)
                    if tokens is not None:
                        token_future = labeller.submit(tokens.append, decoded["plan_tokens"], chunk=chunks, episode=episodes, step=env_steps,
                                                       reset_kind=kind, variant_index=env_task, plan_index=int(decoded["plan_index"]))
                    state = model.state_input(decoded["s"].to(device), decoded.get("state_struct"), batch.get("proprio"))
                    noise = decoded["z"].to(device=device, dtype=torch.float32)
                    a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
                    plan_in = model.plan_input(decoded["plan_pool"].to(device), decoded.get("plan_struct")) if plans > 1 else None
                    k = a_base.shape[0]
                    if k != config.k_candidates:
                        raise RuntimeError("action candidate batching failed")
                    ramp_p = 0.0 if not residual_active else 1.0 if config.residual_ramp_steps == 0 else min(
                        max(env_steps - warmup_end_steps, 0) / config.residual_ramp_steps, 1.0)
                    applied = residual_active and (ramp_p >= 1.0 or np.random.random() < ramp_p)
                    q_d = None
                    with torch.no_grad():
                        state_k = state.expand(k, -1)
                        executed = apply_residual(a_base, model.actor(state_k, noise, a_base)) if applied else a_base
                        if env_steps < config.selection_warmup_steps:
                            star = int(np.random.randint(k))
                        else:
                            pool_k = None if plan_in is None else plan_in.expand(k, -1)
                            q_d = model.select_critic(model.critic_state(state_k, pool_k), executed).reshape(-1)
                            star = int(q_d.argmax())
                            if model.privileged:
                                object_k = torch.from_numpy(sim_state).to(device).expand(k, -1)
                                q_p = model.critic(model.critic_state(state_k, pool_k, object_k), executed).reshape(-1)
                                chunk_info["select_agreement"] = float(star == int(q_p.argmax()))
                        if plans > 1:
                            plans_in = model.plan_input(decoded["plan_pools"].to(device), decoded.get("plan_struct_all"))
                            all_base = decoded["plan_a_base_all"].to(device=device, dtype=torch.float32)
                            flat_state = state.expand(plans * k, -1)
                            flat_base = all_base.reshape(plans * k, HORIZON, ACTION_DIM)
                            critic_all = model.critic_state(flat_state, plans_in.repeat_interleave(k, dim=0))
                            action_all = apply_residual(flat_base, model.actor(flat_state, noise.repeat(plans, 1, 1), flat_base))
                            q_all = model.select_critic(critic_all, action_all).reshape(plans, k).max(dim=1).values
                            v_all = model.plan_critic(state.expand(plans, -1), plans_in).reshape(-1)
                            plan_info = {"plan_agreement": float(int(v_all.argmax()) == int(q_all.argmax())),
                                         "plan_regret": float(q_all.max() - q_all[decoded["plan_index"]]),
                                         "plan_value_spread": float(v_all.max() - v_all.min()),
                                         "plan_random_rate": float(random_plan)}
                            if future is not None:
                                q_mean = torch.stack(model.select_critic(critic_all, action_all, return_all=True)).mean(dim=0).reshape(plans, k).max(dim=1).values
                    if config.plan_target != "rollout" and config.diagnostic_label_every > 0 and chunks % config.diagnostic_label_every == 0:
                        future = pool.submit(capture_live(env._env.env), sim_state, np.stack([_host_array(executed), _host_array(a_base)]),
                                             decoded["first_chunk"], sim_task, stored, bool(config.plus_train), episode_seed)
                    chosen = executed[star:star + 1]
                    policy.commit_executed(chosen.cpu(), first_chunk=decoded["first_chunk"])
                    env_actions = slice_env_actions(chosen.cpu(), decoded["first_chunk"])
                    n_env = env_actions.shape[1]
                    chunk_reward = 0.0
                    done = 0.0
                    executed_n = 0
                    for index in range(n_env):
                        action = decode_action(env_actions[:, index, :].reshape(1, 7), norm)
                        observation, _, terminated, truncated, info = env.step(action)
                        policy.observe_env_step(observation_batch(observation, task["instruction"], device))
                        executed_n += 1
                        episode_length += 1
                        env_steps += 1
                        if env_success(info):
                            chunk_reward = 1.0
                            saw_success = True
                            episode_success = 1.0
                        if saw_success or terminated or truncated or episode_length >= 520:
                            done = 1.0
                            break
                        if env_steps >= budget:
                            break
                    sim_state_next = object_state(env._env.env, layout) if layout is not None else None
                    shaped = shaping_reward(calibration, layout, sim_state, sim_state_next, GAMMA) if config.shaping else 0.0
                    chunk_reward += shaped
                    chunk_info.update({"residual_applied_rate": float(applied), "residual_ramp_p": ramp_p, "shaping_reward": shaped,
                                       "pose_cap": curriculum.cap if curriculum else 0.0,
                                       "pose_offset_l2": 0.0 if offset is None else float(np.linalg.norm(offset))})
                    plan_labels = None
                    if future is not None and config.plan_target != "rollout":
                        labels, label_seconds = future.result()
                        future = None
                        chunk_info.update({"residual_label_gain": float(labels[0].mean() - labels[1].mean()), "label_seconds": label_seconds})
                        if q_d is not None:
                            chunk_info["q_label_agreement"] = float(int(q_d.argmax()) == int(labels[0].argmax()))
                    if future is not None:
                        wait_started = time.perf_counter()
                        plan_labels, label_seconds = future.result()
                        label_wait_seconds, future = time.perf_counter() - wait_started, None
                        plan_mean = plan_labels.mean(axis=1)
                        plan_info.update({
                            "label_agreement_with_q": float(int(plan_mean.argmax()) == int(q_mean.argmax())),
                            "label_spread_within_plan": float(plan_labels.std(axis=1).mean()),
                            "label_spread_across_plans": float(plan_mean.std()),
                            "label_executed_minus_mean": float(plan_mean[decoded["plan_index"]] - plan_mean.mean()),
                            "label_seconds": label_seconds, "label_wait_seconds": label_wait_seconds})
                    if token_future is not None:
                        token_future.result()
                        token_future = None
                    chunk_seconds += time.perf_counter() - started
                    chunk_actions += executed_n
                    episode_return += chunk_reward
                    s_cpu = _host_array(decoded["s"])[0]
                    buffer.add_online({
                        "s": s_cpu,
                        "z": _host_array(noise[star]),
                        "a_base": _host_array(a_base[star]),
                        "z_all": _host_array(noise),
                        "a_base_all": _host_array(a_base),
                        "a": _host_array(chosen)[0],
                        "reward": np.float32(chunk_reward),
                        "done": np.float32(done),
                        "s_next": s_cpu.copy(),
                        "task_id": task_id,
                        "n_env_actions": executed_n,
                        "is_expert": np.float32(0.0),
                        "reset_kind": kind,
                        "variant_index": env_task,
                        "residual_applied": np.float32(applied),
                        **({"plan_labels": plan_labels} if plan_labels is not None else {}),
                        **({"proprio": _host_array(batch["proprio"])[0], "object_state": sim_state,
                            "object_state_next": sim_state_next} if config.proprio else {}),
                        **({key: host_bf16(decoded[key][0]) for key in ("state_struct", "state_tokens", "state_latent") if key in decoded} if features else {}),
                        **({"plan_pool": _host_array(decoded["plan_pool"].to(device))[0], "plan_pools_all": _host_array(decoded["plan_pools"].to(device)),
                            "plan_index": int(decoded["plan_index"]), "plan_a_base_all": _host_array(all_base)} if plans > 1 else {}),
                        **({"plan_struct": host_bf16(decoded["plan_struct"][0]), "plan_struct_all": host_bf16(decoded["plan_struct_all"]),
                            "plan_latent_all": host_bf16(decoded["plan_latent_all"])} if features and plans > 1 and "plan_struct_all" in decoded else {}),
                    })
                    chunks += 1
                    expert_ratio = config.rlpd_expert_ratio(env_steps)
                    if chunks % config.update_every_chunks == 0 and buffer.ready_online_count() >= min_online_rows:
                        update_info = _update_from_buffer(
                            model, buffer, expert_ratio, device, env_steps >= config.bc_filter_warmup_steps, residual_active)
                        if env_steps >= next_sharpen:
                            update_info.update(replay_sharpening(model, _sample_batch(buffer, expert_ratio, device)))
                            next_sharpen = env_steps - env_steps % SHARPEN_EVERY_STEPS + SHARPEN_EVERY_STEPS
                        run.log({
                            **update_info, **plan_info, **chunk_info,
                            **({f"{key}/{RESET_KINDS[kind]}": value for key, value in {**plan_info, **chunk_info}.items()} if config.plus_train else {}),
                            "env_steps": env_steps, "chunks": chunks, "expert_ratio": expert_ratio,
                            "buffer_online_rows": buffer.ready_online_count(),
                            "selection_active": float(env_steps >= config.selection_warmup_steps),
                            "seconds_per_env_step": chunk_seconds / max(chunk_actions, 1),
                            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
                        })
                        chunk_seconds, chunk_actions, chunk_info = 0.0, 0, {}
                    if done:
                        break
                buffer.finalize_episode()
            finally:
                if future is not None:
                    future.wait()
                if token_future is not None:
                    wait([token_future])
                env.close()
                if offset is not None:
                    pose_override.clear()
            if offset is not None:
                curriculum.record(episode_success)
            name = RESET_KINDS[kind]
            run.log({
                "env_steps": env_steps, "episode_return": episode_return,
                "episode_success": episode_success, "episode_length": episode_length,
                "reset_kind": kind, "variant_index": env_task,
                **({f"episode_success_{name}": episode_success, f"episode_length_{name}": episode_length} if config.plus_train else {}),
            })
            episodes += 1
            maybe_checkpoint()
            if episodes % 10 == 0:
                save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id, episodes, warmup_end_steps, curriculum)
    finally:
        if pool is not None:
            pool.close()
        if labeller is not None:
            labeller.shutdown()
    save_inference(output_dir / "residual.pt", model)
    save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id, episodes, warmup_end_steps, curriculum)
    _write_json(output_dir / "summary.json", {"env_steps": env_steps, "chunks": chunks})
    if commit is not None:
        commit(env_steps)
    if hasattr(wandb, "finish"):
        wandb.finish()
    return {"env_steps": env_steps, "chunks": chunks}


def evaluate(config=None, prepared_path=None, output_dir=None, run_name=None, residual_path=None,
             resume=False, commit=None, eval_candidates=None, shard=0, shards=1, plan_ranker="v_plan"):
    from script.lingbot_eval import (
        aggregate_results, describe_suite, file_sha256, read_checkpoint_metadata, read_json, write_json,
    )
    from script.lingbot_eval_config import shard_plan

    config = (config or RLConfig()).validate()
    eval_cfg = EvalConfig(
        source_run=config.source_run, checkpoint_step=config.checkpoint_step, stage="heldout", seed=42,
        task_ids=config.task_ids, shard=shard, shards=shards, video_exec_step=config.video_exec_step,
        video_steps=config.video_steps, action_steps=config.action_steps).validate()
    prepared = _read_prepared(prepared_path)
    device = _device()
    if device == "cuda":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    metadata = read_checkpoint_metadata(prepared["checkpoint"])
    policy = load_residual_policy(
        prepared["checkpoint"], prepared["model_path"], prepared.get("architecture") or metadata.get("architecture") or {},
        eval_cfg.video_exec_step, eval_cfg.video_steps, eval_cfg.action_steps)
    residual_path = Path(residual_path or Path(output_dir) / "residual.pt")
    try:
        payload = torch.load(residual_path, map_location="cpu", weights_only=True)
        layout = DiceResidualModel.layout_of(payload)
        config = replace(config, **layout).validate()
        model = DiceResidualModel(device=device, residual_input=config.residual_input, epsilon=config.epsilon,
                                  plan_candidates=config.plan_candidates, state_features=config.state_features, proprio=config.proprio,
                                  plan_target=config.plan_target)
        model.load_inference_state_dict(payload)
    except Exception as exc:
        raise RuntimeError(f"Failed to load residual weights {residual_path}: {exc}") from exc
    policy.residual_model = model
    policy.eval_candidates = eval_candidates or config.k_candidates
    policy.eval_plans = config.plan_candidates
    if plan_ranker not in ("v_plan", "q", "random"):
        raise ValueError("plan_ranker must be v_plan, q or random")
    policy.eval_plan_ranker = plan_ranker
    suite, tasks = describe_suite(prepared["assets_path"])
    tasks = [task for task in tasks if task["task_id"] in config.task_ids]
    plans = {task["task_id"]: shard_plan(eval_cfg, episode_plan(eval_cfg, task["task_id"], task["initial_state_count"])) for task in tasks}
    expected = sum(len(plan) for plan in plans.values())
    output_dir = Path(output_dir)
    write_json(output_dir / "settings" / f"shard_{shard}.json", {
        "config": eval_cfg.to_dict(), "rl": config.to_dict(), "eval_candidates": policy.eval_candidates,
        "eval_plans": policy.eval_plans, "plan_ranker": plan_ranker, "layout": layout,
        "residual_sha256": file_sha256(residual_path), "episode_plan": [entry for task in tasks for entry in plans[task["task_id"]]]})

    def summarize():
        summary = aggregate_results(rows, task_ids=config.task_ids, episodes_per_task=eval_cfg.episodes_per_task, expected=expected)
        write_json(output_dir / "summary" / f"shard_{shard}.json", summary)
        if commit is not None:
            commit(len(rows))
        return summary

    rows = []
    for task in tasks:
        env = make_env(task["task_id"], suite, proprio=config.proprio)
        try:
            for entry in plans[task["task_id"]]:
                path = output_dir / "episodes" / f"task_{task['task_id']:02d}" / f"episode_{entry['episode_index']:03d}.json"
                if resume and path.exists():
                    rows.append(read_json(path))
                    continue
                rows.append(run_rl_episode(policy, env, entry, task["instruction"], metadata["normalization"], device, config.proprio))
                write_json(path, rows[-1])
                summarize()
        finally:
            env.close()
    return summarize()


def _hook(command):
    import subprocess

    def run(step):
        subprocess.run(command, shell=True, check=False, env={**os.environ, "DICE_STEP": str(int(step))})
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("train", "eval"))
    parser.add_argument("--config-json")
    parser.add_argument("--prepared-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--residual-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--demos-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-volume")
    parser.add_argument("--checkpoint-hook")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--eval-candidates", type=int)
    parser.add_argument("--plan-ranker", default="v_plan")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    config = config_from_dict(json.loads(args.config_json) if args.config_json else {})
    commit = None
    if args.result_volume:
        import modal
        volume = modal.Volume.from_name(args.result_volume)
        commit = lambda step: volume.commit()
    elif args.checkpoint_hook:
        commit = _hook(args.checkpoint_hook)
    if args.operation == "train":
        summary = train(
            config, args.prepared_path, args.output_dir, args.run_name, args.resume, commit,
            max_env_steps=args.max_env_steps, dataset_root=args.dataset_root, demos_root=args.demos_root,
        )
    else:
        summary = evaluate(
            config, args.prepared_path, args.output_dir, args.run_name, args.residual_path, args.resume, commit,
            eval_candidates=args.eval_candidates, shard=args.shard, shards=args.shards, plan_ranker=args.plan_ranker)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
