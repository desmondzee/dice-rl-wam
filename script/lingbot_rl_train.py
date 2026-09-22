import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval_config import TASK_IDS, EvalConfig, episode_plan
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_data import featurize_experts, load_manifest_episodes
from script.lingbot_rl_model import BATCH, UTD, DiceResidualModel, apply_residual
from script.lingbot_rl_policy import histogram_entropy, load_residual_policy, slice_env_actions

LiberoEnv = None
SAVE_EVERY_EPISODES = 10
RESUME_KEYS = {
    "actor", "critic", "target_critic", "actor_opt", "critic_opt",
    "replay", "rng", "env_steps", "chunks", "recipe", "evaluated", "wandb_id",
}


def _env_class():
    global LiberoEnv
    if LiberoEnv is None:
        from lerobot.envs.libero import LiberoEnv as cls
        LiberoEnv = cls
    return LiberoEnv


def make_env(task_id, suite=None, init_states=True):
    return _env_class()(
        task_suite=suite, task_id=task_id, task_suite_name="libero_10",
        episode_length=520, observation_height=128, observation_width=128, obs_type="pixels",
        init_states=init_states, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative",
        hard_reset=True,
    )


def sharpening_metrics(model, state, a_base, action):
    if a_base.ndim == 4:
        batch, candidates, horizon, dim = a_base.shape
        state = state.unsqueeze(1).expand(batch, candidates, -1).reshape(batch * candidates, -1)
        a_base = a_base.reshape(batch * candidates, horizon, dim)
        action = action.reshape(batch * candidates, horizon, dim)
    with torch.no_grad():
        delta_v = float((model.critic(state, action) - model.critic(state, a_base)).mean())
        delta_h = histogram_entropy(a_base) - histogram_entropy(action)
    return {"delta_h": delta_h, "delta_v": delta_v}


def recipe_of(config):
    return config.protocol()


def train_eval_schedule(budget, every):
    if every < 1:
        raise ValueError("train_eval_every must be positive")
    points = set(range(0, budget + 1, every))
    points.add(int(budget))
    return sorted(points)


def train_eval_plan(task_id, episodes, seed=42):
    plan = []
    for index in range(int(episodes)):
        identity = ["libero_10_procedural", int(seed), int(task_id), index]
        digest = hashlib.sha256(json.dumps(identity).encode()).digest()[:4]
        plan.append({"task_id": int(task_id), "episode_index": index, "init_state_id": index,
                     "seed": int.from_bytes(digest, "big")})
    return plan


def due_train_evals(env_steps, schedule, evaluated):
    return [point for point in schedule if env_steps >= point and point not in evaluated]


def _ingest_experts(buffer, rows):
    for row in rows:
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
    if set(payload) - {"actor", "critic", "target_critic"}:
        raise ValueError("Inference dump may only contain actor and critic weights")
    _atomic_torch(path, payload)


def save_inference_checkpoints(output_dir, env_steps, model):
    output_dir = Path(output_dir)
    save_inference(output_dir / "residual.pt", model)
    save_inference(output_dir / "train_eval" / f"step_{int(env_steps):06d}" / "residual.pt", model)


def run_sync_hook(output_dir, env_steps):
    command = os.environ.get("DICE_SYNC_CMD", "").strip()
    if not command:
        return None
    env = {**os.environ, "DICE_OUTPUT_DIR": str(output_dir), "DICE_STEP": str(int(env_steps))}
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        print(f"sync hook exited {result.returncode}", file=sys.stderr)
    return result.returncode


def procedural_reset_difference(env, seeds=(1, 2)):
    frames = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        frames.append(np.asarray(observation["pixels"]["image"], dtype=np.float32))
    return float(np.mean(np.abs(frames[0] - frames[1])))


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


def _evaluated_from_disk(output_dir):
    root = Path(output_dir) / "train_eval"
    if not root.is_dir():
        return set()
    found = set()
    for path in root.glob("step_*"):
        if path.is_dir():
            found.add(int(path.name.split("_", 1)[1]))
    return found


def save_resume(path, model, buffer, env_steps, chunks, recipe, evaluated=(), wandb_id=None):
    payload = {
        **model.resume_state_dict(),
        "replay": buffer.state_dict(),
        "rng": capture_rng(),
        "env_steps": int(env_steps),
        "chunks": int(chunks),
        "recipe": recipe,
        "evaluated": sorted(int(point) for point in evaluated),
        "wandb_id": wandb_id,
    }
    if "transformer" in payload or set(payload) - RESUME_KEYS:
        raise ValueError("Resume payload includes disallowed keys")
    _atomic_torch(path, payload)


def load_resume(path, model, buffer, recipe):
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
    if "evaluated" in payload:
        evaluated = {int(point) for point in payload["evaluated"]}
    else:
        evaluated = _evaluated_from_disk(path.resolve().parent.parent)
    return int(payload["env_steps"]), int(payload.get("chunks", 0)), evaluated, payload.get("wandb_id")


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


def _stack_batches(batches):
    return {
        key: sum((batch[key] for batch in batches), []) if key == "task" else torch.cat([batch[key] for batch in batches])
        for key in batches[0]
    }


def _sample_batch(buffer, expert_ratio, device):
    return _to_model_device(buffer.sample(min(BATCH, len(buffer)), expert_ratio), device)


def _update_from_buffer(model, buffer, expert_ratio, device):
    critic_info = None
    for _ in range(UTD):
        sample = _sample_batch(buffer, expert_ratio, device)
        target = model.n_step_target(
            sample["reward"], sample["done"], sample["s_next"],
            sample["z_next_all"], sample["a_base_next_all"], sample["n_steps"])
        critic_info = model.update_critic(sample["s"], sample["a"], target, sample["is_expert"])
    sample = _sample_batch(buffer, expert_ratio, device)
    actor_info = model.update_actor(
        sample["s"], sample["z_all"], sample["a_base_all"], sample["is_expert"], sample["mc_return"])
    return critic_info, actor_info


def _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes):
    from script.lingbot_eval import run_episode

    rows = []
    for task in tasks:
        env = make_env(task["task_id"], suite, init_states=False)
        try:
            for entry in train_eval_plan(task["task_id"], episodes):
                row = run_episode(policy, env, entry, task["instruction"], norm, device)
                rows.append(row)
                _write_json(
                    Path(output_dir) / "train_eval" / f"step_{env_steps:06d}" / f"task_{task['task_id']:02d}"
                    / f"episode_{entry['episode_index']:03d}.json",
                    row,
                )
        finally:
            env.close()
    per_task = {}
    for task in tasks:
        task_rows = [row for row in rows if int(row["task_id"]) == task["task_id"]]
        per_task[str(task["task_id"])] = float(np.mean([bool(row["success"]) for row in task_rows]))
    return {"env_steps": env_steps, "per_task_success": per_task,
            "macro_success_rate": float(np.mean([bool(row["success"]) for row in rows]))}


def collection_stats(buffer, episodes, episode_lengths):
    return {
        "replay_size": len(buffer),
        "episodes": int(episodes),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }


def _maybe_sharpen(model, policy, batch, device):
    decoded = policy.decode_candidates(batch, k=8)
    a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
    noise = decoded["z"].to(device=device, dtype=torch.float32)
    state = decoded["s"].to(device=device, dtype=torch.float32)
    if state.shape[0] == 1 and a_base.shape[0] > 1:
        state = state.expand(a_base.shape[0], -1)
    with torch.no_grad():
        action = apply_residual(a_base, model.actor(state, noise))
    if a_base.ndim == 3:
        a_base = a_base.unsqueeze(0)
        action = action.unsqueeze(0)
        pooled = decoded["s"].to(device)
    else:
        pooled = state
    return sharpening_metrics(model, pooled, a_base, action)


def train(config=None, prepared_path=None, output_dir=None, run_name=None, resume=False,
          commit=None, max_env_steps=None, dataset_root=None):
    import wandb
    from script.lingbot_eval import decode_action, observation_batch, seed_all

    config = (config or RLConfig()).validate()
    recipe = recipe_of(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _read_prepared(prepared_path)
    if prepared.get("source_run") not in (None, config.source_run) or prepared.get("checkpoint_step") not in (None, config.checkpoint_step):
        raise ValueError("Prepared checkpoint does not match the pinned RL recipe")
    device = _device()
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
        metadata = None
        norm = prepared["normalization"] if "normalization" in prepared else prepared.get("norm") or {
            "q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23,
        }
    policy = load_residual_policy(prepared.get("checkpoint"), prepared.get("model_path"), architecture)
    model = DiceResidualModel(device=device)
    policy.residual_model = model
    policy.eval_candidates = config.k_candidates
    buffer = ChunkReplay()
    tasks = [task for task in prepared["tasks"] if task["task_id"] in config.task_ids]
    if prepared.get("checkpoint") and not resume:
        cache_path = output_dir / "expert_features.pt"
        if dataset_root:
            task_to_id = {task["instruction"]: task["task_id"] for task in prepared["tasks"]}
            episodes = load_manifest_episodes(
                dataset_root, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"),
                task_to_id, norm, task_ids=config.task_ids)
            _ingest_experts(buffer, featurize_experts(
                policy, episodes, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm,
                cache_path, task_ids=config.task_ids))
        elif cache_path.is_file():
            _ingest_experts(buffer, featurize_experts(
                policy, [], read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm,
                cache_path, task_ids=config.task_ids))
    suite = None
    if prepared.get("assets_path"):
        from script.lingbot_eval import describe_suite
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = [task for task in suite_tasks if task["task_id"] in config.task_ids]
    if not tasks:
        raise ValueError("Prepared tasks do not cover the configured task_ids")
    env_steps = 0
    chunks = 0
    evaluated = set()
    wandb_id = None
    resume_path = output_dir / "resume" / "latest.pt"
    if resume:
        env_steps, chunks, evaluated, wandb_id = load_resume(resume_path, model, buffer, recipe)
    if max_env_steps is None and env_steps == 0:
        if not any(float(row["is_expert"]) == 1.0 for row in buffer.rows()):
            raise RuntimeError("Expert features missing; RLPD requires the SFT demos for task_ids")
    _write_json(output_dir / "settings.json", {"config": config.to_dict(), "recipe": recipe})
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
    budget = config.online_env_steps if max_env_steps is None else max_env_steps
    eval_schedule = train_eval_schedule(budget, config.train_eval_every)
    do_train_eval = max_env_steps is None
    envs = {}
    episodes = 0
    episode_lengths = []
    saved_blocks = 0

    def maybe_eval():
        if not do_train_eval:
            return
        due = due_train_evals(env_steps, eval_schedule, evaluated)
        if due:
            evaluated.update(due)
            summary = _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir,
                                  config.train_eval_episodes_per_task)
            logged = {f"train_eval/{key}": value for key, value in summary.items() if key != "per_task_success"}
            logged["env_steps"] = env_steps
            logged.update(collection_stats(buffer, episodes, episode_lengths))
            run.log(logged)
            for task_id, success in summary["per_task_success"].items():
                run.log({f"train_eval/task_{task_id}_success": float(success), "env_steps": env_steps})
            task = tasks[0]
            env = envs.get((0, task["task_id"]))
            borrowed = env is not None
            if not borrowed:
                env = make_env(task["task_id"], suite, init_states=False)
            try:
                sharpened = []
                for entry in train_eval_plan(task["task_id"], 10):
                    observation, _ = env.reset(seed=entry["seed"])
                    policy.reset()
                    metrics = _maybe_sharpen(
                        model, policy, observation_batch(observation, task["instruction"], device), device)
                    if metrics:
                        sharpened.append(metrics)
                if sharpened:
                    run.log({
                        "delta_v": float(np.mean([item["delta_v"] for item in sharpened])),
                        "delta_h": float(np.mean([item["delta_h"] for item in sharpened])),
                        "env_steps": env_steps,
                    })
            finally:
                if not borrowed:
                    env.close()
                policy.reset()
            save_inference_checkpoints(output_dir, env_steps, model)
            save_resume(resume_path, model, buffer, env_steps, chunks, recipe, evaluated, wandb_id)
            run_sync_hook(output_dir, env_steps)
            if commit is not None:
                commit()

    maybe_eval()
    try:
        while env_steps < budget:
            slots = list(range(config.n_envs))
            slot_tasks = [tasks[int(np.random.randint(len(tasks)))] for _ in slots]
            observations = []
            for slot, task in zip(slots, slot_tasks):
                env = envs.get((slot, task["task_id"]))
                if env is None:
                    env = envs[(slot, task["task_id"])] = make_env(task["task_id"], suite, init_states=False)
                observations.append(env.reset(seed=config.seed + env_steps + slot)[0])
            policy.reset()
            stats = [{"return": 0.0, "success": 0.0, "length": 0, "saw_success": False} for _ in slots]
            live = list(slots)

            def live_batch():
                return _stack_batches([
                    observation_batch(observations[slot], slot_tasks[slot]["instruction"], device) for slot in live
                ])

            while live and env_steps < budget:
                decoded = policy.decode_candidates(live_batch(), k=config.k_candidates)
                state = decoded["s"].to(device=device, dtype=torch.float32)
                noise = decoded["z"].to(device=device, dtype=torch.float32)
                a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
                k = config.k_candidates
                if a_base.shape[0] != len(live) * k or state.shape[0] != len(live):
                    raise RuntimeError("action candidate batching failed")
                stars = []
                chosen = []
                delta_v = []
                with torch.no_grad():
                    for position in range(len(live)):
                        rows = slice(position * k, (position + 1) * k)
                        state_k = state[position:position + 1].expand(k, -1)
                        executed = apply_residual(a_base[rows], model.actor(state_k, noise[rows]))
                        q_values = model.critic(state_k, executed)
                        star = int(q_values.argmax())
                        stars.append(position * k + star)
                        chosen.append(executed[star:star + 1])
                        delta_v.append(float(q_values.max() - model.critic(state_k, a_base[rows]).mean()))
                chosen = torch.cat(chosen)
                policy.commit_executed(chosen.cpu(), first_chunk=decoded["first_chunk"])
                env_actions = slice_env_actions(chosen.cpu(), decoded["first_chunk"])
                rewards = [0.0] * len(live)
                dones = [0.0] * len(live)
                executed_n = [0] * len(live)
                over = [False] * len(live)
                for index in range(env_actions.shape[1]):
                    for position, slot in enumerate(live):
                        if over[position]:
                            continue
                        stat = stats[slot]
                        action = decode_action(env_actions[position:position + 1, index, :].reshape(1, 7), norm)
                        observations[slot], _, terminated, truncated, info = envs[
                            (slot, slot_tasks[slot]["task_id"])].step(action)
                        executed_n[position] += 1
                        stat["length"] += 1
                        env_steps += 1
                        if env_success(info) and not stat["saw_success"]:
                            rewards[position] = 1.0
                            stat["saw_success"] = True
                            stat["success"] = 1.0
                        if stat["saw_success"] or terminated:
                            dones[position] = 1.0
                        if dones[position] or truncated or stat["length"] >= 520:
                            over[position] = True
                        if env_steps >= budget:
                            over[:] = [True] * len(live)
                    policy.observe_env_step(live_batch())
                    if all(over):
                        break
                for position, slot in enumerate(live):
                    if executed_n[position] == 0:
                        continue
                    stat = stats[slot]
                    stat["return"] += rewards[position]
                    rows = slice(position * k, (position + 1) * k)
                    s_cpu = _host_array(state[position])
                    row = {
                        "s": s_cpu,
                        "z": _host_array(noise[stars[position]]),
                        "a_base": _host_array(a_base[stars[position]]),
                        "z_all": _host_array(noise[rows]),
                        "a_base_all": _host_array(a_base[rows]),
                        "a": _host_array(chosen[position]),
                        "reward": np.float32(rewards[position]),
                        "done": np.float32(dones[position]),
                        "s_next": s_cpu.copy(),
                        "task_id": slot_tasks[slot]["task_id"],
                        "n_env_actions": executed_n[position],
                        "is_expert": np.float32(0.0),
                    }
                    buffer.add_online(row, stream=slot)
                    chunks += 1
                    expert_ratio = config.rlpd_expert_ratio(env_steps)
                    if buffer.has_ready_online():
                        critic_info, actor_info = _update_from_buffer(model, buffer, expert_ratio, device)
                        log = {
                            "env_steps": env_steps, "chunks": chunks,
                            "actor_loss": actor_info["actor_loss"], "critic_loss": critic_info["critic_loss"],
                            "residual_rms": actor_info["residual_rms"], "q_mean": actor_info["q_mean"],
                            "q_min": actor_info["q_min"], "bc_filter_rate": actor_info["bc_filter_rate"],
                            "expert_ratio": expert_ratio, "episode_return": stat["return"],
                            "episode_success": stat["success"], "episode_length": stat["length"],
                            "chunk_delta_v": delta_v[position],
                        }
                        run.log(log)
                ended = [position for position in range(len(live)) if over[position]]
                for position in ended:
                    buffer.finalize_episode(stream=live[position])
                    episodes += 1
                    episode_lengths.append(stats[live[position]]["length"])
                remaining = [slot for position, slot in enumerate(live) if not over[position]]
                if remaining:
                    for position in reversed(ended):
                        policy.drop_stream(position)
                live = remaining
            maybe_eval()
            if episodes // SAVE_EVERY_EPISODES > saved_blocks:
                saved_blocks = episodes // SAVE_EVERY_EPISODES
                save_inference(output_dir / "residual.pt", model)
                save_resume(resume_path, model, buffer, env_steps, chunks, recipe, evaluated, wandb_id)
                if commit is not None:
                    commit()
    finally:
        for cached in envs.values():
            cached.close()
        envs.clear()
    maybe_eval()
    save_inference(output_dir / "residual.pt", model)
    save_resume(resume_path, model, buffer, env_steps, chunks, recipe, evaluated, wandb_id)
    _write_json(output_dir / "summary.json", {"env_steps": env_steps, "chunks": chunks})
    run_sync_hook(output_dir, env_steps)
    if commit is not None:
        commit()
    if hasattr(wandb, "finish"):
        wandb.finish()
    return {"env_steps": env_steps, "chunks": chunks}


def evaluate(config=None, prepared_path=None, output_dir=None, run_name=None, residual_path=None,
             resume=False, commit=None):
    from script.lingbot_eval import (
        aggregate_results, describe_suite, read_checkpoint_metadata, read_json, run_episode, write_json,
    )

    config = (config or RLConfig()).validate()
    eval_cfg = EvalConfig(
        source_run=config.source_run, checkpoint_step=config.checkpoint_step, stage="eval", seed=42)
    proto = eval_cfg.protocol()
    if proto["episodes_per_task"] != 20 or proto["initial_state_offset"] != 1 or proto["base_seed"] != 42:
        raise ValueError("Comparison eval drifted from the 69% SFT protocol")
    prepared = _read_prepared(prepared_path)
    device = _device()
    if device == "cuda":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    metadata = read_checkpoint_metadata(prepared["checkpoint"])
    policy = load_residual_policy(prepared["checkpoint"], prepared["model_path"], prepared.get("architecture") or metadata.get("architecture") or {})
    model = DiceResidualModel(device=device)
    residual_path = Path(residual_path or Path(output_dir) / "residual.pt")
    try:
        model.load_inference_state_dict(torch.load(residual_path, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise RuntimeError(f"Failed to load residual weights {residual_path}: {exc}") from exc
    policy.residual_model = model
    policy.eval_candidates = config.k_candidates
    suite, tasks = describe_suite(prepared["assets_path"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = {task["task_id"]: episode_plan(eval_cfg, task["task_id"], task["initial_state_count"]) for task in tasks}
    rows = []
    for task in tasks:
        env = make_env(task["task_id"], suite)
        try:
            for entry in plans[task["task_id"]]:
                path = output_dir / "eval" / "episodes" / f"task_{task['task_id']:02d}" / f"episode_{entry['episode_index']:03d}.json"
                if resume and path.exists():
                    rows.append(read_json(path))
                    continue
                row = run_episode(policy, env, entry, task["instruction"], metadata["normalization"], device)
                rows.append(row)
                write_json(path, row)
                if commit is not None:
                    commit()
        finally:
            env.close()
    summary = aggregate_results(rows, episodes_per_task=eval_cfg.episodes_per_task)
    write_json(output_dir / "eval" / "summary.json", summary)
    write_json(output_dir / "eval" / "settings.json", {"config": eval_cfg.to_dict(), "rl": config.to_dict()})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("train", "eval", "check-inits"))
    parser.add_argument("--config-json")
    parser.add_argument("--prepared-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--residual-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-volume")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.operation == "check-inits":
        if args.task_id is None:
            parser.error("check-inits requires --task-id")
        prepared = _read_prepared(args.prepared_path)
        suite = None
        if prepared.get("assets_path"):
            from script.lingbot_eval import describe_suite
            suite, _ = describe_suite(prepared["assets_path"])
        env = make_env(args.task_id, suite, init_states=False)
        try:
            difference = procedural_reset_difference(env)
        finally:
            env.close()
        ok = difference > 0.5
        print(json.dumps({"task_id": args.task_id, "mean_abs_pixel_diff": difference, "ok": ok}))
        if not ok:
            sys.exit(1)
        return
    payload = json.loads(args.config_json) if args.config_json else {}
    config = RLConfig(**{key: payload[key] for key in payload if key in RLConfig.__dataclass_fields__}).validate()
    commit = None
    if args.result_volume:
        import modal
        commit = modal.Volume.from_name(args.result_volume).commit
    if args.operation == "train":
        summary = train(
            config, args.prepared_path, args.output_dir, args.run_name, args.resume, commit,
            max_env_steps=args.max_env_steps, dataset_root=args.dataset_root,
        )
    else:
        summary = evaluate(
            config, args.prepared_path, args.output_dir, args.run_name, args.residual_path, args.resume, commit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
