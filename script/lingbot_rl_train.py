import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval_config import TASK_IDS, EvalConfig, episode_plan
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig, config_from_dict
from script.lingbot_rl_data import featurize_experts, load_manifest_episodes
from script.lingbot_rl_model import (
    ACTOR_EVERY, BATCH, GRADIENT_STEPS, UPDATE_EVERY_CHUNKS, DiceResidualModel, apply_residual,
)
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, load_residual_policy, slice_env_actions,
)

LiberoEnv = None
RESUME_KEYS = {
    "actor", "critic", "target_critic", "actor_opt", "critic_opt", "actor_lr", "critic_lr",
    "replay", "rng", "env_steps", "chunks", "recipe", "checkpointed", "wandb_id",
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
    if set(payload) - {"actor", "critic", "target_critic"}:
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


def save_resume(path, model, buffer, env_steps, chunks, recipe, checkpointed=(), wandb_id=None):
    payload = {
        **model.resume_state_dict(),
        "replay": buffer.state_dict(),
        "rng": capture_rng(),
        "env_steps": int(env_steps),
        "chunks": int(chunks),
        "recipe": recipe,
        "checkpointed": sorted(int(point) for point in checkpointed),
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
    checkpointed = {int(point) for point in payload.get("checkpointed", ())}
    return int(payload["env_steps"]), int(payload.get("chunks", 0)), checkpointed, payload.get("wandb_id")


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


def _sample_batch(buffer, expert_ratio, device):
    return _to_model_device(buffer.sample(min(BATCH, len(buffer)), expert_ratio), device)


def _update_from_buffer(model, buffer, expert_ratio, device, filter_active):
    info = {}
    for step in range(GRADIENT_STEPS):
        sample = _sample_batch(buffer, expert_ratio, device)
        target = model.n_step_target(
            sample["reward"], sample["done"], sample["s_next"],
            sample["z_next_all"], sample["a_base_next_all"], sample["n_steps"])
        if (step + 1) % ACTOR_EVERY == 0:
            info.update(model.update_actor(
                sample["s"], sample["a"], sample["z_all"], sample["a_base_all"], sample["is_expert"],
                sample["mc_return"], filter_active))
        info.update(model.update_critic(sample["s"], sample["a"], target))
        if (step + 1) % ACTOR_EVERY == 0:
            model.polyak_update()
    return info


def _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir, episodes):
    from script.lingbot_eval import run_episode

    plan_cfg = EvalConfig(stage="heldout", seed=42)
    rows = []
    for task in tasks:
        plan = episode_plan(plan_cfg, task["task_id"], plan_cfg.episodes_per_task)[:episodes]
        env = make_env(task["task_id"], suite, init_states=False)
        try:
            for entry in plan:
                row = run_episode(policy, env, entry, task["instruction"], norm, device)
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
        norm = prepared.get("normalization") or {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}
    policy = load_residual_policy(prepared.get("checkpoint"), prepared.get("model_path"), architecture)
    model = DiceResidualModel(device=device)
    policy.residual_model = model
    policy.eval_candidates = config.k_candidates
    buffer = ChunkReplay()
    tasks = prepared["tasks"]
    if prepared.get("checkpoint") and not resume:
        cache_path = output_dir / "expert_features.pt"
        manifest = read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json")
        episodes = []
        if dataset_root:
            task_to_id = {task["instruction"]: task["task_id"] for task in tasks}
            episodes = load_manifest_episodes(dataset_root, manifest, task_to_id, norm)
        if episodes or cache_path.is_file():
            _ingest_experts(buffer, featurize_experts(policy, episodes, manifest, norm, cache_path), config.task_ids)
    suite = None
    if prepared.get("assets_path"):
        from script.lingbot_eval import describe_suite
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = suite_tasks
    tasks = [task for task in tasks if task["task_id"] in config.task_ids]
    env_steps = 0
    chunks = 0
    checkpointed = set()
    wandb_id = None
    resume_path = output_dir / "resume" / "latest.pt"
    if resume:
        env_steps, chunks, checkpointed, wandb_id = load_resume(resume_path, model, buffer, recipe)
    if max_env_steps is None and env_steps == 0:
        if not any(float(row["is_expert"]) == 1.0 for row in buffer.rows()):
            raise RuntimeError("Expert features missing; RLPD requires the SFT demos")
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
    smoke = max_env_steps is not None
    budget = max_env_steps if smoke else config.online_env_steps
    min_online_rows = 1 if smoke else BATCH
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
            summary = _train_eval(policy, tasks, norm, device, suite, point, output_dir, config.train_eval_episodes)
            restore_rng(rng)
            logged = {"train_eval/macro_success_rate": summary["macro_success_rate"], "env_steps": env_steps}
            for task_id, success in summary["per_task_success"].items():
                logged[f"train_eval/task_{task_id}_success"] = success
            env = make_env(tasks[0]["task_id"], suite, init_states=False)
            try:
                observation, _ = env.reset(seed=0)
                policy.reset()
                logged.update(_maybe_sharpen(
                    model, policy, observation_batch(observation, tasks[0]["instruction"], device), device))
            finally:
                env.close()
                policy.reset()
            run.log(logged)
        save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id)
        if commit is not None:
            commit(point)

    maybe_checkpoint()
    episodes = 0
    while env_steps < budget:
        task = tasks[int(np.random.randint(len(tasks)))]
        task_id = task["task_id"]
        env = make_env(task_id, suite, init_states=False)
        policy.reset()
        observation, _ = env.reset(seed=config.seed + env_steps)
        episode_return = 0.0
        episode_success = 0.0
        episode_length = 0
        saw_success = False
        try:
            while env_steps < budget and episode_length < 520:
                batch = observation_batch(observation, task["instruction"], device)
                decoded = policy.decode_candidates(batch, k=config.k_candidates)
                state = decoded["s"].to(device=device, dtype=torch.float32)
                noise = decoded["z"].to(device=device, dtype=torch.float32)
                a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
                k = a_base.shape[0]
                if k != config.k_candidates:
                    raise RuntimeError("action candidate batching failed")
                with torch.no_grad():
                    state_k = state.expand(k, -1)
                    executed = apply_residual(a_base, model.actor(state_k, noise))
                    if env_steps < config.selection_warmup_steps:
                        star = int(np.random.randint(k))
                    else:
                        star = int(model.critic(state_k, executed).reshape(-1).argmax())
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
                episode_return += chunk_reward
                s_cpu = _host_array(state)[0]
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
                })
                chunks += 1
                expert_ratio = config.rlpd_expert_ratio(env_steps)
                if chunks % UPDATE_EVERY_CHUNKS == 0 and buffer.ready_online_count() >= min_online_rows:
                    update_info = _update_from_buffer(
                        model, buffer, expert_ratio, device, env_steps >= config.bc_filter_warmup_steps)
                    run.log({
                        **update_info, "env_steps": env_steps, "chunks": chunks, "expert_ratio": expert_ratio,
                        "buffer_online_rows": buffer.ready_online_count(),
                        "selection_active": float(env_steps >= config.selection_warmup_steps),
                    })
                if done:
                    break
            buffer.finalize_episode()
        finally:
            env.close()
        run.log({
            "env_steps": env_steps, "episode_return": episode_return,
            "episode_success": episode_success, "episode_length": episode_length,
        })
        episodes += 1
        maybe_checkpoint()
        if episodes % 10 == 0:
            save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id)
    save_inference(output_dir / "residual.pt", model)
    save_resume(resume_path, model, buffer, env_steps, chunks, recipe, checkpointed, wandb_id)
    _write_json(output_dir / "summary.json", {"env_steps": env_steps, "chunks": chunks})
    if commit is not None:
        commit(env_steps)
    if hasattr(wandb, "finish"):
        wandb.finish()
    return {"env_steps": env_steps, "chunks": chunks}


def evaluate(config=None, prepared_path=None, output_dir=None, run_name=None, residual_path=None,
             resume=False, commit=None, eval_candidates=None):
    from script.lingbot_eval import (
        aggregate_results, describe_suite, read_checkpoint_metadata, read_json, run_episode, write_json,
    )

    config = (config or RLConfig()).validate()
    eval_cfg = EvalConfig(
        source_run=config.source_run, checkpoint_step=config.checkpoint_step, stage="heldout", seed=42)
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
    policy.eval_candidates = eval_candidates or config.k_candidates
    suite, tasks = describe_suite(prepared["assets_path"])
    tasks = [task for task in tasks if task["task_id"] in config.task_ids]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        env = make_env(task["task_id"], suite)
        try:
            for entry in episode_plan(eval_cfg, task["task_id"], task["initial_state_count"]):
                path = output_dir / "eval" / "episodes" / f"task_{task['task_id']:02d}" / f"episode_{entry['episode_index']:03d}.json"
                if resume and path.exists():
                    rows.append(read_json(path))
                    continue
                row = run_episode(policy, env, entry, task["instruction"], metadata["normalization"], device)
                rows.append(row)
                write_json(path, row)
                if commit is not None:
                    commit(entry["episode_index"])
        finally:
            env.close()
    summary = aggregate_results(rows, task_ids=config.task_ids, episodes_per_task=eval_cfg.episodes_per_task)
    write_json(output_dir / "eval" / "summary.json", summary)
    write_json(output_dir / "eval" / "settings.json",
               {"config": eval_cfg.to_dict(), "rl": config.to_dict(), "eval_candidates": policy.eval_candidates})
    return summary


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-volume")
    parser.add_argument("--checkpoint-hook")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--eval-candidates", type=int)
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
            max_env_steps=args.max_env_steps, dataset_root=args.dataset_root,
        )
    else:
        summary = evaluate(
            config, args.prepared_path, args.output_dir, args.run_name, args.residual_path, args.resume, commit,
            eval_candidates=args.eval_candidates)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
