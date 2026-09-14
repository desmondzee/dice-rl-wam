import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval_config import TASK_IDS, EvalConfig, episode_plan
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_data import featurize_experts, load_manifest_episodes
from script.lingbot_rl_model import BATCH, UTD, DiceResidualModel, apply_residual
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, load_residual_policy, slice_env_actions,
)

LiberoEnv = None
RESUME_KEYS = {
    "actor", "critic", "target_critic", "actor_opt", "critic_opt",
    "replay", "rng", "env_steps", "chunks", "recipe",
}


def _env_class():
    global LiberoEnv
    if LiberoEnv is None:
        from lerobot.envs.libero import LiberoEnv as cls
        LiberoEnv = cls
    return LiberoEnv


def make_env(task_id, suite=None):
    return _env_class()(
        task_suite=suite, task_id=task_id, task_suite_name="libero_10",
        episode_length=520, observation_height=128, observation_width=128, obs_type="pixels",
        init_states=True, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative",
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
    return list(range(0, budget + 1, every))


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


def save_resume(path, model, buffer, env_steps, chunks, recipe):
    payload = {
        **model.resume_state_dict(),
        "replay": buffer.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        "env_steps": int(env_steps),
        "chunks": int(chunks),
        "recipe": recipe,
    }
    if "transformer" in payload or set(payload) - RESUME_KEYS:
        raise ValueError("Resume payload includes disallowed keys")
    _atomic_torch(path, payload)


def load_resume(path, model, buffer, recipe):
    payload = torch.load(path, map_location=model.device, weights_only=False)
    if payload.get("recipe") != recipe:
        raise ValueError("Resume recipe fingerprint mismatch")
    if "transformer" in payload:
        raise ValueError("Resume file contains frozen transformer weights")
    model.load_resume_state_dict(payload)
    buffer.load_state_dict(payload["replay"])
    if payload.get("rng"):
        torch.set_rng_state(payload["rng"]["torch"])
        np.random.set_state(payload["rng"]["numpy"])
    return int(payload["env_steps"]), int(payload.get("chunks", 0))


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


def _update_from_buffer(model, buffer, expert_ratio, device):
    batch_size = min(BATCH, len(buffer))
    sample = _to_model_device(buffer.sample(batch_size, expert_ratio), device)
    target = model.n_step_target(
        sample["reward"], sample["done"], sample["s_next"], sample["a_next"], sample["n_steps"])
    critic_info = None
    for _ in range(UTD):
        critic_info = model.update_critic(sample["s"], sample["a"], target, sample["is_expert"])
    actor_info = model.update_actor(
        sample["s"], sample["z"], sample["a_base"], sample["is_expert"], target)
    return critic_info, actor_info


def _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir):
    from script.lingbot_eval import run_episode

    eval_cfg = EvalConfig(stage="smoke", seed=42)
    rows = []
    for task in tasks:
        plan = episode_plan(eval_cfg, task["task_id"], task.get("initial_state_count", 50))
        env = make_env(task["task_id"], suite)
        try:
            row = run_episode(policy, env, plan[0], task["instruction"], norm, device)
        finally:
            env.close()
        rows.append(row)
        _write_json(
            Path(output_dir) / "train_eval" / f"step_{env_steps:06d}" / f"task_{task['task_id']:02d}.json",
            row,
        )
    successes = {str(row["task_id"]): bool(row["success"]) for row in rows}
    return {"env_steps": env_steps, "per_task_success": successes,
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
        metadata = None
        norm = prepared["normalization"] if "normalization" in prepared else prepared.get("norm") or {
            "q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23,
        }
    policy = load_residual_policy(prepared.get("checkpoint"), prepared.get("model_path"), architecture)
    model = DiceResidualModel(device=device)
    policy.residual_model = model
    buffer = ChunkReplay()
    tasks = prepared["tasks"]
    if prepared.get("checkpoint"):
        cache_path = output_dir / "expert_features.pt"
        if dataset_root:
            task_to_id = {task["instruction"]: task["task_id"] for task in tasks}
            episodes = load_manifest_episodes(
                dataset_root, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"),
                task_to_id, norm)
            _ingest_experts(buffer, featurize_experts(
                policy, episodes, read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm, cache_path))
        elif cache_path.is_file():
            _ingest_experts(buffer, featurize_experts(
                policy, [], read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json"), norm, cache_path))
    suite = None
    if prepared.get("assets_path"):
        from script.lingbot_eval import describe_suite
        suite, suite_tasks = describe_suite(prepared["assets_path"])
        tasks = suite_tasks
    env_steps = 0
    chunks = 0
    resume_path = output_dir / "resume" / "latest.pt"
    if resume:
        env_steps, chunks = load_resume(resume_path, model, buffer, recipe)
    if max_env_steps is None and env_steps == 0:
        if not any(float(row["is_expert"]) == 1.0 for row in buffer.rows()):
            raise RuntimeError("Expert features missing; RLPD requires the 300 SFT demos")
    _write_json(output_dir / "settings.json", {"config": config.to_dict(), "recipe": recipe})
    run = wandb.init(
        project=config.wandb_project, entity=config.wandb_entity, name=run_name or config.default_run_name,
        config=config.to_dict(), resume="allow" if resume else None,
    )
    budget = config.online_env_steps if max_env_steps is None else max_env_steps
    eval_schedule = train_eval_schedule(budget, config.train_eval_every)
    evaluated = set()
    do_train_eval = max_env_steps is None

    def maybe_eval():
        # Episode-boundary only: `_train_eval` / sharpening call `policy.reset()`.
        # Thresholds, not exact equality, so 12/16-step chunks still hit 25k/50k/75k/100k.
        # `--max-env-steps` is unit/cloud smoke: skip the 10-task eval so a few chunks stay cheap.
        if not do_train_eval:
            return
        due = due_train_evals(env_steps, eval_schedule, evaluated)
        if due:
            evaluated.update(due)
            summary = _train_eval(policy, tasks, norm, device, suite, env_steps, output_dir)
            run.log({f"train_eval/{key}": value for key, value in summary.items() if key != "per_task_success"})
            for task_id, success in summary["per_task_success"].items():
                run.log({f"train_eval/task_{task_id}_success": float(success), "env_steps": env_steps})
            task = tasks[0]
            env = make_env(task["task_id"], suite)
            try:
                observation, _ = env.reset(seed=0)
                policy.reset()
                metrics = _maybe_sharpen(
                    model, policy, observation_batch(observation, task["instruction"], device), device)
                if metrics:
                    run.log({**metrics, "env_steps": env_steps})
            finally:
                env.close()
                policy.reset()
            save_inference(output_dir / "residual.pt", model)
            save_resume(resume_path, model, buffer, env_steps, chunks, recipe)
            if commit is not None:
                commit()

    maybe_eval()
    episode_return = 0.0
    episode_success = 0.0
    episode_length = 0
    while env_steps < budget:
        task_id = int(np.random.randint(0, 10))
        task = next(item for item in tasks if item["task_id"] == task_id)
        env = make_env(task_id, suite)
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
                    q_values = model.critic(state_k, executed)
                    star = int(q_values.reshape(-1).argmax())
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
                    if env_success(info) and not saw_success:
                        chunk_reward = 1.0
                        saw_success = True
                        episode_success = 1.0
                    if saw_success or terminated or truncated or episode_length >= 520 or env_steps >= budget:
                        done = 1.0
                        break
                episode_return += chunk_reward
                s_cpu = _host_array(state)[0]
                row = {
                    "s": s_cpu,
                    "z": _host_array(noise[star]),
                    "a_base": _host_array(a_base[star]),
                    "a": _host_array(chosen)[0],
                    "reward": np.float32(chunk_reward),
                    "done": np.float32(done),
                    "s_next": s_cpu.copy(),
                    "task_id": task_id,
                    "n_env_actions": executed_n,
                    "is_expert": np.float32(0.0),
                }
                buffer.add_online(row)
                chunks += 1
                expert_ratio = config.rlpd_expert_ratio(env_steps)
                if buffer.has_ready_online():
                    critic_info, actor_info = _update_from_buffer(model, buffer, expert_ratio, device)
                    log = {
                        "env_steps": env_steps, "chunks": chunks,
                        "actor_loss": actor_info["actor_loss"], "critic_loss": critic_info["critic_loss"],
                        "residual_rms": actor_info["residual_rms"], "q_mean": actor_info["q_mean"],
                        "q_min": actor_info["q_min"], "bc_filter_rate": actor_info["bc_filter_rate"],
                        "expert_ratio": expert_ratio, "episode_return": episode_return,
                        "episode_success": episode_success, "episode_length": episode_length,
                    }
                    run.log(log)
                if done:
                    break
            buffer.finalize_episode()
        finally:
            env.close()
        maybe_eval()
        save_inference(output_dir / "residual.pt", model)
        save_resume(resume_path, model, buffer, env_steps, chunks, recipe)
        if commit is not None:
            commit()
    maybe_eval()
    save_inference(output_dir / "residual.pt", model)
    save_resume(resume_path, model, buffer, env_steps, chunks, recipe)
    _write_json(output_dir / "summary.json", {"env_steps": env_steps, "chunks": chunks})
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
    model.load_inference_state_dict(torch.load(residual_path, map_location="cpu", weights_only=True))
    policy.residual_model = model
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
    parser.add_argument("operation", choices=("train", "eval"))
    parser.add_argument("--config-json")
    parser.add_argument("--prepared-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--residual-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-volume")
    parser.add_argument("--max-env-steps", type=int)
    args = parser.parse_args()
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
