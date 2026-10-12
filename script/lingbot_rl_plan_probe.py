import argparse
import csv
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval import decode_action, describe_suite, observation_batch, read_checkpoint_metadata, seed_all
from script.lingbot_eval_config import EvalConfig, episode_plan
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_model import USED_DOF, DiceResidualModel, apply_residual
from script.lingbot_rl_policy import load_residual_policy, slice_env_actions
from script.lingbot_rl_train import _device, _read_prepared, env_success, make_env

MAX_CHUNKS = 8


def plan_q(model, state, z, a_base):
    plans, k = a_base.shape[:2]
    state_k = state.expand(plans * k, -1)
    base = a_base.reshape(plans * k, *a_base.shape[2:])
    noise = z.unsqueeze(0).expand(plans, -1, -1, -1).reshape(plans * k, *z.shape[1:])
    with torch.no_grad():
        executed = apply_residual(base, model.actor(state_k, noise, base))
        return model.critic(state_k, executed).reshape(plans, k).max(dim=1).values


def probe_row(out, rank_steps, model):
    live = out["a_base"][..., :USED_DOF]
    means = live.mean(dim=1)
    pairs = [float(((means[i] - means[j]) ** 2).mean().sqrt()) for i, j in combinations(range(means.shape[0]), 2)]
    row = {
        "within_spread": float(live.std(dim=1, unbiased=False).mean()),
        "across_spread": float(means.std(dim=0, unbiased=False).mean()),
        "pairwise_rms": float(np.mean(pairs)),
    }
    q = plan_q(model, out["s"], out["z"], out["a_base"]) if model is not None else None
    best = int(q.argmax()) if q is not None else None
    for m in range(live.shape[0]):
        row[f"q_plan_{m}"] = float(q[m]) if q is not None else ""
    row["q_argmax"] = best if q is not None else ""
    row["q_spread"] = float(q.max() - q.min()) if q is not None else ""
    for step in rank_steps:
        partial = out["rank_a_base"][step]
        row[f"rms_step_{step}"] = float(((partial[..., :USED_DOF] - live) ** 2).mean().sqrt())
        row[f"agree_step_{step}"] = "" if q is None else int(int(plan_q(model, out["s"], out["z"], partial).argmax()) == best)
    return row


def step_chunks(policy, env, observation, instruction, norm, device, chunks, k, model):
    for _ in range(chunks):
        decoded = policy.decode_candidates(observation_batch(observation, instruction, device), k=k)
        a_base = decoded["a_base"].to(device=device, dtype=torch.float32)
        index = 0
        executed = a_base
        if model is not None:
            state = decoded["s"].to(device=device, dtype=torch.float32).expand(a_base.shape[0], -1)
            noise = decoded["z"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                executed = apply_residual(a_base, model.actor(state, noise, a_base))
                index = int(model.critic(state, executed).reshape(-1).argmax())
        chosen = executed[index:index + 1].cpu()
        policy.commit_executed(chosen, first_chunk=decoded["first_chunk"])
        env_actions = slice_env_actions(chosen, decoded["first_chunk"])
        for i in range(env_actions.shape[1]):
            observation, _, terminated, truncated, info = env.step(decode_action(env_actions[:, i, :].reshape(1, 7), norm))
            policy.observe_env_step(observation_batch(observation, instruction, device))
            if env_success(info) or terminated or truncated:
                return observation, True
    return observation, False


def summarise(rows, rank_steps):
    def mean(key):
        values = [row[key] for row in rows if row[key] != ""]
        return float(np.mean(values)) if values else None

    summary = {"states": len(rows), "chunks_mean": mean("chunks")}
    for key in ("within_spread", "across_spread", "pairwise_rms", "q_spread"):
        summary[key] = mean(key)
    for step in rank_steps:
        summary[f"rms_step_{step}"] = mean(f"rms_step_{step}")
        summary[f"agree_step_{step}"] = mean(f"agree_step_{step}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task", type=int, default=9)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--plans", type=int, default=4)
    parser.add_argument("--exec-step", type=int, required=True)
    parser.add_argument("--rank-steps", default="3,6,10")
    parser.add_argument("--residual-path", type=Path)
    parser.add_argument("--residual-input", default="base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.plans < 2:
        raise ValueError("The probe needs at least two plans")
    device = _device()
    if device == "cuda":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    prepared = _read_prepared(args.prepared_path)
    metadata = read_checkpoint_metadata(prepared["checkpoint"])
    policy = load_residual_policy(
        prepared["checkpoint"], prepared["model_path"], prepared.get("architecture") or metadata["architecture"])
    norm = metadata["normalization"]
    model = None
    if args.residual_path:
        model = DiceResidualModel(device=device, residual_input=args.residual_input)
        model.load_inference_state_dict(torch.load(args.residual_path, map_location="cpu", weights_only=True))
        policy.residual_model = model
    k = RLConfig().k_candidates
    timesteps, exec_step = policy._video_timesteps(args.exec_step)
    rank_steps = sorted({int(step) for step in args.rank_steps.split(",") if step} & set(range(1, len(timesteps) + 1)))
    suite, tasks = describe_suite(prepared["assets_path"])
    task = next(item for item in tasks if item["task_id"] == args.task)
    plan = episode_plan(EvalConfig(stage="heldout", seed=42), args.task, task["initial_state_count"])
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    env = make_env(args.task, suite, init_states=False)
    try:
        for index in range(args.states):
            entry = plan[index % len(plan)]
            seed = entry["seed"] + index // len(plan)
            chunks = int(rng.integers(0, MAX_CHUNKS + 1))
            seed_all(seed)
            env.init_state_id = entry["init_state_id"]
            observation, _ = env.reset(seed=seed)
            policy.reset()
            seed_all(seed)
            observation, done = step_chunks(
                policy, env, observation, task["instruction"], norm, device, chunks, k if model is not None else 1, model)
            if done:
                continue
            out = policy.decode_plans(
                observation_batch(observation, task["instruction"], device), args.plans, exec_step, k, rank_steps)
            row = {"index": index, "episode_index": entry["episode_index"], "init_state_id": entry["init_state_id"],
                   "seed": seed, "chunks": chunks, **probe_row(out, rank_steps, model)}
            rows.append(row)
            print(json.dumps(row))
    finally:
        env.close()
    if not rows:
        raise RuntimeError("No states were probed")
    with (args.out / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **summarise(rows, rank_steps),
        "task": args.task, "plans": args.plans, "k": k, "exec_step": exec_step, "rank_steps": rank_steps,
        "timesteps": [float(t) for t in timesteps], "sigmas": [float(t) / 1000.0 for t in timesteps],
        "residual_path": str(args.residual_path) if args.residual_path else None, "residual_input": args.residual_input,
        "prepared_path": str(args.prepared_path), "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "seed": args.seed, "max_chunks": MAX_CHUNKS,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
