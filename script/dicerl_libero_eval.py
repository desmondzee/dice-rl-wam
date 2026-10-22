import argparse
import csv
import json
import math
import os

import numpy as np
import torch


def episode_plan(n_states=50, seeds=(0, 1)):
    return [{"init_state_id": i, "seed": s} for s in seeds for i in range(n_states)]


def stack_obs(obs):
    if isinstance(obs, list):
        return {key: np.stack([o[key] for o in obs]) for key in obs[0]}
    return obs


def load_states(path):
    data = np.load(path)
    states, labels = data["states"], data["labels"]
    if states.ndim != 2 or len(states) != len(labels):
        raise ValueError(f"{path}: states {states.shape} and labels {labels.shape} do not match")
    return states, labels


def run_protocol(policy, venv, init_states, plan, n_envs, act_steps=8, max_steps=520, shift=None, labels=None):
    rows = []
    chunks = math.ceil(max_steps / act_steps)
    start = 0
    while start < len(plan):
        end = start + 1
        while end < len(plan) and end - start < n_envs and plan[end]["seed"] == plan[start]["seed"]:
            end += 1
        batch = plan[start:end]
        first = start
        start = end
        live = len(batch)
        batch = batch + [batch[-1]] * (n_envs - live)
        torch.manual_seed(batch[0]["seed"])
        options = [{"init_state": init_states[b["init_state_id"]]} for b in batch]
        if shift is not None:
            for j, option in enumerate(options):
                option["shift"] = {"magnitude": shift["magnitude"], "seed": shift["seed"] + first + min(j, live - 1)}
        obs = stack_obs(venv.reset_arg(options_list=options))
        done = np.zeros(n_envs, dtype=bool)
        reward = np.zeros(n_envs)
        length = np.zeros(n_envs, dtype=int)
        for _ in range(chunks):
            obs, r, term, trunc, _ = venv.step(policy(obs))
            obs = stack_obs(obs)
            active = ~done
            reward[active] += np.asarray(r, dtype=float)[active]
            length[active] += act_steps
            done |= active & (np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool) | (length >= max_steps))
            if done.all():
                break
        for i in range(live):
            label = str(batch[i]["init_state_id"]) if labels is None else str(labels[batch[i]["init_state_id"]])
            rows.append({**batch[i], "label": label, "success": bool(reward[i] >= 1), "length": int(length[i])})
    return rows


def summarise(rows):
    n = len(rows)
    p = sum(r["success"] for r in rows) / n
    return {"success_rate": p, "stderr": math.sqrt(p * (1 - p) / n), "n": n, "mean_length": float(np.mean([r["length"] for r in rows]))}


def run_dir_of(ckpt_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))


def load_cfg(ckpt_path, device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(run_dir_of(ckpt_path), ".hydra", "config.yaml"))
    cfg.device = device
    return cfg


def load_prior(ckpt_path, device):
    import hydra
    cfg = load_cfg(ckpt_path, device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint["model"])
    return cfg, model.to(device).eval()


def resolved_hard_reset(cfg, override=None):
    if override is not None:
        return override == "true"
    return bool(cfg.env.wrappers.get("libero_image", {}).get("hard_reset", True))


def make_env(cfg, n_envs, hard_reset=None):
    import copy
    from env.gym_utils import make_async
    wrappers = copy.deepcopy(cfg.env.wrappers)
    wrappers["libero_image"]["hard_reset"] = resolved_hard_reset(cfg, hard_reset)
    return make_async(
        cfg.env.name, env_type=cfg.env.env_type, num_envs=n_envs, asynchronous=True,
        max_episode_steps=cfg.env.max_episode_steps, wrappers=wrappers, obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim, use_image_obs=True, shape_meta=cfg.shape_meta,
    )


def flow_policy(model, act_steps, device):
    def policy(obs):
        rgb = torch.from_numpy(obs["rgb"]).float().to(device).permute(0, 1, 4, 2, 3)
        state = torch.from_numpy(obs["state"]).float().to(device)
        with torch.no_grad():
            samples = model(cond={"state": state, "rgb": rgb})
        return samples.trajectories[:, :act_steps].cpu().numpy()
    return policy


def shift_totals(venv, shift):
    totals = {"draws": 0, "rejections": 0, "fallbacks": 0}
    if shift is not None:
        for stats, _ in venv.call_sync("shift_stats"):
            for key in totals:
                totals[key] += stats[key]
    return totals


def add_protocol_args(parser):
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--envs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--states", default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--shift-seed", type=int, default=1000)
    parser.add_argument("--hard-reset", choices=("true", "false"), default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")


def protocol_inputs(args, task):
    if args.states:
        init_states, labels = load_states(args.states)
    else:
        from env.gym_utils.wrapper.libero_image import libero_task_files
        init_states, labels = libero_task_files(task)[1], None
    shift = None if not args.shift else {"magnitude": args.shift, "seed": args.shift_seed}
    return init_states, labels, episode_plan(len(init_states), tuple(args.seeds)), shift


def write_results(args, cfg, rows, extra):
    epoch = os.path.splitext(os.path.basename(args.ckpt))[0].split("_")[-1]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"eval_{epoch}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["init_state_id", "label", "seed", "success", "length"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **summarise(rows), "ckpt": os.path.abspath(args.ckpt), "task": cfg.env.name, "max_steps": cfg.env.max_episode_steps,
        "act_steps": cfg.act_steps, "n_envs": args.envs, "seeds": list(args.seeds), "states": args.states or "canonical",
        "hard_reset": resolved_hard_reset(cfg, args.hard_reset), "shift": args.shift or None, "shift_seed": args.shift_seed, **extra,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    add_protocol_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args.ckpt, args.device)
    init_states, labels, plan, shift = protocol_inputs(args, cfg.env.name)
    venv = make_env(cfg, args.envs, args.hard_reset)
    try:
        _, model = load_prior(args.ckpt, args.device)
        rows = run_protocol(flow_policy(model, cfg.act_steps, args.device), venv, init_states, plan, args.envs, cfg.act_steps, cfg.env.max_episode_steps, shift, labels)
        stats = shift_totals(venv, shift)
    finally:
        venv.close()
    write_results(args, cfg, rows, {"weights": "model", "shift_stats": stats})


if __name__ == "__main__":
    main()
