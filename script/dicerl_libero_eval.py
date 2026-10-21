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


def run_protocol(policy, venv, init_states, plan, n_envs, act_steps=8, max_steps=520):
    rows = []
    chunks = math.ceil(max_steps / act_steps)
    start = 0
    while start < len(plan):
        end = start + 1
        while end < len(plan) and end - start < n_envs and plan[end]["seed"] == plan[start]["seed"]:
            end += 1
        batch = plan[start:end]
        start = end
        live = len(batch)
        batch = batch + [batch[-1]] * (n_envs - live)
        torch.manual_seed(batch[0]["seed"])
        obs = stack_obs(venv.reset_arg(options_list=[{"init_state": init_states[b["init_state_id"]]} for b in batch]))
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
            rows.append({**batch[i], "success": bool(reward[i] >= 1), "length": int(length[i])})
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


def make_env(cfg, n_envs):
    from env.gym_utils import make_async
    return make_async(
        cfg.env.name, env_type=cfg.env.env_type, num_envs=n_envs, asynchronous=True,
        max_episode_steps=cfg.env.max_episode_steps, wrappers=cfg.env.wrappers, obs_dim=cfg.obs_dim,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--envs", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from env.gym_utils.wrapper.libero_image import libero_task_files
    cfg = load_cfg(args.ckpt, args.device)
    venv = make_env(cfg, args.envs)
    try:
        _, model = load_prior(args.ckpt, args.device)
        init_states = libero_task_files(cfg.env.name)[1]
        plan = episode_plan(50, (0, 1))
        act_steps, max_steps = cfg.act_steps, cfg.env.max_episode_steps
        rows = run_protocol(flow_policy(model, act_steps, args.device), venv, init_states, plan, args.envs, act_steps, max_steps)
    finally:
        venv.close()
    epoch = os.path.splitext(os.path.basename(args.ckpt))[0].split("_")[-1]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"eval_{epoch}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["init_state_id", "seed", "success", "length"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {**summarise(rows), "ckpt": os.path.abspath(args.ckpt), "task": cfg.env.name, "max_steps": max_steps, "act_steps": act_steps, "n_envs": args.envs}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
