import argparse
import csv
import json
import math
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from env.gym_utils import make_async

BINS = 50
SEED_BASE = 10000


def bin_entropy(candidates):
    live = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
    index = np.clip(np.floor((live + 1.0) / 2.0 * BINS), 0, BINS - 1).astype(np.int16)
    same = index[:, :, None, :] == index[:, None, :, :]
    counts = same.sum(axis=2)
    return (-np.log(counts / candidates.shape[1])).mean(axis=(1, 2))


def log_std(candidates):
    live = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
    return np.log(live.std(axis=1) + 1e-4).mean(axis=1)


def load(ckpt, device):
    checkpoint = torch.load(ckpt, map_location=device)
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), ".hydra", "config.yaml"))
    cfg.device = device
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return cfg, model.to(device).eval()


def make_env(cfg, n):
    venv = make_async(cfg.env.name, env_type=cfg.env.get("env_type"), num_envs=n, asynchronous=True,
                      max_episode_steps=cfg.env.max_episode_steps, wrappers=cfg.env.get("wrappers"),
                      robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path"), shape_meta=cfg.get("shape_meta"),
                      use_image_obs=False, render=False, render_offscreen=False, obs_dim=cfg.obs_dim,
                      action_dim=cfg.action_dim, **(cfg.env.specific if "specific" in cfg.env else {}))
    venv.seed([SEED_BASE + i for i in range(n)])
    return venv


def rollout(cfg, model, venv, episodes, n, k, device):
    act = cfg.act_steps
    max_steps = cfg.env.max_episode_steps
    threshold = cfg.env.get("best_reward_threshold_for_success", 1)
    gamma = cfg.train.gamma
    rows = []
    for ep in range(episodes):
        obs = venv.reset_arg(options_list=[{"seed": SEED_BASE + ep * n + i} for i in range(n)])
        if isinstance(obs, list):
            obs = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        done = np.zeros(n, dtype=bool)
        length = np.zeros(n)
        traj = [[] for _ in range(n)]
        for _ in range(math.ceil(max_steps / act)):
            state = torch.from_numpy(obs["state"]).float().to(device)
            with torch.no_grad():
                action, _ = model.get_exploration_action(state, num_samples=k, exploration_strategy="max_q_min", training_step=10 ** 6)
            obs, reward, term, trunc, _ = venv.step(action[:, :act].cpu().numpy())
            for i in range(n):
                if done[i]:
                    continue
                traj[i].append((state[i].cpu().numpy(), float(reward[i])))
                length[i] += act
                if term[i] or trunc[i] or length[i] >= max_steps:
                    done[i] = True
            if done.all():
                break
        for i in range(n):
            ret = 0.0
            mc = []
            for _, r in reversed(traj[i]):
                ret = r + gamma * ret
                mc.append(ret)
            mc.reverse()
            success = float(sum(r for _, r in traj[i]) >= threshold)
            for t, ((s, r), m) in enumerate(zip(traj[i], mc)):
                rows.append({"episode": ep * n + i, "step": t, "state": s, "reward": r, "mc_return": m, "success": success})
    return rows


def analyse(model, rows, k, device, batch=256):
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            state = torch.from_numpy(np.stack([r["state"] for r in chunk])).float().to(device)
            n = state.shape[0]
            state_k = state.unsqueeze(1).expand(n, k, *state.shape[1:]).reshape(n * k, *state.shape[1:])
            noise = torch.randn(n * k, model.horizon_steps, model.action_dim, device=device)
            action, base = model.get_action(state_k, noise, return_pretrained_actions=True)
            q_base = model.critic(state_k, noise, base).view(n, k)
            q_res = model.critic(state_k, noise, action).view(n, k)
            after = action.view(n, k, model.horizon_steps, model.action_dim).cpu().numpy()
            before = base.view(n, k, model.horizon_steps, model.action_dim).cpu().numpy()
            delta_h = bin_entropy(after) - bin_entropy(before)
            delta_log_std = log_std(after) - log_std(before)
            delta_v = (q_res - q_base).mean(dim=1).cpu().numpy()
            value = q_base.mean(dim=1).cpu().numpy()
            residual = (action - base).view(n, k, -1).norm(dim=-1).mean(dim=1).cpu().numpy()
            for row, dv, dh, ds, v, res in zip(chunk, delta_v, delta_h, delta_log_std, value, residual):
                out.append({"episode": row["episode"], "step": row["step"], "value": float(v), "delta_v": float(dv), "delta_h": float(dh), "delta_log_std": float(ds),
                            "residual_norm": float(res), "mc_return": row["mc_return"], "reward": row["reward"], "success": row["success"]})
    return out


def quartile_means(x, y):
    quartiles = np.quantile(x, [0.25, 0.5, 0.75])
    bands = np.digitize(x, quartiles)
    return [float(y[bands == b].mean()) for b in range(4)], [float(q) for q in quartiles]


def summarise(rows):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episodes = [{"episode": e, "steps": len(items), "success": items[0]["success"],
                 "value": float(np.mean([r["value"] for r in items])), "delta_v": float(np.mean([r["delta_v"] for r in items])),
                 "delta_h": float(np.mean([r["delta_h"] for r in items]))} for e, items in sorted(by_episode.items())]
    v = np.array([r["value"] for r in rows])
    dv = np.array([r["delta_v"] for r in rows])
    dh = np.array([r["delta_h"] for r in rows])
    mc = np.array([r["mc_return"] for r in rows])
    ds = np.array([r["delta_log_std"] for r in rows])
    dh_by_value, value_quartiles = quartile_means(v, dh)
    dh_by_dv, dv_quartiles = quartile_means(dv, dh)
    dh_by_mc, mc_quartiles = quartile_means(mc, dh)
    return {
        "rows": len(rows), "episodes": len(episodes), "episode_success_rate": float(np.mean([e["success"] for e in episodes])),
        "value_mean": float(v.mean()), "delta_v_mean": float(dv.mean()), "delta_h_mean": float(dh.mean()),
        "residual_norm_mean": float(np.mean([r["residual_norm"] for r in rows])),
        "r_value_delta_h": float(np.corrcoef(v, dh)[0, 1]), "r_delta_v_delta_h": float(np.corrcoef(dv, dh)[0, 1]),
        "r_mc_return_delta_h": float(np.corrcoef(mc, dh)[0, 1]),
        "delta_log_std_mean": float(ds.mean()), "r_value_delta_log_std": float(np.corrcoef(v, ds)[0, 1]),
        "r_delta_v_delta_log_std": float(np.corrcoef(dv, ds)[0, 1]), "delta_log_std_by_value_quartile": quartile_means(v, ds)[0],
        "delta_h_by_value_quartile": dh_by_value, "value_quartiles": value_quartiles,
        "delta_h_by_delta_v_quartile": dh_by_dv, "delta_v_quartiles": dv_quartiles,
        "delta_h_by_mc_return_quartile": dh_by_mc, "mc_return_quartiles": mc_quartiles,
        "delta_h_success_episodes": float(dh[np.array([r["success"] for r in rows]) == 1].mean()) if any(r["success"] for r in rows) else None,
        "delta_h_failed_episodes": float(dh[np.array([r["success"] for r in rows]) == 0].mean()) if any(not r["success"] for r in rows) else None,
    }, episodes


def plot(rows, episodes, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, data, xkey, label in ((axes[0], rows, "delta_v", "per state"), (axes[1], episodes, "delta_v", "per episode"), (axes[2], rows, "value", "per state")):
        x = np.array([d[xkey] for d in data])
        y = np.array([d["delta_h"] for d in data])
        ax.scatter(x, y, s=4, alpha=0.25, color="#4a7c59")
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 2)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(grid, slope * grid + intercept, color="#5b4ea6", linewidth=2, label=f"r={r:.3f}, slope={slope:.3f}")
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("ΔV = Q(base+residual) − Q(base)" if xkey == "delta_v" else "V = mean Q(base)")
        ax.set_ylabel("ΔH (action entropy after − before)")
        ax.set_title(f"{title}, {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "delta_h_vs_delta_v.png", dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--envs", type=int, default=30)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg, model = load(str(args.ckpt), args.device)
    venv = make_env(cfg, args.envs)
    states = rollout(cfg, model, venv, args.episodes, args.envs, args.k, args.device)
    venv.close()
    rows = analyse(model, states, args.k, args.device)
    summary, episodes = summarise(rows)
    summary["ckpt"] = str(args.ckpt)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("rows.csv", rows), ("episodes.csv", episodes)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, episodes, args.ckpt.stem, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
