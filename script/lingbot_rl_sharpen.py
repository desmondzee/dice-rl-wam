import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from script.lingbot_rl_model import HORIZON, USED_DOF, DiceResidualModel, apply_residual

BINS = 50


def bin_entropy(candidates):
    live = candidates[..., :USED_DOF].reshape(candidates.shape[0], candidates.shape[1], -1)
    index = np.clip(np.floor((live + 1.0) / 2.0 * BINS), 0, BINS - 1).astype(np.int16)
    same = index[:, :, None, :] == index[:, None, :, :]
    counts = same.sum(axis=2)
    return (-np.log(counts / candidates.shape[1])).mean(axis=(1, 2))


def log_std(candidates):
    live = candidates[..., :USED_DOF].reshape(candidates.shape[0], candidates.shape[1], -1)
    return np.log(live.std(axis=1) + 1e-4).mean(axis=1)


def episode_rows(rows):
    episode = 0
    step = 0
    for row in rows:
        if float(row["is_expert"]) == 1.0:
            continue
        yield episode, step, row
        step += 1
        if float(row["n_steps"]) == 1.0:
            episode += 1
            step = 0


def analyse(resume_path, residual_path, batch=512, device="cpu", residual_input="z"):
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    model = DiceResidualModel(device=device, residual_input=residual_input)
    model.load_inference_state_dict(torch.load(residual_path, map_location="cpu", weights_only=True))
    items = list(episode_rows(payload["replay"]["data"]))
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            state = torch.from_numpy(np.stack([row["s"] for _, _, row in chunk])).float().to(device)
            z_all = torch.from_numpy(np.stack([row["z_all"] for _, _, row in chunk])).float().to(device)
            base_all = torch.from_numpy(np.stack([row["a_base_all"] for _, _, row in chunk])).float().to(device)
            n, k = z_all.shape[0], z_all.shape[1]
            state_k = state.unsqueeze(1).expand(n, k, -1).reshape(n * k, -1)
            base_flat = base_all.reshape(n * k, HORIZON, -1)
            action = apply_residual(base_flat, model.actor(state_k, z_all.reshape(n * k, HORIZON, -1), base_flat))
            delta_v = (model.critic(state_k, action) - model.critic(state_k, base_flat)).reshape(n, k).mean(dim=1).cpu().numpy()
            after = action.reshape(n, k, HORIZON, -1).cpu().numpy()
            before = base_flat.reshape(n, k, HORIZON, -1).cpu().numpy()
            delta_h = bin_entropy(after) - bin_entropy(before)
            delta_std = log_std(after) - log_std(before)
            for (episode, step, row), dv, dh, ds in zip(chunk, delta_v, delta_h, delta_std):
                out.append({"episode": episode, "step": step, "task_id": int(row["task_id"]), "delta_v": float(dv),
                            "delta_h": float(dh), "delta_log_std": float(ds), "mc_return": float(row["mc_return"]),
                            "reward": float(row["reward"]), "done": float(row["done"])})
    return out


def summarise(rows):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episodes = []
    for episode, items in sorted(by_episode.items()):
        episodes.append({"episode": episode, "steps": len(items), "success": float(any(r["reward"] > 0 for r in items)),
                         "delta_v": float(np.mean([r["delta_v"] for r in items])), "delta_h": float(np.mean([r["delta_h"] for r in items])),
                         "delta_log_std": float(np.mean([r["delta_log_std"] for r in items]))})
    dv = np.array([r["delta_v"] for r in rows]); dh = np.array([r["delta_h"] for r in rows]); ds = np.array([r["delta_log_std"] for r in rows])
    quartiles = np.quantile(dv, [0.25, 0.5, 0.75])
    bands = np.digitize(dv, quartiles)
    return {
        "rows": len(rows), "episodes": len(episodes),
        "delta_v_mean": float(dv.mean()), "delta_h_mean": float(dh.mean()), "delta_log_std_mean": float(ds.mean()),
        "r_delta_v_delta_h": float(np.corrcoef(dv, dh)[0, 1]), "r_delta_v_delta_log_std": float(np.corrcoef(dv, ds)[0, 1]),
        "delta_h_by_delta_v_quartile": [float(dh[bands == b].mean()) for b in range(4)],
        "delta_v_quartiles": [float(q) for q in quartiles],
        "episode_r_delta_v_delta_h": float(np.corrcoef([e["delta_v"] for e in episodes], [e["delta_h"] for e in episodes])[0, 1]),
        "episode_success_rate": float(np.mean([e["success"] for e in episodes])),
    }, episodes


def plot(rows, episodes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, label in ((axes[0], rows, "per replay state"), (axes[1], episodes, "per episode")):
        x = np.array([d["delta_v"] for d in data]); y = np.array([d["delta_h"] for d in data])
        ax.scatter(x, y, s=4, alpha=0.25, color="#4a7c59")
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 2)
        ax.plot(grid, slope * grid + intercept, color="#5b4ea6", linewidth=2, label=f"trend (r={np.corrcoef(x, y)[0, 1]:.3f})")
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--"); ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("change in value (ΔV)"); ax.set_ylabel("change in action entropy (ΔH)"); ax.set_title(f"{title}, {label}"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--residual-input", default="z")
    args = parser.parse_args()
    rows = analyse(args.resume, args.residual, residual_input=args.residual_input)
    summary, episodes = summarise(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("rows.csv", rows), ("episodes.csv", episodes)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, episodes, args.title or args.out.name, args.out / "delta_h_vs_delta_v.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
