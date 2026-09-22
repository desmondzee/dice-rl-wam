import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

Z = 1.959963984540054
BOOTSTRAP_SAMPLES = 10000


def load_rows(directory):
    summary = Path(directory) / "summary.json"
    if summary.is_file() and not json.loads(summary.read_text()).get("complete", True):
        raise ValueError(f"Incomplete evaluation under {directory}")
    rows = {}
    for path in sorted(Path(directory).glob("episodes/task_*/episode_*.json")):
        row = json.loads(path.read_text())
        key = (int(row["task_id"]), int(row["init_state_id"]), int(row.get("seed_index", 0)))
        if key in rows:
            raise ValueError(f"Duplicate episode key {key} under {directory}")
        rows[key] = bool(row["success"])
    if not rows:
        raise FileNotFoundError(f"No episode records under {directory}")
    return rows


def arm_of(directory):
    directory = Path(directory)
    settings_path = directory / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    k = settings.get("k")
    step = settings.get("step")
    if k is None:
        match = re.search(r"-k(\d+)", directory.name)
        k = int(match.group(1)) if match else 1
    if step is None:
        match = re.search(r"step(\d+)", directory.name)
        step = int(match.group(1)) if match else 0
    return k, step


def wilson(successes, n):
    if n == 0:
        return math.nan, math.nan, math.nan
    p = successes / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return p, center - half, center + half


def mcnemar_exact(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs, samples=BOOTSTRAP_SAMPLES, seed=0):
    diffs = np.asarray(diffs, dtype=float)
    index = np.random.default_rng(seed).integers(0, len(diffs), size=(samples, len(diffs)))
    means = diffs[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(base, other):
    keys = sorted(set(base) & set(other))
    if not keys:
        raise ValueError("No shared (task, init_state, seed) episodes to pair")
    wins = sum(other[key] and not base[key] for key in keys)
    losses = sum(base[key] and not other[key] for key in keys)
    diffs = [int(other[key]) - int(base[key]) for key in keys]
    low, high = bootstrap_ci(diffs)
    return {"n": len(keys), "diff": float(np.mean(diffs)), "wins": wins, "losses": losses,
            "p": mcnemar_exact(wins, losses), "ci_low": low, "ci_high": high}


def by_task(rows):
    tasks = {}
    for key, success in rows.items():
        tasks.setdefault(key[0], {})[key] = success
    return tasks


def summarize(sft_dir, rl_dirs):
    arms = [("sft-k1", 0, by_task(load_rows(sft_dir)))]
    for directory in rl_dirs:
        k, step = arm_of(directory)
        arms.append((f"rl-k{k}", step, by_task(load_rows(directory))))
    curve, lines = [], []
    for task in sorted({task for _, _, tasks in arms for task in tasks}):
        lines.append(f"task {task}")
        lines.append(f"  {'arm':8s} {'step':>7s} {'n':>4s} {'success':>8s}  wilson95")
        for arm, step, tasks in arms:
            rows = tasks.get(task, {})
            p, low, high = wilson(sum(rows.values()), len(rows))
            curve.append({"task": task, "step": step, "arm": arm, "success": p, "ci_low": low, "ci_high": high})
            lines.append(f"  {arm:8s} {step:7d} {len(rows):4d} {p:8.3f}  [{low:.3f}, {high:.3f}]")
        base = arms[0][2].get(task, {})
        for arm, step, tasks in arms[1:]:
            if task in tasks:
                lines.append(f"  {arm}@{step} vs sft-k1: " + describe(paired(base, tasks[task])))
        for step in sorted({step for arm, step, tasks in arms[1:] if arm == "rl-k1" and task in tasks}):
            k4 = next((tasks[task] for arm, s, tasks in arms if arm == "rl-k4" and s == step and task in tasks), None)
            k1 = next(tasks[task] for arm, s, tasks in arms if arm == "rl-k1" and s == step)
            if k4 is not None:
                lines.append(f"  rl-k4 vs rl-k1 @{step}: " + describe(paired(k1, k4)))
    return curve, lines


def describe(result):
    return (f"diff {result['diff']:+.3f} (n={result['n']}, wins {result['wins']}, losses {result['losses']}), "
            f"McNemar p={result['p']:.3f}, bootstrap95 [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]")


def write_curve(path, curve):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "step", "arm", "success", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True)
    parser.add_argument("--rl", nargs="*", default=[])
    parser.add_argument("--curve", default="curve.csv")
    args = parser.parse_args()
    curve, lines = summarize(args.sft, args.rl)
    print("\n".join(lines))
    write_curve(args.curve, curve)
    print(f"wrote {args.curve}")


if __name__ == "__main__":
    main()
