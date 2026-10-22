import argparse
import json
import os
import re
import subprocess
import sys
import time

BAND = (0.40, 0.60)
LAST_EPOCH = 8000


def in_band(success_rate):
    return BAND[0] <= success_rate <= BAND[1]


def select_checkpoint(rows):
    banded = [r for r in rows if in_band(r["success_rate"])]
    if not banded:
        return None
    return min(banded, key=lambda r: (round(abs(r["success_rate"] - 0.5), 9), r["epoch"]))


def checkpoint_epochs(run_dir, every):
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return []
    epochs = [int(m.group(1)) for f in os.listdir(ckpt_dir) if (m := re.fullmatch(r"state_(\d+)\.pt", f))]
    return sorted(e for e in epochs if e % every == 0)


def evaluate(run_dir, epoch, envs):
    out = os.path.join(run_dir, f"eval_{epoch}")
    ckpt = os.path.join(run_dir, "checkpoint", f"state_{epoch}.pt")
    subprocess.run([sys.executable, "-m", "script.dicerl_libero_eval", "--ckpt", ckpt, "--envs", str(envs), "--out", out], check=True)
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    return {"epoch": epoch, "success_rate": summary["success_rate"], "stderr": summary["stderr"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=25)
    parser.add_argument("--session", required=True)
    parser.add_argument("--poll", type=int, default=300)
    args = parser.parse_args()
    sweep_path = os.path.join(args.run_dir, "prior_sweep.json")
    rows = json.load(open(sweep_path)) if os.path.exists(sweep_path) else []
    done = {r["epoch"] for r in rows}
    while True:
        for epoch in checkpoint_epochs(args.run_dir, args.every):
            if epoch in done:
                continue
            row = evaluate(args.run_dir, epoch, args.envs)
            rows.append(row)
            done.add(epoch)
            with open(sweep_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(json.dumps(row), flush=True)
            if in_band(row["success_rate"]):
                subprocess.run(["tmux", "kill-session", "-t", args.session])
                return 0
            if epoch >= LAST_EPOCH:
                print(f"no checkpoint in band [{BAND[0]}, {BAND[1]}] through epoch {LAST_EPOCH}", file=sys.stderr)
                return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
