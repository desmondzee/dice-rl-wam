import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
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


def schedule_epochs(schedule):
    epochs = set()
    for part in schedule.split(","):
        step, last = (int(x) for x in part.split(":"))
        epochs.update(range(step, last + 1, step))
    return sorted(epochs)


def checkpoint_epochs(run_dir, epochs):
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return []
    found = [int(m.group(1)) for f in os.listdir(ckpt_dir) if (m := re.fullmatch(r"state_(\d+)\.pt", f))]
    return sorted(e for e in found if e in epochs)


def evaluate(run_dir, epoch, envs):
    out = os.path.join(run_dir, f"eval_{epoch}")
    ckpt = os.path.join(run_dir, "checkpoint", f"state_{epoch}.pt")
    subprocess.run([sys.executable, "-m", "script.dicerl_libero_eval", "--ckpt", ckpt, "--envs", str(envs), "--out", out], check=True)
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    return {"epoch": epoch, "success_rate": summary["success_rate"], "stderr": summary["stderr"]}


def session_alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def training_pgid(session):
    result = subprocess.run(["tmux", "display-message", "-p", "-t", session, "#{pane_pid}"], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        return None
    try:
        return os.getpgid(int(result.stdout.strip()))
    except ProcessLookupError:
        return None


def stop_training(session, pgid):
    if pgid is not None and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


def save_rows(path, rows):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix="prior_sweep.json.")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)


def decide(rows, session, pgid):
    if any(in_band(r["success_rate"]) for r in rows):
        stop_training(session, pgid)
        return 0
    if any(r["epoch"] >= LAST_EPOCH for r in rows):
        print(f"no checkpoint in band [{BAND[0]}, {BAND[1]}] through epoch {LAST_EPOCH}", file=sys.stderr)
        return 1
    return None


def watch(run_dir, epochs, envs, session, poll):
    epochs = set(epochs)
    sweep_path = os.path.join(run_dir, "prior_sweep.json")
    rows = json.load(open(sweep_path)) if os.path.exists(sweep_path) else []
    done = {r["epoch"] for r in rows}
    pgid = training_pgid(session)
    code = decide(rows, session, pgid)
    if code is not None:
        return code
    while True:
        for epoch in checkpoint_epochs(run_dir, epochs):
            if epoch in done:
                continue
            row = evaluate(run_dir, epoch, envs)
            rows.append(row)
            done.add(epoch)
            save_rows(sweep_path, rows)
            print(json.dumps(row), flush=True)
            code = decide(rows, session, pgid)
            if code is not None:
                return code
        if not session_alive(session):
            print(f"training session {session} is gone and no checkpoint is in band [{BAND[0]}, {BAND[1]}]", file=sys.stderr)
            return 1
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--schedule", default="100:1000,1000:8000")
    parser.add_argument("--envs", type=int, default=25)
    parser.add_argument("--session", required=True)
    parser.add_argument("--poll", type=int, default=300)
    args = parser.parse_args()
    return watch(args.run_dir, schedule_epochs(args.schedule), args.envs, args.session, args.poll)


if __name__ == "__main__":
    sys.exit(main())
