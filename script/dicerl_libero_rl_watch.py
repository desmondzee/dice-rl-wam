import argparse
import json
import os
import re
import subprocess
import sys
import time

from script.dicerl_libero_watch import save_rows, session_alive

ENV_STEPS_PER_ITERATION = 64


def checkpoint_iterations(run_dir, every):
    found = [int(m.group(1)) for f in os.listdir(run_dir) if (m := re.fullmatch(r"model_step_(\d+)\.pth", f))]
    return sorted(n for n in found if n % every == 0)


def evaluate(run_dir, iteration, states_path, envs):
    out = os.path.join(run_dir, f"heldout_{iteration}")
    ckpt = os.path.join(run_dir, f"model_step_{iteration}.pth")
    subprocess.run([sys.executable, "-m", "script.dicerl_libero_rl_eval", "--ckpt", ckpt, "--states", states_path, "--seeds", "0", "--envs", str(envs), "--out", out], check=True)
    with open(os.path.join(out, "summary.json")) as f:
        return json.load(f)


def watch(run_dir, every, states_path, envs, session, poll):
    run_dir = str(run_dir)
    sweep_path = os.path.join(run_dir, "heldout_sweep.json")
    rows = json.load(open(sweep_path)) if os.path.exists(sweep_path) else []
    done = {r["iteration"] for r in rows}
    while True:
        alive = session_alive(session)
        for iteration in checkpoint_iterations(run_dir, every):
            if iteration in done:
                continue
            summary = evaluate(run_dir, iteration, states_path, envs)
            row = {"iteration": iteration, "env_steps": iteration * ENV_STEPS_PER_ITERATION, "success_rate": summary["success_rate"], "stderr": summary["stderr"]}
            rows.append(row)
            done.add(iteration)
            save_rows(sweep_path, rows)
            print(json.dumps(row), flush=True)
        if not alive:
            return 0
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--every", type=int, default=2000)
    parser.add_argument("--states", required=True)
    parser.add_argument("--envs", type=int, default=25)
    parser.add_argument("--session", required=True)
    parser.add_argument("--poll", type=int, default=300)
    args = parser.parse_args()
    return watch(args.run_dir, args.every, args.states, args.envs, args.session, args.poll)


if __name__ == "__main__":
    sys.exit(main())
