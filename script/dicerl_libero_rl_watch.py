import argparse
import json
import os
import re
import subprocess
import sys
import time

from script.dicerl_libero_watch import save_rows, session_alive

ENV_STEPS_PER_ITERATION = 64
MIN_AGE_SECONDS = 30
MAX_ATTEMPTS = 3


def checkpoint_iterations(run_dir, every):
    found = [int(m.group(1)) for f in os.listdir(run_dir) if (m := re.fullmatch(r"model_step_(\d+)\.pth", f))]
    return sorted(n for n in found if n % every == 0)


def evaluate(run_dir, iteration, states_path, envs):
    out = os.path.join(run_dir, f"heldout_{iteration}")
    ckpt = os.path.join(run_dir, f"model_step_{iteration}.pth")
    subprocess.run([sys.executable, "-m", "script.dicerl_libero_rl_eval", "--ckpt", ckpt, "--states", states_path, "--seeds", "0", "--envs", str(envs), "--out", out], check=True)
    with open(os.path.join(out, "summary.json")) as f:
        return json.load(f)


def row_for(iteration, **fields):
    return {"iteration": iteration, "env_steps": iteration * ENV_STEPS_PER_ITERATION, **fields}


def watch(run_dir, every, states_path, envs, session, poll):
    run_dir = str(run_dir)
    sweep_path = os.path.join(run_dir, "heldout_sweep.json")
    rows = json.load(open(sweep_path)) if os.path.exists(sweep_path) else []
    done = {r["iteration"] for r in rows}
    attempts = {}
    while True:
        alive = session_alive(session)
        pending = False
        for iteration in checkpoint_iterations(run_dir, every):
            if iteration in done:
                continue
            path = os.path.join(run_dir, f"model_step_{iteration}.pth")
            if alive and time.time() - os.path.getmtime(path) < MIN_AGE_SECONDS:
                pending = True
                continue
            try:
                summary = evaluate(run_dir, iteration, states_path, envs)
                row = row_for(iteration, success_rate=summary["success_rate"], stderr=summary["stderr"])
            except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
                attempts[iteration] = attempts.get(iteration, 0) + 1
                message = f"{type(error).__name__}: {error}"[:200]
                print(f"evaluation of iteration {iteration} failed (attempt {attempts[iteration]} of {MAX_ATTEMPTS}): {message}", flush=True)
                if attempts[iteration] < MAX_ATTEMPTS:
                    pending = True
                    continue
                row = row_for(iteration, error=message)
            rows.append(row)
            done.add(iteration)
            save_rows(sweep_path, rows)
            print(json.dumps(row), flush=True)
        if not alive and not pending:
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
