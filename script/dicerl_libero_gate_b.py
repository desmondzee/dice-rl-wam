import argparse
import json
import os
import sys

import numpy as np

MAGNITUDES = (0.1, 0.3, 0.5, 0.6)
DRAWS = 200
MAX_REJECTION_RATE = 0.2
GATED_MAGNITUDE = 0.6
SHAPE_META = {"obs": {"rgb": {"shape": [128, 128, 6]}, "state": {"shape": [8]}}, "action": {"shape": [7]}}


def quantiles(values):
    if len(values) == 0:
        return None
    q = np.percentile(values, [10, 50, 90])
    return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def eef(wrapper):
    return np.array(wrapper.last_raw_obs["robot0_eef_pos"], dtype=np.float64)


def shift_part(task, normalization_path):
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper, libero_task_files, make_libero_env
    _, states = libero_task_files(task)
    env = make_libero_env(task, hard_reset=False)
    wrapper = LiberoImageWrapper(env, SHAPE_META, normalization_path=normalization_path, init_states=states, hard_reset=False)
    drawn = []
    original = wrapper.draw_start

    def recording(state, magnitude, rng):
        out = original(state, magnitude, rng)
        drawn.append(out)
        return out

    wrapper.draw_start = recording
    try:
        baseline, jitter = {}, []
        for i in range(len(states)):
            wrapper.reset(options={"init_state": states[i]})
            baseline[i] = eef(wrapper)
            wrapper.reset(options={"init_state": states[i]})
            jitter.append(float(np.linalg.norm(eef(wrapper) - baseline[i])))
        result = {"task": task, "draws": DRAWS, "baseline_eef_jitter": quantiles(jitter), "magnitudes": {}}
        for magnitude in MAGNITUDES:
            wrapper.set_shift_seed(1)
            wrapper.set_shift(magnitude)
            wrapper.shift_stats()
            norms, displacements, rejected_draws, rejections, fallbacks = [], [], 0, 0, 0
            for k in range(DRAWS):
                i = k % len(states)
                drawn.clear()
                wrapper.reset(options={"seed": i})
                stats = wrapper.shift_stats()
                rejections += stats["rejections"]
                fallbacks += stats["fallbacks"]
                rejected_draws += stats["rejections"] > 0
                if not stats["fallbacks"]:
                    norms.append(float(np.linalg.norm(drawn[0][1:8] - states[i][1:8])))
                displacements.append(float(np.linalg.norm(eef(wrapper) - baseline[i])))
            wrapper.set_shift(None)
            tries = rejections + DRAWS - fallbacks
            result["magnitudes"][str(magnitude)] = {
                "rejections": rejections, "fallbacks": fallbacks, "tries": tries,
                "rejection_rate": rejections / tries, "draws_with_rejection": int(rejected_draws),
                "joint_norm": quantiles(norms), "eef_displacement": quantiles(displacements),
            }
            print(magnitude, json.dumps(result["magnitudes"][str(magnitude)]), flush=True)
    finally:
        env.close()
    return result


def failures(result):
    rate = result["magnitudes"][str(GATED_MAGNITUDE)]["rejection_rate"]
    return [f"rejection rate at {GATED_MAGNITUDE}: {rate}"] if rate > MAX_REJECTION_RATE else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    normalization_path = os.path.join(os.environ["DICE_RL_DATA_DIR"], "libero", args.task, "ph_pretrain_20", "normalization.npz")
    result = shift_part(args.task, normalization_path)
    result["failures"] = failures(result)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate_b.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["failures"] else 0)


if __name__ == "__main__":
    main()
