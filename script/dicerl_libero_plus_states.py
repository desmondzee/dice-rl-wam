import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

CANONICAL_JOINTS = np.array([-0.031, -0.133, 0.015, -2.439, 0.0, 2.207, 0.778])
FIRST_VARIANT, N_VARIANTS, N_STATES, MONITOR_STATES = 640, 42, 7, 2
OFFSET_RANGE = (0.1, 0.5)


def build(canonical, init_qpos, variants):
    rows, labels, monitor = [], [], []
    for v, qpos in zip(variants, init_qpos):
        for s in range(N_STATES):
            state = np.array(canonical[s], dtype=float)
            state[1:8] = qpos
            monitor.append(s < MONITOR_STATES)
            rows.append(state)
            labels.append(f"{int(v)}:{s}")
    rows, labels, monitor = np.stack(rows), np.array(labels), np.array(monitor)
    return {"states": rows, "labels": labels}, {"states": rows[monitor], "labels": labels[monitor]}


def by_variant(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"].split(":")[0]].append(str(row["success"]) == "True")
    return {variant: float(np.mean(values)) for variant, values in groups.items()}


def check_offsets(init_qpos, reference, name):
    offsets = np.linalg.norm(init_qpos - reference, axis=1)
    distinct = len({tuple(np.round(q, 9)) for q in init_qpos})
    print(f"poses {len(init_qpos)} distinct {distinct} offset from {name} min {offsets.min():.4f} max {offsets.max():.4f}")
    return distinct == len(init_qpos) and OFFSET_RANGE[0] - 1e-6 <= offsets.min() and offsets.max() <= OFFSET_RANGE[1] + 1e-6


def export(out):
    from libero.libero import benchmark
    from libero.libero.envs import robots
    suite = benchmark.get_benchmark("libero_10")()
    variants = np.arange(FIRST_VARIANT, FIRST_VARIANT + N_VARIANTS)
    qpos = []
    for v in variants:
        name = suite.get_task(int(v)).name
        match = re.search(r"initstate_(\d+)$", name)
        if match is None or not name.startswith("KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close"):
            raise SystemExit(f"variant {v} is not a robot-init variant of the task: {name}")
        qpos.append(getattr(robots, f"MountedPanda{int(match.group(1))}").init_qpos.fget(None))
    qpos = np.stack(qpos)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = robots.MountedPanda.init_qpos.fget(None)
    np.savez(out, init_qpos=qpos, variants=variants, base_qpos=base)
    print(f"wrote {out}")
    check_offsets(qpos, base, "MountedPanda.init_qpos")
    check_offsets(qpos, CANONICAL_JOINTS, "canonical state 0")


def build_files(qpos_path, task, out_dir):
    from env.gym_utils.wrapper.libero_image import libero_task_files
    canonical = libero_task_files(task)[1]
    data = np.load(qpos_path)
    check_offsets(data["init_qpos"], canonical[0, 1:8], "canonical state 0")
    if not check_offsets(data["init_qpos"], data["base_qpos"], "MountedPanda.init_qpos"):
        raise SystemExit("poses are not distinct or their offsets from the base robot pose fall outside the expected range")
    full, monitor = build(canonical, data["init_qpos"], data["variants"])
    rows = np.linalg.norm(full["states"][:, 1:8] - np.tile(canonical[:N_STATES, 1:8], (len(data["init_qpos"]), 1)), axis=1)
    print(f"offsets of the {len(rows)} rows from their own canonical state: min {rows.min():.4f} max {rows.max():.4f}")
    for name, content in (("plus_robot_init.npz", full), ("plus_robot_init_monitor.npz", monitor)):
        np.savez(os.path.join(out_dir, name), **content)
        print(f"wrote {os.path.join(out_dir, name)} {content['states'].shape}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("--out", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--qpos", required=True)
    build_parser.add_argument("--task", required=True)
    build_parser.add_argument("--out-dir", required=True)
    commands.add_parser("breakdown").add_argument("--episodes", required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.out)
    elif args.command == "build":
        build_files(args.qpos, args.task, args.out_dir)
    else:
        with open(args.episodes) as f:
            for variant, rate in by_variant(csv.DictReader(f)).items():
                print(f"{variant}\t{rate:.3f}")


if __name__ == "__main__":
    main()
