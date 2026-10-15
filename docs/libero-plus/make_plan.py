import argparse
import hashlib
import json
from pathlib import Path

from libero.libero import benchmark, get_libero_path

BASE = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
INSTRUCTION = "put the yellow and white mug in the microwave and close it"
AXES = {"Robot Initial States": "robot_init", "Objects Layout": "layout"}
COUNTS = {"robot_init": 42, "layout": 39}
REPEATS = {"robot_init": 10, "layout": 9}
NOTE = (
    "LIBERO-Plus ships no evaluation harness; this reset rule is our reading of the benchmark. "
    "robot_init entries (reset_rule variant_pose_sampled_objects): LiberoEnv(init_states=False, is_libero_plus=True); "
    "the arm starts at the variant's MountedPandaN.init_qpos and the simulator samples the object layout from the BDDL "
    "regions with np.random seeded by the entry seed (LiberoEnv.reset(seed) -> env.seed -> np.random.seed), so robot_init "
    "episodes are paired across policies by seed, not by canonical init state; init_state_id is null. Each episode uses a freshly "
    "constructed LiberoEnv and exactly one reset: the first seeded reset of a fresh env is reproducible, repeated resets of one "
    "env instance are not. "
    "layout entries (reset_rule stored_state): LiberoEnv(init_states=True) with the variant's single stored state (init_state_id 0). "
    "Seeds follow the standard episode_plan formula sha256(json.dumps(['libero_10', 42, 9, slot] + [1] * repeat))[:4] "
    "with slot = the variant's rank within its axis (ascending suite index); the 19 repeats are pass 2 (repeat = 1). "
    "The policy receives the base task's instruction; the variants' task.language carries a suffix and is not used. "
    "10 settle steps, 520-step horizon, relative control at 20 Hz, 128x128 cameras as in the standard protocol."
)


def seed_for(slot, repeat):
    identity = ["libero_10", 42, 9, slot] + [1] * repeat
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:4], "big")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    classes = json.loads((Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json").read_text())["libero_10"]
    variants = {axis: [] for axis in AXES.values()}
    for index, task in enumerate(suite.tasks):
        entry = classes[index]
        if entry["id"] != index + 1 or entry["name"] != task.name:
            raise ValueError(f"task_classification.json does not match the suite at {index}")
        if task.name.startswith(BASE + "_") and entry["category"] in AXES:
            variants[AXES[entry["category"]]].append((index, task.name, entry["difficulty_level"]))
    if {axis: len(rows) for axis, rows in variants.items()} != COUNTS:
        raise ValueError("Unexpected variant counts")
    episodes = []

    def add(axis, rank, repeat):
        index, name, difficulty = variants[axis][rank]
        robot = axis == "robot_init"
        episodes.append({
            "episode_index": len(episodes), "task_id": 9, "suite": "libero_10", "variant_index": index,
            "variant_name": name, "axis": axis, "difficulty": difficulty, "init_state_id": None if robot else 0,
            "seed": seed_for(rank, repeat), "reset_rule": "variant_pose_sampled_objects" if robot else "stored_state",
        })

    for axis in variants:
        for rank in range(len(variants[axis])):
            add(axis, rank, 0)
    for axis in variants:
        for rank in range(REPEATS[axis]):
            add(axis, rank, 1)
    plan = {"note": NOTE, "suite": "libero_10", "task_id": 9, "base_task": BASE, "instruction": INSTRUCTION,
            "seed": 42, "libero_plus_revision": "4976dc3", "variant_counts": COUNTS, "repeats": REPEATS, "episodes": episodes}
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(len(episodes), [episodes[i]["variant_index"] for i in (0, 41, 42, 80, 81, 90, 91, 99)])


if __name__ == "__main__":
    main()
