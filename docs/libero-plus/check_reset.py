import json
import sys

import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite

from script.lingbot_eval import seed_all

ENV_KWARGS = dict(task_suite_name="libero_10", is_libero_plus=True, episode_length=520, observation_height=128,
                  observation_width=128, obs_type="pixels", n_envs=1, num_steps_wait=10, control_freq=20,
                  control_mode="relative", hard_reset=True)
SUITE = _get_suite("libero_10")


def snapshot(env, seed):
    seed_all(seed)
    observation, _ = env.reset(seed=seed)
    inner = env._env.env
    robot = inner.robots[0]
    return {
        "pixels": np.concatenate([observation["pixels"]["image"], observation["pixels"]["image2"]], axis=1).astype(int),
        "bodies": {name: inner.sim.data.body_xpos[index].round(6).tolist() for name, index in inner.obj_body_id.items()},
        "joints": inner.sim.data.qpos[robot._ref_joint_pos_indexes].round(6).tolist(),
        "init_qpos": np.asarray(robot.robot_model.init_qpos).round(6).tolist(),
        "robot_model": type(robot.robot_model).__name__,
    }


def fresh(variant, seed):
    env = LiberoEnv(task_suite=SUITE, task_id=variant, init_states=False, **ENV_KWARGS)
    try:
        return snapshot(env, seed)
    finally:
        env.close()


def compare(a, b):
    diff = {name: float(np.abs(np.subtract(a["bodies"][name], b["bodies"][name])).max()) for name in a["bodies"]}
    return {"max_pixel_diff": int(np.abs(a["pixels"] - b["pixels"]).max()), "max_body_pos_diff": max(diff.values()),
            "max_joint_diff": float(np.abs(np.subtract(a["joints"], b["joints"])).max()), "body_pos_diff": diff}


def main():
    variant, seed_a, seed_b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    first, second, other = fresh(variant, seed_a), fresh(variant, seed_a), fresh(variant, seed_b)
    env = LiberoEnv(task_suite=SUITE, task_id=variant, init_states=False, **ENV_KWARGS)
    try:
        repeated = [snapshot(env, seed_a), snapshot(env, seed_a)]
    finally:
        env.close()
    print(json.dumps({
        "variant": variant, "robot_model": first["robot_model"], "init_qpos": first["init_qpos"],
        "joints_after_reset": {"seed_a_1": first["joints"], "seed_a_2": second["joints"], "seed_b": other["joints"]},
        "bodies_seed_a": first["bodies"], "bodies_seed_b": other["bodies"],
        "fresh_env_same_seed": compare(first, second), "fresh_env_different_seed": compare(first, other),
        "same_env_first_vs_second_reset": compare(repeated[0], repeated[1]),
        "same_env_first_reset_vs_fresh": compare(repeated[0], first),
    }, indent=2))


if __name__ == "__main__":
    main()
