import argparse
import json
import os
import time

import numpy as np

FLIPS = ("none", "vertical", "both")
MAX_SCORE = 0.05
MIN_RATIO = 2
CAMERAS = {"agentview": ("agentview_image", "agentview_rgb"), "wrist": ("robot0_eye_in_hand_image", "eye_in_hand_rgb")}


def orient(image, flip):
    return {"none": image, "vertical": image[::-1], "both": image[::-1, ::-1]}[flip]


def camera_scores(rendered, stored):
    return {
        flip: {
            cam: float(np.abs(orient(rendered[cam], flip).astype(np.float64) - stored[cam].astype(np.float64)).mean() / 255)
            for cam in stored
        }
        for flip in FLIPS
    }


def flip_scores(rendered, stored):
    return {flip: max(cams.values()) for flip, cams in camera_scores(rendered, stored).items()}


def pick_flip(rendered, stored):
    scores = flip_scores(rendered, stored)
    ranked = sorted(scores, key=scores.get)
    best, second = scores[ranked[0]], scores[ranked[1]]
    if best >= MAX_SCORE or second <= 0 or best * MIN_RATIO > second:
        raise ValueError(f"No flip clearly matches the stored frames: {scores}")
    best = ranked[0]
    return best, scores


def state_of(raw_obs):
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    return LiberoImageWrapper(None, None, normalization_path=None).get_observation(raw_obs)["state"]


def block_diffs(state, demo_state):
    return {
        "ee_pos": float(np.abs(state[:3] - demo_state[:3]).max()),
        "ee_ori": float(np.abs(state[3:6] - demo_state[3:6]).max()),
        "gripper": float(np.abs(state[6:] - demo_state[6:]).max()),
    }


def equivalent_ori_diff(ori, demo_ori):
    def alt(v):
        n = np.linalg.norm(v)
        return v * (1 - 2 * np.pi / n) if n > 0 else v
    return float(min(np.abs(a - b).max() for a in (ori, alt(ori)) for b in (demo_ori, alt(demo_ori))))


def replay_demo(hdf5_path, task_name, demo="demo_0", resets=8):
    import h5py
    from env.gym_utils.wrapper.libero_image import NOOP, SETTLE_STEPS, make_libero_env
    with h5py.File(hdf5_path, "r") as f:
        g = f["data"][demo]
        states, actions = g["states"][:], g["actions"][:]
        o = g["obs"]
        stored = {cam: o[key][0] for cam, (_, key) in CAMERAS.items()}
        demo_state = np.concatenate([o["ee_pos"][0], o["ee_ori"][0], o["gripper_states"][0]])
    env = make_libero_env(task_name)
    sweep = []
    for _ in range(resets):
        env.reset()
        raw = env.set_init_state(states[0])
        rendered = {cam: raw[key] for cam, (key, _) in CAMERAS.items()}
        sweep.append(camera_scores(rendered, stored))
    env.reset()
    raw = env.set_init_state(states[0])
    rendered = {cam: raw[key] for cam, (key, _) in CAMERAS.items()}
    try:
        flip, scores = pick_flip(rendered, stored)
    except ValueError:
        flip, scores = None, flip_scores(rendered, stored)
    camera_sweep = {
        flip: {cam: {"mean": float(np.mean([r[flip][cam] for r in sweep])), "min": float(np.min([r[flip][cam] for r in sweep]))}
               for cam in stored}
        for flip in FLIPS
    }
    state = state_of(raw)
    state_match = {"after_set_init_state": block_diffs(state, demo_state)}
    state_match["after_set_init_state"]["ee_ori_equivalent"] = equivalent_ori_diff(state[3:6], demo_state[3:6])
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = env.step(NOOP)
    settled = state_of(raw)
    state_match["after_settle"] = block_diffs(settled, demo_state)
    state_match["after_settle"]["ee_ori_equivalent"] = equivalent_ori_diff(settled[3:6], demo_state[3:6])
    env.reset()
    env.set_init_state(states[0])
    first = None
    for t, action in enumerate(actions):
        env.step(action)
        if first is None and env.check_success():
            first = t + 1
    final = bool(env.check_success())
    env.close()
    return {
        "success": first is not None, "steps": int(len(actions)), "first_success_step": first,
        "success_at_end": final, "flip": flip, "flip_scores": scores, "best_flip": min(scores, key=scores.get),
        "flip_scores_per_camera": camera_scores(rendered, stored), "flip_sweep": camera_sweep, "flip_sweep_resets": resets, "state_match": state_match,
        "demo_state": [float(x) for x in demo_state], "wrapper_state": [float(x) for x in state],
    }


def throughput(task_name, n_envs, steps=200):
    from omegaconf import OmegaConf
    from env.gym_utils import make_async
    shape_meta = OmegaConf.create({"obs": {"rgb": {"shape": [128, 128, 6]}, "state": {"shape": [8]}}})
    wrappers = OmegaConf.create({
        "libero_image": {"normalization_path": None, "shape_meta": shape_meta},
        "multi_step": {"n_obs_steps": 1, "n_action_steps": 8, "max_episode_steps": 520, "reset_within_step": True},
    })
    venv = make_async(
        task_name, env_type="libero", num_envs=n_envs, asynchronous=True, max_episode_steps=520,
        wrappers=wrappers, obs_dim=8, action_dim=7, use_image_obs=True, shape_meta=shape_meta,
    )
    try:
        venv.reset_arg([{"seed": i} for i in range(n_envs)])
        rng = np.random.default_rng(0)
        start = time.perf_counter()
        for _ in range(steps):
            venv.step(rng.uniform(-1, 1, (n_envs, 8, 7)).astype(np.float32))
        elapsed = time.perf_counter() - start
    finally:
        venv.close()
    return n_envs * steps * 8 / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4, 20])
    args = parser.parse_args()
    result = {"task": args.task}
    rates = {}
    for n in args.envs:
        rates[str(n)] = throughput(args.task, n)
        print(f"throughput {n} envs: {rates[str(n)]:.1f} env steps/s", flush=True)
    result["throughput_env_steps_per_s"] = rates
    result["replay"] = replay_demo(args.hdf5, args.task)
    if "20" in rates:
        chunk_s = 20 * 8 / rates["20"]
        result["projected_100_episode_eval_s"] = 5 * 65 * chunk_s
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate0.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
