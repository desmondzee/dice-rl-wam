import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval import seed_all
from script.lingbot_rl_policy import env_action_count
from script.lingbot_rl_train import make_env, object_state

DEMO_TOLERANCE = 0.02
DEMO_MIN_MATCH = 0.9


def demo_path(demos_root, task_name):
    return Path(demos_root) / "libero_10" / f"{task_name}_demo.hdf5"


def match_demo(handle, episode):
    actions = np.asarray(episode["actions"], np.float32)
    keys = []
    for key in handle["data"]:
        candidate = np.asarray(handle["data"][key]["actions"], np.float32)
        if candidate.shape == actions.shape and float(np.abs(candidate - actions).max(initial=0.0)) < 1e-5:
            keys.append(key)
    if len(keys) != 1:
        raise ValueError(f"demo for episode {episode['episode_index']} not found")
    return keys[0]


def _gripper_position(inner):
    return inner.sim.data.site_xpos[inner.sim.model.site_name2id("gripper0_grip_site")]


def demo_object_states(env, layout, handle, key, episode):
    states = np.asarray(handle["data"][key]["states"])
    dataset = np.asarray(episode["states"], np.float32)
    inner = env._env.env
    offsets = [0]
    first = True
    while offsets[-1] < len(episode["actions"]):
        offsets.append(offsets[-1] + min(env_action_count(first), len(episode["actions"]) - offsets[-1]))
        first = False
    boundaries, hits = [], []
    for offset in offsets:
        inner.sim.set_state_from_flattened(states[min(offset, len(states) - 1)])
        inner.sim.forward()
        boundaries.append(object_state(inner, layout))
        target = dataset[min(offset, len(dataset) - 1)][:3]
        hits.append(float(np.abs(np.asarray(_gripper_position(inner), np.float32) - target).max()) <= DEMO_TOLERANCE)
    matched = float(np.mean(hits))
    if matched < DEMO_MIN_MATCH:
        raise ValueError(f"demo {episode['episode_index']} states disagree with the dataset: matched {matched:.2f}")
    return {"object_states": np.stack(boundaries).astype(np.float32), "matched": matched}


def _fingerprint(episodes, norm, task_id, layout):
    payload = {"episodes": sorted(int(episode["episode_index"]) for episode in episodes), "task_id": int(task_id),
               "layout": None if layout is None else [list(layout["bodies"]), list(layout["joints"])],
               "q01": [float(x) for x in norm["q01"]], "q99": [float(x) for x in norm["q99"]]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expert_object_states(demos_root, task_name, suite, task_id, episodes, norm, cache_path, layout):
    import h5py

    fingerprint = _fingerprint(episodes, norm, task_id, layout)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            return payload["states"]
    env = make_env(task_id, suite, init_states=True, proprio=True)
    try:
        seed_all(0)
        env.reset(seed=0)
        with h5py.File(demo_path(demos_root, task_name), "r") as handle:
            states = {int(episode["episode_index"]): demo_object_states(env, layout, handle, match_demo(handle, episode), episode)["object_states"]
                      for episode in episodes}
    finally:
        env.close()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"fingerprint": fingerprint, "states": states}, cache_path)
    return states


def attach_object_states(rows, states):
    expert = [row for row in rows if float(row["is_expert"]) == 1.0]
    boundaries = np.concatenate([np.stack([value[:-1], value[1:]], axis=1) for value in states.values()])
    if len(boundaries) != len(expert):
        raise ValueError(f"expected {len(boundaries)} expert rows, got {len(expert)}")
    for row, (before, after) in zip(expert, boundaries):
        row["object_state"] = np.array(before, copy=True)
        row["object_state_next"] = np.array(after, copy=True)
    return rows
