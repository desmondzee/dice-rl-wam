import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from script.lingbot_eval import seed_all
from script.lingbot_rl_policy import env_action_count
from script.lingbot_rl_train import make_env, object_state, proprio_from_observation

REPLAY_TOLERANCE = 0.05
REPLAY_MIN_MATCH = 0.9


def _reset(env, index=None):
    if index is not None:
        env.init_state_id = index
    seed_all(0)
    observation, _ = env.reset(seed=0)
    return observation


def _gap(observation, state):
    return float(np.abs(proprio_from_observation(observation) - np.asarray(state, np.float32)).max())


def match_initial_state(env, first_proprio):
    if getattr(env, "_reset_proprios", None) is None:
        env._reset_proprios = np.stack([proprio_from_observation(_reset(env, index)) for index in range(len(env._init_states))])
    gaps = np.abs(env._reset_proprios - np.asarray(first_proprio, np.float32)[None]).max(axis=1)
    index = int(np.argmin(gaps))
    gap = float(gaps[index])
    if gap > REPLAY_TOLERANCE:
        raise ValueError(f"no stored state within {REPLAY_TOLERANCE} of the demo: {gap:.3f}")
    return index


def replay_episode(env, layout, episode, norm):
    actions = np.asarray(episode["actions"], np.float32)
    states = np.asarray(episode["states"], np.float32)
    index = match_initial_state(env, states[0]) if getattr(env, "_init_states", None) is not None else None
    observation = _reset(env, index)
    inner = env._env.env
    boundaries = [object_state(inner, layout)]
    hits = [_gap(observation, states[0]) <= REPLAY_TOLERANCE]
    offset = 0
    first = True
    while offset < len(actions):
        n_env = min(env_action_count(first), len(actions) - offset)
        for action in actions[offset:offset + n_env]:
            observation, *_ = env.step(np.asarray(action, np.float32))
        offset += n_env
        first = False
        boundaries.append(object_state(inner, layout))
        if offset < len(states):
            hits.append(_gap(observation, states[offset]) <= REPLAY_TOLERANCE)
    matched = float(np.mean(hits))
    if matched < REPLAY_MIN_MATCH:
        raise ValueError(f"demo {episode['episode_index']} replay drifted: matched {matched:.2f}")
    return {"object_states": np.stack(boundaries).astype(np.float32), "matched": matched}


def _fingerprint(episodes, norm, task_id, layout):
    payload = {"episodes": sorted(int(episode["episode_index"]) for episode in episodes), "task_id": int(task_id),
               "layout": None if layout is None else [list(layout["bodies"]), list(layout["joints"])],
               "q01": [float(x) for x in norm["q01"]], "q99": [float(x) for x in norm["q99"]]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expert_object_states(suite, task_id, episodes, norm, cache_path, layout):
    fingerprint = _fingerprint(episodes, norm, task_id, layout)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            return payload["states"]
    env = make_env(task_id, suite, init_states=True, proprio=True)
    try:
        states = {int(episode["episode_index"]): replay_episode(env, layout, episode, norm)["object_states"] for episode in episodes}
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
