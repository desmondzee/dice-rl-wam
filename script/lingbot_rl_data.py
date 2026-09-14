import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_model import ACTION_DIM, HORIZON, USED_DOF, mask_unused_dof
from script.lingbot_rl_policy import env_action_count

EXPERT_RECIPE = "dice-rl-expert-v1"
EPS = 1e-6


def normalize_demo_action(raw, norm):
    values = np.asarray(raw, dtype=np.float32)
    low = np.asarray(norm["q01"][:USED_DOF], dtype=np.float32)
    high = np.asarray(norm["q99"][:USED_DOF], dtype=np.float32)
    return 2.0 * (values - low) / (high - low + EPS) - 1.0


def expert_fingerprint(manifest, norm, episodes=None):
    proto = RLConfig().protocol()
    payload = {
        "recipe": EXPERT_RECIPE,
        "video_steps": proto["video_steps"],
        "action_steps": proto["action_steps"],
        "video_exec_step": proto["video_exec_step"],
        "checkpoint_step": 600,
        "source_run": "libero30-sft",
        "q01": [float(x) for x in norm["q01"]],
        "q99": [float(x) for x in norm["q99"]],
    }
    if manifest is not None:
        payload["manifest"] = manifest.get("fingerprint", manifest)
    elif episodes is not None:
        payload["episodes"] = [
            {"task": ep.get("task"), "task_id": int(ep.get("task_id", -1)), "n": int(len(ep["actions"]))}
            for ep in episodes
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_state(state):
    if torch.is_tensor(state):
        state = state.detach().float().cpu().numpy()
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 2:
        state = state[0]
    return state


def _chunk_actions(raw_7, first_chunk, norm):
    take = raw_7.shape[0]
    padded = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    start = 4 if first_chunk else 0
    padded[start:start + take, :USED_DOF] = normalize_demo_action(raw_7, norm)
    chunk = mask_unused_dof(torch.from_numpy(padded)).numpy().astype(np.float32)
    return chunk


def _critic_batch(episode, frame_index, device):
    frames = episode.get("frames") or []
    frame = frames[frame_index] if frame_index < len(frames) else None
    if isinstance(frame, dict) and "pixels" in frame:
        from script.lingbot_eval import observation_batch
        return observation_batch(frame, episode["task"], device)
    return {"task": [episode.get("task", "")]}


def _episode_rows(policy, episode, norm):
    actions = np.asarray(episode["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != USED_DOF:
        raise ValueError("Expert episodes must provide finite (T, 7) actions")
    device = "cpu"
    if hasattr(policy, "config") and getattr(policy.config, "device", None):
        device = policy.config.device
    policy.reset()
    rows = []
    offset = 0
    first = True
    success = bool(episode.get("success", True))
    while offset < len(actions):
        n_env = min(env_action_count(first), len(actions) - offset)
        raw = actions[offset:offset + n_env]
        chunk = _chunk_actions(raw, first, norm)
        if episode.get("latents") is not None:
            latents = episode["latents"]
            if latents.ndim != 5:
                raise ValueError("Published expert latents must be shaped (B, C, T, H, W)")
            frame_count = latents.shape[2]
            index = min(offset // 4, max(frame_count - 1, 0))
            if not hasattr(policy, "extract_critic_state_from_latent"):
                raise TypeError("Policy must pool critic state from published latents")
            state = _tensor_state(
                policy.extract_critic_state_from_latent(
                    latents[:, :, index:index + 1], {"task": [episode["task"]]})
            )
        else:
            state = _tensor_state(policy.extract_critic_state(_critic_batch(episode, offset, device)))
        rows.append({
            "s": state,
            "z": np.zeros((HORIZON, ACTION_DIM), dtype=np.float32),
            "a_base": chunk.copy(),
            "a": chunk.copy(),
            "reward": np.float32(0.0),
            "done": np.float32(0.0),
            "s_next": state.copy(),
            "task_id": int(episode.get("task_id", 0)),
            "n_env_actions": int(n_env),
            "is_expert": np.float32(1.0),
        })
        offset += n_env
        first = False
    if not rows:
        return rows
    for index in range(len(rows) - 1):
        rows[index]["s_next"] = rows[index + 1]["s"].copy()
    rows[-1]["done"] = np.float32(1.0)
    rows[-1]["reward"] = np.float32(1.0 if success else 0.0)
    return rows


def featurize_experts(policy, dataset, manifest=None, norm=None, cache_path=None):
    if norm is None:
        raise ValueError("Checkpoint norm_stats.json is required")
    episodes = list(dataset)
    cache_path = Path(cache_path) if cache_path is not None else None
    fingerprint = expert_fingerprint(manifest, norm, episodes)
    if cache_path is not None and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
            payload = None
        if payload is not None and payload.get("fingerprint") == fingerprint:
            return payload["rows"]
    rows = []
    for episode in episodes:
        rows.extend(_episode_rows(policy, episode, norm))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"fingerprint": fingerprint, "rows": rows, "recipe": EXPERT_RECIPE}, cache_path)
    return rows


def load_manifest_episodes(dataset_root, manifest, task_to_id=None, norm=None):
    """Load the 300 SFT demos listed in the checkpoint manifest from a LeRobot snapshot."""
    import pyarrow.parquet as pq

    if norm is None:
        raise ValueError("Checkpoint norm_stats.json is required")
    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = manifest["episodes"]
    if len(episodes) != 300:
        raise ValueError("Expected 300 SFT demonstrations in the checkpoint manifest")
    if task_to_id is None:
        task_names = []
        for episode in episodes:
            name = episode["tasks"][0]
            if name not in task_names:
                task_names.append(name)
        task_to_id = {name: index for index, name in enumerate(task_names)}
    loaded = []
    for episode in episodes:
        idx = episode["episode_index"]
        chunk = idx // info["chunks_size"]
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=idx)
        parquet_path = root / rel
        try:
            table = pq.read_table(parquet_path, columns=["action", "episode_index", "frame_index"])
        except Exception as exc:
            raise RuntimeError(f"Failed to read episode {idx} parquet {parquet_path}: {exc}") from exc
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.shape[-1] != USED_DOF or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite seven-dimensional LIBERO actions in {parquet_path}")
        latents = _try_load_latents(root, info, episode, actions, norm)
        frames = [] if latents is not None else _load_episode_frames(root, info, episode, table)
        if latents is None and (not frames or frames[0] is None):
            raise ValueError(
                f"Could not load published latents or RGB frames for episode {idx}; "
                "expert critic state requires demo cameras"
            )
        loaded.append({
            "actions": actions,
            "task": episode["tasks"][0],
            "task_id": task_to_id[episode["tasks"][0]],
            "frames": frames,
            "latents": latents,
            "success": True,
            "episode_index": idx,
        })
    return loaded


def _usable_torch_archive(path):
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size < 64:
            return False
        with path.open("rb") as handle:
            magic = handle.read(64)
    except OSError:
        return False
    if magic.startswith(b"version https://git-lfs") or magic.lstrip().startswith(b"<"):
        return False
    return magic.startswith(b"PK") or magic.startswith(b"\x80")


def _try_load_latents(root, info, episode, actions, norm):
    from script.lingbot_sft_data import assemble_streams, episode_paths, load_latent

    _, paths = episode_paths(info, episode)
    files = [root / path for path in paths]
    if not all(_usable_torch_archive(path) for path in files):
        return None
    try:
        assembled = assemble_streams([load_latent(path) for path in files], actions, episode, norm)
        latents = assembled["latents"]
        if latents.ndim != 4:
            raise ValueError("Expected published latents with shape (C, F, H, W)")
        return latents.unsqueeze(0)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        return None


def _read_rgb_video(path):
    import imageio.v2 as imageio

    frames = [np.asarray(frame, dtype=np.uint8) for frame in imageio.get_reader(str(path))]
    stacked = np.stack(frames, axis=0)
    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise ValueError("Expected HxWx3 RGB video frames")
    return stacked


def _load_episode_videos(root, info, episode):
    from script.lingbot_sft_config import CAMERAS as SFT_CAMERAS

    template = info.get("video_path") or (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    chunk = episode["episode_index"] // info["chunks_size"]
    arrays = []
    for camera in SFT_CAMERAS:
        path = root / template.format(
            episode_chunk=chunk, episode_index=episode["episode_index"], video_key=camera)
        try:
            if not path.is_file() or path.stat().st_size < 32:
                return []
            arrays.append(_read_rgb_video(path))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            return []
    if len(arrays) != 2 or arrays[0].shape[0] != arrays[1].shape[0]:
        return []
    return [
        {"pixels": {"image": arrays[0][index], "image2": arrays[1][index]}}
        for index in range(arrays[0].shape[0])
    ]


def _load_episode_frames(root, info, episode, table):
    """RGB frames for critic pooling. Prefer parquet image columns; else video files."""
    from script.lingbot_sft_config import CAMERAS as SFT_CAMERAS

    features = info.get("features", {})
    if all(camera in features and "dtype" in features[camera] for camera in SFT_CAMERAS):
        try:
            import pyarrow.parquet as pq
            idx = episode["episode_index"]
            chunk = idx // info["chunks_size"]
            rel = info["data_path"].format(episode_chunk=chunk, episode_index=idx)
            images = pq.read_table(root / rel, columns=list(SFT_CAMERAS))
            frames = []
            for row in range(images.num_rows):
                pixels = {}
                for short, camera in zip(("image", "image2"), SFT_CAMERAS):
                    pixels[short] = np.asarray(images[camera][row].as_py(), dtype=np.uint8)
                frames.append({"pixels": pixels})
            if frames:
                return frames
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    return _load_episode_videos(root, info, episode)
