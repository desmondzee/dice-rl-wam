import argparse
import json
import os

import h5py
import numpy as np

FLIP_AXES = {"none": (), "vertical": (1,), "both": (1, 2)}


def oriented(frames, flip):
    return np.flip(frames, axis=FLIP_AXES[flip]) if FLIP_AXES[flip] else frames


def normalize(x, lo, hi):
    span = hi - lo
    out = 2 * (x - lo) / np.where(span == 0, 1, span) - 1
    return np.where(span == 0, 0.0, out).astype(np.float32)


def convert(hdf5_path, out_dir, flip="vertical"):
    if flip not in FLIP_AXES:
        raise ValueError(f"Unknown flip {flip!r}")
    keys = ("states", "actions", "images", "rewards", "terminals")
    parts = {k: [] for k in keys}
    lengths = []
    with h5py.File(hdf5_path, "r") as f:
        for name in sorted(f["data"], key=lambda n: int(n.split("_")[1])):
            g = f["data"][name]
            o = g["obs"]
            parts["states"].append(np.concatenate([o["ee_pos"][:], o["ee_ori"][:], o["gripper_states"][:]], -1))
            parts["actions"].append(g["actions"][:])
            agent, wrist = oriented(o["agentview_rgb"][:], flip), oriented(o["eye_in_hand_rgb"][:], flip)
            parts["images"].append(np.concatenate([agent, wrist], -1).transpose(0, 3, 1, 2))
            parts["rewards"].append(g["rewards"][:])
            parts["terminals"].append(g["dones"][:])
            lengths.append(len(g["actions"]))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    stats = {
        "obs_min": data["states"].min(0), "obs_max": data["states"].max(0),
        "action_min": data["actions"].min(0), "action_max": data["actions"].max(0),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        states=normalize(data["states"], stats["obs_min"], stats["obs_max"]),
        actions=normalize(data["actions"], stats["action_min"], stats["action_max"]),
        images=np.ascontiguousarray(data["images"], dtype=np.uint8),
        rewards=data["rewards"].astype(np.float32),
        terminals=data["terminals"].astype(bool),
        traj_lengths=np.array(lengths, dtype=np.int64),
    )
    np.savez_compressed(os.path.join(out_dir, "normalization.npz"), **{k: v.astype(np.float32) for k, v in stats.items()})
    return {"episodes": len(lengths), "steps": int(sum(lengths))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--flip", default="vertical", choices=sorted(FLIP_AXES))
    args = parser.parse_args()
    print(json.dumps(convert(args.hdf5, args.out, args.flip)))


if __name__ == "__main__":
    main()
