import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
from lerobot.envs.libero import LiberoEnv, _get_suite
from PIL import Image

from script.lingbot_eval import (
    file_sha256, load_policy, read_checkpoint_metadata, read_json, video_frame, write_json,
)
from script.lingbot_rl_train import run_rl_episode

ENV_KWARGS = dict(task_suite_name="libero_10", is_libero_plus=True, episode_length=520, observation_height=128,
                  observation_width=128, obs_type="pixels", n_envs=1, num_steps_wait=10, control_freq=20,
                  control_mode="relative", hard_reset=True)
AXES = ("robot_init", "layout")


def read_prepared(path):
    data = read_json(path)
    return read_json(data["prepared_path"]) if "prepared_path" in data else data


def residual_layout(payload):
    from script.lingbot_rl_model import DiceResidualModel

    if payload is None:
        return {"state_features": "pool", "proprio": False}
    return DiceResidualModel.layout_of(payload)


def load(args, prepared, architecture, payload, layout):
    if args.policy == "sft":
        return load_policy(prepared["checkpoint"], prepared["model_path"], architecture, -1, args.video_steps, args.action_steps)
    from script.lingbot_rl_model import DiceResidualModel
    from script.lingbot_rl_policy import load_residual_policy

    policy = load_residual_policy(prepared["checkpoint"], prepared["model_path"], architecture, -1, args.video_steps, args.action_steps)
    model = DiceResidualModel(device="cuda", residual_input=args.residual_input, epsilon=args.epsilon,
                              plan_candidates=args.plan_candidates, **layout)
    model.load_inference_state_dict(payload)
    policy.residual_model = model
    policy.eval_candidates = args.eval_candidates
    policy.eval_plans = args.plan_candidates
    policy.eval_plan_ranker = args.plan_ranker
    return policy


def instrument(env, frame_path):
    original = env.reset
    record = {}

    def reset(seed=None, **kwargs):
        observation, info = original(seed=seed, **kwargs)
        inner = env._env.env
        robot = inner.robots[0]
        record.update(
            robot_model=type(robot.robot_model).__name__,
            init_qpos=[float(v) for v in robot.robot_model.init_qpos],
            joint_pos_after_reset=[float(v) for v in inner.sim.data.qpos[robot._ref_joint_pos_indexes]],
            object_positions={name: [float(v) for v in inner.sim.data.body_xpos[index]] for name, index in inner.obj_body_id.items()},
        )
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(video_frame(observation)).save(frame_path)
        return observation, info

    env.reset = reset
    return record


def summarize(rows, episodes, out):
    summary = {"expected_episodes": len(episodes), "completed_episodes": len(rows), "per_axis": {}}
    for axis in AXES:
        done = [row for row in rows if row["axis"] == axis]
        successes = sum(row["success"] for row in done)
        summary["per_axis"][axis] = {
            "expected_episodes": sum(entry["axis"] == axis for entry in episodes), "completed_episodes": len(done),
            "successes": successes, "success_rate": successes / len(done) if done else None}
    summary["successes"] = sum(row["success"] for row in rows)
    summary["success_rate_completed"] = summary["successes"] / len(rows) if rows else None
    summary["complete"] = len(rows) == len(episodes)
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--policy", choices=("sft", "rl"), required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--residual")
    parser.add_argument("--residual-input", default="z")
    parser.add_argument("--epsilon", type=float, default=-0.5)
    parser.add_argument("--plan-candidates", type=int, default=1)
    parser.add_argument("--plan-ranker", choices=("v_plan", "q", "random"), default="v_plan")
    parser.add_argument("--eval-candidates", type=int, default=4)
    parser.add_argument("--video-steps", type=int, default=5)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.policy == "rl" and not args.residual:
        parser.error("--residual is required for --policy rl")
    plan = read_json(args.plan)
    episodes = plan["episodes"]
    prepared = read_prepared(args.prepared)
    metadata = read_checkpoint_metadata(prepared["checkpoint"])
    instruction = next(task["instruction"] for task in prepared["tasks"] if task["task_id"] == plan["task_id"])
    if instruction != plan["instruction"]:
        raise ValueError("Plan instruction differs from the prepared base task")
    payload = torch.load(args.residual, map_location="cpu", weights_only=True) if args.policy == "rl" else None
    layout = residual_layout(payload)
    settings = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume"},
                "checkpoint": prepared["checkpoint"], "instruction": instruction, "plan_sha256": file_sha256(args.plan),
                "driver_sha256": file_sha256(__file__), "residual_sha256": file_sha256(args.residual) if args.residual else None, "layout": layout}
    settings_path = args.out / "settings" / "run.json"
    if settings_path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; pass --resume or choose a new --out")
        if read_json(settings_path) != settings:
            raise ValueError("Resume requires identical plan, policy, and driver")
    else:
        write_json(settings_path, settings)
        write_json(args.out / "settings" / "plan.json", plan)

    def path_of(entry):
        return args.out / "episodes" / f"task_{entry['task_id']:02d}" / f"episode_{entry['episode_index']:03d}.json"

    rows = []
    for entry in episodes:
        if path_of(entry).exists():
            row = read_json(path_of(entry))
            if any(row.get(key) != value for key, value in entry.items()):
                raise ValueError("Saved episode identity differs from the plan")
            rows.append(row)
    done = {row["episode_index"] for row in rows}
    summary = summarize(rows, episodes, args.out)
    if summary["complete"]:
        print(json.dumps(summary, indent=2))
        return
    torch.cuda.reset_peak_memory_stats()
    policy = load(args, prepared, metadata["architecture"], payload, layout)
    suite = _get_suite("libero_10")
    for entry in episodes:
        if entry["episode_index"] in done:
            continue
        env = LiberoEnv(task_suite=suite, task_id=entry["variant_index"], init_states=entry["reset_rule"] == "stored_state",
                        **{**ENV_KWARGS, "obs_type": "pixels_agent_pos" if layout["proprio"] else "pixels"})
        record = instrument(env, args.out / "frames" / f"episode_{entry['episode_index']:03d}.png")
        writer = None
        video = args.out / "videos" / f"task_{entry['task_id']:02d}.mp4"
        if entry["episode_index"] == 0:
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video), fps=20, codec="libx264", macro_block_size=16)
        try:
            row = run_rl_episode(policy, env, entry, instruction, metadata["normalization"], "cuda", layout["proprio"], video_writer=writer)
        finally:
            if writer is not None:
                writer.close()
            env.close()
        row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        row["reset"] = record
        if writer is not None:
            row["video"] = str(video.relative_to(args.out))
        rows.append(row)
        write_json(path_of(entry), row)
        summary = summarize(rows, episodes, args.out)
        print(json.dumps({key: row[key] for key in ("episode_index", "variant_index", "axis", "success", "policy_steps", "seconds")}), flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
