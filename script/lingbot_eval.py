import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from script.lingbot_eval_config import (
    ARCHITECTURE_KEYS, CAMERAS, LIBERO_ASSETS_REPO, LIBERO_ASSETS_REVISION,
    LEROBOT_REVISION, TASK_IDS, EvalConfig, episode_plan, validate_name,
)
from script.lingbot_sft_config import DATASET_REPO, DATASET_REVISION, MODEL_REPO, MODEL_REVISION, UPSTREAM_REVISION, fingerprint


REQUIRED_FILES = (
    "transformer/diffusion_pytorch_model.safetensors", "transformer/config.json",
    "sft_config.json", "norm_stats.json", "dataset_manifest.json",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint)
    for name in REQUIRED_FILES:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint file: {name}")
    manifest = read_json(checkpoint / "dataset_manifest.json")
    expected = {
        "dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION,
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION, "suite": "libero_10",
        "camera_order": ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"],
        "action_channels": list(range(7)), "resolution_per_camera": [128, 128],
    }
    if any(manifest["provenance"].get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint provenance does not match the LIBERO-10 SFT recipe")
    episodes = manifest["episodes"]
    if len(episodes) != 300 or len({ep["episode_index"] for ep in episodes}) != 300:
        raise ValueError("Expected 300 distinct selected demonstrations")
    if any(len(ep["tasks"]) != 1 for ep in episodes):
        raise ValueError("Each selected episode must have exactly one task")
    counts = dict(Counter(ep["tasks"][0] for ep in episodes))
    if len(counts) != 10 or set(counts.values()) != {30} or counts != manifest["task_counts"]:
        raise ValueError("Expected 30 demonstrations for each of 10 tasks")
    norm = read_json(checkpoint / "norm_stats.json")
    if norm != manifest["norm_stat"]:
        raise ValueError("Checkpoint normalization differs from its selected-demo manifest")
    for key in ("q01", "q99"):
        if len(norm[key]) != 30 or not all(math.isfinite(value) for value in norm[key]):
            raise ValueError("Invalid action quantiles")
    if any(high < low for low, high in zip(norm["q01"], norm["q99"])):
        raise ValueError("Action quantiles are reversed")
    architecture = read_json(checkpoint / "transformer/config.json")
    if architecture.get("_class_name") != "WanTransformer3DModel" or architecture.get("pos_embed_seq_len") is not None:
        raise ValueError("Unsupported transformer architecture")
    if architecture["action_dim"] != 30 or architecture["in_channels"] != 48:
        raise ValueError("Unexpected transformer action/video channels")
    return {"manifest": manifest, "normalization": norm, "architecture": architecture}


def validate_transformer_schema(checkpoint, architecture):
    import torch
    from safetensors import safe_open
    from lerobot.policies.lingbot_va.utils import WanTransformer3DModel

    with torch.device("meta"):
        model = WanTransformer3DModel(**{key: architecture[key] for key in ARCHITECTURE_KEYS}, attn_mode="torch")
    expected = model.state_dict()
    weights = Path(checkpoint) / REQUIRED_FILES[0]
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("Native checkpoint tensor names do not match the pinned LeRobot transformer")
        for key, tensor in expected.items():
            saved = handle.get_slice(key)
            if tuple(saved.get_shape()) != tuple(tensor.shape) or saved.get_dtype() != "BF16":
                raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {key}")


def download_snapshot(repo_id, **kwargs):
    import httpx
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(4):
        try:
            return snapshot_download(repo_id, max_workers=2, **kwargs)
        except (HfHubHTTPError, httpx.TransportError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status is not None and status != 429 and not 500 <= status < 600:
                raise RuntimeError(f"Asset preparation failed with HTTP {status}") from None
            if attempt == 3:
                raise RuntimeError("Asset preparation failed after four attempts; cached files are retained") from None
            delay = 30 * 2**attempt
            retry_after = response.headers.get("Retry-After", "") if response is not None else ""
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    from datetime import datetime, timezone
                    from email.utils import parsedate_to_datetime
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, OverflowError):
                        pass
            if not math.isfinite(delay) or delay > 900:
                raise RuntimeError("Asset download cooldown exceeds 900 seconds; retry preparation later") from None
            time.sleep(delay)


def configure_libero(assets_path):
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The Linux hf-libero package is required")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    assets_path = Path(assets_path)
    for path in (root / "bddl_files", root / "init_files", assets_path):
        if not path.is_dir():
            raise ValueError(f"Missing LIBERO package/assets directory: {path}")
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", "/tmp/lingbot-libero-config"))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    write_json(config_dir / "config.yaml", {
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(assets_path),
        "datasets": str(config_dir / "unused-datasets"),
    })
    import libero.libero as package
    if not hasattr(package, "_assets_path_cache"):
        raise RuntimeError("Unexpected hf-libero assets interface")
    package._assets_path_cache = str(assets_path)


def describe_suite(assets_path):
    configure_libero(assets_path)
    from libero.libero import benchmark, get_libero_path
    from lerobot.envs.libero import get_task_init_states

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if suite.get_num_tasks() != 10:
        raise ValueError("LIBERO-10 must contain 10 tasks")
    tasks = []
    for task_id in TASK_IDS:
        task = suite.get_task(task_id)
        state_file = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states = get_task_init_states(suite, task_id)
        tasks.append({
            "task_id": task_id, "name": task.name, "instruction": task.language,
            "initial_state_count": len(states), "initial_states_sha256": file_sha256(state_file),
            "bddl_sha256": file_sha256(bddl_file),
        })
    return suite, tasks


def prepare_evaluation(config, cache_root, checkpoint):
    config.validate()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required in dice-lingbot-hf for CPU asset preparation")
    checkpoint = Path(checkpoint)
    if checkpoint.name != f"step_{config.checkpoint_step:06d}" or checkpoint.parent.parent.name != config.source_run:
        raise ValueError("Checkpoint path differs from the requested run and step")
    metadata = read_checkpoint_metadata(checkpoint)
    validate_transformer_schema(checkpoint, metadata["architecture"])
    cache_root = Path(cache_root)
    model_path = download_snapshot(MODEL_REPO, revision=MODEL_REVISION,
        allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*"], cache_dir=str(cache_root / "hub"), token=token)
    assets_path = download_snapshot(LIBERO_ASSETS_REPO, repo_type="dataset", revision=LIBERO_ASSETS_REVISION,
        cache_dir=str(cache_root / "hub"), token=token)
    for folder in ("vae", "text_encoder"):
        component = Path(model_path) / folder
        if not (component / "config.json").is_file() or not list(component.glob("*.safetensors")):
            raise ValueError(f"Missing frozen {folder} configuration or weights")
        for index in component.glob("*.index.json"):
            if any(not (component / name).is_file() for name in read_json(index)["weight_map"].values()):
                raise ValueError(f"Missing frozen {folder} weight shard")
    tokenizer = Path(model_path) / "tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file() or not any((tokenizer / name).is_file() for name in ("tokenizer.json", "spiece.model")):
        raise ValueError("Missing frozen tokenizer files")
    _, tasks = describe_suite(assets_path)
    if {task["instruction"] for task in tasks} != set(metadata["manifest"]["task_counts"]):
        raise ValueError("LIBERO task instructions differ from the SFT manifest")
    for task in tasks:
        episode_plan(config, task["task_id"], task["initial_state_count"])
    prepared = {
        "checkpoint": str(checkpoint), "source_run": config.source_run,
        "checkpoint_step": config.checkpoint_step,
        "file_sha256": {name: file_sha256(checkpoint / name) for name in REQUIRED_FILES},
        "file_sizes": {name: (checkpoint / name).stat().st_size for name in REQUIRED_FILES},
        "model_path": model_path, "assets_path": assets_path, "tasks": tasks,
        "manifest_fingerprint": metadata["manifest"]["fingerprint"],
        "lerobot_revision": LEROBOT_REVISION, "libero_assets_revision": LIBERO_ASSETS_REVISION,
    }
    output = cache_root / "lingbot-eval" / fingerprint(prepared) / "prepared.json"
    write_json(output, prepared)
    return str(output)


def observation_batch(observation, instruction, device):
    import numpy as np
    import torch

    batch = {"task": [instruction]}
    for short, key in zip(("image", "image2"), CAMERAS):
        image = observation["pixels"][short]
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError("Expected uint8 128x128 RGB camera observations")
        batch[key] = torch.from_numpy(np.ascontiguousarray(image[::-1])).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    return batch


def decode_action(normalized, norm):
    import numpy as np

    if tuple(normalized.shape) != (1, 7):
        raise ValueError("Expected normalized action shape [1,7]")
    values = normalized.detach().float().cpu().numpy()[0]
    low = np.asarray(norm["q01"][:7], dtype=np.float32)
    high = np.asarray(norm["q99"][:7], dtype=np.float32)
    action = (values + 1.0) / 2.0 * (high - low + 1e-6) + low
    if not np.isfinite(action).all():
        raise ValueError("Policy actions must be finite")
    return action.astype(np.float32)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def video_frame(observation):
    import numpy as np
    return np.ascontiguousarray(np.concatenate([observation["pixels"][key][::-1] for key in ("image", "image2")], axis=1))


def run_episode(policy, env, entry, instruction, norm, device, max_steps=520, video_writer=None):
    import numpy as np
    import torch

    if max_steps < 1:
        raise ValueError("Episode length must be positive")
    started = time.perf_counter()
    seed_all(entry["seed"])
    env.init_state_id = entry["init_state_id"]
    observation, _ = env.reset(seed=entry["seed"])
    policy.reset()
    seed_all(entry["seed"])
    if video_writer is not None:
        video_writer.append_data(video_frame(observation))
    success = terminated = truncated = False
    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            batch = observation_batch(observation, instruction, device)
            normalized = policy.select_action(batch)
            action = decode_action(normalized, norm)
            observation, _, terminated, truncated, info = env.step(action)
            if "is_success" not in info or not isinstance(info["is_success"], (bool, np.bool_)):
                raise ValueError("Environment must report a boolean is_success explicitly")
            success = bool(info["is_success"])
            if video_writer is not None:
                video_writer.append_data(video_frame(observation))
            if success or terminated or truncated:
                break
    truncated = bool(truncated or (step == max_steps and not success and not terminated))
    return {**entry, "instruction": instruction, "success": success,
        "policy_steps": step, "terminated": bool(terminated), "truncated": truncated,
        "seconds": time.perf_counter() - started}


def aggregate_results(rows, task_ids=TASK_IDS, episodes_per_task=20):
    per_task = {str(task): {"successes": 0, "completed_episodes": 0, "success_rate": None} for task in task_ids}
    seen = set()
    for row in rows:
        key = (row["task_id"], row["episode_index"])
        if key in seen:
            raise ValueError("Duplicate evaluation episode")
        if row["task_id"] not in task_ids or not 0 <= row["episode_index"] < episodes_per_task or type(row["success"]) is not bool:
            raise ValueError("Invalid evaluation episode record")
        seen.add(key)
        record = per_task[str(row["task_id"])]
        record["completed_episodes"] += 1
        record["successes"] += int(row["success"])
    for record in per_task.values():
        if record["completed_episodes"]:
            record["success_rate"] = record["successes"] / record["completed_episodes"]
    expected = len(task_ids) * episodes_per_task
    complete = len(rows) == expected
    successes = sum(record["successes"] for record in per_task.values())
    return {"complete": complete, "expected_episodes": expected, "completed_episodes": len(rows),
        "successes": successes, "success_rate_completed": successes / len(rows) if rows else None,
        "macro_success_rate": sum(record["success_rate"] for record in per_task.values()) / len(task_ids) if complete else None,
        "per_task": per_task}


def load_policy(checkpoint, model_path, architecture):
    import torch
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy

    class CachedTextPolicy(LingBotVAPolicy):
        def __init__(self, config):
            super().__init__(config)
            self._text_cache = {}

        def _get_t5_prompt_embeds(self, prompt, max_sequence_length):
            key = (tuple([prompt] if isinstance(prompt, str) else prompt), max_sequence_length)
            if key not in self._text_cache:
                self._text_cache[key] = super()._get_t5_prompt_embeds(prompt, max_sequence_length).detach()
            return self._text_cache[key].clone()

    config = LingBotVAConfig(
        **{key: architecture[key] for key in ARCHITECTURE_KEYS},
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)) for key in CAMERAS},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        wan_pretrained_path=str(model_path), text_encoder_device="cpu", device="cuda", dtype="bfloat16",
        attn_mode="torch", image_hflip=False, camera_layout="width_concat", obs_cam_keys=list(CAMERAS),
        height=128, width=128, action_per_frame=4, frame_chunk_size=4, attn_window=30,
        num_inference_steps=20, video_exec_step=-1, action_num_inference_steps=50,
        guidance_scale=5.0, action_guidance_scale=1.0, snr_shift=5.0, action_snr_shift=0.05,
        used_action_channel_ids=list(range(7)), save_predicted_video=False,
    )
    policy = CachedTextPolicy(config)
    state = load_file(str(Path(checkpoint) / REQUIRED_FILES[0]), device="cpu")
    policy.transformer.load_state_dict(state, strict=True, assign=True)
    del state
    return policy.to("cuda").eval().requires_grad_(False)


def evaluate(config, prepared_path, output_dir, run_name, resume=False, commit=None):
    import torch
    import wandb
    import imageio.v2 as imageio

    config.validate()
    validate_name(run_name)
    prepared = read_json(prepared_path)
    if prepared["source_run"] != config.source_run or prepared["checkpoint_step"] != config.checkpoint_step:
        raise ValueError("Prepared checkpoint does not match the requested evaluation")
    if prepared["lerobot_revision"] != LEROBOT_REVISION or prepared["libero_assets_revision"] != LIBERO_ASSETS_REVISION:
        raise ValueError("Prepared dependency revisions differ")
    checkpoint = Path(prepared["checkpoint"])
    metadata = read_checkpoint_metadata(checkpoint)
    for name in REQUIRED_FILES:
        if (checkpoint / name).stat().st_size != prepared["file_sizes"][name]:
            raise ValueError(f"Checkpoint size changed after preparation: {name}")
        if name.endswith(".json") and file_sha256(checkpoint / name) != prepared["file_sha256"][name]:
            raise ValueError(f"Checkpoint metadata changed after preparation: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the configured CUDA GPU")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    suite, tasks = describe_suite(prepared["assets_path"])
    if tasks != prepared["tasks"]:
        raise ValueError("LIBERO task assets changed after preparation")
    from lerobot.envs.libero import LiberoEnv

    plans = {task["task_id"]: episode_plan(config, task["task_id"], task["initial_state_count"]) for task in tasks}
    settings = {"config": config.to_dict(), "checkpoint": prepared,
        "harness_sha256": file_sha256(__file__),
        "config_module_sha256": file_sha256(Path(__file__).with_name("lingbot_eval_config.py")),
        "packages": {name: importlib.metadata.version(name) for name in ("lerobot", "torch", "diffusers", "transformers", "hf-libero", "mujoco", "robosuite")},
        "episode_plan": [entry for task_id in TASK_IDS for entry in plans[task_id]]}
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not resume:
            raise FileExistsError("Evaluation output exists; use explicit --resume or choose a new run name")
        if read_json(output_dir / "settings.json") != settings:
            raise ValueError("Resume requires identical checkpoint, protocol, dependencies, and harness")
    elif resume:
        raise FileNotFoundError("Cannot resume an evaluation with no saved results")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "settings.json", settings)
    rows = []
    for task_id in TASK_IDS:
        for entry in plans[task_id]:
            path = output_dir / "episodes" / f"task_{task_id:02d}" / f"episode_{entry['episode_index']:03d}.json"
            if path.exists():
                row = read_json(path)
                if any(row.get(key) != value for key, value in entry.items()):
                    raise ValueError("Saved episode identity differs from the evaluation plan")
                rows.append(row)
    summary = aggregate_results(rows, episodes_per_task=config.episodes_per_task)
    write_json(output_dir / "summary.json", summary)
    if commit is not None:
        commit()
    wandb_run = None
    completed = summary["complete"]
    try:
        wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name,
            id=fingerprint([run_name, settings])[:16], resume="must" if resume else "never", config=settings)
        if completed:
            wandb_run.summary["eval/macro_success_rate"] = summary["macro_success_rate"]
            return summary
        torch.cuda.reset_peak_memory_stats()
        policy = load_policy(checkpoint, prepared["model_path"], metadata["architecture"])
        for task in tasks:
            task_id = task["task_id"]
            done = {row["episode_index"] for row in rows if row["task_id"] == task_id}
            if len(done) == config.episodes_per_task:
                continue
            env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name="libero_10",
                episode_length=520, observation_height=128, observation_width=128, obs_type="pixels",
                init_states=True, n_envs=1, num_steps_wait=10, control_freq=20, control_mode="relative", hard_reset=True)
            try:
                for entry in plans[task_id]:
                    if entry["episode_index"] in done:
                        continue
                    writer = None
                    video_path = output_dir / "videos" / f"task_{task_id:02d}.mp4"
                    if entry["episode_index"] == 0:
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = imageio.get_writer(str(video_path), fps=20, codec="libx264", macro_block_size=16)
                    try:
                        row = run_episode(policy, env, entry, task["instruction"], metadata["normalization"], "cuda", video_writer=writer)
                    finally:
                        if writer is not None:
                            writer.close()
                    row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
                    if entry["episode_index"] == 0:
                        row["video"] = str(video_path.relative_to(output_dir))
                    rows.append(row)
                    path = output_dir / "episodes" / f"task_{task_id:02d}" / f"episode_{entry['episode_index']:03d}.json"
                    write_json(path, row)
                    summary = aggregate_results(rows, episodes_per_task=config.episodes_per_task)
                    write_json(output_dir / "summary.json", summary)
                    if commit is not None:
                        commit()
                    wandb_run.log({"eval/completed_episodes": len(rows), "eval/success_rate_completed": summary["success_rate_completed"],
                        f"eval/task_{task_id:02d}/success_rate": summary["per_task"][str(task_id)]["success_rate"],
                        "eval/episode_seconds": row["seconds"], "eval/episode_steps": row["policy_steps"],
                        "eval/peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"]}, step=len(rows))
            finally:
                env.close()
        completed = summary["complete"]
        if not completed:
            raise RuntimeError("Evaluation ended with missing episodes")
        wandb_run.summary["eval/macro_success_rate"] = summary["macro_success_rate"]
        return summary
    finally:
        write_json(output_dir / "status.json", {"state": "completed" if completed else "interrupted_or_failed"})
        if commit is not None:
            commit()
        if wandb_run is not None:
            wandb_run.finish(exit_code=0 if completed else 1)


def main():
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared-output", type=Path)
    parser.add_argument("--prepared-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--result-volume")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = EvalConfig(**json.loads(args.config_json)).validate()
    root = os.environ.get("LEROBOT_SOURCE_ROOT", "/opt/lerobot")
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if revision != LEROBOT_REVISION or importlib.metadata.version("hf-libero") != "0.1.4":
        raise RuntimeError("Evaluation requires the pinned LeRobot and hf-libero revisions")
    if args.operation == "prepare":
        if any(value is None for value in (args.cache_root, args.checkpoint, args.prepared_output)):
            parser.error("prepare requires --cache-root, --checkpoint, and --prepared-output")
        prepared_path = prepare_evaluation(config, args.cache_root, args.checkpoint)
        write_json(args.prepared_output, {"prepared_path": prepared_path})
    else:
        if any(value is None for value in (args.prepared_path, args.output_dir, args.run_name, args.result_volume)):
            parser.error("run requires --prepared-path, --output-dir, --run-name, and --result-volume")
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required in dice-lingbot-wandb")
        import modal
        volume = modal.Volume.from_name(args.result_volume)
        summary = evaluate(config, args.prepared_path, args.output_dir, args.run_name, args.resume, volume.commit)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
