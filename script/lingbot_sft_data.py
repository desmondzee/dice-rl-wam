import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from script.lingbot_sft_config import (
    CAMERAS,
    DATASET_REPO,
    DATASET_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    SFTConfig,
    fingerprint,
    provenance,
)


def load_latent(path):
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    allowed += [type(np.dtype(dtype)) for dtype in (np.int32, np.int64, np.float32, np.float64)]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def choose_episodes(episodes, cfg):
    cfg.validate()
    by_task = defaultdict(list)
    seen = set()
    for episode in episodes:
        idx = episode["episode_index"]
        if not isinstance(idx, int) or idx < 0 or idx in seen:
            raise ValueError("Episode IDs must be unique non-negative integers")
        seen.add(idx)
        if len(episode["tasks"]) != 1:
            raise ValueError("Every LIBERO episode must have exactly one task")
        segments = episode["action_config"]
        if len(segments) != 1 or segments[0]["start_frame"] != 0 or segments[0]["end_frame"] != episode["length"]:
            raise ValueError("Expected a single complete episode segment")
        if segments[0]["action_text"] != episode["tasks"][0]:
            raise ValueError("Task instruction and latent segment instruction disagree")
        by_task[episode["tasks"][0]].append(episode)
    if len(by_task) != 10:
        raise ValueError("Expected exactly ten LIBERO-Long tasks")
    selected = []
    for task in sorted(by_task):
        if len(by_task[task]) < cfg.demos_per_task:
            raise ValueError(f"Insufficient demonstrations for {task}")
        ranked = sorted(by_task[task], key=lambda ep: fingerprint([cfg.seed, task, ep["episode_index"]]))
        selected.extend(sorted(ranked[:cfg.demos_per_task], key=lambda ep: ep["episode_index"]))
    return selected


def episode_paths(info, episode):
    idx = episode["episode_index"]
    chunk = idx // info["chunks_size"]
    data_path = info["data_path"].format(episode_chunk=chunk, episode_index=idx)
    latent_paths = [f"latents/chunk-{chunk:03d}/{cam}/episode_{idx:06d}_0_{episode['length']}.pth" for cam in CAMERAS]
    for path in [data_path, *latent_paths]:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Dataset paths must stay inside the snapshot")
    return data_path, latent_paths


def read_actions(path, episode):
    table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
    if table.num_rows != episode["length"]:
        raise ValueError("Parquet episode length disagrees with metadata")
    if table["episode_index"].to_pylist() != [episode["episode_index"]] * table.num_rows:
        raise ValueError("Parquet contains another episode")
    if table["frame_index"].to_pylist() != list(range(table.num_rows)):
        raise ValueError("Parquet frame order is not contiguous")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.shape != (episode["length"], 7) or not np.isfinite(actions).all():
        raise ValueError("Expected finite seven-dimensional LIBERO actions")
    return actions


def validate_info(info):
    if info["codebase_version"] != "v2.1" or info["total_tasks"] != 10:
        raise ValueError("Expected the pinned LIBERO-Long LeRobot v2.1 dataset")
    if info["features"]["action"]["shape"] != [7] or info["chunks_size"] < 1:
        raise ValueError("Unexpected action schema or chunk size")
    for camera in CAMERAS:
        if info["features"][camera]["shape"] != [128, 128, 3]:
            raise ValueError("LIBERO cameras must be 128x128 RGB")


def assemble_streams(camera_data, actions, episode, norm):
    if actions.shape != (episode['length'],7) or not np.isfinite(actions).all():
        raise ValueError('Expected finite seven-dimensional episode actions')
    views = []
    frame_ids = None
    text = None
    frames = None
    for data in camera_data:
        ids = np.asarray(data["frame_ids"])
        if ids.ndim != 1 or len(ids) < 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Expected integer frame IDs")
        if (data['start_frame'], data['end_frame'], data['video_num_frames'], data['video_height'], data['video_width'], data['fps'], data['ori_fps']) != (0, episode['length'], len(ids), 128, 128, 60, 60):
            raise ValueError('Latent source metadata does not match the pinned LIBERO episode')
        if not np.all(np.diff(ids) == 1) or ids[0] < 0 or ids[-1] >= episode["length"]:
            raise ValueError("LIBERO latents must use contiguous source frames; action/latent ratio is four")
        if frame_ids is not None and not np.array_equal(frame_ids, ids):
            raise ValueError("Camera frame alignment differs")
        frame_ids = ids
        f = int(data["latent_num_frames"])
        if f != (len(ids) - 1) // 4 + 1:
            raise ValueError("Causal VAE temporal compression mismatch")
        if (int(data["latent_height"]), int(data["latent_width"])) != (8, 8):
            raise ValueError("Expected 128x128 views compressed spatially by sixteen")
        latent = data["latent"]
        if latent.shape != (f * 8 * 8, 48) or not torch.isfinite(latent).all():
            raise ValueError("Invalid video latent tensor")
        emb = data["text_emb"]
        if emb.shape == (1, 512, 4096):
            emb = emb[0]
        if emb.shape != (512, 4096) or not torch.isfinite(emb).all():
            raise ValueError("Expected a finite padded UMT5 embedding of shape [512,4096]")
        if data["text"].strip() != episode["tasks"][0].strip():
            raise ValueError("Latent text does not match the selected task")
        if text is not None and not torch.equal(text, emb):
            raise ValueError("Camera task embeddings differ")
        text = emb
        frames = f
        views.append(latent.reshape(f, 8, 8, 48))
    if len(views) != 2:
        raise ValueError("Exactly two ordered camera streams are required")
    raw = np.pad(actions[int(frame_ids[0]):], ((4, 0), (0, 0)))[:frames * 4]
    if raw.shape != (frames * 4, 7):
        raise ValueError("Insufficient actions for the latent sequence")
    q01, q99 = np.asarray(norm["q01"]), np.asarray(norm["q99"])
    if q01.shape != (30,) or q99.shape != (30,) or not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 < q01):
        raise ValueError("Invalid normalization statistics")
    normalized = np.clip((raw - q01[:7]) / (q99[:7] - q01[:7] + 1e-6) * 2 - 1, -1.5, 1.5)
    full = np.zeros((frames * 4, 30), dtype=np.float32)
    full[:, :7] = normalized
    action_tensor = torch.from_numpy(full).reshape(frames, 4, 30).permute(2, 0, 1).unsqueeze(-1)
    mask = torch.zeros_like(action_tensor, dtype=torch.bool)
    mask[:7] = True
    return {
        "latents": torch.cat(views, dim=2).permute(3, 0, 1, 2).contiguous(),
        "actions": action_tensor.contiguous(),
        "actions_mask": mask,
        "text_emb": text.contiguous(),
    }


def build_manifest(root, info, episodes, cfg):
    validate_info(info)
    selected = choose_episodes(episodes, cfg)
    raw_actions = [read_actions(root / episode_paths(info, ep)[0], ep) for ep in selected]
    quantiles = np.quantile(np.concatenate(raw_actions), [0.01, 0.99], axis=0, method="linear")
    norm = {"q01": quantiles[0].tolist() + [0.0] * 23, "q99": quantiles[1].tolist() + [0.0] * 23}
    for ep, actions in zip(selected, raw_actions):
        _, paths = episode_paths(info, ep)
        assemble_streams([load_latent(root / path) for path in paths], actions, ep, norm)
    manifest = {
        "provenance": provenance(), "selection_seed": cfg.seed,
        "selection_algorithm": "per-task SHA256 rank of JSON [seed, task, episode_index], take 30",
        "episodes": selected, "norm_stat": norm, "info": info,
        "training_frames": sum(ep["length"] for ep in selected),
        "task_counts": {task: sum(ep["tasks"][0] == task for ep in selected) for task in sorted({ep["tasks"][0] for ep in selected})},
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def validate_manifest(manifest, cfg):
    contents = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint(contents) != manifest["fingerprint"]:
        raise ValueError("Dataset manifest fingerprint mismatch")
    if manifest["provenance"] != provenance() or manifest["selection_seed"] != cfg.seed:
        raise ValueError("Dataset provenance or selection seed mismatch")
    selected = manifest["episodes"]
    if len(selected) != 300 or choose_episodes(selected, cfg) != selected:
        raise ValueError("Expected exactly 30 distinct complete episodes for each task")
    validate_info(manifest["info"])


class LatentDataset(Dataset):
    def __init__(self, config):
        self.root = Path(config.dataset_path)
        self.manifest = json.loads(Path(config.manifest_path).read_text())
        validate_manifest(self.manifest, SFTConfig(seed=config.seed))
        self.episodes = self.manifest["episodes"]
        self.info = self.manifest["info"]
        self.norm = self.manifest["norm_stat"]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        parquet_path, latent_paths = episode_paths(self.info, episode)
        return assemble_streams([load_latent(self.root / path) for path in latent_paths],
                                read_actions(self.root / parquet_path, episode), episode, self.norm)


def _hf_download(function, *args, **kwargs):
    from huggingface_hub.errors import HfHubHTTPError
    from requests.exceptions import ConnectionError, Timeout

    for attempt in range(4):
        try:
            return function(*args, **kwargs)
        except (HfHubHTTPError, ConnectionError, Timeout) as exc:
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            if isinstance(exc, HfHubHTTPError) and status not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Hugging Face request failed (HTTP {status}); check the HF_TOKEN in Modal secret dice-lingbot-hf and its read permissions."
                ) from None
            if attempt == 3:
                raise RuntimeError(
                    "Hugging Face download failed after four attempts; cached files are retained. Retry after the service cooldown and verify dice-lingbot-hf contains a valid HF_TOKEN."
                ) from None
            delay = 60.0 * 2**attempt
            retry_after = response.headers.get("Retry-After") if response is not None else None
            if retry_after:
                try:
                    advertised = float(retry_after)
                except ValueError:
                    try:
                        advertised = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        advertised = 0.0
                if math.isfinite(advertised):
                    delay = max(delay, advertised)
            if delay > 900:
                raise RuntimeError(
                    "Hugging Face requested a long cooldown; cached files are retained. Retry later instead of holding the preparation container idle."
                ) from None
            logging.getLogger(__name__).warning(
                "Hugging Face download retry %d/3 after HTTP %s; waiting %.0f seconds",
                attempt + 1,
                status,
                delay,
            )
            time.sleep(delay)


@torch.no_grad()
def create_empty_embedding(model_root, output):
    from transformers import T5TokenizerFast, UMT5EncoderModel

    tokenizer = T5TokenizerFast.from_pretrained(str(model_root / "tokenizer"), local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(str(model_root / "text_encoder"), torch_dtype=torch.bfloat16, local_files_only=True).eval()
    tokens = tokenizer([""], padding="max_length", max_length=512, truncation=True,
                       add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    emb = encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state[0]
    length = int(tokens.attention_mask[0].sum())
    emb[length:] = 0
    if emb.shape != (512, 4096) or not torch.isfinite(emb).all():
        raise ValueError("Invalid empty-prompt embedding")
    torch.save(emb.to(torch.bfloat16).cpu(), output)


def prepare(cache_root, cfg):
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg.validate()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for preparation; create Modal secret dice-lingbot-hf using script.lingbot_sft_secret --service hf."
        )
    cache_root = Path(cache_root)
    hf_cache = cache_root / "hub"

    def metadata(name):
        return Path(_hf_download(
            hf_hub_download,
            DATASET_REPO,
            name,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(hf_cache),
            token=token,
        ))

    info = json.loads(metadata("meta/info.json").read_text())
    episodes = [json.loads(line) for line in metadata("meta/episodes.jsonl").read_text().splitlines() if line]
    validate_info(info)
    selected = choose_episodes(episodes, cfg)
    files = ["meta/info.json", "meta/episodes.jsonl"]
    for ep in selected:
        parquet_path, latent_paths = episode_paths(info, ep)
        files.extend([parquet_path, *latent_paths])
    root = Path(_hf_download(
        snapshot_download,
        DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=files,
        cache_dir=str(hf_cache),
        max_workers=2,
        token=token,
    ))
    manifest = build_manifest(root, info, episodes, cfg)
    prepared = cache_root / "prepared" / manifest["fingerprint"]
    prepared.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to overwrite a different prepared manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model_root = Path(_hf_download(
        snapshot_download,
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=["transformer/*", "text_encoder/*", "tokenizer/*"],
        cache_dir=str(hf_cache),
        max_workers=2,
        token=token,
    ))
    empty_path = prepared / "empty_emb.pt"
    if not empty_path.exists():
        temporary = prepared / "empty_emb.tmp"
        create_empty_embedding(model_root, temporary)
        temporary.replace(empty_path)
    empty = torch.load(empty_path, map_location="cpu", weights_only=True)
    if empty.shape != (512, 4096) or not torch.isfinite(empty).all():
        raise ValueError("Invalid cached empty-prompt embedding")
    return {"dataset_path": str(root), "manifest_path": str(manifest_path),
            "model_path": str(model_root), "empty_emb_path": str(empty_path),
            "dataset_fingerprint": manifest["fingerprint"]}
