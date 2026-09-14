import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import modal

from script.lingbot_sft_config import SFTConfig, UPSTREAM_REVISION, fingerprint


APP_NAME = "dice-lingbot-va-sft"
CACHE_VOLUME_NAME = "dice-lingbot-sft-cache"
RUNS_VOLUME_NAME = "dice-lingbot-sft-runs"
SECRET_NAME = "dice-lingbot-wandb"
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "script" / "lingbot_sft_requirements.txt"
SCRIPT_FILES = (
    "lingbot_sft_config.py",
    "lingbot_sft_data.py",
    "lingbot_sft_patch.py",
    "lingbot_sft_train.py",
    "lingbot_sft_modal.py",
    "lingbot_sft_secret.py",
    "lingbot_sft_requirements.txt",
)
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def _build_image():
    image = modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    image = image.apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    image = image.uv_pip_install(
        "torch==2.9.0",
        "torchvision==0.24.0",
        "torchaudio==2.9.0",
        index_url="https://download.pytorch.org/whl/cu126",
        uv_version="0.8.18",
    )
    image = image.uv_pip_install(*REQUIREMENTS.read_text().splitlines(), uv_version="0.8.18")
    for filename in SCRIPT_FILES:
        image = image.add_local_file(ROOT / "script" / filename, f"/workspace/script/{filename}", copy=True)
    image = image.run_commands(
        "git clone https://github.com/Robbyant/lingbot-va.git /opt/lingbot-va",
        f"git -C /opt/lingbot-va checkout {UPSTREAM_REVISION}",
        "PYTHONPATH=/workspace:/opt/lingbot-va python -m script.lingbot_sft_patch --root /opt/lingbot-va",
        "PYTHONPATH=/workspace:/opt/lingbot-va python -c 'from wan_va.train import Trainer; from script.lingbot_sft_data import LatentDataset'",
    )
    return image.env({"PYTHONPATH": "/workspace:/opt/lingbot-va", "HF_HOME": "/cache/hub", "TOKENIZERS_PARALLELISM": "false"})


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
run_locks = modal.Dict.from_name("dice-lingbot-sft-run-locks", create_if_missing=True)
wandb_secret = modal.Secret.from_name(SECRET_NAME, required_keys=["WANDB_API_KEY"])
hf_secret = modal.Secret.from_name("dice-lingbot-hf", required_keys=["HF_TOKEN"])
image = _build_image()


def _config(steps, project, entity):
    cfg = SFTConfig(num_steps=steps, wandb_project=project, wandb_entity=entity)
    return cfg.validate()


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    secrets=[hf_secret],
    cpu=16,
    memory=65536,
    timeout=10800,
    max_containers=1,
)
def prepare(steps=1000, wandb_project=None, wandb_entity=None):
    cache_volume.reload()
    try:
        from script.lingbot_sft_data import prepare as prepare_data

        cfg = _config(steps, wandb_project or SFTConfig().wandb_project, wandb_entity)
        paths = prepare_data("/cache", cfg)
        prepared_dir = Path("/cache") / "prepared" / paths["dataset_fingerprint"]
        prepared_dir.mkdir(parents=True, exist_ok=True)
        output = prepared_dir / "paths.json"
        payload = {"paths": paths, "fingerprint": fingerprint(paths)}
        if output.exists() and json.loads(output.read_text()) != payload:
            raise ValueError(f"Refusing to overwrite prepared paths: {output}")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {"prepared": str(output), "fingerprint": paths["dataset_fingerprint"]}
    except Exception as exc:
        message = str(exc) if type(exc).__module__ == "builtins" else f"{type(exc).__name__}; inspect the preparation input format/dependencies"
        token = os.environ.get("HF_TOKEN", "")
        if token:
            message = message.replace(token, "[redacted]")
        raise RuntimeError(f"SFT preparation failed: {message}") from None
    finally:
        cache_volume.commit()


def _run_train(prepared_json, run_name, gpus, steps, wandb_project, wandb_entity, resume):
    if not RUN_NAME.fullmatch(run_name):
        raise ValueError("A safe --run-name is required for training")
    cfg = _config(steps, wandb_project, wandb_entity)
    owner = uuid.uuid4().hex
    if not run_locks.put(run_name, owner, skip_if_exists=True):
        raise RuntimeError("Run is already active or has a stale lock; confirm the previous job stopped before clearing it")
    try:
        cache_volume.reload()
        runs_volume.reload()
        run_dir = f"/runs/{run_name}"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={gpus}",
            "-m",
            "script.lingbot_sft_train",
            "--upstream-root",
            "/opt/lingbot-va",
            "--prepared",
            prepared_json,
            "--run-dir",
            run_dir,
            "--run-name",
            run_name,
            "--steps",
            str(cfg.num_steps),
            "--wandb-project",
            cfg.wandb_project,
            "--modal-volume",
            RUNS_VOLUME_NAME,
        ]
        if cfg.wandb_entity:
            command.extend(["--wandb-entity", cfg.wandb_entity])
        if resume:
            command.append("--resume")
        subprocess.run(command, cwd="/workspace", check=True)
        return {
            "run_dir": run_dir,
            "latest": f"{run_dir}/resume/latest.pt",
            "prepared": prepared_json,
            "checkpoint": f"{run_name}/checkpoints/step_{cfg.num_steps:06d}",
        }
    finally:
        try:
            runs_volume.commit()
        finally:
            if run_locks.get(run_name) == owner:
                run_locks.pop(run_name)


def download_checkpoint(run_name, steps, download_dir):
    if not RUN_NAME.fullmatch(run_name):
        raise ValueError("Invalid run name")
    SFTConfig(num_steps=steps).validate()
    destination = Path(download_dir) / run_name / f"step_{steps:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{run_name}/checkpoints/step_{steps:06d}"
    try:
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", RUNS_VOLUME_NAME, remote, str(destination)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Training succeeded but checkpoint download failed; remote checkpoint is retained: {exc}") from exc
    for relative in (
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/config.json",
        "norm_stats.json",
        "sft_config.json",
        "dataset_manifest.json",
    ):
        path = destination / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Incomplete checkpoint download: {path}; remote checkpoint is retained")
    return str(destination)


@app.function(
    image=image,
    gpu="H100:4",
    cpu=32,
    memory=262144,
    timeout=28800,
    retries=0,
    volumes={"/cache": cache_volume, "/runs": runs_volume},
    secrets=[wandb_secret],
)
def train4(prepared_json, run_name, steps=1000, wandb_project=None, wandb_entity=None, resume=False):
    return _run_train(prepared_json, run_name, 4, steps, wandb_project or SFTConfig().wandb_project, wandb_entity, resume)


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=28800,
    retries=0,
    volumes={"/cache": cache_volume, "/runs": runs_volume},
    secrets=[wandb_secret],
)
def train8(prepared_json, run_name, steps=1000, wandb_project=None, wandb_entity=None, resume=False):
    return _run_train(prepared_json, run_name, 8, steps, wandb_project or SFTConfig().wandb_project, wandb_entity, resume)


@app.local_entrypoint()
def main(stage: str = "train", gpus: int = 8, steps: int = 1000, run_name: str = "",
         wandb_project: str = "dice-lingbot-va-sft", wandb_entity: str = "", resume: bool = False,
         download_dir: str = "checkpoints/lingbot-sft"):
    if stage not in ("prepare", "train") or gpus not in (4, 8):
        raise ValueError("stage must be prepare/train and gpus must be 4 or 8")
    cfg = _config(steps, wandb_project, wandb_entity or None)
    if stage == "train" and not RUN_NAME.fullmatch(run_name):
        raise ValueError("A safe --run-name is required for training")
    destination = Path(download_dir) / run_name / f"step_{cfg.num_steps:06d}"
    if stage == "train" and destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    prepared = prepare.remote(cfg.num_steps, cfg.wandb_project, cfg.wandb_entity)
    if stage == "prepare":
        print(prepared)
        return
    fn = train4 if gpus == 4 else train8
    remote_result = fn.remote(prepared["prepared"], run_name, cfg.num_steps, cfg.wandb_project, cfg.wandb_entity, resume)
    print(remote_result)
    print(download_checkpoint(run_name, cfg.num_steps, download_dir))
