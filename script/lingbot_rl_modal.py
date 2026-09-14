import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import modal

from script.lingbot_eval_config import EvalConfig, LEROBOT_REVISION, validate_name
from script.lingbot_rl_config import RLConfig
from script.lingbot_sft_config import DATASET_REPO, DATASET_REVISION


ROOT = Path(__file__).resolve().parents[1]
CACHE_VOLUME = "dice-lingbot-sft-cache"
SOURCE_VOLUME = "dice-lingbot-sft-runs"
RESULT_VOLUME = "dice-lingbot-rl-runs"
HF_SECRET_NAME = "dice-lingbot-hf"
WANDB_SECRET_NAME = "dice-lingbot-wandb"
EVAL_PYTHON = "/opt/lerobot/.venv/bin/python"
INFERENCE_PATHS = ("residual.pt", "summary.json", "settings.json", "status.json", "train_eval", "eval")
SMOKE_ENV_STEPS = 32
FILES = (
    "lingbot_eval_config.py", "lingbot_eval.py", "lingbot_eval_report.py", "lingbot_sft_config.py",
    "lingbot_sft_data.py",
    "lingbot_rl_config.py", "lingbot_rl_model.py", "lingbot_rl_buffer.py", "lingbot_rl_policy.py",
    "lingbot_rl_data.py", "lingbot_rl_train.py",
)


def build_image():
    image = modal.Image.debian_slim(python_version="3.12").apt_install(
        "git", "ffmpeg", "libgl1", "libegl1", "libegl1-mesa-dev", "libgl1-mesa-dev",
        "libglib2.0-0", "libglvnd0", "libgles2", "build-essential", "cmake")
    image = image.uv_pip_install("uv==0.8.18", uv_version="0.8.18")
    image = image.run_commands(
        "git clone https://github.com/huggingface/lerobot.git /opt/lerobot",
        f"git -C /opt/lerobot checkout {LEROBOT_REVISION}",
        "uv sync --index-url https://pypi.org/simple --project /opt/lerobot --python 3.12 --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-editable",
        "uv export --index-url https://pypi.org/simple --project /opt/lerobot --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-emit-project --no-hashes --output-file /opt/lingbot-eval-deps.txt",
        f"uv pip install --index-url https://pypi.org/simple --python {EVAL_PYTHON} --constraint /opt/lingbot-eval-deps.txt --exclude-newer 2026-09-05T00:00:00Z modal==1.1.4",
    )
    for filename in FILES:
        image = image.add_local_file(ROOT / "script" / filename, f"/workspace/script/{filename}", copy=True)
    image = image.env({
        "PYTHONPATH": "/workspace", "HF_HOME": "/cache/hub", "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl", "LIBERO_CONFIG_PATH": "/tmp/lingbot-libero-config",
        "TOKENIZERS_PARALLELISM": "false", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
    })
    return image.run_commands(
        f"{EVAL_PYTHON} -c 'from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy; import modal, wandb, imageio'",
        f"{EVAL_PYTHON} -m script.lingbot_rl_config",
        f"{EVAL_PYTHON} -m script.lingbot_rl_train --help",
    )


app = modal.App("dice-lingbot-va-rl")
image = build_image()
cache = modal.Volume.from_name(CACHE_VOLUME)
source = modal.Volume.from_name(SOURCE_VOLUME)
results = modal.Volume.from_name(RESULT_VOLUME, create_if_missing=True)
locks = modal.Dict.from_name("dice-lingbot-rl-run-locks", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])


def _dataset_root():
    repo = DATASET_REPO.replace("/", "--")
    path = Path("/cache/hub") / f"datasets--{repo}" / "snapshots" / DATASET_REVISION
    return str(path) if path.is_dir() else None


def _acquire(run_name):
    owner = uuid.uuid4().hex
    if not locks.put(run_name, owner, skip_if_exists=True):
        raise RuntimeError(
            "RL run is active or has a stale lock; confirm shutdown before manually clearing a stale lock"
        )
    return owner


def _release(run_name, owner):
    if locks.get(run_name) == owner:
        locks.pop(run_name)


@app.function(image=image, cpu=8, memory=32768, timeout=10800, retries=0,
              volumes={"/cache": cache, "/sft": source.read_only()}, secrets=[hf_secret], max_containers=1)
def prepare(config):
    cfg = EvalConfig(**config).validate()
    cache.reload()
    source.reload()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared.json"
            subprocess.run([
                EVAL_PYTHON, "-m", "script.lingbot_eval", "prepare", "--config-json", json.dumps(asdict(cfg)),
                "--cache-root", "/cache", "--checkpoint", f"/sft/{cfg.source_run}/checkpoints/step_{cfg.checkpoint_step:06d}",
                "--prepared-output", str(output),
            ], check=True)
            prepared_path = json.loads(output.read_text())["prepared_path"]
        if _dataset_root() is None:
            token = os.environ.get("HF_TOKEN", "").strip()
            if not token:
                raise RuntimeError("HF_TOKEN is required in dice-lingbot-hf for CPU asset preparation")
            subprocess.run([
                EVAL_PYTHON, "-c",
                "import os; from script.lingbot_eval import download_snapshot; "
                "from script.lingbot_sft_config import DATASET_REPO, DATASET_REVISION; "
                "download_snapshot(DATASET_REPO, repo_type='dataset', revision=DATASET_REVISION, "
                "cache_dir='/cache/hub', token=os.environ['HF_TOKEN'])",
            ], check=True)
        return prepared_path
    finally:
        cache.commit()


@app.function(image=image, gpu="H100", cpu=16, memory=98304, timeout=43200, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/rl": results},
              secrets=[wandb_secret], max_containers=1)
def run_train(config, prepared_path, run_name, resume=False):
    cfg = RLConfig(**config).validate()
    validate_name(run_name)
    owner = _acquire(run_name)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        cache.reload()
        source.reload()
        results.reload()
        output = Path("/rl") / run_name
        output.mkdir(parents=True, exist_ok=True)
        command = [
            EVAL_PYTHON, "-m", "script.lingbot_rl_train", "train",
            "--config-json", json.dumps(asdict(cfg)),
            "--prepared-path", prepared_path,
            "--output-dir", str(output),
            "--run-name", run_name,
            "--result-volume", RESULT_VOLUME,
        ]
        dataset_root = _dataset_root()
        if dataset_root:
            command.extend(["--dataset-root", dataset_root])
        if resume:
            command.append("--resume")
        subprocess.run(command, check=True, env=env)
        status = {"stage": "train", "run_name": run_name, "complete": True}
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        return json.loads((output / "summary.json").read_text())
    finally:
        try:
            results.commit()
        finally:
            _release(run_name, owner)


@app.function(image=image, gpu="H100", cpu=16, memory=98304, timeout=7200, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/rl": results},
              secrets=[wandb_secret], max_containers=1)
def run_smoke(config, prepared_path, run_name):
    """A few env steps on one H100. Does not change the pinned 100k recipe or ingest 300 demos."""
    cfg = RLConfig(**config).validate()
    validate_name(run_name)
    owner = _acquire(run_name)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        cache.reload()
        source.reload()
        results.reload()
        output = Path("/rl") / run_name
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            EVAL_PYTHON, "-m", "script.lingbot_rl_train", "train",
            "--config-json", json.dumps(asdict(cfg)),
            "--prepared-path", prepared_path,
            "--output-dir", str(output),
            "--run-name", run_name,
            "--result-volume", RESULT_VOLUME,
            "--max-env-steps", str(SMOKE_ENV_STEPS),
        ], check=True, env=env)
        status = {"stage": "smoke", "run_name": run_name, "complete": True, "max_env_steps": SMOKE_ENV_STEPS}
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        return json.loads((output / "summary.json").read_text())
    finally:
        try:
            results.commit()
        finally:
            _release(run_name, owner)


@app.function(image=image, gpu="H100", cpu=16, memory=98304, timeout=21600, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/rl": results},
              secrets=[wandb_secret], max_containers=1)
def run_eval(config, prepared_path, run_name, resume=False):
    cfg = RLConfig(**config).validate()
    validate_name(run_name)
    owner = _acquire(f"{run_name}-eval")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        cache.reload()
        source.reload()
        results.reload()
        output = Path("/rl") / run_name
        command = [
            EVAL_PYTHON, "-m", "script.lingbot_rl_train", "eval",
            "--config-json", json.dumps(asdict(cfg)),
            "--prepared-path", prepared_path,
            "--output-dir", str(output),
            "--run-name", run_name,
            "--residual-path", str(output / "residual.pt"),
            "--result-volume", RESULT_VOLUME,
        ]
        if resume:
            command.append("--resume")
        subprocess.run(command, check=True, env=env)
        status = {"stage": "eval", "run_name": run_name, "complete": True}
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        return json.loads((output / "eval" / "summary.json").read_text())
    finally:
        try:
            results.commit()
        finally:
            _release(f"{run_name}-eval", owner)


def download_inference(run_name, download_dir):
    validate_name(run_name)
    destination = Path(download_dir) / run_name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in INFERENCE_PATHS:
        subprocess.run([
            sys.executable, "-m", "modal", "volume", "get", RESULT_VOLUME,
            f"{run_name}/{name}", str(destination.parent),
        ], check=False)
    if not destination.exists():
        destination.mkdir(parents=True)
        for name in INFERENCE_PATHS:
            source_path = destination.parent / name
            if source_path.exists():
                source_path.rename(destination / name)
    eval_dir = destination / "eval"
    if eval_dir.is_dir():
        from script.lingbot_eval_report import create_report
        create_report(eval_dir)
    if not (destination / "residual.pt").is_file():
        raise FileNotFoundError("Inference download is missing residual.pt")
    return str(destination)


@app.local_entrypoint()
def main(stage: str = "train", run_name: str = "", resume: bool = False,
         wandb_project: str = "dice-lingbot-va-rl", wandb_entity: str = "",
         download_dir: str = "result/lingbot-rl"):
    if stage not in ("prepare", "train", "eval", "smoke"):
        raise ValueError("Stage must be prepare, train, eval, or smoke")
    eval_cfg = EvalConfig(
        source_run="libero30-sft", checkpoint_step=600, stage="eval", seed=42,
        wandb_project="dice-lingbot-va-eval", wandb_entity=wandb_entity or None,
    ).validate()
    rl_cfg = RLConfig(
        wandb_project=wandb_project, wandb_entity=wandb_entity or None,
    ).validate()
    if stage == "smoke":
        run_name = validate_name(run_name or "libero30-dice-smoke")
    else:
        run_name = validate_name(run_name or rl_cfg.default_run_name)
    if stage != "prepare" and (Path(download_dir) / run_name).exists():
        raise FileExistsError("Local result directory exists; choose a different --download-dir")
    prepared_path = prepare.remote(asdict(eval_cfg))
    if stage == "prepare":
        print(prepared_path)
        return
    if stage == "train":
        summary = run_train.remote(asdict(rl_cfg), prepared_path, run_name, resume)
    elif stage == "smoke":
        summary = run_smoke.remote(asdict(rl_cfg), prepared_path, run_name)
    else:
        summary = run_eval.remote(asdict(rl_cfg), prepared_path, run_name, resume)
    print(json.dumps(summary, indent=2))
    print(download_inference(run_name, download_dir))
