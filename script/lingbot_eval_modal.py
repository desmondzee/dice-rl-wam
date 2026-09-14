import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import modal

from script.lingbot_eval_config import EvalConfig, LEROBOT_REVISION, validate_name


ROOT = Path(__file__).resolve().parents[1]
CACHE_VOLUME = "dice-lingbot-sft-cache"
SOURCE_VOLUME = "dice-lingbot-sft-runs"
RESULT_VOLUME = "dice-lingbot-eval-results"
EVAL_PYTHON = "/opt/lerobot/.venv/bin/python"
FILES = ("lingbot_eval_config.py", "lingbot_eval.py", "lingbot_sft_config.py")


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
        f"{EVAL_PYTHON} -m script.lingbot_eval --help",
    )


app = modal.App("dice-lingbot-va-eval")
image = build_image()
cache = modal.Volume.from_name(CACHE_VOLUME)
source = modal.Volume.from_name(SOURCE_VOLUME)
results = modal.Volume.from_name(RESULT_VOLUME, create_if_missing=True)
locks = modal.Dict.from_name("dice-lingbot-eval-run-locks", create_if_missing=True)
hf_secret = modal.Secret.from_name("dice-lingbot-hf", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("dice-lingbot-wandb", required_keys=["WANDB_API_KEY"])


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
            return json.loads(output.read_text())["prepared_path"]
    finally:
        cache.commit()


@app.function(image=image, gpu="H100", cpu=16, memory=98304, timeout=21600, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/results": results},
              secrets=[wandb_secret], max_containers=1)
def run_evaluation(config, prepared_path, run_name, resume=False):
    cfg = EvalConfig(**config).validate()
    validate_name(run_name)
    owner = uuid.uuid4().hex
    if not locks.put(run_name, owner, skip_if_exists=True):
        raise RuntimeError("Evaluation is active or has a stale lock; confirm shutdown before manually clearing a stale lock")
    try:
        cache.reload()
        source.reload()
        results.reload()
        command = [
            EVAL_PYTHON, "-m", "script.lingbot_eval", "run", "--config-json", json.dumps(asdict(cfg)),
            "--prepared-path", prepared_path, "--output-dir", f"/results/{run_name}", "--run-name", run_name,
            "--result-volume", RESULT_VOLUME,
        ]
        if resume:
            command.append("--resume")
        subprocess.run(command, check=True)
        return json.loads((Path("/results") / run_name / "summary.json").read_text())
    finally:
        try:
            results.commit()
        finally:
            if locks.get(run_name) == owner:
                locks.pop(run_name)


def download_results(run_name, download_dir):
    from script.lingbot_eval_report import create_report

    validate_name(run_name)
    destination = Path(download_dir) / run_name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", RESULT_VOLUME,
                    run_name, str(destination.parent)], check=True)
    create_report(destination)
    return str(destination)


@app.local_entrypoint()
def main(stage: str = "smoke", source_run: str = "libero30-sft", checkpoint_step: int = 600,
         run_name: str = "", seed: int = 42, resume: bool = False,
         wandb_project: str = "dice-lingbot-va-eval", wandb_entity: str = "",
         download_dir: str = "result/lingbot-eval"):
    if stage not in ("prepare", "smoke", "eval"):
        raise ValueError("Stage must be prepare, smoke, or eval")
    cfg = EvalConfig(source_run=source_run, checkpoint_step=checkpoint_step,
                     stage="eval" if stage == "prepare" else stage, seed=seed,
                     wandb_project=wandb_project, wandb_entity=wandb_entity or None).validate()
    run_name = validate_name(run_name or cfg.default_run_name)
    if stage != "prepare" and (Path(download_dir) / run_name).exists():
        raise FileExistsError("Local result directory exists; choose a different --download-dir")
    prepared_path = prepare.remote(asdict(cfg))
    if stage == "prepare":
        print(prepared_path)
        return
    summary = run_evaluation.remote(asdict(cfg), prepared_path, run_name, resume)
    print(json.dumps(summary, indent=2))
    print(download_results(run_name, download_dir))
