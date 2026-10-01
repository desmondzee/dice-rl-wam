import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

from script.lingbot_eval_config import EvalConfig, LEROBOT_REVISION, eval_config_from_dict, validate_name


ROOT = Path(__file__).resolve().parents[1]
CACHE_VOLUME = "dice-lingbot-sft-cache"
SOURCE_VOLUME = "dice-lingbot-sft-runs"
RESULT_VOLUME = "dice-lingbot-eval-results"
WEIGHTS_VOLUME = "dice-lingbot-rl-weights"
EVAL_PYTHON = "/opt/lerobot/.venv/bin/python"
FILES = ("lingbot_eval_config.py", "lingbot_eval.py", "lingbot_sft_config.py", "lingbot_rl_config.py",
         "lingbot_rl_model.py", "lingbot_rl_buffer.py", "lingbot_rl_data.py", "lingbot_rl_policy.py", "lingbot_rl_train.py")


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
        "PYTHONPATH": "/workspace", "HF_HOME": "/cache/hub", "HF_HUB_DISABLE_XET": "1", "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl", "LIBERO_CONFIG_PATH": "/tmp/lingbot-libero-config",
        "TOKENIZERS_PARALLELISM": "false", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
    })
    return image.run_commands(
        f"{EVAL_PYTHON} -c 'from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy; import modal, imageio'",
        f"{EVAL_PYTHON} -m script.lingbot_eval --help",
    )


app = modal.App("dice-lingbot-va-eval")
image = build_image()
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
source = modal.Volume.from_name(SOURCE_VOLUME)
results = modal.Volume.from_name(RESULT_VOLUME, create_if_missing=True)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("dice-lingbot-hf", required_keys=["HF_TOKEN"])


@app.function(image=image, cpu=8, memory=32768, timeout=10800, retries=0,
              volumes={"/cache": cache, "/sft": source.read_only()}, secrets=[hf_secret], max_containers=1)
def prepare(config):
    cfg = eval_config_from_dict(config)
    cache.reload()
    source.reload()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared.json"
            subprocess.run([
                EVAL_PYTHON, "-m", "script.lingbot_eval", "prepare", "--config-json", json.dumps(cfg.to_dict()),
                "--cache-root", "/cache", "--checkpoint", f"/sft/{cfg.source_run}/checkpoints/step_{cfg.checkpoint_step:06d}",
                "--prepared-output", str(output),
            ], check=True)
            return json.loads(output.read_text())["prepared_path"]
    finally:
        cache.commit()


@app.function(image=image, gpu="L40S", cpu=8, memory=49152, timeout=21600, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/results": results}, max_containers=16)
def run_shard(config, prepared_path, run_dir, resume=False):
    cfg = eval_config_from_dict(config)
    cache.reload()
    source.reload()
    results.reload()
    command = [
        EVAL_PYTHON, "-m", "script.lingbot_eval", "run", "--config-json", json.dumps(cfg.to_dict()),
        "--prepared-path", prepared_path, "--output-dir", f"/results/{run_dir}",
        "--run-name", validate_name(run_dir.replace("/", "-")), "--result-volume", RESULT_VOLUME,
    ] + (["--resume"] if resume else [])
    try:
        subprocess.run(command, check=True)
    finally:
        results.commit()
    return json.loads((Path("/results") / run_dir / "summary" / f"shard_{cfg.shard}.json").read_text())


@app.function(image=image, gpu="L40S", cpu=8, memory=49152, timeout=21600, retries=0,
              volumes={"/cache": cache.read_only(), "/sft": source.read_only(), "/weights": weights.read_only(), "/results": results},
              max_containers=16)
def run_rl_shard(config, prepared_path, run_dir, residual, eval_candidates, resume=False, residual_input="z"):
    cfg = eval_config_from_dict(config)
    cache.reload()
    source.reload()
    weights.reload()
    results.reload()
    command = [
        EVAL_PYTHON, "-m", "script.lingbot_rl_train", "eval", "--config-json", json.dumps({"task_ids": list(cfg.task_ids), "residual_input": residual_input}),
        "--prepared-path", prepared_path, "--output-dir", f"/results/{run_dir}", "--residual-path", f"/weights/{residual}",
        "--eval-candidates", str(eval_candidates), "--shard", str(cfg.shard), "--shards", str(cfg.shards),
        "--result-volume", RESULT_VOLUME,
    ] + (["--resume"] if resume else [])
    try:
        subprocess.run(command, check=True)
    finally:
        results.commit()
    return json.loads((Path("/results") / run_dir / "summary" / f"shard_{cfg.shard}.json").read_text())


def merge_task(local_dir, cfg):
    from script.lingbot_eval import aggregate_results, read_json, write_json

    rows = sorted((read_json(path) for path in (local_dir / "episodes").glob("task_*/episode_*.json")),
                  key=lambda row: row["episode_index"])
    summary = aggregate_results(rows, task_ids=cfg.task_ids, episodes_per_task=cfg.episodes_per_task)
    if not summary["complete"]:
        raise RuntimeError(f"{local_dir} is missing episodes")
    summary["mean_policy_steps"] = sum(row["policy_steps"] for row in rows) / len(rows)
    write_json(local_dir / "summary.json", summary)
    return rows, summary


def write_summary_csv(path, policy, task_id, rows, summary):
    n, k = summary["completed_episodes"], summary["successes"]
    rate = k / n
    record = {"policy": policy, "task_id": task_id, "episodes": n, "successes": k, "success_rate": round(rate, 4),
              "ci95_halfwidth": round(1.96 * math.sqrt(rate * (1 - rate) / n), 4), "mean_policy_steps": round(summary["mean_policy_steps"], 1)}
    existing = []
    if path.exists():
        with path.open() as handle:
            existing = [row for row in csv.DictReader(handle) if not (row["policy"] == policy and int(row["task_id"]) == task_id)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        for row in sorted(existing + [record], key=lambda row: (row["policy"], int(row["task_id"]))):
            writer.writerow(row)


def log_wandb(policy, task_id, rows, summary, settings, project, entity):
    import wandb

    run = wandb.init(project=project, entity=entity or None, name=f"{policy}-task{task_id:02d}-heldout", config=settings)
    for row in rows:
        run.log({"success": int(row["success"]), "policy_steps": row["policy_steps"], "seconds": row["seconds"],
                 "init_state_id": row["init_state_id"]}, step=row["episode_index"])
    run.summary["success_rate"] = summary["success_rate_completed"]
    run.summary["episodes"] = summary["completed_episodes"]
    run.finish()
    return run.url


def download_task(run_dir, download_dir, resume):
    destination = Path(download_dir) / run_dir
    if destination.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", RESULT_VOLUME, run_dir, str(destination.parent), "--force"], check=True)
    return destination


@app.local_entrypoint()
def main(stage: str = "prepare", source_run: str = "libero30-sft", checkpoint_step: int = 600, policy: str = "",
         task_ids: str = "0,4", shards: int = 4, seed: int = 42, resume: bool = False,
         wandb_project: str = "dice-lingbot-va-eval", wandb_entity: str = "james-j-carver-university-of-cambridge",
         download_dir: str = "result/heldout", residual: str = "", eval_candidates: int = 4, residual_input: str = "z"):
    from script.lingbot_eval import read_json

    if stage not in ("prepare", "eval", "merge"):
        raise ValueError("Stage must be prepare, eval, or merge")
    tasks = tuple(int(task) for task in task_ids.split(","))
    if residual and not policy:
        raise ValueError("--policy names the residual evaluation")
    policy = validate_name(policy or f"sft-step{checkpoint_step:06d}")
    base = EvalConfig(source_run=source_run, checkpoint_step=checkpoint_step, stage="heldout", seed=seed,
                      task_ids=tasks, wandb_project=wandb_project, wandb_entity=wandb_entity or None).validate()
    for task in tasks:
        if stage != "prepare" and (Path(download_dir) / policy / f"task_{task:02d}").exists() and not resume:
            raise FileExistsError("Local result directory exists; pass --resume to refresh it")
    if stage != "merge":
        prepared_path = prepare.remote(base.to_dict())
        print(prepared_path)
        if stage == "prepare":
            return
        jobs = [(EvalConfig(**{**base.__dict__, "task_ids": (task,), "shard": shard, "shards": shards}).validate().to_dict(),
                 prepared_path, f"{policy}/task_{task:02d}") for task in tasks for shard in range(shards)]
        if residual:
            summaries = run_rl_shard.starmap([(*job, residual, eval_candidates, resume, residual_input) for job in jobs])
        else:
            summaries = run_shard.starmap([(*job, resume) for job in jobs])
        for shard_summary in summaries:
            print(json.dumps(shard_summary))
    for task in tasks:
        cfg = EvalConfig(**{**base.__dict__, "task_ids": (task,)}).validate()
        local_dir = download_task(f"{policy}/task_{task:02d}", download_dir, resume)
        rows, summary = merge_task(local_dir, cfg)
        write_summary_csv(Path(download_dir) / "summary.csv", policy, task, rows, summary)
        settings = read_json(local_dir / "settings" / "shard_0.json")
        print(task, summary["success_rate_completed"], log_wandb(policy, task, rows, summary, settings, wandb_project, wandb_entity))
