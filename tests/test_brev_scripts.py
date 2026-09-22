import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("push.sh", "setup.sh", "train.sh", "sync.sh", "pull.sh")


def _run(script, args, env, cwd=None):
    merged = {**os.environ, "DICE_DRY_RUN": "1", "DICE_ENV_FILE": "/dev/null", **env}
    for name in ("DICE_SYNC_CMD", "MODAL_PROFILE", "MODAL_SFT_PROFILE", "WANDB_ENTITY"):
        merged.pop(name, None)
    merged.update(env)
    return subprocess.run(["bash", str(ROOT / "brev" / script), *args], env=merged, cwd=cwd,
                          capture_output=True, text=True)


def _box(tmp_path):
    data = tmp_path / "data"
    sft = data / "sft" / "libero30-sft" / "checkpoints" / "step_000600"
    sft.mkdir(parents=True)
    (sft / "norm_stats.json").write_text("{}")
    (data / "prepared.json").write_text("{}")
    (data / "dataset_root").write_text(str(data / "cache" / "hub" / "snap"))
    (data / "lerobot" / ".venv" / "bin").mkdir(parents=True)
    (data / "lerobot" / ".venv" / "bin" / "python").write_text("")
    (data / "lerobot" / ".venv" / "bin" / "modal").write_text("")
    return {"DICE_DATA": str(data), "DICE_REPO": str(ROOT), "HOME": str(tmp_path),
            "WANDB_API_KEY": "wandb-secret-value", "HF_TOKEN": "hf-secret-value",
            "MODAL_PROFILE": "desmond-zee", "MODAL_SFT_PROFILE": "source"}


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_and_are_executable(script):
    path = ROOT / "brev" / script
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_env_example_lists_required_variables_without_values():
    text = (ROOT / "brev" / "env.example").read_text()
    for name in ("DICE_REPO", "DICE_DATA", "WANDB_API_KEY", "HF_TOKEN", "MODAL_PROFILE", "MODAL_SFT_PROFILE"):
        assert f"export {name}=" in text
    assert "brev/env.sh" in (ROOT / ".gitignore").read_text()


def test_train_dry_run_builds_single_task_command_and_resumes_when_state_exists(tmp_path):
    env = _box(tmp_path)
    first = _run("train.sh", ["unit-t4", "4"], env)
    assert first.returncode == 0, first.stderr
    assert '"task_ids": [4]' in first.stdout
    assert "--run-name unit-t4" in first.stdout
    assert "--dataset-root" in first.stdout
    assert "--resume" not in first.stdout
    assert "tmux new-session -d -s dice-unit-t4" in first.stdout
    assert "DICE_SYNC_CMD" in first.stdout
    assert "MODAL_PROFILE='desmond-zee'" in first.stdout
    assert "DICE_DATA='" + env["DICE_DATA"] + "'" in first.stdout
    assert "--config-json" in first.stdout
    assert "source '" + env["DICE_REPO"] + "/brev/env.sh'" in first.stdout
    assert "modal volume ls dice-lingbot-rl-runs unit-t4/resume" in first.stdout
    run_dir = Path(env["DICE_DATA"]) / "runs" / "unit-t4"
    assert (run_dir / "run.sh").is_file()
    assert '"task_ids": [4]' in (run_dir / "config.json").read_text()
    fresh = _run("train.sh", ["unit-t4", "4", "--fresh"], env)
    assert fresh.returncode == 0, fresh.stderr
    assert "modal volume ls dice-lingbot-rl-runs" not in fresh.stdout
    (run_dir / "resume").mkdir()
    (run_dir / "resume" / "latest.pt").write_text("")
    second = _run("train.sh", ["unit-t4", "4"], env)
    assert "--resume" in second.stdout
    assert "modal volume ls dice-lingbot-rl-runs" not in second.stdout
    smoke = _run("train.sh", ["smoke", "0", "--max-env-steps", "32"], env)
    assert "--max-env-steps 32" in smoke.stdout
    assert '"wandb_entity": null' in smoke.stdout
    foreground = _run("train.sh", ["unit-t4", "4", "--no-tmux"], env)
    assert foreground.returncode == 0, foreground.stderr
    assert f"bash {run_dir / 'run.sh'}" in foreground.stdout
    assert "tmux new-session" not in foreground.stdout


def test_train_requires_run_name_task_and_prepared(tmp_path):
    env = _box(tmp_path)
    assert _run("train.sh", [], env).returncode != 0
    assert _run("train.sh", ["only-name"], env).returncode != 0
    Path(env["DICE_DATA"], "prepared.json").unlink()
    missing = _run("train.sh", ["r", "0"], env)
    assert missing.returncode != 0
    assert "prepared.json" in missing.stderr


def test_sync_skips_without_profile_and_puts_step_dir_then_final_state(tmp_path):
    env = _box(tmp_path)
    out = Path(env["DICE_DATA"]) / "runs" / "r"
    (out / "train_eval" / "step_080000").mkdir(parents=True)
    (out / "train_eval" / "step_080000" / "residual.pt").write_text("")
    (out / "residual.pt").write_text("")
    (out / "settings.json").write_text("{}")
    (out / "resume").mkdir()
    (out / "resume" / "latest.pt").write_text("")
    base = {**env, "DICE_OUTPUT_DIR": str(out), "DICE_STEP": "80000"}
    skipped = _run("sync.sh", ["r"], {**base, "MODAL_PROFILE": ""})
    assert skipped.returncode == 0
    assert "skip" in skipped.stdout
    put = _run("sync.sh", ["r"], base)
    assert put.returncode == 0, put.stderr
    assert "modal volume put dice-lingbot-rl-runs" in put.stdout
    assert f"{out}/train_eval/step_080000 r/train_eval/step_080000" in put.stdout
    assert f"{out}/settings.json r/settings.json" in put.stdout
    assert f"{out}/resume/latest.pt r/resume/latest.pt" in put.stdout
    assert "summary.json" not in put.stdout
    (out / "summary.json").write_text("{}")
    (out / "train.log").write_text("")
    final = _run("sync.sh", ["r"], {**base, "DICE_STEP": "660000"})
    assert f"{out}/resume/latest.pt r/resume/latest.pt" in final.stdout
    assert f"{out}/summary.json r/summary.json" in final.stdout
    assert f"{out}/train.log r/train.log" in final.stdout
    assert f"{out}/residual.pt r/residual.pt" in final.stdout


def test_pull_fetches_run_from_modal_volume(tmp_path):
    env = {"MODAL_PROFILE": "desmond-zee"}
    out = _run("pull.sh", ["r"], env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "modal volume get dice-lingbot-rl-runs r" in out.stdout
    assert str(Path("result/brev")) in out.stdout
    assert "MODAL_PROFILE=desmond-zee" in out.stdout
    assert _run("pull.sh", [], env, cwd=tmp_path).returncode != 0


def test_push_syncs_repo_without_heavy_dirs(tmp_path):
    plain = _run("push.sh", ["box"], {}, cwd=ROOT)
    assert plain.returncode == 0, plain.stderr
    for excluded in (".venv", ".cache", "checkpoints", "result", ".git", "brev/env.sh"):
        assert f"--exclude {excluded}" in plain.stdout
    assert "box:dice-rl-wam/" in plain.stdout
    assert _run("push.sh", [], {}, cwd=ROOT).returncode != 0


def test_setup_dry_run_pins_lerobot_revision_and_uses_both_modal_profiles(tmp_path):
    env = _box(tmp_path)
    (tmp_path / ".modal.toml").write_text("[source]\n[desmond-zee]\n")
    out = _run("setup.sh", [], env)
    assert out.returncode == 0, out.stderr
    revision = (ROOT / "script" / "lingbot_eval_config.py").read_text().split('LEROBOT_REVISION = "')[1].split('"')[0]
    assert f"git -C {env['DICE_DATA']}/lerobot checkout {revision}" in out.stdout
    assert "--extra lingbot_va --extra libero --extra evaluation --no-editable" in out.stdout
    assert "modal==1.1.4" in out.stdout
    assert "modal volume create dice-lingbot-rl-runs" in out.stdout
    assert "script.lingbot_eval prepare" in out.stdout
    assert "script.lingbot_rl_train check-inits --task-id 0" in out.stdout
    assert "hf-secret-value" not in out.stdout + out.stderr
    assert "wandb-secret-value" not in out.stdout + out.stderr
    Path(env["DICE_DATA"], "sft/libero30-sft/checkpoints/step_000600/norm_stats.json").unlink()
    pull = _run("setup.sh", [], env)
    assert pull.returncode == 0, pull.stderr
    assert "MODAL_PROFILE=source" in pull.stdout
    assert "modal volume get dice-lingbot-sft-runs libero30-sft/checkpoints/step_000600" in pull.stdout
    (tmp_path / ".modal.toml").unlink()
    missing = _run("setup.sh", [], env)
    assert missing.returncode != 0
    assert "modal token new" in missing.stderr
