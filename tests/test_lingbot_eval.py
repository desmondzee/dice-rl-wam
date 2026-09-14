import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from script.lingbot_eval_config import EvalConfig, episode_plan
from script.lingbot_eval import aggregate_results, observation_batch, decode_action, run_episode


def test_protocol_and_episode_plan():
    smoke = EvalConfig(stage="smoke").validate()
    evaluation = EvalConfig(stage="eval").validate()
    assert smoke.episodes_per_task == 1
    assert evaluation.episodes_per_task == 20
    assert evaluation.protocol()["max_policy_steps"] == 520
    assert evaluation.protocol()["video_steps"] == 20
    assert evaluation.protocol()["action_steps"] == 50
    assert episode_plan(smoke, 0, 50)[0]["init_state_id"] == 0
    plan = episode_plan(evaluation, 0, 50)
    assert [row["init_state_id"] for row in plan] == list(range(1, 21))
    assert len({row["seed"] for row in plan}) == 20
    assert plan == episode_plan(replace(evaluation, checkpoint_step=200), 0, 50)
    assert plan != episode_plan(evaluation, 1, 50)
    with pytest.raises(ValueError, match="initial states"):
        episode_plan(evaluation, 0, 20)


@pytest.mark.parametrize("changes", [
    {"source_run": "../other"}, {"stage": "all"}, {"checkpoint_step": 0},
    {"seed": -1}, {"checkpoint_step": 1001},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(EvalConfig(), **changes).validate()


def observation():
    first = np.zeros((128, 128, 3), dtype=np.uint8)
    first[0, 0] = [10, 20, 30]
    first[-1, 0] = [40, 50, 60]
    second = np.full_like(first, 127)
    return {"pixels": {"image": first, "image2": second}}


def test_camera_order_and_vertical_only_flip():
    batch = observation_batch(observation(), "task text", "cpu")
    image = batch["observation.images.image"]
    assert image.shape == (1, 3, 128, 128)
    torch.testing.assert_close(image[0, :, 0, 0], torch.tensor([40, 50, 60]) / 255)
    torch.testing.assert_close(image[0, :, -1, 0], torch.tensor([10, 20, 30]) / 255)
    assert image[0, :, 0, -1].sum() == 0
    torch.testing.assert_close(batch["observation.images.image2"], torch.full_like(image, 127 / 255))
    assert batch["task"] == ["task text"]


def normalization():
    return {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}


def test_native_action_denormalization():
    values = torch.tensor([[-1.0, 0.0, 1.0, 0.5, -0.5, 2.0, -2.0]])
    decoded = decode_action(values, normalization())
    expected = (values.numpy()[0] + 1) / 2 * (2 + 1e-6) - 1
    np.testing.assert_allclose(decoded, expected)
    assert decoded.shape == (7,)
    assert decoded[5] > 1
    with pytest.raises(ValueError, match="finite"):
        decode_action(torch.full((1, 7), float("nan")), normalization())
    with pytest.raises(ValueError, match="shape"):
        decode_action(torch.zeros(1, 30), normalization())


class FakePolicy:
    def __init__(self):
        self.resets = 0
        self.calls = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        assert not torch.is_grad_enabled()
        self.calls += 1
        return torch.zeros(1, 7)


class FakeEnv:
    def __init__(self, success_at=None, terminate_at=None, truncate_at=None):
        self.success_at = success_at
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.init_state_id = 0
        self.steps = 0
        self.selected_initial_state = None

    def reset(self, seed):
        self.selected_initial_state = self.init_state_id
        self.init_state_id += 1
        self.steps = 0
        return observation(), {"is_success": False}

    def step(self, action):
        assert action.shape == (7,)
        self.steps += 1
        success = self.steps == self.success_at
        return observation(), 0.0, success or self.steps == self.terminate_at, self.steps == self.truncate_at, {"is_success": success}


@pytest.mark.parametrize("env_kwargs,success,steps,truncated", [
    ({"success_at": 2}, True, 2, False),
    ({"terminate_at": 1}, False, 1, False),
    ({"truncate_at": 2}, False, 2, True),
    ({}, False, 3, True),
])
def test_rollout_success_timeout_and_reset(env_kwargs, success, steps, truncated):
    policy = FakePolicy()
    env = FakeEnv(**env_kwargs)
    entry = {"task_id": 0, "episode_index": 0, "init_state_id": 7, "seed": 123}
    result = run_episode(policy, env, entry, "task text", normalization(), "cpu", max_steps=3)
    assert result["success"] is success
    assert result["policy_steps"] == steps
    assert result["truncated"] is truncated
    assert result["init_state_id"] == 7
    assert env.selected_initial_state == 7
    assert policy.resets == 1
    assert policy.calls == steps


def test_aggregation_has_no_missing_or_duplicate_episode_bias():
    rows = [
        {"task_id": 0, "episode_index": 0, "success": True},
        {"task_id": 0, "episode_index": 1, "success": False},
        {"task_id": 1, "episode_index": 0, "success": False},
        {"task_id": 1, "episode_index": 1, "success": False},
    ]
    result = aggregate_results(rows, task_ids=(0, 1), episodes_per_task=2)
    assert result["complete"] is True
    assert result["successes"] == 1
    assert result["completed_episodes"] == 4
    assert result["macro_success_rate"] == 0.25
    assert result["per_task"]["0"]["success_rate"] == 0.5
    partial = aggregate_results(rows[:2], task_ids=(0, 1), episodes_per_task=2)
    assert partial["complete"] is False
    assert partial["macro_success_rate"] is None
    assert partial["per_task"]["1"]["success_rate"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_results(rows + rows[:1], task_ids=(0, 1), episodes_per_task=2)


def test_actual_checkpoint_tensor_schema_without_loading_weights():
    pytest.importorskip("lerobot")
    from script.lingbot_eval import validate_transformer_schema
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "checkpoints/lingbot-sft/libero30-sft/step_000600"
    if not checkpoint.is_dir():
        pytest.skip("Local step-600 checkpoint not present")
    architecture = json.loads((checkpoint / "transformer/config.json").read_text())
    validate_transformer_schema(checkpoint, architecture)


def test_checkpoint_provenance_and_normalization(tmp_path):
    from script.lingbot_eval import read_checkpoint_metadata, write_json, REQUIRED_FILES
    from script.lingbot_sft_config import provenance
    checkpoint = tmp_path / "checkpoint"
    norm = normalization()
    manifest = {"provenance": provenance(), "fingerprint": "test", "norm_stat": norm,
        "episodes": [{"episode_index": i, "tasks": [f"task-{i // 30}"]} for i in range(300)],
        "task_counts": {f"task-{i}": 30 for i in range(10)}}
    checkpoint.mkdir()
    for name in REQUIRED_FILES:
        write_json(checkpoint / name, {})
    write_json(checkpoint / "dataset_manifest.json", manifest)
    write_json(checkpoint / "norm_stats.json", norm)
    write_json(checkpoint / "transformer/config.json", {"_class_name": "WanTransformer3DModel", "action_dim": 30, "in_channels": 48})
    assert read_checkpoint_metadata(checkpoint)["normalization"] == norm
    wrong = {**norm, "q01": [0.0] * 30}
    write_json(checkpoint / "norm_stats.json", wrong)
    with pytest.raises(ValueError, match="normalization differs"):
        read_checkpoint_metadata(checkpoint)
    write_json(checkpoint / "norm_stats.json", norm)
    manifest["provenance"]["suite"] = "libero_object"
    write_json(checkpoint / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError, match="provenance"):
        read_checkpoint_metadata(checkpoint)


def test_download_retry_and_secret_redaction(monkeypatch):
    import huggingface_hub
    import httpx
    import script.lingbot_eval as module
    from huggingface_hub.errors import HfHubHTTPError
    calls = []
    sleeps = []
    request = httpx.Request("GET", "https://example.test/assets")
    response = httpx.Response(429, headers={"Retry-After": "45"}, request=request)
    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        if len(calls) == 1:
            raise HfHubHTTPError("private-test-token", response=response)
        return "cached"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    assert module.download_snapshot("repo", token="private-test-token", revision="pin") == "cached"
    assert calls[0] == calls[1]
    assert calls[0][1]["max_workers"] == 2
    assert sleeps == [45.0]
    def denied(*args, **kwargs):
        raise HfHubHTTPError("private-test-token", response=httpx.Response(403, request=request))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied)
    with pytest.raises(RuntimeError, match="HTTP 403") as exc:
        module.download_snapshot("repo", token="private-test-token")
    assert "private-test-token" not in str(exc.value)


def test_evaluation_persists_and_resumes_without_recounting(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import imageio.v2 as imageio
    import script.lingbot_eval as module
    from script.lingbot_eval_config import LEROBOT_REVISION, LIBERO_ASSETS_REVISION
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in module.REQUIRED_FILES:
        module.write_json(checkpoint / name, {})
    tasks = [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50} for task_id in range(10)]
    prepared = {"checkpoint": str(checkpoint), "source_run": "libero30-sft", "checkpoint_step": 600,
        "lerobot_revision": LEROBOT_REVISION, "libero_assets_revision": LIBERO_ASSETS_REVISION,
        "file_sizes": {name: (checkpoint / name).stat().st_size for name in module.REQUIRED_FILES},
        "file_sha256": {name: module.file_sha256(checkpoint / name) for name in module.REQUIRED_FILES},
        "tasks": tasks, "model_path": "cached-model", "assets_path": "cached-assets"}
    prepared_path = tmp_path / "prepared.json"
    module.write_json(prepared_path, prepared)
    monkeypatch.setattr(module, "read_checkpoint_metadata", lambda path: {"architecture": {}, "normalization": normalization()})
    monkeypatch.setattr(module, "describe_suite", lambda path: (object(), tasks))
    monkeypatch.setattr(module, "load_policy", lambda *args: object())
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "pinned-test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    closes = []
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", SimpleNamespace(LiberoEnv=lambda **kwargs: SimpleNamespace(close=lambda: closes.append(True))))
    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    runs = []
    def start_wandb(**kwargs):
        run = SimpleNamespace(summary={}, log=lambda *args, **kwargs: None, finish=lambda **kwargs: None)
        runs.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=start_wandb))
    visited = []
    fail = [True]
    def rollout(policy, env, entry, instruction, norm, device, **kwargs):
        if entry["task_id"] == 1 and fail[0]:
            fail[0] = False
            raise RuntimeError("simulated interruption")
        visited.append(entry["task_id"])
        return {**entry, "success": entry["task_id"] % 2 == 0, "seconds": 1.0, "policy_steps": 3}
    monkeypatch.setattr(module, "run_episode", rollout)
    commits = []
    output = tmp_path / "results"
    cfg = EvalConfig(stage="smoke")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.evaluate(cfg, prepared_path, output, "test-run", commit=lambda: commits.append(True))
    partial = module.read_json(output / "summary.json")
    assert partial["completed_episodes"] == 1 and partial["macro_success_rate"] is None
    assert module.read_json(output / "status.json")["state"] == "interrupted_or_failed"
    summary = module.evaluate(cfg, prepared_path, output, "test-run", resume=True, commit=lambda: commits.append(True))
    assert summary["complete"] and summary["completed_episodes"] == 10
    assert summary["macro_success_rate"] == 0.5
    assert visited == list(range(10))
    assert runs[0]["resume"] == "never" and runs[1]["resume"] == "must"
    assert runs[0]["id"] == runs[1]["id"]
    assert len(commits) >= 10
    assert len(closes) == 11
    with pytest.raises(FileExistsError):
        module.evaluate(cfg, prepared_path, output, "test-run")
    with pytest.raises(ValueError, match="identical"):
        module.evaluate(replace(cfg, seed=43), prepared_path, output, "test-run", resume=True)
    module.write_json(output / "summary.json", {"complete": False})
    recovered = module.evaluate(cfg, prepared_path, output, "test-run", resume=True)
    assert recovered == summary
    assert module.read_json(output / "summary.json") == summary
    assert module.read_json(output / "status.json")["state"] == "completed"
    assert visited == list(range(10))


def test_non_boolean_success_is_not_counted_as_success():
    class InvalidSuccessEnv(FakeEnv):
        def step(self, action):
            return observation(), 0.0, True, False, {"is_success": "false"}
    entry = {"task_id": 0, "episode_index": 0, "init_state_id": 0, "seed": 123}
    with pytest.raises(ValueError, match="boolean is_success"):
        run_episode(FakePolicy(), InvalidSuccessEnv(), entry, "task", normalization(), "cpu", max_steps=1)


def test_results_download_uses_existing_parent(tmp_path, monkeypatch):
    import script.lingbot_eval_modal as module
    calls = []
    def download(command, check):
        calls.append(command)
        parent = Path(command[-1])
        assert parent.is_dir()
        target = parent / command[-2]
        complete_report_fixture(target)
    monkeypatch.setattr(module.subprocess, "run", download)
    result = module.download_results("eval-run", tmp_path / "downloads")
    assert Path(result) == tmp_path / "downloads/eval-run"
    assert (Path(result) / "report.html").is_file()
    assert (Path(result) / "episodes.csv").is_file()
    assert (Path(result) / "tasks.csv").is_file()
    assert json.loads((Path(result) / "report_data.json").read_text())["overall"]["completed_episodes"] == 4
    assert calls[0][-1] == str(tmp_path / "downloads")
    assert "--force" not in calls[0]
    with pytest.raises(FileExistsError):
        module.download_results("eval-run", tmp_path / "downloads")


def test_download_rejects_missing_episode_despite_complete_summary(tmp_path, monkeypatch):
    import script.lingbot_eval_modal as module

    def download(command, check):
        target = complete_report_fixture(Path(command[-1]) / command[-2])
        (target / "episodes/task_00/episode_001.json").unlink()

    monkeypatch.setattr(module.subprocess, "run", download)
    with pytest.raises(ValueError, match="Missing or unexpected episode JSON"):
        module.download_results("eval-run", tmp_path / "downloads")
    assert not (tmp_path / "downloads/eval-run/report.html").exists()


def test_cpu_preparation_commits_cache_on_failure(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import script.lingbot_eval_modal as module
    events = []
    monkeypatch.setattr(module, "cache", SimpleNamespace(reload=lambda: None, commit=lambda: events.append("commit")))
    monkeypatch.setattr(module, "source", SimpleNamespace(reload=lambda: None))
    def fail(*args, **kwargs):
        raise RuntimeError("simulated preparation failure")
    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="simulated preparation failure"):
        module.prepare.local(asdict(EvalConfig()))
    assert events == ["commit"]


def test_modal_gpu_count_and_read_only_source_mounts():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "script/lingbot_eval_modal.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    gpu = {kw.arg: kw.value for kw in functions["run_evaluation"].decorator_list[0].keywords}
    assert ast.literal_eval(gpu["gpu"]) == "H100"
    assert ast.literal_eval(gpu["retries"]) == 0
    assert not {"region", "cloud", "routing_region"} & set(gpu)
    mounts = gpu["volumes"]
    entries = {ast.literal_eval(key): value for key, value in zip(mounts.keys, mounts.values)}
    assert entries["/cache"].func.attr == "read_only"
    assert entries["/sft"].func.attr == "read_only"


def test_modal_image_preserves_locked_package_index(monkeypatch):
    import shlex
    from unittest.mock import MagicMock
    import script.lingbot_eval_modal as module

    image = MagicMock()
    for method in ("apt_install", "uv_pip_install", "run_commands", "add_local_file", "env"):
        getattr(image, method).return_value = image
    monkeypatch.setattr(module.modal.Image, "debian_slim", lambda **kwargs: image)
    assert module.build_image() is image
    commands = [shlex.split(command) for call in image.run_commands.call_args_list
                for command in call.args if command.startswith("uv ")]
    assert len(commands) == 3
    for command in commands:
        assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
        if command[1] in ("sync", "export"):
            assert "--locked" in command
            assert "--frozen" not in command
        else:
            assert command[1:3] == ["pip", "install"]
            assert "--constraint" in command
            assert command[command.index("--exclude-newer") + 1] == "2026-09-05T00:00:00Z"


def complete_report_fixture(root):
    from script.lingbot_eval import write_json
    root.mkdir(parents=True, exist_ok=True)
    tasks = [{"task_id": task, "name": f"task_{task}", "instruction": f"instruction {task}",
              "initial_state_count": 50} for task in range(2)]
    plan = [{"task_id": task, "episode_index": episode, "init_state_id": episode + 1,
             "seed": 42 + task * 2 + episode} for task in range(2) for episode in range(2)]
    rows = []
    for index, entry in enumerate(plan):
        success = entry["episode_index"] == 0
        row = {**entry, "instruction": tasks[entry["task_id"]]["instruction"], "success": success,
               "policy_steps": 5 if success else 10, "seconds": 10.0 * (index + 1),
               "terminated": success, "truncated": not success, "peak_gpu_memory_bytes": 1024 * (index + 1)}
        if entry["episode_index"] == 0:
            row["video"] = f"videos/task_{entry['task_id']:02d}.mp4"
            (root / "videos").mkdir(exist_ok=True)
            (root / row["video"]).write_bytes(b"test-video")
        write_json(root / "episodes" / f"task_{entry['task_id']:02d}" / f"episode_{entry['episode_index']:03d}.json", row)
        rows.append(row)
    write_json(root / "settings.json", {
        "config": {"stage": "eval", "source_run": "test-sft", "checkpoint_step": 600,
                   "protocol": {"suite": "libero_10", "task_ids": [0, 1], "episodes_per_task": 2,
                                "max_policy_steps": 10, "initial_state_offset": 1,
                                "environment_batch_size": 1, "video_steps": 20, "action_steps": 50}},
        "checkpoint": {"source_run": "test-sft", "checkpoint_step": 600, "tasks": tasks},
        "episode_plan": plan, "packages": {"torch": "test"}})
    write_json(root / "summary.json", aggregate_results(rows, task_ids=(0, 1), episodes_per_task=2))
    write_json(root / "status.json", {"state": "completed"})
    return root


def test_results_report_metrics_exports_and_idempotence(tmp_path):
    import csv
    from script.lingbot_eval_report import create_report
    root = complete_report_fixture(tmp_path / "evaluation")
    original = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report = create_report(root)
    overall = report["overall"]
    assert overall["completed_episodes"] == 4
    assert overall["successes"] == 2 and overall["failures"] == 2
    assert overall["success_rate"] == 0.5
    assert overall["total_policy_steps"] == 30
    assert overall["mean_policy_steps"] == 7.5
    assert overall["total_episode_seconds"] == 100.0
    assert overall["mean_episode_seconds"] == overall["median_episode_seconds"] == 25.0
    assert overall["episodes_at_step_limit"] == 2
    assert overall["recorded_videos"] == 2
    assert overall["peak_gpu_memory_bytes"] == 4096
    assert report["tasks"][0]["total_episode_seconds"] == 30.0
    assert report["tasks"][1]["total_episode_seconds"] == 70.0
    with (root / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with (root / "tasks.csv").open(newline="") as handle:
        tasks = list(csv.DictReader(handle))
    assert len(episodes) == 4 and len(tasks) == 2
    assert [int(row["seed"]) for row in episodes] == [42, 43, 44, 45]
    assert episodes[0]["video"] == "videos/task_00.mp4" and episodes[1]["video"] == ""
    assert episodes[-1]["record_path"] == "episodes/task_01/episode_001.json"
    assert json.loads((root / "report_data.json").read_text()) == report
    html = (root / "report.html").read_text()
    assert "50.0%" in html and "Not recorded" in html
    assert "episodes/task_01/episode_001.json" in html
    assert "model loading" in html and "cumulative" in html
    assert set(report["source_sha256"]) == set(original)
    assert all((root / name).read_bytes() == data for name, data in original.items())
    rendered = {name: (root / name).read_bytes() for name in ("report.html", "episodes.csv", "tasks.csv", "report_data.json")}
    assert create_report(root) == report
    assert all((root / name).read_bytes() == data for name, data in rendered.items())


@pytest.mark.parametrize("damage", [
    "missing_episode", "extra_episode", "wrong_identity", "missing_video", "unsafe_video",
    "wrong_summary", "incomplete_status", "bad_success", "bad_time", "partial_plan",
])
def test_results_report_rejects_incomplete_or_inconsistent_results(tmp_path, damage):
    from script.lingbot_eval_report import create_report
    root = complete_report_fixture(tmp_path / "evaluation")
    episode = root / "episodes/task_00/episode_000.json"
    if damage == "missing_episode":
        episode.unlink()
    elif damage == "extra_episode":
        (episode.parent / "extra.json").write_bytes(episode.read_bytes())
    elif damage == "missing_video":
        (root / "videos/task_00.mp4").unlink()
    elif damage == "wrong_summary":
        summary = json.loads((root / "summary.json").read_text())
        summary["successes"] = 3
        (root / "summary.json").write_text(json.dumps(summary))
    elif damage == "incomplete_status":
        (root / "status.json").write_text('{"state":"interrupted_or_failed"}')
    elif damage == "partial_plan":
        settings = json.loads((root / "settings.json").read_text())
        settings["episode_plan"].pop()
        (root / "settings.json").write_text(json.dumps(settings))
    else:
        row = json.loads(episode.read_text())
        key, value = {"wrong_identity": ("seed", 999), "unsafe_video": ("video", "../../outside.mp4"),
                      "bad_success": ("success", "false"), "bad_time": ("seconds", float("nan"))}[damage]
        row[key] = value
        episode.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        create_report(root)
    assert not (root / "report.html").exists()
    assert not (root / "episodes.csv").exists()


def test_results_report_escapes_html_and_spreadsheet_formulas(tmp_path):
    import csv
    from script.lingbot_eval_report import create_report
    root = complete_report_fixture(tmp_path / "evaluation")
    instruction = '=HYPERLINK("https://example.test", "<script>alert(1)</script>")'
    settings = json.loads((root / "settings.json").read_text())
    settings["checkpoint"]["tasks"][0]["instruction"] = instruction
    (root / "settings.json").write_text(json.dumps(settings))
    for episode in range(2):
        path = root / "episodes/task_00" / f"episode_{episode:03d}.json"
        row = json.loads(path.read_text())
        row["instruction"] = instruction
        path.write_text(json.dumps(row))
    create_report(root)
    html = (root / "report.html").read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    with (root / "episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["instruction"] == "'" + instruction


def test_results_report_does_not_overwrite_existing_reports(tmp_path):
    from script.lingbot_eval_report import create_report
    root = complete_report_fixture(tmp_path / "evaluation")
    (root / "report.html").write_text("user content")
    with pytest.raises(FileExistsError):
        create_report(root)
    assert (root / "report.html").read_text() == "user content"
    assert not (root / "report_data.json").exists()
