import ast
import copy
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from datetime import datetime as real_datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch
from huggingface_hub.errors import HfHubHTTPError
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset

from script import lingbot_sft_data as data
from script.lingbot_sft_config import SFTConfig
from script.lingbot_sft_patch import apply_patch
from script.lingbot_sft_train import capture_rng, restore_rng


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("LINGBOT_UPSTREAM_ROOT", ROOT / ".cache" / "lingbot-va"))


def _fresh_upstream(tmp_path):
    if not UPSTREAM.exists():
        pytest.skip("set LINGBOT_UPSTREAM_ROOT or provide .cache/lingbot-va for upstream integration tests")
    checkout = tmp_path / "lingbot-va"
    checkout.mkdir()
    shutil.copytree(UPSTREAM / ".git", checkout / ".git")
    shutil.copytree(UPSTREAM / "wan_va", checkout / "wan_va")
    for relative in ("wan_va/train.py", "wan_va/modules/model.py"):
        original = subprocess.check_output(["git", "-C", str(UPSTREAM), "show", f"HEAD:{relative}"])
        (checkout / relative).write_bytes(original)
    return checkout


class ForeignCallable:
    def __reduce__(self):
        return (ForeignCallable, ())


def _episode(idx, task="task"):
    return {
        "episode_index": idx,
        "tasks": [task],
        "length": 12,
        "action_config": [{"start_frame": 0, "end_frame": 12, "action_text": task, "skill": ""}],
    }


def _episodes(tasks=10, per_task=50):
    return [_episode(task_id * per_task + idx, f"task-{task_id}") for task_id in range(tasks) for idx in range(per_task)]


def _hf_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response.url = "https://huggingface.example/xet?token=fake-HF_TOKEN"
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return HfHubHTTPError("fake-HF_TOKEN https://huggingface.example/xet", response=response)


def test_hf_download_retries_rate_limit_with_identical_request(monkeypatch, caplog):
    calls = []
    waits = []
    failures = [_hf_error(429)]
    def fake(function_arg, **kwargs):
        calls.append((function_arg, kwargs))
        if failures:
            raise failures.pop(0)
        return "cached/path"
    monkeypatch.setattr(data.time, "sleep", waits.append)
    assert data._hf_download(fake, "repo", revision="rev", token="fake-HF_TOKEN") == "cached/path"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert waits == [60.0]
    assert "fake-HF_TOKEN" not in caplog.text
    assert "huggingface.example" not in caplog.text


@pytest.mark.parametrize("status", [401, 403, 404])
def test_hf_download_auth_and_not_found_fail_without_retry(monkeypatch, status):
    calls = []
    waits = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        raise _hf_error(status)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        data._hf_download(fake, "repo", token="fake-HF_TOKEN")
    assert len(calls) == 1
    assert waits == []


def test_hf_download_retry_after_and_cooldown_limits(monkeypatch):
    waits = []
    failures = [_hf_error(429, "120")]
    def delayed(*args, **kwargs):
        if failures:
            raise failures.pop(0)
        return "ok"
    monkeypatch.setattr(data.time, "sleep", waits.append)
    assert data._hf_download(delayed, "repo") == "ok"
    assert waits == [120.0]
    waits.clear()
    def long_cooldown(*args, **kwargs):
        raise _hf_error(429, "901")
    with pytest.raises(RuntimeError, match="long cooldown"):
        data._hf_download(long_cooldown, "repo")
    assert waits == []


def test_hf_download_http_date_and_malformed_retry_after(monkeypatch):
    waits = []
    class FrozenDatetime:
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(data, "datetime", FrozenDatetime)
    failures = [_hf_error(429, "Thu, 01 Jan 2026 00:02:00 GMT")]
    def date_retry(*args, **kwargs):
        if failures:
            raise failures.pop(0)
        return "ok"
    monkeypatch.setattr(data.time, "sleep", waits.append)
    assert data._hf_download(date_retry, "repo") == "ok"
    assert waits == [120.0]
    waits.clear()
    failures = [_hf_error(429, "malformed")]
    assert data._hf_download(date_retry, "repo") == "ok"
    assert waits == [60.0]


def test_hf_download_network_timeout_and_persistent_rate_limit(monkeypatch):
    waits = []
    failures = [requests.Timeout("temporary")]
    def timeout_then_success(*args, **kwargs):
        if failures:
            raise failures.pop(0)
        return "ok"
    monkeypatch.setattr(data.time, "sleep", waits.append)
    assert data._hf_download(timeout_then_success, "repo") == "ok"
    assert waits == [60.0]
    waits.clear()
    def always_rate_limited(*args, **kwargs):
        raise _hf_error(429)
    with pytest.raises(RuntimeError, match="after four attempts"):
        data._hf_download(always_rate_limited, "repo")
    assert waits == [60.0, 120.0, 240.0]


def test_choose_episodes_is_stable_and_restricted():
    cfg = SFTConfig(seed=42)
    episodes = _episodes()
    chosen = data.choose_episodes(episodes, cfg)
    reversed_chosen = data.choose_episodes(list(reversed(episodes)), cfg)
    changed = data.choose_episodes(episodes, SFTConfig(seed=43))
    assert len(chosen) == 300
    assert chosen == reversed_chosen
    assert {ep["tasks"][0] for ep in chosen} == {f"task-{i}" for i in range(10)}
    assert [ep["episode_index"] for ep in chosen] != [ep["episode_index"] for ep in changed]
    with pytest.raises(ValueError, match="ten"):
        data.choose_episodes(_episodes(tasks=9), cfg)
    duplicate = _episodes()
    duplicate[-1]["episode_index"] = duplicate[0]["episode_index"]
    with pytest.raises(ValueError, match="unique"):
        data.choose_episodes(duplicate, cfg)
    insufficient = _episodes(per_task=50)
    insufficient = [ep for ep in insufficient if ep["tasks"][0] != "task-0" or ep["episode_index"] % 50 < 29]
    with pytest.raises(ValueError, match="Insufficient"):
        data.choose_episodes(insufficient, cfg)
    partial = _episodes()
    partial[0]["action_config"][0]["end_frame"] = 11
    with pytest.raises(ValueError, match="complete"):
        data.choose_episodes(partial, cfg)
    multiple = _episodes()
    multiple[0]["action_config"].append(copy.deepcopy(multiple[0]["action_config"][0]))
    with pytest.raises(ValueError, match="complete"):
        data.choose_episodes(multiple, cfg)
    mismatch = _episodes()
    mismatch[0]["action_config"][0]["action_text"] = "other"
    with pytest.raises(ValueError, match="disagree"):
        data.choose_episodes(mismatch, cfg)


def test_build_manifest_reads_only_selected_actions(monkeypatch, tmp_path):
    episodes = _episodes()
    cfg = SFTConfig(seed=42)
    selected = data.choose_episodes(episodes, cfg)
    selected_ids = {ep["episode_index"] for ep in selected}
    accessed = []

    info = {
        "codebase_version": "v2.1",
        "total_tasks": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {"action": {"shape": [7]}, **{camera: {"shape": [128, 128, 3]} for camera in data.CAMERAS}},
    }

    selected_values = [float((episode["episode_index"] % 17) / 10.0) for episode in selected]
    def fake_read_actions(path, episode):
        accessed.append(episode["episode_index"])
        value = float((episode["episode_index"] % 17) / 10.0) if episode["episode_index"] in selected_ids else 999999.0
        return np.full((episode["length"], 7), value, dtype=np.float32)

    monkeypatch.setattr(data, "read_actions", fake_read_actions)
    monkeypatch.setattr(data, "load_latent", lambda path: None)
    monkeypatch.setattr(data, "assemble_streams", lambda *args: {})
    manifest = data.build_manifest(tmp_path, info, episodes, cfg)
    assert set(accessed) == selected_ids
    assert len(accessed) == 300
    expected = np.quantile(np.asarray(selected_values), [0.01, 0.99], method="linear")
    assert np.allclose(manifest["norm_stat"]["q01"][:7], expected[0])
    assert np.allclose(manifest["norm_stat"]["q99"][:7], expected[1])
    assert 999999.0 not in manifest["norm_stat"]["q99"]


def test_empty_embedding_restores_no_grad_and_zero_pads(monkeypatch, tmp_path):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                input_ids=torch.zeros(1, 512, dtype=torch.long),
                attention_mask=torch.cat([torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 511, dtype=torch.long)], dim=1),
            )

    class FakeEncoder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def eval(self):
            return self
        def __call__(self, **kwargs):
            assert torch.is_grad_enabled() is False
            return SimpleNamespace(last_hidden_state=torch.ones(1, 512, 4096, dtype=torch.bfloat16))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(T5TokenizerFast=FakeTokenizer, UMT5EncoderModel=FakeEncoder))
    output = tmp_path / "empty.pt"
    with torch.enable_grad():
        assert torch.is_grad_enabled() is True
        data.create_empty_embedding(tmp_path, output)
        assert torch.is_grad_enabled() is True
    embedding = torch.load(output, map_location="cpu", weights_only=True)
    assert embedding.shape == (512, 4096)
    assert embedding.dtype == torch.bfloat16
    assert embedding[0].eq(1).all()
    assert embedding[1:].eq(0).all()


def test_prepare_requires_hf_token_before_download(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(data, "_hf_download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download called")))
    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        data.prepare(tmp_path, SFTConfig())


def test_prepare_downloads_use_hf_token_and_two_workers(monkeypatch, tmp_path):
    info = {
        "codebase_version": "v2.1",
        "total_tasks": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {"action": {"shape": [7]}, **{camera: {"shape": [128, 128, 3]} for camera in data.CAMERAS}},
    }
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info))
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text("\n")
    dataset_root = tmp_path / "dataset"
    model_root = tmp_path / "model"
    dataset_root.mkdir()
    model_root.mkdir()
    calls = []
    def fake_download(function, *args, **kwargs):
        calls.append((function.__name__, args, kwargs))
        if function.__name__ == "hf_hub_download":
            return info_path if args[1] == "meta/info.json" else episodes_path
        return dataset_root if args[0] == data.DATASET_REPO else model_root
    monkeypatch.setenv("HF_TOKEN", "fake-HF_TOKEN")
    monkeypatch.setattr(data, "_hf_download", fake_download)
    monkeypatch.setattr(data, "choose_episodes", lambda episodes, cfg: [])
    monkeypatch.setattr(data, "build_manifest", lambda *args: {"fingerprint": "fp", "norm_stat": {"q01": [0.0] * 30, "q99": [1.0] * 30}})
    monkeypatch.setattr(data, "create_empty_embedding", lambda root, output: torch.save(torch.zeros(512, 4096, dtype=torch.bfloat16), output))
    paths = data.prepare(tmp_path, SFTConfig())
    assert paths["dataset_fingerprint"] == "fp"
    assert len(calls) == 4
    for _, _, kwargs in calls:
        assert kwargs["token"] == "fake-HF_TOKEN"
        if "max_workers" in kwargs:
            assert kwargs["max_workers"] == 2


def test_modal_prepare_commits_cache_on_success_and_failure(monkeypatch, tmp_path):
    from script import lingbot_sft_modal as modal_module

    class FakeVolume:
        def __init__(self):
            self.reloads = 0
            self.commits = 0
        def reload(self):
            self.reloads += 1
        def commit(self):
            self.commits += 1
    volume = FakeVolume()
    monkeypatch.setattr(modal_module, "cache_volume", volume)
    real_path = Path
    monkeypatch.setattr(modal_module, "Path", lambda value: tmp_path if value == "/cache" else real_path(value))
    monkeypatch.setenv("HF_TOKEN", "fake-HF_TOKEN")
    raw_prepare = modal_module.prepare.info.raw_f
    monkeypatch.setattr(data, "prepare", lambda *args: {"dataset_fingerprint": "fp", "dataset_path": "d"})
    result = raw_prepare(1, "project", None)
    assert result["fingerprint"] == "fp"
    assert volume.reloads == 1 and volume.commits == 1
    def failing_prepare(*args):
        raise HfHubHTTPError("fake-HF_TOKEN https://huggingface.example", response=_hf_error(429).response)
    monkeypatch.setattr(data, "prepare", failing_prepare)
    with pytest.raises(RuntimeError, match="SFT preparation failed") as failure:
        raw_prepare(1, "project", None)
    assert "fake-HF_TOKEN" not in str(failure.value)
    assert volume.reloads == 2 and volume.commits == 2


def test_assemble_streams_alignment_and_masks():
    episode = _episode(0, "task")
    actions = np.arange(84, dtype=np.float32).reshape(12, 7) / 100
    norm = {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}
    text = torch.zeros(512, 4096, dtype=torch.bfloat16)
    streams = [
        {
            "frame_ids": np.arange(9, dtype=np.int64),
            "start_frame": 0,
            "end_frame": 12,
            "video_num_frames": 9,
            "video_height": 128,
            "video_width": 128,
            "fps": 60,
            "ori_fps": 60,
            "latent_num_frames": 3,
            "latent_height": 8,
            "latent_width": 8,
            "latent": torch.ones(3 * 8 * 8, 48, dtype=torch.bfloat16),
            "text_emb": text,
            "text": "task",
        },
        {
            "frame_ids": np.arange(9, dtype=np.int64),
            "start_frame": 0,
            "end_frame": 12,
            "video_num_frames": 9,
            "video_height": 128,
            "video_width": 128,
            "fps": 60,
            "ori_fps": 60,
            "latent_num_frames": 3,
            "latent_height": 8,
            "latent_width": 8,
            "latent": torch.full((3 * 8 * 8, 48), 2, dtype=torch.bfloat16),
            "text_emb": text,
            "text": "task",
        },
    ]
    result = data.assemble_streams(streams, actions, episode, norm)
    assert result["latents"].shape == (48, 3, 8, 16)
    assert result["latents"][:, :, :, :8].eq(1).all()
    assert result["latents"][:, :, :, 8:].eq(2).all()
    assert result["actions"].shape == (30, 3, 4, 1)
    raw = np.pad(actions, ((4, 0), (0, 0)))[:12]
    expected = np.clip((raw - np.asarray(norm["q01"][:7])) / 2.000001 * 2 - 1, -1.5, 1.5)
    expected = torch.from_numpy(expected.astype(np.float32).reshape(3, 4, 7).transpose(2, 0, 1)).unsqueeze(-1)
    assert torch.allclose(result["actions"][:7], expected)
    assert result["actions"][7:].eq(0).all()
    assert result["actions_mask"][:7].all()
    assert not result["actions_mask"][7:].any()
    with pytest.raises(ValueError, match="finite seven"):
        data.assemble_streams(streams, np.zeros((11, 7), dtype=np.float32), episode, norm)
    nonfinite = actions.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite seven"):
        data.assemble_streams(streams, nonfinite, episode, norm)
    bad = copy.deepcopy(streams)
    bad[0]["fps"] = 30
    with pytest.raises(ValueError, match="source metadata"):
        data.assemble_streams(bad, actions, episode, norm)
    bad = copy.deepcopy(streams)
    bad[1]["frame_ids"] = np.arange(1, 10, dtype=np.int64)
    with pytest.raises(ValueError, match="alignment"):
        data.assemble_streams(bad, actions, episode, norm)
    bad = copy.deepcopy(streams)
    bad[0]["frame_ids"] = np.arange(0, 18, 2, dtype=np.int64)
    with pytest.raises(ValueError, match="contiguous"):
        data.assemble_streams(bad, actions, episode, norm)
    bad = copy.deepcopy(streams)
    bad[0]["latent"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="latent"):
        data.assemble_streams(bad, actions, episode, norm)
    bad = copy.deepcopy(streams)
    bad[0]["text"] = "wrong"
    with pytest.raises(ValueError, match="text"):
        data.assemble_streams(bad, actions, episode, norm)


def test_restricted_latent_loader_accepts_numpy_and_rejects_foreign(tmp_path):
    payload = {
        "latent": torch.zeros(8 * 8, 48, dtype=torch.bfloat16),
        "frame_ids": np.arange(2, dtype=np.int64),
        "latent_num_frames": 1,
        "latent_height": 8,
        "latent_width": 8,
        "text_emb": torch.zeros(512, 4096, dtype=torch.bfloat16),
        "text": "task",
    }
    path = tmp_path / "latent.pth"
    torch.save(payload, path)
    assert np.array_equal(data.load_latent(path)["frame_ids"], np.arange(2, dtype=np.int64))
    foreign = tmp_path / "foreign.pth"
    torch.save({"callable": ForeignCallable()}, foreign)
    with pytest.raises(pickle.UnpicklingError):
        data.load_latent(foreign)


def test_restricted_latent_loader_accepts_published_numpy_core_pickle_name(tmp_path):
    payload = {
        "latent": torch.zeros(8 * 8, 48, dtype=torch.bfloat16),
        "frame_ids": np.arange(2, dtype=np.int64),
        "latent_num_frames": 1,
        "latent_height": 8,
        "latent_width": 8,
        "text_emb": torch.zeros(512, 4096, dtype=torch.bfloat16),
        "text": "task",
    }
    path = tmp_path / "latent.pth"
    torch.save(payload, path)
    raw = path.read_bytes()
    old, new = b"numpy._core.multiarray", b"numpy.core.multiarray"
    if old not in raw:
        pytest.skip("pickle already uses numpy.core.multiarray")
    if raw[:2] == b"PK":
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
            infos = list(zin.infolist())
            contents = {info.filename: zin.read(info.filename) for info in infos}
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zout:
            for info in infos:
                payload = contents[info.filename]
                if old in payload:
                    payload = payload.replace(old, new)
                copied = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                copied.compress_type = info.compress_type
                zout.writestr(copied, payload)
        path.write_bytes(out.getvalue())
    else:
        path.write_bytes(raw.replace(old, new))
    assert np.array_equal(data.load_latent(path)["frame_ids"], np.arange(2, dtype=np.int64))


def test_config_accumulation_and_weight_milestones():
    cfg = SFTConfig()
    assert cfg.accumulation_steps(8) == 10
    assert cfg.accumulation_steps(4) == 20
    assert cfg.num_steps == 1000
    assert all(cfg.save_weights_at(step) for step in (100, 200, 400, 600, 800, 1000))
    assert SFTConfig(num_steps=30).save_weights_at(30)
    with pytest.raises(ValueError, match="clips"):
        SFTConfig(grad_clip=1.0).validate()
    with pytest.raises(ValueError, match="betas"):
        SFTConfig(beta1=1.0).validate()


def _fake_trainer(tmp_path, model, optimizer, scheduler, spec=None):
    from script import lingbot_sft_train as train

    if not UPSTREAM.exists():
        pytest.skip("set LINGBOT_UPSTREAM_ROOT or provide .cache/lingbot-va for native integration tests")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    cls = train._native_sft_class()
    fake = object.__new__(cls)
    fake.run_dir = tmp_path
    fake.step = 2
    fake.config = SimpleNamespace(rank=0, world_size=1)
    fake.spec = spec or SFTConfig()
    fake.paths = {"dataset_fingerprint": "dataset-fingerprint", "manifest_path": str(tmp_path / "manifest.json")}
    fake.patch_hashes = {"train.py": "hash"}
    fake.run_name = "tiny-run"
    fake.resume_requested = False
    fake.modal_volume = None
    fake.wandb_run = None
    fake.transformer = model
    fake.optimizer = optimizer
    fake.lr_scheduler = scheduler
    fake.data_epoch = 0
    fake.data_offset = 2
    fake.train_loader_iter = None
    fake.train_loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=1, shuffle=True,
                                   generator=torch.Generator().manual_seed(19), num_workers=0)
    return fake


def test_production_checkpoint_round_trip_and_atomic_failure(monkeypatch, tmp_path):
    from script import lingbot_sft_train as train

    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
    fake = _fake_trainer(tmp_path, model, optimizer, scheduler)
    fake.save_checkpoint()
    latest = tmp_path / "resume" / "latest.pt"
    saved_bytes = latest.read_bytes()
    expected_model = copy.deepcopy(model)
    expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=0.01)
    expected_scheduler = torch.optim.lr_scheduler.LambdaLR(expected_optimizer, lambda step: 0.9**step)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    expected_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    expected_input = torch.randn(4, 3)
    expected_optimizer.zero_grad()
    expected_model(expected_input).sum().backward()
    expected_optimizer.step()
    expected_scheduler.step()
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 0.9**step)
    resumed = _fake_trainer(tmp_path, restored, restored_optimizer, restored_scheduler)
    resumed.resume_from_latest()
    actual_input = torch.randn(4, 3)
    assert torch.equal(actual_input, expected_input)
    restored_optimizer.zero_grad()
    restored(actual_input).sum().backward()
    restored_optimizer.step()
    restored_scheduler.step()
    assert all(torch.equal(restored.state_dict()[key], expected_model.state_dict()[key]) for key in expected_model.state_dict())
    assert restored_scheduler.state_dict() == expected_scheduler.state_dict()
    monkeypatch.setattr(train.torch, "save", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        fake._write_latest({}, {}, [])
    assert latest.read_bytes() == saved_bytes


def test_production_weights_publish_collision_and_retry(monkeypatch, tmp_path):
    from safetensors.torch import load_file
    from script import lingbot_sft_train as train

    manifest = {"norm_stat": {"q01": [0.0] * 30, "q99": [1.0] * 30}, "fingerprint": "manifest"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    model = torch.nn.Linear(3, 2)
    model.config = {"_name_or_path": "base", "hidden": 2}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    fake = _fake_trainer(tmp_path, model, optimizer, scheduler, SFTConfig(num_steps=2, weight_steps=(2,)))
    stale = tmp_path / "checkpoints" / ".step_000002-stale"
    stale.mkdir(parents=True)
    fake.save_checkpoint()
    milestone = tmp_path / "checkpoints" / "step_000002"
    assert not (milestone / "optimizer.pt").exists()
    assert all(t.dtype == torch.bfloat16 for t in load_file(str(milestone / "transformer" / "diffusion_pytorch_model.safetensors")).values())
    latest = (tmp_path / "resume" / "latest.pt").read_bytes()
    fake.save_checkpoint()
    assert (tmp_path / "resume" / "latest.pt").read_bytes() != b""
    monkeypatch.setattr(train, "save_file", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("weights disk full")))
    with pytest.raises(RuntimeError, match="Checkpoint save failed"):
        fake.save_checkpoint()
    assert (tmp_path / "resume" / "latest.pt").read_bytes() == latest


def test_nonrank_checkpoint_error_broadcast_is_observable(monkeypatch):
    from script import lingbot_sft_train as train

    class FakeDist:
        def broadcast_object_list(self, values, src):
            values[0] = {"ok": False, "error": "rankzero failed"}
    monkeypatch.setattr(train, "_distributed", lambda: True)
    monkeypatch.setattr(train, "_rank", lambda: 1)
    monkeypatch.setattr(train, "dist", FakeDist())
    with pytest.raises(RuntimeError, match="rankzero failed"):
        train._raise_status({"ok": True, "error": ""})


def test_resume_identity_allows_increase_and_rejects_mismatch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    fake = _fake_trainer(tmp_path, model, optimizer, scheduler)
    metadata = fake._metadata([fake._rank_state()])
    fake._validate_resume(metadata)
    fake.spec = SFTConfig(num_steps=2)
    with pytest.raises(ValueError, match="num_steps"):
        fake._validate_resume(metadata)
    fake.spec = SFTConfig(learning_rate=2e-5)
    with pytest.raises(ValueError, match="training identity"):
        fake._validate_resume(metadata)


@pytest.mark.parametrize("start_offset", [2, 4])
def test_production_data_cursor_resume_matches_continuous(tmp_path, start_offset):
    class Episodes(Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return {"episode": idx, "text_emb": torch.ones(2)}

    def setup(run_dir):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        fake = _fake_trainer(run_dir, model, optimizer, scheduler, SFTConfig(cfg_prob=0.5))
        dataset = Episodes()
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
        fake.train_loader = DataLoader(dataset, batch_size=1, sampler=sampler, generator=torch.Generator().manual_seed(73))
        fake.empty_emb = torch.zeros(2)
        fake.data_offset = start_offset
        fake.train_loader_iter = None
        return fake

    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    continuous = setup(tmp_path)
    continuous._position_data_iterator()
    continuous.save_checkpoint()
    expected_batch = continuous._get_next_batch()
    expected_draws = (random.random(), np.random.rand(), torch.rand(3))
    resumed = setup(tmp_path)
    resumed.resume_from_latest()
    actual_batch = resumed._get_next_batch()
    actual_draws = (random.random(), np.random.rand(), torch.rand(3))
    assert torch.equal(expected_batch["episode"], actual_batch["episode"])
    assert torch.equal(expected_batch["text_emb"], actual_batch["text_emb"])
    assert actual_draws[0] == expected_draws[0]
    assert actual_draws[1] == expected_draws[1]
    assert torch.equal(actual_draws[2], expected_draws[2])
    assert torch.equal(continuous.train_loader.generator.get_state(), resumed.train_loader.generator.get_state())


def test_prepared_paths_require_known_keys_and_pinned_model(monkeypatch, tmp_path):
    from script import lingbot_sft_train as train
    from script.lingbot_sft_config import MODEL_REVISION

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"extra": "ok"}))
    with pytest.raises(ValueError, match="missing keys"):
        train._load_prepared(missing, SFTConfig())
    model_root = tmp_path / MODEL_REVISION
    transformer = model_root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fingerprint": "fp", "norm_stat": {"q01": [0.0] * 30, "q99": [1.0] * 30}}))
    paths = {
        "dataset_path": str(tmp_path),
        "manifest_path": str(manifest),
        "model_path": str(model_root),
        "empty_emb_path": str(tmp_path / "empty.pt"),
        "dataset_fingerprint": "fp",
        "extra": "allowed",
    }
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(paths))
    monkeypatch.setattr("script.lingbot_sft_data.validate_manifest", lambda *args: None)
    assert train._load_prepared(prepared, SFTConfig()) == paths


def test_rng_capture_restore_exact():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_patch_idempotence_and_revision_rejection(tmp_path):
    checkout = _fresh_upstream(tmp_path)
    first = apply_patch(checkout)
    second = apply_patch(checkout)
    assert all(first["changed"].values())
    assert not any(second["changed"].values())
    assert first["hashes"] == second["hashes"]
    import script.lingbot_sft_patch as patch

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(patch, "UPSTREAM_REVISION", "wrong-revision")
    try:
        with pytest.raises(RuntimeError, match="revision mismatch"):
            apply_patch(checkout)
    finally:
        monkeypatch.undo()


def test_modal_functions_leave_region_routing_unpinned():
    import inspect
    from script import lingbot_sft_modal as modal_module

    tree = ast.parse(inspect.getsource(modal_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "function":
            assert not {keyword.arg for keyword in node.keywords if keyword.arg} & {"region", "cloud", "routing_region"}


def test_modal_image_source_layout_uses_public_uv_api():
    from script import lingbot_sft_modal as modal_module

    assert modal_module.REQUIREMENTS.exists()
    assert "lingbot_sft_requirements.txt" in modal_module.SCRIPT_FILES
    source_root = modal_module.ROOT / "script"
    assert all((source_root / filename).exists() for filename in modal_module.SCRIPT_FILES)
    assert "/.uv/uv" not in modal_module._build_image.__code__.co_consts


def test_modal_entrypoint_and_gpu_propagation(monkeypatch, tmp_path):
    from script import lingbot_sft_modal as modal_module

    calls = []
    monkeypatch.setattr(modal_module.prepare, "remote", lambda *args: calls.append(("prepare", args)) or {"prepared": "prepared.json"})
    monkeypatch.setattr(modal_module.train4, "remote", lambda *args: calls.append(("train4", args)) or "four")
    monkeypatch.setattr(modal_module.train8, "remote", lambda *args: calls.append(("train8", args)) or "eight")
    downloads = []
    monkeypatch.setattr(modal_module, "download_checkpoint", lambda *args: downloads.append(args) or str(tmp_path / "downloaded"))
    raw_main = modal_module.main.info.raw_f
    raw_main("train", 4, 12, "run", "project", "entity", True, str(tmp_path))
    raw_main("train", 8, 13, "run8", "project8", "", False, str(tmp_path))
    assert calls == [
        ("prepare", (12, "project", "entity")),
        ("train4", ("prepared.json", "run", 12, "project", "entity", True)),
        ("prepare", (13, "project8", None)),
        ("train8", ("prepared.json", "run8", 13, "project8", None, False)),
    ]
    assert downloads == [("run", 12, str(tmp_path)), ("run8", 13, str(tmp_path))]
    raw_main("prepare", 4, 14, "", "project", "", False, str(tmp_path))
    assert len(downloads) == 2
    collision = tmp_path / "collision" / "run" / "step_000012"
    collision.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing"):
        raw_main("train", 4, 12, "run", "project", "", False, str(tmp_path / "collision"))
    with pytest.raises(ValueError, match="stage"):
        raw_main("invalid", 4, 1, "run")
    with pytest.raises(ValueError, match="stage"):
        raw_main("train", 2, 1, "run")
    assert not any(call[0] == "prepare" and call[1][0] == 1 for call in calls)


def test_modal_gpu_run_lock_and_volume_reload(monkeypatch):
    from script import lingbot_sft_modal as modal_module

    class FakeVolume:
        def __init__(self):
            self.events = []
        def reload(self):
            self.events.append("reload")
        def commit(self):
            self.events.append("commit")
    class FakeLocks:
        def __init__(self):
            self.value = None
            self.events = []
        def put(self, key, value, skip_if_exists=False):
            self.events.append(("put", key, skip_if_exists))
            if self.value is not None:
                return False
            self.value = value
            return True
        def get(self, key):
            self.events.append(("get", key))
            return self.value
        def pop(self, key):
            self.events.append(("pop", key))
            self.value = None
    cache = FakeVolume()
    runs = FakeVolume()
    locks = FakeLocks()
    monkeypatch.setattr(modal_module, "cache_volume", cache)
    monkeypatch.setattr(modal_module, "runs_volume", runs)
    monkeypatch.setattr(modal_module, "run_locks", locks)
    commands = []
    monkeypatch.setattr(modal_module.subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    result4 = modal_module._run_train("prepared.json", "run4", 4, 12, "project", None, True)
    result8 = modal_module._run_train("prepared.json", "run8", 8, 13, "project", "entity", False)
    assert result4["checkpoint"] == "run4/checkpoints/step_000012"
    assert result8["checkpoint"] == "run8/checkpoints/step_000013"
    assert any("--nproc_per_node=4" in command for command, _ in commands)
    assert any("--nproc_per_node=8" in command for command, _ in commands)
    assert cache.events == ["reload", "reload"]
    assert runs.events == ["reload", "commit", "reload", "commit"]
    assert [event[0] for event in locks.events] == ["put", "get", "pop", "put", "get", "pop"]
    locks.value = "stale"
    with pytest.raises(RuntimeError, match="already active"):
        modal_module._run_train("prepared.json", "run4", 4, 12, "project", None, False)


def test_download_checkpoint_success_failure_and_collision(monkeypatch, tmp_path):
    from script import lingbot_sft_modal as modal_module

    relative = (
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/config.json",
        "norm_stats.json",
        "sft_config.json",
        "dataset_manifest.json",
    )
    calls = []
    def fake_run(command, check):
        calls.append((command, check))
        destination = Path(command[-1])
        for name in relative:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")
    monkeypatch.setattr(modal_module.subprocess, "run", fake_run)
    result = modal_module.download_checkpoint("run", 12, tmp_path)
    assert result == str(tmp_path / "run" / "step_000012")
    assert "optimizer" not in " ".join(str(path) for path in (tmp_path / "run").rglob("*"))
    assert calls and calls[0][0][-2:] == ["run/checkpoints/step_000012", str(tmp_path / "run" / "step_000012")]
    with pytest.raises(FileExistsError, match="Refusing"):
        modal_module.download_checkpoint("run", 12, tmp_path)
    def failing_run(command, check):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(modal_module.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="remote checkpoint is retained"):
        modal_module.download_checkpoint("failed", 12, tmp_path)
    incomplete = tmp_path / "incomplete"
    def incomplete_run(command, check):
        destination = Path(command[-1])
        (destination / "transformer").mkdir(parents=True, exist_ok=True)
        (destination / "transformer" / "config.json").write_bytes(b"fake")
    monkeypatch.setattr(modal_module.subprocess, "run", incomplete_run)
    with pytest.raises(RuntimeError, match="Incomplete checkpoint download"):
        modal_module.download_checkpoint("incomplete", 12, tmp_path)


def test_native_training_math_functions_are_unchanged():
    if not UPSTREAM.exists():
        pytest.skip("set LINGBOT_UPSTREAM_ROOT or provide .cache/lingbot-va for upstream integration tests")
    current = (UPSTREAM / "wan_va" / "train.py").read_text()
    original = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "show", "HEAD:wan_va/train.py"], text=True
    )
    current_tree = ast.parse(current)
    original_tree = ast.parse(original)
    names = {"_add_noise", "_prepare_input_dict", "compute_loss", "_train_step"}
    def methods(tree):
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
        return {node.name: ast.dump(node) for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert methods(current_tree) == methods(original_tree)


def test_wandb_initializes_before_metrics_and_preserves_rng(monkeypatch, tmp_path):
    from script import lingbot_sft_train as train

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"norm_stat": {"q01": [0.0] * 30, "q99": [1.0] * 30}}))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    fake = _fake_trainer(tmp_path, model, optimizer, scheduler)
    fake.paths["manifest_path"] = str(manifest)
    calls = []
    state = {"initialized": False}
    class Run:
        id = "run-id"
        name = "run-name"
        def define_metric(self, *args, **kwargs):
            assert state["initialized"]
            calls.append(("define", args, kwargs))
        def log_artifact(self, artifact):
            calls.append(("artifact", artifact.files))
        def log(self, *args, **kwargs):
            pass
        def finish(self, **kwargs):
            pass
    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []
        def add_file(self, path, name=None):
            self.files.append((path, name))
    class Wandb:
        def init(self, **kwargs):
            state["initialized"] = True
            calls.append(("init", kwargs))
            return Run()
    Wandb.Artifact = Artifact
    monkeypatch.setitem(sys.modules, "wandb", Wandb())
    torch.manual_seed(123)
    saved = train.capture_rng()
    fake._start_wandb()
    assert calls[0][0] == "init"
    assert calls[0][1]["resume"] == "never"
    assert calls[0][1]["allow_val_change"] is True
    assert calls[0][1]["config"]["patch_hashes"] == fake.patch_hashes
    assert any(call[0] == "artifact" for call in calls)
    assert torch.equal(torch.rand(3), (train.restore_rng(saved), torch.rand(3))[1])
    fake.resume_requested = True
    calls.clear()
    state["initialized"] = False
    fake._start_wandb()
    assert calls[0][1]["resume"] == "must"
    fake.config.rank = 1
    calls.clear()
    fake._start_wandb()
    assert calls == []


def test_secret_sdk_create_hides_key_and_validates_input(monkeypatch, capsys):
    import modal
    from script import lingbot_sft_secret as secret

    calls = []
    monkeypatch.setattr(secret.getpass, "getpass", lambda prompt: "secret-value")
    def fake_create(*args, **kwargs):
        calls.append((args, kwargs))
    monkeypatch.setattr(modal.Secret.objects, "create", fake_create)
    secret.main([])
    output = capsys.readouterr().out
    secret.main(["--service", "hf"])
    output += capsys.readouterr().out
    assert calls == [
        (("dice-lingbot-wandb", {"WANDB_API_KEY": "secret-value"}), {}),
        (("dice-lingbot-hf", {"HF_TOKEN": "secret-value"}), {}),
    ]
    assert "secret-value" not in output
    assert output.count("WANDB_API_KEY") == 1
    assert output.count("HF_TOKEN") == 1

    monkeypatch.setattr(secret.getpass, "getpass", lambda prompt: "")
    with pytest.raises(ValueError, match="nonempty"):
        secret.main([])
    monkeypatch.setattr(secret.getpass, "getpass", lambda prompt: "line1\nline2")
    with pytest.raises(ValueError, match="single-line"):
        secret.main([])
    assert len(calls) == 2

    def failing_create(*args, **kwargs):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(modal.Secret.objects, "create", failing_create)
    monkeypatch.setattr(secret.getpass, "getpass", lambda prompt: "another-fake")
    with pytest.raises(RuntimeError, match="backend unavailable"):
        secret.main([])
    assert "another-fake" not in capsys.readouterr().out


def test_train_and_config_help_without_native_import(monkeypatch, capsys):
    from script import lingbot_sft_config as config
    from script import lingbot_sft_train as train

    monkeypatch.setattr(sys, "argv", ["lingbot_sft_config", "--gpus", "8"])
    config.main()
    assert "gradient_accumulation_steps" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        train.main(["--help"])
    assert exc.value.code == 0
    assert "--upstream-root" in capsys.readouterr().out


def test_cpu_native_import_after_patch(tmp_path):
    checkout = _fresh_upstream(tmp_path)
    apply_patch(checkout)
    sys.path.insert(0, str(checkout))
    try:
        module = __import__("wan_va.train", fromlist=["Trainer"])
        assert hasattr(module, "Trainer")
    finally:
        sys.path.remove(str(checkout))
