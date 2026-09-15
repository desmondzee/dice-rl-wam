import io
import random
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from script.lingbot_eval import decode_action
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_model import (
    ACTION_DIM, ADAM_LR, BETA, ENSEMBLE, EPSILON, GAMMA, HIDDEN, HORIZON,
    K_CANDIDATES, N_STEP_CHUNKS, STATE_DIM, TAU, USED_DOF, UTD, DiceResidualModel, apply_residual,
    mask_unused_dof,
)
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, mlp_to_model, model_to_mlp,
    pool_critic_state, slice_env_actions,
)


def test_config_pins_released_libero_sampler_and_step_600():
    cfg = RLConfig().validate()
    proto = cfg.protocol()
    assert proto["video_steps"] == 20
    assert proto["action_steps"] == 50
    assert proto["video_exec_step"] == -1
    assert proto["sampler"] == "released_lingbot_libero_defaults"
    assert proto["video_guidance"] == 5.0
    assert proto["action_guidance"] == 1.0
    assert cfg.checkpoint_step == 600
    assert cfg.source_run == "libero30-sft"
    assert cfg.wandb_project == "dice-lingbot-va-rl"
    assert cfg.k_candidates == 4
    assert cfg.online_env_steps == 100_000
    assert proto["snr_shift"] == 5.0
    assert proto["action_snr_shift"] == 0.05
    assert proto["dtype"] == "bfloat16"
    assert proto["attention_backend"] == "torch"
    assert proto["attention_window"] == 30
    assert proto["action_normalization_epsilon"] == 1e-6
    assert proto["noisy_history"] is False
    assert proto["text_encoder_device"] == "cpu"
    assert proto["environment_batch_size"] == 1
    assert proto["max_policy_steps"] == 520
    assert proto["used_action_channels"] == list(range(7))
    assert proto["camera_orientation"] == "vertical_flip_only_native_lingbot"
    assert proto["residual_input"] == "z"
    assert proto["mlp_hidden"] == [1024, 1024, 1024]
    assert proto["critic_ensemble"] == 10
    assert proto["beta"] == 100.0
    assert proto["epsilon"] == -0.5
    assert proto["n_step_chunks"] == 3
    assert proto["gamma"] == 0.99
    assert proto["utd"] == 10
    assert proto["tau"] == 0.01
    assert proto["adam_lr"] == 1e-4
    assert proto["batch_size"] == 256
    assert proto["rlpd_start"] == 0.5
    assert proto["rlpd_end"] == 0.1
    assert proto["replay_capacity"] == 100_000
    assert proto["train_eval_every"] == 25_000
    assert proto["train_eval_episodes_per_task"] == 1
    assert proto["comparison_episodes_per_task"] == 20
    assert proto["comparison_initial_state_offset"] == 1
    assert proto["comparison_seed"] == 42
    assert proto["checkpoint_step"] == 600
    assert proto["source_run"] == "libero30-sft"


@pytest.mark.parametrize("changes", [
    {"video_steps": 3},
    {"action_steps": 10},
    {"video_exec_step": 12},
    {"checkpoint_step": 400},
    {"source_run": "other-run"},
    {"k_candidates": 16},
    {"train_eval_every": 5_000},
    {"train_eval_episodes_per_task": 20},
])
def test_config_rejects_sampler_and_recipe_drift(changes):
    with pytest.raises(ValueError):
        replace(RLConfig(), **changes).validate()


def test_rlpd_ratio_anneals_over_env_steps():
    cfg = RLConfig().validate()
    assert cfg.rlpd_expert_ratio(0) == pytest.approx(0.5)
    assert cfg.rlpd_expert_ratio(50_000) == pytest.approx(0.3)
    assert cfg.rlpd_expert_ratio(100_000) == pytest.approx(0.1)
    assert cfg.rlpd_expert_ratio(120_000) == pytest.approx(0.1)


def test_model_hyperparameters_match_baseline_protocol():
    proto = RLConfig().protocol()
    assert (STATE_DIM, HORIZON, ACTION_DIM, USED_DOF) == (3072, 16, 30, 7)
    assert list(HIDDEN) == proto["mlp_hidden"]
    assert ENSEMBLE == proto["critic_ensemble"] == 10
    assert BETA == proto["beta"] == 100.0
    assert EPSILON == proto["epsilon"] == -0.5
    assert N_STEP_CHUNKS == proto["n_step_chunks"] == 3
    assert GAMMA == proto["gamma"] == 0.99
    assert TAU == proto["tau"] == 0.01
    assert ADAM_LR == proto["adam_lr"] == 1e-4
    assert UTD == proto["utd"] == 10
    model = DiceResidualModel(device="cpu")
    assert len(model.critic.heads) == 10


def test_residual_mask_zeros_unused_dof():
    torch.manual_seed(0)
    a_base = torch.ones(2, HORIZON, ACTION_DIM)
    residual = torch.randn(2, HORIZON, ACTION_DIM)
    a = apply_residual(a_base, residual)
    assert a.shape == (2, HORIZON, ACTION_DIM)
    assert torch.count_nonzero(a[:, :, USED_DOF:]) == 0
    torch.testing.assert_close(a[:, :, :USED_DOF], a_base[:, :, :USED_DOF] + residual[:, :, :USED_DOF])


def test_n_step_chunk_return_sparse_terminal():
    model = DiceResidualModel(device="cpu")
    done = torch.tensor([[1.0], [1.0], [1.0]])
    next_state = torch.zeros(3, STATE_DIM)
    next_action = torch.zeros(3, HORIZON, ACTION_DIM)
    n_steps = torch.tensor([[3.0], [2.0], [1.0]])
    summed = torch.tensor([[0.99 ** 2], [0.99], [1.0]])
    target = model.n_step_target(summed, done, next_state, next_action, n_steps)
    assert target.shape == (3, 1)
    torch.testing.assert_close(target[2], torch.tensor([1.0]))
    assert target[0].item() == pytest.approx(0.99 ** 2)
    assert target[1].item() == pytest.approx(0.99)


def test_actor_critic_one_step_update_on_random_tensors():
    torch.manual_seed(1)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(8, STATE_DIM)
    noise = torch.randn(8, HORIZON, ACTION_DIM)
    a_base = mask_unused_dof(torch.randn(8, HORIZON, ACTION_DIM))
    is_expert = torch.zeros(8, 1)
    is_expert[:4] = 1
    reward = torch.zeros(8, 1)
    done = torch.zeros(8, 1)
    n_steps = torch.ones(8, 1) * 3
    next_state = torch.randn(8, STATE_DIM)
    next_action = mask_unused_dof(torch.randn(8, HORIZON, ACTION_DIM))
    target = model.n_step_target(reward, done, next_state, next_action, n_steps)
    before = {k: v.detach().clone() for k, v in model.actor.named_parameters()}
    critic_info = model.update_critic(
        state, apply_residual(a_base, model.actor(state, noise).detach()), target, is_expert)
    actor_info = model.update_actor(state, noise, a_base, is_expert, target)
    assert torch.isfinite(torch.tensor(critic_info["critic_loss"]))
    assert torch.isfinite(torch.tensor(actor_info["actor_loss"]))
    assert 0.0 <= actor_info["bc_filter_rate"] <= 1.0
    assert any(not torch.equal(before[k], v) for k, v in model.actor.named_parameters())


def _replay_row(reward, done, expert=False):
    return {
        "s": np.zeros(STATE_DIM, np.float32),
        "z": np.zeros((HORIZON, ACTION_DIM), np.float32),
        "a_base": np.zeros((HORIZON, ACTION_DIM), np.float32),
        "a": np.zeros((HORIZON, ACTION_DIM), np.float32),
        "reward": np.float32(reward),
        "done": np.float32(done),
        "s_next": np.ones(STATE_DIM, np.float32),
        "task_id": 0,
        "n_env_actions": 16,
        "is_expert": np.float32(expert),
    }


def test_n_step_sparse_terminal_three_chunks():
    buf = ChunkReplay(capacity=32)
    for reward, done in ((0, 0), (0, 0), (1, 1)):
        buf.add_online(_replay_row(reward, done))
    buf.finalize_episode()
    rows = buf.rows()
    assert [float(row["n_steps"]) for row in rows] == [3.0, 2.0, 1.0]
    assert [float(row["done"]) for row in rows] == [1.0, 1.0, 1.0]
    assert float(rows[0]["reward"]) == pytest.approx(GAMMA ** 2)
    assert float(rows[1]["reward"]) == pytest.approx(GAMMA)
    assert float(rows[2]["reward"]) == pytest.approx(1.0)


def test_open_episode_sample_bootstraps_from_successor():
    buf = ChunkReplay(capacity=32)
    first = _replay_row(0, 0)
    first["s"] = np.zeros(STATE_DIM, np.float32)
    first["a"] = np.zeros((HORIZON, ACTION_DIM), np.float32)
    second = _replay_row(0, 0)
    second["s"] = np.ones(STATE_DIM, np.float32)
    second["a"] = np.full((HORIZON, ACTION_DIM), 2.0, np.float32)
    buf.add_online(first)
    assert not buf.has_ready_online()
    with pytest.raises(ValueError, match="no online"):
        buf.sample(1, expert_ratio=0.0)
    buf.add_online(second)
    assert buf.has_ready_online()
    batch = buf.sample(1, expert_ratio=0.0)
    np.testing.assert_array_equal(batch["s"][0].numpy(), np.zeros(STATE_DIM, np.float32))
    np.testing.assert_array_equal(batch["s_next"][0].numpy(), np.ones(STATE_DIM, np.float32))
    np.testing.assert_array_equal(batch["a_next"][0].numpy(), np.full((HORIZON, ACTION_DIM), 2.0, np.float32))
    assert float(batch["n_steps"][0]) == 2.0
    assert float(batch["done"][0]) == 0.0


def test_rlpd_mix_respects_scheduled_ratio():
    buf = ChunkReplay(capacity=200)
    for _ in range(80):
        buf.add_online(_replay_row(0, 0))
    buf.finalize_episode()
    for _ in range(80):
        buf.add_expert(_replay_row(0, 1, expert=True))
    ratio = RLConfig().validate().rlpd_expert_ratio(0)
    assert ratio == pytest.approx(0.5)
    counts = [float(buf.sample(20, expert_ratio=ratio)["is_expert"].mean()) for _ in range(40)]
    assert abs(sum(counts) / len(counts) - 0.5) < 0.15


def _eval_normalization():
    return {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}


def test_first_chunk_execute_length_12_later_16():
    assert env_action_count(True) == 12
    assert env_action_count(False) == 16
    chunk = torch.zeros(1, 16, 30)
    chunk[0, :, 0] = torch.arange(16)
    first = slice_env_actions(chunk, True)
    later = slice_env_actions(chunk, False)
    assert first.shape == (1, 12, 7)
    assert later.shape == (1, 16, 7)
    torch.testing.assert_close(first[0, :, 0], torch.arange(4, 16).float())
    torch.testing.assert_close(later[0, :, 0], torch.arange(16).float())


def test_denorm_matches_eval_decode_action():
    chunk = torch.zeros(1, 16, 30)
    chunk[0, 4, :7] = torch.tensor([-1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0])
    sliced = slice_env_actions(chunk, True)
    decoded = decode_action(sliced[:, :1, :].reshape(1, 7), _eval_normalization())
    expected = decode_action(chunk[0:1, 4, :7], _eval_normalization())
    np.testing.assert_allclose(decoded, expected)


def test_mean_pool_depends_on_text():
    torch.manual_seed(0)
    video = torch.randn(2, 8, 3072)
    s_a = pool_critic_state(video, torch.zeros(2, 4, 3072))
    s_b = pool_critic_state(video, torch.ones(2, 4, 3072))
    assert s_a.shape == (2, 3072)
    assert not torch.allclose(s_a, s_b)


def test_model_mlp_roundtrip():
    torch.manual_seed(0)
    model = torch.randn(2, 30, 4, 4, 1)
    model[..., 7:, :, :, :] = 0
    torch.testing.assert_close(mlp_to_model(model_to_mlp(model)), model)


def test_histogram_entropy_is_per_coordinate():
    torch.manual_seed(0)
    peaked = torch.zeros(32, 16, 30)
    spread = torch.randn(32, 16, 30)
    assert histogram_entropy(peaked) < histogram_entropy(spread)


def test_frozen_prior_kwargs_match_sft_eval_protocol():
    from script.lingbot_rl_policy import frozen_prior_kwargs

    kwargs = frozen_prior_kwargs()
    proto = RLConfig().protocol()
    assert kwargs["num_inference_steps"] == proto["video_steps"] == 20
    assert kwargs["action_num_inference_steps"] == proto["action_steps"] == 50
    assert kwargs["video_exec_step"] == proto["video_exec_step"] == -1
    assert kwargs["guidance_scale"] == proto["video_guidance"] == 5.0
    assert kwargs["action_guidance_scale"] == proto["action_guidance"] == 1.0
    assert kwargs["snr_shift"] == proto["snr_shift"] == 5.0
    assert kwargs["action_snr_shift"] == proto["action_snr_shift"] == 0.05
    assert kwargs["attn_mode"] == proto["attention_backend"] == "torch"
    assert kwargs["dtype"] == proto["dtype"] == "bfloat16"
    assert kwargs["attn_window"] == proto["attention_window"] == 30
    assert kwargs["image_hflip"] is proto["image_hflip"] is False
    assert kwargs["camera_layout"] == proto["camera_layout"] == "width_concat"
    assert kwargs["text_encoder_device"] == proto["text_encoder_device"] == "cpu"
    assert kwargs["used_action_channel_ids"] == proto["used_action_channels"] == list(range(7))
    assert kwargs["obs_cam_keys"] == proto["camera_keys"]
    assert kwargs["save_predicted_video"] is False
    assert kwargs["device"] == "cuda"


def test_policy_class_exposes_collection_api():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    for name in ("decode_candidates", "extract_critic_state", "extract_critic_state_from_latent",
                 "commit_executed", "select_action"):
        assert callable(getattr(ResidualLingBotPolicy, name))


def test_commit_executed_writes_residual_not_base():
    from types import SimpleNamespace

    from script.lingbot_rl_policy import ResidualLingBotPolicy

    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.config = SimpleNamespace(device="cpu")
    policy.dtype = torch.float32
    a_base = torch.zeros(1, HORIZON, ACTION_DIM)
    residual = torch.ones(1, HORIZON, ACTION_DIM)
    executed = apply_residual(a_base, residual)
    policy.commit_executed(executed)
    written = model_to_mlp(policy._executed_actions)
    torch.testing.assert_close(written[:, :, :USED_DOF], executed[:, :, :USED_DOF])
    assert torch.count_nonzero(written[:, :, USED_DOF:]) == 0
    assert not torch.equal(written, a_base)
    first = torch.ones(1, HORIZON, ACTION_DIM)
    policy.commit_executed(first, first_chunk=True)
    written_first = model_to_mlp(policy._executed_actions)
    assert torch.count_nonzero(written_first[:, :4]) == 0
    torch.testing.assert_close(written_first[:, 4:, :USED_DOF], torch.ones(1, 12, USED_DOF))


def test_later_chunk_start_obs_allows_none_batch():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy._first_chunk = True
    policy._extract_raw_obs = lambda batch: {"from": "batch"}
    with pytest.raises(RuntimeError, match="First chunk"):
        policy._start_raw_obs(None)
    assert policy._start_raw_obs({"ok": True}) == {"from": "batch"}
    policy._first_chunk = False
    sentinel = {"from": "buffer"}
    policy._obs_buffer = [sentinel]
    assert policy._start_raw_obs(None) is sentinel
    assert policy._start_raw_obs({"ok": True}) == {"from": "batch"}


def test_isolated_critic_encode_clears_then_restores_streaming_vae():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    dirty = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)

    class Streaming:
        def __init__(self):
            self.feat_cache = [dirty.clone()]
            self.cleared = False

        def clear_cache(self):
            self.cleared = True
            self.feat_cache = [None]

    vae = Streaming()
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy._frozen = {"streaming_vae": vae}
    seen = []

    def encode(_frames):
        seen.append(list(vae.feat_cache))
        vae.feat_cache = [torch.ones_like(dirty)]
        return torch.zeros(1, 48, 1, 8, 16)

    policy._encode_frames = encode
    latent = policy._encode_isolated([{"obs": 1}])
    assert latent.shape[1] == 48
    assert vae.cleared
    assert seen == [[None]]
    torch.testing.assert_close(vae.feat_cache[0], dirty)


def test_expand_conditional_kv_repeats_conditional_batch_only():
    from types import SimpleNamespace

    from script.lingbot_rl_policy import expand_conditional_kv

    orig_k = torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1)
    orig_v = torch.tensor([30.0, 40.0]).reshape(2, 1, 1, 1)
    block = SimpleNamespace(attn1=SimpleNamespace(attn_caches={
        "pos": {"k": orig_k.clone(), "v": orig_v.clone()},
    }))
    expand_conditional_kv(SimpleNamespace(blocks=[block]), 4)
    cache = block.attn1.attn_caches["pos"]
    assert cache["k"].shape[0] == 4
    torch.testing.assert_close(cache["k"], torch.full((4, 1, 1, 1), 10.0))
    torch.testing.assert_close(cache["v"], torch.full((4, 1, 1, 1), 30.0))


def test_expand_conditional_kv_raises_when_cache_missing():
    from types import SimpleNamespace

    from script.lingbot_rl_policy import expand_conditional_kv

    block = SimpleNamespace(attn1=SimpleNamespace(attn_caches={}))
    with pytest.raises(RuntimeError, match="action candidate batching failed"):
        expand_conditional_kv(SimpleNamespace(blocks=[block]), 4)


def test_expert_chunk_stride_and_padding(tmp_path):
    from script.lingbot_rl_data import featurize_experts, normalize_demo_action

    class StubPolicy:
        def extract_critic_state(self, batch):
            return torch.zeros(1, 3072)

        def reset(self):
            return None

    actions = np.zeros((12 + 16 + 4, 7), np.float32)
    actions[:, 0] = np.arange(32)
    episode = {"actions": actions, "task": "put the moka pot", "task_id": 8, "frames": [object()] * 32}
    rows = featurize_experts(
        StubPolicy(), [episode], norm=_eval_normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(rows) == 3
    assert rows[0]["n_env_actions"] == 12
    assert rows[1]["n_env_actions"] == 16
    assert rows[2]["n_env_actions"] == 4
    assert rows[0]["a"].shape == (16, 30)
    assert np.count_nonzero(rows[0]["a"][:, 7:]) == 0
    assert rows[-1]["done"] == 1
    np.testing.assert_array_equal(rows[0]["z"], np.zeros((16, 30), np.float32))
    np.testing.assert_array_equal(rows[0]["a_base"], rows[0]["a"])
    again = featurize_experts(
        StubPolicy(), [episode], norm=_eval_normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(again) == 3
    raw = np.array([-1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0], np.float32)
    normed = normalize_demo_action(raw, _eval_normalization())
    decoded = decode_action(torch.from_numpy(normed).reshape(1, 7), _eval_normalization())
    np.testing.assert_allclose(decoded, raw, atol=1e-5)


def test_featurize_experts_pools_published_latents(tmp_path):
    from script.lingbot_rl_data import featurize_experts

    class StubPolicy:
        def reset(self):
            return None

        def extract_critic_state(self, batch):
            raise AssertionError("RGB path should not run when published latents exist")

        def extract_critic_state_from_latent(self, latent, batch):
            assert latent.shape[2] == 1
            return torch.full((1, 3072), float(latent[0, 0, 0, 0, 0]))

    latents = torch.zeros(1, 48, 8, 8, 16)
    latents[0, 0, 3] = 7
    actions = np.zeros((12 + 16, 7), np.float32)
    episode = {"actions": actions, "task": "put the moka pot", "task_id": 8, "latents": latents}
    rows = featurize_experts(
        StubPolicy(), [episode], norm=_eval_normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(rows) == 2
    assert float(rows[0]["s"][0]) == 0.0
    assert float(rows[1]["s"][0]) == 7.0


def _rewrite_numpy_core_pickle_global(path):
    raw = path.read_bytes()
    old, new = b"numpy._core.multiarray", b"numpy.core.multiarray"
    if old not in raw:
        return False
    if raw[:2] != b"PK":
        path.write_bytes(raw.replace(old, new))
        return True
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
        infos = list(zin.infolist())
        contents = {info.filename: zin.read(info.filename) for info in infos}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zout:
        for info in infos:
            data = contents[info.filename]
            if old in data:
                data = data.replace(old, new)
            copied = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
            copied.compress_type = info.compress_type
            zout.writestr(copied, data)
    path.write_bytes(out.getvalue())
    return True


def test_load_latent_accepts_published_numpy_core_pickle_name(tmp_path):
    from script.lingbot_sft_data import load_latent

    path = tmp_path / "latent.pth"
    torch.save({"frame_ids": np.arange(2, dtype=np.int64), "text": "task"}, path)
    if not _rewrite_numpy_core_pickle_global(path):
        pytest.skip("pickle already uses numpy.core.multiarray")
    loaded = load_latent(path)
    assert np.array_equal(loaded["frame_ids"], np.arange(2, dtype=np.int64))


def _latent_episode_layout(tmp_path, camera_bytes):
    from script.lingbot_sft_config import CAMERAS

    episode = {"episode_index": 0, "length": 12, "tasks": ["put the moka pot"]}
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    files = []
    for camera, payload in zip(CAMERAS, camera_bytes):
        path = tmp_path / f"latents/chunk-000/{camera}/episode_000000_0_12.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append(path)
    return info, episode, files


def test_try_load_latents_returns_none_for_truncated_zip(tmp_path):
    from script.lingbot_rl_data import _try_load_latents

    good = tmp_path / "good.pth"
    torch.save({"frame_ids": np.arange(2, dtype=np.int64)}, good)
    raw = good.read_bytes()
    info, episode, _ = _latent_episode_layout(tmp_path, (raw, raw[:24]))
    assert _try_load_latents(
        tmp_path, info, episode, np.zeros((12, 7), np.float32), _eval_normalization()) is None


def test_try_load_latents_returns_none_for_lfs_pointer_and_empty(tmp_path):
    from script.lingbot_rl_data import _try_load_latents

    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    info, episode, _ = _latent_episode_layout(tmp_path, (pointer, b""))
    assert _try_load_latents(
        tmp_path, info, episode, np.zeros((12, 7), np.float32), _eval_normalization()) is None


def test_load_latent_names_corrupt_path(tmp_path):
    from script.lingbot_sft_data import load_latent

    path = tmp_path / "broken.pth"
    path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(RuntimeError, match="broken.pth"):
        load_latent(path)


def test_featurize_experts_rebuilds_corrupt_cache(tmp_path):
    from script.lingbot_rl_data import featurize_experts

    class StubPolicy:
        def reset(self):
            return None

        def extract_critic_state(self, batch):
            return torch.ones(1, 3072)

    cache = tmp_path / "expert_features.pt"
    cache.write_bytes(b"PK\x03\x04truncated")
    episode = {
        "actions": np.zeros((12, 7), np.float32),
        "task": "put the moka pot",
        "task_id": 8,
        "frames": [object()] * 12,
    }
    rows = featurize_experts(
        StubPolicy(), [episode], norm=_eval_normalization(), cache_path=cache)
    assert len(rows) == 1
    assert float(rows[0]["s"][0]) == 1.0


def test_load_episode_videos_returns_empty_on_truncated_mp4(tmp_path):
    from script.lingbot_rl_data import _load_episode_videos
    from script.lingbot_sft_config import CAMERAS

    info = {
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    for camera in CAMERAS:
        path = tmp_path / f"videos/chunk-000/{camera}/episode_000000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an mp4")
    assert _load_episode_videos(tmp_path, info, {"episode_index": 0, "length": 4}) == []


def test_missing_demo_cameras_do_not_fabricate_none_frames(tmp_path):
    from script.lingbot_rl_data import _load_episode_frames

    info = {
        "features": {},
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    frames = _load_episode_frames(
        tmp_path, info, {"episode_index": 0, "length": 4}, type("Table", (), {"num_rows": 4})())
    assert frames == []


def test_train_eval_fires_after_crossing_chunk_stride():
    from script.lingbot_rl_train import due_train_evals, train_eval_schedule

    schedule = train_eval_schedule(100_000, 25_000)
    assert schedule == [0, 25_000, 50_000, 75_000, 100_000]
    evaluated = set()
    assert due_train_evals(0, schedule, evaluated) == [0]
    evaluated.update([0])
    assert due_train_evals(24_999, schedule, evaluated) == []
    assert due_train_evals(25_012, schedule, evaluated) == [25_000]
    evaluated.update([25_000])
    assert due_train_evals(50_016, schedule, evaluated) == [50_000]
    assert train_eval_schedule(12, 25_000) == [0]


def test_host_array_converts_bfloat16():
    from script.lingbot_rl_train import _host_array

    values = _host_array(torch.tensor([1.0], dtype=torch.bfloat16))
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, [1.0], atol=1e-3)


def test_actor_critic_cast_bfloat16_policy_tensors_to_float32():
    model = DiceResidualModel(device="cpu")
    state = torch.zeros(2, STATE_DIM, dtype=torch.bfloat16)
    noise = torch.zeros(2, HORIZON, ACTION_DIM, dtype=torch.bfloat16)
    action = torch.zeros(2, HORIZON, ACTION_DIM, dtype=torch.bfloat16)
    residual = model.actor(state, noise)
    q = model.critic(state, action)
    applied = apply_residual(action, residual)
    assert residual.dtype == torch.float32
    assert q.dtype == torch.float32
    assert applied.dtype == torch.float32
    assert torch.isfinite(residual).all()
    assert torch.isfinite(q).all()


def test_replay_sample_converts_bf16_tensors_to_float32():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 0)
    row["s"] = torch.zeros(STATE_DIM, dtype=torch.bfloat16)
    row["z"] = torch.zeros(HORIZON, ACTION_DIM, dtype=torch.bfloat16)
    row["a_base"] = torch.zeros(HORIZON, ACTION_DIM, dtype=torch.bfloat16)
    row["a"] = torch.zeros(HORIZON, ACTION_DIM, dtype=torch.bfloat16)
    buf.add_online(row)
    buf.finalize_episode()
    batch = buf.sample(1, expert_ratio=0.0)
    for key in ("s", "z", "a_base", "a", "s_next", "a_next",
                "z_all", "a_base_all", "z_next_all", "a_base_next_all", "mc_return"):
        assert batch[key].dtype == torch.float32, key
        assert torch.isfinite(batch[key]).all()


def test_env_success_requires_explicit_boolean():
    from script.lingbot_rl_train import env_success

    assert env_success({"is_success": False}) is False
    assert env_success({"is_success": np.bool_(True)}) is True
    with pytest.raises(ValueError, match="is_success"):
        env_success({})
    with pytest.raises(ValueError, match="is_success"):
        env_success({"is_success": np.array([False])})


def test_action_batch_failure_does_not_hide_oom():
    from script.lingbot_rl_policy import reraise_action_batch_failure

    with pytest.raises(RuntimeError, match="out of memory"):
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as exc:
            reraise_action_batch_failure(exc)
    with pytest.raises(RuntimeError, match="action candidate batching failed"):
        try:
            raise RuntimeError("shape mismatch")
        except RuntimeError as exc:
            reraise_action_batch_failure(exc)


def test_expert_n_step_is_three_chunks():
    buf = ChunkReplay(capacity=32)
    for reward, done in ((0, 0), (0, 0), (1, 1)):
        buf.add_expert(_replay_row(reward, done, expert=True))
        if done:
            buf.finalize_episode()
    rows = buf.rows()
    assert [float(row["n_steps"]) for row in rows] == [3.0, 2.0, 1.0]
    assert float(rows[0]["reward"]) == pytest.approx(GAMMA ** 2)


def _resume_buffer():
    buf = ChunkReplay(capacity=32)
    buf.add_expert(_replay_row(1, 1, expert=True))
    buf.finalize_episode()
    return buf


def test_save_load_resume_roundtrips_rng_evaluated_and_wandb_id(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    model = DiceResidualModel(device="cpu")
    buffer = _resume_buffer()
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    save_resume(
        path, model, buffer, 0, 0, recipe,
        evaluated={0}, wandb_id="8otkv33a",
    )
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    restored = DiceResidualModel(device="cpu")
    buffer = ChunkReplay()
    env_steps, chunks, evaluated, wandb_id = load_resume(path, restored, buffer, recipe)
    assert env_steps == 0
    assert chunks == 0
    assert evaluated == {0}
    assert wandb_id == "8otkv33a"
    assert any(float(row["is_expert"]) == 1.0 for row in buffer.rows())
    got_t = torch.rand(4)
    got_n = np.random.rand()
    got_p = random.random()
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.testing.assert_close(got_t, torch.rand(4))
    assert got_n == np.random.rand()
    assert got_p == random.random()


def test_load_resume_restores_rng_from_non_byte_mapped_tensor(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    model = DiceResidualModel(device="cpu")
    torch.manual_seed(7)
    save_resume(
        path, model, _resume_buffer(), 12, 3, recipe,
        evaluated={0}, wandb_id="abc",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng"]["torch"] = payload["rng"]["torch"].to(dtype=torch.int32)
    torch.save(payload, path)
    torch.manual_seed(0)
    env_steps, chunks, evaluated, wandb_id = load_resume(
        path, DiceResidualModel(device="cpu"), ChunkReplay(), recipe)
    assert env_steps == 12
    assert chunks == 3
    assert evaluated == {0}
    assert wandb_id == "abc"
    got = torch.rand(3)
    torch.manual_seed(7)
    torch.testing.assert_close(got, torch.rand(3))


def test_load_resume_rejects_recipe_mismatch(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "latest.pt"
    save_resume(path, DiceResidualModel(device="cpu"), _resume_buffer(), 0, 0, recipe, evaluated={0})
    with pytest.raises(ValueError, match="recipe fingerprint"):
        load_resume(path, DiceResidualModel(device="cpu"), ChunkReplay(), {**recipe, "beta": 1.0})


def test_load_resume_infers_evaluated_from_train_eval_dirs(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    save_resume(path, DiceResidualModel(device="cpu"), _resume_buffer(), 0, 0, recipe, evaluated={0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("evaluated")
    torch.save(payload, path)
    (tmp_path / "train_eval" / "step_000000").mkdir(parents=True)
    _, _, evaluated, _ = load_resume(path, DiceResidualModel(device="cpu"), ChunkReplay(), recipe)
    assert evaluated == {0}


def test_load_resume_accepts_legacy_torch_numpy_rng(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    save_resume(path, DiceResidualModel(device="cpu"), _resume_buffer(), 0, 0, recipe, evaluated={0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng"] = {
        "torch": payload["rng"]["torch"].to(dtype=torch.int32),
        "numpy": payload["rng"]["numpy"],
    }
    payload.pop("evaluated")
    payload.pop("wandb_id")
    torch.save(payload, path)
    (tmp_path / "train_eval" / "step_000000").mkdir(parents=True)
    env_steps, _, evaluated, wandb_id = load_resume(
        path, DiceResidualModel(device="cpu"), ChunkReplay(), recipe)
    assert env_steps == 0
    assert evaluated == {0}
    assert wandb_id is None


def test_train_resume_reuses_wandb_id_and_skips_ingest(tmp_path, monkeypatch):
    import sys

    import script.lingbot_rl_train as train

    recipe = RLConfig().protocol()
    train.save_resume(
        tmp_path / "resume" / "latest.pt",
        DiceResidualModel(device="cpu"),
        _resume_buffer(),
        12,
        1,
        recipe,
        evaluated={0},
        wandb_id="run-from-checkpoint",
    )

    def boom(*args, **kwargs):
        raise AssertionError("resume must not re-featurize experts")

    wandb_calls = []

    class Run:
        id = "run-from-checkpoint"

        def log(self, *args, **kwargs):
            return None

        def finish(self, **kwargs):
            return None

        summary = {}

    class StubPolicy:
        def reset(self):
            self._executed_actions = None

        def extract_critic_state(self, batch):
            return torch.zeros(1, STATE_DIM)

        def decode_candidates(self, batch, k=4, **kwargs):
            z = torch.zeros(k, HORIZON, ACTION_DIM)
            a_base = torch.zeros(k, HORIZON, ACTION_DIM)
            return {
                "s": torch.zeros(1, STATE_DIM),
                "z": z,
                "a_base": a_base,
                "video_noise": torch.zeros(1),
                "first_chunk": True,
            }

        def commit_executed(self, chunk, first_chunk=False):
            self._executed_actions = chunk

        def select_action(self, batch):
            return torch.zeros(1, 7)

        def observe_env_step(self, batch):
            return None

    class StubEnv:
        def __init__(self, **kwargs):
            self.steps = 0

        def reset(self, seed=None):
            self.steps = 0
            zeros = np.zeros((128, 128, 3), np.uint8)
            return {"pixels": {"image": zeros, "image2": zeros}}, {}

        def step(self, action):
            self.steps += 1
            done = self.steps >= 12
            return self.reset()[0], 0.0, done, False, {"is_success": False}

        def close(self):
            return None

    monkeypatch.setattr(train, "featurize_experts", boom)
    monkeypatch.setattr(train, "load_manifest_episodes", boom)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: StubPolicy())
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        type("W", (), {
            "init": staticmethod(lambda **k: wandb_calls.append(k) or Run()),
            "finish": staticmethod(lambda **k: None),
        })(),
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "dataset_manifest.json").write_text("{}")
    prepared = {
        "tasks": [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50} for task_id in range(10)],
        "normalization": {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23},
        "checkpoint": str(ckpt),
        "model_path": None,
        "architecture": {},
        "source_run": "libero30-sft",
        "checkpoint_step": 600,
    }
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(__import__("json").dumps(prepared))
    monkeypatch.setattr(
        "script.lingbot_eval.read_checkpoint_metadata",
        lambda path: {"architecture": {}, "normalization": prepared["normalization"]},
    )
    train.train(
        output_dir=tmp_path, run_name="unit", resume=True, max_env_steps=24,
        prepared_path=prepared_path, dataset_root=tmp_path / "dataset",
    )
    assert wandb_calls[0]["id"] == "run-from-checkpoint"
    assert wandb_calls[0]["resume"] == "must"


def test_train_mocked_step_logs_and_saves_small_weights(tmp_path, monkeypatch):
    import sys

    import script.lingbot_rl_train as train
    from script.lingbot_rl_model import ACTION_DIM, HORIZON, STATE_DIM

    class StubPolicy:
        def reset(self):
            self._executed_actions = None

        def extract_critic_state(self, batch):
            return torch.zeros(1, STATE_DIM)

        def decode_candidates(self, batch, k=4, **kwargs):
            z = torch.zeros(k, HORIZON, ACTION_DIM)
            a_base = torch.zeros(k, HORIZON, ACTION_DIM)
            return {
                "s": torch.zeros(1, STATE_DIM),
                "z": z,
                "a_base": a_base,
                "video_noise": torch.zeros(1),
                "first_chunk": True,
            }

        def commit_executed(self, chunk, first_chunk=False):
            self._executed_actions = chunk

        def select_action(self, batch):
            return torch.zeros(1, 7)

        def observe_env_step(self, batch):
            return None

    class StubEnv:
        def __init__(self, **kwargs):
            self.steps = 0
            self.init_state_id = 0

        def reset(self, seed=None):
            self.steps = 0
            zeros = __import__("numpy").zeros((128, 128, 3), __import__("numpy").uint8)
            return {"pixels": {"image": zeros, "image2": zeros}}, {}

        def step(self, action):
            self.steps += 1
            done = self.steps >= 12
            return self.reset()[0], 0.0, done, False, {"is_success": False}

        def close(self):
            return None

    logs = []
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: StubPolicy())
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        type("W", (), {
            "init": staticmethod(lambda **k: type("R", (), {
                "log": logs.append,
                "summary": {},
                "finish": lambda **k: None,
            })()),
            "finish": staticmethod(lambda **k: None),
        })(),
    )
    train.train(output_dir=tmp_path, run_name="unit", max_env_steps=12, prepared_path=tmp_path / "prepared.json")
    assert (tmp_path / "residual.pt").is_file()
    payload = torch.load(tmp_path / "residual.pt", map_location="cpu", weights_only=True)
    assert set(payload) <= {"actor", "critic", "target_critic"}
    resume = torch.load(tmp_path / "resume" / "latest.pt", map_location="cpu", weights_only=False)
    assert "transformer" not in resume
    assert "env_steps" in resume
    assert "recipe" in resume
    assert "evaluated" in resume
    assert "python" in resume["rng"]
    assert "cuda" in resume["rng"]


def test_save_inference_checkpoints_keeps_latest_and_step_copy(tmp_path):
    from script.lingbot_rl_train import save_inference_checkpoints

    model = DiceResidualModel(device="cpu")
    with torch.no_grad():
        model.actor.net[0].bias.fill_(0.25)
    save_inference_checkpoints(tmp_path, 50291, model)
    latest = tmp_path / "residual.pt"
    step = tmp_path / "train_eval" / "step_050291" / "residual.pt"
    assert latest.is_file()
    assert step.is_file()
    loaded_latest = torch.load(latest, map_location="cpu", weights_only=True)
    loaded_step = torch.load(step, map_location="cpu", weights_only=True)
    assert set(loaded_latest) == {"actor", "critic", "target_critic"}
    assert set(loaded_step) == {"actor", "critic", "target_critic"}
    torch.testing.assert_close(loaded_latest["actor"]["net.0.bias"], loaded_step["actor"]["net.0.bias"])
    assert "transformer" not in loaded_latest
    assert "replay" not in loaded_latest


def test_train_eval_writes_step_residual_not_only_latest():
    import script.lingbot_rl_train as train

    source = Path(train.__file__).read_text()
    block = source.split("def maybe_eval", 1)[1].split("maybe_eval()", 1)[0]
    assert "save_inference_checkpoints" in block
    assert 'output_dir / "residual.pt"' not in block


def test_sharpening_metrics_not_residual_rms():
    from script.lingbot_rl_model import DiceResidualModel
    from script.lingbot_rl_train import sharpening_metrics

    model = DiceResidualModel(device="cpu")
    s = torch.zeros(2, 3072)
    a_base = torch.zeros(2, 8, 16, 30)
    a = a_base + 0.1
    metrics = sharpening_metrics(model, s, a_base, a)
    assert "delta_h" in metrics and "delta_v" in metrics
    assert "residual_rms" not in metrics


def test_modal_download_skips_resume_and_refuses_overwrite(tmp_path, monkeypatch):
    import script.lingbot_rl_modal as module

    calls = []

    def download(command, check):
        calls.append(command)
        dest = Path(command[-1]) / "libero30-dice-baseline"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "residual.pt").write_bytes(b"x")
        (dest / "summary.json").write_text("{}")

    monkeypatch.setattr(module.subprocess, "run", download)
    out = module.download_inference("libero30-dice-baseline", tmp_path / "result/lingbot-rl")
    assert Path(out).joinpath("residual.pt").is_file()
    joined = " ".join(str(part) for part in calls[0])
    assert "resume" not in joined
    assert "expert_features" not in joined
    with pytest.raises(FileExistsError):
        module.download_inference("libero30-dice-baseline", tmp_path / "result/lingbot-rl")


def test_download_stage_skips_prepare_and_gpu(tmp_path, monkeypatch):
    import script.lingbot_rl_modal as module

    called = []

    def record(name):
        def remote(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"{name} should not run for download")
        return type("Fn", (), {"remote": staticmethod(remote)})()

    monkeypatch.setattr(module, "prepare", record("prepare"))
    monkeypatch.setattr(module, "run_train", record("train"))
    monkeypatch.setattr(module, "run_eval", record("eval"))
    monkeypatch.setattr(module, "run_smoke", record("smoke"))
    monkeypatch.setattr(module, "download_inference", lambda run_name, download_dir: called.append((run_name, download_dir)) or "out")
    module.main(stage="download", run_name="libero30-dice-baseline", download_dir=str(tmp_path / "result/lingbot-rl"))
    assert called == [("libero30-dice-baseline", str(tmp_path / "result/lingbot-rl"))]


def test_modal_lock_and_help_do_not_print_secrets(capsys):
    import script.lingbot_rl_modal as module

    assert module.WANDB_SECRET_NAME == "dice-lingbot-wandb"
    assert module.HF_SECRET_NAME == "dice-lingbot-hf"
    assert module.RESULT_VOLUME == "dice-lingbot-rl-runs"
    assert module.SMOKE_ENV_STEPS == 32
    source = Path(module.__file__).read_text()
    train_block = source.split("def run_train", 1)[1].split("def run_smoke", 1)[0]
    smoke_block = source.split("def run_smoke", 1)[1].split("def run_eval", 1)[0]
    assert "--max-env-steps" not in train_block
    assert "--max-env-steps" in smoke_block
    assert "lingbot_eval.py" in module.FILES
    assert "lingbot_sft_data.py" in module.FILES


def test_residual_actor_initializes_to_zero_so_policy_starts_at_prior():
    torch.manual_seed(3)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(5, STATE_DIM)
    noise = torch.randn(5, HORIZON, ACTION_DIM)
    residual = model.actor(state, noise)
    torch.testing.assert_close(residual, torch.zeros(5, HORIZON, ACTION_DIM))
    a_base = mask_unused_dof(torch.randn(5, HORIZON, ACTION_DIM))
    torch.testing.assert_close(apply_residual(a_base, residual), a_base)


def test_finalize_stores_monte_carlo_return_and_next_candidates():
    buf = ChunkReplay(capacity=32)
    for step, (reward, done) in enumerate(((0, 0), (0, 0), (1, 1))):
        row = _replay_row(reward, done)
        row["z_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), float(step), np.float32)
        row["a_base_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 10.0 + step, np.float32)
        buf.add_online(row)
    buf.finalize_episode()
    rows = buf.rows()
    assert [float(row["mc_return"]) for row in rows] == pytest.approx([GAMMA ** 2, GAMMA, 1.0])
    np.testing.assert_array_equal(
        rows[0]["z_next_all"], np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 2.0, np.float32))
    np.testing.assert_array_equal(
        rows[0]["a_base_next_all"], np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 12.0, np.float32))


def test_store_defaults_repeat_single_candidate_for_experts():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 1, expert=True)
    row["z"] = np.full((HORIZON, ACTION_DIM), 7.0, np.float32)
    row["a_base"] = np.full((HORIZON, ACTION_DIM), 8.0, np.float32)
    buf.add_expert(row)
    buf.finalize_episode()
    stored = buf.rows()[0]
    assert stored["z_all"].shape == (K_CANDIDATES, HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(stored["z_all"][3], row["z"])
    np.testing.assert_array_equal(stored["a_base_all"][0], stored["a_base_all"][3])


def test_open_episode_view_has_zero_mc_return_and_successor_candidates():
    buf = ChunkReplay(capacity=32)
    first = _replay_row(0, 0)
    second = _replay_row(0, 0)
    second["z_all"] = np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 5.0, np.float32)
    buf.add_online(first)
    buf.add_online(second)
    batch = buf.sample(1, expert_ratio=0.0)
    assert batch["mc_return"].shape == (1, 1)
    assert float(batch["mc_return"][0]) == 0.0
    assert batch["z_all"].shape == (1, K_CANDIDATES, HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(
        batch["z_next_all"][0].numpy(), np.full((K_CANDIDATES, HORIZON, ACTION_DIM), 5.0, np.float32))


def test_store_rejects_wrong_candidate_count():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 0)
    row["z_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    row["a_base_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    with pytest.raises(ValueError, match="candidate"):
        buf.add_online(row)
