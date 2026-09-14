from dataclasses import replace

import numpy as np
import pytest
import torch

from script.lingbot_eval import decode_action
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_model import (
    ACTION_DIM, ADAM_LR, BETA, ENSEMBLE, EPSILON, GAMMA, HIDDEN, HORIZON,
    N_STEP_CHUNKS, STATE_DIM, TAU, USED_DOF, UTD, DiceResidualModel, apply_residual,
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
    assert kwargs["save_predicted_video"] is False
    assert kwargs["device"] == "cuda"


def test_policy_class_exposes_collection_api():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    for name in ("decode_candidates", "extract_critic_state", "commit_executed", "select_action"):
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

        def commit_executed(self, chunk):
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
