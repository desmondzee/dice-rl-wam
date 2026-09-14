from dataclasses import replace

import pytest

from script.lingbot_rl_config import RLConfig


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


@pytest.mark.parametrize("changes", [
    {"video_steps": 3},
    {"action_steps": 10},
    {"video_exec_step": 12},
    {"checkpoint_step": 400},
    {"source_run": "other-run"},
    {"k_candidates": 16},
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
