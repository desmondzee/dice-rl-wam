import importlib.util
import io
import json
import os
import random
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from script.lingbot_eval import decode_action
from script.lingbot_eval_config import EvalConfig, episode_plan
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig, config_from_dict
from script.lingbot_rl_model import (
    ACTION_DIM, ACTOR_EVERY, ADAM_LR, BATCH, BETA, ENSEMBLE, EPSILON, GAMMA, GRADIENT_STEPS,
    HIDDEN, HORIZON, K_CANDIDATES, LR_MIN, MAX_GRAD_NORM, N_STEP_CHUNKS, STATE_DIM, TAU,
    UPDATE_EVERY_CHUNKS, USED_DOF, WEIGHT_DECAY, DiceResidualModel, apply_residual,
    cosine_restart_lr, mask_unused_dof,
)
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, mlp_to_model, model_to_mlp,
    pool_critic_state, slice_env_actions,
)


def test_config_pins_released_libero_sampler_and_step_600():
    cfg = RLConfig().validate()
    proto = cfg.protocol()
    assert proto["sampler"] == "released_lingbot_libero_defaults"
    assert cfg.checkpoint_step == 600
    assert cfg.source_run == "libero30-sft"
    assert cfg.wandb_project == "dice-lingbot-va-rl"
    assert cfg.k_candidates == 4
    assert cfg.task_ids == (0,)
    assert cfg.default_run_name == "dice-t0"
    assert cfg.online_env_steps == 660_000
    assert proto["action_normalization_epsilon"] == 1e-6
    assert proto["noisy_history"] is False
    assert proto["environment_batch_size"] == 1
    assert proto["max_policy_steps"] == 520
    assert proto["camera_orientation"] == "vertical_flip_only_native_lingbot"
    assert proto["residual_input"] == "z"
    assert proto["lr_schedule"] == "cosine_restarts_1000_warmup_10_min_1e-6"
    assert proto["rlpd_start"] == 0.9
    assert proto["rlpd_end"] == 0.1
    assert proto["rlpd_steps"] == 208_000
    assert proto["replay_capacity"] == 100_000
    assert proto["task_ids"] == [0]
    assert proto["train_init_states"] == "procedural"
    assert proto["selection"] == "max_q_min"
    assert proto["selection_warmup_steps"] == 64_000
    assert proto["bc_filter_anchor"] == "q_stored_minus_mc_return"
    assert proto["bc_filter_warmup_steps"] == 128_000
    assert proto["expert_rows_keep_bc"] is True
    assert proto["actor_q_normalization"] == "mean_abs_q_online"
    assert proto["checkpoint_every"] == 80_000
    assert proto["train_eval_episodes"] == 10
    assert proto["checkpoint_step"] == 600
    assert proto["source_run"] == "libero30-sft"
    assert proto["version"] == 3
    assert proto["multi_sample_candidates"] == 4
    assert proto["actor_final_init"] == "zeros"
    assert proto["eval_best_of_n"] == 4


@pytest.mark.parametrize("changes", [
    {"video_steps": 0}, {"video_steps": 51}, {"action_steps": 0}, {"action_steps": 51}, {"video_steps": 6, "video_exec_step": 10},
    {"video_exec_step": 0}, {"video_exec_step": 21}, {"checkpoint_step": 400},
    {"source_run": "other-run"}, {"k_candidates": 16}, {"task_ids": ()}, {"task_ids": (0, 0)},
    {"task_ids": (11,)}, {"online_env_steps": 0}, {"checkpoint_every": 0}, {"train_eval_episodes": 0},
])
def test_config_rejects_sampler_and_recipe_drift(changes):
    with pytest.raises(ValueError):
        replace(RLConfig(), **changes).validate()


def test_config_epsilon_and_budget_enter_the_fingerprint_and_model():
    cfg = config_from_dict({"epsilon": -0.3, "online_env_steps": 480_000, "task_ids": [9]})
    assert cfg.protocol()["epsilon"] == -0.3 and cfg.protocol()["online_env_steps"] == 480_000
    assert cfg.protocol() != RLConfig().protocol()
    assert cfg.checkpoint_schedule()[-1] == 480_000 and cfg.train_eval_schedule() == [0, 80_000, 240_000, 400_000]
    assert DiceResidualModel(device="cpu", epsilon=cfg.epsilon).epsilon == -0.3
    with pytest.raises(ValueError, match="epsilon"):
        config_from_dict({"epsilon": 0.1})


def test_config_video_exec_step_enters_the_fingerprint():
    cfg = config_from_dict({"video_exec_step": 10})
    assert cfg.protocol()["video_exec_step"] == 10
    assert cfg.protocol() != RLConfig().protocol()
    strip = lambda protocol: {key: value for key, value in protocol.items() if key != "video_exec_step"}
    assert strip(cfg.protocol()) == strip(RLConfig().protocol())


def test_config_sampler_steps_enter_the_fingerprint():
    cfg = config_from_dict({"video_steps": 6, "action_steps": 8, "video_exec_step": 6})
    proto = cfg.protocol()
    assert (proto["video_steps"], proto["action_steps"], proto["video_exec_step"]) == (6, 8, 6)
    keys = ("video_steps", "action_steps", "video_exec_step")
    strip = lambda protocol: {key: value for key, value in protocol.items() if key not in keys}
    assert proto != RLConfig().protocol() and strip(proto) == strip(RLConfig().protocol())
    assert config_from_dict({"video_steps": 10, "video_exec_step": 10}).video_exec_step == 10


def test_config_plan_fields_enter_the_fingerprint():
    base = RLConfig().protocol()
    assert base["plan_candidates"] == 1 and base["plan_critic"] == "none" and base["plan_pool"] == "none"
    plan = RLConfig(plan_candidates=4).protocol()
    assert plan["plan_critic"] == "regression_max_k_min_q" and plan["plan_pool"] == "critic_pool_of_denoised_chunk"
    assert plan != base
    with pytest.raises(ValueError):
        RLConfig(plan_candidates=9).validate()


def test_config_feature_fields_enter_the_fingerprint_for_both_runs():
    from script.lingbot_rl_config import plus_split, plus_train_variants
    base = RLConfig().protocol()
    assert plus_split("robot_init")[1] == tuple(range(642, 682, 3)) and plus_split("layout")[1] == tuple(range(1984, 2021, 3))
    train = plus_train_variants()
    assert (len(train["robot_init"]), len(train["layout"])) == (28, 26) and 642 not in train["robot_init"]
    run_a = RLConfig(task_ids=(9,), plan_candidates=4, epsilon=-0.3, residual_input="base", proprio=True, plus_train=train).protocol()
    run_b = RLConfig(task_ids=(9,), plan_candidates=4, epsilon=-0.3, residual_input="base", proprio=True, state_features="struct",
                     plus_train=train).protocol()
    assert run_a["collection_mix"] == "thirds_standard_robot_init_layout" and run_a["plus_train"]["layout"][:3] == [1982, 1983, 1985]
    assert len({json.dumps(protocol, sort_keys=True) for protocol in (base, run_a, run_b)}) == 3
    restored = config_from_dict(json.loads(json.dumps(RLConfig(task_ids=(9,), plus_train=train).to_dict())))
    assert restored.plus_train == train
    for bad in (dict(state_features="tokens"), dict(proprio=1), dict(task_ids=(9,), plus_train={"view": (1,)}),
                dict(task_ids=(9,), plus_train={"robot_init": (640,), "layout": (1984,)}), dict(plus_train=train)):
        with pytest.raises(ValueError):
            RLConfig(**bad).validate()


def test_config_bounded_recipe_fields_enter_the_fingerprint_only_when_set():
    base = RLConfig().validate().protocol()
    config = RLConfig(task_ids=(9,), residual_input="base", proprio=True, state_features="struct", residual_bound=0.3, bc_weight=0.0,
                      actor_warmup_episodes=50, residual_ramp_steps=50_000, shaping=True, privileged_critic=True, standardize_features=True,
                      update_every_chunks=1, rlpd_start=0.5, rlpd_end=0.5, diagnostic_label_every=10,
                      procedural_robot_init={"cap_start": 0.3}).validate()
    protocol = config.protocol()
    assert len(base) == 78 and "residual_bound" not in base and base["beta"] == 100.0 and base["update_every_chunks"] == 4
    assert protocol["residual_bound"] == 0.3 and protocol["beta"] == 0.0 and protocol["bc_filter"] == "off"
    assert (protocol["actor_warmup_episodes"], protocol["residual_ramp_steps"]) == (50, 50_000)
    assert protocol["reward"] == "success_plus_potential_shaping" and protocol["critic_input"] == "features_proprio_object_state"
    assert protocol["feature_standardization"] == "expert_global_struct_per_dim_object_state"
    assert protocol["update_every_chunks"] == 1 and protocol["diagnostic_label_every"] == 10
    assert protocol["procedural_robot_init"] == {"magnitude_lo": 0.1, "cap_start": 0.3, "cap_end": 0.5, "trigger_success": 0.5, "trigger_window": 20}
    assert protocol["collection_mix"] == "thirds_standard_procedural_robot_init_layout"
    assert config.procedural()["cap_end"] == 0.5 and RLConfig().procedural() is None


@pytest.mark.parametrize("changes", [
    {"residual_bound": -0.1}, {"bc_weight": -1.0}, {"actor_warmup_episodes": -1}, {"residual_ramp_steps": -1},
    {"update_every_chunks": 0}, {"diagnostic_label_every": -1}, {"procedural_robot_init": {"cap_start": 0.6}},
    {"procedural_robot_init": {"bogus": 1}}, {"shaping": True, "proprio": False}, {"privileged_critic": True, "proprio": False},
    {"diagnostic_label_every": 10, "proprio": False}, {"procedural_robot_init": {"cap_start": 0.3}, "task_ids": (0,)},
])
def test_config_rejects_invalid_bounded_recipe_values(changes):
    with pytest.raises(ValueError):
        RLConfig(**{"task_ids": (9,), "proprio": True, **changes}).validate()


def test_default_protocol_is_unchanged_by_the_feature_fields():
    base = RLConfig().protocol()
    assert not {"state_features", "proprio", "plus_train", "collection_mix"} & set(base) and len(base) == 78
    assert set(RLConfig(proprio=True).protocol()) == set(base) | {"state_features", "proprio", "plus_train", "collection_mix"}


def test_config_plan_target_enters_the_fingerprint_only_for_rollout():
    base = RLConfig().protocol()
    assert "plan_target" not in base and len(base) == 78
    rollout = RLConfig(task_ids=(9,), plan_candidates=4, proprio=True, plan_target="rollout")
    protocol = rollout.protocol()
    assert protocol["plan_target"] == "rollout" and protocol["plan_critic"] == "regression_mean_k_rollout_progress"
    assert set(protocol) == set(RLConfig(task_ids=(9,), plan_candidates=4, proprio=True).protocol()) | {"plan_target"}
    assert config_from_dict(json.loads(json.dumps(rollout.to_dict()))).plan_target == "rollout"
    for bad in (dict(plan_target="mc"), dict(task_ids=(9,), proprio=True, plan_target="rollout"),
                dict(task_ids=(9,), plan_candidates=4, plan_target="rollout")):
        with pytest.raises(ValueError):
            RLConfig(**bad).validate()


def test_progress_rewards_approach_inside_and_closing_and_calibrates_the_door_from_the_rest_limit():
    from script.lingbot_rl_progress import INSIDE_RADIUS, calibrate, phi, progress
    layout = {"bodies": ["microwave_1", "porcelain_mug_1", "white_yellow_mug_1"], "joints": ["microwave_1_microjoint"], "body_quaternion": "wxyz"}

    def state(mug, door):
        poses = np.zeros((3, 7), np.float32)
        poses[2, :3] = mug
        return np.concatenate([poses.reshape(-1), [door]]).astype(np.float32)

    model = SimpleNamespace(jnt_range=np.array([[-2.094, 0.0]]), joint_name2id=lambda name: {"microwave_1_microjoint": 0}[name])
    inner = SimpleNamespace(sim=SimpleNamespace(model=model))
    calibration = calibrate(9, inner, layout, state([0.3, 0.0, 0.0], -2.0))
    assert (calibration["door_closed"], calibration["door_open"], calibration["door_initial"]) == (0.0, -2.094, -2.0)
    assert calibration["mug"] == "white_yellow_mug_1" and calibration["phi_initial"] == pytest.approx(-0.3)
    far, near = state([0.3, 0.0, 0.0], -2.094), state([0.05, 0.0, 0.0], -2.094)
    assert phi(calibration, layout, far, False) == pytest.approx(-0.3) and phi(calibration, layout, near, False) == pytest.approx(0.95)
    assert phi(calibration, layout, state([0.05, 0.0, 0.0], 0.0), False) == pytest.approx(1.95)
    assert phi(calibration, layout, state([0.05, 0.0, 0.0], 0.0), True) == pytest.approx(3.95)
    assert progress(calibration, layout, far, near, False) == pytest.approx(1.25)
    assert progress(calibration, layout, far, state([0.3, 0.0, 0.0], -1.0), False) == pytest.approx(0.0)
    assert INSIDE_RADIUS == 0.12
    with pytest.raises(ValueError):
        calibrate(0, inner, layout, far)
    with pytest.raises(ValueError):
        calibrate(9, inner, {**layout, "bodies": ["microwave_1", "porcelain_mug_1"]}, far)


def test_evaluate_records_video_exec_step_and_passes_it_to_the_policy(tmp_path, monkeypatch):
    import script.lingbot_eval as eval_module
    import script.lingbot_rl_train as train

    class StubEnv:
        def __init__(self, **env_kwargs):
            pass

        def close(self):
            return None

    tasks = [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50} for task_id in range(10)]
    norm = {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}
    loaded = []
    monkeypatch.setattr(train, "load_residual_policy", lambda *a: loaded.append(a) or _StubPolicy())
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setattr(eval_module, "read_checkpoint_metadata", lambda path: {"architecture": {}, "normalization": norm})
    monkeypatch.setattr(eval_module, "describe_suite", lambda path: (object(), tasks))
    monkeypatch.setattr(eval_module, "run_episode", lambda policy, env, entry, instruction, norm, device, **k: {
        **entry, "success": True, "seconds": 1.0, "policy_steps": 3})
    torch.save(DiceResidualModel(device="cpu").inference_state_dict(), tmp_path / "residual.pt")
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"checkpoint": "ckpt", "model_path": "model", "assets_path": "assets",
                                    "source_run": "libero30-sft", "checkpoint_step": 600}))
    summary = train.evaluate(config_from_dict({"task_ids": [9], "video_exec_step": 10, "video_steps": 12, "action_steps": 8}),
                             prepared, tmp_path / "out", residual_path=tmp_path / "residual.pt", eval_candidates=2, shard=1, shards=50)
    assert summary["complete"] and summary["completed_episodes"] == 2
    assert loaded[0][3:] == (10, 12, 8)
    settings = json.loads((tmp_path / "out" / "settings" / "shard_1.json").read_text())
    assert settings["config"]["video_exec_step"] == 10 and settings["config"]["protocol"]["video_exec_step"] == 10
    assert (settings["config"]["video_steps"], settings["config"]["action_steps"]) == (12, 8)
    assert (settings["config"]["protocol"]["video_steps"], settings["config"]["protocol"]["action_steps"]) == (12, 8)
    assert settings["rl"]["video_exec_step"] == 10 and settings["eval_candidates"] == 2
    assert (settings["rl"]["video_steps"], settings["rl"]["action_steps"]) == (12, 8)
    assert [entry["episode_index"] for entry in settings["episode_plan"]] == [1, 51]


def test_evaluate_reconstructs_the_input_layout_from_the_residual_checkpoint(tmp_path, monkeypatch):
    import script.lingbot_eval as eval_module
    import script.lingbot_rl_train as train
    envs = []

    class StubEnv:
        def __init__(self, **env_kwargs):
            envs.append(env_kwargs)

        def close(self):
            return None

    tasks = [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50} for task_id in range(10)]
    stub = _StubPolicy()
    monkeypatch.setattr(train, "load_residual_policy", lambda *a: stub)
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setattr(eval_module, "read_checkpoint_metadata", lambda path: {"architecture": {}, "normalization": _eval_normalization()})
    monkeypatch.setattr(eval_module, "describe_suite", lambda path: (object(), tasks))
    monkeypatch.setattr(train, "run_rl_episode", lambda policy, env, entry, instruction, norm, device, proprio, **k: {
        **entry, "success": True, "seconds": 1.0, "policy_steps": 3, "proprio": proprio})
    torch.save(DiceResidualModel(device="cpu", state_features="struct", proprio=True).inference_state_dict(), tmp_path / "residual.pt")
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"checkpoint": "ckpt", "model_path": "model", "assets_path": "assets",
                                    "source_run": "libero30-sft", "checkpoint_step": 600}))
    summary = train.evaluate(config_from_dict({"task_ids": [9]}), prepared, tmp_path / "out",
                             residual_path=tmp_path / "residual.pt", shard=1, shards=50)
    assert summary["complete"] and (stub.residual_model.features, stub.residual_model.proprio) == ("struct", True)
    assert envs[0]["obs_type"] == "pixels_agent_pos"
    settings = json.loads((tmp_path / "out" / "settings" / "shard_1.json").read_text())
    assert settings["layout"] == {"state_features": "struct", "proprio": True} and settings["rl"]["state_features"] == "struct"
    episode = json.loads((tmp_path / "out" / "episodes" / "task_09" / "episode_001.json").read_text())
    assert episode["proprio"] is True
    pool_payload = DiceResidualModel(device="cpu").inference_state_dict()
    torch.save(pool_payload, tmp_path / "pool.pt")
    with pytest.raises(RuntimeError, match="layout mismatch"):
        train.evaluate(config_from_dict({"task_ids": [9], "plan_candidates": 2}), prepared, tmp_path / "bad",
                       residual_path=tmp_path / "pool.pt", shard=1, shards=50)
    pool_payload.pop("layout")
    torch.save(pool_payload, tmp_path / "legacy.pt")
    train.evaluate(config_from_dict({"task_ids": [9]}), prepared, tmp_path / "legacy", residual_path=tmp_path / "legacy.pt", shard=1, shards=50)
    assert (stub.residual_model.features, stub.residual_model.proprio) == ("pool", False) and envs[-1]["obs_type"] == "pixels"


def test_config_from_dict_restores_task_ids_and_validates():
    cfg = RLConfig(task_ids=(0, 3)).validate()
    payload = cfg.to_dict()
    assert payload["task_ids"] == [0, 3]
    assert config_from_dict(payload) == cfg
    assert config_from_dict({}) == RLConfig()
    assert cfg.default_run_name == "dice-t0-3"
    with pytest.raises(ValueError):
        config_from_dict({"task_ids": [11]})


def test_checkpoint_and_train_eval_schedules():
    cfg = RLConfig().validate()
    assert cfg.checkpoint_schedule() == [step * 80_000 for step in range(9)] + [660_000]
    assert cfg.train_eval_schedule() == [0, 80_000, 240_000, 400_000, 560_000]


def test_due_points_fire_once_per_schedule_entry():
    from script.lingbot_rl_train import due_points

    schedule = RLConfig().checkpoint_schedule()
    done = set()
    assert due_points(0, schedule, done) == [0]
    done.add(0)
    assert due_points(79_999, schedule, done) == []
    assert due_points(80_012, schedule, done) == [80_000]
    done.add(80_000)
    assert due_points(240_000, schedule, done) == [160_000, 240_000]


def test_cosine_restart_lr_warms_up_then_restarts():
    assert cosine_restart_lr(0) == cosine_restart_lr(1000) == pytest.approx(LR_MIN / ADAM_LR)
    assert cosine_restart_lr(10) == pytest.approx(1.0)
    assert cosine_restart_lr(505) < cosine_restart_lr(10)


def test_rlpd_ratio_anneals_over_env_steps():
    cfg = RLConfig().validate()
    assert cfg.rlpd_expert_ratio(0) == pytest.approx(0.9)
    assert cfg.rlpd_expert_ratio(104_000) == pytest.approx(0.5)
    assert cfg.rlpd_expert_ratio(208_000) == pytest.approx(0.1)
    assert cfg.rlpd_expert_ratio(500_000) == pytest.approx(0.1)


def test_heldout_plan_covers_fifty_states_twice_from_state_zero():
    cfg = EvalConfig(stage="heldout").validate()
    assert cfg.episodes_per_task == 100
    assert cfg.initial_state_offset == 0
    plan = episode_plan(cfg, 0, 50)
    assert [entry["init_state_id"] for entry in plan] == list(range(50)) * 2
    assert plan[0]["seed"] != plan[50]["seed"]


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
    assert WEIGHT_DECAY == proto["weight_decay"] == 1e-5
    assert MAX_GRAD_NORM == proto["max_grad_norm"] == 1.0
    assert GRADIENT_STEPS == proto["gradient_steps"] == 10
    assert UPDATE_EVERY_CHUNKS == proto["update_every_chunks"] == 4
    assert ACTOR_EVERY == proto["actor_every"] == 2
    assert BATCH == proto["batch_size"] == proto["min_online_rows"] == 256
    model = DiceResidualModel(device="cpu")
    assert len(model.critic.heads) == 10
    assert model.actor_opt.param_groups[0]["weight_decay"] == WEIGHT_DECAY


def test_residual_mask_zeros_unused_dof():
    torch.manual_seed(0)
    a_base = torch.ones(2, HORIZON, ACTION_DIM)
    residual = torch.randn(2, HORIZON, ACTION_DIM)
    a = apply_residual(a_base, residual)
    assert a.shape == (2, HORIZON, ACTION_DIM)
    assert torch.count_nonzero(a[:, :, USED_DOF:]) == 0
    torch.testing.assert_close(a[:, :, :USED_DOF], a_base[:, :, :USED_DOF] + residual[:, :, :USED_DOF])


def test_actor_critic_one_step_update_on_random_tensors():
    torch.manual_seed(1)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(8, STATE_DIM)
    z_all = torch.randn(8, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_all = mask_unused_dof(torch.randn(8, K_CANDIDATES, HORIZON, ACTION_DIM))
    is_expert = torch.zeros(8, 1)
    is_expert[:4] = 1
    target = torch.randn(8, 1)
    before = {k: v.detach().clone() for k, v in model.actor.named_parameters()}
    critic_info = model.update_critic(state, a_base_all[:, 0], target)
    actor_info = model.update_actor(
        state, state, a_base_all[:, 0], z_all, a_base_all, is_expert, torch.zeros(8, 1), True)
    assert torch.isfinite(torch.tensor(critic_info["critic_loss"]))
    assert set(critic_info) == {"critic_loss", "critic_grad_norm", "critic_q_mean"}
    assert torch.isfinite(torch.tensor(actor_info["actor_loss"]))
    assert set(actor_info) == {
        "actor_loss", "actor_q_loss", "actor_bc_loss", "actor_grad_norm", "residual_rms",
        "q_mean", "q_min", "pretrained_q_mean", "q_advantage", "better_than_base_rate",
        "bc_filter_rate", "q_overestimation", "q_overestimation_online",
    }
    assert 0.0 <= actor_info["bc_filter_rate"] <= 1.0
    assert any(not torch.equal(before[k], v) for k, v in model.actor.named_parameters())


def test_actor_q_term_is_normalised_by_online_mean_abs_q():
    torch.manual_seed(2)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(4, STATE_DIM)
    z_all = torch.zeros(4, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_all = mask_unused_dof(torch.randn(4, K_CANDIDATES, HORIZON, ACTION_DIM))
    is_expert = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    online = (is_expert == 0).float()
    with torch.no_grad():
        state_k = state.unsqueeze(1).expand(4, K_CANDIDATES, STATE_DIM).reshape(-1, STATE_DIM)
        q = model.critic(state_k, a_base_all.reshape(-1, HORIZON, ACTION_DIM)).reshape(4, K_CANDIDATES)
    q_scale = float((q.abs() * online).sum() / (online.sum() * K_CANDIDATES))
    expected = float(-(q.mean(dim=1) * online.squeeze(1)).sum() / 4) / q_scale
    info = model.update_actor(
        state, state, a_base_all[:, 0], z_all, a_base_all, is_expert, torch.zeros(4, 1), False)
    assert info["actor_q_loss"] == pytest.approx(expected, rel=1e-4)
    assert info["q_mean"] == pytest.approx(float(q.mean()), rel=1e-4)
    assert info["q_advantage"] == pytest.approx(0.0, abs=1e-6)


class _SumCritic(torch.nn.Module):
    def forward(self, state, action, return_all=False):
        q = action.float().reshape(action.shape[0], -1).sum(dim=-1, keepdim=True)
        return [q] if return_all else q


def test_bc_filter_drops_better_overestimated_online_rows_only():
    model = DiceResidualModel(device="cpu")
    model.critic = _SumCritic()
    state = torch.zeros(3, STATE_DIM)
    z_all = torch.zeros(3, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_all = torch.zeros(3, K_CANDIDATES, HORIZON, ACTION_DIM)
    stored = torch.zeros(3, HORIZON, ACTION_DIM)
    is_expert = torch.tensor([[0.0], [0.0], [1.0]])
    mc_return = torch.tensor([[1.0], [0.0], [1.0]])
    rates = []
    for filter_active in (False, True):
        with torch.no_grad():
            model.actor.net[-1].bias.fill_(1.0)
        info = model.update_actor(state, state, stored, z_all, a_base_all, is_expert, mc_return, filter_active)
        rates.append(info["bc_filter_rate"])
    assert info["residual_rms"] == pytest.approx(1.0, rel=1e-3)
    assert info["actor_bc_loss"] == pytest.approx(rates[1], rel=1e-3)
    assert info["better_than_base_rate"] == 1.0
    assert info["q_overestimation"] == pytest.approx(-2.0 / 3.0)
    assert info["q_overestimation_online"] == pytest.approx(-0.5)
    assert rates == [1.0, pytest.approx(2.0 / 3.0)]


def test_plan_critic_regresses_onto_max_candidate_min_q_and_leaves_q_and_actor_alone():
    torch.manual_seed(0)
    single = DiceResidualModel(device="cpu")
    model = DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2)
    assert single.plan_critic is None and single.critic.heads[0][0].in_features == STATE_DIM + HORIZON * ACTION_DIM
    assert model.critic.heads[0][0].in_features == 2 * STATE_DIM + HORIZON * ACTION_DIM
    assert set(single.inference_state_dict()) == {"actor", "critic", "target_critic", "layout"}
    assert {"plan_critic", "plan_opt", "plan_lr"} <= set(model.resume_state_dict())
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state, pools = torch.randn(5, STATE_DIM), torch.randn(5, 2, STATE_DIM)
    z_all = torch.randn(5, 4, HORIZON, ACTION_DIM)
    a_base = mask_unused_dof(torch.randn(5, 2, 4, HORIZON, ACTION_DIM))
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    actor_before = [p.detach().clone() for p in model.actor.parameters()]
    with torch.no_grad():
        flat_state = state[:, None, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_pool = pools[:, :, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_z = z_all[:, None].expand(5, 2, 4, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
        flat_base = a_base.reshape(-1, HORIZON, ACTION_DIM)
        action = apply_residual(flat_base, model.actor(flat_state, flat_z, flat_base))
        expected = model.critic(model.critic_state(flat_state, flat_pool), action).reshape(5, 2, 4).max(dim=2).values
    losses = [model.update_plan_critic(state, pools, z_all, a_base)["plan_critic_loss"] for _ in range(30)]
    info = model.update_plan_critic(state, pools, z_all, a_base)
    assert losses[-1] < losses[0] and abs(info["plan_target_mean"] - float(expected.mean())) < 1e-5
    assert all(torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actor_before, model.actor.parameters()))
    values = model.plan_critic(state[:1].expand(2, -1), pools[0])
    assert model.rank_plans(state[:1], pools[0]) == int(values.reshape(-1).argmax())


def test_plan_critic_trains_on_rollout_labels_and_ranks_by_the_mean_readout():
    torch.manual_seed(0)
    model = DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2, plan_target="rollout")
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state, pools = torch.randn(5, STATE_DIM), torch.randn(5, 2, STATE_DIM)
    z_all = torch.randn(5, 4, HORIZON, ACTION_DIM)
    a_base = mask_unused_dof(torch.randn(5, 2, 4, HORIZON, ACTION_DIM))
    labels = torch.randn(5, 2, 4)
    with torch.no_grad():
        flat_state = state[:, None, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_pool = pools[:, :, None].expand(5, 2, 4, STATE_DIM).reshape(-1, STATE_DIM)
        flat_z = z_all[:, None].expand(5, 2, 4, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
        flat_base = a_base.reshape(-1, HORIZON, ACTION_DIM)
        action = apply_residual(flat_base, model.actor(flat_state, flat_z, flat_base))
        q_target = model.critic(model.critic_state(flat_state, flat_pool), action).reshape(5, 2, 4).max(dim=2).values
    with pytest.raises(ValueError):
        model.update_plan_critic(state, pools, z_all, a_base)
    losses = [model.update_plan_critic(state, pools, z_all, a_base, labels)["plan_critic_loss"] for _ in range(30)]
    info = model.update_plan_critic(state, pools, z_all, a_base, labels)
    assert losses[-1] < losses[0] and info["plan_target_mean"] == pytest.approx(float(labels.mean()), abs=1e-5)
    assert info["plan_target_q_mean"] == pytest.approx(float(q_target.mean()), abs=1e-5)
    assert info["plan_label_vs_q_agreement"] == pytest.approx(float((labels.mean(-1).argmax(1) == q_target.argmax(1)).float().mean()))
    online = model.update_plan_critic(state, pools, z_all, a_base, labels, torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]]))
    assert online["plan_target_mean"] == pytest.approx(float(labels[::2].mean()), abs=1e-5) and online["plan_label_vs_q_agreement"] == \
        pytest.approx(float((labels[::2].mean(-1).argmax(1) == q_target[::2].argmax(1)).float().mean()))
    heads = torch.stack(model.plan_critic(state[:1].expand(2, -1), pools[0], return_all=True))
    assert model.rank_plans(state[:1], pools[0]) == int(heads.mean(dim=0).reshape(-1).argmax())
    payload = model.inference_state_dict()
    assert payload["layout"]["plan_target"] == "rollout" and DiceResidualModel.layout_of(payload)["plan_target"] == "rollout"
    assert "plan_target" not in DiceResidualModel(device="cpu", plan_candidates=2).inference_state_dict()["layout"]
    with pytest.raises(ValueError, match="layout mismatch"):
        DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2).load_inference_state_dict(payload)


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


def _online_buffer():
    buf = ChunkReplay(capacity=32)
    for reward, done in ((0, 0), (0, 0), (1, 1)):
        buf.add_online(_replay_row(reward, done))
    buf.finalize_episode()
    return buf


def test_model_input_layout_follows_state_features_and_proprio():
    from script.lingbot_rl_model import PROPRIO_DIM
    struct = DiceResidualModel(device="cpu", state_features="struct", proprio=True)
    mixed = DiceResidualModel(device="cpu", plan_candidates=2, proprio=True)
    assert (struct.state_dim, struct.plan_dim) == (2 * STATE_DIM + PROPRIO_DIM, 0)
    assert (mixed.state_dim, mixed.plan_dim) == (STATE_DIM + PROPRIO_DIM, STATE_DIM)
    assert struct.actor.net[0].in_features == struct.state_dim + HORIZON * ACTION_DIM and struct.plan_critic is None
    assert struct.critic.heads[0][0].in_features == struct.state_dim + HORIZON * ACTION_DIM
    assert mixed.critic.heads[0][0].in_features == mixed.state_dim + mixed.plan_dim + HORIZON * ACTION_DIM
    assert mixed.plan_critic.heads[0][0].in_features == mixed.state_dim + mixed.plan_dim
    s, st, pr = torch.zeros(3, STATE_DIM), torch.arange(3 * 2 * STATE_DIM).reshape(3, 2, STATE_DIM).float(), torch.ones(3, PROPRIO_DIM)
    state = struct.state_input(s, st, pr)
    assert torch.equal(state[:, :2 * STATE_DIM], st.reshape(3, -1)) and torch.equal(state[:, -PROPRIO_DIM:], pr)
    assert torch.equal(mixed.state_input(s, st, pr)[:, :STATE_DIM], s) and mixed.plan_input(None, None) is None
    assert struct.plan_input(torch.zeros(3, 2, STATE_DIM), torch.ones(3, 2, 8, STATE_DIM)).shape == (3, 2, 8 * STATE_DIM)
    with pytest.raises(ValueError):
        struct.state_input(s, st, None)
    payload = struct.inference_state_dict()
    assert payload["layout"] == {"state_features": "struct", "proprio": True, "plan_candidates": 1}
    assert DiceResidualModel.layout_of({"actor": {}}) == {"state_features": "pool", "proprio": False}
    with pytest.raises(ValueError, match="layout mismatch"):
        DiceResidualModel(device="cpu", proprio=True).load_inference_state_dict(payload)
    info = mixed.update_plan_critic(mixed.state_input(s, None, pr), torch.randn(3, 2, STATE_DIM), torch.randn(3, 4, HORIZON, ACTION_DIM),
                                    mask_unused_dof(torch.randn(3, 2, 4, HORIZON, ACTION_DIM)))
    assert set(info) == {"plan_critic_loss", "v_plan_mean", "plan_target_mean"}
    del struct, mixed
    torch.manual_seed(0)
    a = DiceResidualModel(device="cpu", plan_candidates=2)
    torch.manual_seed(0)
    b = DiceResidualModel(device="cpu", plan_candidates=2, state_features="pool", proprio=False)
    assert all(torch.equal(x, y) for x, y in zip(a.critic.parameters(), b.critic.parameters()))


def test_update_runs_actor_and_polyak_on_every_second_gradient_step():
    from script.lingbot_rl_train import _update_from_buffer

    torch.manual_seed(7)
    model = DiceResidualModel(device="cpu")
    buf = _online_buffer()
    samples, order, targets = [], [], []
    critic, actor = model.update_critic, model.update_actor
    weight = model.target_critic.heads[0][0].weight
    original = buf.sample

    def spy_critic(state, action, target_q):
        order.append(("critic", id(state)))
        targets.append(weight.detach().clone())
        return critic(state, action, target_q)

    def spy_actor(state, *args):
        order.append(("actor", id(state)))
        return actor(state, *args)

    buf.sample = lambda size, expert_ratio: samples.append(size) or original(size, expert_ratio)
    model.update_critic = spy_critic
    model.update_actor = spy_actor
    info = _update_from_buffer(model, buf, 0.0, "cpu", False)
    actor_steps = GRADIENT_STEPS // ACTOR_EVERY
    assert len(samples) == GRADIENT_STEPS
    assert [kind for kind, _ in order] == ["critic", "actor", "critic"] * actor_steps
    assert all(order[i * 3 + 2][1] == order[i * 3 + 1][1] for i in range(actor_steps))
    moved = [not torch.equal(targets[i - 1], targets[i]) for i in range(1, GRADIENT_STEPS)]
    assert moved == [False, True] * (actor_steps - 1) + [False]
    assert {"critic_loss", "actor_loss"} <= set(info)


def test_n_step_sparse_terminal_three_chunks():
    buf = _online_buffer()
    rows = buf.rows()
    assert [float(row["n_steps"]) for row in rows] == [3.0, 2.0, 1.0]
    assert [float(row["done"]) for row in rows] == [1.0, 1.0, 1.0]
    assert float(rows[0]["reward"]) == pytest.approx(GAMMA ** 2)
    assert float(rows[1]["reward"]) == pytest.approx(GAMMA)
    assert float(rows[2]["reward"]) == pytest.approx(1.0)


def test_only_finalized_online_rows_are_ready_to_sample():
    buf = ChunkReplay(capacity=32)
    for _ in range(3):
        buf.add_online(_replay_row(0, 0))
    assert buf.ready_online_count() == 0
    with pytest.raises(ValueError, match="no online"):
        buf.sample(1, expert_ratio=0.0)
    buf.finalize_episode()
    assert buf.ready_online_count() == 3
    for _ in range(4):
        buf.add_expert(_replay_row(0, 1, expert=True))
    buf.finalize_episode()
    assert buf.ready_online_count() == 3
    assert len(buf) == 7


def test_rlpd_mix_respects_scheduled_ratio():
    buf = ChunkReplay(capacity=200)
    for _ in range(80):
        buf.add_online(_replay_row(0, 0))
    buf.finalize_episode()
    for _ in range(80):
        buf.add_expert(_replay_row(0, 1, expert=True))
    buf.finalize_episode()
    ratio = RLConfig().validate().rlpd_expert_ratio(104_000)
    assert ratio == pytest.approx(0.5)
    counts = [float(buf.sample(20, expert_ratio=ratio)["is_expert"].mean()) for _ in range(40)]
    assert abs(sum(counts) / len(counts) - 0.5) < 0.15


def test_plan_rows_default_finalize_and_sample_with_one_shape():
    buffer = ChunkReplay(plans=3)
    first = _replay_row(0.0, 0.0)
    first["plan_pool"] = np.full(STATE_DIM, 1.0, np.float32)
    buffer.add_online(first)
    second = _replay_row(1.0, 1.0)
    second["plan_pool"] = np.full(STATE_DIM, 2.0, np.float32)
    second["plan_pools_all"] = np.stack([np.full(STATE_DIM, v, np.float32) for v in (2.0, 5.0, 7.0)])
    second["plan_index"] = 0
    buffer.add_online(second)
    buffer.finalize_episode()
    rows = buffer.rows()
    assert rows[0]["plan_pools_all"].shape == (3, STATE_DIM) and float(rows[0]["plan_pools_all"][2, 0]) == 1.0
    assert np.array_equal(rows[0]["plan_a_base_all"][1], rows[0]["a_base_all"])
    assert float(rows[0]["plan_pool_next"][0]) == 2.0 and float(rows[1]["plan_pool_next"][0]) == 2.0
    batch = buffer.sample(2, 0.0)
    assert batch["plan_pools_all"].shape == (2, 3, STATE_DIM) and batch["plan_index"].dtype == torch.long
    assert batch["plan_a_base_all"].shape == (2, 3, K_CANDIDATES, HORIZON, ACTION_DIM)
    legacy = ChunkReplay(plans=3)
    payload = buffer.state_dict()
    for row in payload["data"]:
        for key in ("plan_pool", "plan_pools_all", "plan_index", "plan_a_base_all", "plan_pool_next"):
            row.pop(key)
    legacy.load_state_dict(payload)
    assert legacy.sample(2, 0.0)["plan_pools_all"].shape == (2, 3, STATE_DIM)


def test_feature_rows_default_finalize_and_sample_in_bf16_with_store_only_keys():
    buffer = ChunkReplay(plans=2)
    first = _replay_row(0.0, 0.0)
    first.update(proprio=np.full(8, 0.5, np.float32), state_struct=np.full((2, STATE_DIM), 1.0, np.float32),
                 plan_struct=np.full((8, STATE_DIM), 1.0, np.float32), object_state=np.arange(11, dtype=np.float32),
                 reset_kind=2, variant_index=1989)
    buffer.add_online(first)
    second = _replay_row(1.0, 1.0)
    second.update(proprio=np.full(8, 2.0, np.float32), state_struct=torch.full((2, STATE_DIM), 2.0, dtype=torch.bfloat16),
                  plan_struct_all=torch.full((2, 8, STATE_DIM), 3.0, dtype=torch.bfloat16),
                  plan_latent_all=torch.zeros(2, 48, 4, 8, 16, dtype=torch.bfloat16))
    buffer.add_online(second)
    buffer.finalize_episode()
    rows = buffer.rows()
    assert rows[0]["state_struct"].dtype == torch.bfloat16 and rows[0]["plan_struct_all"].shape == (2, 8, STATE_DIM)
    assert float(rows[0]["plan_struct_all"][1, 0, 0]) == 1.0 and rows[1]["plan_latent_all"].shape == (2, 48, 4, 8, 16)
    assert all(rows[0][key].numel() == 0 and rows[0][key].dtype == torch.bfloat16 for key in ("state_tokens", "state_latent", "plan_latent_all"))
    assert float(rows[0]["proprio_next"][0]) == 2.0 and float(rows[1]["proprio_next"][0]) == 2.0
    assert float(rows[0]["state_struct_next"][0, 0]) == 2.0 and float(rows[0]["plan_struct_next"][0, 0]) == 0.0
    assert rows[0]["object_state"].shape == (11,) and rows[1]["object_state"].shape == (0,)
    assert (rows[0]["reset_kind"], rows[0]["variant_index"], rows[1]["reset_kind"], rows[1]["variant_index"]) == (2, 1989, 0, 0)
    batch = buffer.sample(2, 0.0)
    assert batch["state_struct"].shape == (2, 2, STATE_DIM) and batch["state_struct"].dtype == torch.bfloat16
    assert batch["proprio"].shape == (2, 8) and batch["plan_struct_all"].shape == (2, 2, 8, STATE_DIM)
    assert batch["plan_struct"].shape == batch["plan_struct_next"].shape == (2, 8, STATE_DIM)
    assert not {"state_tokens", "state_latent", "plan_latent_all", "object_state", "reset_kind", "variant_index"} & set(batch)
    legacy = ChunkReplay(plans=2)
    payload = buffer.state_dict()
    for row in payload["data"]:
        for key in ("proprio", "proprio_next", "state_struct", "state_struct_next", "state_tokens", "state_latent", "plan_struct",
                    "plan_struct_all", "plan_struct_next", "plan_latent_all", "object_state", "reset_kind", "variant_index"):
            row.pop(key)
    legacy.load_state_dict(payload)
    assert legacy.sample(2, 0.0)["plan_struct"].shape == (2, 8, STATE_DIM) and legacy.rows()[0]["object_state"].shape == (0,)
    assert legacy.rows()[1]["plan_latent_all"].numel() == 0 and legacy.rows()[1]["state_struct"].shape == (2, STATE_DIM)


def test_plan_labels_default_to_zeros_and_sample_with_the_plan_shape():
    buffer = ChunkReplay(plans=2)
    first = _replay_row(0.0, 0.0)
    first["plan_labels"] = np.arange(8, dtype=np.float32).reshape(2, 4)
    buffer.add_online(first)
    buffer.add_online(_replay_row(1.0, 1.0))
    buffer.finalize_episode()
    rows = buffer.rows()
    assert rows[0]["plan_labels"].dtype == np.float32 and float(rows[0]["plan_labels"][1, 3]) == 7.0
    assert rows[1]["plan_labels"].shape == (2, K_CANDIDATES) and not rows[1]["plan_labels"].any()
    batch = buffer.sample(2, 0.0)
    assert batch["plan_labels"].shape == (2, 2, K_CANDIDATES) and batch["plan_labels"].dtype == torch.float32
    bad = _replay_row(0.0, 0.0)
    bad["plan_labels"] = np.zeros((3, K_CANDIDATES), np.float32)
    with pytest.raises(ValueError):
        buffer.add_online(bad)
    legacy = ChunkReplay(plans=2)
    payload = buffer.state_dict()
    for row in payload["data"]:
        row.pop("plan_labels")
    legacy.load_state_dict(payload)
    assert legacy.sample(2, 0.0)["plan_labels"].shape == (2, 2, K_CANDIDATES) and not legacy.rows()[0]["plan_labels"].any()


def test_token_store_round_trips_chunks_rolls_shards_and_continues_ordinals_across_reopen(tmp_path, monkeypatch):
    import script.lingbot_rl_tokens as tokens_module
    from script.lingbot_rl_tokens import TokenStore
    monkeypatch.setattr(tokens_module, "SHARD_CHUNKS", 2)
    chunk = lambda value: torch.full((2, 3, 4), value, dtype=torch.bfloat16)
    store = TokenStore(tmp_path / "tokens", 0)
    first = store.append(chunk(1.5), chunk=0, episode=0, step=12, reset_kind=0, variant_index=9, plan_index=1)
    store.append(chunk(-2.0), chunk=1, episode=0, step=28, reset_kind=0, variant_index=9, plan_index=0)
    assert first == {"chunk": 0, "episode": 0, "step": 12, "reset_kind": 0, "variant_index": 9, "plan_index": 1, "shard": 0, "offset": 0}
    with pytest.raises(ValueError):
        store.append(chunk(0.0), chunk=5, episode=0, step=44, reset_kind=0, variant_index=9, plan_index=0)
    reopened = TokenStore(tmp_path / "tokens", 2)
    third = reopened.append(chunk(0.25), chunk=2, episode=1, step=44, reset_kind=1, variant_index=673, plan_index=1)
    assert (third["shard"], third["offset"]) == (1, 0) and reopened.ordinal == 3
    lines = [json.loads(line) for line in (tmp_path / "tokens" / "index.jsonl").read_text().splitlines()]
    assert [line["chunk"] for line in lines] == [0, 1, 2] and lines[1] == {**lines[1], "shard": 0, "offset": 1}
    assert torch.equal(reopened.read(lines[1]), chunk(-2.0)) and torch.equal(reopened.read(third), chunk(0.25))
    assert np.load(tmp_path / "tokens" / "shard_0000.npy", mmap_mode="r").shape == (2, 2, 3, 4)
    truncated = TokenStore(tmp_path / "tokens", 1)
    assert truncated.ordinal == 1 and len((tmp_path / "tokens" / "index.jsonl").read_text().splitlines()) == 1
    assert truncated.append(chunk(9.0), chunk=1, episode=0, step=28, reset_kind=0, variant_index=9, plan_index=0)["offset"] == 1
    with pytest.raises(RuntimeError, match="behind"):
        TokenStore(tmp_path / "tokens", 7)


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

    for name in ("decode_candidates", "extract_critic_state", "extract_features_from_latent",
                 "commit_executed", "select_action"):
        assert callable(getattr(ResidualLingBotPolicy, name))


def test_policy_feeds_proprio_from_the_batch_or_the_select_action_stash():
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates, policy.eval_plans, policy.eval_plan_ranker = 2, 1, "v_plan"
    policy.residual_model = DiceResidualModel(device="cpu", state_features="struct", proprio=True)
    captured = {}
    policy.decode_candidates = lambda batch, k: {"s": torch.zeros(1, STATE_DIM), "state_struct": torch.ones(1, 2, STATE_DIM),
                                                 "z": torch.zeros(2, HORIZON, ACTION_DIM), "a_base": torch.zeros(2, HORIZON, ACTION_DIM),
                                                 "first_chunk": False, "latents": torch.zeros(1)}
    policy._apply_residual_choice = lambda dec, index: captured.update(state=dec["state_in"]) or torch.zeros(1, 16, USED_DOF)
    policy.predict_action_chunk({"task": ["t"], "proprio": torch.full((1, 8), 3.0)})
    assert captured["state"].shape == (1, 2 * STATE_DIM + 8) and float(captured["state"][0, -1]) == 3.0 and float(captured["state"][0, 0]) == 1.0
    policy._last_proprio = torch.full((1, 8), 5.0)
    policy.predict_action_chunk(None)
    assert float(captured["state"][0, -1]) == 5.0
    original = LingBotVAPolicy.select_action
    LingBotVAPolicy.select_action = lambda self, batch, **kwargs: "ok"
    try:
        assert policy.select_action({"proprio": torch.full((1, 8), 7.0)}) == "ok" and float(policy._last_proprio[0, 0]) == 7.0
    finally:
        LingBotVAPolicy.select_action = original


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


def test_ingest_experts_skips_rows_from_other_tasks():
    from script.lingbot_rl_train import _ingest_experts

    buf = ChunkReplay(capacity=32)
    mine = _replay_row(1, 1, expert=True)
    other = _replay_row(1, 1, expert=True)
    other["task_id"] = 5
    _ingest_experts(buf, [other, mine, other], (0,))
    assert len(buf) == 1
    assert buf.rows()[0]["task_id"] == 0
    assert buf.ready_online_count() == 0


def test_featurize_experts_pools_published_latents(tmp_path):
    from script.lingbot_rl_data import featurize_experts

    class StubPolicy:
        def reset(self):
            return None

        def extract_critic_state(self, batch):
            raise AssertionError("RGB path should not run when published latents exist")

        def extract_features_from_latent(self, latent, batch):
            assert latent.shape[2] in (1, 4)
            value, frames = float(latent[0, 0, 0, 0, 0]), latent.shape[2]
            return {"pool": torch.full((1, 3072), value), "struct": torch.full((1, frames, 2, 3072), value),
                    "tokens": torch.full((1, 32 * frames, 3072), value)}

    latents = torch.zeros(1, 48, 8, 8, 16)
    latents[0, 0, 3] = 7
    actions = np.zeros((12 + 16, 7), np.float32)
    episode = {"actions": actions, "task": "put the moka pot", "task_id": 8, "latents": latents}
    rows = featurize_experts(
        StubPolicy(), [episode], norm=_eval_normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(rows) == 2
    assert float(rows[0]["s"][0]) == 0.0
    assert float(rows[1]["s"][0]) == 7.0


def test_expert_rows_pool_the_true_plan_frames(tmp_path):
    from script.lingbot_rl_data import featurize_experts
    calls = []

    class StubPolicy:
        config = type("C", (), {"device": "cpu"})()

        def reset(self):
            return None

        def extract_features_from_latent(self, latent, batch):
            calls.append(tuple(latent.shape))
            value, frames = float(latent[0, 0, 0, 0, 0]), latent.shape[2]
            return {"pool": torch.full((1, STATE_DIM), value), "struct": torch.full((1, frames, 2, STATE_DIM), value),
                    "tokens": torch.full((1, 32 * frames, STATE_DIM), value)}

    latents = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1, 1).expand(1, 48, 10, 2, 2).clone()
    episode = {"actions": np.zeros((36, USED_DOF), np.float32), "task": "t", "task_id": 9, "latents": latents, "success": True,
               "states": np.arange(36 * 8, dtype=np.float32).reshape(36, 8)}
    rows = featurize_experts(StubPolicy(), [episode], None, _eval_normalization(), tmp_path / "cache.pt")
    assert [row["s"][0] for row in rows] == [0.0, 3.0, 7.0]
    assert [row["plan_pool"][0] for row in rows] == [0.0, 4.0, 8.0]
    assert (1, 48, 4, 2, 2) in calls
    assert [float(row["proprio"][0]) for row in rows] == [0.0, 96.0, 224.0]
    assert [float(row["proprio_next"][0]) for row in rows] == [96.0, 224.0, 224.0]
    assert rows[1]["plan_struct"].shape == (8, STATE_DIM) and rows[1]["plan_struct"].dtype == torch.bfloat16 and float(rows[1]["plan_struct"][0, 0]) == 4.0
    assert rows[0]["state_struct"].shape == (2, STATE_DIM) and rows[0]["state_tokens"].shape == (32, STATE_DIM)
    assert rows[1]["state_latent"].shape == (48, 1, 2, 2) and float(rows[1]["state_latent"][0, 0, 0, 0]) == 3.0
    assert rows[1]["plan_latent"].shape == (48, 4, 2, 2) and float(rows[1]["plan_latent"][0, 0, 0, 0]) == 4.0


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
    residual = model.actor(state, noise, action)
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


def test_save_load_resume_roundtrips_rng_checkpointed_and_schedulers(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    model = DiceResidualModel(device="cpu")
    for _ in range(12):
        model.actor_opt.step()
        model.critic_opt.step()
        model.actor_lr.step()
        model.critic_lr.step()
    buffer = _resume_buffer()
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    save_resume(path, model, buffer, 0, 0, recipe, checkpointed={0, 80_000}, wandb_id="8otkv33a")
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    restored = DiceResidualModel(device="cpu")
    buffer = ChunkReplay()
    env_steps, chunks, checkpointed, wandb_id, episodes = load_resume(path, restored, buffer, recipe)
    assert env_steps == 0
    assert chunks == 0
    assert checkpointed == {0, 80_000}
    assert wandb_id == "8otkv33a"
    assert episodes == 0
    assert restored.actor_lr.state_dict()["last_epoch"] == 12 == restored.critic_lr.last_epoch
    assert restored.actor_opt.param_groups[0]["lr"] == model.actor_opt.param_groups[0]["lr"]
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
    save_resume(path, model, _resume_buffer(), 12, 3, recipe, checkpointed={0}, wandb_id="abc")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng"]["torch"] = payload["rng"]["torch"].to(dtype=torch.int32)
    torch.save(payload, path)
    torch.manual_seed(0)
    env_steps, chunks, checkpointed, wandb_id, episodes = load_resume(
        path, DiceResidualModel(device="cpu"), ChunkReplay(), recipe)
    assert env_steps == 12
    assert chunks == 3
    assert checkpointed == {0}
    assert wandb_id == "abc"
    got = torch.rand(3)
    torch.manual_seed(7)
    torch.testing.assert_close(got, torch.rand(3))


def test_load_resume_rejects_recipe_mismatch(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "latest.pt"
    save_resume(path, DiceResidualModel(device="cpu"), _resume_buffer(), 0, 0, recipe, checkpointed={0})
    with pytest.raises(ValueError, match="recipe fingerprint"):
        load_resume(path, DiceResidualModel(device="cpu"), ChunkReplay(), {**recipe, "beta": 1.0})


def test_load_resume_accepts_legacy_torch_numpy_rng(tmp_path):
    from script.lingbot_rl_train import load_resume, save_resume

    recipe = RLConfig().protocol()
    path = tmp_path / "resume" / "latest.pt"
    save_resume(path, DiceResidualModel(device="cpu"), _resume_buffer(), 0, 0, recipe, checkpointed={0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng"] = {
        "torch": payload["rng"]["torch"].to(dtype=torch.int32),
        "numpy": payload["rng"]["numpy"],
    }
    payload.pop("checkpointed")
    payload.pop("wandb_id")
    torch.save(payload, path)
    env_steps, _, checkpointed, wandb_id, episodes = load_resume(
        path, DiceResidualModel(device="cpu"), ChunkReplay(), recipe)
    assert env_steps == 0
    assert checkpointed == set()
    assert wandb_id is None


class _Observable:
    def __init__(self, modality):
        self.modality, self.enabled = modality, True

    def set_enabled(self, enabled):
        self.enabled = enabled


class _TwinSim:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0, qpos=np.array([0.3, 0.0, 0.0, -2.0]), qvel=np.zeros(4), qacc_warmstart=np.zeros(4),
                                    body_xpos=np.zeros((2, 3)), body_xquat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
        self.model = SimpleNamespace(get_joint_qpos_addr=lambda name: 3)
        self.forward()

    def get_state(self):
        return SimpleNamespace(time=self.data.time, qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())

    def set_state(self, state):
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    def forward(self):
        self.data.body_xpos[1] = self.data.qpos[:3]


class _TwinInner:
    def __init__(self, drift):
        self.sim = _TwinSim()
        self.obj_body_id = {"micro": 0, "mug": 1}
        self.ignore_done = False
        self._observables = {"agentview_image": _Observable("image"), "robot0_eef_pos": _Observable("proprio")}
        self.robots = [SimpleNamespace(
            controller=SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3), initial_joint=np.zeros(7), update=lambda force=False: None),
            gripper=SimpleNamespace(current_action=np.zeros(2)))]
        self.drift, self.steps = drift, 0

    def step(self, action):
        self.sim.forward()
        self.sim.data.qpos[:3] += 0.01 * action[:3] + self.drift
        self.sim.data.qpos[3] += 0.5 * max(float(action[6]), 0.0)
        self.steps += 1
        return {}, 0.0, False, {}

    def _check_success(self):
        return bool(self.sim.data.qpos[3] >= 0.0)


class _TwinEnv:
    def __init__(self, drift=0.0):
        self._env = SimpleNamespace(env=_TwinInner(drift))

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        inner = self._env.env
        obs, reward, done, info = inner.step(action)
        success = inner._check_success()
        return obs, reward, done or success, False, {**info, "is_success": success}

    def close(self):
        return None


_TWIN_LAYOUT = {"bodies": ["micro", "mug"], "joints": ["door"], "body_quaternion": "wxyz"}
_TWIN_CALIBRATION = {"task_id": 9, "mug": "mug", "microwave": "micro", "door": "door", "door_closed": 0.0, "door_open": -2.0}


def test_label_plans_scores_every_candidate_from_the_live_snapshot_with_the_chunk_length():
    from script.lingbot_rl_train import capture_live, label_plans, object_state
    live, shadow = _TwinEnv(), _TwinEnv()
    shadow._env.env.sim.data.qpos[:] = [9.0, 9.0, 9.0, 9.0]
    live._env.env.robots[0].controller.initial_joint[:] = 0.5
    candidates = torch.zeros(2, 2, HORIZON, ACTION_DIM)
    candidates[0, :, :, 0] = 1.0
    candidates[1, :, :, 0] = -1.0
    candidates[1, 1, 4:, 6] = 1.0
    norm = _eval_normalization()
    snapshot, before = capture_live(live._env.env), object_state(live._env.env, _TWIN_LAYOUT)
    labels = label_plans(shadow, snapshot, before, candidates, True, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    assert labels.shape == (2, 2) and labels.dtype == np.float32
    assert labels[0, 0] == pytest.approx(-0.11, abs=1e-4) and labels[0, 1] == labels[0, 0]
    assert labels[1, 0] == pytest.approx(0.11, abs=1e-4) and labels[1, 1] == pytest.approx(2.03, abs=1e-4)
    assert shadow._env.env.steps == 12 + 12 + 12 + 4 and live._env.env.steps == 0 and live._env.env.sim.data.qpos[0] == 0.3
    assert float(shadow._env.env.robots[0].controller.initial_joint[0]) == 0.5
    later = label_plans(shadow, snapshot, before, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    assert later[0, 0] == pytest.approx(-0.15, abs=1e-4) and later[1, 0] == pytest.approx(0.15, abs=1e-4)
    with pytest.raises(RuntimeError, match="shadow state mismatch"):
        label_plans(shadow, snapshot, before + 1.0, candidates, True, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    live._env.env.step(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], np.float32))
    snapshot, before = capture_live(live._env.env), object_state(live._env.env, _TWIN_LAYOUT)
    assert float(snapshot["state"].qpos[0] - before[7]) == pytest.approx(5e-4, abs=1e-7)
    assert label_plans(shadow, snapshot, before, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION).shape == (2, 1)
    live._env.env.step(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], np.float32))
    snapshot, before = capture_live(live._env.env), object_state(live._env.env, _TWIN_LAYOUT)
    with pytest.raises(RuntimeError, match="shadow state mismatch 0.02"):
        label_plans(shadow, snapshot, before, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)


def _twin_shadow(task_id, suite, init_states, plus, seed):
    import script.lingbot_rl_train as train
    train.LiberoEnv = lambda **kwargs: _TwinEnv()
    shadow = train.make_shadow(task_id, suite, init_states, plus, seed)
    shadow._env.env.sim.data.qpos[:] = 9.0
    return shadow


def test_label_pool_matches_label_plans_across_two_workers_and_propagates_the_mismatch():
    from script.lingbot_rl_label import LabelPool
    from script.lingbot_rl_train import capture_live, label_plans, object_state
    live, shadow = _TwinEnv(), _TwinEnv()
    shadow._env.env.sim.data.qpos[:] = 9.0
    live._env.env.robots[0].controller.initial_joint[:] = 0.5
    candidates = torch.rand(2, 3, HORIZON, ACTION_DIM, generator=torch.Generator().manual_seed(0)) * 2.0 - 1.0
    norm = _eval_normalization()
    snapshot, before = capture_live(live._env.env), object_state(live._env.env, _TWIN_LAYOUT)
    pool = LabelPool(2, None, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION, shadow_factory=_twin_shadow)
    try:
        labels, seconds = pool.submit(snapshot, before, candidates.numpy(), True, 9, False, False, 0).result()
        expected = label_plans(shadow, snapshot, before, candidates, True, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
        assert labels.shape == (2, 3) and labels.dtype == np.float32 and np.array_equal(labels, expected) and seconds >= 0.0
        assert len(set(labels.reshape(-1).tolist())) > 1
        later = pool.submit(snapshot, before, candidates[:, :1].numpy(), False, 9, False, False, 0).result()[0]
        assert np.array_equal(later, label_plans(shadow, snapshot, before, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION))
        with pytest.raises(RuntimeError, match="shadow state mismatch"):
            pool.submit(snapshot, before + 1.0, candidates.numpy(), True, 9, False, False, 0).result()
        assert np.array_equal(pool.submit(snapshot, before, candidates.numpy(), True, 9, False, False, 0).result()[0], expected)
    finally:
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)


def test_shadow_fidelity_measures_object_pose_gaps_per_chunk_and_gates_on_them(monkeypatch):
    import script.lingbot_rl_train as train
    from script.lingbot_rl_train import make_shadow, shadow_fidelity
    monkeypatch.setattr(train, "LiberoEnv", lambda **kwargs: _TwinEnv())
    calls = []
    act = lambda observation: calls.append(observation) or np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], np.float32)
    live, shadow = _TwinEnv(), make_shadow(9, None, False, False, 0)
    observables = shadow._env.env._observables
    assert shadow._env.env.ignore_done is True and observables["agentview_image"].enabled is False and observables["robot0_eef_pos"].enabled is True
    result = shadow_fidelity(live, shadow, _TWIN_LAYOUT, act, {"step": 0}, chunks=3)
    assert result == {"position_m": 0.0, "angle_rad": 0.0, "chunks": 3} and len(calls) == 12 + 16 + 16
    assert shadow._env.env.steps == live._env.env.steps == 44 and calls[0] == {"step": 0}
    with pytest.raises(RuntimeError, match="fidelity"):
        shadow_fidelity(_TwinEnv(), _TwinEnv(drift=2e-4), _TWIN_LAYOUT, act, {}, chunks=3)


class _StubPolicy:
    plan_calls = []

    def reset(self):
        self._executed_actions = None

    def decode_candidates(self, batch, k=4, **kwargs):
        a_base = torch.zeros(k, HORIZON, ACTION_DIM)
        for index in range(k):
            a_base[index, :, 0] = index
        return {
            "s": torch.zeros(1, STATE_DIM),
            "z": torch.zeros(k, HORIZON, ACTION_DIM),
            "a_base": a_base,
            "video_noise": torch.zeros(1),
            "first_chunk": True,
            "state_struct": torch.zeros(1, 2, STATE_DIM),
            "state_tokens": torch.zeros(1, 32, STATE_DIM),
            "state_latent": torch.zeros(1, 48, 1, 8, 16),
        }

    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False, return_tokens=False):
        type(self).plan_calls.append((plans, k, random_plan, decode_all, return_tokens))
        base = self.decode_candidates(batch, k)
        pools = torch.stack([torch.full((STATE_DIM,), float(m)) for m in range(plans)])
        struct_all = pools[:, None].expand(plans, 8, STATE_DIM)
        all_base = torch.stack([base["a_base"] + 10.0 * m for m in range(plans)])
        model = self.residual_model
        index = 1 if random_plan else int(rank_fn(
            model.state_input(base["s"], base["state_struct"], batch.get("proprio")), model.plan_input(pools, struct_all)))
        return {**base, "plan_pools": pools, "plan_struct_all": struct_all, "plan_latent_all": torch.zeros(plans, 48, 4, 8, 16),
                "plan_pool": pools[index:index + 1], "plan_struct": struct_all[index:index + 1], "plan_index": index,
                "plan_a_base_all": all_base, "a_base": all_base[index],
                **({"plan_tokens": torch.full((plans, 8, STATE_DIM), 0.5, dtype=torch.bfloat16)} if return_tokens else {})}

    def commit_executed(self, chunk, first_chunk=False):
        self._executed_actions = chunk

    def observe_env_step(self, batch):
        return None


def _sum_critic_model(device="cpu", residual_input="z", epsilon=-0.5, plan_candidates=1, **kwargs):
    model = DiceResidualModel(device=device, residual_input=residual_input, plan_candidates=plan_candidates, **kwargs)
    model.critic = _SumCritic()
    return model


def _smoke_train(tmp_path, monkeypatch, budget, prepared=None, model_factory=None, **kwargs):
    import sys

    import script.lingbot_rl_train as train

    envs, logs, wandb_calls = [], [], []

    class StubEnv:
        def __init__(self, **env_kwargs):
            envs.append(env_kwargs)
            self.steps = 0
            self.proprio = env_kwargs.get("obs_type") == "pixels_agent_pos"
            self._init_states = [0]
            self._env = SimpleNamespace(env=SimpleNamespace(_observables={}))

        def _obs(self):
            zeros = np.zeros((128, 128, 3), np.uint8)
            obs = {"pixels": {"image": zeros, "image2": zeros}}
            if self.proprio:
                obs["robot_state"] = {"eef": {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                                      "gripper": {"qpos": np.array([0.04, -0.04])}}
            return obs

        def reset(self, seed=None):
            self.steps = 0
            return self._obs(), {}

        def step(self, action):
            self.steps += 1
            return self._obs(), 0.0, self.steps >= 12, False, {"is_success": False}

        def close(self):
            return None

    class Run:
        id = "stub-run"
        summary = {}

        def log(self, payload):
            logs.append(payload)

        def finish(self, **finish_kwargs):
            return None

    prepared_path = tmp_path / "prepared.json"
    if prepared is not None:
        prepared_path.write_text(json.dumps(prepared))
    if model_factory is not None:
        monkeypatch.setattr(train, "DiceResidualModel", model_factory)
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: _StubPolicy())
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setattr(train, "_plus_suite", lambda base: (None, 2519))
    monkeypatch.setattr(train, "object_layout", lambda inner: {"bodies": ["mug"], "joints": ["door"]})
    monkeypatch.setattr(train, "object_state", lambda inner, layout: np.array([7.0, 8.0], np.float32))
    monkeypatch.setattr(train.np.random, "randint", lambda n: 0 if n == 1 else 1)
    monkeypatch.setitem(sys.modules, "wandb", type("W", (), {
        "init": staticmethod(lambda **k: wandb_calls.append(k) or Run()),
        "finish": staticmethod(lambda **k: None),
    })())
    train.train(output_dir=tmp_path, run_name="unit", max_env_steps=budget,
                prepared_path=prepared_path, **kwargs)
    resume = torch.load(tmp_path / "resume" / "latest.pt", map_location="cpu", weights_only=False)
    return {"rows": resume["replay"]["data"], "logs": logs, "envs": envs,
            "wandb": wandb_calls, "resume": resume}


def test_train_smoke_loop_gates_updates_saves_weights_and_runs_hook(tmp_path, monkeypatch):
    from script.lingbot_rl_train import _hook

    stamp = tmp_path / "hook.txt"
    out = _smoke_train(tmp_path, monkeypatch, 42, commit=_hook(f"printf %s $DICE_STEP > {stamp}"))
    assert out["envs"] and all(env["init_states"] is False for env in out["envs"])
    assert stamp.read_text() == "42"
    payload = torch.load(tmp_path / "residual.pt", map_location="cpu", weights_only=True)
    assert set(payload) <= {"actor", "critic", "target_critic", "layout"}
    assert "transformer" not in out["resume"]
    assert out["resume"]["env_steps"] == 42
    assert out["resume"]["checkpointed"] == []
    assert "recipe" in out["resume"]
    assert "python" in out["resume"]["rng"] and "cuda" in out["resume"]["rng"]
    updates = [entry for entry in out["logs"] if "critic_loss" in entry]
    assert len(out["rows"]) == 4
    assert [entry["chunks"] for entry in updates] == [UPDATE_EVERY_CHUNKS]
    assert updates[0]["buffer_online_rows"] == 3
    assert [float(row["done"]) for row in out["rows"]] == [1.0, 1.0, 1.0, 0.0]
    assert [int(row["n_env_actions"]) for row in out["rows"]] == [12, 12, 12, 6]
    assert all(out["rows"][0][key].numel() == 0 for key in ("state_tokens", "state_latent", "plan_latent_all"))


def test_selection_is_random_during_warmup_and_argmax_afterwards(tmp_path, monkeypatch):
    warmup = _smoke_train(
        tmp_path / "warm", monkeypatch, 36, model_factory=_sum_critic_model,
        config=RLConfig(selection_warmup_steps=10_000))
    assert [float(row["a"][0][0]) for row in warmup["rows"]] == [1.0, 1.0, 1.0]
    greedy = _smoke_train(
        tmp_path / "greedy", monkeypatch, 36, model_factory=_sum_critic_model,
        config=RLConfig(selection_warmup_steps=0))
    assert [float(row["a"][0][0]) for row in greedy["rows"]] == [3.0, 3.0, 3.0]
    assert [entry["selection_active"] for entry in greedy["logs"] if "selection_active" in entry] == []


def test_train_with_plans_stores_plan_rows_trains_the_plan_critic_and_logs_metrics(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    out = _smoke_train(tmp_path / "ranked", monkeypatch, 42, model_factory=lambda **kw: DiceResidualModel(**kw),
                       config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base"))
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert rows[0]["plan_pools_all"].shape == (3, STATE_DIM)
    assert rows[0]["plan_a_base_all"].shape == (3, K_CANDIDATES, HORIZON, ACTION_DIM)
    assert float(rows[0]["plan_pool"][0]) == float(rows[0]["plan_index"])
    assert all(call[2] is False and call[3] is True for call in _StubPolicy.plan_calls)
    logged = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert logged and {"plan_agreement", "plan_regret", "plan_value_spread", "plan_random_rate", "seconds_per_env_step",
                       "replay_plan_delta_v", "replay_plan_delta_h"} <= set(logged[0])
    assert {"plan_critic", "plan_opt", "plan_lr"} <= set(out["resume"])
    _StubPolicy.plan_calls = []
    _smoke_train(tmp_path / "warm", monkeypatch, 42, model_factory=lambda **kw: DiceResidualModel(**kw),
                 config=RLConfig(plan_candidates=3, plan_random_prob=0.0, selection_warmup_steps=10_000))
    assert all(call[2] is True for call in _StubPolicy.plan_calls)


def test_single_plan_training_never_calls_select_plan(tmp_path, monkeypatch):
    _StubPolicy.plan_calls = []
    out = _smoke_train(tmp_path, monkeypatch, 42, model_factory=lambda **kw: DiceResidualModel(**kw), config=RLConfig(selection_warmup_steps=0))
    assert _StubPolicy.plan_calls == [] and not any("plan_agreement" in entry for entry in out["logs"])


def test_train_resume_reuses_wandb_id_and_skips_ingest(tmp_path, monkeypatch):
    import script.lingbot_rl_train as train

    def boom(*args, **kwargs):
        raise AssertionError("resume must not re-featurize experts")

    train.save_resume(
        tmp_path / "resume" / "latest.pt", DiceResidualModel(device="cpu"), _resume_buffer(),
        12, 1, RLConfig().protocol(), checkpointed={0}, wandb_id="run-from-checkpoint")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "dataset_manifest.json").write_text("{}")
    prepared = {
        "tasks": [{"task_id": task_id, "instruction": f"task-{task_id}", "initial_state_count": 50}
                  for task_id in range(10)],
        "normalization": _eval_normalization(),
        "checkpoint": str(ckpt),
        "model_path": None,
        "architecture": {},
        "source_run": "libero30-sft",
        "checkpoint_step": 600,
    }
    monkeypatch.setattr(train, "featurize_experts", boom)
    monkeypatch.setattr(train, "load_manifest_episodes", boom)
    monkeypatch.setattr(
        "script.lingbot_eval.read_checkpoint_metadata",
        lambda path: {"architecture": {}, "normalization": _eval_normalization()},
    )
    out = _smoke_train(
        tmp_path, monkeypatch, 24, prepared=prepared, resume=True, dataset_root=tmp_path / "dataset")
    assert out["wandb"][0]["id"] == "run-from-checkpoint"
    assert out["wandb"][0]["resume"] == "must"


def test_draw_reset_is_per_episode_deterministic_and_covers_every_kind():
    from script.lingbot_rl_config import plus_train_variants
    from script.lingbot_rl_train import draw_reset
    config = RLConfig(task_ids=(9,), plus_train=plus_train_variants())
    draws = [draw_reset(config, episode) for episode in range(300)]
    assert draws == [draw_reset(config, episode) for episode in range(300)]
    assert {draw[0] for draw in draws} == {0, 1, 2} and all(0.0 <= draw[2] < 1.0 for draw in draws)
    assert all(draw[1] is None for draw in draws if draw[0] == 0)
    assert all(draw[1] in config.plus_train[("robot_init", "layout")[draw[0] - 1]] for draw in draws if draw[0])
    assert [draw[:2] for draw in draws[:4]] == [(0, None), (1, 673), (0, None), (2, 1989)]
    assert draw_reset(RLConfig(), 5) == (0, None, 0.0)


def test_proprio_from_observation_follows_the_dataset_convention():
    from script.lingbot_rl_train import proprio_from_observation
    observation = {"robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])},
                                   "gripper": {"qpos": np.array([0.04, -0.04])}}}
    state = proprio_from_observation(observation)
    assert state.dtype == np.float32 and state.shape == (8,)
    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 0.0, 0.0, 0.6, 0.04, -0.04], atol=1e-6)
    observation["robot_state"]["eef"]["quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(proprio_from_observation(observation)[3:6], 0.0)


def test_object_layout_walks_articulated_joints_under_base_bodies_and_reads_poses():
    from script.lingbot_rl_train import object_layout, object_state
    model = SimpleNamespace(njnt=3, jnt_type=[0, 3, 3], jnt_bodyid=[2, 4, 5], body_parentid=[0, 0, 0, 0, 3, 1],
                            joint_id2name=lambda joint: ["mug_free", "door_hinge", "robot_joint"][joint],
                            get_joint_qpos_addr=lambda name: {"door_hinge": 9}[name])
    data = SimpleNamespace(body_xpos=np.arange(18, dtype=np.float64).reshape(6, 3), body_xquat=np.arange(24, dtype=np.float64).reshape(6, 4),
                           qpos=np.arange(12, dtype=np.float64))
    inner = SimpleNamespace(sim=SimpleNamespace(model=model, data=data), obj_body_id={"mug": 2, "microwave": 3})
    layout = object_layout(inner)
    assert layout == {"bodies": ["microwave", "mug"], "joints": ["door_hinge"], "body_quaternion": "wxyz"}
    expected = np.concatenate([data.body_xpos[3], data.body_xquat[3], data.body_xpos[2], data.body_xquat[2], [9.0]]).astype(np.float32)
    np.testing.assert_array_equal(object_state(inner, layout), expected)
    inner.obj_body_id["cookies"] = 5
    np.testing.assert_array_equal(object_state(inner, layout), expected)
    with pytest.raises(KeyError):
        object_state(inner, {"bodies": ["bowl"], "joints": []})


def test_train_under_plus_resets_with_proprio_stores_features_and_logs_per_kind(tmp_path, monkeypatch):
    import script.lingbot_rl_model as model_module
    from script.lingbot_rl_config import plus_train_variants
    from script.lingbot_rl_train import draw_reset
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base",
                      proprio=True, state_features="struct", plus_train=plus_train_variants())
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    probe, envs = out["envs"][0], out["envs"][1:]
    assert (probe["task_id"], probe["init_states"], probe["obs_type"]) == (2519, False, "pixels_agent_pos")
    draws = [draw_reset(config, episode) for episode in range(len(envs))]
    assert [draw[0] for draw in draws] == [0, 1, 0, 2, 1] and all(env["obs_type"] == "pixels_agent_pos" for env in envs)
    assert [(env["task_id"], env["init_states"], env["is_libero_plus"]) for env in envs] == \
        [(2519 if variant is None else variant, kind != 1, True) for kind, variant, _ in draws]
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    np.testing.assert_allclose(rows[0]["proprio"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04], atol=1e-6)
    assert rows[0]["plan_pool"].shape == rows[0]["s"].shape == (STATE_DIM,) and rows[0]["plan_pools_all"].shape == (2, STATE_DIM)
    assert rows[0]["state_struct"].dtype == torch.bfloat16 and rows[0]["object_state"].tolist() == [7.0, 8.0]
    assert rows[0]["state_tokens"].shape == (32, STATE_DIM) and rows[0]["plan_latent_all"].shape == (2, 48, 4, 8, 16)
    assert [row["reset_kind"] for row in rows] == [0, 1, 0, 2, 1] and [row["variant_index"] for row in rows[:2]] == [9, 673]
    episode_logs = [entry for entry in out["logs"] if "reset_kind" in entry]
    assert episode_logs[1]["variant_index"] == 673 and {"episode_success_robot_init", "episode_length_robot_init"} <= set(episode_logs[1])
    updates = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert updates and any(key.startswith("plan_agreement/") for key in updates[0])
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["object_layout"] == {"bodies": ["mug"], "joints": ["door"]} and settings["config"]["plus_train"]["layout"][0] == 1982
    assert out["resume"]["episodes"] == 5 and out["resume"]["layout"] == {"state_features": "struct", "proprio": True, "plan_candidates": 2}
    assert out["wandb"][0]["config"]["proprio"] is True and out["wandb"][0]["config"]["state_features"] == "struct"
    assert not any("episode_success_standard" in entry for entry in _smoke_train(tmp_path / "plain", monkeypatch, 24)["logs"])


def test_train_with_rollout_labels_overlaps_the_live_chunk_stores_them_and_records_both_fidelity_gates(tmp_path, monkeypatch):
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    from script.lingbot_rl_config import plus_train_variants
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    labelled, pools, closed = [], [], []
    monkeypatch.setattr(train, "calibrate", lambda task_id, inner, layout, initial: {"task_id": task_id, "door_closed": 0.0, "door_open": -2.094})
    monkeypatch.setattr(train, "capture_live", lambda live: {"stub": True})
    monkeypatch.setattr(train, "shadow_fidelity", lambda env, shadow, layout, act, observation, chunks=20: {
        "position_m": 2e-4, "angle_rad": 1e-3, "chunks": chunks})

    class StubPool:
        def __init__(self, workers, spec, norm, layout, calibration, shadow_factory=None):
            pools.append((workers, spec, layout, calibration["task_id"]))

        def submit(self, snapshot, before, candidates, first_chunk, env_task, init_states, plus, seed):
            labelled.append((snapshot, before.tolist(), type(candidates), candidates.dtype, tuple(candidates.shape), first_chunk, env_task, init_states, plus, seed))
            labels = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], np.float32)
            return SimpleNamespace(result=lambda: (labels, 0.0), wait=lambda: None)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(train, "LabelPool", StubPool)
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base",
                      proprio=True, plus_train=plus_train_variants(), plan_target="rollout")
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    envs = out["envs"]
    strip = lambda env: {key: value for key, value in env.items() if key != "obs_type"}
    assert len(envs) == 4 + 5 and all(envs[i]["obs_type"] == "pixels" and strip(envs[i]) == strip(envs[i - 1]) for i in (1, 3))
    assert (envs[2]["task_id"], envs[2]["init_states"], envs[2]["is_libero_plus"], envs[2]["obs_type"]) == (1982, True, True, "pixels_agent_pos")
    assert all(env["obs_type"] == "pixels_agent_pos" for env in envs[4:])
    assert pools == [(4, None, {"bodies": ["mug"], "joints": ["door"]}, 9)] and closed == [True]
    assert labelled and len(labelled) == 5 and labelled[0][:6] == ({"stub": True}, [7.0, 8.0], np.ndarray, np.float32, (2, K_CANDIDATES, HORIZON, ACTION_DIM), True)
    assert [(entry[6], entry[7], entry[8]) for entry in labelled] == [(env["task_id"], env["init_states"], True) for env in envs[4:]]
    assert labelled[0][9] == 42 and labelled[1][9] == 42 + 12
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert rows[0]["plan_labels"].dtype == np.float32 and rows[0]["plan_labels"].tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    updates = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert updates and {"label_agreement_with_q", "label_spread_within_plan", "label_spread_across_plans", "label_executed_minus_mean",
                        "label_seconds", "label_wait_seconds", "plan_target_q_mean", "plan_label_vs_q_agreement"} <= set(updates[0])
    assert any(key.startswith("label_seconds/") for key in updates[0]) and updates[0]["label_wait_seconds"] >= 0.0
    assert updates[0]["label_spread_within_plan"] == pytest.approx(float(np.std([0.0, 1.0, 2.0, 3.0])))
    assert updates[0]["label_spread_across_plans"] == pytest.approx(2.0) and abs(updates[0]["label_executed_minus_mean"]) == pytest.approx(2.0)
    assert updates[0]["label_agreement_with_q"] in (0.0, 1.0)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["shadow_fidelity"] == {"base": {"position_m": 2e-4, "angle_rad": 1e-3, "chunks": 20},
                                           "plus": {"variant_index": 1982, "position_m": 2e-4, "angle_rad": 1e-3, "chunks": 3}}
    assert settings["progress_calibration"]["door_open"] == -2.094
    assert out["resume"]["recipe"]["plan_target"] == "rollout" and out["resume"]["layout"]["plan_target"] == "rollout"
    plain = _smoke_train(tmp_path / "plain", monkeypatch, 24)
    assert len(pools) == 1 and not any("label_seconds" in entry for entry in plain["logs"])
    assert json.loads((tmp_path / "plain" / "settings.json").read_text())["shadow_fidelity"] is None


def test_train_with_store_tokens_writes_the_sidecar_from_the_worker_and_the_default_recipe_writes_none(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    from script.lingbot_rl_tokens import TokenStore
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    pools = []
    monkeypatch.setattr(train, "ThreadPoolExecutor", lambda **kwargs: pools.append(kwargs) or ThreadPoolExecutor(**kwargs))
    _StubPolicy.plan_calls = []
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base", store_tokens=True)
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    assert pools == [{"max_workers": 1}] and all(call[4] is True for call in _StubPolicy.plan_calls)
    lines = [json.loads(line) for line in (tmp_path / "tokens" / "index.jsonl").read_text().splitlines()]
    assert [line["chunk"] for line in lines] == [0, 1, 2, 3, 4] and [line["step"] for line in lines] == [0, 12, 24, 36, 48]
    assert lines[0]["episode"] == 0 and lines[1]["episode"] == 1 and {line["plan_index"] for line in lines} <= {0, 1}
    assert np.load(tmp_path / "tokens" / "shard_0000.npy", mmap_mode="r").shape == (256, 2, 8, STATE_DIM)
    assert torch.equal(TokenStore(tmp_path / "tokens", 5).read(lines[4]), torch.full((2, 8, STATE_DIM), 0.5, dtype=torch.bfloat16))
    assert not any("plan_tokens" in row for row in out["rows"])
    assert out["resume"]["recipe"] == config_from_dict({**config.to_dict(), "store_tokens": False}).protocol()
    _StubPolicy.plan_calls = []
    _smoke_train(tmp_path / "plain", monkeypatch, 24, config=RLConfig(plan_candidates=2, selection_warmup_steps=0))
    assert pools == [{"max_workers": 1}] and all(call[4] is False for call in _StubPolicy.plan_calls) and not (tmp_path / "plain" / "tokens").exists()
    with pytest.raises(ValueError):
        RLConfig(store_tokens=True).validate()


_GB10 = bool(os.environ.get("DICE_DATA")) and importlib.util.find_spec("libero") is not None


@pytest.mark.skipif(not _GB10, reason="needs DICE_DATA and the LIBERO install on a GB10")
def test_env_proprio_matches_the_first_demo_frame_of_a_stored_initial_state():
    import pyarrow.parquet as pq
    from pathlib import Path
    from script.lingbot_eval import describe_suite, read_json, seed_all
    from script.lingbot_rl_train import _env_class, proprio_from_observation
    data = Path(os.environ["DICE_DATA"])
    prepared = read_json(data / "prepared.json")
    prepared = read_json(prepared["prepared_path"]) if "prepared_path" in prepared else prepared
    manifest = read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json")
    root = Path((data / "dataset_root").read_text().strip())
    instruction = next(task["instruction"] for task in prepared["tasks"] if task["task_id"] == 9)
    demos = []
    for episode in [episode for episode in manifest["episodes"] if episode["tasks"][0] == instruction][:3]:
        chunk = episode["episode_index"] // manifest["info"]["chunks_size"]
        path = root / manifest["info"]["data_path"].format(episode_chunk=chunk, episode_index=episode["episode_index"])
        demos.append(np.asarray(pq.read_table(path, columns=["observation.state"])["observation.state"][0].as_py(), np.float32))
    suite, _ = describe_suite(prepared["assets_path"])
    env = _env_class()(task_suite=suite, task_id=9, task_suite_name="libero_10", episode_length=520, observation_height=128,
                       observation_width=128, obs_type="pixels_agent_pos", init_states=True, n_envs=1, num_steps_wait=0,
                       control_freq=20, control_mode="relative", hard_reset=True)
    try:
        states = []
        for state_id in range(len(env._init_states)):
            env.init_state_id = state_id
            seed_all(0)
            observation, _ = env.reset(seed=0)
            states.append(proprio_from_observation(observation))
    finally:
        env.close()
    states = np.stack(states)
    assert states.shape == (50, 8)
    for demo in demos:
        gaps = np.abs(states - demo[None]).max(axis=1)
        assert gaps.min() < 5e-2, (gaps.min(), np.abs(states - demo[None])[gaps.argmin()])


def test_save_inference_checkpoints_keeps_latest_and_step_copy(tmp_path):
    from script.lingbot_rl_train import checkpoint_dir, save_inference_checkpoints

    model = DiceResidualModel(device="cpu")
    with torch.no_grad():
        model.actor.net[0].bias.fill_(0.25)
    save_inference_checkpoints(tmp_path, 50291, model)
    latest = tmp_path / "residual.pt"
    step = checkpoint_dir(tmp_path, 50291) / "residual.pt"
    assert step == tmp_path / "checkpoints" / "step_050291" / "residual.pt"
    assert latest.is_file()
    assert step.is_file()
    loaded_latest = torch.load(latest, map_location="cpu", weights_only=True)
    loaded_step = torch.load(step, map_location="cpu", weights_only=True)
    assert set(loaded_latest) == {"actor", "critic", "target_critic", "layout"}
    assert set(loaded_step) == {"actor", "critic", "target_critic", "layout"}
    torch.testing.assert_close(loaded_latest["actor"]["net.0.bias"], loaded_step["actor"]["net.0.bias"])
    assert "transformer" not in loaded_latest
    assert "replay" not in loaded_latest


def test_replay_sharpening_reports_per_state_candidate_spread():
    from script.lingbot_rl_train import replay_sharpening
    model = DiceResidualModel(device="cpu")
    torch.manual_seed(0)
    sample = {"s": torch.randn(6, STATE_DIM), "z_all": torch.randn(6, 4, HORIZON, ACTION_DIM),
              "a_base_all": torch.rand(6, 4, HORIZON, ACTION_DIM) * 2 - 1, "is_expert": torch.tensor([0, 0, 1, 0, 0, 1.0])}
    metrics = replay_sharpening(model, sample)
    assert set(metrics) == {"replay_delta_v", "replay_delta_h", "replay_delta_log_std"}
    assert metrics["replay_delta_h"] == 0.0 and metrics["replay_delta_log_std"] == 0.0 and metrics["replay_delta_v"] == 0.0
    assert replay_sharpening(model, {**sample, "is_expert": torch.ones(6)}) == {}


def test_sharpening_metrics_not_residual_rms():
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
    a_base = mask_unused_dof(torch.randn(5, HORIZON, ACTION_DIM))
    residual = model.actor(state, noise, a_base)
    torch.testing.assert_close(residual, torch.zeros(5, HORIZON, ACTION_DIM))
    torch.testing.assert_close(apply_residual(a_base, residual), a_base)


def test_finalize_stores_monte_carlo_return_and_next_candidates():
    buf = ChunkReplay(capacity=32)
    for step, (reward, done) in enumerate(((0, 0), (0, 0), (1, 1))):
        row = _replay_row(reward, done)
        row["s"] = np.full(STATE_DIM, float(step), np.float32)
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
    expected_returns = {0: GAMMA ** 2, 1: GAMMA, 2: 1.0}
    batch = buf.sample(12, expert_ratio=0.0)
    for i in range(batch["mc_return"].shape[0]):
        s_val = int(batch["s"][i][0].item())
        assert float(batch["mc_return"][i].item()) == pytest.approx(expected_returns[s_val])


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


def test_store_rejects_wrong_candidate_count():
    buf = ChunkReplay(capacity=8)
    row = _replay_row(0, 0)
    row["z_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    row["a_base_all"] = np.zeros((2, HORIZON, ACTION_DIM), np.float32)
    with pytest.raises(ValueError, match="candidate"):
        buf.add_online(row)


def test_n_step_target_averages_current_policy_over_candidates():
    torch.manual_seed(11)
    model = DiceResidualModel(device="cpu")
    next_state = torch.randn(2, STATE_DIM)
    z_next_all = torch.randn(2, K_CANDIDATES, HORIZON, ACTION_DIM)
    a_base_next_all = mask_unused_dof(torch.randn(2, K_CANDIDATES, HORIZON, ACTION_DIM))
    reward = torch.tensor([[0.0], [1.0]])
    done = torch.tensor([[0.0], [1.0]])
    n_steps = torch.tensor([[3.0], [1.0]])
    target = model.n_step_target(reward, done, next_state, z_next_all, a_base_next_all, n_steps)
    assert target.shape == (2, 1)
    with torch.no_grad():
        qs = torch.stack([
            model.target_critic(next_state, a_base_next_all[:, k]) for k in range(K_CANDIDATES)
        ], dim=0).mean(dim=0)
    torch.testing.assert_close(target[0], (GAMMA ** 3) * qs[0])
    torch.testing.assert_close(target[1], torch.tensor([1.0]))


def test_predict_action_chunk_executes_highest_q_candidate():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    torch.manual_seed(9)
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates = 4
    policy.eval_plans = 1
    model = DiceResidualModel(device="cpu")
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    policy.residual_model = model
    decoded = {
        "s": torch.zeros(1, STATE_DIM),
        "z": torch.randn(4, HORIZON, ACTION_DIM),
        "a_base": mask_unused_dof(torch.randn(4, HORIZON, ACTION_DIM)),
        "first_chunk": False,
        "latents": torch.zeros(1),
    }
    captured = {}

    def fake_decode(batch, k):
        captured["k"] = k
        return decoded

    def fake_apply(dec, index):
        captured["index"] = index
        return torch.zeros(1, 12, USED_DOF)

    policy.decode_candidates = fake_decode
    policy._apply_residual_choice = fake_apply
    policy.predict_action_chunk({"task": ["t"]})
    state = decoded["s"].expand(4, -1)
    executed = apply_residual(decoded["a_base"], model.actor(state, decoded["z"], decoded["a_base"]))
    expected = int(model.critic(state, executed).reshape(-1).argmax())
    assert captured["k"] == 4
    assert captured["index"] == expected


def test_predict_action_chunk_stays_single_candidate_without_residual_model():
    from script.lingbot_rl_policy import ResidualLingBotPolicy

    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates = 4
    policy.residual_model = None
    captured = {}

    def fake_decode(batch, k):
        captured["k"] = k
        return {"s": torch.zeros(1, STATE_DIM), "z": torch.zeros(1, HORIZON, ACTION_DIM),
                "a_base": torch.zeros(1, HORIZON, ACTION_DIM), "first_chunk": False}

    def fake_apply(dec, index):
        captured["index"] = index
        return torch.zeros(1, 16, USED_DOF)

    policy.decode_candidates = fake_decode
    policy._apply_residual_choice = fake_apply
    policy.predict_action_chunk({"task": ["t"]})
    assert captured["k"] == 1
    assert captured["index"] == 0


def test_parse_run_names_splits_validates_and_rejects_duplicates():
    from script.lingbot_rl_modal import parse_run_names

    assert parse_run_names("a-1,b-2,c-3") == ["a-1", "b-2", "c-3"]
    assert parse_run_names("solo") == ["solo"]
    with pytest.raises(ValueError):
        parse_run_names("a-1,a-1")
    with pytest.raises(ValueError):
        parse_run_names("bad name")
    with pytest.raises(ValueError):
        parse_run_names("")


def test_eval_command_targets_each_run_directory():
    from script.lingbot_rl_modal import eval_command

    command = eval_command({"seed": 42}, "/cache/prep.json", "run-a", resume=True)
    assert command[command.index("--output-dir") + 1] == "/rl/run-a"
    assert command[command.index("--run-name") + 1] == "run-a"
    assert command[command.index("--residual-path") + 1] == "/rl/run-a/residual.pt"
    assert command[-1] == "--resume"
    assert "--resume" not in eval_command({"seed": 42}, "/cache/prep.json", "run-a", resume=False)


def test_eval_stage_fans_out_multiple_run_names(tmp_path, monkeypatch):
    import script.lingbot_rl_modal as module

    called = []

    def fn(name, result=None):
        def remote(*args, **kwargs):
            called.append((name, args))
            return result
        return type("Fn", (), {"remote": staticmethod(remote)})()

    monkeypatch.setattr(module, "prepare", fn("prepare", "/cache/prep.json"))
    monkeypatch.setattr(module, "run_eval", fn("eval", {"a-1": {}, "b-2": {}}))
    monkeypatch.setattr(module, "download_inference", lambda run_name, download_dir: called.append(("download", run_name)) or "out")
    module.main(stage="eval", run_name="a-1,b-2", download_dir=str(tmp_path / "dl"))
    eval_calls = [c for c in called if c[0] == "eval"]
    assert len(eval_calls) == 1
    assert eval_calls[0][1][2] == ["a-1", "b-2"]
    assert [c[1] for c in called if c[0] == "download"] == ["a-1", "b-2"]
    (tmp_path / "dl" / "b-2").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.main(stage="eval", run_name="a-1,b-2", download_dir=str(tmp_path / "dl"))


def test_bin_entropy_counts_distinct_bins_per_coordinate():
    from script.lingbot_rl_sharpen import bin_entropy, log_std

    same = np.zeros((1, 4, HORIZON, ACTION_DIM), dtype=np.float32)
    spread = np.zeros((1, 4, HORIZON, ACTION_DIM), dtype=np.float32)
    spread[0, :, :, :USED_DOF] = np.linspace(-0.9, 0.9, 4)[:, None, None]
    assert bin_entropy(same)[0] == pytest.approx(0.0)
    assert bin_entropy(spread)[0] == pytest.approx(np.log(4))
    assert log_std(spread)[0] > log_std(same)[0]


def test_sharpen_episode_rows_skip_experts_and_split_on_done():
    from script.lingbot_rl_sharpen import episode_rows

    rows = [{"is_expert": 1.0, "n_steps": 1.0}, {"is_expert": 0.0, "n_steps": 3.0}, {"is_expert": 0.0, "n_steps": 2.0},
            {"is_expert": 0.0, "n_steps": 1.0}, {"is_expert": 0.0, "n_steps": 3.0}]
    assert [(e, s) for e, s, _ in episode_rows(rows)] == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_train_loop_runs_with_base_conditioned_residual(tmp_path, monkeypatch):
    out = _smoke_train(tmp_path, monkeypatch, 42, config=RLConfig(residual_input="base").validate())
    assert out["envs"]
    actor = torch.load(tmp_path / "residual.pt", map_location="cpu", weights_only=True)["actor"]
    assert actor["net.0.weight"].shape[1] == STATE_DIM + HORIZON * ACTION_DIM
    assert json.loads((tmp_path / "settings.json").read_text())["recipe"]["residual_input"] == "base"


def _mock_policy(seed=0, blocks=2):
    from types import SimpleNamespace

    from einops import rearrange
    from lerobot.policies.lingbot_va.utils import FlowMatchScheduler

    from script.lingbot_rl_policy import ResidualLingBotPolicy

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(48 * 4, STATE_DIM)
            self.proj_out = torch.nn.Linear(STATE_DIM, 48 * 4)
            self.action_map = torch.nn.Linear(ACTION_DIM, ACTION_DIM)
            self.condition_embedder = SimpleNamespace(text_embedder=torch.nn.Linear(8, STATE_DIM))
            self.blocks = [SimpleNamespace(attn1=SimpleNamespace(attn_caches={
                "pos": {"k": torch.zeros(2, 1, 1, 1), "v": torch.zeros(2, 1, 1, 1)}})) for _ in range(blocks)]
            self.calls = []
            self.create_empty_cache = lambda *args, **kwargs: None

        def forward(self, payload, update_cache=0, cache_name="pos", action_mode=False):
            self.calls.append((action_mode, update_cache))
            x = payload["noisy_latents"]
            bias = self.blocks[0].attn1.attn_caches["pos"]["k"].float().reshape(x.shape[0], -1).mean(dim=1)[:, None, None]
            if action_mode:
                return self.action_map(rearrange(x, "b c f h w -> b (f h w) c")) + bias
            tokens = rearrange(x, "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)", p1=1, p2=2, p3=2)
            hidden = torch.tanh(self.embed(tokens) + payload["timesteps"][:, :1, None] / 1000 + bias)
            out = rearrange(self.proj_out(hidden), "b l (n c) -> b (l n) c", n=4)
            if update_cache == 1:
                for block in self.blocks:
                    cache = block.attn1.attn_caches["pos"]
                    cache["k"] = cache["k"] + hidden.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
            return out

    torch.manual_seed(seed)
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        device="cpu", frame_chunk_size=4, action_dim=ACTION_DIM, action_per_frame=4, num_inference_steps=4,
        video_exec_step=-1, action_num_inference_steps=3, guidance_scale=5.0, action_guidance_scale=1.0,
        patch_size=(1, 2, 2), used_action_channel_ids=list(range(USED_DOF)), camera_layout="width_concat",
        height=32, width=32, obs_cam_keys=("a", "b"), attn_window=30)
    policy.dtype = torch.float32
    policy._use_cfg = True
    policy._frozen = None
    policy._prompt_embeds = torch.randn(1, 2, 8)
    policy._negative_prompt_embeds = torch.randn(1, 2, 8)
    policy._scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    policy._action_scheduler = FlowMatchScheduler(shift=0.05, sigma_min=0.0, extra_one_step=True)
    policy.transformer = Transformer()
    policy._ensure_frozen_modules = lambda: None
    policy._first_chunk = False
    policy._started = True
    policy._exec_step = 5
    policy._frame_st_id = 4
    policy._obs_buffer = []
    policy._executed_actions = None
    policy.residual_model = None
    policy._critic_cache_batches = {}
    return policy


def _reference_infer(policy, video_noise, action_noise):
    import torch.nn.functional as F
    from einops import rearrange
    from lerobot.policies.lingbot_va.utils import data_seq_to_patch

    from script.lingbot_rl_policy import expand_conditional_kv

    cfg = policy.config
    h, w = policy._latent_hw
    latents, actions, k = video_noise.clone(), action_noise.clone(), action_noise.shape[0]
    policy._scheduler.set_timesteps(cfg.num_inference_steps)
    policy._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
    timesteps = F.pad(policy._scheduler.timesteps, (0, 1), value=0)
    action_timesteps = F.pad(policy._action_scheduler.timesteps, (0, 1), value=0)
    for i, t in enumerate(timesteps):
        last = i == len(timesteps) - 1
        payload = policy._prepare_latent_input(latents, None, t, t, None, None, frame_st_id=4)["latent_res_lst"]
        pred = policy.transformer(policy._repeat_input_for_cfg(payload), update_cache=int(last), action_mode=False)
        if not last:
            pred = data_seq_to_patch(cfg.patch_size, pred, cfg.frame_chunk_size, h, w, batch_size=2)
            pred = pred[1:] + cfg.guidance_scale * (pred[:1] - pred[1:])
            latents = policy._scheduler.step(pred, t, latents, return_dict=False)
    snap = policy._snapshot_kv()
    expand_conditional_kv(policy.transformer, k)
    for i, t in enumerate(action_timesteps):
        last = i == len(action_timesteps) - 1
        payload = policy._prepare_latent_input(None, actions, t, t, None, None, frame_st_id=4)["action_res_lst"]
        payload["text_emb"] = policy._prompt_embeds.expand(k, -1, -1).clone()
        payload["grid_id"] = payload["grid_id"][None].repeat(k, 1, 1)
        payload["timesteps"] = payload["timesteps"][None].repeat(k, 1)
        pred = policy.transformer(payload, update_cache=int(last), action_mode=True)
        if not last:
            pred = rearrange(pred, "b (f n) c -> b c f n 1", f=cfg.frame_chunk_size)
            actions = policy._action_scheduler.step(pred, t, actions, return_dict=False)
    actions[:, ~policy._action_mask] *= 0
    policy._restore_kv(snap)
    return actions, latents


def _mock_features(pool):
    return lambda latent: {"pool": pool(latent), "struct": pool(latent)[:, None, None].expand(-1, latent.shape[2], 2, -1),
                           "tokens": pool(latent)[:, None].expand(-1, 32 * latent.shape[2], -1)}


def test_single_plan_infer_reproduces_baseline_bit_for_bit():
    policy = _mock_policy()
    torch.manual_seed(1)
    actions, latents, z, video = policy._infer(None, 4, k=4)
    torch.manual_seed(1)
    video_noise = torch.randn(1, 48, 4, 2, 4)
    action_noise = torch.randn(4, ACTION_DIM, 4, 4, 1)
    ref_actions, ref_latents = _reference_infer(_mock_policy(), video_noise, action_noise)
    assert torch.equal(video, video_noise) and torch.equal(z, action_noise)
    assert torch.equal(actions, ref_actions) and torch.equal(latents, ref_latents)
    given = _mock_policy()._infer(None, 4, video_noise=video_noise, action_noise=action_noise, k=4)
    assert torch.equal(given[0], ref_actions) and torch.equal(given[1], ref_latents)


def test_decode_candidates_keeps_its_keys_on_the_mocked_policy():
    policy = _mock_policy()
    policy._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy._features_from_latent = _mock_features(lambda latent: torch.zeros(1, STATE_DIM))
    decoded = policy.decode_candidates({"task": ["t"]}, k=4)
    assert set(decoded) == {"s", "state_struct", "state_tokens", "state_latent", "z", "a_base", "video_noise", "latents", "first_chunk"}
    assert decoded["a_base"].shape == decoded["z"].shape == (4, HORIZON, ACTION_DIM)
    assert decoded["latents"].shape == decoded["video_noise"].shape == (1, 48, 4, 2, 4)
    assert decoded["first_chunk"] is False and policy._exec_step == 0
    assert [call for call in policy.transformer.calls if call == (False, 1)] == [(False, 1)]


def test_structured_means_split_cameras_and_match_the_pool_without_text():
    from script.lingbot_rl_policy import structured_means
    tokens = torch.zeros(2, 4 * 4 * 8, STATE_DIM)
    grid = tokens.reshape(2, 4, 4, 8, STATE_DIM)
    grid[:, :, :, :4] = 1.0
    grid[:, :, :, 4:] = 3.0
    grid[1, 2] += 10.0
    struct = structured_means(tokens, 4, 4, 8)
    assert struct.shape == (2, 4, 2, STATE_DIM)
    assert torch.equal(struct[0, :, 0], torch.ones(4, STATE_DIM)) and torch.equal(struct[0, :, 1], torch.full((4, STATE_DIM), 3.0))
    assert torch.equal(struct[1, 2, 0], torch.full((STATE_DIM,), 11.0)) and torch.equal(struct[1, 1, 0], torch.ones(STATE_DIM))
    torch.testing.assert_close(pool_critic_state(tokens[:1], torch.full((1, 512, STATE_DIM), 2.0))[0], struct[0].mean(dim=(0, 1)))


def test_features_from_latent_splits_the_real_token_grid_by_camera_in_one_pass():
    policy = _mock_policy()
    latent = torch.randn(1, 48, 4, 2, 4)
    calls = len(policy.transformer.calls)
    feats = policy._features_from_latent(latent)
    assert len(policy.transformer.calls) == calls + 1
    assert torch.equal(feats["struct"][0, :, 0], feats["tokens"][0, 0::2]) and torch.equal(feats["struct"][0, :, 1], feats["tokens"][0, 1::2])


def test_decode_plans_shapes_and_restores_policy_state():
    policy = _mock_policy()
    policy._begin_chunk = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy._pool_from_latent = lambda latent: torch.zeros(1, STATE_DIM)
    with pytest.raises(ValueError, match="exec_step"):
        policy.decode_plans({"task": ["t"]}, plans=2, exec_step=6, k=3)
    before = policy._snapshot_chunk_state()
    out = policy.decode_plans({"task": ["t"]}, plans=2, exec_step=4, k=3)
    assert set(out) == {"s", "z", "a_base", "last_pass_pools", "video_noise", "latents", "first_chunk", "exec_step"}
    assert out["a_base"].shape == (2, 3, HORIZON, ACTION_DIM)
    assert out["z"].shape == (3, HORIZON, ACTION_DIM)
    assert out["last_pass_pools"].shape == (2, STATE_DIM)
    assert out["latents"].shape == (2, 48, 4, 2, 4)
    assert out["video_noise"].shape == (2, 48, 4, 2, 4)
    assert out["exec_step"] == 4 and out["first_chunk"] is False
    assert not torch.equal(out["last_pass_pools"][0], out["last_pass_pools"][1])
    assert torch.count_nonzero(out["a_base"][..., USED_DOF:]) == 0
    for m in range(2):
        single = _mock_policy()
        single.config.video_exec_step = 4
        actions, latents, *_ = single._infer(
            None, 4, video_noise=out["video_noise"][m:m + 1], action_noise=mlp_to_model(out["z"]), k=3)
        assert torch.equal(out["a_base"][m], model_to_mlp(actions).float())
        assert torch.equal(out["latents"][m:m + 1], latents)
    after = policy._snapshot_chunk_state()
    assert after["attrs"] == before["attrs"]
    for old, new in zip(before["kv"], after["kv"]):
        torch.testing.assert_close(new["k"], old["k"])


def test_video_pass_batched_single_plan_equals_video_pass_bit_for_bit():
    torch.manual_seed(2)
    noise = torch.randn(1, 48, 4, 2, 4)
    single, batched = _mock_policy(), _mock_policy()
    timesteps, exec_step = single._video_timesteps()
    batched._video_timesteps()
    x_single, x_batched = noise.clone(), noise.clone()
    for i, t in enumerate(timesteps):
        last = i == len(timesteps) - 1
        advance = not last or exec_step != -1
        x_single = single._video_pass(x_single, t, None, 4, write=last, advance=advance)
        x_batched = batched._video_pass_batched(x_batched, t, None, 4, write=last, advance=advance)
    assert torch.equal(x_single, x_batched)
    for a, b in zip(single._snapshot_kv(), batched._snapshot_kv()):
        assert torch.equal(a["k"], b["k"]) and torch.equal(a["v"], b["v"])


def test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows():
    pool = lambda latent: latent.float().mean(dim=(1, 2, 3, 4)).reshape(-1, 1).expand(-1, STATE_DIM)
    begin = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    looped = _mock_policy()
    looped._begin_chunk, looped._pool_from_latent = begin, pool
    torch.manual_seed(11)
    ref = looped.decode_plans({"task": ["t"]}, plans=2, exec_step=-1, k=3)
    policy = _mock_policy()
    policy._begin_chunk, policy._features_from_latent = begin, _mock_features(pool)
    torch.manual_seed(11)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 1, decode_all=True)
    assert out["plan_index"] == 1 and torch.equal(out["plan_pool"], out["plan_pools"][1:2])
    assert {"state_struct", "state_tokens", "state_latent", "plan_struct_all", "plan_latent_all"} <= set(out)
    assert torch.equal(out["plan_struct"], out["plan_struct_all"][1:2]) and torch.equal(out["plan_latent_all"][1:2], out["latents"])
    torch.testing.assert_close(out["plan_a_base_all"], ref["a_base"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["latents"], ref["latents"][1:2], rtol=1e-5, atol=1e-6)
    cache = policy.transformer.blocks[0].attn1.attn_caches["pos"]["k"]
    assert cache.shape[0] == 2
    winner = _mock_policy()
    winner._begin_chunk, winner._features_from_latent = begin, _mock_features(pool)
    winner.decode_candidates({"task": ["t"]}, k=3, video_noise=ref["video_noise"][1:2], action_noise=mlp_to_model(ref["z"]))
    torch.testing.assert_close(cache, winner.transformer.blocks[0].attn1.attn_caches["pos"]["k"], rtol=1e-5, atol=1e-6)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0)
    assert "plan_a_base_all" not in out and out["a_base"].shape == (3, HORIZON, ACTION_DIM)


def test_select_plan_returns_last_pass_tokens_only_when_asked_without_extra_transformer_work():
    pool = lambda latent: latent.float().mean(dim=(1, 2, 3, 4)).reshape(-1, 1).expand(-1, STATE_DIM)
    begin = lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))
    policy, hooked = _mock_policy(), _mock_policy()
    policy._begin_chunk, policy._features_from_latent = begin, _mock_features(pool)
    hooked._begin_chunk, hooked._features_from_latent = begin, _mock_features(pool)
    torch.manual_seed(3)
    plain = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0)
    torch.manual_seed(3)
    out = hooked.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0, return_tokens=True)
    assert "plan_tokens" not in plain and set(out) == set(plain) | {"plan_tokens"} and hooked.transformer.calls == policy.transformer.calls
    tokens = out["plan_tokens"]
    assert tokens.shape == (2, 8, STATE_DIM) and tokens.dtype == torch.bfloat16 and tokens.device.type == "cpu"
    assert torch.equal(out["plan_pools"], plain["plan_pools"]) and not torch.equal(tokens[0], tokens[1])


def test_predict_action_chunk_ranks_plans_with_the_plan_critic():
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    policy = ResidualLingBotPolicy.__new__(ResidualLingBotPolicy)
    policy.eval_candidates, policy.eval_plans, policy.eval_plan_ranker = 2, 3, "v_plan"
    policy.residual_model = DiceResidualModel(device="cpu", plan_candidates=3)
    captured = {}

    def fake_select(batch, plans, k, rank_fn=None, random_plan=False, decode_all=False, rank_candidates=None):
        captured.update(plans=plans, k=k, rank=rank_fn, random=random_plan, decode_all=decode_all, cands=rank_candidates)
        return {"s": torch.zeros(1, STATE_DIM), "z": torch.zeros(2, HORIZON, ACTION_DIM),
                "a_base": torch.zeros(2, HORIZON, ACTION_DIM), "first_chunk": False, "latents": torch.zeros(1),
                "plan_pool": torch.zeros(1, STATE_DIM), "plan_index": 0}

    policy.select_plan = fake_select
    policy._apply_residual_choice = lambda dec, index: torch.zeros(1, 16, USED_DOF)
    policy.predict_action_chunk({"task": ["t"]})
    assert captured == {"plans": 3, "k": 2, "rank": policy.residual_model.rank_plans, "random": False, "decode_all": False,
                        "cands": None}
    policy.eval_plan_ranker = "q"
    policy.predict_action_chunk({"task": ["t"]})
    assert captured["cands"] == policy.residual_model.rank_candidates
    policy.eval_plan_ranker = "random"
    policy.predict_action_chunk({"task": ["t"]})
    assert captured["random"] and captured["cands"] is None


def test_rank_candidates_returns_the_plan_of_the_best_candidate():
    torch.manual_seed(4)
    model = DiceResidualModel(device="cpu", residual_input="base", plan_candidates=3)
    for parameter in model.critic.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state, pools = torch.randn(1, STATE_DIM), torch.randn(3, STATE_DIM)
    a_base = mask_unused_dof(torch.randn(3, 2, HORIZON, ACTION_DIM))
    z = torch.randn(2, HORIZON, ACTION_DIM)
    best = None
    for m in range(3):
        for j in range(2):
            action = apply_residual(a_base[m, j:j + 1], model.actor(state, z[j:j + 1], a_base[m, j:j + 1]))
            q = float(model.critic(model.critic_state(state, pools[m:m + 1]), action))
            best = (q, m) if best is None or q > best[0] else best
    assert model.rank_candidates(state, pools, a_base, z) == best[1]
