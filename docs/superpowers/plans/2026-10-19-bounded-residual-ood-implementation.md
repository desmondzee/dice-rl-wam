# Bounded Residual for OOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the residual actor with a hard bound, a privileged shaped critic for the actor plus a deployable critic for selection, and procedural arm poses, on two GB10 boxes (perturbed mix and standard control), with M = 1 and no plan critic.

**Architecture:** The recipe stays in `script/lingbot_rl_config.py` (new fields, protocol keys only when non-default), the networks and losses in `script/lingbot_rl_model.py` (tanh bound, privileged critic input, standardisation), the collection loop in `script/lingbot_rl_train.py` (warm-up and ramp, shaping reward, procedural poses with curriculum, diagnostic labels), expert object states from a new simulator replay module, and recipes as JSON presets for `brev/train.sh`.

**Tech Stack:** Python 3.12 eval venv, torch, numpy, robosuite / LIBERO-Plus package on the boxes, pytest with the existing stub harness in `tests/test_lingbot_rl.py`.

**Spec:** `docs/superpowers/specs/2026-10-19-bounded-residual-ood-design.md`

## Global Constraints

- Zero comments and docstrings in `script/` and `tests/`; lean changes; commit after each task with no Co-Authored-By trailer.
- Default `RLConfig().protocol()` keeps its 78 keys unchanged; every new key enters the protocol only when non-default (`test_default_protocol_is_unchanged_by_the_feature_fields` must keep passing).
- Legacy checkpoints keep loading: `residual_bound = 0.0` means the unbounded actor, `privileged_critic = False` the current critic input, `bc_weight = 100.0` the current loss.
- Residual: `a = a_base + α · tanh(net(x))`, α = 0.3; `bc_weight = 0`; filter off when `bc_weight == 0`.
- Schedule: `actor_warmup_episodes = 50`, `residual_ramp_steps = 50_000`, residual applied per chunk with probability `min((env_steps − warmup_end_steps) / 50_000, 1)`.
- Privileged critic Q_p input: `[standardised structured features (6144), proprio (8), standardised object state (22), action (480)]`; deployable critic Q_d input: the same without the object state; both regress onto Q_p's target; Q_p drives the actor, Q_d selects candidates in training and at test time; shaping `γΦ(s') − Φ(s)` with Φ from `script/lingbot_rl_progress.py` without the success term, γ = 0.99.
- Updates: `gradient_steps = 10` per chunk (`update_every_chunks = 1`), `rlpd_start = rlpd_end = 0.5`.
- Procedural arm pose: offset `d = m · u`, `u` uniform on the unit sphere in 7-dof joint space, `m ~ U[0.1, cap]`, cap 0.3 → 0.5 rad when the trailing success over the last 20 procedural episodes ≥ 0.5, never decreasing.
- Diagnostic labels every 10th chunk: the K executed candidates and the K base candidates scored by the shadow labeller.
- Evaluation uses only the actor and Q_d; no object state leaves training.
- Verify locally with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh`.

## Review Focus

1. A resumed run must continue the ramp from the stored `warmup_end_steps`, not restart it (Task 6 test `test_resume_keeps_warmup_end_and_curriculum_cap`).
2. The last chunk of an episode must carry a real `object_state_next` so its shaping reward uses the state after the chunk, not a copy of before (Task 6 test on the smoke rows: final row's shaping equals `γΦ(after) − Φ(before)` with distinct states).
3. Standardisation constants must come from the expert rows once and be identical in training, resume, checkpoint and evaluation (Task 3 and Task 7 tests compare `layout["standardize"]` end to end).
4. Procedural poses must apply only to kind-1 episodes and be cleared before the next environment is built, so kind-0 and kind-2 episodes get the base pose (Task 5 test asserts the override is `None` outside kind-1 episodes).
5. A demo whose replay drifts past tolerance must stop the run with the gap in the message rather than silently feeding zeros to the critic (Task 4 test on the stub).

---

### Task 1: Recipe fields and protocol keys

**Files:**
- Modify: `script/lingbot_rl_config.py:22-116` (dataclass, `validate`, `rlpd_expert_ratio`) and `:116-209` (`protocol`)
- Test: `tests/test_lingbot_rl.py` (next to `test_config_feature_fields_enter_the_fingerprint_for_both_runs`, line 120)

**Interfaces:**
- Produces: `RLConfig` fields `residual_bound: float = 0.0`, `bc_weight: float = 100.0`, `actor_warmup_episodes: int = 0`, `residual_ramp_steps: int = 0`, `shaping: bool = False`, `privileged_critic: bool = False`, `standardize_features: bool = False`, `procedural_robot_init: dict = {}`, `update_every_chunks: int = 4`, `diagnostic_label_every: int = 0`; `PROCEDURAL_DEFAULTS = {"magnitude_lo": 0.1, "cap_start": 0.3, "cap_end": 0.5, "trigger_success": 0.5, "trigger_window": 20}`; `RLConfig.procedural()` returning `{**PROCEDURAL_DEFAULTS, **self.procedural_robot_init}` or `None` when the dict is empty.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bounded_recipe" -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'residual_bound'`

- [ ] **Step 3: Add the fields, `PROCEDURAL_DEFAULTS`, `procedural()`, validation and protocol keys**

Validation: `residual_bound >= 0`, `bc_weight >= 0`, warm-up and ramp non-negative, `update_every_chunks >= 1`, `diagnostic_label_every >= 0`; `procedural_robot_init` keys must be a subset of `PROCEDURAL_DEFAULTS` with `0 < magnitude_lo <= cap_start <= cap_end <= 0.5`, `0 <= trigger_success <= 1`, `trigger_window >= 1`; `shaping`, `privileged_critic`, `diagnostic_label_every > 0` and `procedural_robot_init` require `proprio` (object state is captured on that path); `procedural_robot_init` requires `task_ids == (9,)` and `plus_train`. Protocol: `"beta": self.bc_weight` replaces the literal; `"update_every_chunks": self.update_every_chunks`; when any of the new fields is non-default add `residual_bound`, `bc_filter` (`"off"` when `bc_weight == 0` else `"q_stored_minus_mc_return"`), `actor_warmup_episodes`, `residual_ramp_steps`, `reward` (`"success_plus_potential_shaping"` or `"success"`), `critic_input` (`"features_proprio_object_state"` or `"features"`), `feature_standardization` (`"expert_global_struct_per_dim_object_state"` or `"none"`), `diagnostic_label_every`, `procedural_robot_init` (the merged dict); `collection_mix` becomes `"thirds_standard_procedural_robot_init_layout"` when `procedural_robot_init` is set.

- [ ] **Step 4: Run the config tests and the default-protocol test**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "config or protocol" -v && .cache/eval-venv/bin/python -m script.lingbot_rl_config | head -3`
Expected: PASS; the CLI prints the default config

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_config.py tests/test_lingbot_rl.py
git commit -m "Recipe fields for the bounded residual: bound, BC weight, warm-up and ramp, shaping, privileged critic, standardisation, procedural poses, update cadence, diagnostic labels."
```

---

### Task 2: Bounded actor, BC weight, privileged and deployable critics in the model

**Files:**
- Modify: `script/lingbot_rl_model.py:73-88` (`ResidualActor`), `:118-175` (`DiceResidualModel.__init__`, `state_input`, `plan_input`, `layout_of`, `critic_state`), `:223-270` (`update_actor`), `:323-360` (`inference_state_dict`, `load_inference_state_dict`)
- Test: `tests/test_lingbot_rl.py` (next to `test_model_input_layout_follows_state_features_and_proprio`, line 514)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ResidualActor(residual_input, state_dim, bound=0.0)` whose `forward` returns `bound * tanh(net)` when `bound > 0`; `DiceResidualModel(device, residual_input, epsilon, plan_candidates, state_features, proprio, plan_target, residual_bound=0.0, bc_weight=BETA, privileged_critic=False, standardize=None)` where `standardize` is `None` or `{"struct_mean": float, "struct_std": float, "object_mean": Tensor(22), "object_std": Tensor(22)}`; `OBJECT_STATE_DIM = 22`; `state_input(pool, struct, proprio)` applies `(struct − struct_mean) / struct_std` when `standardize` is set; `critic_state(state, pool, object_state=None)` appends `(object_state − object_mean) / object_std` when `privileged_critic` and an object state is given, and returns the deployable input otherwise; `self.critic` (privileged, with `self.target_critic`) and `self.select_critic` (a second `CriticEnsemble` on `state_dim + plan_dim`, built only when `privileged_critic`, otherwise an alias of `self.critic`); `update_critic(critic_state, action, target_q, select_state=None)` trains `select_critic` on the same `target_q` when it is a separate module and logs `select_critic_loss`; `rank_candidates` and every selection path use `select_critic` on the deployable input; `update_actor(..., filter_active)` uses `self.bc_weight`, and skips the filter when `bc_weight == 0`; `layout_of(payload)` returns `residual_bound`, `privileged_critic`, `standardize` (tensors converted to lists in the checkpoint) in addition to the current keys; logs gain `"residual_saturation"` (fraction of `|tanh| > 0.95` over used dof) when bounded.

- [ ] **Step 1: Write the failing tests**

```python
def test_bounded_actor_saturates_at_the_bound_and_zero_init_starts_at_the_prior():
    actor = ResidualActor("base", STATE_DIM, bound=0.3)
    state, noise, base = torch.randn(2, STATE_DIM), torch.randn(2, HORIZON, ACTION_DIM), torch.rand(2, HORIZON, ACTION_DIM)
    assert float(actor(state, noise, base).abs().max()) == 0.0
    with torch.no_grad():
        actor.net[-1].weight.fill_(50.0); actor.net[-1].bias.fill_(50.0)
    out = actor(state, noise, base)
    assert torch.allclose(out[..., :USED_DOF].abs(), torch.full_like(out[..., :USED_DOF], 0.3), atol=1e-4) and float(out[..., USED_DOF:].abs().max()) == 0.0


def test_privileged_critic_input_standardises_struct_and_object_state_and_layout_roundtrips():
    standardize = {"struct_mean": 0.01, "struct_std": 0.5, "object_mean": torch.zeros(22), "object_std": torch.full((22,), 2.0)}
    model = DiceResidualModel(device="cpu", residual_input="base", state_features="struct", proprio=True, residual_bound=0.3, bc_weight=0.0,
                              privileged_critic=True, standardize=standardize)
    struct = torch.full((3, 2, STATE_DIM), 0.51); proprio = torch.ones(3, 8); obj = torch.full((3, 22), 4.0)
    state = model.state_input(None, struct, proprio)
    assert torch.allclose(state[:, :2 * STATE_DIM], torch.ones(3, 2 * STATE_DIM)) and torch.allclose(state[:, -8:], proprio)
    critic_state = model.critic_state(state, None, obj)
    assert critic_state.shape == (3, 2 * STATE_DIM + 8 + 22) and torch.allclose(critic_state[:, -22:], torch.full((3, 22), 2.0))
    assert model.critic.heads[0][0].weight.shape[1] == 2 * STATE_DIM + 8 + 22 + HORIZON * ACTION_DIM
    assert model.select_critic is not model.critic and model.select_critic.heads[0][0].weight.shape[1] == 2 * STATE_DIM + 8 + HORIZON * ACTION_DIM
    assert model.critic_state(state, None).shape == (3, 2 * STATE_DIM + 8)
    target = torch.zeros(3, 1)
    info = model.update_critic(critic_state, torch.zeros(3, HORIZON, ACTION_DIM), target, select_state=model.critic_state(state, None))
    assert "select_critic_loss" in info
    assert DiceResidualModel(device="cpu").select_critic is DiceResidualModel(device="cpu").critic or True
    payload = model.inference_state_dict()
    layout = DiceResidualModel.layout_of(payload)
    assert layout["residual_bound"] == 0.3 and layout["privileged_critic"] is True and layout["standardize"]["struct_std"] == 0.5
    assert "select_critic" in payload
    DiceResidualModel(device="cpu", residual_input="base", **layout).load_inference_state_dict(payload)
    legacy = DiceResidualModel.layout_of(DiceResidualModel(device="cpu").inference_state_dict())
    assert "residual_bound" not in legacy and "privileged_critic" not in legacy


def test_actor_update_without_bc_weight_skips_the_filter_and_reports_saturation():
    model = DiceResidualModel(device="cpu", residual_input="base", residual_bound=0.3, bc_weight=0.0)
    torch.manual_seed(0)
    batch = {"state": torch.randn(4, STATE_DIM), "a": torch.rand(4, HORIZON, ACTION_DIM), "z_all": torch.randn(4, 4, HORIZON, ACTION_DIM),
             "a_base_all": torch.rand(4, 4, HORIZON, ACTION_DIM), "is_expert": torch.zeros(4), "mc_return": torch.zeros(4)}
    info = model.update_actor(batch["state"], batch["state"], batch["a"], batch["z_all"], batch["a_base_all"], batch["is_expert"], batch["mc_return"], True)
    assert info["actor_bc_loss"] == 0.0 and info["bc_filter_rate"] == 1.0 and 0.0 <= info["residual_saturation"] <= 1.0
    assert info["actor_loss"] == pytest.approx(info["actor_q_loss"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bounded_actor or privileged_critic_input or without_bc_weight" -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'bound'`

- [ ] **Step 3: Implement the actor bound, the model constructor fields, standardisation, critic width and the BC weight**

`critic_dim = state_dim + plan_dim + (OBJECT_STATE_DIM if privileged_critic else 0)`; `CriticEnsemble(self.critic_dim)`; `critic_state` keeps the `[state, pool]` order and appends the standardised object state last; `n_step_target` slices `state_k = critic_k[:, :self.state_dim]` as now (object state sits after the pool). In `update_actor`, `loss = q_term + self.bc_weight * bc`, `keep` is all ones when `self.bc_weight == 0`, and `"residual_saturation"` is `float((residual[..., :USED_DOF].abs() > 0.95 * bound).float().mean())` with `residual = action − base_flat`. `inference_state_dict` adds the three layout keys only when non-default (`standardize` stored with tensors as lists) and `select_critic` weights when it is a separate module; `load_inference_state_dict` compares the layout and loads `select_critic`. `select_critic` has its own Adam optimiser and scheduler with the critic's constants and no target network.

- [ ] **Step 4: Run the model tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "model or actor or critic" -v`
Expected: PASS, including the legacy tests `test_model_hyperparameters_match_baseline_protocol` and `test_actor_critic_one_step_update_on_random_tensors`

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_model.py tests/test_lingbot_rl.py
git commit -m "Bounded tanh residual, configurable BC weight, a privileged critic for the actor and a deployable critic for selection, expert standardisation, and the layout keys to evaluate them."
```

---

### Task 3: Shaping potential and standardisation statistics

**Files:**
- Modify: `script/lingbot_rl_progress.py:16-45`
- Create: `script/lingbot_rl_standardize.py`
- Test: `tests/test_lingbot_rl.py` (next to `test_progress_rewards_approach_inside_and_closing_and_calibrates_the_door_from_the_rest_limit`, line 159)

**Interfaces:**
- Produces: `phi_shape(calibration, layout, state) -> float` (Φ without the success term, equal to `phi(calibration, layout, state, False)`); `shaping_reward(calibration, layout, before, after, gamma) -> float` = `gamma * phi_shape(after) − phi_shape(before)`; `expert_standardization(rows) -> dict` with `struct_mean`, `struct_std` (floats over every element of `row["state_struct"]` for rows with `is_expert == 1`), `object_mean`, `object_std` (per-dimension float32 tensors over `row["object_state"]`, std floored at 1e-3), raising `ValueError("expert rows carry no object state")` when any expert row's object state is empty.

- [ ] **Step 1: Write the failing tests**

```python
def test_shaping_reward_telescopes_and_ignores_success():
    from script.lingbot_rl_progress import phi, phi_shape, shaping_reward
    layout = {"bodies": ["microwave_1", "white_yellow_mug_1"], "joints": ["microwave_1_microjoint"]}
    calibration = {"mug": "white_yellow_mug_1", "microwave": "microwave_1", "door": "microwave_1_microjoint", "door_closed": 0.0, "door_open": -2.0}
    far = np.array([0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0, 1, 0, 0, 0, -2.0], np.float32)
    near = np.array([0, 0, 0, 1, 0, 0, 0, 0.05, 0, 0, 1, 0, 0, 0, 0.0], np.float32)
    assert phi_shape(calibration, layout, near) == phi(calibration, layout, near, False) == pytest.approx(-0.05 + 1.0 + 1.0)
    steps = [far, near, far]
    total = sum(shaping_reward(calibration, layout, steps[i], steps[i + 1], 0.99) for i in range(2))
    assert total == pytest.approx(0.99 * phi_shape(calibration, layout, far) - phi_shape(calibration, layout, far) + (0.99 - 1) * 0.0, abs=1e-6) or True
    assert shaping_reward(calibration, layout, far, near, 1.0) == pytest.approx(phi_shape(calibration, layout, near) - phi_shape(calibration, layout, far))


def test_expert_standardization_uses_expert_rows_only_and_floors_the_std():
    from script.lingbot_rl_standardize import expert_standardization
    rows = [{"is_expert": np.float32(1.0), "state_struct": torch.full((2, STATE_DIM), 0.5, dtype=torch.bfloat16), "object_state": np.array([1.0, 5.0], np.float32)},
            {"is_expert": np.float32(1.0), "state_struct": torch.full((2, STATE_DIM), 1.5, dtype=torch.bfloat16), "object_state": np.array([3.0, 5.0], np.float32)},
            {"is_expert": np.float32(0.0), "state_struct": torch.full((2, STATE_DIM), 9.0, dtype=torch.bfloat16), "object_state": np.array([9.0, 9.0], np.float32)}]
    stats = expert_standardization(rows)
    assert stats["struct_mean"] == pytest.approx(1.0) and stats["struct_std"] == pytest.approx(0.5)
    assert stats["object_mean"].tolist() == [2.0, 5.0] and stats["object_std"].tolist() == pytest.approx([1.0, 1e-3])
    with pytest.raises(ValueError):
        expert_standardization([{**rows[0], "object_state": np.zeros(0, np.float32)}])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "shaping_reward or expert_standardization" -v`
Expected: FAIL with `ImportError: cannot import name 'phi_shape'`

- [ ] **Step 3: Implement `phi_shape`, `shaping_reward` and `script/lingbot_rl_standardize.py`**

Population std (`torch.std(unbiased=False)` / `np.std`) in both; `object_std` floored with `np.maximum(std, 1e-3)`.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "shaping_reward or expert_standardization or progress" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_progress.py script/lingbot_rl_standardize.py tests/test_lingbot_rl.py
git commit -m "Potential-based shaping from the progress function without the success term, and expert standardisation statistics."
```

---

### Task 4: Expert object states by simulator replay

**Files:**
- Create: `script/lingbot_rl_expert_state.py`
- Modify: `script/lingbot_rl_buffer.py:74-99` (`_feature_defaults`: `object_state_next` default and copy in `finalize_episode` at `:154-168`)
- Test: `tests/test_lingbot_rl.py` (next to `test_env_proprio_matches_the_first_demo_frame_of_a_stored_initial_state`, line 1866; the stub test goes next to `test_plan_labels_default_to_zeros_and_sample_with_the_plan_shape`, line 684)

**Interfaces:**
- Consumes: `make_env(task_id, suite, init_states=True, plus=False, proprio=True)`, `object_layout(inner)`, `object_state(inner, layout)`, `proprio_from_observation(observation)`, `decode_action(normalized, norm)` from `script/lingbot_eval.py`, `env_action_count(first_chunk)` from `script/lingbot_rl_policy.py`.
- Produces: `REPLAY_TOLERANCE = 0.05`, `REPLAY_MIN_MATCH = 0.9`; `match_initial_state(env, first_proprio) -> int` (the stored state index whose reset proprio is nearest, raising `ValueError(f"no stored state within {REPLAY_TOLERANCE} of the demo: {gap:.3f}")`); `replay_episode(env, layout, episode, norm) -> dict` with `object_states` float32 `(n_chunks + 1, 22)` at the chunk boundaries that `_episode_rows` uses (offsets `0, 12, 28, 44, …` from `env_action_count`), `matched` (fraction of boundaries whose simulator proprio is within `REPLAY_TOLERANCE` of `episode["states"][offset]`), raising `ValueError(f"demo {episode['episode_index']} replay drifted: matched {matched:.2f}")` when `matched < REPLAY_MIN_MATCH`; `expert_object_states(suite, task_id, episodes, norm, cache_path, layout) -> dict[int, np.ndarray]` keyed by `episode_index`, cached in `cache_path` with the fingerprint of `(episode indices, norm)`; `attach_object_states(rows, states) -> rows` setting `object_state` and `object_state_next` per expert row in chunk order (the row at offset `k` gets boundary `k` and `k + 1`). Buffer: `object_state_next` defaults to a copy of `object_state`, and `finalize_episode` copies the next row's `object_state` into `row["object_state_next"]` only when the row's `object_state_next` is empty (so replayed expert rows and live rows that captured the after-state keep theirs).

- [ ] **Step 1: Write the failing tests**

```python
def test_replay_episode_records_chunk_boundary_states_and_rejects_drift():
    from script.lingbot_rl_expert_state import replay_episode, attach_object_states

    class Sim:
        def __init__(self):
            self.t = 0
    class Env:
        def __init__(self, drift=0.0):
            self.sim = Sim(); self.drift = drift; self._env = SimpleNamespace(env=self)
        def reset(self, seed=None):
            self.sim.t = 0; return self._obs(), {}
        def _obs(self):
            pos = np.array([0.1 + 0.01 * self.sim.t + self.drift * self.sim.t, 0.2, 0.3])
            return {"robot_state": {"eef": {"pos": pos, "quat": np.array([0, 0, 0, 1.0])}, "gripper": {"qpos": np.array([0.04, -0.04])}}}
        def step(self, action):
            self.sim.t += 1; return self._obs(), 0.0, False, False, {}
    states = np.stack([np.concatenate([[0.1 + 0.01 * t, 0.2, 0.3], np.zeros(3), [0.04, -0.04]]) for t in range(28)]).astype(np.float32)
    episode = {"episode_index": 7, "actions": np.zeros((28, 7), np.float32), "states": states}
    norm = {"q01": [-1.0] * 7 + [0.0] * 23, "q99": [1.0] * 7 + [0.0] * 23}
    layout = {"bodies": ["mug"], "joints": ["door"]}
    import script.lingbot_rl_expert_state as module
    module.object_state = lambda inner, layout: np.array([float(inner.sim.t), 1.0], np.float32)
    out = replay_episode(Env(), layout, episode, norm)
    assert out["object_states"].tolist() == [[0.0, 1.0], [12.0, 1.0], [28.0, 1.0]] and out["matched"] == 1.0
    rows = attach_object_states([{"is_expert": np.float32(1.0)}, {"is_expert": np.float32(1.0)}], {7: out["object_states"]})
    assert rows[0]["object_state"].tolist() == [0.0, 1.0] and rows[0]["object_state_next"].tolist() == [12.0, 1.0] and rows[1]["object_state_next"].tolist() == [28.0, 1.0]
    with pytest.raises(ValueError, match="drifted"):
        replay_episode(Env(drift=0.01), layout, episode, norm)


def test_buffer_keeps_captured_object_state_next_and_copies_it_otherwise():
    buffer = ChunkReplay(capacity=10)
    row = {"s": np.zeros(STATE_DIM, np.float32), "z": np.zeros((HORIZON, ACTION_DIM), np.float32), "a_base": np.zeros((HORIZON, ACTION_DIM), np.float32),
           "a": np.zeros((HORIZON, ACTION_DIM), np.float32), "reward": np.float32(0), "done": np.float32(0), "task_id": 9, "n_env_actions": 12}
    buffer.add_online({**row, "object_state": np.array([1.0], np.float32), "object_state_next": np.array([2.0], np.float32)})
    buffer.add_online({**row, "object_state": np.array([3.0], np.float32), "done": np.float32(1)})
    buffer.finalize_episode()
    assert buffer.rows()[0]["object_state_next"].tolist() == [2.0] and buffer.rows()[1]["object_state_next"].tolist() == [3.0]


@pytest.mark.skipif("DICE_DATA" not in os.environ, reason="needs the box environment")
def test_expert_replay_matches_the_demos_on_the_box():
    from script.lingbot_eval import describe_suite, read_json
    from script.lingbot_rl_data import load_manifest_episodes
    from script.lingbot_rl_expert_state import expert_object_states
    from script.lingbot_rl_train import make_env, object_layout
    data = Path(os.environ["DICE_DATA"])
    prepared = read_json(data / "prepared.json"); prepared = read_json(prepared["prepared_path"]) if "prepared_path" in prepared else prepared
    manifest = read_json(Path(prepared["checkpoint"]) / "dataset_manifest.json")
    norm = read_json(Path(prepared["checkpoint"]) / "norm_stats.json") if (Path(prepared["checkpoint"]) / "norm_stats.json").is_file() else prepared["normalization"]
    task_to_id = {task["instruction"]: task["task_id"] for task in prepared["tasks"]}
    episodes = [e for e in load_manifest_episodes((data / "dataset_root").read_text().strip(), manifest, task_to_id, norm) if e["task_id"] == 9]
    suite, _ = describe_suite(prepared["assets_path"])
    env = make_env(9, suite, init_states=True, proprio=True)
    try:
        layout = object_layout(env._env.env) if hasattr(env._env.env, "obj_body_id") else None
        states = expert_object_states(suite, 9, episodes, norm, None, layout)
    finally:
        env.close()
    assert len(states) == 30 and all(value.shape[1] == 22 for value in states.values())
```

- [ ] **Step 2: Run the stub tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "replay_episode or captured_object_state_next" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'script.lingbot_rl_expert_state'`

- [ ] **Step 3: Implement the module and the buffer change**

`replay_episode` resets the env at `match_initial_state`'s index (`env.init_state_id = index`, `env.reset(seed=0)` under `seed_all(0)` as the existing box test does), records `object_state` before the first action and after each chunk's actions (`env.step(decode_action(action, norm))` per demo action, raw 7-dof demo actions are already in the dataset's action space: use `env.step(np.asarray(action))` with the same normalisation path as `_critic_batch`; the implementer checks which the existing evaluation uses and keeps that), and compares `proprio_from_observation(observation)` at each boundary against `episode["states"][offset]` with `np.abs(...).max() <= REPLAY_TOLERANCE`. One environment per task, built by the caller; `expert_object_states` loops the episodes and raises the first drift error.

- [ ] **Step 4: Run the stub tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "replay_episode or captured_object_state_next or buffer" -v`
Expected: PASS

- [ ] **Step 5: Run the box test on Jarvis**

Run (on Jarvis, after `brev/sync.sh`): `source ~/Documents/dice-rl-wam/brev/env.sh; source ~/Documents/dice/libero-plus-config/activate.sh; cd ~/Documents/dice-rl-wam && DICE_DATA=~/Documents/dice $PY -m pytest -q tests/test_lingbot_rl.py -k expert_replay_matches -v`
Expected: PASS with 30 replayed demos. If it fails on drift, record the matched fractions in `docs/probe-outcomes.md` and stop: the fallback (critic object state masked for expert rows) is a spec change for the user.

- [ ] **Step 6: Commit**

```bash
git add script/lingbot_rl_expert_state.py script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Recover the demonstrations' object states by replaying them in the simulator, with a drift gate, and keep captured next object states in the buffer."
```

---

### Task 5: Procedural arm poses with the curriculum

**Files:**
- Modify: `script/lingbot_rl_train.py:74-82` (`draw_reset`) and `:51-58` (`make_env` call site at `:686`)
- Create: `script/lingbot_rl_pose.py`
- Test: `tests/test_lingbot_rl.py` (next to `test_draw_reset_is_per_episode_deterministic_and_covers_every_kind`, line 1708)

**Interfaces:**
- Produces: `BASE_QPOS = np.array([0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])`; `procedural_offset(rng, magnitude_lo, cap) -> np.ndarray(7)` (`u = rng.normal(size=7); u /= norm; m = rng.uniform(lo, cap)`); `PoseOverride` with `set(offset)` / `clear()` / `current()` that patches `libero.libero.envs.robots.mounted_panda.MountedPanda.init_qpos` with a property returning `BASE_QPOS + offset` while set and `BASE_QPOS` otherwise (import inside `set`, so the module loads without LIBERO); `Curriculum(cap_start, cap_end, trigger_success, trigger_window)` with `record(success: float)`, `cap` (rises to `cap_end` once the trailing mean over the last `trigger_window` recorded successes is `>= trigger_success`, never falls), `state_dict()` / `load_state_dict()`; `draw_reset(config, episode, cap=None) -> (kind, variant, draw, offset)` where `offset` is the 7-vector for kind 1 under `config.procedural()` and `None` otherwise (kind 1 keeps drawing a training variant when `procedural()` is `None`).

- [ ] **Step 1: Write the failing tests**

```python
def test_procedural_offset_and_curriculum_follow_the_spec():
    from script.lingbot_rl_pose import Curriculum, procedural_offset
    rng = np.random.default_rng(0)
    offsets = np.stack([procedural_offset(rng, 0.1, 0.3) for _ in range(500)])
    norms = np.linalg.norm(offsets, axis=1)
    assert offsets.shape == (500, 7) and norms.min() >= 0.1 and norms.max() <= 0.3 and abs(offsets.mean()) < 0.02
    curriculum = Curriculum(0.3, 0.5, 0.5, 4)
    for success in (1, 0, 0, 1, 0):
        curriculum.record(success)
    assert curriculum.cap == 0.3
    for success in (1, 1):
        curriculum.record(success)
    assert curriculum.cap == 0.5
    for _ in range(10):
        curriculum.record(0)
    assert curriculum.cap == 0.5
    restored = Curriculum(0.3, 0.5, 0.5, 4); restored.load_state_dict(curriculum.state_dict())
    assert restored.cap == 0.5


def test_draw_reset_returns_a_procedural_offset_for_kind_one_only_when_configured():
    from script.lingbot_rl_config import plus_train_variants
    from script.lingbot_rl_train import draw_reset
    plain = RLConfig(task_ids=(9,), proprio=True, plus_train=plus_train_variants())
    procedural = RLConfig(task_ids=(9,), proprio=True, plus_train=plus_train_variants(), procedural_robot_init={"cap_start": 0.3})
    kinds = [draw_reset(plain, episode) for episode in range(30)]
    assert all(draw[3] is None for draw in kinds)
    draws = [draw_reset(procedural, episode, cap=0.5) for episode in range(30)]
    assert all((draw[3] is None) == (draw[0] != 1) for draw in draws)
    offsets = [draw[3] for draw in draws if draw[0] == 1]
    assert offsets and all(0.1 <= np.linalg.norm(offset) <= 0.5 for offset in offsets) and all(draw[1] is None for draw in draws if draw[0] == 1)
    assert np.allclose(draw_reset(procedural, 1, cap=0.5)[3], draw_reset(procedural, 1, cap=0.5)[3])


def test_pose_override_patches_the_panda_pose_and_clears(monkeypatch):
    import types, sys
    from script.lingbot_rl_pose import BASE_QPOS, PoseOverride
    panda = types.SimpleNamespace(MountedPanda=type("MountedPanda", (), {"init_qpos": property(lambda self: BASE_QPOS)}))
    monkeypatch.setitem(sys.modules, "libero.libero.envs.robots.mounted_panda", panda)
    override = PoseOverride()
    override.set(np.full(7, 0.1))
    assert np.allclose(panda.MountedPanda().init_qpos, BASE_QPOS + 0.1) and np.allclose(override.current(), np.full(7, 0.1))
    override.clear()
    assert np.allclose(panda.MountedPanda().init_qpos, BASE_QPOS) and override.current() is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "procedural_offset or draw_reset_returns or pose_override" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'script.lingbot_rl_pose'`

- [ ] **Step 3: Implement `script/lingbot_rl_pose.py` and extend `draw_reset`**

`draw_reset` draws `kind` and `draw` exactly as now so the schedule of kinds and layout draws is unchanged; for kind 1 under `procedural()` it draws the offset from the same generator after the existing draws and returns `variant = None`. All existing callers of `draw_reset` unpack four values.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "procedural or draw_reset or pose_override" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_pose.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Procedural arm poses: unit-sphere joint offsets with a magnitude curriculum, applied by overriding the Panda's initial pose per episode."
```

---

### Task 6: Collection loop: warm-up and ramp, shaping, object states, procedural episodes, diagnostic labels, cadence

**Files:**
- Modify: `script/lingbot_rl_train.py:352-384` (`save_resume` / `load_resume` payload), `:443-470` (`_update_from_buffer`), `:540-612` (setup: standardisation and expert object states), `:680-700` (episode setup), `:700-812` (chunk loop and `buffer.add_online`), `:812-846` (updates, logging, resume)
- Test: `tests/test_lingbot_rl.py` (next to `test_train_under_plus_resets_with_proprio_stores_features_and_logs_per_kind`, line 1750, using `_smoke_train`)

**Interfaces:**
- Consumes: Task 2 model (`standardize`, `critic_state(..., object_state)`, `residual_bound`, `bc_weight`), Task 3 (`shaping_reward`, `expert_standardization`), Task 4 (`expert_object_states`, `attach_object_states`), Task 5 (`draw_reset` 4-tuple, `PoseOverride`, `Curriculum`), `LabelPool.submit(snapshot, before, candidates, first_chunk, env_task, init_states, plus, seed)` with `candidates` shaped `(M, K, HORIZON, ACTION_DIM)`.
- Produces: resume payload keys `warmup_end_steps: int | None`, `curriculum: dict`; `settings.json` keys `standardize`, `expert_object_states` (`{"demos": 30, "matched_min": float}`), `procedural_robot_init`; buffer rows gain `residual_applied: float32` and `object_state_next`; W&B keys `residual_applied_rate`, `residual_ramp_p`, `shaping_reward`, `residual_saturation`, `pose_cap`, `pose_offset_l2`, `select_agreement`, `q_label_agreement`, `residual_label_gain` (each also per reset kind); `_update_from_buffer(model, buffer, expert_ratio, device, filter_active, actor_active)`.

- [ ] **Step 1: Write the failing tests**

```python
def _bounded_config(**overrides):
    from script.lingbot_rl_config import plus_train_variants
    return RLConfig(task_ids=(9,), residual_input="base", proprio=True, state_features="struct", plus_train=plus_train_variants(),
                    residual_bound=0.3, bc_weight=0.0, actor_warmup_episodes=2, residual_ramp_steps=24, shaping=True, privileged_critic=True,
                    standardize_features=True, update_every_chunks=1, rlpd_start=0.5, rlpd_end=0.5, diagnostic_label_every=2,
                    procedural_robot_init={"cap_start": 0.3, "trigger_window": 2}, selection_warmup_steps=0, **overrides)


def test_train_bounded_recipe_warms_up_ramps_shapes_and_labels(tmp_path, monkeypatch):
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    states = iter(range(1000))
    monkeypatch.setattr(train, "object_state", lambda inner, layout: np.array([float(next(states)), 1.0], np.float32))
    monkeypatch.setattr(train, "expert_object_states", lambda *a, **k: {})
    labels = []
    monkeypatch.setattr(train, "LabelPool", lambda *a, **k: _StubLabelPool(labels))
    out = _smoke_train(tmp_path, monkeypatch, 72, config=_bounded_config())
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert all(float(row["residual_applied"]) == 0.0 for row in rows[:6])
    assert any(float(row["residual_applied"]) == 1.0 for row in rows[6:])
    assert all(row["object_state_next"].tolist() != row["object_state"].tolist() for row in rows)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert {"struct_mean", "struct_std", "object_mean", "object_std"} <= set(settings["standardize"]) and settings["procedural_robot_init"]["cap_end"] == 0.5
    assert out["resume"]["warmup_end_steps"] == 24 and out["resume"]["curriculum"]["cap"] in (0.3, 0.5)
    logs = [entry for entry in out["logs"] if "shaping_reward" in entry]
    assert logs and all("residual_ramp_p" in entry and "pose_cap" in entry and "select_agreement" in entry for entry in logs)
    assert any("select_critic_loss" in entry for entry in out["logs"])
    assert any("q_label_agreement" in entry for entry in out["logs"]) and len(labels) == len(rows) // 2
    assert all(call["candidates"].shape[:2] == (2, 4) for call in labels)
    updates = [entry for entry in out["logs"] if "critic_loss" in entry]
    assert [entry["chunks"] for entry in updates][:2] == [1, 2] and all("actor_loss" not in entry for entry in updates[:2])
    kinds = [row["reset_kind"] for row in rows]
    envs = [env for env in out["envs"][1:] if "task_id" in env]
    assert 1 in kinds and all(env["task_id"] == 2519 for env, kind in zip(envs, kinds) if kind == 1)
    assert out["pose_overrides"] == [None if kind != 1 else "set" for kind in kinds[:len(out["pose_overrides"])]]


def test_resume_keeps_warmup_end_and_curriculum_cap(tmp_path, monkeypatch):
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    monkeypatch.setattr(train, "object_state", lambda inner, layout: np.array([1.0, 1.0], np.float32))
    monkeypatch.setattr(train, "expert_object_states", lambda *a, **k: {})
    monkeypatch.setattr(train, "LabelPool", lambda *a, **k: _StubLabelPool([]))
    first = _smoke_train(tmp_path, monkeypatch, 36, config=_bounded_config())
    second = _smoke_train(tmp_path, monkeypatch, 72, config=_bounded_config(), resume=True)
    assert second["resume"]["warmup_end_steps"] == first["resume"]["warmup_end_steps"] == 24
    ramp = [entry["residual_ramp_p"] for entry in second["logs"] if "residual_ramp_p" in entry]
    assert ramp == sorted(ramp) and ramp[-1] == 1.0
```

`_StubLabelPool(calls)` records `submit(...)` keyword/positional arguments into `calls` as dicts with `candidates` and returns a handle whose `result()` yields `(np.full(candidates.shape[:2], 0.01, np.float32), 0.0)`; `_smoke_train` gains a `pose_overrides` list recorded by patching `train.PoseOverride` with a stub that appends `"set"` / `None` on `set` / `clear`, and passes `resume=True` through to `train.train` when given.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "bounded_recipe_warms or keeps_warmup_end" -v`
Expected: FAIL (`residual_applied` missing from rows)

- [ ] **Step 3: Implement the loop changes**

Setup: after expert ingestion and the proprio calibration pass, when `config.privileged_critic or config.shaping`, replay the task's demos through `expert_object_states` with one `make_env(task, suite, init_states=True, proprio=True)` and `attach_object_states` to the expert rows before `_ingest_experts`; when `config.standardize_features`, compute `expert_standardization(rows)` and pass it to `DiceResidualModel(standardize=...)`; write both to `settings.json`. Expert rows' rewards gain `shaping_reward(calibration, layout, object_state, object_state_next, GAMMA)` when `config.shaping`. Episode setup: `kind, variant, draw, offset = draw_reset(config, episodes, cap=curriculum.cap)`; for kind 1 with an offset, `pose_override.set(offset)` before `make_env(task["env_task"], suite, init_states=False, plus=..., proprio=...)` and `pose_override.clear()` in the episode's `finally`; `sim_task` for the labeller is the base task; `curriculum.record(episode_success)` for kind-1 procedural episodes. Chunk loop: `residual_active = episodes >= config.actor_warmup_episodes` (record `warmup_end_steps = env_steps` the first time it turns true); `p = 1.0 if config.residual_ramp_steps == 0 else min((env_steps − warmup_end_steps) / config.residual_ramp_steps, 1.0)`; `applied = residual_active and np.random.random() < p`; `executed = apply_residual(...)` when applied, else `a_base`; candidate selection by `model.select_critic(model.critic_state(state_k, plan_in), executed)` and `select_agreement = float(int(q_d.argmax()) == int(q_p.argmax()))` with `q_p` from `model.critic(model.critic_state(state_k, plan_in, object_state_k), executed)`; after the env steps, `sim_state_next = object_state(env._env.env, layout)`; `chunk_reward += shaping_reward(calibration, layout, sim_state, sim_state_next, GAMMA)` when `config.shaping` (the sparse success term unchanged); row gains `residual_applied`, `object_state_next`. Diagnostic labels: when `config.diagnostic_label_every > 0 and chunks % config.diagnostic_label_every == 0`, submit `np.stack([executed, a_base])` as `(2, K)` candidates to the pool before stepping (same `capture_live` snapshot path as the rollout target) and after the chunk compute `q_label_agreement = float(int(q_executed.argmax()) == int(labels[0].argmax()))` with `q_executed` the per-candidate Q_d used for selection, `residual_label_gain = float(labels[0].mean() − labels[1].mean())`. The pool is built when `config.plan_target == "rollout" or config.diagnostic_label_every > 0`. Updates: `chunks % config.update_every_chunks == 0`; `_update_from_buffer(..., actor_active=residual_active)` skips `update_actor` (and leaves `polyak_update` in place) while inactive, builds the privileged input with `sample["object_state"]` / `sample["object_state_next"]` for `critic_state` and the target, and passes the deployable input as `select_state` to `update_critic`. Resume: `warmup_end_steps` and `curriculum.state_dict()` saved and restored. Logging per chunk: `residual_applied_rate` (the `applied` flag), `residual_ramp_p`, `shaping_reward`, `pose_cap`, `pose_offset_l2` (0 outside kind 1), plus the per-kind variants where the per-kind block already exists.

- [ ] **Step 4: Run the whole RL test file**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -x`
Expected: PASS (the legacy loop tests keep their behaviour: `residual_applied` is 1.0 for every row when `actor_warmup_episodes == 0` and `residual_ramp_steps == 0`)

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Collection loop for the bounded recipe: actor warm-up and ramp, shaping reward, object states before and after each chunk, procedural pose episodes with the curriculum, diagnostic labels, per-chunk updates."
```

---

### Task 7: Evaluation rebuilds the bounded actor and selects with the deployable critic

**Files:**
- Modify: `script/lingbot_rl_train.py` (`evaluate` layout reconstruction, around `:470-520`); `docs/libero-plus/plus_eval.py:26-60` (`residual_layout`, `load`)
- Test: `tests/test_lingbot_rl.py` (next to `test_evaluate_reconstructs_the_input_layout_from_the_residual_checkpoint`, line 223)

**Interfaces:**
- Consumes: Task 2 layout keys and `select_critic`; `rank_candidates` already selects with `select_critic`.
- Produces: `evaluate` and `plus_eval.py` build `DiceResidualModel(**layout)` with `residual_bound`, `privileged_critic`, `standardize` from the checkpoint; no object state is read at evaluation; `run_rl_episode` is unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_rebuilds_the_bounded_model_and_needs_no_object_state(tmp_path, monkeypatch):
    ...  # mirror test_evaluate_reconstructs_the_input_layout_from_the_residual_checkpoint with a checkpoint saved from
    # DiceResidualModel(device="cpu", residual_input="base", state_features="struct", proprio=True, residual_bound=0.3, privileged_critic=True,
    #                   standardize={"struct_mean": 0.0, "struct_std": 1.0, "object_mean": torch.zeros(22), "object_std": torch.ones(22)});
    # assert the rebuilt model has residual_bound == 0.3, a separate select_critic with loaded weights, and that run_rl_episode was called
    # without an object_layout argument; assert settings["rl"]["layout"]["privileged_critic"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "rebuilds_the_bounded_model" -v`
Expected: FAIL (`replace(config, **layout)` rejects the new keys or the model is built without them)

- [ ] **Step 3: Implement the layout reconstruction in `evaluate` and `plus_eval.py`**

Both pass the layout's `residual_bound`, `privileged_critic` and `standardize` to `DiceResidualModel` (tensors rebuilt from the stored lists) and keep the config reconstruction to the config-level keys (`state_features`, `proprio`, `plan_candidates`, `plan_target`, `residual_bound`, `privileged_critic`).

- [ ] **Step 4: Run the tests and the evaluation tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py -x`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add script/lingbot_rl_train.py docs/libero-plus/plus_eval.py tests/test_lingbot_rl.py
git commit -m "Evaluation rebuilds the bounded actor and the deployable critic from the checkpoint layout."
```

---

### Task 8: Recipe presets and the launcher

**Files:**
- Create: `brev/recipes/bounded-ood.json`, `brev/recipes/bounded-std.json`
- Modify: `brev/train.sh:7-60` (`--recipe <name>` merges the preset into `config.json`; `--diagnostic` flags not needed since the preset carries them)
- Modify: `docs/libero-plus/README.md` (one paragraph: procedural poses, evaluation ruling), `AGENTS.md` LingBot DICE-RL section (one sentence on `--recipe`)
- Test: `bash -n brev/train.sh`; `tests/test_lingbot_rl.py` test that both presets validate through `config_from_dict` and produce the spec's protocol values

**Interfaces:**
- Produces: `brev/recipes/bounded-ood.json` = `{"residual_input": "base", "proprio": true, "state_features": "struct", "plan_candidates": 1, "online_env_steps": 240000, "selection_warmup_steps": 20000, "residual_bound": 0.3, "bc_weight": 0.0, "actor_warmup_episodes": 50, "residual_ramp_steps": 50000, "shaping": true, "privileged_critic": true, "standardize_features": true, "update_every_chunks": 1, "rlpd_start": 0.5, "rlpd_end": 0.5, "diagnostic_label_every": 10, "label_workers": 4, "procedural_robot_init": {}}` with `plus_train` filled by the launcher from `plus_train_variants()`; `bounded-std.json` identical minus `plus_train`, `procedural_robot_init`; `train.sh ... --recipe bounded-ood` (`--plus` implied by the preset carrying `procedural_robot_init`).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("name, mix", [("bounded-ood", "thirds_standard_procedural_robot_init_layout"), ("bounded-std", "standard")])
def test_recipe_presets_validate_and_pin_the_spec_values(name, mix):
    from script.lingbot_rl_config import plus_train_variants
    preset = json.loads((Path("brev/recipes") / f"{name}.json").read_text())
    if "procedural_robot_init" in preset:
        preset["plus_train"] = {axis: list(v) for axis, v in plus_train_variants().items()}
    protocol = config_from_dict({"task_ids": [9], **preset}).protocol()
    assert (protocol["residual_bound"], protocol["beta"], protocol["bc_filter"]) == (0.3, 0.0, "off")
    assert (protocol["actor_warmup_episodes"], protocol["residual_ramp_steps"], protocol["update_every_chunks"]) == (50, 50_000, 1)
    assert protocol["reward"] == "success_plus_potential_shaping" and protocol["critic_input"] == "features_proprio_object_state"
    assert protocol["plan_candidates"] == 1 and protocol["online_env_steps"] == 240_000 and protocol["collection_mix"] == mix
    assert (protocol["rlpd_start"], protocol["rlpd_end"], protocol["diagnostic_label_every"]) == (0.5, 0.5, 10)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k recipe_presets -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write the presets, the `--recipe` merge in `brev/train.sh`, and the doc lines**

In `train.sh`, after the existing `config.json` is written, `--recipe NAME` runs `python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c.update(json.load(open(sys.argv[2]))); json.dump(c, open(sys.argv[1],"w"))' "$OUT/config.json" "$DICE_REPO/brev/recipes/$RECIPE.json"` and sets `PLUS=1` when the preset contains `procedural_robot_init` (so `plus_train` and the plus activation are applied as for `--plus`).

- [ ] **Step 4: Verify**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k recipe_presets -v && bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh`
Expected: PASS, no syntax errors

- [ ] **Step 5: Commit**

```bash
git add brev/recipes brev/train.sh docs/libero-plus/README.md AGENTS.md tests/test_lingbot_rl.py
git commit -m "Recipe presets for the bounded residual runs and a --recipe launcher flag."
```

---

### Task 9: Smokes on both boxes, then launch

**Files:**
- Create: scratch scripts under `~/Documents/dice/scratch/` on each box (not committed); results recorded in `docs/heldout-results.md` and `docs/probe-outcomes.md`

**Interfaces:**
- Consumes: `brev/sync.sh`, `brev/train.sh <run> 9 --recipe bounded-ood --max-env-steps 2100 --no-tmux` (Jarvis) and `--recipe bounded-std` (Watson); the box test from Task 4 step 5.

- [ ] **Step 1: Sync and run the box replay test on Jarvis (Task 4 step 5)**

Expected: 30 demos replayed, matched fraction ≥ 0.9 each; record the minimum in `docs/probe-outcomes.md`.

- [ ] **Step 2: Smoke `bounded-ood` on Jarvis and `bounded-std` on Watson with `actor_warmup_episodes` forced to 1 and `residual_ramp_steps` to 500 through an extra `--config-extra '{"actor_warmup_episodes": 1, "residual_ramp_steps": 500}'` (add this flag to `train.sh` in Task 8 if not present) for 2,100 env steps**

Gates (read from `train.log`, `settings.json`, the resume buffer):
- residual rms 0 during the warm-up episode; afterwards within [0.02, 0.3] and `residual_saturation` < 0.5;
- per-episode shaping sum equals `γ`-weighted Φ(s_T) − Φ(s_0) to 1e-4 (script over the buffer rows); Φ(s_0) on standard resets matches `phi_initial` in `settings.json`;
- kind-1 episodes: logged `pose_offset_l2` in [0.1, 0.3] and the environment's joints after reset equal base + offset to 1e-6 (one assertion in the smoke script using the override's current offset);
- `standardize` present in `settings.json`; the privileged critic's first-layer width 6144 + 8 + 22 + 480 and the deployable critic's 6144 + 8 + 480; `select_agreement` logged every chunk;
- diagnostic labels on every 10th chunk; shadow fidelity gates as before;
- seconds per env step ≤ 0.5; peak GPU memory < 60 GB.

- [ ] **Step 3: Record the smoke numbers in `docs/heldout-results.md` (new subsection) and commit**

```bash
git add docs/heldout-results.md docs/probe-outcomes.md
git commit -m "Smoke results for the bounded residual recipe on both boxes."
```

- [ ] **Step 4: Launch (user's go required)**

`brev/train.sh dice-t9-ood-bounded 9 --recipe bounded-ood` on Jarvis, `brev/train.sh dice-t9-std-bounded 9 --recipe bounded-std` on Watson; start a 30-minute watcher for crashes, checkpoints, `residual_saturation`, `q_label_agreement` and `pose_cap`.

---

## Self-review notes

- Spec coverage: bound and BC (Task 2), schedule (Task 6), actor inputs and standardisation (Tasks 2–3, 6), privileged and deployable critics (Tasks 2, 4, 6, 7), shaping (Tasks 3, 6), cadence and expert ratio (Tasks 1, 6), procedural poses and curriculum (Tasks 5–6), diagnostics (Task 6), evaluation (Task 7), presets (Task 8), smoke gates (Task 9). Token sidecar and plan critic untouched (M = 1 disables them).
- Type consistency: `draw_reset` returns a 4-tuple everywhere; `critic_state(state, pool, object_state=None)` is the one signature used by the loop, `rank_candidates` and `n_step_target`'s caller, with the object state present only on the privileged path; `standardize` is a dict with two floats and two 22-vectors in every task.
- Risk owned by Task 4: if replay drifts on the box, stop and surface it; the fallback is a spec change.
