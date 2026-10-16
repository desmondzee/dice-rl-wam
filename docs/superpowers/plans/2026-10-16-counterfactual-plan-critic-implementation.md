# Plan critic trained on counterfactual simulator rollouts: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace run A (`dice-t9-plus-m4-pool`) by `dice-t9-plus-m4-pool-cf`, the same recipe with `plan_target = "rollout"`: every one of the M × K candidate chunks is executed in a shadow simulator before the live chunk, the per-candidate progress of the task 9 object poses is stored as `plan_labels` (M, K), and V_plan regresses onto the mean label per plan instead of the Q-derived target, ranking by the mean over its heads.

**Architecture:** The progress function and its startup calibration live in a new stdlib-plus-numpy module `script/lingbot_rl_progress.py`. The trainer gains four environment helpers (`make_shadow`, `sync_shadow`, `label_plans`, `shadow_fidelity`): a second `LiberoEnv` with the live environment's kwargs is reset once per episode and then driven only by `sim.set_state` plus a copy of the OSC goal and the gripper ramp; candidates are stepped through the LeRobot wrapper's `step` with `decode_action`, so the controller and control frequency match the live environment. The model carries `plan_target`; under `"rollout"` `update_plan_critic` regresses onto `plan_labels.mean(-1)` and still computes the Q target for logging, and `rank_plans` reads the mean over heads. The buffer stores `plan_labels` with a legacy default of zeros. A startup fidelity check gates the run and its maxima go to `settings.json`.

**Tech Stack:** Torch 2.11 (cu130 on the GB10s), the pinned LeRobot LingBot-VA port and its `LiberoEnv`, robosuite's MuJoCo binding (`sim.get_state` / `set_state` / `forward`, OSC `goal_pos` / `goal_ori`, Panda gripper `current_action`), numpy replay rows, pytest with the `_StubPolicy` / `_smoke_train` fixtures in `tests/test_lingbot_rl.py`, LIBERO-Plus at `4976dc3` on the GB10s.

**Spec:** docs/superpowers/specs/2026-10-16-counterfactual-plan-critic-design.md

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests.
- No secrets in logs, argv, metadata or artifacts.
- The default recipe (`plan_target` `"q"`) must reproduce bit-for-bit: no new RNG draws on the default path, no behaviour change in `rank_plans` / `update_plan_critic` when `"q"`, no change to the default `protocol()` (78 keys), no change to the default checkpoint `layout` dict.
- The resume fingerprint changes with `plan_target` (`test_load_resume_rejects_recipe_mismatch` is the refusal mechanism).
- The run name is `dice-t9-plus-m4-pool-cf`; run B (`dice-t9-plus-m4-struct`, Watson) is untouched.
- What stays fixed (spec section 2): the residual actor, the Q ensemble and its targets, the sampler, M = 4, K = 4, the reset schedule, the buffer's other keys, both evaluation protocols.
- Do not edit `.cache/lerobot`; do not edit `script/lingbot_eval.py` (its `harness_sha256` enters evaluation provenance); do not instantiate Hydra `DistillResidualRLModel`; tests must not call `modal run` or `.remote()`.
- No GPU jobs, Modal, ssh or remote machines from implementers; Task 7 is run by the controller on Jarvis.
- Verify every task with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `.cache/eval-venv/bin/python -m compileall -q script/`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`.

## Review Focus

1. A shadow stepped more than LIBERO's `horizon` (1000) raises `ValueError("executing action in terminated episode")` from robosuite unless `ignore_done` is set: Task 5 tests that `make_shadow` sets `ignore_done = True` on the inner env (a shadow executes about 8,000 steps per episode).
2. A candidate that reaches success mid-chunk must stop there and carry +2, like the live loop stops at success: Task 5 tests the step count and label of a succeeding candidate.
3. The door term must not reward closing the door while the mug is outside (progress myopia, spec section 7): Task 2 tests that a door change with the mug outside has zero progress.
4. A rollout checkpoint evaluated by a model built for the Q readout would rank with the min over heads: Task 4 tests that `layout` carries `plan_target` and that loading it into a `"q"` model raises `layout mismatch`.
5. A rollout recipe without object poses or with a single plan has nothing to label: Task 1 tests that `validate()` refuses `plan_target == "rollout"` without `proprio` or with `plan_candidates == 1`.

## Decisions the spec left open (binding for every task)

- **Door calibration.** The spec says the door range is measured at startup; the brief proposed "closed = the limit nearer the initial value of stored state 0". Task 9's bddl initialises `(Open microwave_1)` (`libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl` line 76) and `microwave.xml` gives `microjoint` the range `-2.094 0` with LIBERO's `default_close_ranges = [-0.005, 0.0]`, so the nearer-to-initial rule would label the open limit as closed. Decision: `closed` = the `jnt_range` limit nearer zero (the articulated object's rest pose), `open` = the other; the initial door value and `phi_initial` are recorded in `settings.json` under `progress_calibration`, and Task 7 checks on the box that `door_initial` lies nearer `door_open`.
- **Interior target.** As the brief decided: `micro` = `microwave_1` body position, `inside` = ‖mug − micro‖ < 0.12 m (`INSIDE_RADIUS`). The heating-region site sits at (0, 0.01, 0.096) from the body origin with half-extents (0.12, 0.0835, 0.08), so the ball covers the cavity centre and the +2 success term carries the goal; a site-based target is a later refinement, not part of this plan.
- **Progress signature.** `calibrate(task_id, inner, layout, initial) -> dict` (task-specific names, door limits, `door_initial`, `phi_initial`), `phi(calibration, layout, state, success) -> float`, `progress(calibration, layout, before, after, success) -> float` where `before` has `success=False`. The task is selected in `calibrate` through `TASKS`; unknown tasks raise `ValueError` at startup.
- **Shadow state transfer.** `sync_shadow(live, twin)` copies `sim.get_state()` (time, qpos, qvel), `data.qacc_warmstart`, calls `sim.forward()`, then per robot `controller.update(force=True)`, copies `controller.goal_pos` / `goal_ori` from the live controller (the OSC keeps `goal_ori` across steps when the rotation delta is zero, so a `reset_goal` would differ from the live controller) and copies `gripper.current_action` (the Panda gripper ramps its command by 0.01 per substep, so it is real controller state). `use_delta` is already True on both after `reset`.
- **Shadow stepping.** Candidates go through `shadow.step(decode_action(...))` (the LeRobot wrapper), so rendering happens on every shadow step; its cost is measured in the smoke (`label_seconds`). The `use_camera_obs = False` shortcut is not taken: `LiberoEnv._format_raw_obs` reads the camera keys, so that shortcut would require stepping `shadow._env` directly; Task 7 names it as the fallback if `label_seconds` exceeds 2 s.
- **Shadow construction.** `make_shadow(task_id, suite, init_states, plus, seed)` builds `make_env(task_id, suite, init_states=init_states, plus=plus)` (obs_type `"pixels"`), resets it inside `capture_rng()` / `restore_rng()` (LIBERO's reset seeding must not touch the collection stream), then sets `shadow._env.env.ignore_done = True`. One shadow per episode, reset with the live seed `config.seed + env_steps`; the startup probe's shadow uses `plus=False` and `init_states=False` like the probe env.
- **Label timing.** `label_plans` runs after decoding and before `commit_executed`, inside the region timed for `seconds_per_env_step`, so that metric reports the true cost; `label_seconds` is logged separately.
- **Q reference for the agreement diagnostics.** In the loop, `label_agreement_with_q` compares the label argmax with the argmax over plans of the max over K of the mean over the ten Q heads (a second `critic(..., return_all=True)` call on the same inputs, only under rollout; the existing `q_all` min-readout path is untouched). In `update_plan_critic`, `plan_label_vs_q_agreement` compares the label argmax with the argmax of the Q-derived target (max over K of the min over heads), the quantity the critic used to train on.
- **Checkpoint layout.** `inference_state_dict()["layout"]` gains `"plan_target"` only when it is not `"q"`; `layout_of` returns it only when present; `load_inference_state_dict` compares `layout.get("plan_target", "q")` with the model's. `evaluate()` and `docs/libero-plus/plus_eval.py` already apply `**layout` to the config / model, so a rollout checkpoint evaluates with the mean readout without further edits.
- **Protocol.** Under rollout `protocol()` adds `"plan_target": "rollout"` and overrides `"plan_critic"` with `"regression_mean_k_rollout_progress"`; the default protocol is byte-identical.
- **Validation.** `plan_target == "rollout"` requires `plan_candidates > 1` and `proprio` (object poses are read only when `proprio`).
- **Fidelity rollout.** `shadow_fidelity(env, shadow, layout, act, observation, chunks=20)` drives the live probe env with `act(observation) -> (7,) env action`, chunk lengths from `env_action_count(first)` (12 then 16), syncs the shadow at every chunk start, steps both with the same action, and compares `object_state` after each chunk; it raises `RuntimeError` when the maximum position gap is ≥ 1e-3 m or the maximum joint gap ≥ 0.02 rad. The trainer's `act` is the prior through `policy.select_action` on `rl_observation_batch`, as `run_rl_episode` does. Thresholds: `FIDELITY_POSITION_M = 1e-3`, `FIDELITY_DOOR_RAD = 0.02`, `FIDELITY_CHUNKS = 20`.
- **Sharpening blocks.** `replay_sharpening` and `_maybe_sharpen` keep calling `plan_critic` with the min readout; they are diagnostics, not ranking.

## File map

| file | responsibility after this plan |
|---|---|
| `script/lingbot_rl_config.py` | `plan_target`; validation; protocol keys under rollout |
| `script/lingbot_rl_progress.py` | `TASKS`, `INSIDE_RADIUS`, `body_position`, `joint_value`, `calibrate`, `phi`, `progress` |
| `script/lingbot_rl_buffer.py` | `plan_labels` (M, K) float32 with a zeros default |
| `script/lingbot_rl_model.py` | `plan_target` on the model; rollout target and diagnostics; mean readout in `rank_plans`; `plan_target` in `layout` |
| `script/lingbot_rl_train.py` | `make_shadow`, `sync_shadow`, `label_plans`, `shadow_fidelity`; the probe calibration and fidelity gate; per-episode shadow; labels stored and logged |
| `brev/train.sh` | `--plan-target q|rollout` |

---

### Task 1: `plan_target` in the config and the fingerprint

**Files:**
- Modify: `script/lingbot_rl_config.py` (field after line 49, `validate()` before line 89, `protocol()` after line 194)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: `RLConfig.plan_target: str = "q"` (`"q"` | `"rollout"`); `validate()` raises `ValueError` for other values and for rollout with `plan_candidates < 2` or `proprio` False; `protocol()` adds `"plan_target"` and sets `"plan_critic": "regression_mean_k_rollout_progress"` only under rollout; `config_from_dict` needs no change (string field).

- [ ] **Step 1: Write the failing test** (after `test_default_protocol_is_unchanged_by_the_feature_fields`)

```python
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
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_target_enters`
Expected: FAIL with `TypeError: RLConfig.__init__() got an unexpected keyword argument 'plan_target'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_config.py`

After `plus_train: dict = field(default_factory=dict)` (line 49):

```python
    plan_target: str = "q"
```

In `validate()` before `return self` (line 89):

```python
        if self.plan_target not in ("q", "rollout"):
            raise ValueError("plan_target must be q or rollout")
        if self.plan_target == "rollout" and (self.plan_candidates < 2 or not self.proprio):
            raise ValueError("rollout plan labels need plan_candidates > 1 and proprio")
```

In `protocol()` after the feature `protocol.update({...})` block (line 194), before `return protocol`:

```python
        if self.plan_target != "q":
            protocol.update({"plan_target": self.plan_target, "plan_critic": "regression_mean_k_rollout_progress"})
```

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config`

```bash
git add script/lingbot_rl_config.py tests/test_lingbot_rl.py
git commit -m "Add plan_target to the RL recipe: rollout labels enter the fingerprint and require plans and proprioception."
```

---

### Task 2: Progress function and door calibration

**Files:**
- Create: `script/lingbot_rl_progress.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: the `object_state` vector layout (`[pos(3), quat(4)]` per body in `layout["bodies"]` order, then `layout["joints"]` qpos values) and `inner.sim.model.jnt_range` / `joint_name2id`.
- Produces: `INSIDE_RADIUS = 0.12`; `TASKS = {9: {"mug": "white_yellow_mug_1", "microwave": "microwave_1", "door": "microwave_1_microjoint"}}`; `body_position(state, layout, name) -> float64 (3,)`; `joint_value(state, layout, name) -> float`; `calibrate(task_id, inner, layout, initial) -> {"task_id", "mug", "microwave", "door", "door_closed", "door_open", "door_initial", "phi_initial"}` (raises `ValueError` for a task not in `TASKS`, `ValueError` from `.index` for a name missing from the layout); `phi(calibration, layout, state, success) -> float`; `progress(calibration, layout, before, after, success) -> float` = `phi(after, success) - phi(before, False)`.

- [ ] **Step 1: Write the failing test** (after `test_config_plan_target_enters_the_fingerprint_only_for_rollout`)

```python
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
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k progress_rewards`
Expected: FAIL with `ModuleNotFoundError: No module named 'script.lingbot_rl_progress'`.

- [ ] **Step 3: Create** `script/lingbot_rl_progress.py`

```python
import numpy as np

INSIDE_RADIUS = 0.12
TASKS = {9: {"mug": "white_yellow_mug_1", "microwave": "microwave_1", "door": "microwave_1_microjoint"}}


def body_position(state, layout, name):
    start = 7 * layout["bodies"].index(name)
    return np.asarray(state[start:start + 3], dtype=np.float64)


def joint_value(state, layout, name):
    return float(state[7 * len(layout["bodies"]) + layout["joints"].index(name)])


def phi(calibration, layout, state, success):
    distance = float(np.linalg.norm(
        body_position(state, layout, calibration["mug"]) - body_position(state, layout, calibration["microwave"])))
    inside = float(distance < INSIDE_RADIUS)
    door_open = (joint_value(state, layout, calibration["door"]) - calibration["door_closed"]) / (
        calibration["door_open"] - calibration["door_closed"])
    return -distance + inside + (1.0 - door_open) * inside + 2.0 * float(success)


def calibrate(task_id, inner, layout, initial):
    if task_id not in TASKS:
        raise ValueError(f"No progress function for task {task_id}")
    names = TASKS[task_id]
    model = inner.sim.model
    low, high = (float(limit) for limit in model.jnt_range[model.joint_name2id(names["door"])])
    closed, opened = (low, high) if abs(low) <= abs(high) else (high, low)
    calibration = {"task_id": int(task_id), **names, "door_closed": closed, "door_open": opened,
                   "door_initial": joint_value(initial, layout, names["door"])}
    calibration["phi_initial"] = phi(calibration, layout, initial, False)
    return calibration


def progress(calibration, layout, before, after, success):
    return phi(calibration, layout, after, success) - phi(calibration, layout, before, False)
```

`phi_initial` validates every name in `TASKS[task_id]` against the layout at startup (a missing body raises `ValueError` from `list.index`). `inner.sim.model.jnt_range` is the (njnt, 2) MuJoCo array exposed by robosuite's `MjModel` delegate; `joint_name2id` is the robosuite binding method already used next to `get_joint_qpos_addr`.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_progress.py tests/test_lingbot_rl.py
git commit -m "Task 9 progress: mug-to-microwave distance, inside and door-closed terms, success bonus, door range from the rest limit."
```

---

### Task 3: `plan_labels` in the replay buffer

**Files:**
- Modify: `script/lingbot_rl_buffer.py` (`PLAN_FLOAT_KEYS` line 6, `_plan_defaults` lines 60–70)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces: row key `plan_labels` (M, K) float32, default zeros; `sample()` returns `plan_labels` (B, M, K) float32 on `self.device`; a row whose `plan_labels` is not shaped `(plans, K_CANDIDATES)` raises `ValueError`; legacy `state_dict` rows get the zeros default.

- [ ] **Step 1: Write the failing test** (after `test_feature_rows_default_finalize_and_sample_in_bf16_with_store_only_keys`)

```python
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
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k plan_labels_default`
Expected: FAIL with `KeyError: 'plan_labels'` on `rows[1]["plan_labels"]`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_buffer.py`

Line 6 becomes:

```python
PLAN_FLOAT_KEYS = ("plan_pool", "plan_pools_all", "plan_a_base_all", "plan_pool_next", "plan_labels")
```

(`FLOAT_KEYS` and `KEYS` splat `PLAN_FLOAT_KEYS`, so `plan_labels` is hosted as float32, stacked by `sample`, and re-hosted in `load_state_dict` without further edits.) In `_plan_defaults`, replace the `plan_pool_next` default and the shape check (lines 65–68) with:

```python
        _default(stored, "plan_pool_next", lambda: np.array(stored["plan_pool"], copy=True))
        _default(stored, "plan_labels", lambda: np.zeros((self.plans, K_CANDIDATES), np.float32))
        if _host_float(stored["plan_pools_all"]).shape[0] != self.plans or \
                _host_float(stored["plan_a_base_all"]).shape[0] != self.plans or \
                _host_float(stored["plan_labels"]).shape != (self.plans, K_CANDIDATES):
            raise ValueError(f"Rows must carry exactly {self.plans} plans")
```

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`

```bash
git add script/lingbot_rl_buffer.py tests/test_lingbot_rl.py
git commit -m "Store per-candidate plan labels (M, K) in replay rows with a zeros default for legacy and expert rows."
```

---

### Task 4: Model: rollout target, mean readout, diagnostics, layout

**Files:**
- Modify: `script/lingbot_rl_model.py` (`__init__` 119–140, `layout_of` 162–165, `rank_plans` 182–185, `update_plan_critic` 264–285, `inference_state_dict` 292–301, `load_inference_state_dict` 303–314), `script/lingbot_rl_train.py` (model construction at lines 455 and 725 only)
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `plan_labels` (B, M, K) from `sample()` (Task 3).
- Produces: `DiceResidualModel(device, residual_input, epsilon, plan_candidates=1, state_features="pool", proprio=False, plan_target="q")` with `self.plan_target`; `update_plan_critic(state, pools_all, z_all, plan_a_base_all, plan_labels=None)` returns `{"plan_critic_loss", "v_plan_mean", "plan_target_mean"}` under `"q"` (unchanged) and additionally `{"plan_target_q_mean", "plan_label_vs_q_agreement"}` under `"rollout"`, where the trained target is `plan_labels.mean(-1)` and `ValueError` is raised when `plan_labels` is None; `rank_plans(state, pools)` returns the argmax of the mean over heads under `"rollout"` and of the min readout under `"q"`; `inference_state_dict()["layout"]["plan_target"]` present only when not `"q"`; `layout_of(payload)` returns `plan_target` only when present; `load_inference_state_dict` raises `ValueError("residual checkpoint layout mismatch")` when `layout.get("plan_target", "q") != self.plan_target`.

- [ ] **Step 1: Write the failing test** (after `test_plan_critic_regresses_onto_max_candidate_min_q_and_leaves_q_and_actor_alone`)

```python
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
    heads = torch.stack(model.plan_critic(state[:1].expand(2, -1), pools[0], return_all=True))
    assert model.rank_plans(state[:1], pools[0]) == int(heads.mean(dim=0).reshape(-1).argmax())
    payload = model.inference_state_dict()
    assert payload["layout"]["plan_target"] == "rollout" and DiceResidualModel.layout_of(payload)["plan_target"] == "rollout"
    assert "plan_target" not in DiceResidualModel(device="cpu", plan_candidates=2).inference_state_dict()["layout"]
    with pytest.raises(ValueError, match="layout mismatch"):
        DiceResidualModel(device="cpu", residual_input="base", plan_candidates=2).load_inference_state_dict(payload)
```

- [ ] **Step 2: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k rollout_labels_and_ranks`
Expected: FAIL with `TypeError: DiceResidualModel.__init__() got an unexpected keyword argument 'plan_target'`.

- [ ] **Step 3: Implement** in `script/lingbot_rl_model.py`

`__init__` signature and first lines:

```python
    def __init__(self, device="cpu", residual_input="z", epsilon=EPSILON, plan_candidates=1, state_features="pool", proprio=False,
                 plan_target="q"):
        self.device = device
        self.epsilon = epsilon
        self.plans = int(plan_candidates)
        self.plan_target = plan_target
```

(the rest of `__init__` unchanged.) `layout_of`:

```python
    @staticmethod
    def layout_of(payload):
        layout = payload.get("layout", {})
        out = {"state_features": layout.get("state_features", "pool"), "proprio": bool(layout.get("proprio", False))}
        if "plan_target" in layout:
            out["plan_target"] = layout["plan_target"]
        return out
```

`rank_plans`:

```python
    def rank_plans(self, state, pools):
        with torch.no_grad():
            pools = mlp_float(pools)
            state = mlp_float(state).expand(pools.shape[0], -1)
            if self.plan_target == "rollout":
                return int(torch.stack(self.plan_critic(state, pools, return_all=True)).mean(dim=0).reshape(-1).argmax())
            return int(self.plan_critic(state, pools).reshape(-1).argmax())
```

`update_plan_critic`: signature `def update_plan_critic(self, state, pools_all, z_all, plan_a_base_all, plan_labels=None):`; after the `with torch.no_grad():` block (which still ends with `target = self.critic(...).max(dim=2).values`) insert:

```python
        extra = {}
        if self.plan_target == "rollout":
            if plan_labels is None:
                raise ValueError("rollout plan target needs plan_labels")
            labels = mlp_float(plan_labels).mean(dim=-1)
            extra = {"plan_target_q_mean": float(target.mean()),
                     "plan_label_vs_q_agreement": float((labels.argmax(dim=1) == target.argmax(dim=1)).float().mean())}
            target = labels
```

and the return becomes:

```python
        return {"plan_critic_loss": float(loss.detach()), "v_plan_mean": float(torch.stack(preds).mean().detach()),
                "plan_target_mean": float(target.mean()), **extra}
```

`inference_state_dict`: after `payload["layout"] = {...}` add:

```python
        if self.plan_target != "q":
            payload["layout"]["plan_target"] = self.plan_target
```

`load_inference_state_dict`: the mismatch condition becomes

```python
        if layout is not None and (layout["state_features"], bool(layout["proprio"]), int(layout.get("plan_candidates", self.plans)),
                                   layout.get("plan_target", "q")) != (self.features, self.proprio, self.plans, self.plan_target):
            raise ValueError("residual checkpoint layout mismatch")
```

In `script/lingbot_rl_train.py`, both `DiceResidualModel(...)` constructions (line 455 in `train`, line 725 in `evaluate`) gain `plan_target=config.plan_target` after `proprio=config.proprio`. `evaluate` already does `config = replace(config, **layout).validate()`, so a rollout checkpoint sets `plan_target` on the eval config and the model; `docs/libero-plus/plus_eval.py` passes `**layout` to the constructor and needs no edit.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_model.py script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Plan critic rollout target: train on the mean stored label per plan, rank by the mean over heads, log the Q target for comparison."
```

---

### Task 5: Shadow environment helpers: `make_shadow`, `sync_shadow`, `label_plans`, `shadow_fidelity`

**Files:**
- Modify: `script/lingbot_rl_train.py` (imports lines 13–24; new functions after `object_state`, line 145)
- Test: `tests/test_lingbot_rl.py` (two fixtures, two tests)

**Interfaces:**
- Consumes: `calibrate` / `progress` (Task 2); `object_state(inner, layout)`, `env_success(info)`, `capture_rng` / `restore_rng`, `make_env`; `slice_env_actions`, `env_action_count` (already imported from `script.lingbot_rl_policy`); `decode_action` from `script.lingbot_eval`.
- Produces: `FIDELITY_CHUNKS = 20`, `FIDELITY_POSITION_M = 1e-3`, `FIDELITY_DOOR_RAD = 0.02`; `make_shadow(task_id, suite, init_states, plus, seed) -> LiberoEnv` (reset once inside an RNG capture, `ignore_done = True` on `shadow._env.env`); `sync_shadow(live, twin)` (inner robosuite envs); `label_plans(shadow, live, candidates (M, K, 16, 30) tensor, first_chunk, norm, layout, calibration) -> np.float32 (M, K)`; `shadow_fidelity(env, shadow, layout, act, observation, chunks=FIDELITY_CHUNKS) -> {"position_m", "door_rad", "chunks"}` raising `RuntimeError` above the thresholds.

- [ ] **Step 1: Fixtures** in `tests/test_lingbot_rl.py` (before `_StubPolicy`)

```python
class _TwinSim:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0, qpos=np.array([0.3, 0.0, 0.0, -2.0]), qvel=np.zeros(4), qacc_warmstart=np.zeros(4),
                                    body_xpos=np.zeros((2, 3)), body_xquat=np.zeros((2, 4)))
        self.model = SimpleNamespace(get_joint_qpos_addr=lambda name: 3)
        self.forward()

    def get_state(self):
        return SimpleNamespace(time=self.data.time, qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())

    def set_state(self, state):
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    def forward(self):
        self.data.body_xpos[1] = self.data.qpos[:3]


class _TwinEnv:
    def __init__(self, drift=0.0):
        controller = SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3), update=lambda force=False: None)
        self._env = SimpleNamespace(env=SimpleNamespace(
            sim=_TwinSim(), obj_body_id={"micro": 0, "mug": 1}, ignore_done=False,
            robots=[SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=np.zeros(2)))]))
        self.drift, self.steps = drift, 0

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        inner = self._env.env
        inner.sim.data.qpos[:3] += 0.01 * action[:3] + self.drift
        inner.sim.data.qpos[3] += 0.5 * max(float(action[6]), 0.0)
        inner.sim.forward()
        self.steps += 1
        success = bool(inner.sim.data.qpos[3] >= 0.0)
        return {}, 0.0, success, False, {"is_success": success}

    def close(self):
        return None


_TWIN_LAYOUT = {"bodies": ["micro", "mug"], "joints": ["door"], "body_quaternion": "wxyz"}
_TWIN_CALIBRATION = {"task_id": 9, "mug": "mug", "microwave": "micro", "door": "door", "door_closed": 0.0, "door_open": -2.0}
```

The stub's `object_state` (the real function) reads body 0 (microwave at the origin), body 1 (the mug at `qpos[:3]`) and `qpos[3]` (the door, open at −2.0; a positive gripper command closes it by 0.5 per step and success fires at 0.0).

- [ ] **Step 2: Write the failing tests** (after the fixtures)

```python
def test_label_plans_scores_every_candidate_from_the_live_state_with_the_chunk_length():
    from script.lingbot_rl_train import label_plans
    live, shadow = _TwinEnv(), _TwinEnv()
    shadow._env.env.sim.data.qpos[:] = [9.0, 9.0, 9.0, 9.0]
    candidates = torch.zeros(2, 2, HORIZON, ACTION_DIM)
    candidates[0, :, :, 0] = 1.0
    candidates[1, :, :, 0] = -1.0
    candidates[1, 1, 4:, 6] = 1.0
    norm = _eval_normalization()
    labels = label_plans(shadow, live._env.env, candidates, True, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    assert labels.shape == (2, 2) and labels.dtype == np.float32
    assert labels[0, 0] == pytest.approx(-0.12, abs=1e-4) and labels[0, 1] == labels[0, 0]
    assert labels[1, 0] == pytest.approx(0.12, abs=1e-4) and labels[1, 1] == pytest.approx(2.04, abs=1e-4)
    assert shadow.steps == 12 + 12 + 12 + 4 and live._env.env.sim.data.qpos[0] == 0.3 and live.steps == 0
    later = label_plans(shadow, live._env.env, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    assert later[0, 0] == pytest.approx(-0.16, abs=1e-4) and later[1, 0] == pytest.approx(0.16, abs=1e-4)


def test_shadow_fidelity_measures_object_pose_gaps_per_chunk_and_gates_on_them(monkeypatch):
    import script.lingbot_rl_train as train
    from script.lingbot_rl_train import make_shadow, shadow_fidelity
    monkeypatch.setattr(train, "LiberoEnv", lambda **kwargs: _TwinEnv())
    calls = []
    act = lambda observation: calls.append(observation) or np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], np.float32)
    live, shadow = _TwinEnv(), make_shadow(9, None, False, False, 0)
    assert shadow._env.env.ignore_done is True
    result = shadow_fidelity(live, shadow, _TWIN_LAYOUT, act, {"step": 0}, chunks=3)
    assert result == {"position_m": 0.0, "door_rad": 0.0, "chunks": 3} and len(calls) == 12 + 16 + 16
    assert shadow.steps == live.steps == 44 and calls[0] == {"step": 0}
    with pytest.raises(RuntimeError, match="fidelity"):
        shadow_fidelity(_TwinEnv(), _TwinEnv(drift=2e-4), _TWIN_LAYOUT, act, {}, chunks=3)
```

Expected values: a first chunk executes 12 steps of 0.01 m, so the mug moves from 0.3 to 0.42 (label −0.12) or 0.18 (+0.12); `decode_action` with the unit normalisation scales by `1 + 5e-7`, hence `abs=1e-4`. Candidate (1, 1) closes the door at 0.5 rad per step and succeeds after four steps: Δ(−distance) = +0.04, plus 2.0. The second call uses `first_chunk=False` (16 steps). With a drift of 2e-4 m per step the shadow is 2.4e-3 m off after the first chunk, above `FIDELITY_POSITION_M`.

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "label_plans_scores or shadow_fidelity_measures"`
Expected: FAIL with `ImportError: cannot import name 'label_plans'`.

- [ ] **Step 4: Implement** in `script/lingbot_rl_train.py`

Imports: add `from script.lingbot_rl_progress import calibrate, progress` after the `script.lingbot_rl_policy` import (line 23). After `SHARPEN_EVERY_STEPS = 5_000` add:

```python
FIDELITY_CHUNKS = 20
FIDELITY_POSITION_M = 1e-3
FIDELITY_DOOR_RAD = 0.02
```

After `object_state` (line 145):

```python
def make_shadow(task_id, suite, init_states, plus, seed):
    shadow = make_env(task_id, suite, init_states=init_states, plus=plus)
    rng = capture_rng()
    try:
        shadow.reset(seed=seed)
    finally:
        restore_rng(rng)
    shadow._env.env.ignore_done = True
    return shadow


def sync_shadow(live, twin):
    twin.sim.set_state(live.sim.get_state())
    twin.sim.data.qacc_warmstart[:] = live.sim.data.qacc_warmstart
    twin.sim.forward()
    for source, target in zip(live.robots, twin.robots):
        target.controller.update(force=True)
        target.controller.goal_pos = np.array(source.controller.goal_pos)
        target.controller.goal_ori = np.array(source.controller.goal_ori)
        target.gripper.current_action = np.array(source.gripper.current_action)


def label_plans(shadow, live, candidates, first_chunk, norm, layout, calibration):
    from script.lingbot_eval import decode_action
    twin = shadow._env.env
    plans, k = candidates.shape[:2]
    env_actions = slice_env_actions(candidates.detach().float().cpu().reshape(plans * k, HORIZON, ACTION_DIM), first_chunk)
    labels = np.zeros(plans * k, np.float32)
    for index in range(plans * k):
        sync_shadow(live, twin)
        before = object_state(twin, layout)
        success = False
        for step in range(env_actions.shape[1]):
            _, _, terminated, truncated, info = shadow.step(decode_action(env_actions[index, step].reshape(1, 7), norm))
            success = env_success(info)
            if success or terminated or truncated:
                break
        labels[index] = progress(calibration, layout, before, object_state(twin, layout), success)
    return labels.reshape(plans, k)


def shadow_fidelity(env, shadow, layout, act, observation, chunks=FIDELITY_CHUNKS):
    live, twin = env._env.env, shadow._env.env
    bodies = 7 * len(layout["bodies"])
    worst = np.zeros(2)
    for chunk in range(chunks):
        sync_shadow(live, twin)
        for _ in range(env_action_count(chunk == 0)):
            action = act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            shadow.step(action)
        diff = np.abs(object_state(live, layout) - object_state(twin, layout))
        worst = np.maximum(worst, [diff[:bodies].reshape(-1, 7)[:, :3].max(), diff[bodies:].max(initial=0.0)])
        if env_success(info) or terminated or truncated:
            break
    result = {"position_m": float(worst[0]), "door_rad": float(worst[1]), "chunks": chunk + 1}
    if worst[0] >= FIDELITY_POSITION_M or worst[1] >= FIDELITY_DOOR_RAD:
        raise RuntimeError(f"shadow fidelity check failed: {result}")
    return result
```

Source facts behind `sync_shadow` (robosuite at `~/.cache/uv/git-v0/checkouts/79abe6cb204f004b/c848ca8`): `MjSim.get_state()` returns `MjSimState(time, qpos copy, qvel copy)` and `set_state` copies them back (`utils/binding_utils.py:1140-1155`, `na == 0` asserted), `forward()` is `mj_forward`; `Controller.update(force=True)` recomputes `ee_pos` / `ee_ori_mat` from the sim (`controllers/base_controller.py:125-161`); OSC `set_goal` only rewrites `goal_ori` when the rotation delta is non-zero (`controllers/osc.py:271-278`), which is why the live goal is copied; `PandaGripper.format_action` ramps `current_action` by `speed = 0.01` per substep (`models/grippers/panda_gripper.py:43-62`). Robosuite's `MujocoEnv.step` raises once `done` is set, and `done = timestep >= horizon and not ignore_done` (`environments/base.py:378, 432`); LIBERO's `ControlEnv` defaults to `horizon=1000, ignore_done=False`, hence `ignore_done = True` on the shadow.

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Shadow environment helpers: state and controller sync, per-candidate progress labels, startup fidelity check."
```

---

### Task 6: Trainer integration, storage, logging and `train.sh`

**Files:**
- Modify: `script/lingbot_rl_train.py` (`_update_from_buffer` lines 370–371; `train()` probe block 479–488, settings line 500, episode head 552–563, chunk loop 572–606, row 634–655, `finally` 677), `brev/train.sh`, `tests/test_lingbot_rl.py` (`_smoke_train` fixture, one test)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `settings.json` gains `"progress_calibration"` and `"shadow_fidelity"` (None unless rollout); a shadow env per episode under rollout; rows carry `plan_labels` under rollout; `plan_info` gains `label_agreement_with_q`, `label_spread_within_plan`, `label_spread_across_plans`, `label_executed_minus_mean`, `label_seconds` (and their `/<kind>` copies under `plus_train`); `_update_from_buffer` passes `sample["plan_labels"]` to `update_plan_critic`; `brev/train.sh --plan-target q|rollout` writes `"plan_target"` into `config.json`.

- [ ] **Step 1: Fixture** in `_smoke_train`

In `StubEnv.__init__` change `self._env = SimpleNamespace(env=None)` to `self._env = SimpleNamespace(env=SimpleNamespace())` (the shadow sets `ignore_done` on it; `object_layout` / `object_state` are monkeypatched already).

- [ ] **Step 2: Write the failing test** (after `test_train_under_plus_resets_with_proprio_stores_features_and_logs_per_kind`)

```python
def test_train_with_rollout_labels_stores_them_logs_the_diagnostics_and_records_the_fidelity_gate(tmp_path, monkeypatch):
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    from script.lingbot_rl_config import plus_train_variants
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    labelled = []
    monkeypatch.setattr(train, "calibrate", lambda task_id, inner, layout, initial: {"task_id": task_id, "door_closed": 0.0, "door_open": -2.094})
    monkeypatch.setattr(train, "shadow_fidelity",
                        lambda env, shadow, layout, act, observation: {"position_m": 2e-4, "door_rad": 1e-3, "chunks": 20})
    monkeypatch.setattr(train, "label_plans", lambda shadow, live, candidates, first_chunk, norm, layout, calibration: labelled.append(
        (tuple(candidates.shape), first_chunk, calibration["task_id"])) or np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], np.float32))
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base",
                      proprio=True, plus_train=plus_train_variants(), plan_target="rollout")
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    envs = out["envs"]
    strip = lambda env: {key: value for key, value in env.items() if key != "obs_type"}
    assert len(envs) == 2 + 2 * 5 and all(envs[i]["obs_type"] == "pixels" and strip(envs[i]) == strip(envs[i - 1]) for i in range(1, 12, 2))
    assert labelled and labelled[0] == ((2, K_CANDIDATES, HORIZON, ACTION_DIM), True, 9) and len(labelled) == 5
    rows = [row for row in out["rows"] if float(row["is_expert"]) == 0.0]
    assert rows[0]["plan_labels"].dtype == np.float32 and rows[0]["plan_labels"].tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    updates = [entry for entry in out["logs"] if "plan_critic_loss" in entry]
    assert updates and {"label_agreement_with_q", "label_spread_within_plan", "label_spread_across_plans", "label_executed_minus_mean",
                        "label_seconds", "plan_target_q_mean", "plan_label_vs_q_agreement"} <= set(updates[0])
    assert any(key.startswith("label_seconds/") for key in updates[0]) and updates[0]["label_seconds"] >= 0.0
    assert updates[0]["label_spread_within_plan"] == pytest.approx(float(np.std([0.0, 1.0, 2.0, 3.0])))
    assert updates[0]["label_spread_across_plans"] == pytest.approx(2.0) and abs(updates[0]["label_executed_minus_mean"]) == pytest.approx(2.0)
    assert updates[0]["label_agreement_with_q"] in (0.0, 1.0)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["shadow_fidelity"]["chunks"] == 20 and settings["progress_calibration"]["door_open"] == -2.094
    assert out["resume"]["recipe"]["plan_target"] == "rollout" and out["resume"]["layout"]["plan_target"] == "rollout"
    plain = _smoke_train(tmp_path / "plain", monkeypatch, 24)
    assert not any("label_seconds" in entry for entry in plain["logs"])
    assert json.loads((tmp_path / "plain" / "settings.json").read_text())["shadow_fidelity"] is None
```

The stub env ends every episode after 12 steps, so a budget of 60 gives five episodes (reset kinds 0, 1, 0, 2, 1 for seed 42) and one update at chunk 4; `envs` alternate live / shadow after the probe pair, and the shadow's kwargs equal the live ones except `obs_type`. `label_executed_minus_mean` is ±2.0 depending on which plan the untrained critic picks.

- [ ] **Step 3: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k rollout_labels_stores`
Expected: FAIL with `KeyError: 'plan_labels'` (or `assert len(envs) == 12` failing with 6).

- [ ] **Step 4: Implement** in `script/lingbot_rl_train.py`

`_update_from_buffer` (lines 370–371):

```python
            info.update(model.update_plan_critic(
                state, model.plan_input(sample["plan_pools_all"], sample["plan_struct_all"]), sample["z_all"], sample["plan_a_base_all"],
                sample["plan_labels"]))
```

Probe block (replace lines 479–488):

```python
    layout = calibration = fidelity = None
    if config.proprio:
        rng = capture_rng()
        env = make_env(tasks[0]["env_task"], suite, init_states=False, proprio=True)
        try:
            observation, _ = env.reset(seed=config.seed)
            layout = object_layout(env._env.env)
            if config.plan_target == "rollout":
                calibration = calibrate(tasks[0]["task_id"], env._env.env, layout, object_state(env._env.env, layout))
                shadow = make_shadow(tasks[0]["env_task"], suite, False, False, config.seed)
                try:
                    policy.reset()
                    fidelity = shadow_fidelity(env, shadow, layout, lambda current: decode_action(
                        policy.select_action(rl_observation_batch(current, tasks[0]["instruction"], device, True)), norm), observation)
                finally:
                    shadow.close()
                    policy.reset()
        finally:
            env.close()
        restore_rng(rng)
```

(`decode_action` is already imported at the top of `train()`; `policy.eval_candidates` / `eval_plans` are set before this block and `eval_plan_ranker` defaults to `"v_plan"`, so the probe rollout is the prior with the zero-initialised residual, exactly the `run_rl_episode` action path.) Settings (line 500):

```python
    _write_json(output_dir / "settings.json", {"config": config.to_dict(), "recipe": recipe, "runtime": runtime, "object_layout": layout,
                                               "progress_calibration": calibration, "shadow_fidelity": fidelity})
```

Episode head (lines 557–563) becomes:

```python
        sim_task = task["env_task"] if variant is None else variant
        env = make_env(sim_task, suite, init_states=stored, plus=bool(config.plus_train), proprio=config.proprio)
        if stored:
            env.init_state_id = int(draw * len(env._init_states))
        policy.reset()
        observation, _ = env.reset(seed=config.seed + env_steps)
        shadow = make_shadow(sim_task, suite, stored, bool(config.plus_train), config.seed + env_steps) if calibration is not None else None
```

After the decode branches (line 579, `decoded = policy.decode_candidates(batch, k=config.k_candidates)`), still inside the chunk loop:

```python
                plan_labels = None
                if shadow is not None:
                    label_started = time.perf_counter()
                    plan_labels = label_plans(shadow, env._env.env, decoded["plan_a_base_all"], decoded["first_chunk"], norm, layout, calibration)
                    label_seconds = time.perf_counter() - label_started
```

Replace the `q_all` expression and the `plan_info` dict (lines 600–606) with:

```python
                        critic_all = model.critic_state(flat_state, plans_in.repeat_interleave(k, dim=0))
                        action_all = apply_residual(flat_base, model.actor(flat_state, noise.repeat(plans, 1, 1), flat_base))
                        q_all = model.critic(critic_all, action_all).reshape(plans, k).max(dim=1).values
                        v_all = model.plan_critic(state.expand(plans, -1), plans_in).reshape(-1)
                        plan_info = {"plan_agreement": float(int(v_all.argmax()) == int(q_all.argmax())),
                                     "plan_regret": float(q_all.max() - q_all[decoded["plan_index"]]),
                                     "plan_value_spread": float(v_all.max() - v_all.min()),
                                     "plan_random_rate": float(random_plan)}
                        if plan_labels is not None:
                            plan_mean = plan_labels.mean(axis=1)
                            q_mean = torch.stack(model.critic(critic_all, action_all, return_all=True)).mean(dim=0).reshape(plans, k).max(dim=1).values
                            plan_info.update({
                                "label_agreement_with_q": float(int(plan_mean.argmax()) == int(q_mean.argmax())),
                                "label_spread_within_plan": float(plan_labels.std(axis=1).mean()),
                                "label_spread_across_plans": float(plan_mean.std()),
                                "label_executed_minus_mean": float(plan_mean[decoded["plan_index"]] - plan_mean.mean()),
                                "label_seconds": label_seconds})
```

(`critic_all` / `action_all` are the same two expressions the old `q_all` line nested, so `q_all` is bit-identical; `v_all` keeps the min readout for the agreement diagnostic, as before.) In the row (lines 634–655) add, after the `"variant_index": env_task,` line:

```python
                    **({"plan_labels": plan_labels} if plan_labels is not None else {}),
```

The `finally` (line 677):

```python
        finally:
            env.close()
            if shadow is not None:
                shadow.close()
```

- [ ] **Step 5: `brev/train.sh`**

Both usage strings append ` [--plan-target q|rollout]` after `[--plus]`. After `PLUS=0` add `PLAN_TARGET=q`; in the `case` after `--plus) PLUS=1; shift ;;` add `--plan-target) PLAN_TARGET="$2"; shift 2 ;;`. The config JSON python gains `, "plan_target": sys.argv[13]` before the closing `}` and the argument `"$PLAN_TARGET"` after `"$PLUS_TRAIN"`.

- [ ] **Step 6: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && .cache/eval-venv/bin/python -m compileall -q script/ && bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`

```bash
git add script/lingbot_rl_train.py brev/train.sh tests/test_lingbot_rl.py
git commit -m "Label every candidate plan in a shadow simulator under plan_target rollout, store and log the labels, gate the run on shadow fidelity; add --plan-target to train.sh."
```

---

### Task 7: GB10 checks, smoke, stop run A, launch (controller)

Run on Jarvis from `~/Documents/dice-rl-wam` after `git pull`; `DICE_DATA=~/Documents/dice`; the plus install is activated by `brev/train.sh --plus` (`~/Documents/dice/libero-plus-config/activate.sh`). Nothing here is run by implementers. Run A keeps running until Step 4.

- [ ] **Step 1: Smoke with the fidelity gate** (`DICE_NO_SYNC=1` in env.sh as for the previous runs):

```bash
bash brev/train.sh smoke-pool-cf 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --plus --proprio --state-features pool --plan-target rollout --max-env-steps 2100 --no-tmux
```

Copy `$DICE_DATA/runs/dice-t9-plus-m4-pool/expert_features.pt` into `$DICE_DATA/runs/smoke-pool-cf/` first to skip the v3 rebuild (same recipe fingerprint). The run starts with the probe: layout, calibration, then the 20-chunk fidelity rollout of the prior (about 3 minutes). If it raises `shadow fidelity check failed: {...}`, stop and report the printed maxima; do not loosen `FIDELITY_POSITION_M` / `FIDELITY_DOOR_RAD`. Read:

- `$DICE_DATA/runs/smoke-pool-cf/settings.json`: `recipe.plan_target` `rollout`, `recipe.plan_critic` `regression_mean_k_rollout_progress`; `progress_calibration` with `door_closed` 0.0, `door_open` −2.094, `door_initial` nearer `door_open` (task 9 starts with the microwave open; if `door_initial` is instead nearer 0.0, stop and report), `phi_initial` about −‖mug − micro‖ at reset; `shadow_fidelity.position_m` < 1e-3, `door_rad` < 0.02, `chunks` 20 (fewer only if the prior succeeded within 20 chunks).
- Also confirm from the log that the shadow is unconstrained by LIBERO's horizon: with `ignore_done` set the smoke's episodes run their full 520 steps without a `ValueError("executing action in terminated episode")`.
- W&B run `smoke-pool-cf` (or `$DICE_REPO/wandb/latest-run/files/wandb-summary.json`): `label_seconds` (expect about 1 s per chunk; the spec's budget is about 10% on 0.54 s per env step), `seconds_per_env_step` against the run A smoke, `label_agreement_with_q`, `label_spread_within_plan`, `label_spread_across_plans`, `label_executed_minus_mean`, `plan_target_q_mean`, `plan_label_vs_q_agreement`, `peak_gpu_memory_gb`. If `label_seconds` exceeds 2 s, the fallback is to step `shadow._env` directly with `use_camera_obs = False` on the inner env and read success through `shadow._env.check_success()` (the LeRobot `step` needs the camera keys); that is a follow-up change, not something to patch on the box.
- Buffer rows from the resume file:

```bash
$PY - <<'EOF'
import torch, numpy as np
payload = torch.load("/home/$USER/Documents/dice/runs/smoke-pool-cf/resume/latest.pt", map_location="cpu", weights_only=False)
rows = [row for row in payload["replay"]["data"] if float(row["is_expert"]) == 0.0]
labels = np.stack([row["plan_labels"] for row in rows])
print(labels.shape, labels.dtype, labels.mean(), labels.std(), np.abs(labels).max())
print("within", labels.std(axis=2).mean(), "across", labels.mean(axis=2).std(axis=1).mean())
print("expert", next(row for row in payload["replay"]["data"] if float(row["is_expert"]) == 1.0)["plan_labels"].shape)
EOF
```

Expected: `(n_chunks, 4, 4) float32`, non-zero labels with magnitudes around the per-chunk distance change (centimetres) and occasional +2 jumps near success; expert rows `(4, 4)` zeros.

- [ ] **Step 2: Record** the fidelity maxima, calibration, `label_seconds`, `seconds_per_env_step`, memory and the label statistics in `docs/heldout-results.md` next to the run A entry.

- [ ] **Step 3: Stop run A** (keep its run directory and W&B run; it is the reference for the Q-target recipe):

```bash
tmux kill-session -t dice-dice-t9-plus-m4-pool
```

- [ ] **Step 4: Launch** the replacement run on Jarvis with run A's flags plus `--plan-target rollout`:

```bash
bash brev/train.sh dice-t9-plus-m4-pool-cf 9 --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4 --plus --proprio --state-features pool --plan-target rollout
```

Copy `expert_features.pt` into `$DICE_DATA/runs/dice-t9-plus-m4-pool-cf/` first. Checkpoints every 80k; the offline measurements of spec section 5 run on the stored `plan_labels` at 80k and 320k.

## Self-review notes

- Spec coverage: section 2 run / recipe → Task 7 launch command and Task 1 validation; label per chunk (shadow, 16 candidates, base actions, physics only, φ difference, mean over K, (M, K) float32 stored) → Tasks 5 (`label_plans`), 3 (buffer), 6 (storage) and 4 (mean over K as the target); φ for task 9 with startup calibration → Task 2 (door limits from the model, interior = microwave body position per the brief); critic unchanged architecture, label target, mean readout, Q target still logged → Task 4; what stays fixed → no task touches the actor, Q updates, sampler, reset draw or evaluation protocols; shadow per episode with the live kwargs, reset once, state-driven, live never restored → Tasks 5 and 6 (`make_shadow`, `sync_shadow`, per-episode construction); logged diagnostics → Task 6 (`label_agreement_with_q`, within / across spreads, executed minus mean, `label_seconds`, per-episode logs unchanged); cost measured in the smoke → Task 7; section 4 fidelity check (20 chunks, 1 mm, door within 0.02 rad as the brief's substitute for 1°) → Tasks 5 and 6 with the controller goal copied along with the state as section 4 anticipates; section 6 code list → Tasks 1–6 and `train.sh --plan-target`; section 6 tests (progress on hand-built poses, `label_plans` on a stub environment, the plan-critic update on stored labels, fingerprint, default recipe unchanged) → Tasks 2, 5, 4, 1 and the `plain` smoke assertions in Task 6. Not in this plan: the offline measurements and held-out evaluations of section 5, which the spec schedules after 80k / 320k.
- Placeholder scan: every code step carries the code; the only deferred item is the `use_camera_obs` fallback, named as a follow-up in Task 7 with the reason it is not patched on the box.
- Type consistency: `label_plans(shadow, live, candidates, first_chunk, norm, layout, calibration)` has the same signature in Task 5, the Task 6 call site and the Task 6 monkeypatch; `shadow_fidelity(env, shadow, layout, act, observation, chunks)` in Task 5 and the Task 6 call and monkeypatch (the call omits `chunks`, so the monkeypatch's five-argument lambda matches); `calibrate(task_id, inner, layout, initial)` in Tasks 2, 6; `progress(calibration, layout, before, after, success)` in Tasks 2, 5; `update_plan_critic(..., plan_labels=None)` in Tasks 4, 6; `make_shadow(task_id, suite, init_states, plus, seed)` in Tasks 5, 6; `plan_labels` is numpy (M, K) float32 from `label_plans`, in the row, and torch (B, M, K) from `sample`.
- Bit-for-bit with `plan_target "q"`: the protocol and checkpoint `layout` are unchanged; `rank_plans` executes the same `plan_critic(state.expand, pools)` min readout; `update_plan_critic` computes the same target and returns the same keys; the loop's `q_all` is the same expression split over two names; `plan_labels` enters the row only under rollout and the buffer default is zeros that the `"q"` update ignores; the shadow, calibration and fidelity code runs only when `calibration is not None`, which requires `plan_target == "rollout"`; the probe block only binds the reset's return value.
- RNG: `make_shadow` resets inside `capture_rng()` / `restore_rng()`; the startup fidelity rollout runs inside the probe's existing RNG capture; `label_plans` draws nothing.
- Fingerprint: `plan_target` enters `protocol()` whenever it is `"rollout"`, so `test_load_resume_rejects_recipe_mismatch` refuses a resume across the change; `test_config_plan_target_enters_the_fingerprint_only_for_rollout` pins the default protocol at 78 keys.
- Verified from source: robosuite `MjSim.get_state` / `set_state` / `forward`, `Controller.update(force=True)`, OSC `set_goal` goal_ori behaviour and `reset_goal`, `PandaGripper.format_action` ramp (`~/.cache/uv/git-v0/checkouts/79abe6cb204f004b/c848ca8`); robosuite `MujocoEnv.step` done check; LIBERO `ControlEnv(horizon=1000, ignore_done=False)`, `regenerate_obs_from_state` pattern, task 9 bddl `(Open microwave_1)` init and `(In white_yellow_mug_1 microwave_1_heating_region) (Close microwave_1)` goal, `microwave.xml` joint range `-2.094 0` and heating-region site (LIBERO copy at `/Users/desmondzee/Hardware/ICL-so101/third_party/LIBERO`, not the pinned GB10 install); LeRobot `LiberoEnv.step` / `reset` / `_format_raw_obs` (`.cache/lerobot`).
- Box-only checks (Task 7): the shadow's determinism against the live env (the fidelity maxima), the recorded door calibration direction, the LIBERO-Plus install's `horizon` / `ignore_done` values, and the label cost.
