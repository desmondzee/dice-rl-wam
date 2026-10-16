# Plan critic trained on counterfactual simulator rollouts: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace run A (`dice-t9-plus-m4-pool`) by `dice-t9-plus-m4-pool-cf`, the same recipe with `plan_target = "rollout"`: every one of the M × K candidate chunks is executed in a shadow simulator while the live chunk runs, the per-candidate progress of the task 9 object poses is stored as `plan_labels` (M, K), and V_plan regresses onto the mean label per plan instead of the Q-derived target, ranking by the mean over its heads.

**Architecture:** The progress function and its startup calibration live in a new numpy-only module `script/lingbot_rl_progress.py`. The trainer gains environment helpers (`make_shadow`, `capture_live`, `sync_shadow`, `label_plans`, `shadow_fidelity`): a second `LiberoEnv` with the live environment's kwargs is reset once per episode with its image observables disabled, then driven only by `sim.set_state` plus a copy of the OSC goal, nullspace reference and gripper ramp taken from a snapshot of the live environment; candidates are stepped on the shadow's robosuite environment directly (physics only, no rendering) on a one-worker thread that overlaps the live chunk, and every row waits for its labels before it is stored. The model carries `plan_target`; under `"rollout"` `update_plan_critic` regresses onto `plan_labels.mean(-1)` and still computes the Q target for logging, and `rank_plans` reads the mean over heads. The buffer stores `plan_labels` with a legacy default of zeros. A startup fidelity check (live env through the LeRobot wrapper, shadow through the direct path) gates the run and its maxima go to `settings.json`.

**Tech Stack:** Torch 2.11 (cu130 on the GB10s), the pinned LeRobot LingBot-VA port and its `LiberoEnv`, robosuite's MuJoCo binding (`sim.get_state` / `set_state` / `forward`, OSC `goal_pos` / `goal_ori` / `initial_joint`, Panda gripper `current_action`, `Observable.set_enabled`), `concurrent.futures.ThreadPoolExecutor`, numpy replay rows, pytest with the `_StubPolicy` / `_smoke_train` fixtures in `tests/test_lingbot_rl.py`, LIBERO-Plus at `4976dc3` on the GB10s.

**Spec:** docs/superpowers/specs/2026-10-16-counterfactual-plan-critic-design.md

## Global Constraints

- Zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests.
- No secrets in logs, argv, metadata or artifacts.
- The default recipe (`plan_target` `"q"`) must reproduce bit-for-bit: no new RNG draws on the default path, no behaviour change in `rank_plans` / `update_plan_critic` when `"q"`, no change to the default `protocol()` (78 keys), no change to the default checkpoint `layout` dict, no thread pool creation and no shadow construction on the default path.
- The resume fingerprint changes with `plan_target` (`test_load_resume_rejects_recipe_mismatch` is the refusal mechanism).
- The run name is `dice-t9-plus-m4-pool-cf`; run B (`dice-t9-plus-m4-struct`, Watson) is untouched.
- What stays fixed (spec section 2): the residual actor, the Q ensemble and its targets, the sampler, M = 4, K = 4, the reset schedule, the buffer's other keys, both evaluation protocols.
- Do not edit `.cache/lerobot`; do not edit `script/lingbot_eval.py` (its `harness_sha256` enters evaluation provenance); do not instantiate Hydra `DistillResidualRLModel`; tests must not call `modal run` or `.remote()`.
- No GPU jobs, Modal, ssh or remote machines from implementers; Task 8 is run by the controller on Jarvis.
- `store_tokens` changes storage only: never in the protocol fingerprint, never in the replay buffer; with it off there is no sidecar, no `proj_out` capture and no extra transformer work.
- Verify every task with `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py`, `.cache/eval-venv/bin/python -m script.lingbot_rl_config`, `.cache/eval-venv/bin/python -m compileall -q script/`, `bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`.

## Review Focus

1. A shadow stepped more than LIBERO's `horizon` (1000) raises `ValueError("executing action in terminated episode")` from robosuite unless `ignore_done` is set, and a shadow with image observables enabled renders two cameras on every substep: Task 5 tests that `make_shadow` sets `ignore_done = True` and disables exactly the image-modality observables (a shadow executes about 8,000 steps per episode).
2. A candidate that reaches success mid-chunk must stop there and carry +2, like the live loop stops at success: Task 5 tests the step count and label of a succeeding candidate.
3. The door term must not reward closing the door while the mug is outside (progress myopia, spec section 7): Task 2 tests that a door change with the mug outside has zero progress.
4. A rollout checkpoint evaluated by a model built for the Q readout would rank with the min over heads: Task 4 tests that `layout` carries `plan_target` and that loading it into a `"q"` model raises `layout mismatch`.
5. A shadow whose qpos layout or object identities differ from the live environment (a LIBERO-Plus variant) would mislabel silently: Task 5 tests that `label_plans` raises `shadow state mismatch` when the synced shadow's object state differs from the live one, and Task 6 runs the fidelity check on a layout variant with `plus=True` as well as on the base task.

## Decisions the spec left open (binding for every task)

- **Door calibration.** The spec says the door range is measured at startup; the brief proposed "closed = the limit nearer the initial value of stored state 0". Task 9's bddl initialises `(Open microwave_1)` (`libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl` line 76) and `microwave.xml` gives `microjoint` the range `-2.094 0` with LIBERO's `default_close_ranges = [-0.005, 0.0]`, so the nearer-to-initial rule would label the open limit as closed. Decision: `closed` = the `jnt_range` limit nearer zero (the articulated object's rest pose), `open` = the other; the initial door value and `phi_initial` are recorded in `settings.json` under `progress_calibration`, and Task 8 checks on the box that `door_initial` lies nearer `door_open`.
- **Interior target.** As the brief decided: `micro` = `microwave_1` body position, `inside` = ‖mug − micro‖ < 0.12 m (`INSIDE_RADIUS`). Geometry caveat (recorded again in Task 8): relative to the microwave body origin the cavity spans x ∈ [−0.1435, 0.0635], y ∈ [−0.088, 0.0805] with the floor at z = 0.024, so the 0.12 m ball from the mug origin misses about 2.5 cm of the far floor and includes about 3 cm beyond the door opening; the +2 success term carries the goal. A site-based target is a later refinement, not part of this plan.
- **Progress signature.** `calibrate(task_id, inner, layout, initial) -> dict` (task-specific names, door limits, `door_initial`, `phi_initial`), `phi(calibration, layout, state, success) -> float`, `progress(calibration, layout, before, after, success) -> float` where `before` has `success=False`. The task is selected in `calibrate` through `TASKS`; unknown tasks raise `ValueError` at startup.
- **Live snapshot and shadow sync.** The worker thread must never read the live environment while the main thread steps it, so the main thread captures `capture_live(live) -> {"state", "warmstart", "robots"}`: `sim.get_state()` (time, qpos, qvel), a copy of `data.qacc_warmstart`, and per robot copies of the OSC `goal_pos`, `goal_ori`, `initial_joint` (the nullspace reference, drawn with noise at reset) and the gripper's `current_action` (the Panda gripper ramps its command by 0.01 per substep, so it is real controller state). `sync_shadow(snapshot, twin)` writes them into the shadow (`set_state`, `qacc_warmstart`, `forward`, `controller.update(force=True)`, then the controller and gripper fields). The live goal is copied rather than `reset_goal` because OSC `set_goal` keeps `goal_ori` across steps when the rotation delta is zero. `use_delta` is already True on both after `reset`.
- **Shadow stepping (physics only).** `label_plans` steps the shadow's robosuite environment directly, `inner.step(action)` on `shadow._env.env`, and reads success from `inner._check_success()`. This mirrors the LeRobot wrapper exactly: `LiberoEnv.step` (`.cache/lerobot/src/lerobot/envs/libero.py:366-391`) asserts `action.ndim == 1`, forwards the (7,) float32 action unchanged to `OffScreenRenderEnv.step`, which is `self.env.step(action)` on the robosuite env, then reports `terminated = done or self._env.check_success()` where `check_success` is `self.env._check_success()`; there is no scaling, clipping or gripper sign handling, and relative control is the `robot.controller.use_delta = True` set by `LiberoEnv.reset` (persisting until the next reset). Rendering is avoided by disabling the image-modality observables once in `make_shadow`: robosuite renders cameras inside `Observable.update` (called from `_update_observables` on every substep of `MujocoEnv.step`), and `use_camera_obs` is only read when observables are created, so setting it after reset would not stop rendering; `observable.set_enabled(False)` does (`utils/observables.py:214-224`), and the shadow is not reset again afterwards.
- **Labelling thread.** One `ThreadPoolExecutor(max_workers=1)` per run, created only under rollout. After `select_plan` returns, the main thread captures the snapshot and submits `_timed(label_plans, ...)`; the live chunk executes as today; `future.result()` is called before `buffer.add_online`, so no row is stored without its labels and the training order is unchanged. `label_seconds` is the future's own wall time, `label_wait_seconds` the time blocked in `result()`; both enter `plan_info`. The wait sits inside the region timed for `seconds_per_env_step`. MuJoCo's bindings release the GIL in `mj_step` and the shadow owns its MjModel / MjData, so the overlap is real. On the way out of an episode a pending future is waited for (`concurrent.futures.wait`) before the shadow is closed.
- **Q reference for the agreement diagnostics.** In the loop, `label_agreement_with_q` compares the label argmax with the argmax over plans of the max over K of the mean over the ten Q heads (a second `critic(..., return_all=True)` call on the same inputs, only under rollout, computed before the live chunk; the existing `q_all` min-readout path is untouched). In `update_plan_critic`, `plan_label_vs_q_agreement` compares the label argmax with the argmax of the Q-derived target (max over K of the min over heads), the quantity the critic used to train on.
- **Fidelity check.** `shadow_fidelity(env, shadow, layout, act, observation, chunks=20)` drives the live probe env through the LeRobot wrapper with `act(observation) -> (7,) env action`, chunk lengths from `env_action_count(first)` (12 then 16), syncs the shadow from a fresh snapshot at every chunk start, steps the shadow through the direct path with the same action, and compares `object_state` after each chunk: per body the position gap and the angle `2·arccos(min(1, |q_live·q_shadow|))` on the quaternion slice, and the joint gap; it raises `RuntimeError` when the maximum position gap is ≥ `FIDELITY_POSITION_M = 1e-3` or the maximum of the angles and joint gaps is ≥ `FIDELITY_ANGLE_RAD = 0.01745` (1°). A correct sync lands near 1e-12, so the tight gate catches missed state. The trainer's `act` is the prior through `policy.select_action` on `rl_observation_batch`, as `run_rl_episode` does. It runs twice at startup: 20 chunks on the base task and, when `plus_train` is set, 3 chunks on the first layout training variant with `init_states=True, plus=True`.
- **Shadow construction.** `make_shadow(task_id, suite, init_states, plus, seed)` builds `make_env(task_id, suite, init_states=init_states, plus=plus)` (obs_type `"pixels"`), resets it inside `capture_rng()` / `restore_rng()` (LIBERO's reset seeding must not touch the collection stream), sets `shadow._env.env.ignore_done = True` and disables the image observables. One shadow per episode, reset with the live seed `config.seed + env_steps`; the startup probe's shadow uses `plus=False` and `init_states=False` like the probe env.
- **Checkpoint layout.** `inference_state_dict()["layout"]` gains `"plan_target"` only when it is not `"q"`; `layout_of` returns it only when present; `load_inference_state_dict` compares `layout.get("plan_target", "q")` with the model's. `evaluate()` and `docs/libero-plus/plus_eval.py` already apply `**layout` to the config / model, so a rollout checkpoint evaluates with the mean readout without further edits. `script/lingbot_eval_modal.py` builds its eval config without `plan_candidates`, so a rollout checkpoint fails validation there; that is pre-existing for every M = 4 checkpoint and out of scope.
- **Protocol.** Under rollout `protocol()` adds `"plan_target": "rollout"` and overrides `"plan_critic"` with `"regression_mean_k_rollout_progress"`; the default protocol is byte-identical.
- **Validation.** `plan_target == "rollout"` requires `plan_candidates > 1` and `proprio` (object poses are read only when `proprio`).
- **Sharpening blocks.** `replay_sharpening` and `_maybe_sharpen` keep calling `plan_critic` with the min readout; they are diagnostics, not ranking.
- **Token sidecar (Task 7).** The stored pools and `state_tokens` come from the history-free critic pass, so the 128 history-conditioned tokens of each plan cannot be recomputed from the stored latents; under `store_tokens` `select_plan(return_tokens=True)` returns the conditional rows of the last batched denoising pass (`captured[0][:M]`, CFG batch 2M with conditional rows first) as `plan_tokens` (M, 128, 3072) bf16 on CPU. They never enter the replay buffer: the Task 6 worker thread writes them to `<run>/tokens/shard_<n>.npy` (a valid `.npy` of shape (256, M, 128, 3072) int16 bf16 bit patterns, created sparse by `open_memmap`, one slot per chunk) and appends one `index.jsonl` line (`chunk`, `episode`, `step`, `reset_kind`, `variant_index`, `plan_index`, `shard`, `offset`). `TokenStore(root, chunks)` at startup truncates the index to the replay's chunk count so `len(index) == chunks` always holds; `append` refuses a `chunk` that is not the next ordinal; write failures propagate through the future before the row is stored. About 3.1 MB per chunk, about 63 GB per 320k run on the 3.6 TB NVMe. `store_tokens` is not in the protocol, so a run can be resumed with or without it.

## File map

| file | responsibility after this plan |
|---|---|
| `script/lingbot_rl_config.py` | `plan_target`; validation; protocol keys under rollout |
| `script/lingbot_rl_progress.py` | `TASKS`, `INSIDE_RADIUS`, `body_position`, `joint_value`, `calibrate`, `phi`, `progress` |
| `script/lingbot_rl_buffer.py` | `plan_labels` (M, K) float32 with a zeros default |
| `script/lingbot_rl_model.py` | `plan_target` on the model; rollout target and diagnostics; mean readout in `rank_plans`; `plan_target` in `layout` |
| `script/lingbot_rl_train.py` | `make_shadow`, `capture_live`, `sync_shadow`, `label_plans`, `shadow_fidelity`, `_timed`; the probe calibration and fidelity gates; per-episode shadow and the labelling thread; labels stored and logged |
| `script/lingbot_rl_tokens.py` | `TokenStore`: sharded `.npy` + `index.jsonl` sidecar for the plan tokens |
| `script/lingbot_rl_policy.py` | `_video_pass_batched(capture=)`, `select_plan(return_tokens=)` returning `plan_tokens` |
| `brev/train.sh` | `--plan-target q|rollout`, `--store-tokens` |

---

### Task 1: `plan_target` in the config and the fingerprint

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

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

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

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

`phi_initial` validates every name in `TASKS[task_id]` against the layout at startup (a missing body raises `ValueError` from `list.index`). `inner.sim.model.jnt_range` is the (njnt, 2) MuJoCo array exposed by robosuite's `MjModel` delegate; `joint_name2id` is the robosuite binding method (`utils/binding_utils.py:378`) next to `get_joint_qpos_addr`.

- [ ] **Step 4: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_progress.py tests/test_lingbot_rl.py
git commit -m "Task 9 progress: mug-to-microwave distance, inside and door-closed terms, success bonus, door range from the rest limit."
```

---

### Task 3: `plan_labels` in the replay buffer

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

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

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

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

### Task 5: Shadow environment helpers: `make_shadow`, `capture_live`, `sync_shadow`, `label_plans`, `shadow_fidelity`

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

**Files:**
- Modify: `script/lingbot_rl_train.py` (imports lines 13–24; constants after line 27; new functions after `object_state`, line 145)
- Test: `tests/test_lingbot_rl.py` (fixtures, two tests)

**Interfaces:**
- Consumes: `calibrate` / `progress` (Task 2); `object_state(inner, layout)`, `env_success(info)`, `capture_rng` / `restore_rng`, `make_env`; `slice_env_actions`, `env_action_count` (already imported from `script.lingbot_rl_policy`); `decode_action` from `script.lingbot_eval`.
- Produces: `FIDELITY_CHUNKS = 20`, `FIDELITY_POSITION_M = 1e-3`, `FIDELITY_ANGLE_RAD = 0.01745`; `make_shadow(task_id, suite, init_states, plus, seed) -> LiberoEnv` (reset once inside an RNG capture, `ignore_done = True` and image observables disabled on `shadow._env.env`); `capture_live(live) -> {"state", "warmstart", "robots": [(goal_pos, goal_ori, initial_joint, gripper_action), ...]}`; `sync_shadow(snapshot, twin)`; `label_plans(shadow, snapshot, before (object_state of the live env), candidates (M, K, 16, 30) tensor, first_chunk, norm, layout, calibration) -> np.float32 (M, K)` raising `RuntimeError("shadow state mismatch")` when the synced shadow's object state differs from `before`; `shadow_fidelity(env, shadow, layout, act, observation, chunks=FIDELITY_CHUNKS) -> {"position_m", "angle_rad", "chunks"}` raising `RuntimeError` above the thresholds.

- [ ] **Step 1: Fixtures** in `tests/test_lingbot_rl.py` (before `_StubPolicy`)

```python
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
        self.sim.data.qpos[:3] += 0.01 * action[:3] + self.drift
        self.sim.data.qpos[3] += 0.5 * max(float(action[6]), 0.0)
        self.sim.forward()
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
```

`_TwinInner` plays the robosuite env (direct `step` returning the 4-tuple, `_check_success`, `_observables`, `robots`); `_TwinEnv` plays the LeRobot wrapper (`step` mirrors `LiberoEnv.step`: forward the action, `terminated = done or success`, `info["is_success"]`). The real `object_state` reads body 0 (the microwave at the origin), body 1 (the mug at `qpos[:3]`) and `qpos[3]` (the door, open at −2.0; a positive gripper command closes it by 0.5 per step and success fires at 0.0). Quaternions are identity so the orientation gap is exactly zero.

- [ ] **Step 2: Write the failing tests** (after the fixtures)

```python
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
    assert labels[0, 0] == pytest.approx(-0.12, abs=1e-4) and labels[0, 1] == labels[0, 0]
    assert labels[1, 0] == pytest.approx(0.12, abs=1e-4) and labels[1, 1] == pytest.approx(2.04, abs=1e-4)
    assert shadow._env.env.steps == 12 + 12 + 12 + 4 and live._env.env.steps == 0 and live._env.env.sim.data.qpos[0] == 0.3
    assert float(shadow._env.env.robots[0].controller.initial_joint[0]) == 0.5
    later = label_plans(shadow, snapshot, before, candidates[:, :1], False, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)
    assert later[0, 0] == pytest.approx(-0.16, abs=1e-4) and later[1, 0] == pytest.approx(0.16, abs=1e-4)
    with pytest.raises(RuntimeError, match="shadow state mismatch"):
        label_plans(shadow, snapshot, before + 1.0, candidates, True, norm, _TWIN_LAYOUT, _TWIN_CALIBRATION)


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
```

Expected values: a first chunk executes 12 steps of 0.01 m, so the mug moves from 0.3 to 0.42 (label −0.12) or 0.18 (+0.12); `decode_action` with the unit normalisation scales by `1 + 5e-7`, hence `abs=1e-4`. Candidate (1, 1) closes the door at 0.5 rad per step and succeeds after four steps: Δ(−distance) = +0.04, plus 2.0. The second call uses `first_chunk=False` (16 steps). The mismatch case offsets `before` by 1.0. With a drift of 2e-4 m per step the shadow is 2.4e-3 m off after the first chunk, above `FIDELITY_POSITION_M`.

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "label_plans_scores or shadow_fidelity_measures"`
Expected: FAIL with `ImportError: cannot import name 'capture_live'`.

- [ ] **Step 4: Implement** in `script/lingbot_rl_train.py`

Imports: add `from script.lingbot_rl_progress import calibrate, progress` after the `script.lingbot_rl_policy` import (line 23). After `SHARPEN_EVERY_STEPS = 5_000` add:

```python
FIDELITY_CHUNKS = 20
FIDELITY_POSITION_M = 1e-3
FIDELITY_ANGLE_RAD = 0.01745
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
    inner = shadow._env.env
    inner.ignore_done = True
    for observable in inner._observables.values():
        if observable.modality == "image":
            observable.set_enabled(False)
    return shadow


def capture_live(live):
    return {"state": live.sim.get_state(), "warmstart": np.array(live.sim.data.qacc_warmstart),
            "robots": [(np.array(robot.controller.goal_pos), np.array(robot.controller.goal_ori),
                        np.array(robot.controller.initial_joint), np.array(robot.gripper.current_action)) for robot in live.robots]}


def sync_shadow(snapshot, twin):
    twin.sim.set_state(snapshot["state"])
    twin.sim.data.qacc_warmstart[:] = snapshot["warmstart"]
    twin.sim.forward()
    for robot, (goal_pos, goal_ori, initial_joint, gripper_action) in zip(twin.robots, snapshot["robots"]):
        robot.controller.update(force=True)
        robot.controller.goal_pos = np.array(goal_pos)
        robot.controller.goal_ori = np.array(goal_ori)
        robot.controller.initial_joint = np.array(initial_joint)
        robot.gripper.current_action = np.array(gripper_action)


def label_plans(shadow, snapshot, before, candidates, first_chunk, norm, layout, calibration):
    from script.lingbot_eval import decode_action
    twin = shadow._env.env
    plans, k = candidates.shape[:2]
    env_actions = slice_env_actions(candidates.detach().float().cpu().reshape(plans * k, HORIZON, ACTION_DIM), first_chunk)
    labels = np.zeros(plans * k, np.float32)
    for index in range(plans * k):
        sync_shadow(snapshot, twin)
        if index == 0 and not np.allclose(before, object_state(twin, layout), atol=1e-6):
            raise RuntimeError("shadow state mismatch")
        success = False
        for step in range(env_actions.shape[1]):
            _, _, done, _ = twin.step(decode_action(env_actions[index, step].reshape(1, 7), norm))
            success = bool(twin._check_success())
            if success or done:
                break
        labels[index] = progress(calibration, layout, before, object_state(twin, layout), success)
    return labels.reshape(plans, k)


def shadow_fidelity(env, shadow, layout, act, observation, chunks=FIDELITY_CHUNKS):
    live, twin = env._env.env, shadow._env.env
    bodies = 7 * len(layout["bodies"])
    worst = np.zeros(2)
    for chunk in range(chunks):
        sync_shadow(capture_live(live), twin)
        for _ in range(env_action_count(chunk == 0)):
            action = act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            twin.step(action)
        live_state, twin_state = object_state(live, layout), object_state(twin, layout)
        poses = np.stack([live_state[:bodies].reshape(-1, 7), twin_state[:bodies].reshape(-1, 7)]).astype(np.float64)
        position = np.abs(poses[0, :, :3] - poses[1, :, :3]).max()
        angle = (2.0 * np.arccos(np.minimum(1.0, np.abs((poses[0, :, 3:] * poses[1, :, 3:]).sum(axis=1))))).max()
        joint = np.abs(live_state[bodies:] - twin_state[bodies:]).max(initial=0.0)
        worst = np.maximum(worst, [position, max(angle, joint)])
        if env_success(info) or terminated or truncated:
            break
    result = {"position_m": float(worst[0]), "angle_rad": float(worst[1]), "chunks": chunk + 1}
    if worst[0] >= FIDELITY_POSITION_M or worst[1] >= FIDELITY_ANGLE_RAD:
        raise RuntimeError(f"shadow fidelity check failed: {result}")
    return result
```

Source facts (robosuite at `~/.cache/uv/git-v0/checkouts/79abe6cb204f004b/c848ca8`): `MjSim.get_state()` returns `MjSimState(time, qpos copy, qvel copy)` and `set_state` copies them back (`utils/binding_utils.py:1140-1155`); the `na == 0` assertion lives in `MjSimState.from_flattened` (`binding_utils.py:236`), the path LIBERO's `set_init_state` uses, so the model has no actuator activations to carry; `forward()` is `mj_forward`; `MjData` delegates `qacc_warmstart` to `mujoco.MjData` (`binding_utils.py:555-574`); `Controller.update(force=True)` recomputes `ee_pos` / `ee_ori_mat` from the sim (`controllers/base_controller.py:125-161`); OSC `set_goal` only rewrites `goal_ori` when the rotation delta is non-zero (`controllers/osc.py:271-278`), which is why the live goal is copied; `initial_joint` feeds the OSC nullspace torques (`controllers/osc.py:351`) and is set from the joint positions at reset (`base_controller.py:91, 185`); `PandaGripper.format_action` ramps `current_action` by `speed = 0.01` per substep (`models/grippers/panda_gripper.py:43-62`). Robosuite's `MujocoEnv.step` raises once `done` is set, `done = timestep >= horizon and not ignore_done` (`environments/base.py:378, 432`); LIBERO's `ControlEnv` defaults to `horizon=1000, ignore_done=False`, hence `ignore_done = True` on the shadow. Cameras render in `Observable.update` (`utils/observables.py:214-224`, run from `_update_observables` on every substep), and `Observable.modality` / `set_enabled` are the public accessors (`observables.py:289, 396`), so disabling the image observables is what stops rendering; `use_camera_obs` is only consulted when observables are created (`environments/robot_env.py:335, 524`). The direct path `twin.step(action)` is exactly what `LiberoEnv.step` forwards (`.cache/lerobot/src/lerobot/envs/libero.py:366-391`: `ndim == 1` assert, no scaling, `terminated = done or check_success()`).

- [ ] **Step 5: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py && .cache/eval-venv/bin/python -m compileall -q script/`

```bash
git add script/lingbot_rl_train.py tests/test_lingbot_rl.py
git commit -m "Shadow environment helpers: live snapshot and controller sync, physics-only candidate labels, startup fidelity check with pose and orientation gates."
```

---

### Task 6: Trainer integration: labelling thread, storage, logging, `train.sh`

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`) reproduces bit-for-bit, with no thread pool creation and no shadow construction.

**Files:**
- Modify: `script/lingbot_rl_train.py` (imports; `_update_from_buffer` lines 370–371; `train()` probe block 479–488, settings line 500, executor before the `while`, episode head 552–563, chunk loop 572–606, after the env step loop (before line 629 `chunk_seconds += ...`), row 634–655, `finally` 677, after the loop), `brev/train.sh`, `tests/test_lingbot_rl.py` (`_smoke_train` fixture, one test)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `_timed(fn, *args) -> (result, seconds)`; `settings.json` gains `"progress_calibration"` and `"shadow_fidelity"` (None unless rollout; under rollout `{"base": {...}, "plus": {"variant_index", ...} | None}`); a shadow env per episode and one `ThreadPoolExecutor(max_workers=1)` per run under rollout; rows carry `plan_labels` under rollout; `plan_info` gains `label_agreement_with_q`, `label_spread_within_plan`, `label_spread_across_plans`, `label_executed_minus_mean`, `label_seconds`, `label_wait_seconds` (and their `/<kind>` copies under `plus_train`); `_update_from_buffer` passes `sample["plan_labels"]` to `update_plan_critic`; `brev/train.sh --plan-target q|rollout` writes `"plan_target"` into `config.json`.

- [ ] **Step 1: Fixture** in `_smoke_train`

In `StubEnv.__init__` change `self._env = SimpleNamespace(env=None)` to `self._env = SimpleNamespace(env=SimpleNamespace(_observables={}))` (the shadow sets `ignore_done` and walks `_observables` on it; `object_layout` / `object_state` are monkeypatched already).

- [ ] **Step 2: Write the failing test** (after `test_train_under_plus_resets_with_proprio_stores_features_and_logs_per_kind`)

```python
def test_train_with_rollout_labels_overlaps_the_live_chunk_stores_them_and_records_both_fidelity_gates(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import script.lingbot_rl_model as model_module
    import script.lingbot_rl_train as train
    from script.lingbot_rl_config import plus_train_variants
    monkeypatch.setattr(model_module, "HIDDEN", (32, 32, 32))
    labelled, pools = [], []
    monkeypatch.setattr(train, "ThreadPoolExecutor", lambda **kwargs: pools.append(kwargs) or ThreadPoolExecutor(**kwargs))
    monkeypatch.setattr(train, "calibrate", lambda task_id, inner, layout, initial: {"task_id": task_id, "door_closed": 0.0, "door_open": -2.094})
    monkeypatch.setattr(train, "capture_live", lambda live: {"stub": True})
    monkeypatch.setattr(train, "shadow_fidelity", lambda env, shadow, layout, act, observation, chunks=20: {
        "position_m": 2e-4, "angle_rad": 1e-3, "chunks": chunks})
    monkeypatch.setattr(train, "label_plans", lambda shadow, snapshot, before, candidates, first_chunk, norm, layout, calibration: labelled.append(
        (snapshot, before.tolist(), tuple(candidates.shape), first_chunk, calibration["task_id"]))
        or np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], np.float32))
    config = RLConfig(task_ids=(9,), plan_candidates=2, plan_random_prob=0.0, selection_warmup_steps=0, residual_input="base",
                      proprio=True, plus_train=plus_train_variants(), plan_target="rollout")
    out = _smoke_train(tmp_path, monkeypatch, 60, model_factory=lambda **kw: DiceResidualModel(**kw), config=config)
    envs = out["envs"]
    strip = lambda env: {key: value for key, value in env.items() if key != "obs_type"}
    assert len(envs) == 4 + 2 * 5 and all(envs[i]["obs_type"] == "pixels" and strip(envs[i]) == strip(envs[i - 1]) for i in range(1, 14, 2))
    assert (envs[2]["task_id"], envs[2]["init_states"], envs[2]["is_libero_plus"], envs[2]["obs_type"]) == (1982, True, True, "pixels_agent_pos")
    assert pools == [{"max_workers": 1}]
    assert labelled and labelled[0] == ({"stub": True}, [7.0, 8.0], (2, K_CANDIDATES, HORIZON, ACTION_DIM), True, 9) and len(labelled) == 5
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
    assert pools == [{"max_workers": 1}] and not any("label_seconds" in entry for entry in plain["logs"])
    assert json.loads((tmp_path / "plain" / "settings.json").read_text())["shadow_fidelity"] is None
```

The stub env ends every episode after 12 steps, so a budget of 60 gives five episodes (reset kinds 0, 1, 0, 2, 1 for seed 42) and one update at chunk 4. `envs` are the probe pair, the layout-variant pair (variant 1982 with stored states on the plus package), then live / shadow per episode; each shadow's kwargs equal its live env's except `obs_type`. The stub `label_plans` is synchronous but still runs on the executor's thread; `future.result()` joins it before the row is stored, so every assertion sees completed work. `label_executed_minus_mean` is ±2.0 depending on which plan the untrained critic picks. The `plain` run must not add a second executor.

- [ ] **Step 3: Run it**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k rollout_labels_overlaps`
Expected: FAIL with `AttributeError: module 'script.lingbot_rl_train' has no attribute 'ThreadPoolExecutor'`.

- [ ] **Step 4: Implement** in `script/lingbot_rl_train.py`

Imports: add `from concurrent.futures import ThreadPoolExecutor, wait` after `import argparse`'s group (before `from dataclasses import replace`). After `_host_array` (line 330):

```python
def _timed(fn, *args):
    started = time.perf_counter()
    return fn(*args), time.perf_counter() - started
```

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
                act = lambda current: decode_action(
                    policy.select_action(rl_observation_batch(current, tasks[0]["instruction"], device, True)), norm)
                shadow = make_shadow(tasks[0]["env_task"], suite, False, False, config.seed)
                try:
                    policy.reset()
                    fidelity = {"base": shadow_fidelity(env, shadow, layout, act, observation), "plus": None}
                finally:
                    shadow.close()
                    policy.reset()
        finally:
            env.close()
        if calibration is not None and config.plus_train:
            variant = int(config.plus_train["layout"][0])
            env, shadow = make_env(variant, suite, init_states=True, plus=True, proprio=True), make_shadow(variant, suite, True, True, config.seed)
            try:
                env.init_state_id = 0
                observation, _ = env.reset(seed=config.seed)
                policy.reset()
                fidelity["plus"] = {"variant_index": variant, **shadow_fidelity(env, shadow, layout, act, observation, chunks=3)}
            finally:
                env.close()
                shadow.close()
                policy.reset()
        restore_rng(rng)
```

(`decode_action` is already imported at the top of `train()`; `policy.eval_candidates` / `eval_plans` are set before this block and `eval_plan_ranker` defaults to `"v_plan"`, so the probe rollouts use the prior with the zero-initialised residual, exactly the `run_rl_episode` action path; the whole block sits inside the existing RNG capture.) Settings (line 500):

```python
    _write_json(output_dir / "settings.json", {"config": config.to_dict(), "recipe": recipe, "runtime": runtime, "object_layout": layout,
                                               "progress_calibration": calibration, "shadow_fidelity": fidelity})
```

Before `while env_steps < budget:` (line 552):

```python
    labeller = ThreadPoolExecutor(max_workers=1) if calibration is not None else None
```

(Task 7 widens this condition to `calibration is not None or tokens is not None`.)

Episode head (lines 557–563) becomes:

```python
        sim_task = task["env_task"] if variant is None else variant
        env = make_env(sim_task, suite, init_states=stored, plus=bool(config.plus_train), proprio=config.proprio)
        if stored:
            env.init_state_id = int(draw * len(env._init_states))
        policy.reset()
        observation, _ = env.reset(seed=config.seed + env_steps)
        shadow = make_shadow(sim_task, suite, stored, bool(config.plus_train), config.seed + env_steps) if calibration is not None else None
        future = None
```

After the decode branches (line 579, `decoded = policy.decode_candidates(batch, k=config.k_candidates)`), still inside the chunk loop:

```python
                if shadow is not None:
                    future = labeller.submit(_timed, label_plans, shadow, capture_live(env._env.env), sim_state,
                                             decoded["plan_a_base_all"].detach().float().cpu(), decoded["first_chunk"], norm, layout, calibration)
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
                        if future is not None:
                            q_mean = torch.stack(model.critic(critic_all, action_all, return_all=True)).mean(dim=0).reshape(plans, k).max(dim=1).values
```

(`critic_all` / `action_all` are the two expressions the old `q_all` line nested, so `q_all` is bit-identical; `v_all` keeps the min readout for the existing agreement diagnostic.) After the env step `for` loop and before `chunk_seconds += time.perf_counter() - started` (line 629):

```python
                plan_labels = None
                if future is not None:
                    wait_started = time.perf_counter()
                    plan_labels, label_seconds = future.result()
                    label_wait_seconds, future = time.perf_counter() - wait_started, None
                    plan_mean = plan_labels.mean(axis=1)
                    plan_info.update({
                        "label_agreement_with_q": float(int(plan_mean.argmax()) == int(q_mean.argmax())),
                        "label_spread_within_plan": float(plan_labels.std(axis=1).mean()),
                        "label_spread_across_plans": float(plan_mean.std()),
                        "label_executed_minus_mean": float(plan_mean[decoded["plan_index"]] - plan_mean.mean()),
                        "label_seconds": label_seconds, "label_wait_seconds": label_wait_seconds})
```

In the row (lines 634–655) add, after the `"variant_index": env_task,` line:

```python
                    **({"plan_labels": plan_labels} if plan_labels is not None else {}),
```

The `finally` (line 677):

```python
        finally:
            if future is not None:
                wait([future])
            env.close()
            if shadow is not None:
                shadow.close()
```

After the `while` loop, before `save_inference(output_dir / "residual.pt", model)`:

```python
    if labeller is not None:
        labeller.shutdown()
```

On the default path `calibration` is None, so no executor, no shadow, `future` stays None, `plan_labels` is None and the row is unchanged; `sim_task` is the same value the old inline expression produced.

- [ ] **Step 5: `brev/train.sh`**

Both usage strings append ` [--plan-target q|rollout]` after `[--plus]`. After `PLUS=0` add `PLAN_TARGET=q`; in the `case` after `--plus) PLUS=1; shift ;;` add `--plan-target) PLAN_TARGET="$2"; shift 2 ;;`. The config JSON python gains `, "plan_target": sys.argv[13]` before the closing `}` and the argument `"$PLAN_TARGET"` after `"$PLUS_TRAIN"`.

- [ ] **Step 6: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && .cache/eval-venv/bin/python -m compileall -q script/ && bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`

```bash
git add script/lingbot_rl_train.py brev/train.sh tests/test_lingbot_rl.py
git commit -m "Label every candidate plan in a physics-only shadow on a worker thread under plan_target rollout; store and log the labels; gate the run on base and variant shadow fidelity; add --plan-target to train.sh."
```

---

### Task 7: Token sidecar for the offline encoder study

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`, `store_tokens` False) reproduces bit-for-bit, with no thread pool creation, no shadow construction, no sidecar, no extra `proj_out` capture and no extra transformer work.

**Files:**
- Create: `script/lingbot_rl_tokens.py`
- Modify: `script/lingbot_rl_config.py` (field after `plan_target`, `validate()`), `script/lingbot_rl_policy.py` (`_video_pass_batched` line 416, `select_plan` lines 604–657), `script/lingbot_rl_train.py` (imports; `train()`: store construction after the resume load, the executor condition, the `select_plan` call, the token job after the label submit, the wait before `add_online`, the episode `finally`), `brev/train.sh`, `tests/test_lingbot_rl.py` (`_StubPolicy.select_plan`, three tests)

**Interfaces:**
- Consumes: Task 6's executor (`labeller`), `future` handling and episode `finally`; `select_plan`'s batched last pass (`_video_pass_batched`) and the `proj_out` capture pattern of `_video_pass`.
- Produces: `RLConfig.store_tokens: bool = False` (storage only, never in `protocol()`; requires `plan_candidates > 1`); `ResidualLingBotPolicy._video_pass_batched(..., capture=None)`; `select_plan(..., return_tokens=False)` adding `out["plan_tokens"]` = the conditional rows of the last denoising pass's pre-`proj_out` video tokens, `(M, 128, 3072)` bf16 on CPU, only when `return_tokens` is True; `script/lingbot_rl_tokens.py`: `SHARD_CHUNKS = 256`, `TokenStore(root, ordinal)` (reopens `index.jsonl`, keeps its first `ordinal` lines, raises `RuntimeError("token index behind the replay")` when it has fewer), `TokenStore.append(tokens, chunk, episode, step, reset_kind, variant_index, plan_index) -> entry dict` (writes slot `offset` of `shard_<n>.npy`, appends the entry line, advances `ordinal`; raises `ValueError` when `chunk != ordinal`), `TokenStore.read(entry) -> bf16 tensor`; `train.sh --store-tokens`; `<run>/tokens/` exists only when `store_tokens`.

- [ ] **Step 1: Fixture** in `tests/test_lingbot_rl.py`

`_StubPolicy.select_plan` becomes `def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False, return_tokens=False):`, records `(plans, k, random_plan, decode_all, return_tokens)` in `plan_calls`, and its returned dict gains `**({"plan_tokens": torch.full((plans, 8, STATE_DIM), 0.5, dtype=torch.bfloat16)} if return_tokens else {})`. (The existing assertions index `call[2]` / `call[3]` and stay valid.)

- [ ] **Step 2: Write the failing tests**

After `test_select_plan_batched_matches_looped_decode_plans_and_keeps_winner_rows`:

```python
def test_select_plan_returns_last_pass_tokens_only_when_asked_without_extra_transformer_work():
    pool = lambda latent: latent.float().mean(dim=(1, 2, 3, 4)).reshape(-1, 1).expand(-1, STATE_DIM)
    policy = _mock_policy()
    policy._begin_chunk, policy._features_from_latent = (lambda batch: (False, None, 4, torch.zeros(1, 48, 1, 2, 4))), _mock_features(pool)
    torch.manual_seed(3)
    plain = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0)
    calls = len(policy.transformer.calls)
    torch.manual_seed(3)
    out = policy.select_plan({"task": ["t"]}, plans=2, k=3, rank_fn=lambda s, pools: 0, return_tokens=True)
    assert "plan_tokens" not in plain and set(out) == set(plain) | {"plan_tokens"} and len(policy.transformer.calls) == 2 * calls
    tokens = out["plan_tokens"]
    assert tokens.shape == (2, 8, STATE_DIM) and tokens.dtype == torch.bfloat16 and tokens.device.type == "cpu"
    assert torch.equal(out["plan_pools"], plain["plan_pools"]) and not torch.equal(tokens[0], tokens[1])
```

After `test_plan_labels_default_to_zeros_and_sample_with_the_plan_shape`:

```python
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
```

After the Task 6 trainer test:

```python
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
```

The stub env ends every episode after 12 steps, so five chunks are five episodes; the stub tokens are a constant 0.5 so the round trip is exact; the recipe equality pins that `store_tokens` never enters the fingerprint.

- [ ] **Step 3: Run them**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py -k "last_pass_tokens or token_store or store_tokens"`
Expected: FAIL with `TypeError: select_plan() got an unexpected keyword argument 'return_tokens'` and `ModuleNotFoundError: No module named 'script.lingbot_rl_tokens'`.

- [ ] **Step 4: Config** in `script/lingbot_rl_config.py`

After `plan_target: str = "q"`:

```python
    store_tokens: bool = False
```

In `validate()` after the `plan_target` checks:

```python
        if type(self.store_tokens) is not bool or (self.store_tokens and self.plan_candidates < 2):
            raise ValueError("store_tokens must be a bool and needs plan_candidates > 1")
```

`protocol()` is not touched: the sidecar changes storage only, so a run may be resumed with or without it.

- [ ] **Step 5: Policy** in `script/lingbot_rl_policy.py`

`_video_pass_batched` gains `capture=None` and wraps its transformer call exactly as `_video_pass` does:

```python
    def _video_pass_batched(self, latents, t, latent_cond, frame_st_id, write, advance, capture=None):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        m = latents.shape[0]
        cond = None if latent_cond is None else latent_cond.expand(m, -1, -1, -1, -1)
        payload = self._prepare_latent_input(latents, None, t, t, cond, None, frame_st_id=frame_st_id)
        payload = payload["latent_res_lst"]
        payload["noisy_latents"] = payload["noisy_latents"].repeat(2, 1, 1, 1, 1)
        payload["text_emb"] = torch.cat([
            self._prompt_embeds.to(self.dtype).expand(m, -1, -1),
            self._negative_prompt_embeds.to(self.dtype).expand(m, -1, -1)]).clone()
        payload["grid_id"] = payload["grid_id"][None].repeat(2 * m, 1, 1)
        payload["timesteps"] = payload["timesteps"][None].repeat(2 * m, 1)
        handle = None
        if capture is not None:
            handle = self.transformer.proj_out.register_forward_hook(lambda module, inputs, output: capture.append(inputs[0]))
        try:
            pred = self.transformer(payload, update_cache=1 if write else 0, cache_name="pos", action_mode=False)
        finally:
            if handle is not None:
                handle.remove()
```

(the rest of the method unchanged). `select_plan` signature becomes `def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False, rank_candidates=None, return_tokens=False):`; the denoising loop becomes

```python
        captured = []
        for i, t in enumerate(timesteps):
            last = i == len(timesteps) - 1
            latents = self._video_pass_batched(
                latents, t, latent_cond, frame_st_id, write=last, advance=not last or exec_step != -1,
                capture=captured if last and return_tokens else None)
```

and right after the `out = {...}` dict is built:

```python
        if return_tokens:
            out["plan_tokens"] = captured[0][:plans].to(device="cpu", dtype=torch.bfloat16)
```

The last batched pass runs the transformer at CFG batch 2M with the M conditional rows first (`text_emb` is `cat([prompt × M, negative × M])`), so `captured[0][:plans]` are the history-conditioned (KV cache `"pos"`) pre-`proj_out` video tokens of the M plans: 4 frames × 4 rows × 8 columns = 128 tokens of width 3072, the same grid `_features_from_latent` reads for the pool but from the history-free critic cache. Without `return_tokens` no hook is registered and the pass is byte-identical.

- [ ] **Step 6: Sidecar** `script/lingbot_rl_tokens.py`

```python
import json
from pathlib import Path

import numpy as np
import torch

SHARD_CHUNKS = 256
FIELDS = ("chunk", "episode", "step", "reset_kind", "variant_index", "plan_index")


class TokenStore:
    def __init__(self, root, ordinal):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        lines = self.index.read_text().splitlines() if self.index.is_file() else []
        if len(lines) < ordinal:
            raise RuntimeError("token index behind the replay")
        self.index.write_text("".join(line + "\n" for line in lines[:ordinal]))
        self.ordinal = int(ordinal)

    def shard_path(self, shard):
        return self.root / f"shard_{shard:04d}.npy"

    def append(self, tokens, **entry):
        if entry["chunk"] != self.ordinal:
            raise ValueError(f"token chunk {entry['chunk']} does not match ordinal {self.ordinal}")
        shard, offset = divmod(self.ordinal, SHARD_CHUNKS)
        array = tokens.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.int16).numpy()
        path = self.shard_path(shard)
        if offset == 0 and not path.is_file():
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(SHARD_CHUNKS, *array.shape))
        else:
            store = np.load(path, mmap_mode="r+")
        store[offset] = array
        store.flush()
        del store
        entry = {**{key: int(entry[key]) for key in FIELDS}, "shard": shard, "offset": offset}
        with self.index.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.ordinal += 1
        return entry

    def read(self, entry):
        array = np.array(np.load(self.shard_path(entry["shard"]), mmap_mode="r")[entry["offset"]])
        return torch.from_numpy(array).view(torch.bfloat16)
```

Each shard is a valid `.npy` of shape `(256, M, 128, 3072)` int16 (bf16 bit patterns, about 800 MB, created sparse by `open_memmap` and filled slot by slot); `index.jsonl` is the only source of truth for which slots are valid, so a crash between the slot write and the index line leaves an orphan slot that the next append overwrites. `TokenStore(root, chunks)` at resume truncates the index to the replay's chunk count (the resume file is written every ten episodes, the index every chunk), so `len(index) == chunks` always holds and later chunks overwrite their slots; an index shorter than the replay means tokens are missing and raises. `store[offset] = array` raises on a shape mismatch, and `flush()` raises on a write failure; both propagate through the future.

- [ ] **Step 7: Trainer** in `script/lingbot_rl_train.py`

Imports: `from script.lingbot_rl_tokens import TokenStore`. In `train()`, after the resume load (the `if resume:` block) and before the expert-rows check:

```python
    tokens = TokenStore(output_dir / "tokens", chunks) if config.store_tokens else None
```

The executor line from Task 6 becomes:

```python
    labeller = ThreadPoolExecutor(max_workers=1) if calibration is not None or tokens is not None else None
```

The episode head's `future = None` becomes `future = token_future = None`. The `select_plan` call gains `return_tokens=tokens is not None`:

```python
                    decoded = policy.select_plan(batch, plans, config.k_candidates, rank_fn=model.rank_plans,
                                                 random_plan=random_plan, decode_all=True, return_tokens=tokens is not None)
```

Directly after the Task 6 label `submit` block (same indentation):

```python
                if tokens is not None:
                    token_future = labeller.submit(tokens.append, decoded["plan_tokens"], chunk=chunks, episode=episodes, step=env_steps,
                                                   reset_kind=kind, variant_index=env_task, plan_index=int(decoded["plan_index"]))
```

(`chunks`, `episodes` and `env_steps` are read before the live chunk executes, so `step` is the chunk's first env step and `chunk` equals the row's position in the replay.) After the Task 6 `future.result()` block, before `chunk_seconds += ...`:

```python
                if token_future is not None:
                    token_future.result()
                    token_future = None
```

The episode `finally` waits for both: `wait([f for f in (future, token_future) if f is not None])` replaces the `if future is not None: wait([future])` lines (keep the `env.close()` / shadow close after it). The replay row is unchanged (`plan_tokens` never enters the buffer). On the default path `tokens` is None, no executor is created unless rollout is on, `select_plan` receives `return_tokens=False` and `token_future` stays None.

- [ ] **Step 8: `brev/train.sh`**

Both usage strings append ` [--store-tokens]` after `[--plan-target q|rollout]`. After `PLAN_TARGET=q` add `STORE_TOKENS=0`; in the `case` after the `--plan-target` line add `--store-tokens) STORE_TOKENS=1; shift ;;`. The config JSON python gains `, "store_tokens": sys.argv[14] == "1"` before the closing `}` and the argument `"$STORE_TOKENS"` after `"$PLAN_TARGET"`.

- [ ] **Step 9: Verify and commit**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py && .cache/eval-venv/bin/python -m script.lingbot_rl_config && .cache/eval-venv/bin/python -m compileall -q script/ && bash -n brev/setup.sh brev/train.sh brev/sync.sh brev/pull.sh brev/eval.sh`

```bash
git add script/lingbot_rl_tokens.py script/lingbot_rl_config.py script/lingbot_rl_policy.py script/lingbot_rl_train.py brev/train.sh tests/test_lingbot_rl.py
git commit -m "Store every plan's history-conditioned last-pass tokens in an on-disk sidecar under --store-tokens, written by the worker thread."
```

---

### Task 8: GB10 checks, smoke, stop run A, launch (controller)

**Standing constraints:** zero comments and docstrings in `script/lingbot_rl_*.py` and tests; lean code; only necessary tests; the default recipe (`plan_target` `"q"`, `store_tokens` False) reproduces bit-for-bit, with no thread pool creation, no shadow construction and no sidecar. Nothing in this task changes code; a failure here is reported, not patched on the box.

Run on Jarvis from `~/Documents/dice-rl-wam` after `git pull`; `DICE_DATA=~/Documents/dice`; the plus install is activated by `brev/train.sh --plus` (`~/Documents/dice/libero-plus-config/activate.sh`). Nothing here is run by implementers. Run A keeps running until Step 4.

- [ ] **Step 1: Smoke with the fidelity gates** (`DICE_NO_SYNC=1` in env.sh as for the previous runs):

```bash
bash brev/train.sh smoke-pool-cf 9 --residual-input base --epsilon -0.3 --video-steps 5 --action-steps 10 --plan-candidates 4 --env-steps 320000 --plus --proprio --state-features pool --plan-target rollout --store-tokens --max-env-steps 2100 --no-tmux
```

Copy `$DICE_DATA/runs/dice-t9-plus-m4-pool/expert_features.pt` into `$DICE_DATA/runs/smoke-pool-cf/` first to skip the v3 rebuild (same recipe fingerprint). The run starts with the probe: layout, calibration, the 20-chunk fidelity rollout of the prior on the base task (about 3 minutes), then 3 chunks on layout variant 1982. If either raises `shadow fidelity check failed: {...}`, stop and report the printed maxima; do not loosen `FIDELITY_POSITION_M` / `FIDELITY_ANGLE_RAD`. A `shadow state mismatch` from `label_plans` during collection means the variant's qpos layout or object identities differ between the two environments; report the episode's `variant_index`. Read:

- `$DICE_DATA/runs/smoke-pool-cf/settings.json`: `recipe.plan_target` `rollout`, `recipe.plan_critic` `regression_mean_k_rollout_progress`; `progress_calibration` with `door_closed` 0.0, `door_open` −2.094, `door_initial` nearer `door_open` (task 9 starts with the microwave open; if `door_initial` is instead nearer 0.0, stop and report), `phi_initial` about −‖mug − micro‖ at reset; `shadow_fidelity.base.position_m` and `.plus.position_m` well under 1e-3 (a correct sync lands near 1e-12), `angle_rad` well under 0.01745, `base.chunks` 20 (fewer only if the prior succeeded within 20 chunks), `plus.chunks` 3, `plus.variant_index` 1982.
- Confirm from the log that the shadow is unconstrained by LIBERO's horizon: the smoke's episodes run their full 520 steps without a `ValueError("executing action in terminated episode")`.
- W&B run `smoke-pool-cf` (or `$DICE_REPO/wandb/latest-run/files/wandb-summary.json`): `label_seconds` (physics only; expect well under the roughly 1 s the spec budgeted with rendering), `label_wait_seconds` (expect near zero: the shadow's 16 × ≤16 steps overlap a live chunk of about 16 × 0.54 s), `seconds_per_env_step` against the run A smoke, `label_agreement_with_q`, `label_spread_within_plan`, `label_spread_across_plans` (logged as two values; the spec's within-over-across ratio is taken offline), `label_executed_minus_mean`, `plan_target_q_mean`, `plan_label_vs_q_agreement`, `peak_gpu_memory_gb`. If `label_wait_seconds` is a sizeable fraction of `label_seconds`, the overlap is not happening (the GIL is not released where expected); report it rather than tune.
- Token sidecar: `wc -l $DICE_DATA/runs/smoke-pool-cf/tokens/index.jsonl` equals `chunks` in `$DICE_DATA/runs/smoke-pool-cf/summary.json` (about 130 for 2,100 steps); `ls -l` and `du -h` on `tokens/shard_0000.npy` (the header says `(256, 4, 128, 3072)` int16, about 805 MB apparent, with `du` growing by about 3.1 MB per chunk since the file is sparse); `df -h $DICE_DATA` leaves room for about 63 GB per 320k run. Then:

```bash
$PY - <<'EOF'
import json, numpy as np, torch
from script.lingbot_rl_tokens import TokenStore
root = "/home/$USER/Documents/dice/runs/smoke-pool-cf/tokens"
lines = [json.loads(line) for line in open(root + "/index.jsonl")]
store = TokenStore(root, len(lines))
tokens = store.read(lines[-1]).float()
print(len(lines), lines[-1], tuple(tokens.shape), float(tokens.abs().mean()), bool(torch.isfinite(tokens).all()))
print(np.load(root + "/shard_0000.npy", mmap_mode="r").shape)
EOF
```

Expected: the last entry's `chunk` is `len(lines) - 1`, `step` its chunk's first env step, tokens `(4, 128, 3072)` finite with a non-zero mean magnitude (the stub-free real pass; all-zero tokens would mean the hook captured the wrong rows). Also read `label_wait_seconds` again with the sidecar on: the 3 MB write shares the worker with the labels and should not raise it.
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

- [ ] **Step 2: Record** in `docs/heldout-results.md` next to the run A entry: the two fidelity results, the calibration, `label_seconds`, `label_wait_seconds`, `seconds_per_env_step`, memory, the label statistics, and the interior-ball geometry caveat: relative to the microwave body origin the cavity spans x ∈ [−0.1435, 0.0635], y ∈ [−0.088, 0.0805] with the floor at z = 0.024, so the 0.12 m ball from the mug origin misses about 2.5 cm of the far floor and includes about 3 cm beyond the door opening; the +2 success term carries the goal.

- [ ] **Step 3: Stop run A** (keep its run directory and W&B run; it is the reference for the Q-target recipe):

```bash
tmux kill-session -t dice-dice-t9-plus-m4-pool
```

- [ ] **Step 4: Launch** the replacement run on Jarvis with run A's flags plus `--plan-target rollout`:

```bash
bash brev/train.sh dice-t9-plus-m4-pool-cf 9 --residual-input base --epsilon -0.3 --env-steps 320000 --video-steps 5 --action-steps 10 --plan-candidates 4 --plus --proprio --state-features pool --plan-target rollout --store-tokens
```

Copy `expert_features.pt` into `$DICE_DATA/runs/dice-t9-plus-m4-pool-cf/` first. Checkpoints every 80k; the offline measurements of spec section 5 run on the stored `plan_labels` at 80k and 320k.

## Self-review notes

- Spec coverage: section 2 run / recipe → Task 8 launch command and Task 1 validation; label per chunk (shadow, 16 candidates, base actions, physics only, φ difference, mean over K, (M, K) float32 stored) → Tasks 5 (`label_plans` on the direct robosuite path with image observables disabled), 3 (buffer), 6 (storage, labelling thread) and 4 (mean over K as the target); φ for task 9 with startup calibration → Task 2 (door limits from the model, interior = microwave body position per the brief); critic unchanged architecture, label target, mean readout, Q target still logged → Task 4; what stays fixed → no task touches the actor, Q updates, sampler, reset draw or evaluation protocols; shadow per episode with the live kwargs, reset once, state-driven from a snapshot, live never restored → Tasks 5 and 6; logged diagnostics → Task 6 (`label_agreement_with_q`, the within and across spreads logged as two values rather than the spec's ratio, executed minus mean, `label_seconds`, `label_wait_seconds`; per-episode logs unchanged); cost measured in the smoke → Task 8; section 4 fidelity check (20 chunks, 1 mm, 1° on the body orientations and the door) → Tasks 5 and 6, plus the 3-chunk check on a layout variant, with the controller goal copied along with the state as section 4 anticipates; section 6 code list → Tasks 1–6 and `train.sh --plan-target`; the token sidecar for the later encoder study (not in the spec; coordinator-added) → Task 7 (`store_tokens`, `select_plan(return_tokens=True)`, `TokenStore`, `train.sh --store-tokens`), its three tests and the Task 8 sidecar readouts; section 6 tests (progress on hand-built poses, `label_plans` on a stub environment, the plan-critic update on stored labels, fingerprint, default recipe unchanged) → Tasks 2, 5, 4, 1 and the `plain` smoke assertions in Task 6. Not in this plan: the offline measurements and held-out evaluations of section 5, which the spec schedules after 80k / 320k; `script/lingbot_eval_modal.py` building its eval config without `plan_candidates` (pre-existing for M = 4 checkpoints).
- Placeholder scan: every code step carries the code; nothing is deferred to the box except reading measurements.
- Sidecar invariants: `plan_tokens` leaves `select_plan` only under `return_tokens`, with no hook and no extra transformer call otherwise (pinned by the equal call count); the row never carries `plan_tokens`; `store_tokens` is absent from `protocol()` (pinned by the recipe equality in the Task 7 trainer test); `TokenStore(root, chunks)` keeps the index aligned with the replay across resume; the token job runs on the same single worker as the labels, after the label job, and its `result()` precedes the next chunk so a write failure stops the run.
- Type consistency: `label_plans(shadow, snapshot, before, candidates, first_chunk, norm, layout, calibration)` has the same signature in Task 5, the Task 6 `submit` (through `_timed`) and the Task 6 monkeypatch; `shadow_fidelity(env, shadow, layout, act, observation, chunks)` in Task 5, both Task 6 calls and the monkeypatch (which accepts `chunks`); `capture_live(live)` in Tasks 5, 6; `calibrate(task_id, inner, layout, initial)` in Tasks 2, 6; `progress(calibration, layout, before, after, success)` in Tasks 2, 5; `update_plan_critic(..., plan_labels=None)` in Tasks 4, 6; `make_shadow(task_id, suite, init_states, plus, seed)` in Tasks 5, 6; `plan_labels` is numpy (M, K) float32 from `label_plans`, in the row, and torch (B, M, K) from `sample`; the fidelity result keys are `position_m`, `angle_rad`, `chunks` in Tasks 5, 6, 7.
- Bit-for-bit with `plan_target "q"`: the protocol and checkpoint `layout` are unchanged; `rank_plans` executes the same `plan_critic(state.expand, pools)` min readout; `update_plan_critic` computes the same target and returns the same keys; the loop's `q_all` is the same expression split over two names; `plan_labels` enters the row only under rollout and the buffer default is zeros that the `"q"` update ignores; the executor, shadow, snapshot, calibration and fidelity code run only when `calibration is not None`, which requires `plan_target == "rollout"` (pinned by the `pools == [{"max_workers": 1}]` assertion after the `plain` run); the probe block only binds the reset's return value.
- Threading: the worker reads only the snapshot, the shadow and cpu tensors; the main thread never touches the shadow while a future is pending; `future.result()` precedes `buffer.add_online`, so rows and training order are unchanged; a pending future is waited for before the shadow is closed; the test's stub labeller is synchronous and joined before any assertion.
- RNG: `make_shadow` resets inside `capture_rng()` / `restore_rng()`; both startup fidelity rollouts run inside the probe's RNG capture; `label_plans` draws nothing.
- Fingerprint: `plan_target` enters `protocol()` whenever it is `"rollout"`, so `test_load_resume_rejects_recipe_mismatch` refuses a resume across the change; `test_config_plan_target_enters_the_fingerprint_only_for_rollout` pins the default protocol at 78 keys.
- Verified from source: robosuite `MjSim.get_state` / `set_state` / `forward`, `MjSimState.from_flattened` (`na == 0`), `MjData` attribute delegation, `Controller.update(force=True)`, `initial_joint`, OSC `set_goal` goal_ori behaviour and `reset_goal`, `PandaGripper.format_action` ramp, `MujocoEnv.step` done check, `Observable.update` rendering and `modality` / `set_enabled`, `use_camera_obs` consulted only at observable creation (`~/.cache/uv/git-v0/checkouts/79abe6cb204f004b/c848ca8`); LIBERO `ControlEnv(horizon=1000, ignore_done=False)`, `OffScreenRenderEnv.step` forwarding, `check_success`, `regenerate_obs_from_state` pattern, task 9 bddl `(Open microwave_1)` init and `(In white_yellow_mug_1 microwave_1_heating_region) (Close microwave_1)` goal, `microwave.xml` joint range `-2.094 0` and heating-region site (LIBERO copy at `/Users/desmondzee/Hardware/ICL-so101/third_party/LIBERO`, not the pinned GB10 install); LeRobot `LiberoEnv.step` / `reset` / `_format_raw_obs` (`.cache/lerobot`).
- Box-only checks (Task 8): the shadow's determinism against the live env (the fidelity maxima on the base task and the layout variant), the recorded door calibration direction, the LIBERO-Plus install's `horizon` / `ignore_done` values, the physics-only label cost and the overlap (`label_wait_seconds`), the sidecar's index length against `chunks` and the sparse shard growth, and that the real last-pass tokens are non-zero.
