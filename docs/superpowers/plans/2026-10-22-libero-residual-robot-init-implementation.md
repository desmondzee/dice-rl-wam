# Residual RL on Shifted Robot Initial States Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the DICE-RL noise-space residual over the frozen 20-demo LIBERO prior on procedurally shifted robot initial poses and measure it on the LIBERO-Plus robot-init variants.

**Architecture:** The `libero_image` wrapper gains a soft reset, a controller-reference update and a procedural joint shift with a validity check; a subclass of the vendored image fine-tune agent ramps the shift magnitude through the vector env; a residual evaluator mirrors the agent's feature path; the LIBERO-Plus variant poses are exported once and composed onto canonical stored states for the held-out set. Three gates (soft reset, shift distribution, smoke) run on Jarvis before the launch.

**Tech Stack:** the LIBERO venv on Jarvis (`brev/dicerl/env_libero.sh`: Python 3.11, torch 2.11.0+cu128, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1); the LeRobot venv on Jarvis only for the LIBERO-Plus export; local tests with `.cache/eval-venv/bin/python` (numpy, torch; no gym, no h5py).

**Spec:** `docs/superpowers/specs/2026-10-22-libero-residual-robot-init-design.md`

## Global Constraints

- Vendored DICE-RL algorithm code is not edited: `agent/`, `model/`, `util/`, `script/run.py`, `script/eval_rl_checkpoint.py`, `cfg/robomimic/`. Edits in `env/gym_utils/` are limited to the libero branch of `make_async` and the libero wrapper files. `.cache/lerobot`, `script/lingbot_*`, the venvs' pins: untouched.
- No comments or docstrings in new code; lean files; commit after each task with no attribution trailer; never `pkill -f` (kill tmux sessions by name, processes by pid or process group); one GPU training job on the box at a time; nothing printed that could carry a secret.
- Task: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`. Prior: `state_100.pt` of run `2026-10-22_10-11-34_42` under `$DICE_RL_LOG_DIR/libero-pretrain/`. Data: `$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain_20/`.
- State layout: 47 entries, joints at indices 1–7 (`slice(1, 8)`), gripper 8–9, objects 10–24, velocities after.
- Shift: offset direction uniform on the 7-sphere, norm uniform in [0, magnitude]; ramp 0 until iteration 3,000, linear to 0.6 rad at iteration 6,000, then 0.6; validity = joints inside model range with 0.02 rad margin and no robot/gripper geom contact with a non-gripper geom; 20 resamples then unshifted fallback.
- Recipe: released Square image fine-tune config with the changes listed in the spec's "Training recipe" (`condition_residual_on_base_action: true`, `use_soft_q_filtering: true`, `always_retain_bc_loss_for_expert_data: false`, `q_underestimation_threshold: -0.55`, `n_step: 3`, warm-ups 3000/3000, `n_envs: 8`, `success_steps_before_termination: 1`, `hard_reset: false`, `noise_dim: 56`, `eval_n_envs: 10`, `num_eval_episodes: 30`).
- Evaluation protocols: 8-step chunks, 520 steps, 25 envs; held-out = variants 640–681 × canonical stored states 0–6 (294 episodes) at torch seed 0; monitoring = states 0–1 (84 episodes) at seed 0 every 2,000 iterations; in-distribution = canonical 50 × seeds 0, 1, 2; procedural curve = magnitudes 0.1, 0.3, 0.5, 0.6 on canonical states, shift seed 1000, 100 episodes each; residual checkpoints use max-Q-min with 16 samples, the prior uses single-sample flow.
- Box: `ssh -F ~/.brev/ssh_config jarvis`, `source brev/dicerl/env_libero.sh` (sets `PY`, `PYTHONPATH`, `DICE_RL_DATA_DIR`, `DICE_RL_LOG_DIR`, `DICE_RL_WANDB_ENTITY`, `MUJOCO_GL=egl`); rsync changed directories before each box run.

## Review Focus

- A shifted start that places a joint outside its range or the arm in contact with the table or microwave: must be rejected and resampled, never trained on (Task 2, `test_valid_start_rejects_contact_and_range`).
- An evaluation reset with an explicit `init_state` while a training shift is active: must load the state unshifted (Task 2, `test_explicit_init_state_ignores_training_shift`).
- A held-out variant pose loaded without the controller's reference update: the arm would drift toward the canonical pose; the wrapper must update the reference whenever joints differ (Task 1, `test_controller_reference_follows_loaded_joints`).
- The ramp outside its window: flat at 0 before iteration 3,000 and flat at 0.6 after 6,000, never negative or above the end (Task 4, `test_ramp_magnitude_segments`).
- A `--states` file whose labels and states disagree in length, or a seeds list of one: the evaluator must refuse the former and accept the latter (Task 3, `test_states_file_validation`, `test_single_seed_plan`).

---

### Task 1: Soft reset, `hard_reset` plumbing and controller reference (gate A)

**Files:**
- Modify: `env/gym_utils/wrapper/libero_image.py` (`make_libero_env`, `LiberoImageWrapper.__init__`, `reset`)
- Modify: `env/gym_utils/__init__.py` (libero branch of `_make_env`)
- Create: `script/dicerl_libero_gate_a.py`
- Create: `docs/libero-residual.md`
- Test: `tests/test_dicerl_libero_rl.py` (new file; reuse `FakeLibero` and `make_wrapper` from `tests/test_dicerl_libero.py` by import)

**Interfaces:**
- Produces: `make_libero_env(task_name, camera_names=..., image_hw=128, hard_reset=True)`; `LiberoImageWrapper.__init__(..., hard_reset=True)` stores the flag only (the factory pops `hard_reset` from the wrapper block and passes it to `make_libero_env`); `LiberoImageWrapper.load_state(state) -> raw_obs` = `env.set_init_state(state)` followed by `env.env.robots[0].controller.update_initial_joints(state[1:8])` when `state[1:8]` differs from `self.canonical_joints` (the joints of `init_states[0]`, or `None` when no init states) by more than 1e-9; `reset` uses `load_state`.
- `FakeLibero` gains `env.robots[0].controller.update_initial_joints` recording calls (a nested fake in the new test file).

- [ ] **Step 1: Write the failing tests**

```python
def test_controller_reference_follows_loaded_joints(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    canonical = w.init_states[0]
    w.reset(options={"init_state": canonical})
    assert env.controller_calls == []
    shifted = canonical.copy(); shifted[1:8] += 0.1
    w.reset(options={"init_state": shifted})
    assert len(env.controller_calls) == 1 and np.allclose(env.controller_calls[0], shifted[1:8])

def test_hard_reset_flag_is_stored_and_passed():
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    w = LiberoImageWrapper(FakeRLLibero(), None, hard_reset=False)
    assert w.hard_reset is False
```

`make_rl_wrapper` builds the wrapper with `init_states` = 50 copies of a base state whose joints are the canonical `[-0.031, -0.133, 0.015, -2.439, 0.0, 2.207, 0.778]` at indices 1–7, and `FakeRLLibero` extends `FakeLibero` with `env.robots = [obj]` where `obj.controller.update_initial_joints(j)` appends `np.asarray(j)` to `controller_calls`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py` (mark these two `needs_gym` like the wrapper tests; they run on the box)
Expected: fail on the missing `hard_reset` argument and `controller_calls`.

- [ ] **Step 3: Implement**

`make_libero_env(..., hard_reset=True)` passes `hard_reset=hard_reset` to `OffScreenRenderEnv`. The factory's libero branch: `args = dict(args); hard = args.pop("hard_reset", True)` for the `libero_image` block and `env = make_libero_env(id, hard_reset=hard)` (resolve the block before building the env). The wrapper: `self.hard_reset`, `self.canonical_joints`, `load_state`.

- [ ] **Step 4: Write `script/dicerl_libero_gate_a.py`**

`replay(hard_reset) -> dict` (demo 0 to success, as gate 0 but on `make_libero_env(task, hard_reset=hard_reset)`); `equivalence(states=(0, 17, 42)) -> dict` comparing the first wrapper observation after `reset(options={"init_state": s})` across four conditions (hard/soft × fresh build/after 20 random-action episodes), reporting max abs proprioception difference and mean abs image difference per state; `drift(n=200) -> dict` (soft resets to state 0, max proprio spread); `latency() -> dict` (mean reset time hard vs soft on one env; env steps per second through `make_async` with 8 envs over 20 full episodes with resets, soft); `controller(magnitude=0.3) -> dict` (after a soft reset and a shifted load with and without `update_initial_joints`, 50 NOOP steps, max joint drift from the loaded joints). CLI `--task --hdf5 --out` writes `gate_a.json`, prints it, and exits 1 if replay fails, any proprio equivalence difference exceeds 1e-3, or drift exceeds 1e-4.

- [ ] **Step 5: Run gate A on the box**

Run: rsync `env/ script/ tests/` then `ssh ... '"$PY" -m script.dicerl_libero_gate_a --task <task> --hdf5 ~/Documents/dice/libero-demos/libero_10/<task>_demo.hdf5 --out $DICE_RL_DATA_DIR/libero/<task>'` and `"$PY" -m pytest -q tests/test_dicerl_libero.py tests/test_dicerl_libero_rl.py`
Expected: exit 0; proprio equivalence ≤ 1e-4; soft reset latency well under the hard one; the with-update controller drift ≤ 2e-2 rad. If the equivalence check fails beyond 1e-3, stop and report (the spec's fallback is hard resets and a rebuilt budget).

- [ ] **Step 6: Start `docs/libero-residual.md`**

Sections: task and prior, gate A table (replay, equivalence per state, drift, latencies, env steps/s, controller drift with and without update), then empty headings for gates B and C, config, launch, sweep, final results.

- [ ] **Step 7: Commit**

```bash
git add env/gym_utils/wrapper/libero_image.py env/gym_utils/__init__.py script/dicerl_libero_gate_a.py tests/test_dicerl_libero_rl.py docs/libero-residual.md
git commit -m "LIBERO wrapper: soft reset option and controller reference update; gate A on Jarvis."
```

---

### Task 2: Procedural robot-init shift (gate B, part 1)

**Files:**
- Create: `env/gym_utils/wrapper/libero_shift.py`
- Modify: `env/gym_utils/wrapper/libero_image.py` (`__init__`, `reset`, new methods)
- Create: `script/dicerl_libero_gate_b.py`
- Test: `tests/test_dicerl_libero_rl.py` (append)

**Interfaces:**
- Produces in `libero_shift.py`: `JOINTS = slice(1, 8)`, `MARGIN = 0.02`, `MAX_TRIES = 20`; `sample_offset(rng, magnitude) -> np.ndarray (7,)`; `shift_state(state, offset) -> np.ndarray` (copy; only `JOINTS` changed); `valid_start(env) -> bool` (after `env.sim.forward()`: every joint in `env.sim.model.jnt_range[j]` ± `MARGIN` for the seven robot joints, and no contact in `env.sim.data.contact[:env.sim.data.ncon]` whose two geoms, by `env.sim.model.geom_id2name`, are one robot-or-gripper geom (name starting `robot0_` or `gripper0_`) and one geom not starting `gripper0_`); `ramp_magnitude(step, start, end, magnitude) -> float`.
- Wrapper: `self.shift = None`; `set_shift_seed(seed)`; `set_shift(magnitude)` (`None` or 0 disables); `shift_stats() -> dict(draws, rejections, fallbacks)` resetting counters; `reset(options)` draws a shifted stored state when `self.shift` is active and no `init_state` is supplied, or applies `options["shift"] = {"magnitude", "seed"}` to a supplied `init_state`; `draw_start(state, magnitude, rng) -> np.ndarray` implements the resample loop against `valid_start` and the fallback.
- `FakeRLLibero` gains a `sim` with `model.jnt_range` (7 rows), `model.geom_id2name`, `data.ncon`, `data.contact`, and a `forward()`; `valid_start` consumes only those.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_offset_norm_and_zero():
    from env.gym_utils.wrapper.libero_shift import sample_offset
    rng = np.random.RandomState(0)
    norms = [np.linalg.norm(sample_offset(rng, 0.6)) for _ in range(500)]
    assert max(norms) <= 0.6 + 1e-9 and min(norms) >= 0 and abs(np.mean(norms) - 0.3) < 0.03
    assert np.array_equal(sample_offset(rng, 0.0), np.zeros(7))

def test_shift_state_changes_only_joints_and_not_input():
    from env.gym_utils.wrapper.libero_shift import shift_state
    s = np.arange(47, dtype=float); s.setflags(write=False)
    out = shift_state(s, np.ones(7))
    assert np.array_equal(out[1:8], s[1:8] + 1) and np.array_equal(np.delete(out, range(1, 8)), np.delete(s, range(1, 8))) and out is not s

def test_valid_start_rejects_contact_and_range():
    from env.gym_utils.wrapper.libero_shift import valid_start
    env = FakeRLLibero()
    assert valid_start(env)
    env.sim.add_contact("robot0_link4_collision", "microwave_1_body")
    assert not valid_start(env)
    env.sim.clear_contacts(); env.sim.add_contact("gripper0_finger1_collision", "gripper0_finger2_collision")
    assert valid_start(env)
    env.sim.clear_contacts(); env.state[1] = env.sim.model.jnt_range[0][1] - 0.01
    assert not valid_start(env)

def test_training_shift_draws_shifted_stored_state_and_counts(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3)
    w.reset(options={"seed": 3})
    assert 0 < np.linalg.norm(env.state[1:8] - w.init_states[3][1:8]) <= 0.3
    assert w.shift_stats() == {"draws": 1, "rejections": 0, "fallbacks": 0}

def test_explicit_init_state_ignores_training_shift(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    w.set_shift_seed(7); w.set_shift(0.3)
    w.reset(options={"init_state": w.init_states[5]})
    assert np.array_equal(env.state, w.init_states[5])

def test_shift_option_applies_to_explicit_state_with_fallback(tmp_path):
    env, w = make_rl_wrapper(tmp_path)
    env.sim.always_contact = True
    w.reset(options={"init_state": w.init_states[5], "shift": {"magnitude": 0.3, "seed": 11}})
    assert np.array_equal(env.state, w.init_states[5]) and w.shift_stats()["fallbacks"] == 1 and w.shift_stats()["rejections"] == 0

def test_ramp_magnitude_segments():
    from env.gym_utils.wrapper.libero_shift import ramp_magnitude
    assert ramp_magnitude(0, 3000, 6000, 0.6) == 0.0 and ramp_magnitude(2999, 3000, 6000, 0.6) == 0.0
    assert abs(ramp_magnitude(4500, 3000, 6000, 0.6) - 0.3) < 1e-12
    assert ramp_magnitude(6000, 3000, 6000, 0.6) == 0.6 and ramp_magnitude(20000, 3000, 6000, 0.6) == 0.6
```

`shift_stats()` resets its counters each call, so the fallback test reads it once into a variable. The first four and the ramp test are numpy-only; the wrapper ones carry `needs_gym`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py -k "offset or shift_state or valid_start or ramp"`
Expected: ImportError on `libero_shift`.

- [ ] **Step 3: Implement `libero_shift.py` and the wrapper changes**

`draw_start`: for `MAX_TRIES` draws, `load_state(shift_state(state, sample_offset(rng, magnitude)))`, `env.sim.forward()`, return on `valid_start`; count rejections; after the loop count a fallback and `load_state(state)`. The training path's rng is `np.random.RandomState(self.shift_seed + self.reset_count)` with `reset_count` incremented per reset; the option path's rng is `RandomState(options["shift"]["seed"])`.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py` locally and the full two files on the box after rsync.
Expected: all pass (gym-marked ones on the box).

- [ ] **Step 5: Write `script/dicerl_libero_gate_b.py` (part 1) and run it**

For magnitudes 0.1, 0.3, 0.5, 0.6: 200 draws from the 50 canonical states on one soft-reset env with `set_shift_seed(1)`; record rejection rate, fallbacks, realised l2 norm quantiles (10/50/90) and post-settle end-effector displacement quantiles (from the wrapper's unnormalised eef position versus the unshifted start); write `gate_b.json`, print it, exit 1 if the 0.6 rejection rate exceeds 0.2. Record the table in `docs/libero-residual.md`.

- [ ] **Step 6: Commit**

```bash
git add env/gym_utils/wrapper/libero_shift.py env/gym_utils/wrapper/libero_image.py script/dicerl_libero_gate_b.py tests/test_dicerl_libero_rl.py docs/libero-residual.md
git commit -m "Procedural robot-init shift in the LIBERO wrapper with validity check and ramp; gate B part 1."
```

---

### Task 3: Evaluators, LIBERO-Plus poses and the prior baselines (gate B, part 2)

**Files:**
- Modify: `script/dicerl_libero_eval.py`
- Create: `script/dicerl_libero_rl_eval.py`
- Create: `script/dicerl_libero_plus_states.py`
- Test: `tests/test_dicerl_libero_rl.py` (append)

**Interfaces:**
- `script/dicerl_libero_eval.py`: `load_states(path) -> (states, labels)` validating equal lengths; `run_protocol(policy, venv, init_states, plan, n_envs, act_steps=8, max_steps=520, shift=None, labels=None)` where `shift = {"magnitude": m, "seed": s}` adds `"shift": {"magnitude": m, "seed": s + plan index}` to each reset option and rows carry `label`; CLI flags `--states PATH`, `--shift M`, `--shift-seed S` (default 1000), `--seeds` accepting one or more ints; `summary.json` gains `states`, `shift`, `shift_seed`.
- `script/dicerl_libero_rl_eval.py`: `load_residual(ckpt_path, device) -> (cfg, model)` reading `<run>/.hydra/config.yaml` (run dir = `dirname(ckpt_path)`), setting `cfg.obs_dim` to `8 + visual_feature_dim` from the prior's config as the agent does, instantiating `cfg.model`, loading `checkpoint["model_state_dict"]`; `augment(model, obs, device) -> torch.Tensor` (the agent's feature path: rgb permuted, `model.pretrained_flow_policy.network.extract_visual_features`, concatenated with state); `residual_policy(model, act_steps, device)` calling `model.get_exploration_action(augmented, num_samples=16, exploration_strategy="max_q_min", training_step=10**6)` and returning `actions[:, :act_steps]`; the same CLI flags as the prior evaluator plus `--iteration` recorded in the summary (parsed from the filename `model_step_<N>.pth`).
- `script/dicerl_libero_plus_states.py`: subcommand `export` (LeRobot env on the box) writing `plus_init_qpos.npz` with `init_qpos` (42, 7) and `variants` (42,) from the variants' robot classes (the builder resolves the class lookup in the LIBERO-Plus package; check: 42 distinct vectors with l2 offset 0.1–0.5 from the canonical joints); subcommand `build` (LIBERO venv) reading it plus the canonical states from `libero_task_files` and writing `plus_robot_init.npz` (294 rows, labels `<variant>:<state>`) and `plus_robot_init_monitor.npz` (84 rows, states 0–1).

- [ ] **Step 1: Write the failing tests**

```python
def test_states_file_validation(tmp_path):
    from script.dicerl_libero_eval import load_states
    np.savez(tmp_path / "s.npz", states=np.zeros((3, 47)), labels=np.array(["a", "b", "c"]))
    s, l = load_states(tmp_path / "s.npz"); assert s.shape == (3, 47) and list(l) == ["a", "b", "c"]
    np.savez(tmp_path / "bad.npz", states=np.zeros((3, 47)), labels=np.array(["a"]))
    with pytest.raises(ValueError): load_states(tmp_path / "bad.npz")

def test_single_seed_plan():
    from script.dicerl_libero_eval import episode_plan
    assert len(episode_plan(4, (0,))) == 4

def test_run_protocol_shift_option_and_labels():
    from script.dicerl_libero_eval import run_protocol, episode_plan
    venv = FakeVenv(2, succeed_at=[10 ** 6, 10 ** 6]); venv.options = []
    def reset_arg(options_list): venv.options.append(options_list); return FakeVenv.reset_arg(venv, options_list)
    venv.reset_arg = reset_arg
    rows = run_protocol(lambda o: np.zeros((2, 8, 7)), venv, np.zeros((2, 47)), episode_plan(2, (0,)), 2, max_steps=16, shift={"magnitude": 0.3, "seed": 1000}, labels=["x", "y"])
    assert [o["shift"]["seed"] for o in venv.options[0]] == [1000, 1001] and venv.options[0][0]["shift"]["magnitude"] == 0.3
    assert [r["label"] for r in rows] == ["x", "y"]

def test_build_plus_states(tmp_path):
    from script.dicerl_libero_plus_states import build
    canonical = np.tile(np.arange(47, dtype=float), (50, 1)); qpos = np.full((42, 7), 9.0); variants = np.arange(640, 682)
    full, monitor = build(canonical, qpos, variants)
    assert full["states"].shape == (294, 47) and monitor["states"].shape == (84, 47)
    assert full["labels"][0] == "640:0" and np.array_equal(full["states"][0][1:8], np.full(7, 9.0)) and np.array_equal(full["states"][0][8:], canonical[0][8:])
    assert monitor["labels"][-1] == "681:1"
```

Reuse `FakeVenv` from `tests/test_dicerl_libero.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py -k "states or seed_plan or shift_option or plus"`
Expected: failures on missing functions and the seeds validation.

- [ ] **Step 3: Implement**

Prior evaluator changes as in Interfaces (the `--seeds` check becomes "at least one"); `dicerl_libero_rl_eval.py` importing `episode_plan`, `run_protocol`, `summarise`, `stack_obs`, `make_env`, `load_states` from the prior evaluator; `dicerl_libero_plus_states.py` with `build(canonical, init_qpos, variants) -> (dict, dict)` and the two subcommands.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py`
Expected: all pass locally except gym-marked ones.

- [ ] **Step 5: Export and build the LIBERO-Plus poses on the box**

Run (LeRobot env): `ssh ... 'source ~/Documents/dice/libero-plus-config/activate.sh; ~/Documents/dice/lerobot/.venv/bin/python -m script.dicerl_libero_plus_states export --out $HOME/Documents/dice/dicerl/data_dir/libero/<task>/plus_init_qpos.npz'`, then (LIBERO venv) `"$PY" -m script.dicerl_libero_plus_states build --qpos .../plus_init_qpos.npz --task <task> --out-dir $DICE_RL_DATA_DIR/libero/<task>`
Expected: 42 distinct poses with l2 offsets in [0.1, 0.5] from the canonical joints (print the min and max); the two npz files written.

- [ ] **Step 6: Prior baselines (gate B part 2)**

Run the prior evaluator four times on `state_100.pt` with `--envs 25` and `--out` under the prior's run directory: held-out (`--states plus_robot_init.npz --seeds 0`), monitoring set (`--states plus_robot_init_monitor.npz --seeds 0`), in-distribution (`--seeds 0 1 2`), procedural curve (four runs, `--shift 0.1`, `0.3`, `0.5`, `0.6`, each with `--shift-seed 1000 --seeds 0 1` on the canonical states, 100 episodes); record success, stderr and the per-variant breakdown (mean over the 7 states per variant) in `docs/libero-residual.md`.

- [ ] **Step 7: Commit**

```bash
git add script/dicerl_libero_eval.py script/dicerl_libero_rl_eval.py script/dicerl_libero_plus_states.py tests/test_dicerl_libero_rl.py docs/libero-residual.md
git commit -m "LIBERO evaluators: explicit state sets, procedural shift option, residual checkpoints; LIBERO-Plus robot-init poses exported; prior baselines."
```

---

### Task 4: RL config, ramp agent, launcher, watcher and smoke (gate C)

**Files:**
- Create: `brev/dicerl/cfg/libero/<task>/ft_distill_residual_flow_unet_img.yaml`
- Create: `script/dicerl_libero_rl.py`
- Create: `script/dicerl_libero_rl_watch.py`
- Create: `brev/dicerl/finetune_libero.sh`
- Modify: `script/dicerl_libero_watch.py` only if a function needs a parameter to be reused (pattern, command); no behaviour change for the pretraining watcher
- Test: `tests/test_dicerl_libero_rl.py` (append)

**Interfaces:**
- `script/dicerl_libero_rl.py`: `class ShiftedResidualAgent(TrainDistillResidualFlowImgAgent)`; `__init__(cfg)` calls the parent then `self.venv.call_sync_arg("set_shift_seed", "seed", [cfg.shift.seed + i for i in range(cfg.env.n_envs)])` and sets `self._shift_magnitude = None`, `self._shift_logged_step = -1`; `update_networks(self, training_step=0)` computes `m = ramp_magnitude(training_step, cfg.shift.ramp_start, cfg.shift.ramp_end, cfg.shift.end_magnitude)`, calls `self.venv.call_sync("set_shift", magnitude=m)` when `m != self._shift_magnitude`, and when `training_step != self._shift_logged_step` and W&B is on, logs `{"shift/magnitude": m, "shift/rejections": R, "shift/fallbacks": F}` (summed over the `call_sync("shift_stats")` results) at `step=training_step * self.horizon_steps * self.n_envs`; then returns the parent's result. `ramp_magnitude` comes from `env/gym_utils/wrapper/libero_shift.py`.
- Config: `_target_: script.dicerl_libero_rl.ShiftedResidualAgent`, `shift: {end_magnitude: 0.6, ramp_start: 3000, ramp_end: 6000, seed: 42}`, everything else per Global Constraints and the spec's "Training recipe"; `train.num_train_steps: 10000000` (never reached); `train.save_freq: 1000`.
- `script/dicerl_libero_rl_watch.py`: `checkpoint_iterations(run_dir, every) -> list[int]` from `model_step_<N>.pth` files with `N % every == 0`; `evaluate(run_dir, iteration, states_path, envs)` running `script.dicerl_libero_rl_eval` with `--states --seeds 0 --envs` into `<run>/heldout_<N>/` and returning the summary; `watch(run_dir, every, states_path, envs, session, poll)` appending `{"iteration", "env_steps", "success_rate", "stderr"}` rows to `heldout_sweep.json` (atomic write, idempotent on restart), exiting 0 when the session is gone, never stopping training; CLI `--run-dir --every 2000 --states --envs 25 --session --poll 300`.
- `brev/dicerl/finetune_libero.sh <task> [seed] [extra hydra overrides...]`: sessions `dicerl-rl-<short>-<seed>` and `dicerl-rlwatch-<seed>`, log `$DICE_RL_LOG_DIR/finetune_<short>_<seed>.log`, run directory detection as `pretrain_libero.sh`, watcher started with the monitoring states file.

- [ ] **Step 1: Write the failing tests**

```python
def test_checkpoint_iterations(tmp_path):
    from script.dicerl_libero_rl_watch import checkpoint_iterations
    for n in (1000, 2000, 2500, 4000): (tmp_path / f"model_step_{n}.pth").write_bytes(b"x")
    assert checkpoint_iterations(tmp_path, 2000) == [2000, 4000]

def test_rl_watch_rows_and_exit(tmp_path, monkeypatch):
    import script.dicerl_libero_rl_watch as w
    (tmp_path / "model_step_2000.pth").write_bytes(b"x")
    monkeypatch.setattr(w, "evaluate", lambda run_dir, it, states, envs: {"success_rate": 0.3, "stderr": 0.05})
    alive = iter([True, False]); monkeypatch.setattr(w, "session_alive", lambda s: next(alive))
    monkeypatch.setattr(w.time, "sleep", lambda s: None)
    assert w.watch(tmp_path, 2000, "states.npz", 25, "sess", 0) == 0
    rows = json.load(open(tmp_path / "heldout_sweep.json"))
    assert rows == [{"iteration": 2000, "env_steps": 128000, "success_rate": 0.3, "stderr": 0.05}]

def test_shift_agent_update_sets_magnitude_once_per_change(monkeypatch):
    from script.dicerl_libero_rl import ShiftedResidualAgent
    calls = []
    agent = ShiftedResidualAgent.__new__(ShiftedResidualAgent)
    agent.cfg = type("C", (), {"shift": type("S", (), {"end_magnitude": 0.6, "ramp_start": 3000, "ramp_end": 6000, "seed": 42})()})()
    agent.venv = type("V", (), {"call_sync": lambda self, name, **kw: calls.append((name, kw)) or [({"draws": 1, "rejections": 0, "fallbacks": 0}, True)]})()
    agent._shift_magnitude = None; agent._shift_logged_step = -1; agent.use_wandb = False; agent.horizon_steps = 8; agent.n_envs = 8
    monkeypatch.setattr(ShiftedResidualAgent.__mro__[1], "update_networks", lambda self, training_step=0: ("parent", training_step))
    assert agent.update_networks(training_step=10) == ("parent", 10)
    agent.update_networks(training_step=10); agent.update_networks(training_step=4500)
    assert [c for c in calls if c[0] == "set_shift"] == [("set_shift", {"magnitude": 0.0}), ("set_shift", {"magnitude": 0.3})]
```

The agent test imports the vendored agent module, which imports torch and gym; mark it `needs_gym` and run it on the box.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py -k "checkpoint_iterations or rl_watch"`
Expected: ImportError.

- [ ] **Step 3: Write the config**

Copy `cfg/robomimic/finetune/square/ft_distill_residual_flow_unet_img.yaml` and apply exactly the spec's "Training recipe" list and the Global Constraints values; repoint `expert_dataset.success_steps_for_termination` to `${env.wrappers.libero_image.success_steps_before_termination}`; set `_target_` and the `shift` block. Keep the released comments out (it is a new file).

- [ ] **Step 4: Implement the agent subclass, the watcher and the launcher**

Per Interfaces. The launcher passes `--config-name=ft_distill_residual_flow_unet_img --config-dir=brev/dicerl/cfg/libero/$TASK seed=$SEED "$@"` for extra overrides.

- [ ] **Step 5: Run the tests and syntax checks**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero_rl.py && bash -n brev/dicerl/finetune_libero.sh`, then the full files on the box.
Expected: all pass.

- [ ] **Step 6: Gate C smoke on the box**

Launch with `train.num_train_steps=200` (200 iterations) through the launcher's extra overrides, and `--every 100` for the watcher so one monitoring evaluation runs on `model_step_100.pth` (also pass `train.save_freq=100` for the smoke). Record: iterations per hour from the log (excluding the iteration-0 evaluation), the iteration-0 in-training evaluation's wall time, the monitoring evaluation's wall time and success, the shift logging in W&B (`shift/magnitude` present), and that the residual evaluator loaded the checkpoint. Project 13,000 iterations plus 6 monitoring evaluations; if above 48 h, stop and report before Task 5. Kill both smoke sessions by name afterwards and confirm no orphan workers (`pgrep -fc "cfg/libero"` should be 0 for the smoke's processes).

- [ ] **Step 7: Record and commit**

Gate C numbers and the config diff against the released Square image config go into `docs/libero-residual.md`.

```bash
git add brev/dicerl/cfg/libero brev/dicerl/finetune_libero.sh script/dicerl_libero_rl.py script/dicerl_libero_rl_watch.py script/dicerl_libero_watch.py tests/test_dicerl_libero_rl.py docs/libero-residual.md
git commit -m "LIBERO residual RL: shifted-start agent with ramp, config, launcher, held-out watcher; smoke recorded."
```

---

### Task 5: Launch, monitor, final evaluation, records

**Files:**
- Modify: `docs/libero-residual.md`, `docs/probe-outcomes.md`

- [ ] **Step 1: Launch**

Run: `bash brev/dicerl/finetune_libero.sh <task> 42` on the box.
Expected: two sessions, a W&B run in `dice-rl-libero`, the run directory under `$DICE_RL_LOG_DIR/libero-finetune/`.

- [ ] **Step 2: Monitor**

A background loop every 30 minutes reports the latest in-training evaluation row (from the run's `evaluation_results.csv`), the latest `heldout_sweep.json` row, `shift/magnitude`, and session liveness; stop on any traceback. Each held-out row goes into the sweep table in `docs/libero-residual.md` with the residual norm and filter activity from W&B beside it.

- [ ] **Step 3: Stop and final evaluation**

Stop at 850k env steps (iteration 13,281) or after three consecutive monitoring rows without improvement once the ramp has ended, by killing the training session by name after the next checkpoint save. On the final and the best-monitoring checkpoints run the residual evaluator on the four protocols (held-out 294 at seed 0, in-distribution 150, procedural curve 4 × 100), and compare with the prior's baselines from Task 3.

- [ ] **Step 4: Record and commit**

Final table (prior vs residual per protocol with errors and the per-variant breakdown), the decision on the control run, and a digest paragraph in `docs/probe-outcomes.md`.

```bash
git add docs/libero-residual.md docs/probe-outcomes.md
git commit -m "LIBERO residual on shifted robot inits: run record, held-out results against the prior."
```
