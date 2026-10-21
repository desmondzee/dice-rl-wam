# LIBERO-10 Flow Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain an image-conditioned flow-matching prior on one LIBERO-10 task inside the DICE-RL codebase and select a checkpoint at about 50% held-out success.

**Architecture:** A `libero_image` gym wrapper and a `libero` branch in the DICE-RL env factory make LIBERO look like the robomimic image path, so the vendored pretraining agent, multi-step wrappers and configs run unchanged. A converter writes the task's 50 demos in the DICE-RL npz format. A standalone evaluator runs the held-out protocol on pretrain checkpoints, and a launcher runs pretraining with a companion watcher that evaluates every 1,000 epochs and stops in the 40–60% band.

**Tech Stack:** Python 3.11 uv venv on Jarvis (`~/Documents/dice/dicerl-libero-venv`): torch 2.11.0+cu128, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1, numpy 1.26.4, gym 0.22.0, hydra-core 1.3.2, h5py; tests run locally with `.cache/eval-venv/bin/python` (no simulator, no GPU).

**Spec:** `docs/superpowers/specs/2026-10-21-libero-prior-design.md`

## Global Constraints

- Vendored DICE-RL algorithm code is not edited: `agent/`, `model/`, `util/`, `script/run.py`, `script/eval_rl_checkpoint.py`, `cfg/robomimic/`. Only registrations change: `env/gym_utils/wrapper/__init__.py` and one `env_type == "libero"` branch in `env/gym_utils/__init__.py`.
- `.cache/lerobot`, `script/lingbot_*`, the SFT environment, `dicerl-venv311` and the Tool Hang run directories are untouched.
- No comments or docstrings in new code; lean files; commit after each task with no attribution trailer; nothing printed that could carry a secret.
- Task: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10). Demo file on Jarvis: `~/Documents/dice/libero-demos/libero_10/<task>_demo.hdf5`.
- Observation: `rgb` uint8 (128, 128, 6) agentview then wrist on the last axis; `state` float (8,) = eef pos (3), eef axis-angle (3), gripper qpos (2), min–max normalised to [−1, 1]. Actions 7-d normalised in [−1, 1].
- Reward 1.0 on the LIBERO success predicate; termination after 5 cumulative success steps or 520 steps; episode success = reward sum ≥ 1.
- Settle: 10 steps of `[0, 0, 0, 0, 0, 0, -1]` after every reset before the first observation.
- Held-out protocol: initial states 0–49 of the task's stored init file, torch seeds 0 and 1, 100 episodes, 520 steps, 8 actions per chunk, one noise draw per chunk.
- Pretrain config: released Tool Hang image recipe with `obs_dim 8`, `cond_dim 8`, images 128 crop 116, `horizon_steps 8`, `act_steps 8`, `flow_steps 10`, `n_epochs 8000`, `save_model_freq 200`, `env.n_envs 10`, `train.render.num 1`, `max_episode_steps 520`.
- Selection: evaluate every 1,000 epochs during training; stop at the first checkpoint in [0.40, 0.60]; select the in-band checkpoint closest to 0.50, earlier on ties; evaluate raw `model` weights, not `ema`.
- Box operations run through `ssh -F ~/.brev/ssh_config jarvis`, in box-local tmux sessions, never `pkill -f` by pattern (kill by session name or pid).

## Review Focus

- A demo whose images are stored upside-down relative to the live render: gate 0 check 2 must fail loudly rather than pass with the wrong flip (Task 4 test `test_pick_flip_rejects_mismatch`).
- A proprioception dimension whose demo range never crosses zero (eef z): normalisation must use the true min and max, never zero-initialised extremes (Task 3 test `test_normalization_true_extremes`).
- An auto-reset inside `multi_step` between held-out episodes: the evaluator must start every episode with an explicit `reset_arg` carrying `init_state` (Task 5 test `test_eval_resets_with_init_state_every_episode`).
- A success predicate that flickers: the wrapper's cumulative counter must count 5 success steps in total, and success for reporting must be reward sum ≥ 1 after the first success step (Task 2 test `test_termination_after_five_cumulative_successes`).
- A checkpoint sweep where no checkpoint lands in the band: the watcher must not stop the run and must report, not select (Task 6 test `test_select_checkpoint_none_in_band`).

---

### Task 1: LIBERO environment on Jarvis

**Files:**
- Create: `brev/dicerl/setup_libero.sh`
- Create: `brev/dicerl/env_libero.sh`

**Interfaces:**
- Produces: `brev/dicerl/env_libero.sh` exports `DICE_DATA`, `DICE_RL_DATA_DIR`, `DICE_RL_LOG_DIR`, `DICE_RL_WANDB_ENTITY`, `PYTHONPATH`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, and sets `PY="$DICE_DATA/dicerl-libero-venv/bin/python"`. Later tasks source it before every box command.

- [ ] **Step 1: Write `brev/dicerl/setup_libero.sh`**

Copy the structure of `brev/dicerl/setup.sh` with `VENV="$DICE_DATA/dicerl-libero-venv"`, Python 3.11, and this package set in one `uv pip install`: `"numpy==1.26.4" "gym==0.22.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "einops==0.8.0" "imageio==2.35.1" "imageio-ffmpeg==0.6.0" "matplotlib==3.7.5" "pandas==2.0.3" "tqdm==4.66.5" "wandb==0.27.2" "h5py==3.16.0" "hf-libero==0.1.4" "robosuite==1.4.0" "mujoco==3.8.1" "robomimic==0.2.0"`, then torch/torchvision 2.11.0 from the cu128 index as in `setup.sh`. Create `$DICE_DATA/dicerl/data_dir/libero` and `$DICE_DATA/dicerl/log_dir/libero-pretrain`. Final check line: `MUJOCO_GL=egl "$PY" -c 'import torch, gym, robosuite, mujoco, libero, h5py; print(torch.__version__, torch.cuda.is_available(), robosuite.__version__, mujoco.__version__, numpy.__version__)'` with `numpy` imported too.

- [ ] **Step 2: Write `brev/dicerl/env_libero.sh`**

Copy `brev/dicerl/env.sh` and change only `PY` to the new venv.

- [ ] **Step 3: Syntax-check and copy to the box**

Run: `bash -n brev/dicerl/setup_libero.sh brev/dicerl/env_libero.sh && rsync -a -e "ssh -F $HOME/.brev/ssh_config" brev/ jarvis:Documents/dice-rl-wam/brev/`
Expected: no output from `bash -n`; rsync completes.

- [ ] **Step 4: Build the environment on the box**

Run: `ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && bash brev/dicerl/setup_libero.sh 2>&1 | tail -3'`
Expected: last line prints `2.11.0+cu128 True 1.4.0 3.8.1 1.26.4`. If `uv` resolution fails on a pin, report the conflict; do not change pins silently.

- [ ] **Step 5: Smoke the simulator**

Run: `ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && source brev/dicerl/env_libero.sh && "$PY" -c "from libero.libero import benchmark; b=benchmark.get_benchmark(\"libero_10\")(); print([b.get_task(i).name for i in range(10)])"'`
Expected: a list of ten task names including `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`. If LIBERO asks for a config file, note the path it wants; Task 2 writes it.

- [ ] **Step 6: Commit**

```bash
git add brev/dicerl/setup_libero.sh brev/dicerl/env_libero.sh
git commit -m "DICE-RL LIBERO environment: uv venv with the LIBERO stack beside the reproduction venv."
```

---

### Task 2: `libero_image` wrapper and factory branch

**Files:**
- Create: `env/gym_utils/wrapper/libero_image.py`
- Modify: `env/gym_utils/wrapper/__init__.py` (add import and `"libero_image"` entry)
- Modify: `env/gym_utils/__init__.py:200-240` (add the `libero` branch inside `_make_env` before the robomimic branch)
- Test: `tests/test_dicerl_libero.py`

**Interfaces:**
- Produces in `env/gym_utils/wrapper/libero_image.py`:
  - `libero_task_files(task_name: str) -> tuple[str, np.ndarray]` returning the bddl file path and the (50, D) stored initial states for a LIBERO-10 task by name, writing `~/.libero/config.yaml` pointing at the installed package's `bddl_files`, `init_files` and `assets` folders if it is missing (same keys as `script/lingbot_eval.py:141-148`).
  - `quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray` (3,), the conversion LIBERO's dataset used (robosuite `quat2axisangle`).
  - `NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0])`, `SETTLE_STEPS = 10`, `FLIP = "vertical"` module constants; `orient(image: np.ndarray, flip: str) -> np.ndarray` applying `"none"`, `"vertical"` (`[::-1]`) or `"both"` (`[::-1, ::-1]`).
  - `class LiberoImageWrapper(gym.Env)` with `__init__(self, env, shape_meta, normalization_path=None, init_states=None, image_keys=("agentview_image", "robot0_eye_in_hand_image"), success_steps_before_termination=5, max_episode_steps=520, flip=FLIP, render_camera_name="agentview")`, methods `seed(seed=None)`, `reset(options={}, **kwargs)`, `step(action)`, `get_observation(raw_obs) -> dict`, `normalize_obs`, `unnormalize_action`, `get_sim_state()`, `render(mode="rgb_array")` and the `video_path` behaviour of the robomimic wrapper.
  - `make_libero_env(task_name: str, camera_names=("agentview", "robot0_eye_in_hand"), image_hw=128) -> OffScreenRenderEnv`.
- The factory branch: when `env_type == "libero"`, `_make_env` calls `make_libero_env(id)`, then applies `wrappers` through `wrapper_dict` as the robomimic branch does, passing `init_states` from `libero_task_files(id)` into the `libero_image` wrapper's arguments, and returns before any robomimic or d4rl import.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

class FakeLibero:
    def __init__(self, success_after=None, state_dim=47):
        self.t = 0; self.success_after = success_after; self.state = np.zeros(state_dim); self.actions = []
    def reset(self): self.t = 0; return self._obs()
    def set_init_state(self, state): self.state = np.asarray(state); self.t = 0; return self._obs()
    def step(self, action):
        self.actions.append(np.asarray(action)); self.t += 1
        return self._obs(), 0.0, False, {}
    def check_success(self): return self.success_after is not None and self.t >= self.success_after
    def get_sim_state(self): return self.state
    def _obs(self):
        return {"agentview_image": np.full((128, 128, 3), 10, np.uint8), "robot0_eye_in_hand_image": np.full((128, 128, 3), 20, np.uint8),
                "robot0_eef_pos": np.array([0.1, 0.2, 0.9]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]), "robot0_gripper_qpos": np.array([0.04, -0.04])}

def make_wrapper(tmp_path, **kw):
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    np.savez(tmp_path / "norm.npz", obs_min=np.full(8, -1.0), obs_max=np.full(8, 1.0), action_min=np.full(7, -1.0), action_max=np.full(7, 1.0))
    env = kw.pop("env", FakeLibero())
    return env, LiberoImageWrapper(env, shape_meta={"obs": {"rgb": {"shape": [128, 128, 6]}, "state": {"shape": [8]}}, "action": {"shape": [7]}},
                                   normalization_path=str(tmp_path / "norm.npz"), init_states=np.arange(50)[:, None] * np.ones((50, 47)), **kw)

def test_observation_layout(tmp_path):
    env, w = make_wrapper(tmp_path)
    obs = w.reset(options={"seed": 3})
    assert obs["rgb"].shape == (128, 128, 6) and obs["rgb"].dtype == np.uint8
    assert obs["rgb"][0, 0, 0] == 10 and obs["rgb"][0, 0, 3] == 20
    assert obs["state"].shape == (8,) and np.allclose(obs["state"][:3], [0.1, 0.2, 0.9]) and np.allclose(obs["state"][3:6], 0.0)
    assert np.allclose(obs["state"][6:], [0.04, -0.04])

def test_reset_seed_selects_stored_state_and_settles(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"seed": 53})
    assert np.allclose(env.state, 3.0)
    assert len(env.actions) == 10 and np.allclose(env.actions[0], [0, 0, 0, 0, 0, 0, -1])

def test_reset_init_state_option(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.reset(options={"init_state": np.full(47, 7.0)})
    assert np.allclose(env.state, 7.0)

def test_termination_after_five_cumulative_successes(tmp_path):
    env, w = make_wrapper(tmp_path, env=FakeLibero(success_after=3))
    w.reset(options={"seed": 0})
    rewards, terms = [], []
    for _ in range(8):
        _, r, term, trunc, _ = w.step(np.zeros(7)); rewards.append(r); terms.append(term)
    assert rewards[:2] == [0.0, 0.0] and rewards[2] == 1.0
    assert terms.index(True) == 6 and sum(rewards) >= 1

def test_truncation_at_max_steps(tmp_path):
    env, w = make_wrapper(tmp_path, max_episode_steps=4)
    w.reset(options={"seed": 0})
    out = [w.step(np.zeros(7))[3] for _ in range(4)]
    assert out == [False, False, False, True]

def test_orient():
    from env.gym_utils.wrapper.libero_image import orient
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(orient(img, "vertical"), img[::-1]) and np.array_equal(orient(img, "both"), img[::-1, ::-1]) and np.array_equal(orient(img, "none"), img)

def test_action_unnormalize(tmp_path):
    env, w = make_wrapper(tmp_path)
    w.action_min = np.full(7, -0.5); w.action_max = np.full(7, 0.5)
    assert np.allclose(w.unnormalize_action(np.ones(7)), 0.5) and np.allclose(w.unnormalize_action(-np.ones(7)), -0.5)

def test_wrapper_registered():
    from env.gym_utils.wrapper import wrapper_dict
    from env.gym_utils.wrapper.libero_image import LiberoImageWrapper
    assert wrapper_dict["libero_image"] is LiberoImageWrapper
```

Unnormalise formula is the robomimic one: `(a + 1) / 2 * (max - min) + min`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py`
Expected: ImportError on `env.gym_utils.wrapper.libero_image`. If importing `env.gym_utils.wrapper` fails locally because `robomimic` or `gym` is missing in the eval venv, guard the libero import in the test with `pytest.importorskip("gym")` and run that test on the box with `$PY -m pytest` instead; record which.

- [ ] **Step 3: Implement `env/gym_utils/wrapper/libero_image.py`**

Mirror `RobomimicImageWrapper` for `get_observation`, `normalize_obs`, `unnormalize_action`, `seed`, video writing and the success counter. `reset`: close any video writer; open one if `video_path` given; pick the state (`init_state` option, else stored state `seed % len(init_states)` after `np.random.seed(seed)`, else a random stored state); call `env.reset()` once on first use then `env.set_init_state(state)`; step `NOOP` `SETTLE_STEPS` times; zero the counters; return `get_observation`. `step`: un-normalise, `env.step`, reward `float(env.check_success())`, counters as the robomimic wrapper, `truncated` when `step_count >= max_episode_steps`. Images: `orient(raw[key], self.flip)` then concatenate on the last axis. State: `[eef_pos, quat_to_axis_angle(eef_quat), gripper_qpos]`.

- [ ] **Step 4: Register the wrapper and add the factory branch**

In `env/gym_utils/wrapper/__init__.py` import `LiberoImageWrapper` and add `"libero_image": LiberoImageWrapper`. In `env/gym_utils/__init__.py` `_make_env`, add before the `if robomimic_env_cfg_path is not None:` line:

```python
if env_type == "libero":
    from env.gym_utils.wrapper.libero_image import make_libero_env, libero_task_files
    env = make_libero_env(id)
    _, init_states = libero_task_files(id)
    for wrapper, args in wrappers.items():
        extra = {"init_states": init_states} if wrapper == "libero_image" else {}
        env = wrapper_dict[wrapper](env, **args, **extra)
    return env
```

Also set `os.environ["MUJOCO_GL"] = "egl"` in that branch as the robomimic branch does.

- [ ] **Step 5: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py`
Expected: all pass.

- [ ] **Step 6: Live check on the box**

Run: `rsync -a -e "ssh -F $HOME/.brev/ssh_config" env/ tests/ jarvis:Documents/dice-rl-wam/ --relative` (or rsync the two directories) then `ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && source brev/dicerl/env_libero.sh && "$PY" -c "
from env.gym_utils.wrapper.libero_image import make_libero_env, libero_task_files, LiberoImageWrapper
t=\"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it\"; bddl, states = libero_task_files(t); print(bddl, states.shape)
w = LiberoImageWrapper(make_libero_env(t), shape_meta=None, init_states=states); o = w.reset(options={\"seed\": 0}); print(o[\"rgb\"].shape, o[\"state\"]); print(w.step(__import__(\"numpy\").zeros(7))[1:4])"'`
Expected: `(50, 47)`-shaped states (or the task's state dimension), `(128, 128, 6)`, an 8-vector, and `(0.0, False, False)`.

- [ ] **Step 7: Commit**

```bash
git add env/gym_utils/wrapper/libero_image.py env/gym_utils/wrapper/__init__.py env/gym_utils/__init__.py tests/test_dicerl_libero.py
git commit -m "LIBERO image wrapper and env factory branch for DICE-RL."
```

---

### Task 3: Demo converter

**Files:**
- Create: `script/dataset/process_libero_dataset.py`
- Test: `tests/test_dicerl_libero.py` (append)

**Interfaces:**
- Produces: `convert(hdf5_path: str, out_dir: str, flip: str = "vertical") -> dict` writing `out_dir/train.npz` with keys `states` (N, 8) float32 normalised, `actions` (N, 7) float32 normalised, `images` (N, 6, 128, 128) uint8, `rewards` (N,) float32, `terminals` (N,) bool, `traj_lengths` (E,) int64, and `out_dir/normalization.npz` with `obs_min`, `obs_max`, `action_min`, `action_max` (true extremes over all demos), returning the shapes. CLI: `python -m script.dataset.process_libero_dataset --hdf5 PATH --out DIR [--flip vertical]`.
- Consumes: `orient` and the state order from Task 2 (proprio = `ee_pos`, `ee_ori`, `gripper_states` from the demo `obs` group; `ee_ori` is already axis-angle).

- [ ] **Step 1: Write the failing tests**

```python
def make_demo_h5(path, lengths=(5, 7)):
    import h5py
    with h5py.File(path, "w") as f:
        for i, n in enumerate(lengths):
            g = f.create_group(f"data/demo_{i}")
            o = g.create_group("obs")
            o["agentview_rgb"] = np.full((n, 128, 128, 3), i + 1, np.uint8)
            o["eye_in_hand_rgb"] = np.full((n, 128, 128, 3), i + 11, np.uint8)
            o["ee_pos"] = np.tile([0.0, 0.5, 1.0], (n, 1)) + i
            o["ee_ori"] = np.zeros((n, 3)); o["gripper_states"] = np.tile([0.02, -0.02], (n, 1))
            g["actions"] = np.linspace(-0.5, 0.5, n * 7).reshape(n, 7)
            g["rewards"] = np.r_[np.zeros(n - 1), 1.0]; g["dones"] = np.r_[np.zeros(n - 1), 1.0]
            g["states"] = np.zeros((n, 47))

def test_convert_shapes_and_order(tmp_path):
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5")
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"))
    d = np.load(tmp_path / "out" / "train.npz")
    assert d["states"].shape == (12, 8) and d["actions"].shape == (12, 7) and d["images"].shape == (12, 6, 128, 128)
    assert d["images"].dtype == np.uint8 and d["images"][0, 0, 0, 0] == 1 and d["images"][0, 3, 0, 0] == 11 and d["images"][5, 0, 0, 0] == 2
    assert list(d["traj_lengths"]) == [5, 7] and d["rewards"].sum() == 2 and d["terminals"].sum() == 2

def test_normalization_true_extremes(tmp_path):
    from script.dataset.process_libero_dataset import convert
    make_demo_h5(tmp_path / "d.hdf5")
    convert(str(tmp_path / "d.hdf5"), str(tmp_path / "out"))
    n = np.load(tmp_path / "out" / "normalization.npz"); d = np.load(tmp_path / "out" / "train.npz")
    assert np.allclose(n["obs_min"][:3], [0.0, 0.5, 1.0]) and np.allclose(n["obs_max"][:3], [1.0, 1.5, 2.0])
    assert n["obs_min"][2] > 0
    assert d["states"].min() >= -1 - 1e-6 and d["states"].max() <= 1 + 1e-6 and np.isfinite(d["states"]).all()
    assert np.isclose(n["action_min"].min(), -0.5) and np.isclose(n["action_max"].max(), 0.5)
```

Normalisation: `2 * (x - min) / (max - min) - 1`, with a constant dimension (max == min) mapped to 0 (e.g. `ee_ori` here).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py -k convert_or_normalization`
Expected: ImportError on `script.dataset.process_libero_dataset`.

- [ ] **Step 3: Implement `script/dataset/process_libero_dataset.py`**

`convert` iterates demos in sorted numeric order (`demo_0`, `demo_1`, …), builds per-step arrays, images as `np.concatenate([orient(agent, flip), orient(wrist, flip)], -1).transpose(0, 3, 1, 2)`, computes extremes, normalises, writes with `np.savez_compressed`, and returns `{"episodes": E, "steps": N}`. `main()` parses `--hdf5`, `--out`, `--flip` and prints the returned dict.

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py`
Expected: all pass.

- [ ] **Step 5: Convert the real file on the box**

Run: `rsync -a -e "ssh -F $HOME/.brev/ssh_config" script/dataset/process_libero_dataset.py jarvis:Documents/dice-rl-wam/script/dataset/ && ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && source brev/dicerl/env_libero.sh && T=KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it && "$PY" -m script.dataset.process_libero_dataset --hdf5 ~/Documents/dice/libero-demos/libero_10/${T}_demo.hdf5 --out $DICE_RL_DATA_DIR/libero/$T/ph_pretrain && ls -la $DICE_RL_DATA_DIR/libero/$T/ph_pretrain'`
Expected: `{"episodes": 50, "steps": <about 15000>}` and two files; `train.npz` around 1 GB.

- [ ] **Step 6: Commit**

```bash
git add script/dataset/process_libero_dataset.py tests/test_dicerl_libero.py
git commit -m "LIBERO demo converter to the DICE-RL pretraining format."
```

---

### Task 4: Gate 0 — replay, image orientation, throughput

**Files:**
- Create: `script/dicerl_libero_gate0.py`
- Create: `docs/libero-prior.md`
- Test: `tests/test_dicerl_libero.py` (append)

**Interfaces:**
- Produces: `pick_flip(rendered: dict[str, np.ndarray], stored: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]` returning the flip in `("none", "vertical", "both")` whose mean absolute pixel difference against the stored demo frame is below 2/255 for both cameras, raising `ValueError` with the three scores if none qualifies; `replay_demo(hdf5_path, task_name, demo="demo_0") -> dict` with keys `success` (bool), `steps`, `first_success_step`; `throughput(task_name, n_envs, steps=200) -> float` env steps per second through `make_async` with the `libero_image` and `multi_step` wrappers and random actions. CLI writes `gate0.json` next to the run data and prints it.

- [ ] **Step 1: Write the failing test**

```python
def test_pick_flip_rejects_mismatch():
    from script.dicerl_libero_gate0 import pick_flip
    img = np.random.RandomState(0).randint(0, 255, (128, 128, 3), np.uint8)
    rendered = {"agentview": img, "wrist": img}
    stored = {"agentview": img[::-1], "wrist": img[::-1]}
    flip, scores = pick_flip(rendered, stored)
    assert flip == "vertical" and scores["vertical"] < 2 / 255
    with pytest.raises(ValueError):
        pick_flip(rendered, {"agentview": np.roll(img, 7, 0), "wrist": img})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py -k pick_flip`
Expected: ImportError.

- [ ] **Step 3: Implement `script/dicerl_libero_gate0.py`**

`replay_demo`: build the raw env with `make_libero_env`, `reset()`, `set_init_state(states[0])` from the hdf5, step the demo's actions in order, record the first step where `check_success()` is true; also capture the raw rendered images at step 0 and compare them with `obs/agentview_rgb[0]` and `obs/eye_in_hand_rgb[0]` through `pick_flip` (scores as mean absolute difference over float pixels divided by 255). `throughput`: `make_async(task, env_type="libero", num_envs=n, asynchronous=True, max_episode_steps=520, wrappers=<libero_image with normalization_path None and multi_step n_obs_steps 1, n_action_steps 8, reset_within_step True>, obs_dim=8, action_dim=7, use_image_obs=True)`, `reset_arg` with seeds, then 200 steps of uniform random chunks; return `n * 200 * 8 / elapsed`. CLI `--task --hdf5 --out` runs replay, flip and throughput for 1, 4, 20 envs and writes `gate0.json`.

- [ ] **Step 4: Run the unit test**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py`
Expected: all pass.

- [ ] **Step 5: Run gate 0 on the box**

Run: `rsync -a -e "ssh -F $HOME/.brev/ssh_config" script/dicerl_libero_gate0.py jarvis:Documents/dice-rl-wam/script/ && ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && source brev/dicerl/env_libero.sh && T=KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it && "$PY" -m script.dicerl_libero_gate0 --task $T --hdf5 ~/Documents/dice/libero-demos/libero_10/${T}_demo.hdf5 --out $DICE_RL_DATA_DIR/libero/$T 2>&1 | tail -5'`
Expected: `success: true`, a flip name, three throughput numbers. If the flip chosen is not `vertical`, set `FLIP` in `libero_image.py` to the chosen value and re-run Task 3 Step 5 with `--flip <value>` before continuing; record it.

- [ ] **Step 6: Write `docs/libero-prior.md`**

Sections: task and data (file, demos, lengths), environment table (from the spec), gate 0 results (replay success and first-success step, flip and the three scores, throughput at 1/4/20 envs and the projected 100-episode evaluation time), and empty headings for pretraining and selection.

- [ ] **Step 7: Commit**

```bash
git add script/dicerl_libero_gate0.py tests/test_dicerl_libero.py docs/libero-prior.md
git commit -m "LIBERO gate 0: demo replay, image orientation and throughput on Jarvis."
```

---

### Task 5: Held-out evaluator for pretrain checkpoints

**Files:**
- Create: `script/dicerl_libero_eval.py`
- Test: `tests/test_dicerl_libero.py` (append)

**Interfaces:**
- Produces: `episode_plan(n_states: int = 50, seeds=(0, 1)) -> list[dict]` of `{"init_state_id", "seed"}` in state-major order; `run_protocol(policy, venv, init_states, plan, n_envs, act_steps=8, max_steps=520) -> list[dict]` with per-episode `{"init_state_id", "seed", "success", "length"}` where each batch of `n_envs` episodes starts with `venv.reset_arg(options_list=[{"init_state": init_states[id]} ...])` and `torch.manual_seed(seed)` before the batch; `summarise(rows) -> dict` with `success_rate`, `stderr` (binomial), `n`, `mean_length`; `load_prior(ckpt_path, device) -> (cfg, model)` loading `.hydra/config.yaml` from the run directory (two levels up from `checkpoint/`), instantiating `cfg.model`, loading `checkpoint["model"]`; CLI `--ckpt --envs 20 --out DIR` writing `episodes.csv` and `summary.json`.
- `policy(obs_batch) -> np.ndarray (n, act_steps, 7)` is the callable `run_protocol` takes; the CLI wraps the flow model exactly like `agent/pretrain/train_flow_matching_img_agent.py:118-128` (rgb permuted to channels-first, `samples.trajectories[:, :act_steps]`).

- [ ] **Step 1: Write the failing tests**

```python
def test_episode_plan_order():
    from script.dicerl_libero_eval import episode_plan
    plan = episode_plan(3, (0, 1))
    assert plan == [{"init_state_id": 0, "seed": 0}, {"init_state_id": 0, "seed": 1}, {"init_state_id": 1, "seed": 0}, {"init_state_id": 1, "seed": 1}, {"init_state_id": 2, "seed": 0}, {"init_state_id": 2, "seed": 1}]

class FakeVenv:
    def __init__(self, n, succeed_at):
        self.n = n; self.succeed_at = succeed_at; self.resets = []; self.t = 0
    def _obs(self):
        return {"rgb": np.zeros((self.n, 1, 128, 128, 6), np.uint8), "state": np.zeros((self.n, 1, 8))}
    def reset_arg(self, options_list):
        self.resets.append([o["init_state"][0] for o in options_list]); self.t = 0
        return self._obs()
    def step(self, actions):
        self.t += 1
        r = np.array([1.0 if self.t == self.succeed_at[i] else 0.0 for i in range(self.n)])
        term = np.array([self.t >= self.succeed_at[i] + 4 for i in range(self.n)])
        return self._obs(), r, term, np.zeros(self.n, bool), [{}] * self.n

def test_eval_resets_with_init_state_every_episode():
    from script.dicerl_libero_eval import episode_plan, run_protocol, summarise
    venv = FakeVenv(2, succeed_at=[3, 10 ** 6])
    init_states = np.arange(4)[:, None] * np.ones((4, 47))
    rows = run_protocol(lambda obs: np.zeros((2, 8, 7)), venv, init_states, episode_plan(4, (0,)), n_envs=2, max_steps=40)
    assert venv.resets == [[0.0, 1.0], [2.0, 3.0]]
    assert [r["success"] for r in rows] == [True, False, True, False] and rows[0]["length"] == 7 * 8 and rows[1]["length"] == 40
    s = summarise(rows); assert s["n"] == 4 and s["success_rate"] == 0.5 and abs(s["stderr"] - 0.25) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py -k "episode_plan or eval_resets"`
Expected: ImportError.

- [ ] **Step 3: Implement `script/dicerl_libero_eval.py`**

`run_protocol` processes the plan in batches of `n_envs` (padding the last batch by repeating its last entry and discarding the extra rows), stepping `ceil(max_steps / act_steps)` chunks, accumulating reward and `act_steps` of length per env until termination, truncation or `max_steps`; `success = reward_sum >= 1`. The CLI builds the venv through `make_async` with the run config's `env` block (`n_envs` from `--envs`), `libero_task_files` for the states, and writes outputs under `--out` (default: next to the checkpoint, `eval_<epoch>/`).

- [ ] **Step 4: Run the tests**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add script/dicerl_libero_eval.py tests/test_dicerl_libero.py
git commit -m "Held-out evaluator for LIBERO pretrain checkpoints (50 states x 2 seeds, 520 steps)."
```

---

### Task 6: Pretrain config, launcher, watcher and smoke run

**Files:**
- Create: `brev/dicerl/cfg/libero/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/pre_flow_matching_unet_img.yaml`
- Create: `brev/dicerl/pretrain_libero.sh`
- Create: `script/dicerl_libero_watch.py`
- Test: `tests/test_dicerl_libero.py` (append)

**Interfaces:**
- Produces: `select_checkpoint(rows: list[dict]) -> dict | None` over rows `{"epoch", "success_rate"}` returning the in-band row closest to 0.5 (earlier epoch on ties) or `None`; `in_band(success_rate: float) -> bool` for [0.40, 0.60]. `script/dicerl_libero_watch.py --run-dir DIR --every 1000 --envs 20 --session NAME` polls `DIR/checkpoint/state_<epoch>.pt` for epochs that are multiples of `--every`, evaluates each new one with Task 5's CLI into `DIR/eval_<epoch>/`, appends `{"epoch", "success_rate", "stderr"}` to `DIR/prior_sweep.json`, and when `in_band` holds runs `tmux kill-session -t NAME` and exits 0; exits 1 with a message when epoch 8000 has been evaluated without a band hit.
- `brev/dicerl/pretrain_libero.sh <task> [seed] [--epochs N]`: sources `env_libero.sh`, starts tmux session `dicerl-pre-<task-short>-<seed>` running `script/run.py --config-name=pre_flow_matching_unet_img --config-dir=brev/dicerl/cfg/libero/<task> seed=<seed> [train.n_epochs=N]`, waits for the run directory to appear, then starts tmux session `dicerl-watch-<seed>` running the watcher against it.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_checkpoint_none_in_band():
    from script.dicerl_libero_watch import select_checkpoint, in_band
    assert select_checkpoint([{"epoch": 1000, "success_rate": 0.7}, {"epoch": 2000, "success_rate": 0.65}]) is None
    assert not in_band(0.39) and in_band(0.40) and in_band(0.60) and not in_band(0.61)

def test_select_checkpoint_closest_to_half_earlier_on_ties():
    from script.dicerl_libero_watch import select_checkpoint
    rows = [{"epoch": 1000, "success_rate": 0.42}, {"epoch": 2000, "success_rate": 0.56}, {"epoch": 3000, "success_rate": 0.44}]
    assert select_checkpoint(rows)["epoch"] == 2000
    rows = [{"epoch": 1000, "success_rate": 0.45}, {"epoch": 2000, "success_rate": 0.55}]
    assert select_checkpoint(rows)["epoch"] == 1000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py -k select_checkpoint`
Expected: ImportError.

- [ ] **Step 3: Write the config**

Copy `cfg/robomimic/pretrain/tool_hang/pre_flow_matching_unet_img.yaml` and change exactly: `logdir: ${oc.env:DICE_RL_LOG_DIR}/libero-pretrain/${name}/${now:%Y-%m-%d}_${now:%H-%M-%S}_${seed}`; remove `robomimic_env_cfg_path`; `train_dataset_path: data_dir/libero/${env_name}/ph_pretrain/train.npz`; `normalization_path: data_dir/libero/${env_name}/ph_pretrain/normalization.npz`; `env_name: <task>`; `obs_dim: 8`; `flow_steps: 10`; `env.env_type: libero`; `env.n_envs: 10`; `env.max_episode_steps: 520`; `env.wrappers` = `libero_image: {normalization_path: ${normalization_path}, shape_meta: ${shape_meta}}` followed by the unchanged `multi_step` block; `shape_meta.obs.rgb.shape: [128, 128, 6]`, `shape_meta.obs.state.shape: [8]`; `wandb.project: dice-rl-libero`, `wandb.entity: ${oc.env:DICE_RL_WANDB_ENTITY}`; `train.render.num: 1`; `model.network.cond_dim: 8`, `img_height: 128`, `img_width: 128`; the network's random-crop size (find its key in `model/flow_matching/vision_resnet_mlp_flow_matching.py` and the Tool Hang yaml, e.g. `random_crop`/`crop_shape`) set to 116. Leave `horizon_steps: 8`, `act_steps: 8`, `n_epochs: 8000`, `save_model_freq: 200`, batch and optimiser as released.

- [ ] **Step 4: Implement `script/dicerl_libero_watch.py` and `brev/dicerl/pretrain_libero.sh`**

The watcher sleeps 300 s between polls, runs the evaluator as a subprocess with `$PY -m script.dicerl_libero_eval --ckpt <path> --envs <n> --out <run>/eval_<epoch>`, and reads its `summary.json`. The launcher follows `brev/dicerl/finetune.sh`'s tmux and logging pattern (log `$DICE_RL_LOG_DIR/pretrain_<task-short>_<seed>.log`).

- [ ] **Step 5: Run the tests and syntax checks**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_dicerl_libero.py && bash -n brev/dicerl/pretrain_libero.sh`
Expected: all pass, no output from `bash -n`.

- [ ] **Step 6: Smoke run on the box (100 epochs)**

Run: `rsync -a -e "ssh -F $HOME/.brev/ssh_config" brev/ script/ jarvis:Documents/dice-rl-wam/ --relative` then `ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && bash brev/dicerl/pretrain_libero.sh KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it 42 --epochs 100'`; poll the log every few minutes with `tail`.
Expected: the run reaches epoch 100 with one in-training rollout logged and no traceback; note seconds per epoch and the rollout's wall time; project 8,000 epochs. If the projection exceeds 24 h, stop and report before Task 7. Kill the smoke session by name when done (`tmux kill-session -t <name>`); the watcher will have found no 1,000-epoch checkpoint and should be killed by name too.

- [ ] **Step 7: Evaluate the smoke checkpoint once**

Run the Task 5 CLI on `state_100.pt` of the smoke run with `--envs 20`.
Expected: `summary.json` with `n: 100`, a success rate near zero, and the wall time recorded for `docs/libero-prior.md` (gate 0 item 3).

- [ ] **Step 8: Commit**

```bash
git add brev/dicerl/cfg/libero brev/dicerl/pretrain_libero.sh script/dicerl_libero_watch.py tests/test_dicerl_libero.py docs/libero-prior.md
git commit -m "LIBERO prior pretraining config, launcher and band-stopping watcher; smoke run recorded."
```

---

### Task 7: Full pretraining run and checkpoint selection

**Files:**
- Modify: `docs/libero-prior.md`
- Modify: `docs/probe-outcomes.md` (one digest paragraph)

**Interfaces:**
- Consumes: Task 6's launcher and watcher; Task 5's outputs.

- [ ] **Step 1: Launch**

Run: `ssh -F ~/.brev/ssh_config jarvis 'cd ~/Documents/dice-rl-wam && bash brev/dicerl/pretrain_libero.sh KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it 42'`
Expected: two tmux sessions (pretrain and watcher) and a W&B run in `dice-rl-libero`.

- [ ] **Step 2: Monitor**

Check every 30 minutes with a background loop that prints the last `prior_sweep.json` row and whether both sessions are alive; stop on any traceback. Each new row goes into the table in `docs/libero-prior.md`.

- [ ] **Step 3: Selection**

When the watcher exits 0, record the selected checkpoint (epoch, success rate, standard error, path) in `docs/libero-prior.md` and a one-paragraph digest in `docs/probe-outcomes.md`. If the watcher exits 1 above the band, stop and report to the user with the sweep table before any 20-demo retrain; if below the band at 8,000 epochs, report likewise.

- [ ] **Step 4: Commit**

```bash
git add docs/libero-prior.md docs/probe-outcomes.md
git commit -m "LIBERO prior: pretraining sweep and selected checkpoint."
```
