# LingBot-VA DSRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train DSRL-SAC (RL over LingBot-VA's action input noise) on the same frozen LIBERO-10 step-600 prior, budget, critic state, and eval protocol as the DICE-RL runs. Then compare the two on sample efficiency, stability, and 200-episode success, running on a personal Slurm HPC.

**Spec:** `docs/superpowers/specs/2026-09-17-lingbot-dsrl-design.md`. Read §3, §5, §7, and §8 before starting.

**Architecture:** New `script/lingbot_dsrl_*.py` modules. They import (never modify semantics of) the DICE port's pinned policy loader, pooled critic state, env factory, eval harness, and resume helpers. The frozen 5B prior is called only at collection and eval. Updates are pure MLP SAC over a 7-d tanh-bounded noise vector, tiled across the 16-token action chunk. The only edit to existing code is a behaviour-preserving split of `ResidualLingBotPolicy.decode_candidates` into "pool state" and "decode with given noise", so the actor can see `s` before choosing `w`. HPC launch is Slurm + a uv venv (or Apptainer) and replaces Modal for this work only.

**Tech Stack:** PyTorch, numpy, pytest, pinned LeRobot `3f2c29ef` (`lingbot_va`, `libero`, `evaluation` extras), hf-libero 0.1.4, MuJoCo EGL, W&B, Slurm.

## Global Constraints

- The prior pin is unchanged: `load_residual_policy` + `frozen_prior_kwargs()` (20 video / 50 action Euler, `video_exec_step=-1`, CFG 5/1, SNR 5/0.05). `RLConfig.validate()` rules apply to DSRL too.
- With `actor is None`, the DSRL policy must decode with `ε_a ~ N(0,I)` and reproduce the SFT path exactly (it is the prior).
- Budget is 100,000 env action steps including warm-up. One `LiberoEnv`, uniform task sampling, 520-step horizon, success reward 1 on the first success chunk, and `done` on success/termination/horizon, all identical to `lingbot_rl_train.train`.
- Do not edit `.cache/lerobot`, `.cache/lingbot-va`, the Hydra robomimic stack, `script/lingbot_eval.py` protocol fields, or DICE training semantics in `lingbot_rl_{model,buffer,train,data}.py`.
- Local verification uses `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_dsrl.py`. Local tests never allocate a GPU, download models, call Slurm, or call Modal.
- Repo style: no comments or docstrings in new code; plain imperative commit messages; no conventional-commit prefixes.
- No secrets in argv, logs, configs, or artifacts. `WANDB_API_KEY` comes from the environment, and `HF_TOKEN` is used only in the prepare step.

## File Map

| File | Status | Responsibility |
| --- | --- | --- |
| `script/lingbot_rl_policy.py` | modify | split `decode_candidates` into `pool_chunk_state` + `decode_with_noise` (no behaviour change) |
| `script/lingbot_dsrl_config.py` | create | pinned DSRL recipe, protocol fingerprint, eval schedule |
| `script/lingbot_dsrl_model.py` | create | noise tiling, tanh-Gaussian noise actor, Q^W ensemble, temperature, SAC updates, state dicts |
| `script/lingbot_dsrl_buffer.py` | create | chunk replay `(s, w, r, done, s', discount)` with pending-transition completion |
| `script/lingbot_dsrl_policy.py` | create | `SteeredLingBotPolicy` for eval (deterministic / best-of-4) |
| `script/lingbot_dsrl_train.py` | create | collection loop, warm-up, updates, timers, train-time evals, resume, `train`/`eval` CLI |
| `script/lingbot_dsrl_probe.py` | create | Phase 0 steerability probe (GPU, JSON report) |
| `script/lingbot_compare_report.py` | create | stdlib CSV/JSON comparison of SFT vs DICE vs DSRL with Fisher/z tests |
| `hpc/env.sh`, `hpc/setup.sh`, `hpc/prepare.sbatch`, `hpc/probe.sbatch`, `hpc/train.sbatch`, `hpc/eval.sbatch`, `hpc/submit_chain.sh` | create | Slurm launchers |
| `tests/test_lingbot_dsrl.py` | create | CPU tests |
| `AGENTS.md`, `README.md` | modify | DSRL section + HPC commands |

---

### Task 0: Stand up the HPC environment and stage assets

**Files:**
- Create: `hpc/env.sh`, `hpc/setup.sh`, `hpc/prepare.sbatch`

This task needs the operator. It is where Modal-specific assumptions break, so do it first.

- [ ] **Step 1: Record cluster facts in `hpc/env.sh`.** GPU type/VRAM, partition, max wall time, whether compute nodes have internet, and whether Apptainer is available. Template:

```bash
export DSRL_ROOT=${DSRL_ROOT:-$HOME/dice-rl-wam}
export CACHE_ROOT=${CACHE_ROOT:-/scratch/$USER/lingbot-cache}
export RUNS_ROOT=${RUNS_ROOT:-/scratch/$USER/lingbot-runs}
export LEROBOT_SOURCE_ROOT=$DSRL_ROOT/.cache/lerobot
export EVAL_PY=$DSRL_ROOT/.cache/eval-venv/bin/python
export HF_HOME=$CACHE_ROOT/hub
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
export LIBERO_CONFIG_PATH=${TMPDIR:-/tmp}/lingbot-libero-config-$SLURM_JOB_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$DSRL_ROOT
export SLURM_PARTITION=${SLURM_PARTITION:-gpu}
export SLURM_MAX_HOURS=${SLURM_MAX_HOURS:-24}
```

- [ ] **Step 2: Write `hpc/setup.sh`.** It mirrors `script/lingbot_rl_modal.py::build_image` minus Modal: clone LeRobot at `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e` into `.cache/lerobot`, run `UV_PROJECT_ENVIRONMENT=$DSRL_ROOT/.cache/eval-venv uv sync --index-url https://pypi.org/simple --project .cache/lerobot --python 3.12 --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation`, export constraints, then `uv pip install --constraint ... wandb pytest==8.4.1 imageio==2.37.4 imageio-ffmpeg==0.6.0`. It must end with:

```bash
$EVAL_PY -c 'from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy; import wandb'
$EVAL_PY -m script.lingbot_rl_config > /dev/null
$EVAL_PY -m pytest -q tests/test_lingbot_rl.py
```

Expected: the import succeeds, the config prints, and the DICE suite passes on the cluster.

- [ ] **Step 3: Copy the SFT checkpoint off Modal (operator, once, anywhere with Modal auth):**

```bash
mkdir -p checkpoints/lingbot-sft/libero30-sft/checkpoints
uv run --no-project --with modal==1.1.4 modal volume get dice-lingbot-sft-runs libero30-sft/checkpoints/step_000600 checkpoints/lingbot-sft/libero30-sft/checkpoints
rsync -a checkpoints/lingbot-sft/ hpc-login:/scratch/$USER/lingbot-sft/
```

`prepare_evaluation` requires `.../libero30-sft/checkpoints/step_000600` (it checks `checkpoint.parent.parent.name == "libero30-sft"`). Keep that layout.

- [ ] **Step 4: `hpc/prepare.sbatch`** (CPU partition or login node, needs internet + `HF_TOKEN` in env):

```bash
source hpc/env.sh
$EVAL_PY -m script.lingbot_eval prepare \
  --config-json '{"source_run":"libero30-sft","checkpoint_step":600,"stage":"eval","seed":42}' \
  --cache-root "$CACHE_ROOT" \
  --checkpoint /scratch/$USER/lingbot-sft/libero30-sft/checkpoints/step_000600 \
  --prepared-output "$CACHE_ROOT/prepared-pointer.json"
```

Expected: `prepared-pointer.json` holds `{"prepared_path": ".../lingbot-eval/<fp>/prepared.json"}`.

- [ ] **Step 5: GPU smoke of the unchanged SFT path.** Run one episode of task 5 through `script.lingbot_eval.run_episode` with `load_policy`, and log per-chunk time and `torch.cuda.max_memory_allocated()`. Expected: finite actions, a boolean `is_success`, and peak memory ≈ 14 GB. This proves EGL + weights + assets on the cluster.

- [ ] **Step 6: Commit** `hpc/env.sh hpc/setup.sh hpc/prepare.sbatch`: "Add Slurm environment setup and asset preparation for LingBot RL on HPC."

---

### Task 1: Split chunk decoding so a noise policy can see the state first

**Files:**
- Modify: `script/lingbot_rl_policy.py` (`ResidualLingBotPolicy.decode_candidates`)
- Test: `tests/test_lingbot_dsrl.py`

Today `decode_candidates` pools `s` and then calls `_infer` with `action_noise` fixed at call time. DSRL needs `s → π(w|s) → noise → decode`.

**Interfaces produced:**
- `policy.pool_chunk_state(batch) -> dict(s, init_latent, frame_st_id, first_chunk)`: does the obs encoding, streaming-cache init / `_compute_kv_cache`, and `_pool_from_latent`. It does not run `_infer`.
- `policy.decode_with_noise(ctx, k=1, video_noise=None, action_noise=None) -> dict(s, z, a_base, video_noise, latents, first_chunk)`: runs `_infer`, flips `_first_chunk`, resets `_exec_step`, sets `_started`.
- `decode_candidates(batch, k, video_noise, action_noise)` becomes `decode_with_noise(pool_chunk_state(batch), ...)` and returns an identical dict.

- [ ] **Step 1: Write the failing test.** Use `ResidualLingBotPolicy.__new__` with monkeypatched `_ensure_frozen_modules`, `_maybe_init_prompt`, `_start_raw_obs`, `_encode_frames`, `_encode_isolated`, `_init_streaming_cache`, `_compute_kv_cache`, `_pool_from_latent`, and `_infer` (all recording calls, returning small fixed tensors). Assert that:
  1. `decode_candidates(batch, k=4)` and `decode_with_noise(pool_chunk_state(batch), k=4)` on two fresh instances produce equal tensors.
  2. The call order is `…_pool_from_latent` before `_infer`.
  3. `_first_chunk` is still `True` after `pool_chunk_state` and `False` after `decode_with_noise`.

- [ ] **Step 2: Run** `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_dsrl.py -k split`. Expected: FAIL (`AttributeError: pool_chunk_state`).

- [ ] **Step 3: Implement** by moving the body of `decode_candidates` up to and including `state = self._pool_from_latent(critic_latent)` into `pool_chunk_state`, and the rest into `decode_with_noise`. Decorate both `@torch.no_grad()`.

- [ ] **Step 4: Run** the new test and the full DICE suite `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py`. Expected: all pass. DICE collection must be untouched.

- [ ] **Step 5: Commit:** "Split LingBot chunk decoding into state pooling and noise-conditioned decode."

---

### Task 2: Phase 0 steerability probe

**Files:**
- Create: `script/lingbot_dsrl_probe.py`, `hpc/probe.sbatch`
- Test: `tests/test_lingbot_dsrl.py` (pure-math helpers only)

This implements spec §8 and gates §5.2/§5.3. Run it before writing Tasks 3–8 in full, because its result can change the latent space.

**Interfaces produced:**
- `variance_shares(actions)`: `actions` shaped `(V, A, 16, 7)` (V video noises × A action noises). Returns `{"total", "action_share", "video_share", "interaction_share"}` from a two-way random-effects decomposition, averaged over the 112 dims.
- `tie_noise(w7, generator)` is imported from Task 4. The probe temporarily defines its own copy and replaces it with the import after Task 4 lands.
- CLI: `python -m script.lingbot_dsrl_probe --prepared-path P --output out.json --states-per-task 5 --grid 4`.

- [ ] **Step 1: Failing test for `variance_shares`.** Synthetic `a[v,j] = v_effect[v] + a_effect[j]` with known variances (e.g. video var 4, action var 1, no interaction). Assert `action_share ≈ 0.2` and `video_share ≈ 0.8` within 0.05 using large V, A (e.g. 200×200). Also check that a pure-interaction tensor yields `interaction_share ≈ 1`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `variance_shares` with grand mean, row means, column means, and cell residuals. Use unbiased two-way ANOVA components clipped at 0 and normalised by their sum.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Implement the GPU probe body.**
  - For each task and state: `policy.reset()`, `env.reset(seed=42 + i)`, step the prior for `i*16` actions to reach a later chunk (i ∈ {0, 3}), then `ctx = pool_chunk_state(batch)`.
  - For each of V=4 video noises, decode A=4 action noises in one batched call. Use `_snapshot_kv`/`_restore_kv` and the streaming state snapshot so every decode starts from the same cache. Collect `a_base[..., :7]`.
  - Tied reach: 32 tied `w ~ U[−b_W, b_W]^7` for b_W ∈ {1, 1.5, 2.5} versus 32 untied draws rescaled to the same norm. Record per-dim std.
  - Mode coverage on tasks 4, 8, and 9: 8 episodes each with a fixed random tied `w` per episode, recording success.
  - Timers: per-chunk wall-clock for k=1 and k=4, plus `max_memory_allocated`.
- [ ] **Step 6: `hpc/probe.sbatch`** (1 GPU, 6 h), then run it. Write `$RUNS_ROOT/probe/probe.json`.
- [ ] **Step 7: Decision.** Apply the spec §8 rule and record the chosen latent space in the PR description and in `RLConfig`-style pinning in Task 3 (`latent_mode`: `"action_tied"` | `"action_untied"` | `"action_video"`).
- [ ] **Step 8: Commit:** "Add the LingBot noise steerability probe."

---

### Task 3: Pin the DSRL recipe

**Files:**
- Create: `script/lingbot_dsrl_config.py`
- Test: `tests/test_lingbot_dsrl.py`

- [ ] **Step 1: Failing tests**

```python
from dataclasses import replace

import pytest

from script.lingbot_dsrl_config import DSRLConfig
from script.lingbot_rl_config import RLConfig


def test_dsrl_config_reuses_frozen_prior_pin():
    proto = DSRLConfig().validate().protocol()
    dice = RLConfig().protocol()
    for key in ("video_steps", "action_steps", "video_exec_step", "video_guidance", "action_guidance",
                "snr_shift", "action_snr_shift", "checkpoint_step", "source_run", "lerobot_revision",
                "model_revision", "max_policy_steps", "frame_chunk_size", "action_per_frame",
                "used_action_channels", "online_env_steps", "replay_capacity"):
        assert proto[key] == dice[key]
    assert proto["algorithm"] == "dsrl_sac"
    assert proto["latent_dim"] == 7 and proto["latent_mode"] == "action_tied"
    assert proto["action_magnitude"] == 1.0 and proto["utd"] == 20 and proto["critic_ensemble"] == 10
    assert proto["critic_reduction"] == "mean" and proto["backup_entropy"] is False
    assert proto["warmup_chunks"] == 500 and proto["gamma_env"] == 0.999


@pytest.mark.parametrize("changes", [
    {"video_steps": 10}, {"action_steps": 10}, {"video_exec_step": 3},
    {"online_env_steps": 50_000}, {"checkpoint_step": 800}, {"latent_mode": "bogus"},
])
def test_dsrl_config_rejects_drift(changes):
    with pytest.raises(ValueError):
        replace(DSRLConfig(), **changes).validate()


def test_dsrl_eval_schedule():
    cfg = DSRLConfig()
    assert cfg.train_eval_points() == list(range(0, 100_001, 10_000))
    assert cfg.full_eval_points() == [25_000, 50_000, 75_000, 100_000]
```

- [ ] **Step 2: Run, expect FAIL** (module missing).

- [ ] **Step 3: Implement**

```python
from dataclasses import asdict, dataclass

from script.lingbot_rl_config import RLConfig

LATENT_MODES = {"action_tied": 7, "action_untied": 112}


@dataclass(frozen=True)
class DSRLConfig:
    source_run: str = "libero30-sft"
    checkpoint_step: int = 600
    seed: int = 42
    wandb_project: str = "dice-lingbot-va-dsrl"
    wandb_entity: str | None = None
    video_steps: int = 20
    action_steps: int = 50
    video_exec_step: int = -1
    online_env_steps: int = 100_000
    latent_mode: str = "action_tied"
    train_eval_every: int = 10_000
    train_eval_episodes_per_task: int = 3

    def _prior(self):
        return RLConfig(
            source_run=self.source_run, checkpoint_step=self.checkpoint_step, seed=self.seed,
            video_steps=self.video_steps, action_steps=self.action_steps,
            video_exec_step=self.video_exec_step, online_env_steps=self.online_env_steps,
        )

    def validate(self):
        self._prior().validate()
        if self.latent_mode not in LATENT_MODES:
            raise ValueError("Unknown DSRL latent mode")
        if self.train_eval_every != 10_000 or self.train_eval_episodes_per_task != 3:
            raise ValueError("DSRL train-time eval is 3 rollouts/task every 10,000 env steps")
        return self

    @property
    def default_run_name(self):
        return f"{self.source_run}-dsrl-{self.latent_mode}-s{self.seed}"

    def train_eval_points(self):
        return list(range(0, self.online_env_steps + 1, self.train_eval_every))

    def full_eval_points(self):
        return [25_000, 50_000, 75_000, 100_000]

    def protocol(self):
        self.validate()
        prior = self._prior().protocol()
        shared = {key: prior[key] for key in (
            "suite", "task_ids", "max_policy_steps", "settling_steps", "control_freq", "control_mode",
            "hard_reset", "environment_batch_size", "camera_keys", "camera_orientation", "resolution",
            "frame_chunk_size", "action_per_frame", "video_steps", "action_steps", "video_exec_step",
            "video_guidance", "action_guidance", "snr_shift", "action_snr_shift", "attention_window",
            "action_normalization_epsilon", "sampler", "dtype", "attention_backend", "noisy_history",
            "text_encoder_device", "image_hflip", "camera_layout", "used_action_channels",
            "online_env_steps", "replay_capacity", "checkpoint_step", "source_run", "lerobot_revision",
            "model_repo", "model_revision", "libero_assets_repo", "libero_assets_revision",
            "comparison_episodes_per_task", "comparison_initial_state_offset", "comparison_seed",
        )}
        return {
            **shared,
            "version": 1,
            "algorithm": "dsrl_sac",
            "latent_mode": self.latent_mode,
            "latent_dim": LATENT_MODES[self.latent_mode],
            "action_magnitude": 1.0,
            "warmup_chunks": 500,
            "warmup_noise": "standard_normal_unclipped",
            "mlp_hidden": [1024, 1024, 1024],
            "layer_norm": True,
            "critic_ensemble": 10,
            "critic_reduction": "mean",
            "backup_entropy": False,
            "actor_lr": 1e-4,
            "critic_lr": 3e-4,
            "temperature_lr": 3e-4,
            "init_temperature": 1.0,
            "target_entropy": -LATENT_MODES[self.latent_mode] / 2,
            "log_std_bounds": [-20.0, 2.0],
            "tau": 0.005,
            "gamma_env": 0.999,
            "chunk_discount": "gamma_env ** n_env_actions",
            "utd": 20,
            "batch_size": 256,
            "reward": "success_on_first_success_chunk",
            "critic_state": "pooled_video_text_3072",
            "eval_policy": "deterministic_tanh_mean",
            "train_eval_every": self.train_eval_every,
            "train_eval_episodes_per_task": self.train_eval_episodes_per_task,
            "full_eval_points": self.full_eval_points(),
        }

    def to_dict(self):
        return {**asdict(self), "protocol": self.protocol()}


def main():
    import json
    print(json.dumps(DSRLConfig().validate().to_dict(), indent=2))


if __name__ == "__main__":
    main()
```

The `"comparison_*"` keys must exist in `RLConfig.protocol()`. They do (`comparison_episodes_per_task`, `comparison_initial_state_offset`, `comparison_seed`).

- [ ] **Step 4: Run tests, expect PASS.** Also run `.cache/eval-venv/bin/python -m script.lingbot_dsrl_config`.
- [ ] **Step 5: Commit:** "Pin the LingBot DSRL-SAC recipe."

---

### Task 4: Map the latent action onto LingBot action noise

**Files:**
- Create: `script/lingbot_dsrl_model.py` (noise section)
- Test: `tests/test_lingbot_dsrl.py`

Model-layout action noise is `(B, 30, 4, 4, 1)` = `(batch, channel, frame, action_per_frame, 1)`. LeRobot zeroes channels 7–29 of the noisy action input on every Euler step, so only channels 0–6 matter.

- [ ] **Step 1: Failing tests**

```python
import torch

from script.lingbot_dsrl_model import LATENT_DIM_TIED, noise_from_latent


def test_tied_latent_fills_used_channels_on_every_token():
    w = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]])
    noise = noise_from_latent(w, "action_tied", generator=torch.Generator().manual_seed(0))
    assert noise.shape == (1, 30, 4, 4, 1)
    for c in range(7):
        assert torch.allclose(noise[0, c], torch.full((4, 4, 1), float(w[0, c])))
    assert noise[0, 7:].abs().sum() > 0


def test_untied_latent_orders_tokens_like_model_to_mlp():
    w = torch.arange(112, dtype=torch.float32).reshape(1, 112)
    noise = noise_from_latent(w, "action_untied", generator=torch.Generator().manual_seed(0))
    mlp = noise.squeeze(-1).permute(0, 2, 3, 1).reshape(1, 16, 30)
    assert torch.equal(mlp[0, :, :7].reshape(-1), w[0])


def test_unused_channels_are_standard_normal_and_seeded():
    w = torch.zeros(2048, LATENT_DIM_TIED)
    a = noise_from_latent(w, "action_tied", generator=torch.Generator().manual_seed(1))
    b = noise_from_latent(w, "action_tied", generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    tail = a[:, 7:].reshape(-1)
    assert abs(float(tail.mean())) < 0.02 and abs(float(tail.std()) - 1.0) < 0.02
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement**

```python
import torch

from script.lingbot_rl_model import ACTION_DIM, HORIZON, USED_DOF

LATENT_DIM_TIED = USED_DOF
LATENT_DIM_UNTIED = USED_DOF * HORIZON
FRAMES = 4
ACTIONS_PER_FRAME = 4


def latent_dim(mode):
    return {"action_tied": LATENT_DIM_TIED, "action_untied": LATENT_DIM_UNTIED}[mode]


def noise_from_latent(w, mode, generator=None):
    w = torch.as_tensor(w, dtype=torch.float32)
    batch = w.shape[0]
    noise = torch.randn(batch, ACTION_DIM, FRAMES, ACTIONS_PER_FRAME, 1, generator=generator)
    if mode == "action_tied":
        used = w.reshape(batch, USED_DOF, 1, 1, 1).expand(batch, USED_DOF, FRAMES, ACTIONS_PER_FRAME, 1)
    elif mode == "action_untied":
        used = w.reshape(batch, FRAMES, ACTIONS_PER_FRAME, USED_DOF).permute(0, 3, 1, 2).unsqueeze(-1)
    else:
        raise ValueError("Unknown DSRL latent mode")
    noise[:, :USED_DOF] = used
    return noise
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit:** "Map DSRL latent actions onto LingBot action noise."

---

### Task 5: Noise actor, Q^W ensemble, temperature, and SAC updates

**Files:**
- Modify: `script/lingbot_dsrl_model.py`
- Test: `tests/test_lingbot_dsrl.py`

Matches `nakamotoo/dsrl_pi0` `pixel_sac`: the target has no entropy backup, `mean` critic reduction, the temperature loss is `α·(entropy − target)`, and log-std is clipped to [−20, 2].

- [ ] **Step 1: Failing tests**

```python
import torch

from script.lingbot_dsrl_model import DSRLModel
from script.lingbot_rl_model import STATE_DIM


def _batch(n=32, d=7):
    return {
        "s": torch.randn(n, STATE_DIM), "w": torch.rand(n, d) * 2 - 1,
        "reward": torch.randint(0, 2, (n, 1)).float(), "done": torch.randint(0, 2, (n, 1)).float(),
        "s_next": torch.randn(n, STATE_DIM), "discount": torch.full((n, 1), 0.999 ** 16),
    }


def test_actor_samples_are_bounded_and_log_probs_finite():
    torch.manual_seed(0)
    model = DSRLModel(latent_dim=7, device="cpu")
    w, logp = model.actor.sample(torch.randn(64, STATE_DIM))
    assert w.shape == (64, 7) and float(w.abs().max()) <= 1.0
    assert torch.isfinite(logp).all() and logp.shape == (64,)
    w_det, _ = model.actor.sample(torch.randn(3, STATE_DIM), deterministic=True)
    assert float(w_det.abs().max()) <= 1.0


def test_critic_target_uses_mean_reduction_without_entropy_backup():
    torch.manual_seed(1)
    model = DSRLModel(latent_dim=7, device="cpu")
    model.log_alpha.data.fill_(5.0)
    batch = _batch()
    torch.manual_seed(2)
    target = model.critic_target(batch)
    torch.manual_seed(2)
    next_w, _ = model.actor.sample(batch["s_next"])
    expected = batch["reward"] + batch["discount"] * (1 - batch["done"]) * model.target_critic(batch["s_next"], next_w).mean(0)
    torch.testing.assert_close(target, expected)


def test_one_update_changes_actor_critic_and_temperature():
    torch.manual_seed(3)
    model = DSRLModel(latent_dim=7, device="cpu")
    before = [p.detach().clone() for p in (*model.actor.parameters(), *model.critic.parameters(), model.log_alpha)]
    info = model.update(_batch())
    after = [p.detach() for p in (*model.actor.parameters(), *model.critic.parameters(), model.log_alpha)]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    for key in ("critic_loss", "actor_loss", "temperature", "entropy", "q_mean", "q_std_ensemble", "w_abs_mean"):
        assert key in info and torch.isfinite(torch.tensor(info[key]))


def test_resume_state_dict_roundtrip():
    model = DSRLModel(latent_dim=7, device="cpu")
    model.update(_batch())
    clone = DSRLModel(latent_dim=7, device="cpu")
    clone.load_resume_state_dict(model.resume_state_dict())
    s = torch.randn(4, STATE_DIM)
    torch.testing.assert_close(clone.actor.sample(s, deterministic=True)[0], model.actor.sample(s, deterministic=True)[0])
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (append to `lingbot_dsrl_model.py`)

```python
import copy

import torch.nn as nn
import torch.nn.functional as F

from script.lingbot_rl_model import STATE_DIM, _mlp, mlp_float

ACTION_MAGNITUDE = 1.0
ENSEMBLE = 10
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
TEMPERATURE_LR = 3e-4
TAU = 0.005
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class NoiseActor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = _mlp(STATE_DIM, 2 * latent_dim)

    def forward(self, state):
        mean, log_std = self.net(mlp_float(state)).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, state, deterministic=False):
        mean, log_std = self(state)
        if deterministic:
            return ACTION_MAGNITUDE * torch.tanh(mean), None
        std = log_std.exp()
        pre = mean + std * torch.randn_like(mean)
        squashed = torch.tanh(pre)
        log_prob = torch.distributions.Normal(mean, std).log_prob(pre).sum(-1)
        log_prob = log_prob - torch.log(ACTION_MAGNITUDE * (1 - squashed.pow(2)) + 1e-6).sum(-1)
        return ACTION_MAGNITUDE * squashed, log_prob


class NoiseCritic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(STATE_DIM + latent_dim, 1) for _ in range(ENSEMBLE)])

    def forward(self, state, w):
        x = torch.cat([mlp_float(state), mlp_float(w)], dim=-1)
        return torch.stack([head(x) for head in self.heads], dim=0)


class DSRLModel:
    def __init__(self, latent_dim, device="cpu"):
        self.device = device
        self.latent_dim = latent_dim
        self.target_entropy = -latent_dim / 2
        self.actor = NoiseActor(latent_dim).to(device)
        self.critic = NoiseCritic(latent_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=TEMPERATURE_LR)

    def critic_target(self, batch):
        with torch.no_grad():
            next_w, _ = self.actor.sample(batch["s_next"])
            next_q = self.target_critic(batch["s_next"], next_w).mean(0)
            return batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * next_q

    def update(self, batch):
        target = self.critic_target(batch)
        qs = self.critic(batch["s"], batch["w"])
        critic_loss = ((qs - target.unsqueeze(0)) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        w, log_prob = self.actor.sample(batch["s"])
        q_pi = self.critic(batch["s"], w).mean(0).squeeze(-1)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -log_prob.detach().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_parameter.data.mul_(1.0 - TAU).add_(parameter.data, alpha=TAU)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "temperature": float(self.log_alpha.exp().detach()),
            "entropy": float(entropy),
            "q_mean": float(qs.detach().mean()),
            "q_std_ensemble": float(qs.detach().std(0).mean()),
            "target_q_mean": float(target.mean()),
            "w_abs_mean": float(w.detach().abs().mean()),
        }

    def inference_state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(), "log_alpha": self.log_alpha.detach().cpu()}

    def load_inference_state_dict(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))

    def resume_state_dict(self):
        return {**self.inference_state_dict(), "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(), "alpha_opt": self.alpha_opt.state_dict()}

    def load_resume_state_dict(self, payload):
        self.load_inference_state_dict(payload)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.alpha_opt.load_state_dict(payload["alpha_opt"])
```

`init_temperature = 1.0` means `log_alpha = 0`, which matches `zeros`.

- [ ] **Step 4: Run, expect PASS.** If the "changes" test is flaky because the actor's last layer barely moves, assert on the critic and `log_alpha` separately. Do not loosen the finite checks.
- [ ] **Step 5: Commit:** "Add the DSRL-SAC noise actor, latent critic ensemble, and updates."

---

### Task 6: Chunk replay for the latent-action MDP

**Files:**
- Create: `script/lingbot_dsrl_buffer.py`
- Test: `tests/test_lingbot_dsrl.py`

A transition is completed when the next chunk's `s` exists, or at `done`. `s_next` is the next chunk's pooled state. Terminal rows use their own `s` (masked by `done`). `discount = 0.999 ** n_env_actions`.

- [ ] **Step 1: Failing tests**
  - `test_transition_completes_on_next_state`: `begin(s0, w0, task_id=0, is_warmup=True)` then `end(reward=0, done=0, n_env=12)`, then `begin(s1, w1, 0, False)` then `end(1, 1, 16)`. After the second `end` the buffer has 2 rows. Row 0 has `s_next == s1` and `discount == 0.999**12`; row 1 has `done == 1` and `discount == 0.999**16`.
  - `test_incomplete_transition_is_not_sampled`: after `begin`/`end(0, 0)` with no successor, `len(buffer) == 0` and `sample` raises.
  - `test_capacity_ring_and_sample_shapes`: capacity 5, insert 8 completed transitions. `len == 5`, the oldest are dropped, and `sample(4)` returns tensors with shapes `s (4, 3072)`, `w (4, d)`, `reward/done/discount (4, 1)`.
  - `test_state_dict_roundtrip_including_pending`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `LatentReplay(capacity, latent_dim, device)`: preallocated numpy arrays (`s` float32 `(C, 3072)` ≈ 1.2 GB at 100k, acceptable on host), ring index, `_pending` dict. Methods:
  - `begin(s, w, task_id, is_warmup)`: if a non-terminal row is waiting, complete it with `s_next = s`;
  - `end(reward, done, n_env)`: sets reward, done, and `discount = 0.999 ** n_env`, and completes the row immediately when `done`;
  - `abandon_pending()`, `sample(batch)`, `state_dict()`, `load_state_dict()`, `__len__`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit:** "Add the DSRL chunk replay buffer."

---

### Task 7: Steered policy for evaluation

**Files:**
- Create: `script/lingbot_dsrl_policy.py`
- Test: `tests/test_lingbot_dsrl.py`

**Interfaces produced:**
- `class SteeredLingBotPolicy(ResidualLingBotPolicy)` with attributes `dsrl_model=None`, `latent_mode="action_tied"`, and `eval_mode in {"deterministic", "best_of_4"}`.
- `predict_action_chunk(batch)` does `ctx = pool_chunk_state(batch)`, then:
  - if `dsrl_model is None`: `decode_with_noise(ctx, k=1)` (prior);
  - `deterministic`: `w = actor.sample(s, deterministic=True)` → `noise_from_latent` → `decode_with_noise(ctx, k=1, action_noise=noise)`;
  - `best_of_4`: sample 4 `w`, decode with `k=4`, pick `argmax mean-ensemble Q^W(s, w_k)`.

  It then commits the chosen `a_base` via `commit_executed` and returns `slice_env_actions` (same as `_apply_residual_choice` with no residual).
- `load_steered_policy(checkpoint, model_path, architecture)`, the same body as `load_residual_policy` but constructing `SteeredLingBotPolicy`. Factor out the shared validation into a private helper in `lingbot_rl_policy.py` only if it stays behaviour-identical, otherwise duplicate the ~15-line config check.

- [ ] **Step 1: Failing tests** use `SteeredLingBotPolicy.__new__`, monkeypatching `pool_chunk_state`/`decode_with_noise`/`commit_executed`:
  1. With no model, `decode_with_noise` is called with `action_noise=None, k=1`.
  2. Deterministic mode passes a noise whose channels 0–6 equal `tanh(mean)` tiled.
  3. Best-of-4 decodes with `k=4` and commits the index of the highest-Q `w`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit:** "Add the steered LingBot policy for DSRL evaluation."

---

### Task 8: Training loop, train-time evals, resume, and CLI

**Files:**
- Create: `script/lingbot_dsrl_train.py`
- Test: `tests/test_lingbot_dsrl.py`

Mirror `script/lingbot_rl_train.py::train` and import its helpers instead of copying: `make_env`, `env_success`, `_read_prepared`, `_write_json`, `_atomic_torch`, `capture_rng`, `restore_rng`, `train_eval_schedule`, `due_train_evals`, `_host_array`.

**Loop (per episode, uniform task, `env.reset(seed=config.seed + env_steps)`):**

```text
while env_steps < budget and episode_length < 520:
    ctx = policy.pool_chunk_state(batch)
    s = ctx["s"].float()
    if chunks < warmup_chunks: w = randn(1, d); noise = noise_from_latent(w, mode)
    else: w, _ = model.actor.sample(s); noise = noise_from_latent(w, mode)
    decoded = policy.decode_with_noise(ctx, k=1, action_noise=noise)
    a = decoded["a_base"][:1]
    policy.commit_executed(a.cpu(), first_chunk=decoded["first_chunk"])
    replay.begin(s, w, task_id, is_warmup=chunks < warmup_chunks)
    step env over slice_env_actions(a) exactly as DICE (reward, done, observe_env_step)
    replay.end(reward, done, n_env=executed_n)
    chunks += 1
    if chunks >= warmup_chunks and len(replay) >= batch: 20 × model.update(replay.sample(256))
    log every chunk: env_steps, chunks, is_warmup, episode_*, update info, timers
```

Warm-up `w` is stored unclipped. SAC's tanh log-prob never evaluates stored `w`, so no clipping is needed for the critic. Log `warmup_w_abs_max` so the |w|>1 fraction is visible.

**Timers (W&B):** `time/pool`, `time/decode`, `time/env`, `time/update`, `time/eval`, cumulative `gpu_hours`.

**Episode plans:** add `plan_entries(seed, task_id, init_state_ids)`, which reproduces `episode_plan`'s identity hash `["libero_10", seed, task_id, init_state_id]` for arbitrary IDs. Train-time evals use init states **{0, 48, 49}** and final evals use **1–20** (or **1–40** when extended). The two sets are disjoint, so train-time curves never touch final-eval states. Test: `plan_entries(42, t, range(1, 21))` equals `episode_plan(EvalConfig(stage="eval"), t, 50)` for every task.

**Train-time evals:** at each `DSRLConfig.train_eval_points()` crossed at an episode boundary, run 3 episodes/task on init states {0, 48, 49} in `deterministic` mode. Save `train_eval/step_XXXXXX/{task_*.json, dsrl.pt}` and log macro + per-task. At `full_eval_points` also save `checkpoints/step_XXXXXX/dsrl.pt` for the separate 200-episode job.

**Sharpening diagnostics (parity with DICE's `delta_v`/`delta_h`):** at each train-time eval point, on task 0's reset state, decode 8 prior samples (`w ~ N(0,I)`) and 8 steered samples (`w ~ π`) with one shared video noise. Log `delta_v = mean Q^W(s, w_π) − mean Q^W(s, w_prior)` and `delta_h = H(a_prior) − H(a_π)` using `histogram_entropy`. Positive values mean sharpened, the same sign convention as DICE.

**Resume:** `resume/latest.pt` holds `model.resume_state_dict()`, `replay.state_dict()`, RNG, `env_steps`, `chunks`, `evaluated`, `wandb_id`, and `recipe = DSRLConfig.protocol()`. Refuse on recipe mismatch. Save at every episode end (atomic). Never store transformer weights.

**CLI:** `python -m script.lingbot_dsrl_train {train,eval} --prepared-path P --output-dir D [--run-name N] [--seed S] [--latent-mode M] [--resume] [--max-env-steps K] [--weights PATH] [--eval-mode deterministic|best_of_4] [--episodes-per-task 20|40]`. `eval` uses `plan_entries(42, task, 1..N)` (N=20 matches the 69% protocol exactly). It then writes `eval/episodes/task_XX/episode_XXX.json`, `eval/summary.json`, and `eval/settings.json`, resuming per episode like `lingbot_rl_train.evaluate`.

- [ ] **Step 1: Failing tests** (CPU, fully mocked):
  - `test_warmup_uses_standard_normal_then_actor` checks that the first `warmup_chunks` decode calls receive noise whose used channels are not bounded by 1 in aggregate, and later calls are bounded by 1. Use a fake policy (records `action_noise`, returns zeros `a_base`), a fake env (`is_success` False, fixed obs), and a warm-up of 3 chunks via monkeypatched constant.
  - `test_updates_start_after_warmup_with_utd_20` monkeypatches `DSRLModel.update` to count calls. After warm-up + 2 chunks it expects 40 calls.
  - `test_budget_counts_warmup_env_steps` stops at `max_env_steps` exactly and checks `summary["env_steps"] == max_env_steps`.
  - `test_resume_refuses_recipe_mismatch`.
  - `test_resume_payload_excludes_transformer`.
  - `test_plan_entries_match_episode_plan_and_train_ids_are_disjoint`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep W&B behind the same `wandb.init(**kwargs)` pattern, and in tests monkeypatch `wandb` with a stub module (see how `tests/test_lingbot_rl.py` handles it).
- [ ] **Step 4: Run** `pytest -q tests/test_lingbot_dsrl.py tests/test_lingbot_rl.py`. Expected: all pass.
- [ ] **Step 5: Commit:** "Add the LingBot DSRL-SAC training loop with warm-up, timers, resume, and evaluation CLI."

---

### Task 9: Slurm launchers with resume chaining

**Files:**
- Create: `hpc/train.sbatch`, `hpc/eval.sbatch`, `hpc/submit_chain.sh`

- [ ] **Step 1: `hpc/train.sbatch`** (`--gres=gpu:1 --cpus-per-task=16 --mem=96G --time=${SLURM_MAX_HOURS}:00:00 --signal=B:USR1@600`):
  - source `hpc/env.sh`;
  - take a run lock via `mkdir "$RUNS_ROOT/$RUN/LOCK.d"` (atomic) and write `$SLURM_JOB_ID` into it. If the directory already exists, print the recorded job ID and its `squeue` state, then exit non-zero. Never remove a foreign lock automatically: an operator confirms that job is gone and deletes `LOCK.d`;
  - `trap 'touch $RUNS_ROOT/$RUN/STOP' USR1`, where the trainer checks `STOP` at episode boundaries, saves resume, and exits 0 so the chain continues;
  - run `$EVAL_PY -m script.lingbot_dsrl_train train --prepared-path "$(jq -r .prepared_path $CACHE_ROOT/prepared-pointer.json)" --output-dir "$RUNS_ROOT/$RUN" --run-name "$RUN" --seed "$SEED" $( [ -f "$RUNS_ROOT/$RUN/resume/latest.pt" ] && echo --resume )`;
  - release the lock on exit (the trap removes `LOCK.d` only if `jobid` matches).
- [ ] **Step 2: `hpc/submit_chain.sh RUN SEED N`** submits N train jobs chained with `--dependency=afterany:<prev>`. Each resumes and exits immediately if `summary.json` exists.
- [ ] **Step 3: `hpc/eval.sbatch RUN STEP MODE EPISODES`** runs `lingbot_dsrl_train eval --weights $RUNS_ROOT/$RUN/checkpoints/step_$STEP/dsrl.pt --output-dir $RUNS_ROOT/$RUN/eval-$STEP-$MODE-$EPISODES` with `--resume`.
- [ ] **Step 4: Add the trainer `STOP`-file check** (test: `STOP` present → loop exits after the current episode with resume saved), then commit.
- [ ] **Step 5: Commit:** "Add Slurm launchers with resumable chained training and detached evaluation."

---

### Task 10: Smoke, runs, evals, comparison

**Files:**
- Create: `script/lingbot_compare_report.py`
- Test: `tests/test_lingbot_dsrl.py`

- [ ] **Step 1: Cluster smoke.** Submit `train.sbatch` with `--max-env-steps 2000` and a warm-up of 50 chunks. Pass the warm-up through `--smoke-warmup-chunks`, which the trainer accepts only together with `--max-env-steps`, records in `settings.json`, and folds into the resume fingerprint. That way a smoke run can never resume into a full run. Pass criteria:
  - W&B shows updates;
  - `a_base` and `w` are finite;
  - `time/decode` is within 1.3× the probe's k=1 number;
  - resume works: `scancel` mid-run, resubmit, and `env_steps` continues;
  - no OOM.
- [ ] **Step 2: Main runs.** `hpc/submit_chain.sh libero30-dsrl-tied-s42 42 N` and `… s43 43 N`. N comes from the probe timing: `ceil(estimated_hours / (SLURM_MAX_HOURS − 0.25))`.
- [ ] **Step 3: Detached evals.** For each seed at 25k/50k/75k/100k submit `eval.sbatch … deterministic 20`. Also submit `best_of_4 20` at 50k and 100k, which matches the DICE v2 eval selection. At 100k, extend the better mode to 40 episodes/task (init states 1–40, 400 episodes). If compute allows, also extend SFT and DICE v2 @100k to 40 episodes/task on the same init states, using the steered policy with no model for SFT and the DICE `evaluate` path with a 40-entry plan.
- [ ] **Step 4: Failing test for the report math:** `fisher_two_sided(13, 6, 20)` ≈ 0.056 (±1e-3; task 4 SFT vs DICE@50k), `two_proportion_z(0.69, 0.63, 200)` ≈ −1.27 (±0.01), and `wilson_interval(138, 200)` contains 0.69 with half-width ≈ 0.064.
- [ ] **Step 5: Implement `lingbot_compare_report.py`** (stdlib only). Inputs: the SFT `summary.json`, the DICE CSV (`dice-rl-eval-results.csv` format), DSRL `eval/summary.json` files, and W&B-exported train-time curves (CSV). Outputs `result/lingbot-compare/`:
  - `tasks.csv`: per-task successes for SFT, DICE@50k/100k, and DSRL@25/50/75/100k per seed and mode, with Fisher p against SFT;
  - `macro.csv`: macro rates, 95% Wilson CIs, z vs SFT, and z DSRL vs DICE at equal steps;
  - `curves.csv`: train-time success vs env steps plus AUC;
  - `compute.csv`: GPU-hours split by phase.
- [ ] **Step 6: Run the report**, then commit the script and the (small) CSVs: "Add the SFT/DICE-RL/DSRL comparison report."

---

### Task 11: Docs

- [ ] Add a "LingBot DSRL" section to `AGENTS.md`: narrow checks (`pytest -q tests/test_lingbot_dsrl.py`, `python -m script.lingbot_dsrl_config`), HPC boundaries (no Modal, no edits to DICE semantics), and the lock/STOP discipline.
- [ ] Add a README section with the HPC commands from Tasks 0, 9, and 10.
- [ ] Commit: "Document the LingBot DSRL experiment and HPC launch."

---

### Task 12 (conditional, after Task 10 reports): Phase 3

Only start this after the Phase 1 numbers exist. Each is a separate branch/plan.

- **A1 untied** (`latent_mode="action_untied"`): config + one run. The code already supports it.
- **A3 video steering**: requires a probe result first. Add `latent_mode="action_video"` with a 48-d tied video offset (spec §9) and `video_noise` built in `noise_from_latent`.
- **DSRL-NA with demos**:
  - add `Q^A(s, a)` trained by TD on online chunks + the `featurize_experts` rows (import, do not modify);
  - distil `Q^W` on stored `(s, w_k, a_k)` with K=4 per chunk from a batched decode;
  - run 10 `Q^W` steps per update.
- **DICE v3 fix**: mask `z[:, :, 7:]` before `ResidualActor` and raise stored K to 16. This belongs in the DICE code path under its own recipe version, not in this plan's files.

---

## Execution order and gates

```text
Task 0 (env/assets) ─► Task 1 (split) ─► Task 2 (probe, GPU) ──gate: latent mode──►
Tasks 3–7 (CPU TDD, parallelizable after 3) ─► Task 8 (loop) ─► Task 9 (Slurm) ─►
Task 10 smoke ─gate: pass criteria─► main runs ─► evals ─► report ─► Task 11 ─► (Task 12)
```

## Open questions for the operator (answers change Tasks 0, 9, 10)

1. Cluster: GPU model/VRAM, max wall time, Slurm or PBS, Apptainer or not, and internet on compute nodes?
2. How many GPU-hours are available? This decides 2 seeds vs 1, the 50-episode final evals, and whether DICE v2 is rerun with a second seed.
3. Is the SFT step-600 checkpoint reachable (Modal volume access), or does a collaborator need to send it?
4. W&B: same entity as `dice-lingbot-va-rl`? This plan uses project `dice-lingbot-va-dsrl`.
