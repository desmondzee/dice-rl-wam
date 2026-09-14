# LingBot-VA DICE-RL Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Modal-launchable DICE-RL trainer that freezes LingBot-VA `libero30-sft` step 000600, trains residual + critic MLPs on LIBERO-10, logs to W&B, and can be evaluated with the same 20/50 full-video protocol that scored 69%.

**Architecture:** New `script/lingbot_rl_*.py` on the **eval** LeRobot environment. Do not instantiate Hydra `DistillResidualRLModel` or edit `.cache/lerobot`. Copy only residual/critic **loss math**. Collection: one `LiberoEnv`, shared video Euler, K=4 batched action Euler, execute `argmax Q`. Train-time MLP updates use stored `(s, z, a_base)` — never call the 5B model inside the optimizer loop.

**Tech Stack:** Python 3.12 eval venv (`.cache/eval-venv`), LeRobot `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`, PyTorch, Modal 1.1.4, W&B, pytest. GPU work is Modal-only.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-14-lingbot-dice-rl-design.md`
- Sampler is immutable: `video_steps=20`, `video_exec_step=-1`, `action_steps=50`, CFG 5.0/1.0, SNR 5.0/0.05, `source_run=libero30-sft`, `checkpoint_step=600`
- Do not edit `.cache/lerobot`, `.cache/lingbot-va`, `script/lingbot_eval.py` protocol fields, `agent/finetune/`, `cfg/robomimic/`, or root `pyproject.toml`
- Do not import or instantiate `DistillResidualRLModel`
- Transformer/VAE/UMT5 stay frozen; never write them into `dice-lingbot-rl-runs`
- Local download is residual+critic inference weights + summaries/eval only — not `resume/latest.pt`, replay, Adam, expert cache, or the 5B model
- Local tests: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py` — no GPU, no `modal run` `.remote()`, no network downloads
- Commit at task boundaries when the user asked to save progress. Never add `Co-authored-by`
- Work in the current `dice-rl-wam` checkout (user is reviewing files here)
- Non-goals (do not implement): DSRL; Hydra `agent/finetune/` / `cfg/robomimic/`; vectorized envs; KV-cache critic; s=0.6 / 10-step action sampler; re-running the 69% SFT eval; 100 rollouts/task in the train loop; β/n/M/BC-filter sweeps; editing SFT weights or the 30-demo subset; `script/lingbot_eval.py` protocol fields

---

## Spec lock (every immutable from the design spec)

Copy these into code. A task that contradicts this table is wrong.

| Spec | Value | Plan task |
| --- | --- | --- |
| Suite / env | LIBERO-10, one `LiberoEnv`, `n_envs=1`, `batch_size=1` | 7, 8 |
| Checkpoint | `/sft/libero30-sft/checkpoints/step_000600` on `dice-lingbot-sft-runs` | 5, 7, 8 |
| Frozen weights | Native transformer safetensors; VAE + UMT5 from `robbyant/lingbot-va-base`; `requires_grad_(False)`, `.eval()`; UMT5 on **CPU**; never write transformer/VAE/UMT5 into `dice-lingbot-rl-runs` | 5, 7, 8 |
| Cameras | 128×128 agentview then wrist, width-concat latents, **vertical-only flip**, `image_hflip=False` (reuse `observation_batch`) | 5, 7 |
| Actions | 7 DoF in channels 0–6 of 30; unused **zero-masked**; denorm `q01`/`q99` ε=`1e-6` via `decode_action` | 2, 4, 7 |
| Sampler | `released_lingbot_libero_defaults`: video **20**, `video_exec_step=-1` (no cutoff; last video forward writes KV only, no `scheduler.step`), action **50**, CFG **5.0 / 1.0**, SNR **5.0 / 0.05** | 1, 5 |
| Chunk | `K_AR=4` latent frames × 4 actions = 16 tokens; first env chunk **12** (drop frame 0); later **16**; residual always on full 16×30; frame-0 stays the **zero action condition** | 4, 5 |
| Noisy-history | **Off** at inference | 1, 5 |
| Attention / dtype | SDPA `attn_mode=torch`, bf16, `attn_window=30` | 1, 5 |
| Residual Eq. (2) | `a = a_base + s_θ(s, z)` in **normalized** space; condition on **z** not `a_base`; then mask 7–29; then denorm 0–6 | 2, 5 |
| History | Write **residual-edited** chunk to `_executed_actions` before `_compute_kv_cache`; never `a_base` | 5 |
| Collection K | **4** action noises, **one** video noise/decode, batched action Euler; expand **conditional** KV only (action batch 4, not 8); `argmax_k Q(s, a_k)`; OOM → **raise**, never drop K or ensemble size | 5, 7 |
| Train-time π_pre | Updates use stored `(s, z, a_base)` of the chosen candidate; **no** 5B call inside optimizer | 2, 7 |
| Critic `s` | Action-free video-stream on **real** VAE obs (not imagined video, not action tokens, not KV); pre-`proj_out` 3072-d; text after `condition_embedder.text_embedder`; concat seq + **mean-pool**; compute once at collection / expert featurize; store vector only | 4, 5, 6 |
| MLPs | Residual `[1024]³` GELU LN, in `(s, z_flat)` out 16×30; critic **10** heads, in `(s, a_flat)`, **min**; target Polyak τ=`0.01`; Adam `1e-4` both | 2 |
| Losses | β=`100`; ε=`-0.5`; BC filter **on**; n-step **3 chunks**, γ=`0.99`/chunk; UTD **10** critic + **1** actor per online chunk; RLPD 0.5→0.1 linear over 100k env steps; **expert actor Q-max off** (`disable_q_loss_for_expert_data`); critic TD **does** use expert rows; batch **256** or `min(256, len(buffer))` | 2, 3, 7 |
| Replay | One AR chunk per row; fields `s,z,a_base,a,reward,done,s_next,task_id,n_env_actions,is_expert` (+ `n_steps`,`a_next` for backup); sparse `is_success` → 1 on first success chunk else 0; capacity **100k**; CPU host storage | 3 |
| Experts | Same **300** SFT demos from checkpoint manifest; GPU featurize at **train start** (CPU prepare is assets only); cache on RL volume; pad 7-d → 16×30 with unused 0; checkpoint normalizer; published latents only if cameras align, else RGB from LeRobot dataset | 6, 8 |
| Budget | **100,000 env action steps** (not chunks); task_id ~ U{0..9}; cap **520** policy steps; `LiberoEnv` kwargs match eval | 7 |
| Resume | `resume/latest.pt` = MLPs + Adam + replay + RNG + `env_steps` + **recipe fingerprint**; refuse resume if sampler/β/n/step 600 drifted; never store transformer | 7 |
| Inference dump | `residual.pt` = actor + critic (+ target) only | 7, 8 |
| Train-time eval | env steps 0 / 25k / 50k / 75k / 100k; **1**/task; init-state **0**; recorded seeds via `episode_plan(EvalConfig(stage="smoke"))`; per-task success logged | 7 |
| Comparison eval | `--stage eval`; **20**/task; init-states **1–20**; seed **42**; identical to `EvalConfig(stage="eval")`; **do not** re-run the SFT 69% job; 100/task is out of spec | 7, 8 |
| W&B | project `dice-lingbot-va-rl`; `--wandb-entity` optional; log every optimizer step; scalars `env_steps, chunks, actor_loss, critic_loss, residual_rms, q_mean, q_min, bc_filter_rate, expert_ratio, episode_return, episode_success, episode_length`; **ΔH/ΔV** on demo anchors at train-eval points only (same frozen video noise, several action z); ΔH = drop in **per-coordinate** histogram entropy; residual RMS is **not** ΔH; no secrets in logs/configs/artifacts | 7, 8 |
| Modal image | Debian 3.12; LeRobot `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`; extras `lingbot_va`+`libero`+`evaluation`; `--index-url https://pypi.org/simple` locked uv; `MUJOCO_GL=egl` | 8 |
| Volumes | cache RO, sft RO, `dice-lingbot-rl-runs` RW | 8 |
| Secrets | `dice-lingbot-hf` on CPU prepare only; `dice-lingbot-wandb` on GPU; GPU train sets `HF_HUB_OFFLINE=1` like eval | 8 |
| GPU job | 1× H100; timeout **12h**; retries **0**; `max_containers=1`; persistent lock `dice-lingbot-rl-run-locks`; **no** auto stale-lock takeover; **no** region pin | 8 |
| Download | after successful train/eval, client stays attached; `result/lingbot-rl/<run>/`; `--download-dir` override; no overwrite; **only** `residual.pt`, summaries, train-eval rows, and after eval the 20-rollout JSON/`report.html`; never resume/replay/Adam/expert cache/5B | 8 |
| CLI | `modal run -m script.lingbot_rl_modal --stage train --run-name libero30-dice-baseline` | 8 |
| Local tests | eval venv; listed in spec §8; no GPU; no `.remote()` | 1–8 |
| Cloud smoke | operator-launched: a few H100 chunks, W&B step, no OOM, finite `a_base`/residual, K stays 4 | notes |

---

## File map

| File | Responsibility |
| --- | --- |
| `script/lingbot_rl_config.py` | Frozen recipe dataclass, W&B project, protocol dict, name validation |
| `script/lingbot_rl_model.py` | Residual MLP, 10-critic ensemble, DICE-RL actor/critic losses, Polyak, save/load |
| `script/lingbot_rl_buffer.py` | Chunk replay, n-step returns, RLPD mix schedule |
| `script/lingbot_rl_policy.py` | Geometry helpers + `ResidualLingBotPolicy` (video freeze, K-batch action Euler, critic pool, residual apply, history writeback) |
| `script/lingbot_rl_data.py` | One-shot expert featurization of the 300 SFT demos |
| `script/lingbot_rl_train.py` | Single-env loop, W&B, resume, train-time eval, comparison eval |
| `script/lingbot_rl_modal.py` | Eval-image Modal: prepare / train / eval, inference-only download |
| `tests/test_lingbot_rl.py` | All local tests |
| `README.md`, `AGENTS.md` | Document RL commands (last task) |

---

## Shared types (locked)

```python
STATE_DIM = 3072          # 24 heads × 128, pre-proj_out
HORIZON = 16              # 4 latent frames × 4 actions
ACTION_DIM = 30           # LingBot action channels
USED_DOF = 7              # LIBERO DoF in channels 0–6
VIDEO_STEPS = 20
ACTION_STEPS = 50
VIDEO_EXEC_STEP = -1
K_CANDIDATES = 4
ENSEMBLE = 10
HIDDEN = (1024, 1024, 1024)
BETA = 100.0
EPSILON = -0.5
GAMMA = 0.99
N_STEP_CHUNKS = 3
UTD = 10
TAU = 0.01
ADAM_LR = 1e-4
BATCH = 256
ONLINE_ENV_STEPS = 100_000
REPLAY_CAPACITY = 100_000
RLPD_START = 0.5
RLPD_END = 0.1
MAX_EPISODE_STEPS = 520
WANDB_PROJECT = "dice-lingbot-va-rl"
```

Tensor layouts:

- Pooled critic state `s`: `(B, 3072)`
- MLP chunk `z` / `a_base` / `a`: `(B, 16, 30)`
- LingBot model chunk: `(B, 30, 4, 4, 1)`
- Env action: length-7 float32, denormed with `script.lingbot_eval.decode_action`

---

### Task 1: Pinned RL config

**Files:**
- Create: `script/lingbot_rl_config.py`
- Test: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `script.lingbot_eval_config.validate_name`, `TASK_IDS`, `CAMERAS`, `LEROBOT_REVISION`, `EvalConfig.protocol` field names
- Produces: `RLConfig` frozen dataclass with `validate()`, `protocol()`, `to_dict()`, `rlpd_expert_ratio(env_steps)`, `default_run_name`

- [x] **Step 1: Write the failing tests**

Create `tests/test_lingbot_rl.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_config_pins_released_libero_sampler_and_step_600 tests/test_lingbot_rl.py::test_config_rejects_sampler_and_recipe_drift tests/test_lingbot_rl.py::test_rlpd_ratio_anneals_over_env_steps -v`

Expected: FAIL with `ModuleNotFoundError: script.lingbot_rl_config`

- [x] **Step 3: Write minimal implementation**

`script/lingbot_rl_config.py`:

```python
from dataclasses import asdict, dataclass

from script.lingbot_eval_config import (
    CAMERAS, LEROBOT_REVISION, LIBERO_ASSETS_REPO, LIBERO_ASSETS_REVISION,
    TASK_IDS, validate_name,
)
from script.lingbot_sft_config import MODEL_REPO, MODEL_REVISION


@dataclass(frozen=True)
class RLConfig:
    source_run: str = "libero30-sft"
    checkpoint_step: int = 600
    seed: int = 42
    wandb_project: str = "dice-lingbot-va-rl"
    wandb_entity: str | None = None
    video_steps: int = 20
    action_steps: int = 50
    video_exec_step: int = -1
    k_candidates: int = 4
    online_env_steps: int = 100_000
    train_eval_every: int = 25_000
    train_eval_episodes_per_task: int = 1

    def validate(self):
        validate_name(self.source_run)
        if self.source_run != "libero30-sft":
            raise ValueError("RL is pinned to libero30-sft")
        if self.checkpoint_step != 600:
            raise ValueError("RL is pinned to checkpoint step 600")
        if self.video_steps != 20 or self.action_steps != 50 or self.video_exec_step != -1:
            raise ValueError("RL must use the released 20/50 full-video sampler")
        if self.k_candidates != 4:
            raise ValueError("Collection K is pinned to 4")
        if self.online_env_steps != 100_000:
            raise ValueError("Online budget is 100,000 env action steps")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if not self.wandb_project:
            raise ValueError("W&B project is required")
        return self

    @property
    def default_run_name(self):
        return f"{self.source_run}-dice-baseline"

    def rlpd_expert_ratio(self, env_steps):
        span = self.online_env_steps
        t = min(max(env_steps, 0), span) / span
        return 0.5 + (0.1 - 0.5) * t

    def protocol(self):
        self.validate()
        return {
            "version": 1,
            "suite": "libero_10",
            "task_ids": list(TASK_IDS),
            "max_policy_steps": 520,
            "camera_keys": list(CAMERAS),
            "camera_orientation": "vertical_flip_only_native_lingbot",
            "resolution": [128, 128],
            "frame_chunk_size": 4,
            "action_per_frame": 4,
            "settling_steps": 10,
            "control_freq": 20,
            "control_mode": "relative",
            "hard_reset": True,
            "environment_batch_size": 1,
            "video_steps": self.video_steps,
            "action_steps": self.action_steps,
            "video_exec_step": self.video_exec_step,
            "video_guidance": 5.0,
            "action_guidance": 1.0,
            "snr_shift": 5.0,
            "action_snr_shift": 0.05,
            "attention_window": 30,
            "action_normalization_epsilon": 1e-6,
            "sampler": "released_lingbot_libero_defaults",
            "dtype": "bfloat16",
            "attention_backend": "torch",
            "noisy_history": False,
            "text_encoder_device": "cpu",
            "image_hflip": False,
            "camera_layout": "width_concat",
            "used_action_channels": list(range(7)),
            "k_candidates": self.k_candidates,
            "online_env_steps": self.online_env_steps,
            "lerobot_revision": LEROBOT_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "libero_assets_repo": LIBERO_ASSETS_REPO,
            "libero_assets_revision": LIBERO_ASSETS_REVISION,
        }

    def to_dict(self):
        return {**asdict(self), "protocol": self.protocol()}
```

- [x] **Step 4: Run tests and make sure they pass**

Same pytest command. Expected: PASS

- [ ] **Step 5: Commit**

Commit this task's files. No `Co-authored-by`.

---

### Task 2: Residual MLP, critic ensemble, DICE-RL losses

**Files:**
- Create: `script/lingbot_rl_model.py`
- Modify: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: constants from Task 1 / shared types
- Produces:
  - `ResidualActor.forward(state, noise) -> (B, 16, 30)`
  - `CriticEnsemble.forward(state, action, return_all=False) -> (B, 1)` or list of `(B, 1)`
  - `mask_unused_dof(chunk) -> chunk` with channels 7–29 zero
  - `apply_residual(a_base, residual) -> a`
  - `DiceResidualModel.actor_loss(state, noise, a_base, is_expert) -> dict`
  - `DiceResidualModel.critic_loss(state, action, target_q, is_expert) -> dict`
  - `DiceResidualModel.n_step_target(reward, done, next_state, next_action, n_steps) -> (B, 1)`
  - `DiceResidualModel.polyak_update()`
  - `inference_state_dict()` / `load_inference_state_dict()` — residual+critic only
  - `resume_state_dict()` / `load_resume_state_dict()` — MLPs + Adam (no replay)

Actor update uses **stored** `(s, z, a_base)`, not new 5B samples. Expert rows skip **actor Q maximization** (`disable_q_loss_for_expert_data`). Critic TD **is applied to expert rows** (spec does not set `disable_td_loss_for_expert_data`). BC filter on: drop BC when `Q(s,a) > Q(s,a_base)` **and** TD residual `Q(s,a) - target_q < ε` if `target_q` is passed; if `target_q` is omitted, drop BC when `Q(s,a) > Q(s,a_base)` only if that would make the filter vacuous — instead pass `target_q` from the critic update. For the isolated actor-loss unit test, pass `target_q=None` and apply: `should_filter = (q_a > q_base) & ((q_a - q_base) < EPSILON)` is **wrong** (contradictory). Use:

```
q_td_error = q_a.detach() - target_q   # if target_q is not None
underestimated = q_td_error < EPSILON
better = (q_a.detach() > q_base)
should_filter = better & underestimated
bc_keep = 1 - should_filter
```

When `target_q` is None (unit test of unfiltered path), `bc_keep = 1`. The train loop always passes the n-step target.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lingbot_rl.py`:

```python
import torch

from script.lingbot_rl_model import (
    ACTION_DIM, EPSILON, HORIZON, STATE_DIM, USED_DOF,
    DiceResidualModel, apply_residual, mask_unused_dof,
)


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
    reward = torch.tensor([[0.0], [0.0], [1.0]])
    done = torch.tensor([[0.0], [0.0], [1.0]])
    next_state = torch.zeros(3, STATE_DIM)
    next_action = torch.zeros(3, HORIZON, ACTION_DIM)
    n_steps = torch.tensor([[3.0], [2.0], [1.0]])
    # For the last row, done=1 so backup is 0 regardless of Q.
    target = model.n_step_target(reward, done, next_state, next_action, n_steps)
    assert target.shape == (3, 1)
    torch.testing.assert_close(target[2], torch.tensor([1.0]))
    # First row: 0 + 0 + γ² * 1 if we were computing returns in the buffer;
    # n_step_target treats `reward` as already n-step summed. Pass the summed return.
    summed = torch.tensor([[0.99 ** 2], [0.99], [1.0]])
    target = model.n_step_target(summed, done, next_state, next_action, n_steps)
    torch.testing.assert_close(target[2], torch.tensor([1.0]))
    assert target[0].item() == pytest.approx(0.99 ** 2)


def test_actor_critic_one_step_update_on_random_tensors():
    torch.manual_seed(1)
    model = DiceResidualModel(device="cpu")
    state = torch.randn(8, STATE_DIM)
    noise = torch.randn(8, HORIZON, ACTION_DIM)
    a_base = torch.randn(8, HORIZON, ACTION_DIM)
    a_base = mask_unused_dof(a_base)
    is_expert = torch.zeros(8, 1)
    is_expert[:4] = 1
    reward = torch.zeros(8, 1)
    done = torch.zeros(8, 1)
    n_steps = torch.ones(8, 1) * 3
    next_state = torch.randn(8, STATE_DIM)
    next_action = mask_unused_dof(torch.randn(8, HORIZON, ACTION_DIM))
    target = model.n_step_target(reward, done, next_state, next_action, n_steps)
    before = {k: v.detach().clone() for k, v in model.actor.named_parameters()}
    critic_info = model.update_critic(state, apply_residual(a_base, model.actor(state, noise).detach()), target, is_expert)
    actor_info = model.update_actor(state, noise, a_base, is_expert, target)
    assert torch.isfinite(critic_info["critic_loss"])
    assert torch.isfinite(actor_info["actor_loss"])
    assert 0.0 <= actor_info["bc_filter_rate"] <= 1.0
    changed = any(not torch.equal(before[k], v) for k, v in model.actor.named_parameters())
    assert changed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py::test_residual_mask_zeros_unused_dof tests/test_lingbot_rl.py::test_n_step_chunk_return_sparse_terminal tests/test_lingbot_rl.py::test_actor_critic_one_step_update_on_random_tensors -v`

Expected: FAIL `ModuleNotFoundError: script.lingbot_rl_model`

- [ ] **Step 3: Write implementation**

`script/lingbot_rl_model.py` (complete):

```python
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 3072
HORIZON = 16
ACTION_DIM = 30
USED_DOF = 7
HIDDEN = (1024, 1024, 1024)
ENSEMBLE = 10
BETA = 100.0
EPSILON = -0.5
GAMMA = 0.99
N_STEP_CHUNKS = 3
TAU = 0.01
ADAM_LR = 1e-4


def mask_unused_dof(chunk):
    out = chunk.clone()
    out[..., USED_DOF:] = 0
    return out


def apply_residual(a_base, residual):
    return mask_unused_dof(a_base + residual)


def _mlp(in_dim, out_dim):
    dims = [in_dim, *HIDDEN, out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(STATE_DIM + HORIZON * ACTION_DIM, HORIZON * ACTION_DIM)

    def forward(self, state, noise):
        b = state.shape[0]
        x = torch.cat([state, noise.reshape(b, -1)], dim=-1)
        residual = self.net(x).reshape(b, HORIZON, ACTION_DIM)
        return mask_unused_dof(residual)


class CriticEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            [_mlp(STATE_DIM + HORIZON * ACTION_DIM, 1) for _ in range(ENSEMBLE)]
        )

    def forward(self, state, action, return_all=False):
        b = state.shape[0]
        x = torch.cat([state, mask_unused_dof(action).reshape(b, -1)], dim=-1)
        qs = [head(x) for head in self.heads]
        if return_all:
            return qs
        return torch.min(torch.stack(qs, dim=0), dim=0).values


class DiceResidualModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.actor = ResidualActor().to(device)
        self.critic = CriticEnsemble().to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ADAM_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=ADAM_LR)

    def n_step_target(self, reward, done, next_state, next_action, n_steps):
        with torch.no_grad():
            backup = self.target_critic(next_state, next_action)
            discount = GAMMA ** n_steps
            return reward + discount * (1.0 - done) * backup

    def update_critic(self, state, action, target_q, is_expert):
        preds = self.critic(state, action, return_all=True)
        losses = [F.mse_loss(pred, target_q) for pred in preds]
        loss = torch.stack(losses).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.polyak_update()
        return {"critic_loss": float(loss.detach()), "q_mean": float(torch.stack(preds).mean().detach())}

    def update_actor(self, state, noise, a_base, is_expert, target_q):
        residual = self.actor(state, noise)
        action = apply_residual(a_base, residual)
        q_a = self.critic(state, action)
        with torch.no_grad():
            q_base = self.critic(state, a_base)
            better = (q_a > q_base).float()
            td_error = q_a - target_q
            underestimated = (td_error < EPSILON).float()
            bc_keep = 1.0 - better * underestimated
        online = (is_expert == 0).float()
        q_term = -(q_a * online).sum() / online.sum().clamp(min=1.0)
        mse = ((action - a_base) ** 2).mean(dim=(1, 2), keepdim=True)
        bc = (bc_keep * mse).mean()
        loss = q_term + BETA * bc
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        rms = float(((action - a_base).detach() ** 2).mean().sqrt())
        return {
            "actor_loss": float(loss.detach()),
            "residual_rms": rms,
            "q_mean": float(q_a.detach().mean()),
            "q_min": float(q_a.detach().min()),
            "bc_filter_rate": float(bc_keep.mean()),
        }

    def polyak_update(self):
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1.0 - TAU).add_(p.data, alpha=TAU)

    def inference_state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def load_inference_state_dict(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])

    def resume_state_dict(self):
        return {
            **self.inference_state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_resume_state_dict(self, payload):
        self.load_inference_state_dict(payload)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
```

Fix `test_n_step_chunk_return_sparse_terminal` so the first assertion uses already-summed n-step rewards (the buffer, Task 3, does the summing). Keep the last-row `done=1 → target=reward` check.

- [ ] **Step 4: Run tests**

Same pytest command. Expected: PASS. Also re-run Task 1 tests.

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 3: Chunk replay, n-step returns, RLPD mix

**Files:**
- Create: `script/lingbot_rl_buffer.py`
- Modify: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `RLConfig.rlpd_expert_ratio`, tensor layouts from Task 2
- Produces:
  - `ChunkReplay.add_online(transition: dict)` — one AR chunk
  - `ChunkReplay.add_expert(transition: dict)`
  - `ChunkReplay.finalize_episode(rewards, dones)` — writes n-step fields onto the last episode's online rows
  - `ChunkReplay.sample(batch_size, expert_ratio) -> dict` of tensors
  - Stored keys: `s, z, a_base, a, reward, done, s_next, a_next, n_steps, task_id, n_env_actions, is_expert`

n-step math (chunk index, sparse terminal):

```
# rewards r_0..r_{T-1} in {0,1}, typically all 0 then a single 1 on the success chunk
# For t, n = min(3, T-t)
# R_t = sum_{k=0}^{n-1} γ^k r_{t+k}
# done_t = 1 if any done in [t, t+n)
# s_next = s_{t+n} if t+n < T else s_{T-1}  (bootstrap 0 when done)
```

Worked example for the unit test: three chunks, rewards `[0, 0, 1]`, done `[0, 0, 1]`:

- t=0, n=3: `R = 0 + 0 + γ²·1 = γ²`, done=1, n_steps=3
- t=1, n=2: `R = 0 + γ·1 = γ`, done=1, n_steps=2
- t=2, n=1: `R = 1`, done=1, n_steps=1

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from script.lingbot_rl_buffer import ChunkReplay
from script.lingbot_rl_config import RLConfig
from script.lingbot_rl_model import ACTION_DIM, GAMMA, HORIZON, STATE_DIM


def _row(reward, done, expert=False):
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
        buf.add_online(_row(reward, done))
    buf.finalize_episode()
    batch = buf.sample(3, expert_ratio=0.0)
    # Order is insertion order for a 3-row buffer
    r = batch["reward"].squeeze().tolist()
    n = batch["n_steps"].squeeze().tolist()
    d = batch["done"].squeeze().tolist()
    assert n == [3.0, 2.0, 1.0]
    assert d == [1.0, 1.0, 1.0]
    assert r[0] == pytest.approx(GAMMA ** 2)
    assert r[1] == pytest.approx(GAMMA)
    assert r[2] == pytest.approx(1.0)


def test_rlpd_mix_respects_scheduled_ratio():
    buf = ChunkReplay(capacity=200)
    for _ in range(80):
        buf.add_online(_row(0, 0))
    buf.finalize_episode()
    for _ in range(80):
        buf.add_expert(_row(0, 1, expert=True))
    cfg = RLConfig().validate()
    ratio = cfg.rlpd_expert_ratio(0)
    assert ratio == pytest.approx(0.5)
    counts = []
    for _ in range(40):
        batch = buf.sample(20, expert_ratio=ratio)
        counts.append(float(batch["is_expert"].mean()))
    assert abs(sum(counts) / len(counts) - 0.5) < 0.15
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError: script.lingbot_rl_buffer`

- [ ] **Step 3: Implement `script/lingbot_rl_buffer.py`**

```python
import numpy as np
import torch

from script.lingbot_rl_model import ACTION_DIM, GAMMA, HORIZON, N_STEP_CHUNKS, STATE_DIM


class ChunkReplay:
    def __init__(self, capacity=100_000, device="cpu"):
        self.capacity = capacity
        self.device = device
        self._data = []
        self._episode_start = 0

    def __len__(self):
        return len(self._data)

    def add_online(self, row):
        row = {**row, "is_expert": np.float32(0.0)}
        self._data.append(row)
        if len(self._data) > self.capacity:
            drop = len(self._data) - self.capacity
            self._data = self._data[drop:]
            self._episode_start = max(0, self._episode_start - drop)

    def add_expert(self, row):
        row = {**row, "is_expert": np.float32(1.0), "n_steps": np.float32(1.0)}
        self._data.append(row)
        if len(self._data) > self.capacity:
            self._data = self._data[len(self._data) - self.capacity:]

    def finalize_episode(self):
        ep = self._data[self._episode_start:]
        t_len = len(ep)
        rewards = [float(r["reward"]) for r in ep]
        dones = [float(r["done"]) for r in ep]
        for t, row in enumerate(ep):
            n = min(N_STEP_CHUNKS, t_len - t)
            ret = 0.0
            done_n = 0.0
            for k in range(n):
                ret += (GAMMA ** k) * rewards[t + k]
                if dones[t + k]:
                    done_n = 1.0
            nxt = t + n if t + n < t_len else t_len - 1
            row["reward"] = np.float32(ret)
            row["done"] = np.float32(done_n)
            row["n_steps"] = np.float32(n)
            row["s_next"] = np.array(ep[nxt]["s"] if done_n else ep[nxt]["s"], copy=True)
            row["a_next"] = np.array(ep[nxt]["a"], copy=True)
        self._episode_start = len(self._data)

    def sample(self, batch_size, expert_ratio):
        online = [i for i, r in enumerate(self._data) if r["is_expert"] == 0]
        expert = [i for i, r in enumerate(self._data) if r["is_expert"] == 1]
        n_expert = int(round(batch_size * expert_ratio)) if expert else 0
        n_online = batch_size - n_expert
        if not online:
            raise ValueError("Replay has no online chunks")
        idx = list(np.random.choice(online, size=n_online, replace=len(online) < n_online))
        if n_expert:
            idx += list(np.random.choice(expert, size=n_expert, replace=len(expert) < n_expert))
        keys = ("s", "z", "a_base", "a", "reward", "done", "s_next", "a_next", "n_steps", "is_expert", "task_id", "n_env_actions")
        batch = {}
        for key in keys:
            stacked = np.stack([self._data[i][key] for i in idx])
            batch[key] = torch.from_numpy(np.asarray(stacked)).to(self.device)
            if key in ("reward", "done", "n_steps", "is_expert") and batch[key].ndim == 1:
                batch[key] = batch[key].unsqueeze(-1)
        return batch
```

Export `N_STEP_CHUNKS = 3` from `lingbot_rl_model.py` and import it in the buffer (do not duplicate a second `N_STEP` if it would drift). Use `from script.lingbot_rl_model import GAMMA, STATE_DIM, HORIZON, ACTION_DIM` and `N_STEP_CHUNKS = 3` as a module constant in the buffer.

For `s_next` when `done_n=1`, bootstrap is already zeroed by `done` in `n_step_target`; storing `s` of the terminal chunk is fine.

- [ ] **Step 4: Run tests** — PASS

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 4: Action geometry, first-chunk length, critic pooling

**Files:**
- Create: `script/lingbot_rl_policy.py` (helpers only in this task; `ResidualLingBotPolicy` in Task 5)
- Modify: `tests/test_lingbot_rl.py`

**Interfaces:**
- Produces:
  - `model_to_mlp(actions) -> (B, 16, 30)` from `(B, 30, 4, 4, 1)`
  - `mlp_to_model(chunk) -> (B, 30, 4, 4, 1)`
  - `env_action_count(first_chunk: bool) -> 12 or 16`
  - `slice_env_actions(mlp_chunk, first_chunk) -> (B, 12|16, 7)` normalized used DoF
  - `pool_critic_state(video_tokens, text_tokens) -> (B, 3072)`
  - `histogram_entropy(samples) -> float` for ΔH

- [ ] **Step 1: Failing tests**

```python
from script.lingbot_eval import decode_action
from script.lingbot_rl_policy import (
    env_action_count, histogram_entropy, mlp_to_model, model_to_mlp,
    pool_critic_state, slice_env_actions,
)


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
    from tests.test_lingbot_eval import normalization
    chunk = torch.zeros(1, 16, 30)
    chunk[0, 4, :7] = torch.tensor([-1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0])
    sliced = slice_env_actions(chunk, True)  # index 0 of first-chunk slice is model index 4
    decoded = decode_action(sliced[:, :1, :].reshape(1, 7), normalization())
    expected = decode_action(chunk[0:1, 4, :7], normalization())
    np.testing.assert_allclose(decoded, expected)


def test_mean_pool_depends_on_text():
    torch.manual_seed(0)
    video = torch.randn(2, 8, 3072)
    text_a = torch.zeros(2, 4, 3072)
    text_b = torch.ones(2, 4, 3072)
    s_a = pool_critic_state(video, text_a)
    s_b = pool_critic_state(video, text_b)
    assert s_a.shape == (2, 3072)
    assert not torch.allclose(s_a, s_b)


def test_model_mlp_roundtrip():
    torch.manual_seed(0)
    model = torch.randn(2, 30, 4, 4, 1)
    model[..., 7:, :, :, :] = 0
    back = mlp_to_model(model_to_mlp(model))
    torch.testing.assert_close(back, model)
```

- [ ] **Step 2: Run to fail** — missing helpers

- [ ] **Step 3: Implement helpers at the top of `script/lingbot_rl_policy.py`**

```python
import torch

from script.lingbot_rl_model import ACTION_DIM, HORIZON, STATE_DIM, USED_DOF


def env_action_count(first_chunk):
    return 12 if first_chunk else 16


def model_to_mlp(actions):
    # (B, 30, 4, 4, 1) -> (B, 16, 30)
    b = actions.shape[0]
    return actions.squeeze(-1).permute(0, 2, 3, 1).reshape(b, HORIZON, ACTION_DIM)


def mlp_to_model(chunk):
    b = chunk.shape[0]
    return chunk.reshape(b, 4, 4, ACTION_DIM).permute(0, 3, 1, 2).unsqueeze(-1)


def slice_env_actions(mlp_chunk, first_chunk):
    start = 4 if first_chunk else 0  # drop frame-0's 4 actions
    return mlp_chunk[:, start:, :USED_DOF]


def pool_critic_state(video_tokens, text_tokens):
    if video_tokens.shape[-1] != STATE_DIM or text_tokens.shape[-1] != STATE_DIM:
        raise ValueError("Critic tokens must be 3072-d pre-proj_out / text features")
    return torch.cat([video_tokens, text_tokens], dim=1).mean(dim=1)


def histogram_entropy(samples, bins=32):
    """Mean per-coordinate histogram entropy (paper ΔH). `samples` is (N, 16, 30) or (N, 480)."""
    x = samples.detach().float().cpu().numpy()
    if x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    entropies = []
    for dim in range(x.shape[1]):
        hist, _ = np.histogram(x[:, dim], bins=bins)
        total = hist.sum()
        if total == 0:
            continue
        p = hist.astype(np.float64) / total
        p = p[p > 0]
        entropies.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0
```

`test_denorm_matches_eval_decode_action` imports `normalization` from `tests.test_lingbot_eval`. If pytest collection makes that awkward, copy the tiny `normalization()` helper into `test_lingbot_rl.py` instead of importing from the other test module.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 5: ResidualLingBotPolicy (video freeze, K-batch, critic features, history)

**Files:**
- Modify: `script/lingbot_rl_policy.py`, `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: `script.lingbot_eval.load_policy` (reuse for constructing the frozen prior — do not change eval protocol). Subclass the returned class or wrap after load.
- Produces: `ResidualLingBotPolicy` with:
  - `extract_critic_state(batch) -> (1, 3072)` — action-free video-stream hook on **real** obs, pool with text
  - `decode_candidates(batch, k=4) -> dict(z, a_base, video_noise)` — one video Euler, K action Eulers batched
  - `commit_executed(mlp_chunk)` — writes residual-edited chunk into `_executed_actions` via `mlp_to_model`
  - `select_action(batch)` — eval path: K=1 residual policy, same first-chunk slicing as parent

Do **not** edit `.cache/lerobot`. Override `_infer` on a subclass defined in `lingbot_rl_policy.py`.

**Video / action split (copy parent `_infer`, then change):**

1. Sample or accept `video_noise` `(1, 48, 4, H, W)` once.
2. Run the existing video Euler (20 steps, `video_exec_step=-1`, CFG 5.0). Last video index: **do not** call `scheduler.step`; that forward only writes already-denoised latents into KV (parent `_infer`).
3. Sample `z` `(K, 30, 4, 4, 1)` — **do not** resample video noise.
4. Action CFG is 1.0. Expand the **conditional** KV (batch index 0) to K. Do **not** expand uncond and run action CFG (that would be batch 8). If KV tensors cannot be expanded, **raise** `RuntimeError("action candidate batching failed")` — never silently drop K. If the smoke OOMs, raise; never drop ensemble size either.
5. Action Euler 50 steps, batched K. On the first chunk, frame-0 action slots stay the **zero** condition (parent `action_cond`).
6. Zero unused DoF. Convert to MLP layout. Apply residual **outside** `_infer` (collection and eval both apply residual after `a_base`). Residual is predicted on the **full** 16×30 chunk even when only 12 env actions execute.

**Critic state:** register a forward hook on `transformer.proj_out` capturing its **input** (3072-d). Run one video-stream forward on the VAE-encoded **current** observation (`action_mode=False`, `update_cache=0`, no CFG). Text tokens: `transformer.condition_embedder.text_embedder(prompt_embeds)`. `pool_critic_state`. Remove hook. Do not store KV.

**History:** `commit_executed` must set `_executed_actions` to the residual-edited model chunk **before** the next `_compute_kv_cache`. Never write `a_base`.

**Determinism test:** mocked transformer that returns zeros + identity scheduler step is enough locally. Assert same `(video_noise, action_noise)` → same `a_base`.

- [ ] **Step 1: Failing tests (mocked, no weights)**

```python
def test_decode_candidates_batches_k_and_freezes_video_noise():
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    policy = ResidualLingBotPolicy.fake_for_test(k=4)
    batch = {"task": ["pick the mug"], "observation.images.image": torch.zeros(1, 3, 8, 8),
             "observation.images.image2": torch.zeros(1, 3, 8, 8)}
    out = policy.decode_candidates(batch, k=4, video_noise=policy.fixed_video_noise, action_noise=policy.fixed_action_noise)
    assert out["a_base"].shape == (4, 16, 30)
    assert out["z"].shape == (4, 16, 30)
    again = policy.decode_candidates(batch, k=4, video_noise=policy.fixed_video_noise, action_noise=policy.fixed_action_noise)
    torch.testing.assert_close(out["a_base"], again["a_base"])
    assert policy.video_forwards == 1 or policy.video_loops == 2  # one loop per decode call, not 4


def test_commit_executed_writes_residual_not_base():
    policy = ResidualLingBotPolicy.fake_for_test(k=1)
    a_base = torch.zeros(1, 16, 30)
    residual = torch.ones(1, 16, 30)
    from script.lingbot_rl_model import apply_residual
    a = apply_residual(a_base, residual)
    policy.commit_executed(a)
    written = model_to_mlp(policy._executed_actions)
    torch.testing.assert_close(written[:, :, :7], a[:, :, :7])
```

Implement `ResidualLingBotPolicy.fake_for_test` only as a **test helper in the test file** (a duck-typed stub with the same methods), not as a production factory. Production class must still exist for the import to work.

Prefer: test `commit_executed` and layout helpers on the real class with a `SimpleNamespace` stub for `_executed_actions`. Test `pool_critic_state` already covers text dependence. Add one test that `decode_candidates` **contract** is documented via a small fake:

```python
class FakeResidualPolicy:
    def decode_candidates(self, batch, k=4, video_noise=None, action_noise=None):
        z = video_noise if action_noise is None else action_noise
        a_base = torch.zeros(k, 16, 30)
        return {"z": z, "a_base": a_base, "video_noise": video_noise}
```

This is weaker than instantiating `LingBotVAPolicy`. Local CI cannot load the 5B model. Keep production `ResidualLingBotPolicy` implementing the real `_infer` split; unit-test the pure functions plus a **thin** test that `ResidualLingBotPolicy` defines `decode_candidates`, `extract_critic_state`, `commit_executed`, and `select_action`.

```python
def test_policy_class_exposes_collection_api():
    from script.lingbot_rl_policy import ResidualLingBotPolicy
    for name in ("decode_candidates", "extract_critic_state", "commit_executed", "select_action"):
        assert callable(getattr(ResidualLingBotPolicy, name))
```

- [ ] **Step 2: Fail** — class missing

- [ ] **Step 3: Implement `ResidualLingBotPolicy`**

Subclass `lerobot.policies.lingbot_va.modeling_lingbot_va.LingBotVAPolicy`. Wrap construction:

```python
def load_residual_policy(checkpoint, model_path, architecture):
    from script.lingbot_eval import load_policy
    # load_policy returns CachedTextPolicy. We need ResidualLingBotPolicy.
```

`load_policy` hard-codes `CachedTextPolicy(LingBotVAPolicy)`. Do **not** change `lingbot_eval.py`. Instead, in `lingbot_rl_policy.py`, copy the `LingBotVAConfig` construction from `load_policy` **verbatim** (same kwargs: `num_inference_steps=20`, `video_exec_step=-1`, `action_num_inference_steps=50`, CFG 5/1, SNR 5.0/0.05, `attn_mode="torch"`, `dtype="bfloat16"`, `text_encoder_device="cpu"`, `image_hflip=False`, `camera_layout="width_concat"`, `used_action_channel_ids=list(range(7))`, `save_predicted_video=False`) and instantiate `ResidualLingBotPolicy`. Duplicate the safetensors load + `requires_grad_(False)` + eval. Keep it in `load_residual_policy()` so train/eval share it.

`ResidualLingBotPolicy` also copies `_get_t5_prompt_embeds` caching from eval.

`_infer(self, init_latent, frame_st_id=0, video_noise=None, action_noise=None)`:
- Use provided noises if not None, else `torch.randn` as parent
- Video loop: copy parent (do not change `video_exec_step` handling)
- After video, if `action_noise` has batch K>1: expand cond KV, run action loop with batch K
- Return `actions` with batch K

`decode_candidates`: `predict_action_chunk` analogue that returns all K `a_base` without applying residual and **without** setting `_executed_actions` yet.

`select_action` for eval: sample k=1, `extract_critic_state`, `actor(s,z)`, `apply_residual`, `commit_executed`, then enqueue sliced env actions (12 vs 16) like parent `predict_action_chunk`.

Implement KV expand:

```python
def expand_conditional_kv(transformer, k):
    cache = transformer.cache  # inspect actual attribute names in utils.py init_kv_cache
    # take batch index 0, repeat K
```

Read `init_kv_cache` / block cache dict (`k`,`v` shape `[B, T, H, D]`) at implementation time and repeat dim 0. If `guidance_scale>1` made B=2, slice `[0:1]` first.

- [ ] **Step 4: Tests pass** (API + helpers). Cloud smoke later.

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 6: Expert featurization of 300 demos

**Files:**
- Create: `script/lingbot_rl_data.py`
- Modify: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: checkpoint `dataset_manifest.json` (300 episodes), checkpoint `norm_stats.json`, `ResidualLingBotPolicy.extract_critic_state`, `mask_unused_dof`
- Produces: `featurize_experts(policy, dataset, manifest, norm, cache_path) -> list[row]`
  - Cache file `expert_features.pt` under the RL run dir
  - If cache exists and fingerprint matches manifest+norm+recipe, load and return
  - **GPU** work: VAE + action-free transformer. Modal CPU `prepare` does **eval assets only**. Featurize at GPU train start if the cache is missing.
  - Prefer published latents **only if** they match demo cameras; otherwise encode RGB from LeRobot `robbyant/libero-long-lerobot`. Either path uses the checkpoint `norm_stats.json` (ε=1e-6).
  - Each demo walked at execute stride: first chunk 12 env actions, later 16
  - Pad 7-d actions to `(16, 30)` normalized with unused DoF 0
  - `z` zeros `(16, 30)`; `a_base = a` (expert residual target 0)
  - `is_expert=1`, `reward=1` on last chunk of a successful demo else 0, `done=1` on last chunk

Normalize env actions to model space (inverse of `decode_action`):

```
normed = 2 * (raw - q01) / (q99 - q01 + 1e-6) - 1
```

Local test: **do not** load LeRobot dataset. Feed a fake iterator of one episode with 12+16+4 actions and a stub `extract_critic_state`.

- [ ] **Step 1: Failing test**

```python
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
    rows = featurize_experts(StubPolicy(), [episode], norm=normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(rows) == 3
    assert rows[0]["n_env_actions"] == 12
    assert rows[1]["n_env_actions"] == 16
    assert rows[2]["n_env_actions"] == 4
    assert rows[0]["a"].shape == (16, 30)
    assert np.count_nonzero(rows[0]["a"][:, 7:]) == 0
    assert rows[-1]["done"] == 1
    again = featurize_experts(StubPolicy(), [episode], norm=normalization(), cache_path=tmp_path / "expert_features.pt")
    assert len(again) == 3
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement `script/lingbot_rl_data.py`** with cache fingerprint `sha256(manifest fingerprint + "dice-rl-expert-v1")`. Real GPU path loads LeRobot `robbyant/libero-long-lerobot` episodes listed in the checkpoint manifest (same 300). If a frame count is not divisible, last chunk uses actual leftover length in `n_env_actions` and pads the rest of the 16-slot tensor with zeros.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 7: Training loop, W&B, resume, eval hooks

**Files:**
- Create: `script/lingbot_rl_train.py`
- Modify: `tests/test_lingbot_rl.py`

**Interfaces:**
- Consumes: Tasks 1–6, `script.lingbot_eval.{observation_batch, decode_action, run_episode, aggregate_results, seed_all, describe_suite, read_checkpoint_metadata}`, `EvalConfig` + `episode_plan` for comparison eval
- Produces: `train(config, prepared_path, output_dir, run_name, resume=False, commit=None)`
  - `save_resume(path)` writes `resume/latest.pt` (MLPs, Adam, replay list, RNG, `env_steps`, recipe fingerprint) atomically
  - `save_inference(path)` writes `residual.pt` (actor+critic only)
  - Train-time eval at env steps `{0, 25k, 50k, 75k, 100k}`: 1 rollout/task, init-state 0, recorded seeds via `episode_plan(EvalConfig(stage="smoke"), ...)`; log per-task success
  - Comparison eval uses `EvalConfig(stage="eval")` 20/task, init-states 1–20, seed 42. **Do not** re-run `libero30-sft-step000600-eval`.
  - Optimizer batch: `min(256, len(buffer))`
  - Resume refuses a fingerprint mismatch (sampler, β, n, checkpoint step)

Loop (single env):

```
while env_steps < 100_000:
    task_id ~ U{0..9}
    reset LiberoEnv for that task
    policy.reset()
    first_chunk = True
    until success/term or 520 env steps:
        s = extract_critic_state(obs)
        cands = decode_candidates(obs, k=4)   # may OOM → raise, do not drop K
        residual = actor(s.expand(K), z)
        a = apply_residual(a_base, residual)
        k* = argmax_k Q(s, a_k)
        commit_executed(a[k*])
        execute 12 or 16 env actions; reward 1 on first is_success else 0
        store online row; env_steps += n_env_actions
        if buffer has any online:
            batch = buffer.sample(min(256, len(buffer)), expert_ratio)
            for _ in range(10): critic update   # UTD 10; critic TD includes expert rows
            1 actor update                      # actor Q-max online-only; BC on all
            wandb.log(...)
    finalize_episode()
    maybe eval, maybe save residual.pt at the 25k grid
```

W&B scalars every optimizer step: `env_steps, chunks, actor_loss, critic_loss, residual_rms, q_mean, q_min, bc_filter_rate, expert_ratio, episode_return, episode_success, episode_length`.

ΔH / ΔV at train-eval points only, on a small expert-anchor batch. Sample K_entropy=8 action noises through **frozen π_pre with the same video noise** (5B allowed here, not in the optimizer). `delta_v = Q(s,a) − Q(s,a_base)`, `delta_h = mean_d (H_d(a_base) − H_d(a))` using **per-coordinate** histograms. Residual RMS is logged on every optimizer step separately; it is **not** ΔH.

Refuse to save transformer weights: `save_resume` keys must be a fixed allow-list; assert `"transformer"` not in payload.

- [ ] **Step 1: Failing tests**

```python
def test_train_mocked_step_logs_and_saves_small_weights(tmp_path, monkeypatch):
    import torch
    import script.lingbot_rl_train as train
    from script.lingbot_rl_model import ACTION_DIM, HORIZON, STATE_DIM, DiceResidualModel

    class StubPolicy:
        def reset(self):
            self._executed_actions = None
        def extract_critic_state(self, batch):
            return torch.zeros(1, STATE_DIM)
        def decode_candidates(self, batch, k=4, **kwargs):
            z = torch.zeros(k, HORIZON, ACTION_DIM)
            a_base = torch.zeros(k, HORIZON, ACTION_DIM)
            return {"z": z, "a_base": a_base, "video_noise": torch.zeros(1)}
        def commit_executed(self, chunk):
            self._executed_actions = chunk
        def select_action(self, batch):
            return torch.zeros(1, 7)

    class StubEnv:
        def __init__(self, **kwargs):
            self.steps = 0
            self.init_state_id = 0
        def reset(self, seed=None):
            self.steps = 0
            return {"pixels": {"image": __import__("numpy").zeros((128, 128, 3), __import__("numpy").uint8),
                               "image2": __import__("numpy").zeros((128, 128, 3), __import__("numpy").uint8)}}, {}
        def step(self, action):
            self.steps += 1
            done = self.steps >= 12
            return self.reset()[0], 0.0, done, False, {"is_success": False}
        def close(self):
            return None

    logs = []
    monkeypatch.setattr(train, "load_residual_policy", lambda *a, **k: StubPolicy())
    monkeypatch.setattr(train, "LiberoEnv", StubEnv)
    monkeypatch.setitem(__import__("sys").modules, "wandb",
                        type("W", (), {"init": staticmethod(lambda **k: type("R", (), {"log": logs.append, "summary": {}, "finish": lambda **k: None})()), "finish": staticmethod(lambda **k: None)})())
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
```

`max_env_steps` is a `train()` argument used only when provided; default `None` means `config.online_env_steps`. Production Modal never passes it. Tests pass `max_env_steps=12`. Do not `replace(RLConfig(), online_env_steps=12)` — validate rejects that.

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement `script/lingbot_rl_train.py`**. Reuse eval `LiberoEnv` constructor kwargs from `lingbot_eval.evaluate`. Comparison eval (`operation=eval`) loads `residual.pt`, wraps `select_action`, calls `run_episode` per `episode_plan(EvalConfig(stage="eval"), ...)`, writes the same episode JSON layout so `lingbot_eval_report.create_report` works.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

### Task 8: Modal app, inference-only download, docs

**Files:**
- Create: `script/lingbot_rl_modal.py`
- Modify: `tests/test_lingbot_rl.py`, `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: eval image builder in `script/lingbot_eval_modal.py` (`build_image` pattern, `--index-url https://pypi.org/simple`, locked uv, `MUJOCO_GL=egl`)
- Produces: `modal run -m script.lingbot_rl_modal --stage train --run-name libero30-dice-baseline`

Volumes:

| Mount | Volume | Mode |
| --- | --- | --- |
| `/cache` | `dice-lingbot-sft-cache` | RO on GPU |
| `/sft` | `dice-lingbot-sft-runs` | RO |
| `/rl` | `dice-lingbot-rl-runs` | RW |

Secrets: `dice-lingbot-hf` on **CPU prepare only**; `dice-lingbot-wandb` on GPU. GPU train/eval set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` like eval. Timeouts: prepare 3h (assets), train **12h**, eval 6h (200 episodes). `retries=0`, `max_containers=1`, persistent `modal.Dict` lock `dice-lingbot-rl-run-locks`. No region pin. No automatic stale-lock takeover.

Stages: `prepare` = eval asset prep only (CPU). `train` = GPU loop; expert featurize at start if cache missing. `eval` = 20-rollout comparison with residual weights. Local entrypoint accepts `--wandb-entity` and `--download-dir`. After success, **keep the client attached** until `download_inference` finishes.

`download_inference(run_name, download_dir)`:
- Refuse if local dest exists
- `modal volume get` **only** these remote paths if present: `residual.pt`, `summary.json`, `settings.json`, `status.json`, `train_eval/` (if any), `eval/` (comparison)
- Do **not** get `resume/`, `replay`, `expert_features.pt`
- After eval stage, run `create_report` on the eval folder

Add local files to the image: `lingbot_rl_*.py` plus the eval/sft config modules eval already needs.

- [ ] **Step 1: Failing tests** (mocked argv, no secrets printed)

```python
def test_modal_download_skips_resume_and_refuses_overwrite(tmp_path, monkeypatch):
    import script.lingbot_rl_modal as module
    calls = []
    def download(command, check):
        calls.append(command)
        dest = Path(command[-1]) / "libero30-dice-baseline"
        dest.mkdir(parents=True)
        (dest / "residual.pt").write_bytes(b"x")
        (dest / "summary.json").write_text("{}")
    monkeypatch.setattr(module.subprocess, "run", download)
    out = module.download_inference("libero30-dice-baseline", tmp_path / "result/lingbot-rl")
    assert Path(out).joinpath("residual.pt").is_file()
    joined = " ".join(calls[0])
    assert "resume" not in joined
    assert "expert_features" not in joined
    with pytest.raises(FileExistsError):
        module.download_inference("libero30-dice-baseline", tmp_path / "result/lingbot-rl")


def test_modal_lock_and_help_do_not_print_secrets(capsys):
    import script.lingbot_rl_modal as module
    assert module.WANDB_SECRET_NAME == "dice-lingbot-wandb"
    # import module; help path used in compile/help check below
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement `script/lingbot_rl_modal.py`** mirroring `lingbot_eval_modal.py` (`prepare.remote` then `run_train.remote` / `run_eval.remote`). GPU functions `subprocess.run` the eval Python: `/opt/lerobot/.venv/bin/python -m script.lingbot_rl_train ...`. Local entrypoint downloads after success.

Update `AGENTS.md` with:

```text
.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py
.cache/eval-venv/bin/python -m compileall -q script/lingbot_rl_*.py
.cache/eval-venv/bin/python -m script.lingbot_rl_config
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 modal run -m script.lingbot_rl_modal --help
```

Update `README.md`: replace “RL is not implemented” / “Planned RL architecture” with the launched command (user-run, not executed here):

```text
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_rl_modal --stage train --run-name libero30-dice-baseline
```

Do not launch paid jobs from tests.

- [ ] **Step 4: Run full local verification**

```text
.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py tests/test_lingbot_eval.py
.cache/eval-venv/bin/python -m compileall -q script/lingbot_rl_config.py script/lingbot_rl_model.py script/lingbot_rl_buffer.py script/lingbot_rl_policy.py script/lingbot_rl_data.py script/lingbot_rl_train.py script/lingbot_rl_modal.py
.cache/eval-venv/bin/python -m script.lingbot_rl_config
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 modal run -m script.lingbot_rl_modal --help
```

Expected: pytest pass, compileall silent, config JSON prints pinned 20/50/-1/600, modal help works, no `.remote()`.

- [ ] **Step 5: Commit** — include this task's files; no `Co-authored-by`.

---

## Self-review

**Spec coverage**

| Spec section | Task |
| --- | --- |
| §4 frozen prior + sampler pin | 1, 5, 7 |
| §5.1 residual on z, first-chunk 12, history writeback, K=4 batch | 2, 4, 5 |
| §5.2 critic pool video+text 3072 | 4, 5 |
| §5.3 MLP/ensemble/β/ε/UTD/RLPD/BC filter/expert Q off | 2, 3, 7 |
| §5.4 replay fields + n-step chunks | 3 |
| §5.5 expert 300 demos + cache | 6 |
| §5.6 100k env steps, 1 env, resume without transformer | 7 |
| §5.7 train-time 1/task vs 20/task comparison | 7 |
| §6 W&B + ΔH/ΔV | 7 |
| §7 Modal + inference-only download | 8 |
| §8 local tests | 1–8 |
| Non-goals (no DSRL, no Hydra, no s=0.6, no 100-rollout in loop) | Global constraints |

**Placeholder scan:** no TBD / `...` test bodies. `fake_for_test` on the production class was rejected in Task 5 in favor of a test stub + API presence test. Critic TD on expert is on; actor Q-max on expert is off. ΔH is per-coordinate.

**Type consistency:** `s` is `(B, 3072)` everywhere; chunks `(B, 16, 30)`; `is_expert` `(B, 1)`; `n_steps` `(B, 1)` float.

**Gaps closed vs first draft:** noisy-history off, bf16/SDPA/attn_window/ε, UMT5 CPU, last video step KV-only, first-chunk zero action cond, per-coordinate ΔH, critic-on-expert, GPU expert featurize (CPU prepare = assets), batch `min(256,n)`, recipe fingerprint resume, do not re-run 69% SFT eval, `--wandb-entity` / `--download-dir` / attached client, OOM must not drop ensemble, `HF_HUB_OFFLINE` on GPU.

---

## Execution notes

Cloud smoke (operator-launched, not this plan's local verification): a few chunks on one H100, W&B step appears, no OOM, `a_base`/`residual` finite, K stays 4.
