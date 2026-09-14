# LingBot-VA DICE-RL baseline — design

Date: 2026-09-14
Status: draft for review

Repository: `dice-rl-wam`.

Port DICE-RL residual finetuning onto the LIBERO-10 LingBot-VA SFT prior, with
the same inference sampler already used to measure that prior. First deliverable
is one baseline run on a single Modal H100, logged to W&B.

## 1. Goal

Train a frozen-prior residual policy on **LIBERO-10 (LIBERO-Long)** starting from
`libero30-sft` **step 000600**. The SFT checkpoint is already measured at **69%**
macro success (20 rollouts/task, 138/200). That number is the SFT point on the
curve. DICE-RL must use the same decoder so a later success rate is attributable
to the residual, not to a sampler change.

Success for this phase: a Modal-launchable trainer that collects online LIBERO
rollouts, updates residual and critic MLPs, logs losses and success to W&B, and
can be evaluated with the existing LIBERO-10 protocol.

## 2. Non-goals

- DSRL, or any second RL algorithm.
- Robomimic, PushT, D3IL, Furniture, or the Hydra residual agent under
  `agent/finetune/` and `cfg/robomimic/`. That stack is left untouched.
- Changing SFT weights, the 30-demo subset, or the eval harness provenance
  (`script/lingbot_eval.py` 20/50 protocol).
- Re-evaluating step 600 under LingBot’s paper real-time sampler (3 video steps
  to s=0.6, 10 action steps). That is a different π_pre than the 69% eval.
- Vectorized **environments**. Collection is one `LiberoEnv`. K=4 action
  candidates share one video decode (batched action Euler on one GPU).
- Critic state from KV cache or long-horizon memory. Phase 1 pools the current
  chunk only; LIBERO-Long memory dependence is a known limitation (task 8 is
  10% under SFT).
- Ablation sweeps (β, n, M, BC filter). Defaults below are the only v1 run.
- 100 rollouts/task inside the training loop. That job is a follow-up comparison
  after the baseline trains.

## 3. Why LingBot has a video denoise step (π0 does not)

DICE-RL Appendix A finetunes **π0**, a VLA. π0’s backbone is a vision-language
model; its action expert is flow-matching on **actions only**. There is no video
generator. “Sample z, decode an action chunk” is the whole prior. Critic state
is one action-free backbone pass, mean-pooled over image and text tokens.

**LingBot-VA is not that architecture.** It is a dual-stream video-action world
model: the same transformer denoises **Wan VAE video latents** and then **actions**.
LeRobot inference (`LingBotVAPolicy._infer`) is:

1. Sample video noise, run the video Euler loop, write imagined latents into the
   KV cache.
2. Sample action noise, run the action Euler loop conditioned on that cache
   (and on text via cross-attention).

The video loop exists because **LingBot produces actions by imagining video**,
not because DICE-RL requires video. The π0 port has no analogue.

What we copy from DICE-RL+π0 is the *control* recipe, not the backbone:

| DICE-RL + π0 | This port + LingBot-VA |
| --- | --- |
| Frozen VLA; residual on actions | Frozen world model; residual on actions |
| Sample action noise z | Sample action noise z; **hold video imagination fixed for a chunk** |
| Critic: pool image + text tokens | Critic: one action-free pass; pool **video-stream and text** tokens |
| No video sampler to pin | Pin LingBot’s **released LIBERO sampler** (below) |

“Fix video, resample action noise” is the LingBot-specific translation of “only
tune the action decode.” Video noise is drawn once per chunk and not resampled
when exploring action residuals.

### 3.1 Two LingBot samplers (paper real-time vs released LIBERO)

DICE-RL never partially denoises video. Its only iterative sampler is **action**
flow/diffusion: Robomimic uses 10 Euler steps on the action chunk (20 on Tool
Hang); π0 uses its usual action-flow denoiser after one action-free VLM pass.
There is no `s=0.6` anywhere in DICE-RL.

`s` in LingBot is **video flow time**: 0 = Gaussian noise, 1 = clean imagined
latents. `s=0.6` means stop video integration 60% of the way to clean, then
decode **actions fully** (`s=1`). It is not DICE-RL, and it is not “40% of 20
LIBERO steps” (that was a discarded back-of-envelope on an unshifted σ grid).

LingBot publishes two different recipes:

| | Paper real-time (§3.4 Algorithm 1, §4.2) | Released LIBERO code |
| --- | --- | --- |
| Video | Euler **3** steps, integrate **0→0.6** (Algorithm 1 also writes 0.5) | **20** steps, `video_exec_step=-1` → **0→1** |
| Action | **10** steps to s=1.0 | **50** steps to s=1.0 |
| CFG | 5.0 video / 1.0 action | same |
| Why it works | Noisy-history aug at train (`p=0.5`, `s_aug∈[0.5,1]`) so the action expert can read partially noisy video | Default in `wan_va/configs/va_libero_cfg.py` and LeRobot LIBERO |

The 98.5% LIBERO table does not name the sampler. The config they shipped for
LIBERO is the 20/50 full-video column. The 3-step / s=0.6 numbers sit in the
**real-time deployment** writeup. Our SFT used noisy-history probability 0.5, so
the weights *can* run at s=0.6; we did not evaluate that decoder.

**This RL run pins the released LIBERO sampler**, because that is the decoder
that produced the 69% SFT number. Switching to s=0.6 (or 10 action steps) is a
different π_pre and would require a new SFT eval before any RL delta is
meaningful.

### 3.2 Evidence: the Modal SFT eval fully denoised video

The completed 200-episode job
`result/lingbot-eval/libero30-sft-step000600-eval/` recorded the protocol in
`settings.json` (not reconstructed after the fact):

- `"sampler": "released_lingbot_libero_defaults"`
- `"video_steps": 20`
- `"video_exec_step": -1`
- `"action_steps": 50`
- `"video_guidance": 5.0`, `"action_guidance": 1.0`
- `source_run: libero30-sft`, `checkpoint_step: 600`
- macro success **0.69** (`summary.json`, 138/200)

That protocol is what `script/lingbot_eval.py` `load_policy` actually
constructed:

```text
num_inference_steps=20, video_exec_step=-1, action_num_inference_steps=50
```

LeRobot `_infer` implements the cutoff as a **prefix of the video timestep
list**. `-1` means do not slice:

```text
timesteps = pad(scheduler.timesteps, value=0)   # 20 Euler times + terminal t=0
if video_exec_step != -1:
    timesteps = timesteps[:video_exec_step]     # skipped on our eval
```

On the last video index, when `video_exec_step == -1`, the code **does not**
call `scheduler.step`; that forward only writes the **already fully denoised**
latents into the KV cache. So video was integrated to clean (`s=1`), then
actions ran 50 Euler steps. Upstream `va_libero_cfg.py` is the same triple
`(20, -1, 50)`.

We are confident the 69% prior is this full-video decoder. DICE-RL collection
and the 20-rollout comparison must use it.

## 4. Frozen prior (π_pre)

| Item | Value |
| --- | --- |
| Checkpoint | Modal `dice-lingbot-sft-runs` / `libero30-sft/checkpoints/step_000600` |
| Weights | Native transformer safetensors; frozen VAE + UMT5 from `robbyant/lingbot-va-base` |
| Suite | LIBERO-10, 128×128 agentview then wrist, width-concat latents |
| Actions | 7 LIBERO DoF in channels 0–6 of 30; unused channels zero-masked |
| Normalization | Checkpoint `norm_stats.json` (300-demo q01/q99, ε=1e-6) |
| Video Euler | **20** steps, `video_exec_step=-1` (full denoise) |
| Action Euler | **50** steps |
| CFG | 5.0 video / 1.0 action |
| SNR shift | 5.0 video / 0.05 action |
| Chunk | `K_AR=4` latent frames × 4 actions/frame = 16 action tokens |
| Noisy-history aug | Off at inference |
| Attention | SDPA (`attn_mode=torch`), bf16 |
| Cameras | Vertical-only flip, native LingBot client orientation |

These match `script/lingbot_eval.py` `load_policy` / `EvalConfig.protocol()`.
They are immutable for this run. Do not introduce s=0.6 or 10 action steps.

Measured prior: 69% macro, 20 rollouts/task, init-state IDs 1–20, seed 42.
That eval is **not** rerun. RL eval that claims a delta against 69% must use the
same sampler, cameras, denormalization, 520-step horizon, and the same 20-rollout
episode plan.

Transformer parameters stay frozen (`requires_grad_(False)`, eval mode). Only
residual and critic MLPs are trained.

## 5. Architecture

New scripts on the **eval** Python environment (pinned LeRobot + LIBERO), not
the SFT FSDP env and not the root `pyproject.toml`. Do not edit
`.cache/lerobot` or `.cache/lingbot-va`. Do not modify `script/lingbot_eval.py`
protocol fields.

```text
script/lingbot_rl_config.py    # pinned recipe, names, W&B project
script/lingbot_rl_policy.py    # LingBotVAPolicy subclass: video freeze, features, residual
script/lingbot_rl_model.py     # residual + critic ensemble + DICE-RL losses
script/lingbot_rl_buffer.py    # chunk replay, n-step, RLPD mix
script/lingbot_rl_data.py      # one-time expert featurization of 300 demos
script/lingbot_rl_train.py     # single-env loop, W&B, checkpoints
script/lingbot_rl_modal.py     # CPU prep + one-H100 train/eval
tests/test_lingbot_rl.py       # math, masks, config, mocked step
```

Copy residual/critic **loss math** from `model/rl/distill_residual_rl.py` where
it matches this spec. Do not instantiate `DistillResidualRLModel` (Hydra flow
loader, Robomimic obs). Do not import Robomimic env wrappers.

### 5.1 Policy wrapper

Subclass `LingBotVAPolicy`.

**Chunk decode.** Split `_infer` into (1) video imagination with a stored video
noise, **20 full Euler steps, no cutoff**, (2) action Euler from supplied
action noise(s), **50 steps**. For one env chunk: sample video noise once,
fully denoise video, cache KV; sample **K=4** action noises; denoise those four
action chunks **in one batched Euler** (repeat the video KV along batch; action
CFG stays 1.0 so action batch is 4, not 8). Eval already peaked ~13.7 GB of an
80 GB H100; four action-token forwards share the frozen 5B weights and should
fit. If the smoke OOMs, fail explicitly rather than silently dropping K.

**First-chunk length.** LingBot drops frame-0 actions on the first chunk (12
env actions). Later chunks execute 16. Residual is always predicted on the
full 16×30 model chunk; frame-0 on the first chunk stays the zero action
condition; env execution uses the same slice as `select_action`.

**Residual apply (DICE-RL Eq. (2)).** `a = π_pre(s,z) + s_θ(s, z)` in
**normalized** space, before `decode_action`. The MLP is conditioned on pooled
`s` and the **action-noise** `z` (flattened 16×30), not on decoded `a_base`.
The residual is still *added* to `a_base = π_pre(s,z)`. Zero residual and
outputs on channels 7–29, then denormalize channels 0–6 as eval (`q01`/`q99`,
ε=1e-6). Conditioning on `a_base` instead of `z` is a codebase flag, not the
paper default; this baseline follows the paper.

**History.** Write the **executed** (residual-edited) chunk into
`_executed_actions` before `_compute_kv_cache`. Never feed `a_base` back as if
it had been executed.

**Determinism.** Same `(video_noise, action_noise, obs, text, history)` must
yield the same `a_base`. Covered by a unit test with mocked or tiny tensors;
full 5B equality is a cloud smoke check.

### 5.2 Critic state

LingBot text is **cross-attention**, not π0’s joint image/text sequence. The
faithful port:

1. Action-free video-stream forward on the **real** VAE-encoded observation at
   the start of the chunk (not imagined video, not action tokens).
2. Take **pre-`proj_out`** hidden states, dim **3072**
   (`24 heads × 128`).
3. Take text tokens after `condition_embedder.text_embedder` (also 3072).
4. Concatenate video tokens and text tokens on the sequence axis; **mean-pool**
   to one vector `s ∈ R^{3072}`.

Q is therefore task-conditioned. Encoder frozen: compute `s` once at collection
(and once when featurizing demos); store the vector in replay. Do not store KV
caches.

Phase-1 limitation: `s` has no AR memory. The frozen prior still uses KV for
generation; the critic does not.

### 5.3 Residual and critic networks

| Item | Value |
| --- | --- |
| Residual | MLP `[1024, 1024, 1024]`, GELU, LayerNorm on; input `(s, z_flat)`; output 16×30 |
| Residual input | Action-noise `z` (480 dims), paper Eq. (2); **not** decoded `a_base` |
| Critic | Ensemble of **10** Q heads, same MLP width, GELU, LayerNorm; input `(s, a_flat)`; min over ensemble |
| Target critic | Polyak **τ = 0.01** |
| Optimizers | Adam, **1e-4**, both actor and critic |
| β | 100 (BC / residual regularizer weight) |
| ε | −0.5 (Q-filter underestimation threshold) |
| n-step | **3 chunks** (not 3 env steps), γ = 0.99 per chunk |
| UTD | 10 critic gradient steps per collected chunk |
| RLPD | offline fraction 0.5 → 0.1 linear over the run |
| BC filter | on |
| K (best-of-N at collection) | **4**, batched action Euler after one shared video decode |

Paper Algorithm 1 / Table 2 use **K=16** at both collection and in the MLP
update, because Robomimic `π_pre(s,z)` is a small flow net on already-pooled
`s`. LingBot `π_pre` needs live images/KV, and replay stores pooled `s` not
pixels, so **train-time** resampling of new `z` cannot call the 5B model.
Collection uses K=4 (batched action Euler + execute `argmax_k Q`). Actor/critic
updates use the stored `(s, z, a_base)` from that chosen candidate. That is the
π0-scale translation of multi-sample, not K=16 inside every MLP step.

Expert Q-loss is disabled (`disable_q_loss_for_expert_data`); expert data still
mixes into the replay for RLPD.

### 5.4 Replay buffer

One transition = one AR chunk.

Stored fields: `s` (3072), `z` (16×30 action noise), `a_base` (16×30), `a`
(16×30 executed), `reward` (scalar), `done`, `s_next`, `task_id`,
`n_env_actions` (12 or 16), `is_expert`.

n-step returns are computed in **chunk** index. Reward is sparse terminal
success already reported by LIBERO (`info["is_success"]`): 1 on the chunk that
first sees success, else 0.

Capacity: 100,000 chunk transitions (online + expert). CPU numpy / torch host
storage; `s` is small.

### 5.5 Expert data (RLPD)

The same 300 SFT demonstrations (manifest in the checkpoint). Featurize once on
GPU at train start (or a dedicated Modal prepare stage): walk each demo at
chunk stride matching executed lengths, encode start-of-chunk images with the
frozen VAE + action-free transformer, pad 7-d demo actions into 16×30
normalized chunks with unused DoF zero. Write a feature cache under the RL run
volume so resume does not recompute.

Do not require a second copy of the latent training set beyond what SFT cache
and the LeRobot LIBERO dataset already provide. If published latents align with
demo cameras, use them; otherwise encode RGB from the LeRobot dataset. Either
path must use the checkpoint normalizer.

### 5.6 Training loop

Single `LiberoEnv`, `batch_size=1`, same camera/action mapping as eval.

Each episode: sample `task_id` uniformly from 0–9; reset; run chunks until
success, terminate, or 520 policy steps.

Per chunk: decode π_pre with fixed video noise and **K=4** batched action
noises → four `a_base`; residual each; execute `argmax_k Q(s, a_k)`; step the
env for 12 or 16 actions; store the chosen `(s, z, a_base, a)`.

After each collected **online** chunk: 10 critic updates and 1 actor update
from a mixed RLPD batch (batch size 256, or the whole buffer if smaller).

**Online budget: 100,000 environment action steps** (not chunks). That is ~280
episodes at the SFT eval mean length (~350 steps). On Modal, keep one rolling
full trainer state at `resume/latest.pt` (MLP weights, Adam state, replay, RNG,
env-step counter) so a job can resume. **Never write the frozen transformer,
VAE, or UMT5 into the RL run volume** — those stay on `dice-lingbot-sft-runs`
/ `dice-lingbot-sft-cache`. Milestone inference dumps are residual + critic
state dicts only (a few tens of MB, not 10 GB).

If wall-clock hits the Modal timeout, resume continues from `latest.pt`. The
recipe (sampler, β, n, checkpoint step) is immutable across resume.

### 5.7 Evaluation

| When | Protocol |
| --- | --- |
| Train-time | 1 rollout/task (10 episodes) at env steps 0, 25k, 50k, 75k, 100k. Same sampler and cameras. Init-state index 0, recorded seeds. |
| Comparison to 69% | After the run (or on demand): **20 rollouts/task**, init-state IDs **1–20**, seed 42 — identical plan to `EvalConfig(stage="eval")`. |
| Paper-scale | 100 rollouts/task — **out of this spec**; do not put it in the training loop. |

Train-time eval is smoke-scale so a 20-rollout job cannot dominate a 100k-step
run (~4.5 h). The 20-rollout job is the number that may be plotted next to 69%.

## 6. Logging (W&B)

Project: `dice-lingbot-va-rl`. Entity optional via `--wandb-entity`. Rank/process
is one GPU; log every optimizer step from that process.

Required scalars: `env_steps`, `chunks`, `actor_loss`, `critic_loss`,
`residual_rms`, `q_mean`, `q_min`, `bc_filter_rate`, `expert_ratio`,
`episode_return`, `episode_success`, `episode_length`.

**Sharpening check (ΔH vs ΔV), paper §5.3:** on demo-anchor states, sample
several action chunks from π_pre and from π_pre+residual (different action
noises, **same frozen video noise**), then log (i) `delta_v = Q(s,a) −
Q(s,a_base)` and (ii) `delta_h` = drop in per-coordinate histogram entropy of
the action chunk vs the prior. Residual RMS is logged separately; it is **not**
ΔH. Full K-sample ΔH is expensive on LingBot; compute it on a small demo-anchor
batch at train-time eval points, not every optimizer step.

Per-task success counters on train-time eval. No secrets in logs, configs, or
artifacts.

## 7. Modal

Reuse the eval image pattern (`script/lingbot_eval_modal.py`): Debian 3.12,
pinned LeRobot revision `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`, extras
`lingbot_va` + `libero` + `evaluation`, `MUJOCO_GL=egl`, PyPI index URL as in
eval (locked uv).

| Resource | Role |
| --- | --- |
| `dice-lingbot-sft-cache` | RO model/LIBERO assets |
| `dice-lingbot-sft-runs` | RO step-600 checkpoint |
| `dice-lingbot-rl-runs` | RW checkpoints, replay dump, expert features, logs |
| `dice-lingbot-hf` | CPU/GPU prep Hub token only |
| `dice-lingbot-wandb` | `WANDB_API_KEY` on the GPU function |

GPU: **one H100**. Eval already peaked ~13.7 GB allocated; transformer stays
inference; UMT5 stays on CPU; MLPs are small. If smoke OOMs, fail explicitly
(do not silently drop ensemble size).

Timeout: **12 hours**, retries 0, `max_containers=1`, persistent run lock
(same discipline as SFT/eval: no automatic stale-lock takeover).

Stages: `prepare` (assets + expert features if not cached), `train`, `eval`
(20-rollout comparison). After a successful `train` or `eval` stage the local
entrypoint **downloads** inference artifacts only (no overwrite) to
`result/lingbot-rl/<run-name>/`: residual actor + critic ensemble weights
(`residual.pt` / equivalent), `summary.json`, train-time eval rows, and —
after `--stage eval` — the 20-rollout comparison (`summary.json`, per-episode
JSON, `report.html`). Do **not** download `resume/latest.pt`, the replay
buffer, Adam state, expert feature cache, or the 5B transformer. Those stay on
Modal, same pattern as SFT (weights-only local copy). Keep the client attached
until that download finishes. Use `--download-dir` to change the local parent.

Local CLI: `modal run -m script.lingbot_rl_modal --stage train --run-name libero30-dice-baseline`.

No region pin. Do not launch paid jobs from local tests.

## 8. Tests (local, no GPU, no Modal `.remote()`)

`tests/test_lingbot_rl.py` in the eval venv:

- Config rejects sampler drift (video steps ≠ 20, action steps ≠ 50,
  `video_exec_step` ≠ −1, checkpoint step ≠ 600 unless overridden in tests).
- Residual masking zeros DoF 7–29; denorm matches `decode_action`.
- First-chunk execute length 12, later 16.
- Mean-pool of fake video+text tokens has shape (B, 3072) and depends on text
  (different prompt embeds → different `s`).
- n-step chunk returns: sparse terminal 1 discounted over 3 chunks.
- RLPD mix respects the scheduled expert ratio.
- Actor/critic one-step update on random tensors (no LingBot weights).
- Modal helpers: mocked argv, no secret printing.

Cloud smoke (operator-launched, not CI): a few chunks on H100, W&B step appears,
no OOM, `a_base` finite, residual finite.

## 9. Out of scope until this baseline exists

Multi-env rollouts, LingBot paper s=0.6 video early-exit, 10 action Euler steps,
KV-cache critic, DSRL, 100-rollout eval, β/n sweeps, multi-GPU.

The DICE-RL + π0 LIBERO-10 curve (Appendix A, Fig. 11) is a **shape reference**,
not a claim that this LingBot residual matches π0 architecture or demo IDs.

## 10. Vanilla DICE-RL vs this LingBot port

Checked against [DICE-RL v1](https://arxiv.org/abs/2603.10263) Algorithm 1,
Appendix A (π0 / LIBERO-10), and Appendix C Table 2 (Robomimic defaults).
LIBERO-Long is closer to Transport/Tool Hang than to Can, so we take β=100,
n=3, ensemble 10, ε=−0.5, UTD 10 from that long-horizon column, not Can’s
β=50.

| Mechanism | Paper vanilla | This spec |
| --- | --- | --- |
| Frozen π_pre; residual only | Yes | Yes |
| π_pre(s,z) deterministic given z | Action flow only | Action flow; **video noise also frozen per chunk** |
| Residual input | s_θ(s, **z**) Eq. (2) | Same: condition on `z`, add to `a_base` |
| Critic state | Frozen encoder; π0 = mean-pool image+text tokens → 2048 | Mean-pool LingBot video-stream + text tokens → 3072; **not** KV |
| Chunk critic + h-step backup | Eq. (4) | One AR chunk = 12 or 16 env actions |
| n-step | 3 (5 on Tool Hang) | 3 **chunks** |
| Ensemble min, GELU [1024]³, Adam 1e-4, τ=0.01 | Table 2 | Same |
| β, ε, BC filter Eq. (6)–(8) | β=100, ε=−0.5 on long tasks | Same, filter on |
| RLPD | 0.5→0.1 (Square/Tool Hang) | 0.5→0.1 over the 100k-step run |
| Multi-sample K / best-of-K | K=16 train + collection | **K=4 at collection** (batched action Euler); train uses stored `(s,z,a_base)` |
| Parallel envs | 4–8 | **1 env**; K action noises batched on one GPU |
| Online budget | LIBERO Fig. 10/11 plotted vs env steps (~100k scale) | 100k env actions |
| Eval | 100 rollouts/task (Fig. 10) | 20/task to match our SFT number; 100/task later |
| Sampler pin | Robomimic 10 action Euler; π0 native action denoise | LingBot **released** 20 video / 50 action, full video denoise |
| DSRL | LIBERO-10 flow-policy baseline only | Out of scope |

π0 vs LingBot (Appendix A vs this model):

- π0: one VLM forward (no denoise) → pooled s; then **action-only** iterative
  denoise → a_base. Residual never enters the VLM.
- LingBot: VAE(obs) → **video** Euler (20) → KV; **action** Euler (50) →
  a_base. Residual never enters the 5B transformer. Extra video loop is why
  collection is slower than π0 and why we freeze video noise when resampling
  action z.
- π0 text lives in the same token sequence as images. LingBot text is
  cross-attention; pooling both token sets is the analogue so Q is not
  task-blind.
- π0 has no AR video KV. LingBot does; phase-1 critic ignores it (known
  LIBERO-Long limitation).

What we will **not** copy from the LingBot paper into this RL run: 3-step
video to s=0.6, 10-step action Euler, noisy-history augmentation at inference.
Those are a different decoder than
`libero30-sft-step000600-eval` (`video_exec_step=-1`, 20/50).
