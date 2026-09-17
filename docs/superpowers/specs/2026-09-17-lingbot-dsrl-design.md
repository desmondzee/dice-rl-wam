# LingBot-VA DSRL (latent-noise steering) vs DICE-RL — design

Date: 2026-09-17
Status: draft for review

Repository: `dice-rl-wam`. Compute: personal HPC (Slurm assumed), not Modal.

Run DSRL ([Wagenmaker et al., arXiv 2506.15799](https://arxiv.org/abs/2506.15799))
on the exact LIBERO-10 LingBot-VA prior that the DICE-RL residual runs used, and
compare the two on sample efficiency, stability, and final 200-episode success.
Everything except the RL algorithm is held fixed.

## 1. Question and hypothesis

**Question.** For a frozen video-action world model (LingBot-VA, step-600
SFT, 69% macro on LIBERO-10), is RL over the **input action noise** (DSRL) more
sample-efficient and more stable than a **residual on the decoded action**
(DICE-RL), at the same 100k env-step budget?

**Hypotheses, stated so they can fail.**

- H1 (efficiency): at 50k env steps, where DICE-RL v2 regressed to 63.0%,
  DSRL beats the SFT 200-episode macro rate by a margin significant at n=200
  (≥ +9pp, i.e. ≥ 156/200; see §7.4).
- H2 (stability): DSRL's train-time success curve never drops below the SFT
  rate after warm-up, whereas DICE-RL v2 dipped at 50k.
- H3 (mechanism): steering only changes *which* prior sample is decoded, so
  DSRL cannot leave the prior's support. Tasks where the prior almost never
  succeeds (task 8, 2/20) will not improve under either method.

H1 fails if DSRL at 50k is below 156/200. If DSRL lands within ±6.4pp of
69% at both 50k and 100k, the result is "no detectable change".

## 2. What the DICE-RL v2 evidence actually says

From `dice-rl-eval-results.csv` (200 episodes, 20/task, IDs 1–20, seed 42):

| checkpoint | macro | Δ vs SFT | z (two-proportion) |
| --- | --- | --- | --- |
| SFT step 600 | 138/200 = 69.0% | — | — |
| DICE-RL v2 @50k | 126/200 = 63.0% | −6.0pp | −1.27 (p≈0.20) |
| DICE-RL v2 @100k | 139/200 = 69.5% | +0.5pp | +0.11 |

Per task, no change is significant under Fisher's exact test. The closest is
task 4 at 50k (13→6/20, p=0.056). At n=200 the 95% CI on a ~69% rate is
±6.4pp. **The 50k "regression" is suggestive but not established.** Detecting
+10pp with 80% power needs ~300 episodes per arm. This drives the eval design
in §7.

W&B (`libero30-dice-v2`): `residual_rms` spikes to ~0.035 then settles at
~0.017. `q_mean` overshoots to ~1.7 and then flattens at ~0.8. `critic_loss`
collapses to ~0 within a few hundred updates. `delta_v` peaks at +0.028 and
turns negative. `delta_h` goes to −0.13, so action entropy *increased* and the
residual broadened the distribution instead of sharpening it. A near-zero
critic loss with a flat `q_mean` is consistent with the critic fitting stored
targets that barely depend on the actor. That is the concern in §3.

## 3. Fresh latents: what DICE-RL needs, and why DSRL mostly avoids it

DICE-RL's objective (Eq. 4–5) is an expectation over **fresh** `z ~ N(0,I)`
drawn through `π_pre(s, z)` at every gradient step, for both the critic target
at `s'` and the actor objective at `s`. The LingBot port replaces this with
K=4 `(z, a_base)` pairs stored per chunk. Two consequences follow:

1. **Coverage, not staleness.** `a_base = π_pre(s, z)` is exact forever
   because the prior is frozen. But each state only ever sees the same 4
   noises. The actor is trained to make `s_θ(s, z_k)` good for those 4 points
   and is unconstrained elsewhere in z-space. At deployment it gets new z,
   which is extrapolation. The contraction DICE-RL relies on (pull *all*
   prior samples toward the high-value mode) needs the residual to see the
   whole noise distribution at each state. K=4 fixed points cannot measure or
   enforce that, which matches the flat `delta_h`.
2. **Noise-input distractors.** LeRobot's `_prepare_latent_input` zeroes
   the noisy action input on channels 7–29 at every Euler step
   (`noisy_latents[:, ~action_mask] *= 0`). Only 7 × 16 = 112 of the 480 z
   dims influence `a_base`. The stored `z` in replay is the *raw* pre-mask
   noise, so the residual MLP receives 368 dims of pure noise as input. This
   is a small, independent DICE-port fix: mask `z` before the actor.

Cheap mitigations if DICE-RL is kept (not in this spec's scope):

- Raise stored K to 16 per chunk, one batched action Euler (decode cost is
  dominated by the 20×CFG video loop, not the action batch).
- Add a z-agnostic regulariser: at update time, also evaluate the actor on
  fresh `z ~ N(0,I)` with `a_base` approximated by the nearest stored
  candidate. This is cheap and approximate.

**DSRL-SAC never calls the prior inside the update loop.** The replay
transition is `(s, w, r, s')` in the latent-action MDP. The prior is part of
the environment, so there is nothing to go stale. The only prior calls are
at collection (1 decode per chunk). **DSRL-NA** does need prior samples to
distil `Q^A(s, π(s,w))` into `Q^W(s,w)`. Its distillation targets
`(s, w_k, a_k)` are exact and never stale, because the prior is frozen and
`Q^W` generalises across states. So the stored-K trick is much less harmful
there than in DICE-RL's actor objective.

## 4. DSRL recap and variant choice

DSRL treats `π_dp^W(s, w)` (deterministic given noise `w`) as part of the
environment and runs RL over `w`. Two instantiations:

| | DSRL-SAC | DSRL-NA (Alg. 1) |
| --- | --- | --- |
| Critic | `Q^W(s, w)`, TD in latent MDP | `Q^A(s, a)` by TD + `Q^W` distilled from `Q^A(s, π_dp(s, w))`, `w ~ N(0,I)` |
| Offline A-space data (demos) | cannot use | can use |
| Prior calls in update loop | none | yes (distillation) |
| Used for | Robomimic Transport; **π0 on LIBERO / Aloha / real** | Robomimic Lift/Can/Square, OGBench |
| Sample efficiency | ≈2× slower than NA on Square | best |

**Phase 1 runs DSRL-SAC.** It is the variant the paper used for the only
large pretrained VLA (π0), including a LIBERO task (20% → ~100% in ~10k
samples, single task). It needs no prior calls during updates, so collection
is the only expensive part. DSRL-NA is Phase 3 (§9).

## 5. Mapping DSRL onto LingBot-VA

### 5.1 Frozen prior

Identical to `2026-09-14-lingbot-dice-rl-design.md` §4: `libero30-sft`
step 600, released LIBERO sampler (20 video Euler steps, `video_exec_step=-1`,
50 action Euler steps, CFG 5.0/1.0, SNR shift 5.0/0.05), 128×128 agentview +
wrist width-concat, 7-DoF in channels 0–6, checkpoint `norm_stats.json`,
bf16, SDPA. Reuse `load_residual_policy` and `frozen_prior_kwargs()`
unchanged.

### 5.2 Two noise sources

LingBot draws **two** noises per chunk:

| noise | shape | effective dims | effect |
| --- | --- | --- | --- |
| video `ε_v` | `(1, 48, 4, 8, 16)` | 24,576 (frame 0 overwritten on first chunk) | imagined future frames → KV cache → actions |
| action `ε_a` | `(1, 30, 4, 4, 1)` | 7 × 16 = 112 (channels 7–29 zeroed each step; frame 0 overwritten on first chunk) | action flow ODE |

π0 has only the action noise. DICE-RL's port perturbs only the action side and
lets video noise stay i.i.d. per chunk.

**Phase 1 steers `ε_a` only. `ε_v ~ N(0,I)` is i.i.d. per chunk, exactly as
in the prior.** The latent-action MDP is then stochastic in `ε_v`, which SAC
handles. This matches the DICE port's "freeze video, act on the action decode"
translation, so the comparison isolates *residual-on-a* vs *steer-ε_a*.

**Risk: action noise may have little leverage.** With `action_snr_shift=0.05`
most of the 50-step schedule sits at low noise, and the video KV may pin the
action mode. §8's Phase 0 probe measures what fraction of action-chunk
variance comes from `ε_a` versus `ε_v` *before* anything is trained. If `ε_a`
explains little, Phase 1 changes to the video-steering design in §9 A3.

### 5.3 Latent action space

Following DSRL-π0 (32-d noise tiled across a 50-step chunk):

- **Latent action `w ∈ [−b_W, b_W]^7`**, one value per used channel.
- Tiled across all 16 action tokens (4 frames × 4 actions/frame) into channels
  0–6. Channels 7–29 get i.i.d. N(0,1), which has no effect since they are
  masked, but keeps the tensor distributionally identical to the prior.
- `b_W = 1.0` (DSRL-π0 LIBERO). The probe checks the steering range at
  `b_W ∈ {1, 1.5, 2.5}`.
- Ablation A1: untied `w ∈ R^{112}` (7 × 16).

Tied noise is a strong prior. A single 7-vector moves the whole chunk
coherently in noise space. DSRL reports this is "expressive enough" for π0.

### 5.4 Critic / actor state

The same `s ∈ R^{3072}` as the DICE port: one action-free video-stream pass on
the real start-of-chunk observation, pre-`proj_out` tokens concatenated with
projected text tokens, mean-pooled. `ResidualLingBotPolicy._pool_from_latent`
already computes this at collection. DSRL-π0 used a 64×64 CNN + proprio in
sim and VLM last-token features in the real world. Keeping the pooled LingBot
feature is the closer analogue to the latter and keeps the comparison clean.
Ablation A4 adds the 8-d proprio state.

### 5.5 One RL step

One transition is one AR chunk (12 env actions on the first chunk, 16 after),
exactly as DICE. Reward is 1 on the chunk that first sees `info["is_success"]`,
else 0, with `done` at success/termination/520 steps. The discount per chunk is
`γ^{n_env}` with `γ = 0.999` per env step (DSRL-π0 LIBERO), i.e.
0.984 for a full chunk. DICE uses 0.99 per chunk. The two are within 1% of each
other, and each method keeps its own paper value.

Ablation A2: DSRL-π0's `−1 per step until success, 0 at success` reward.

### 5.6 Collection

Per chunk:

1. `s ← pool(real start-of-chunk obs)` (critic cache pass).
2. **Warm-up (first 500 chunks ≈ 8k env steps):** `w ~ N(0,I)`, unclipped. This
   is exactly the SFT policy. Afterwards: `w ~ π_φ(·|s)` (tanh-Gaussian scaled
   by `b_W`).
3. Decode one candidate: `_infer(..., action_noise=tile(w), k=1)` → `a`.
4. Execute, `commit_executed(a)`, step env, store `(s, w, r, done, s', n_env)`.
5. After each online chunk past warm-up: 20 critic + 20 actor/temperature
   gradient steps on fresh 256 minibatches.

No best-of-N at collection, and no expert demos (DSRL-SAC cannot consume
A-space actions). Warm-up rollouts from the prior play the role DSRL-π0's
"initial rollouts" play. See §7.2 for the fairness caveat.

### 5.7 Networks and optimisation (pinned)

| item | DSRL-LingBot | source |
| --- | --- | --- |
| Actor | tanh-Gaussian, learned diag std, MLP [1024]×3, LayerNorm, GELU; input `s`; output 7-d | π0-real hidden 1024 with backbone features; DSRL App. B (LayerNorm, large nets) |
| Critic | 10 Q heads, same MLP, input `(s, w)`, **mean** reduction for target and actor | π0-LIBERO: `num_qs=10`, `critic_reduction='mean'` |
| Target critic | Polyak τ = 0.005 | Table 11 |
| LR | actor 1e-4, critic 3e-4, temperature 3e-4, Adam | Table 11 / `launch_train_sim.py` |
| Target entropy | −d/2 = −3.5 | Table 11 (`−d/2`) |
| Batch | 256 | Table 11 |
| UTD | 20 gradient steps per collected chunk | Table 11 (`multi_grad_step 20`) |
| `b_W` | 1.0 | Table 11 (LIBERO) |
| Warm-up | 500 chunks with `w ~ N(0,I)` | `start_online_updates 500` |
| Replay | 100k chunks (all online) | = DICE capacity |
| Budget | **100,000 env action steps, warm-up included** | = DICE |

### 5.8 Evaluation policy

Deterministic `w = b_W · tanh(μ_φ(s))`, `ε_v ~ N(0,I)`, one decode.
Secondary eval (matches DICE v2, which ranks 4 candidates by its critic):
sample 4 `w ~ π_φ`, batched action Euler after one shared video decode, execute
`argmax_k Q^W(s, w_k)`. Both use the pinned 200-episode protocol.

## 6. What is identical vs different

| | DICE-RL v2 (existing) | DSRL (this spec) |
| --- | --- | --- |
| Prior, sampler, cameras, norm, horizon, env wrapper | pinned | **same code path** |
| Critic state `s` | 3072-d pooled | **same** |
| Chunk transition, success reward, done | yes | **same** (ablation A2) |
| Budget | 100k env steps, 1 env, uniform task sampling | **same** |
| Train-time / final eval protocol | 1 ep/task every 25k; 200-ep at 50k, 100k | **extended** for both (§7.3) |
| What is learned | residual `s_θ(s,z)` on `a` + chunk critic `Q(s,a)` | noise policy `π(w|s)` + `Q^W(s,w)` |
| Prior calls per chunk (collection) | 1 video + K=4 batched action | 1 video + 1 action |
| Prior calls in updates | none (stored K=4) | none |
| Offline data | 300 SFT demos via RLPD 0.5→0.1 | none; 8k env-step prior warm-up |
| Exploration | best-of-4 by Q | SAC entropy |
| Discount | 0.99 / chunk | 0.999 / env step |
| Hyperparameters | DICE-RL Table 2 long-horizon column | DSRL Table 11 LIBERO column |

## 7. Comparison protocol

### 7.1 Primary metrics

1. **Final success**: 200-episode pinned protocol (IDs 1–20, seed 42) at 50k
   and 100k env steps, matching the CSV columns.
2. **Sample efficiency**: success vs env steps from the train-time curve
   (§7.3), plus area under that curve over [0, 100k].
3. **Stability**: minimum post-warm-up train-time success, and the number of
   eval points below SFT.
4. **Compute**: wall-clock and GPU-hours to 100k, split into collection,
   updates, and evals (logged timers).

### 7.2 Fairness caveats (report, do not hide)

- DICE had 300 demonstrations in replay. DSRL-SAC has none, which is a
  handicap for DSRL. Phase 3 DSRL-NA closes this gap.
- DSRL's warm-up spends 8% of the budget without learning, and it is counted
  in the budget.
- DICE's collection decodes 4 action candidates, DSRL's decodes 1. Report
  per-env-step wall-clock rather than assuming equal cost.
- Each method uses its own paper hyperparameters. Neither is tuned on the eval
  set.

### 7.3 Eval schedule (both methods, rerun DICE if affordable)

The DICE runs' train-time eval (1 episode/task = 10 episodes) cannot
distinguish 60% from 80%. For DSRL:

- **Every 10k env steps:** 3 episodes/task (30 episodes), fixed init states
  {0, 48, 49}, seed 42. These are disjoint from the final-eval states 1–40. Stored per point, with `residual.pt`-equivalent weights.
- **At 25k, 50k, 75k, 100k:** full 200-episode protocol, run as a separate
  Slurm job from the saved weights so it never blocks training.

### 7.4 Statistics

- Two-proportion z / Fisher exact per checkpoint against SFT 138/200 and
  between methods at equal env steps. Significance needs ≈ ±9pp at n=200.
- Episodes share init-state IDs across methods, so McNemar on
  init-state-paired outcomes is a valid secondary test. Pairing only removes
  init-state variance, not sampler noise.
- The minimum effect detectable with 80% power at n=200 per arm (α=0.05) is
  ≈ +12pp from 69%. To claim a smaller gap, extend the final checkpoint to
  40 episodes/task (400, init states 1–40; LIBERO-10 has 50 per task) for
  SFT, DICE, and DSRL alike. At n=400 the significance threshold is ≈ ±6pp.
- **Seeds:** at least 2 DSRL seeds (42, 43). One seed per method cannot
  separate method from run variance. If budget allows, one extra DICE v2 seed.

## 8. Phase 0: steerability probe (before training)

A one-GPU script. No RL, and it produces a JSON report. It decides §5.2/§5.3
before any money is spent.

For 10 tasks × 5 start states (demo anchors or env resets) × first and a later
chunk:

1. **Variance decomposition.** Sample a 4×4 grid (4 `ε_v` × 4 `ε_a`), decode
   16 chunks, and compute per-dim variance of used channels. Report
   `Var_between(ε_a | ε_v)` / total and `Var_between(ε_v)` / total.
2. **Tied vs untied reach.** For `b_W ∈ {1, 1.5, 2.5}` and 32 random tied `w`
   (7-d), measure spread of `a` versus 32 untied (112-d) draws of the same
   norm.
3. **Mode coverage (cheap H3 check).** On tasks 4, 8, and 9, execute 8 rollouts
   with fixed random tied `w` per episode and record success. If any fixed `w`
   succeeds more often than the prior, steering has something to find.
4. **Cost.** Per-chunk wall-clock for `k=1` and `k=4`, and peak memory. This
   sizes the Slurm time limits.

**Decision rule.**

- If `ε_a` explains ≥ 20% of action variance, run Phase 1 as specified.
- If it explains < 5%, switch to A3, joint video steering (§9).
- In between, run both.

## 9. Ablations and follow-ups (Phase 3, only after Phase 1 reports)

- **A1** untied 112-d `w`.
- **A2** −1/0 per-step reward.
- **A3 video steering**: additionally steer `ε_v` through a per-channel 48-d
  tied offset, `ε_v = √(1−ρ²)·ξ + ρ·tile(u)` with `ξ ~ N(0,I)` and a fixed
  `ρ=0.5`. The actor outputs `u ∈ [−b_W, b_W]^{48}`, so the noise stays
  near-Gaussian marginally. It is OOD-risky for a video model, so probe
  first.
- **A4** state = pooled ⊕ 8-d proprio.
- **DSRL-NA**: `Q^A(s, a)` TD-trained on online + 300 demo chunks (reuses
  `lingbot_rl_data.featurize_experts` rows). `Q^W` is distilled on stored
  `(s, w_k, a_k)`, K=4 per chunk from the same batched decode DICE uses, with
  10 `Q^W` steps per update. This gives DSRL the demos and makes the
  comparison symmetric.
- **DICE-RL fix**: mask `z` channels 7–29 before the residual MLP (§3.2) and
  store K=16. Report it as DICE v3 if run.

## 10. HPC port (replaces Modal)

- **Environment.** Python 3.12, LeRobot checkout at
  `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e` with `uv sync --locked --extra lingbot_va --extra libero --extra evaluation`
  (the Modal image recipe, minus Modal). Optionally an Apptainer image built
  from the same commands. `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
  `MUJOCO_EGL_DEVICE_ID` set from the allocated GPU, and
  `LEROBOT_SOURCE_ROOT` pointing at the checkout.
- **Assets** (login node with internet, once):
  - `robbyant/lingbot-va-base@68b7bc1b` (vae, text_encoder, tokenizer)
  - `lerobot/libero-assets@0b3ea86b`
  - SFT checkpoint `libero30-sft/checkpoints/step_000600` from Modal volume
    `dice-lingbot-sft-runs`, via `modal volume get`
  - `script.lingbot_eval prepare` writes `prepared.json` under `$CACHE_ROOT`

  Compute nodes run with `HF_HUB_OFFLINE=1`.
- **Jobs.** 1 GPU (≥ 40 GB assumed; probe measures K=4 peak), 16 CPU, 64–96 GB
  RAM. Training resumes from `resume/latest.pt`, and a job array or
  `--dependency=afterany` chain continues past wall-time limits. Resume
  already restores RNG, eval schedule, and W&B id in the DICE trainer, and the
  DSRL trainer copies that.
- **Locks.** A file lock (`O_EXCL` create of `<run>/LOCK`) replaces
  `modal.Dict`, with no automatic stale takeover (same discipline).
- **Secrets.** `WANDB_API_KEY` from the environment or `wandb login` on the
  cluster. `HF_TOKEN` only on the prepare step. Never in argv or logs.

## 11. Risks

| risk | signal | response |
| --- | --- | --- |
| `ε_a` has little leverage (§5.2) | Phase 0 variance share < 5% | A3 video steering |
| Tied 7-d `w` too restrictive | train success plateaus at SFT; probe reach small | A1 untied |
| Multi-task single actor (DSRL-π0 was single-task) | per-task curves diverge; some tasks regress | per-task analysis; task-id one-hot into actor/critic |
| Sparse reward + no demos → slow critic | `Q^W` flat at 0 after warm-up | A2 reward; DSRL-NA with demos |
| `b_W=1` clips the prior's typical noise | post-warm-up success drops below warm-up rate at step 0 of learning | probe `b_W`; raise to 1.5 |
| Eval noise hides effects | CIs overlap | 500-episode final eval, 2 seeds |
| EGL/MuJoCo on HPC nodes | env reset fails | smoke job on a GPU node before prepare/train |

## 12. Non-goals

- Changing the SFT checkpoint, sampler, or eval harness provenance.
- Multi-env rollouts, multi-GPU training.
- Modal launchers for DSRL (HPC only; Modal code untouched).
- Editing `.cache/lerobot`, `.cache/lingbot-va`, the Hydra robomimic stack, or
  any `lingbot_rl_*` DICE semantics (DSRL reuses them by import only, plus one
  behaviour-preserving split in the policy wrapper).
