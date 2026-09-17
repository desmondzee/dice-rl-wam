# DICE-RL on LingBot-VA: configurations and results of all four evaluations

Reference: DICE-RL paper (arXiv 2603.10263v2). All evaluations use the same pinned protocol:
LIBERO-10 (LIBERO-Long), 20 episodes/task × 10 tasks = 200 episodes, init-state offset 1, seed 42,
max 520 env steps/episode, 128×128 two-camera pixels, relative control at 20 Hz.

## 1. Results

### Macro success (200 episodes each)

| eval | checkpoint | macro | vs SFT |
|---|---|---|---|
| SFT baseline | `libero30-sft/step_000600` | **69.0%** (138/200) | — |
| v1 RL, endpoint | `libero30-dice-baseline` @ 100k env steps | 69.0% (138/200) | ±0 |
| v2 RL, mid-run | `libero30-dice-v2` @ 50,454 env steps | **63.0%** (126/200) | **−6.0** |
| v2 RL, endpoint | `libero30-dice-v2` @ 100k env steps | 69.5% (139/200) | +0.5 |

200-episode 95% CI ≈ ±6.5pp on the macro rate; per-task (n=20) ≈ ±20pp.

### Per-task successes /20

| task | description | SFT | v1@100k | v2@50k | v2@100k |
|---|---|---|---|---|---|
| 0 | soup + sauce in basket | 12 | 16 | 12 | 11 |
| 1 | cream cheese + butter in basket | 16 | 19 | 16 | 19 |
| 2 | stove on + moka pot | 17 | 16 | 14 | 19 |
| 3 | bowl in drawer + close | 17 | 18 | **20** | 18 |
| 4 | two mugs on plates | 13 | 13 | **6** | 12 |
| 5 | book in caddy | 18 | 19 | 18 | 18 |
| 6 | mug on plate + pudding | 16 | 14 | 15 | 18 |
| 7 | soup + cream cheese in basket | 18 | 15 | 17 | 15 |
| 8 | both moka pots on stove | 2 | 0 | 1 | 2 |
| 9 | mug in microwave + close | 9 | 8 | 7 | 7 |

### Train-time evals during RL (1 episode/task, fixed seeds — noisy, ±1–2)

| env steps | 0 | ~25k | ~50k | ~75k | 100k |
|---|---|---|---|---|---|
| v1 successes /10 | 1 | 7 | 9 | 9 | 5 |
| v2 successes /10 | 5 | 5 | 8 | 6 | 5 |

v1's step-0 = 1/10 reflects its random-init residual corrupting the initial policy; v2's step-0 = 5/10 ≈ prior.
The v2@50k 8/10 train-eval did not survive the 200-episode protocol (63.0%): single-episode
train-evals are unreliable for checkpoint selection.

## 2. Common stack (all runs)

### Base model and SFT prior

| variable | value |
|---|---|
| backbone | LingBot-VA (`robbyant/lingbot-va-base`), ~5B video-action world model (WAM), AR video context via KV cache |
| paper's counterpart | flow-matching BC 1D U-Net (Robomimic) / π₀ VLA (LIBERO appendix) |
| SFT data | 30 demos/task × 10 LIBERO-10 tasks (300 total), precomputed VAE latents |
| SFT training | full transformer, effective batch 80, ≤1000 optimizer updates; **checkpoint step 600** pinned (no checkpoint sweep) |
| frozen during RL | entire transformer, VAE, text encoder |

### Inference sampler (identical in SFT eval and all RL runs — "the 69% sampler")

| variable | value | LingBot vanilla? |
|---|---|---|
| video denoise steps | 20, guidance 5.0, snr_shift 5.0 | released defaults |
| action denoise steps | 50, guidance 1.0, action_snr_shift 0.05 | released defaults |
| video_exec_step | −1 (full video decode each chunk) | released default |
| action chunk | frame_chunk 4 × action_per_frame 4 = 16 env actions (12 on first chunk) | native |
| action space | 30-dim padded, 7 DOF used, normalized q01/q99 → [−1,1] | native |
| attention | torch backend, window 30, bf16 | native |
| cameras | 2 × 128×128, width-concat, native vertical flip | pinned to SFT eval |
| text encoder | CPU | pinned |

### RL-specific machinery (both v1 and v2)

| variable | value | paper |
|---|---|---|
| critic state `s` | 3072-d mean-pool of pre-proj_out video tokens + text tokens, single isolated frame encode, frozen | π₀ variant: 2048-d pooled transformer latent, frozen (same idea) |
| residual actor | MLP 1024×1024×1024, GELU+LayerNorm, input (s, z flattened), output 16×30 chunk residual | same widths, GELU |
| critic | ensemble 10 × MLP 1024³ on (s, action chunk), min-ensemble | ensemble 10, min |
| reward | sparse terminal 1.0 at first success, per-chunk, γ=0.99 per chunk | same |
| n-step | 3 chunks | 3–5 per task |
| RLPD expert data | the same 300 SFT demos, featurized; ratio 0.5 → 0.1 linear over full run | 0.5→0.1 over warm-start window |
| optimizer | Adam 1e-4, batch 256, τ=0.01, β=100, ε=−0.5, replay 100k chunks | same value ranges (β 50–100, ε −0.25..−0.75) |
| environments | 1 (sequential), random task per episode | 4–8 parallel (Robomimic); LIBERO random task/iteration |
| online budget | 100,000 env steps ≈ 6.4k chunks ≈ **~30 episodes/task** | LIBERO: ~1.1M env steps ≈ ~300+ episodes/task |

## 3. What differed between v1 and v2 (and vs the paper)

| variable | v1 (`libero30-dice-baseline`) | v2 (`libero30-dice-v2`) | DICE-RL paper |
|---|---|---|---|
| residual init | default Linear init (random ~0.4σ output) | **zero-init output layer** (starts at prior) | starts at prior (implied) |
| critic target (Eq. 4) | 1-sample SARSA: stored executed next chunk `a_next` | mean over **stored K=4** next-state candidates, residual recomputed with current actor | mean over **K=16 fresh** prior samples drawn in-loop |
| actor objective (Eq. 5) | single stored (z, a_base) | mean over stored K=4 candidate set | mean over K=16 fresh samples |
| BC filter anchor Ĝ (Eq. 6) | bootstrapped n-step TD target | **Monte-Carlo return-to-go** from episode finalize | Monte-Carlo return from replay |
| UTD=10 | 10 critic steps on ONE frozen batch+target | 10 × fresh minibatch + fresh target, +1 actor batch | fresh minibatches |
| collection | best-of-4, critic argmax | same (best-of-4) | best-of-16 |
| eval-time selection | none (k=1, single prior sample + residual) | **best-of-4, critic argmax** | best-of-N used during interaction |
| checkpoints kept | final only | every train-eval point | n/a (stable curves, endpoint reported) |
| recipe version | 1 | 2 | — |

### Remaining deviations from the paper after v2

| deviation | status / reason |
|---|---|
| K=4 not 16 | pinned for GPU memory of batched action denoise; paper ablation: 16 > 4 > 1 (mostly sample efficiency) |
| stored candidates, never redrawn | fresh in-loop sampling infeasible: prior needs the AR video KV context, unstorable per replay row (WAM tax). Per-state sampling error is fixed, not averaged away across updates |
| 1 env, 100k steps | ~1/10 the paper's LIBERO budget → ~30 online episodes/task |
| truncation at 520 = terminal (no bootstrap) | pre-existing; paper silent |
| single SFT checkpoint (step 600) | paper: intermediate BC checkpoints finetune best (GoodCov/BadEnt tradeoff, Fig. 4); never swept |

## 4. Reading of the results

- v1 (4 implementation bugs + endpoint-only eval): net zero. Fixed in v2.
- v2 trains *healthily* (calibrated Q, residual RMS ~0.017 ≈ half of v1, no q_min pathology) but the
  200-episode evals bracket the baseline: 63.0% at 50k, 69.5% at 100k.
- The 50k eval kills two hypotheses at once: "peak then fade" (the peak was train-eval noise) and
  "policy never leaves the prior" (task 3 → 20/20 while task 4 → 6/20 — the residual moves behavior
  strongly, in both directions). Diagnosis: the critic is decisive but unreliable per task at
  ~30 episodes/task, and best-of-4 selection amplifies its mistakes where it is miscalibrated.
- Paper comparison: on LIBERO the paper's early gains are modest (+3–5pp by ~100k steps of a 1.1M run)
  but its curves **never dip below the pretrained baseline**. Our 50k dip below baseline is a
  qualitative mismatch not explainable by budget alone; the paper's stabilizers we lack are fresh
  K=16 sampling (candidate-set staleness) and data density.
- Task 8 (prior 2/20) never moved in any run: contraction cannot create modes the prior does not
  sample — a prior-quality problem (SFT checkpoint/data), not an RL problem.

## 5. Candidate next experiments

| experiment | cost | what it discriminates |
|---|---|---|
| single-task RL (task 9, 45% prior), same recipe, 100k steps | ~11 H100-h + config change | paper's Robomimic regime (~10× per-task data): steep early gains ⇒ data starvation was binding; flat/dip ⇒ candidate staleness is binding |
| re-eval v2@50k task 4 with best-of-N off (k=1) | ~40 min | whether critic-argmax selection actively harms miscalibrated tasks |
| store 16 candidates at collection, subsample 4 fresh per gradient step | ~1.3× rollout cost | restores update-to-update sampling variation (closest feasible analog of fresh K=16) |
| 300k-step multitask run | ~35 H100-h | pure budget test at 3× per-task data |
| SFT checkpoint sweep (400/600/800) probed with GoodCov-style metric | cheap, offline | prior finetunability, esp. tasks 8/9 (paper Fig. 4 lever) |
