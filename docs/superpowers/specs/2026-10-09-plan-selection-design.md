# Plan-level selection for LingBot-VA DICE-RL

Status: v1 design, approved decisions, not implemented. Code base: main at d6e2a6b (`decode_plans` is the only
multi-plan primitive; sampler options `video_steps`, `action_steps`, `video_exec_step` in both configs).

## 1. Decisions

| item | decision |
|---|---|
| tasks, residual input, ε | tasks 9 and 0, one run each, `residual_input: base`, ε = −0.3 (action-branch filter only) |
| sampler | 5 video steps fully denoised, 10 action steps, CFG 5 / 1, shift 5 / 0.05: the authors' default (VA2 §2.4.1, `va_franka_cfg`) |
| plan candidates M | 4, fully denoised, batched through one transformer pass; no early exit, no partial decode |
| plan ranking | argmax over m of min-ensemble V_plan(s, p_m); V_plan trained by regression onto Q |
| action ranking | argmax over k of min-ensemble Q(s, p_m*, a_k) inside the chosen plan, K = 4, decoded as one batch of M × K |
| actor | residual actor on (s, a_base), zero-init, β = 100, Q-normalised TD3+BC, filter at ε; unchanged |
| budget | 320k env steps per run; checkpoints every 80k plus the end; train-evals at 0 / 80k / 240k; resume every 10 episodes; W&B as today |
| plan_random_prob | 0.1, constant, collection only, after `selection_warmup_steps` (64k) during which the plan is always random; evaluation executes the argmax plan |
| evaluation | the method only (M = 4, K = 4); the SFT re-bench is the sampler check |
| compute | the two GB10s (DGX Spark), one run each; no Modal |

## 2. Why

Given one video plan, the four action candidates differ by 0.033 (std, normalised units) against 0.44 across
states; the residual is 0.034 RMS. The action decoder is near-deterministic given the plan, so candidate
selection and the residual are substitutes (63 → 75 and 73 → 73 on task 9) and no action-level recipe
produced the paper's negative ΔV–ΔH trend. The policy's modes live in the plan, sampled once per chunk and
never resampled. Choosing among M sampled plans with a learned plan value is the contraction operator that
fits this model: it stays on the policy's own support and removes mass where the distribution has it. This
is best-of-N re-ranking with a learned value (V-GPS, Nakamoto et al. 2024) lifted from actions to plans,
with the value trained by regression onto the TD critic in the manner of DSRL's noise-space critic
(Wagenmaker et al. 2025).

## 3. Sampler facts

- A chunk is 4 latent frames = 16 env actions; the chunk latent is (48, 4, h, w).
- The port builds `FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)`: n steps are n denoising
  passes at the shifted levels below (σ = 1 is pure noise), the last of which lands on σ = 0, followed by one
  write-only pass at σ = 0 that puts the clean chunk into the cache. So "5 steps" is 6 transformer passes.

  | steps | σ of each denoising pass |
  |---|---|
  | 4 | 1.00, 0.94, 0.83, 0.63 |
  | 5 | 1.00, 0.95, 0.88, 0.77, 0.56 |
  | 6 | 1.00, 0.96, 0.91, 0.83, 0.71, 0.50 |
  | 20 | 1.00, 0.99, … 0.56 (pass 17), 0.47, 0.36, 0.21 |

- `video_exec_step = E` runs only the first E passes; pass E writes the cache from its input and still
  advances the latent, so the action stream reads history at the σ of pass E. Post-training noises the
  history to σ ∈ [0, 0.5] with probability 0.5, so cache levels σ ≤ 0.5 are in distribution. Under shift 5
  the cheapest in-range truncation is 6 steps / E = 6 (6 passes, cache at 0.50), the same count as 5 full
  steps, so partial denoising saves nothing and is dropped. The paper's "3 Euler steps to s = 0.6" belongs to
  an unshifted schedule the released checkpoint and configs do not use; every released config runs
  `video_exec_step = -1`.
- Re-bench result (task 9, 100 held-out episodes, the seeds of every result in `docs/heldout-results.md`):
  5 / 10 full = 60 / 100 (33 gained / 15 lost against 42 / 100 at 20 / 50); 5 steps / E = 4 / 10 (cache at
  σ = 0.77) = 39 / 100. The 5 / 10 sampler is adopted for training and every evaluation; the old residual
  results at 20 / 50 are not comparable to what follows. Tasks 0 and 8 get the same SFT re-bench.
- Episode wall time on a GB10 at 5 / 10, M = 1: about 65 s (520 policy steps) against 1213 s at 20 / 50.

## 4. Throughput

The GB10 is memory-bound on this model: a video pass streams the 10 GB of bf16 transformer weights at about
240 GB/s whatever the batch, so a CFG batch-2 pass takes about 0.35 s at 55 W with no power cap. Two
changes raise the arithmetic intensity; both are in v1:

1. **M plans in one pass.** The five video passes run at batch 2M = 8 (M conditional rows then M
   unconditional rows, the port's CFG order), reading the weights once for all plans. Expected cost about
   1.5× a batch-2 pass, so the four plans cost about 0.4× of four looped passes.
2. **M × K action decode in one pass.** The ten action steps run at batch M × K = 16 against a cache whose
   rows are each plan's conditional rows repeated K times (the existing `expand_conditional_kv` pattern,
   per plan), reading the video cache once for all candidates.

Everything else stays: Torch 2.11.0+cu130 (sm_121 kernels, installed by `brev/setup.sh` through
`DICE_TORCH_INDEX`), `F.scaled_dot_product_attention` in bf16 (`attn_mode: torch`), no FlashAttention
package, TensorRT, FP8, CUDA graphs or `torch.compile` (sequence length changes every chunk and the cache
is mutated in Python inside the forward, so compile would recompile or break graphs, and fused bf16 kernels
would void the bit-for-bit tests), no parallel environments (a recipe change). The trainer refuses to start
on an aarch64 host whose torch is not a CUDA 13 build, records the CUDA version and the SDPA flash flag in
`settings.json`, and logs `seconds_per_env_step` and the peak GPU memory with every update; the P0 smoke at
5 / 10 reads them at M = 4 and at M = 1. Working set at M = 4, K = 4: about 25 GB of KV cache (2M-row
history, its snapshot, and the M × K-row decode cache) on top of 10 GB of weights.

## 5. Policy at inference

Per chunk, with the history cache holding 2 rows (conditional, unconditional):

1. Snapshot the history. Repeat each cache row M times so the cache has 2M rows ordered [cond × M, uncond × M].
2. Draw M video noises (M, 48, 4, h, w). Run the 5 denoising passes and the clean write pass at batch 2M;
   the write pass puts each plan's chunk into its own rows, so the cache now holds 2M rows. Pool each denoised chunk latent
   (M, 48, 4, h, w) through the history-free critic pool (`_pool_from_latent`, one batch-M pass against
   the empty critic cache, the same function that produces s from the current frame) → p_m (M, 3072).
3. m* = argmax over m of min over heads j of V_plan_j(s, p_m), where s is the current-frame pool
   (`_pool_from_latent`, 3072-d) as today.
4. Action decode at evaluation: repeat plan m*'s conditional row K times into a 1 × K cache, draw K action
   noises, run the 10 action steps at batch K, restore. The K candidates cross-attend to the chosen plan
   only. At collection the decode instead covers all plans: an M × K cache (row m repeated K times, the same
   K noises for every plan), batch M × K, giving a_base (M, K, 16, 30); the rejected plans' candidates exist
   only as V_plan regression targets and for the logged metrics, never for execution.
5. Keep rows m* and M + m* of the written cache as the new 2-row history; drop the rest. Rank plan m*'s K
   candidates with Q(s, p_m*, a_k), apply the residual, execute. From here on identical to today.

Rejected plans leave no trace in the cache. `plan_candidates = 1` reproduces today's policy at the sampler
bit for bit (batch 2 passes, no row expansion), which is the regression test for the batched path together
with the looped `decode_plans` output for M = 2.

## 6. Policy at collection

As at inference with two differences: the executed plan is uniform over the M with probability 0.1 (always
during the first 64k env steps, when `eval_candidates` is also 1 as today); and all M × K candidates are
decoded and scored every chunk so V_plan has targets for the plans it rejected. Stored per chunk: s, the M
pools, the chosen index, the M × K base actions and the K action noises, the executed action and today's
fields.

Logged per chunk: plan_agreement (V_plan's argmax equals the argmax over m of max_k Q(s, p_m, a_mk)),
plan_regret (that maximum minus the executed plan's value), plan_value_spread (max − min of V_plan over
the M), plan_random_rate; per update bin: plan_critic_loss, v_plan_mean.

## 7. Critics and actor

**Q(s, p, a)**, the action critic: input current-frame pool s (3072) ‖ plan pool p (3072) ‖ action chunk
(480); 10 heads of [1024, 1024, 1024]; min-ensemble; Polyak target ensemble; TD with the multi-sample n-step
target as today, with the executed next chunk's pool p′ in the bootstrap. Input width grows from 3,552 to
6,624; hidden widths, optimiser, schedule, update cadence unchanged.

**V_plan(s, p)**, the video (plan) critic:

- Inputs s ‖ p (6,144); 10 heads of [1024, 1024, 1024] → scalar, same init as the critic heads.
- Target, per stored plan m of a row, computed under `no_grad`: a_mk = a_base_mk + actor(s, a_base_mk) for
  k = 1..K; q_mk = min over j of Q_j(s, p_m, a_mk); y_m = max over k of q_mk. Expert rows carry one plan
  (the true future's pool, §8) and one base action (the demo action), so y = min_j Q_j(s, p, a_demo +
  actor(s, a_demo)).
- Loss = mean over rows, plans and heads of (V_plan_j(s, p_m) − y_m)². Online rows contribute M plans each,
  expert rows one; the batch and expert ratio are the critic's.
- One V_plan step per critic step, on the same batch, after the critic update, using the live Q and the
  live actor (what collection-time selection uses). Adam 1e-4, wd 1e-5, clip 1.0, the critic's cosine
  schedule. No target network, no bootstrap through V_plan, no filter, no BC term.
- Use: collection and evaluation rank plans by min over heads (§5). V_plan weights are saved in every
  inference checkpoint; parameters, optimiser and schedule state live in `resume/latest.pt`.
- Nothing flows back: Q, the actor and the stored base actions supply numbers, not gradients; the actor is
  never trained to make plans look good to V_plan; Q stays the single TD-trained source of truth.

Why regression rather than TD: rewards exist only for executed plans, so a TD plan critic would train on its
own argmax choices and extrapolate to the plans it rejects; it would be a second bootstrapped critic with its
own drift; its targets arrive only after episode finalisation. The regression target exists for all M plans
at every update and inherits Q's pessimism.

**Residual actor**: unchanged, inputs (s, a_base), zero-init, β = 100, Q-normalised TD3+BC, filter at ε
anchored on Q(s, a_stored) against the Monte Carlo return, after `bc_filter_warmup_steps`.

## 8. Expert rows

Demos have no sampled plan but have the true future. For each demo chunk the plan pool is the history-free
critic pool of the published latents of the next 4 frames (clamped and padded at the episode end), the
same function and cache the online rows use on their denoised chunk, so expert and online pools differ
only in where the chunk came from. One-off precompute in the expert featurisation (about 900 pooling
passes for 30 demos), stored beside `expert_features.pt` under expert recipe v2. Expert rows keep z = 0 and
a_base = demo action.

## 9. Replay rows

Added per row: plan_pool (executed, 3072), plan_pools_all (M × 3072), plan_index, plan_a_base_all
(M × K × 480), plan_pool_next (filled at finalisation). Expert rows carry their single pool repeated M
times and their K-copied demo action repeated M times, so every row has one shape and one target formula.
Row size grows from about 60 KB to about 165 KB in float32; 20k rows at 320k ≈ 3.3 GB in the resume file.

## 10. Config

```json
{
  "task_ids": [9],
  "residual_input": "base",
  "epsilon": -0.3,
  "online_env_steps": 320000,
  "video_steps": 5,
  "action_steps": 10,
  "video_exec_step": -1,
  "plan_candidates": 4,
  "plan_random_prob": 0.1
}
```

The task 0 run is the same file with `task_ids: [0]`. The random-plan warm-up reuses
`selection_warmup_steps` (64k). All plan and sampler fields enter the recipe fingerprint, so no resume file
saved before this change loads; the runs use fresh names. Evaluation takes M from the same RL config
(`brev/eval.sh --plans M`, policy name suffix `-mM`) and records it as `eval_plans` in the shard settings;
a mismatched M fails on the critic width. The Modal evaluation path is not extended (GB10 only).

## 11. Code changes (no comments or docstrings)

- `script/lingbot_rl_policy.py`: `expand_kv_rows(transformer, reps)` and `select_kv_rows(transformer, rows)`
  beside `expand_conditional_kv`; `_video_passes(latents, ...)` at batch 2M with its own CFG payload;
  `_decode_actions` taking an (M, K) layout; `select_plan(batch, plans, k, rank_fn, random)` implementing
  §5, returning s, pools, a_base (M, K, 16, 30), z, chosen index, latents; `decode_candidates` and
  `select_action` call it; `decode_plans` kept as the looped reference for tests.
- `script/lingbot_rl_model.py`: `PlanCritic`; `CriticEnsemble` input 2·STATE_DIM + action; `n_step_target`
  takes p′; `update_plan_critic(state, pools, a_base_all, is_expert)`; resume and inference states include
  the plan critic, its optimiser and schedule.
- `script/lingbot_rl_buffer.py`: the keys of §9; finalisation fills plan_pool_next.
- `script/lingbot_rl_data.py`: expert plan pools (§8).
- `script/lingbot_rl_train.py`: M-plan collection with the random-plan schedule, the metrics of §6, V_plan
  update after each critic update, train-eval and held-out eval with M plans.
- `script/lingbot_rl_config.py`, `script/lingbot_eval_config.py`: fields, validation, fingerprints.
- `brev/train.sh`, `brev/eval.sh`: `--plan-candidates`.
- Tests: fingerprints; buffer keys and finalisation; plan-critic update on a stub (loss falls, Q and actor
  untouched); `select_plan` on the mocked transformer (M = 1 bit-for-bit against today's `_infer`; M = 2
  batched equals the looped `decode_plans` per plan; the history after selection equals the winner's rows
  and nothing else); eval option plumbing.

## 12. Experiments

Pre-run (SFT, no training): the re-benches of §3 fix the sampler for both tasks.

Training, 5 / 10, M = 4, K = 4, 320k env steps, one per GB10:

| run | task | box | SFT held-out at 20 / 50 |
|---|---|---|---|
| dice-t9-plan-m4 | 9 | Jarvis | 42 / 100 |
| dice-t0-plan-m4 | 0 | Watson | 63 / 100 |

Task 0 is the second task closest to a 50% prior with an existing 100-episode baseline and full recipe
results (81 / 90 at 660k).

Held-out per task, 100 episodes, the held-out seeds of every previous result, all at 5 / 10:

| policy | role |
|---|---|
| SFT, M = 1 | the sampler check (re-bench, running) |
| m4 run, residual, V_plan argmax over 4, Q over 4 inside it | the method |

Whether ranking beats a random plan is read from plan_regret and plan_agreement during training, not from
an extra evaluation.

Training signals: plan_agreement, plan_regret, plan_value_spread and plan_critic_loss per update bin;
replay sharpening every 5k env steps as today, extended with V_plan's spread across the M stored plans.
Contraction for the write-up is read at the plan level: V_plan spread across sampled plans and plan_regret
on the random-plan chunks, not ΔH on actions.

Go / no-go: unit tests and the 300-step smoke at 5 / 10 (plan_agreement computed, plan_critic_loss falling,
seconds per env step logged); at 160k, plan_agreement ≥ 0.7 and plan_value_spread clearly above V_plan's
regression error, else stop and diagnose before spending the second half.

## 13. Costs

Per chunk at M = 4: 5 video passes at batch 8 plus one action decode at batch 16, against 20 passes at
batch 2 plus one decode at batch 4 over 50 steps today. Collection should run at or below the 20 / 50
smoke's 0.64 s per env step; the P0 smoke pins it. 320k env steps ≈ 2 to 2.5 days per run. Evaluation at
M = 4 is a few times the 65 s per episode of M = 1, roughly 3 to 5 h per 100 episodes.

## 14. Risks

- The 5 / 10 re-bench falls outside 42 ± 10 on task 9 (or 63 ± 10 on task 0): fall back to 20 / 50 with
  M = 4 (4× video cost).
- Plans collapse at video CFG 5 (across-plan spread not above the 0.033 within-plan spread): read from the
  stored M pools and plan_value_spread in the first 10k env steps; stop and revisit guidance.
- Selection bias in V_plan's training distribution: mitigated by the random plan and by regressing on rejected
  plans.
- Expert plan pools come from true futures, online pools from samples: diagnostic V_plan on expert versus
  online rows at matched returns.
- Batched cache rows: covered by the M = 1 bit-for-bit test, the M = 2 batched-equals-looped test and the
  winner-only history test.

## 15. Open items

1. Re-bench outcomes at 5 / 10 on tasks 9 and 0.
