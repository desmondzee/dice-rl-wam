# Plan-level selection for LingBot-VA DICE-RL

Status: design under discussion, not implemented. Code base: main at 1bb03c8 (RL scripts
comment-free, `script/modal_volume_tools.py` for verified volume transfers, DSRL reference at
`.cache/dsrl`).

## 1. Decisions taken

| item | decision |
|---|---|
| task, residual input, ε | task 9, `residual_input: base`, ε = −0.3 |
| plan candidates M | 4 |
| budget | 320k env steps; checkpoints every 80k plus the end; train-evals at 0 / 80k / 240k; rolling resume every 10 episodes and at every checkpoint; W&B as today |
| plan_random_prob | 0.1, constant, collection only; evaluation always executes the argmax plan |
| execution schedule | partial video denoising to step E, adopted after probe P1; all baselines re-run on the chosen E |
| ranking step k | from probe P3 |
| route for rejected plans | partial decode at step k if P3 shows agreement, else continue to E |
| compute | GB10 units (DGX Spark); two units run two independent jobs, not one faster run |
| critics | Q extended with the plan pool; V_plan a distilled 10-head critic; actor unchanged |

## 2. Why

Given one video plan, the four action candidates differ by a standard deviation of 0.033 in
normalised units against 0.44 across states; the residual is 0.034 RMS; the ΔH bin is 0.04. The
action decoder is near-deterministic given the plan, so selection among candidates and the residual
are substitutes (base run 63 → 75, filtered run 73 → 73) and no recipe produced the paper's negative
ΔV–ΔH trend. The policy's modes live in the plan, sampled once per chunk. Choosing among M plans with
a learned plan critic is the contraction operator that fits this model: it keeps the policy on its
own support and acts where the distribution has mass to remove.

## 3. Facts the design relies on

- A chunk is 4 latent frames = 16 env actions; the chunk latent is (48, 4, h, w).
- Video denoise: 20 scheduler steps, CFG 5, each step a transformer pass attending to the KV cache
  of observed history; only the last step writes the chunk into the cache. The released config
  exposes `video_exec_step`, which truncates the schedule.
- Action denoise: 50 steps on the 768-wide action stream after the video, reading the plan through
  the cache; K candidates run against a K-expanded cache.
- The pinned post-training Trainer noises the video conditioning with probability 0.5 to
  s ∈ [0.5, 1.0] (`_add_noise(noisy_cond_prob=0.5)`), and the action stream attends to those
  latents, so our SFT checkpoint was trained to decode actions from plans denoised to s ≥ 0.5. The
  paper's implementation integrates 3 steps to s ≈ 0.6.
- The critic state is a history-free pool of the current frame (3072-d). The same hook pools the
  video stream's tokens at any denoise step; those tokens are history-conditioned because the pass
  attends to the cache, so a plan pool needs no explicit previous-frame input.
- Schedule steps are not evenly spaced in s (shifted schedule); every probe logs sigma per step.

## 4. Probes before the run

- **P0 Stack and throughput.** The 300-step smoke on a GB10 (aarch64 CUDA torch, MuJoCo, robosuite,
  EGL). Record s/env-step against the L40S's 0.29 at 20 steps.
- **P1 Execution schedule.** SFT held-out on task 9, 100 episodes, same seeds, at `video_exec_step`
  20, 12, 10, 6 and 3, with sigma logged per cut. E = the shortest cut whose success stays inside
  the 20-step interval (42 ± 10). Fallback if every short cut degrades: re-post-train on the
  30 × 10 demos with the augmentation range set to the target s_E (one bound change in the
  adapter's `_add_noise` call), 1,000 updates as in the SFT recipe, then repeat the sweep. Either
  way the SFT, base-run and ε-run residual baselines are re-run at E before the M-plan run.
- **P2 Plan diversity.** For ~200 train-eval states sample M = 4 plans at E, decode base actions,
  report across-plan vs within-plan action spread (within is 0.033 today) and the current Q's
  spread across plans. If plans collapse at video CFG 5, revisit guidance before building.
- **P3 Partial-decode agreement.** Same states and noises: decode actions from the plan at step k
  and at E; report chunk distance and Q agreement for k ∈ {3, 6, 10} ∩ [1, E]. Fixes k and decides
  the rejected-plan route.
- **P4 Rankability (during the run).** Per chunk, agreement between V_plan's argmax at step k and
  Q's argmax over fully decoded candidates; logged from the first update.

## 5. Policy at inference

Per chunk:

1. Draw M video noises; run steps 1..k for each (loop over M, or batch M × 2 for CFG). Nothing is
   written to the cache before the last step, so rejected plans leave no trace.
2. At step k, pool each plan's pre-projection tokens → p_m (3072-d).
3. Choose m* = argmax over m of min-ensemble V_plan(s, p_m).
4. Continue plan m* to E (nothing to continue if k = E). Decode K candidates, rank with
   Q(s, p_m*, a), apply the residual, execute. Unchanged from today after this point.

Video passes per chunk: M·k + (E − k), against E for the single-plan policy at the same schedule.

## 6. Policy at collection

Same as inference with two differences: with probability 0.1 (after the 64k warm-up, during which
the plan is always random) the executed plan is uniform over the M; and every plan gets K decoded
candidates so it can be scored. Rejected plans are decoded from the step-k plan when P3 allows,
otherwise continued to E. Stored per chunk: s, the M pools, the chosen index, each plan's K base
actions and noises, the executed action and the usual fields.

Logged per chunk: plan_agreement (P4), plan_regret (Q of the best-scored plan minus Q of the
executed one; on random chunks this measures V_plan's error off its own argmax), plan_value_spread
(max − min of V_plan over the M), plan_random_rate, plan_critic_loss, v_plan_mean.

## 7. Critics and actor

**Q(s, p, a)**: current-frame pool ‖ plan pool at step k ‖ action chunk. 10 heads of
[1024, 1024, 1024], min-ensemble, TD with the multi-sample target as today, plus p′_exec in the
bootstrap. Input width grows from 3,552 to 6,624; hidden widths unchanged.

**V_plan(s, p)**: same inputs minus the action, 10 heads of [1024, 1024, 1024]. Trained by
regression only, on every stored (s, p_m) including rejected plans:

- a_mk = a_base_mk + actor(s, a_base_mk) for the K candidates of plan m.
- q_mk = min over heads j of Q_j(s, p_m, a_mk).
- y_m = max over k of q_mk.
- loss = sum over heads of (V_plan_j(s, p_m) − y_m)².

Everything in y_m is computed without gradient: Q, the actor and the stored base actions supply
numbers, not gradients; only V_plan's parameters move. Q stays the single TD-trained source of
truth and the actor is never trained to make plans look good to V_plan. Two choices inside the
target, both logged: live Q (matches what collection-time selection uses; chosen) versus its
Polyak copy; max over candidates (matches the executed within-plan selection; chosen) versus mean.
Optimiser settings as the critic (Adam 1e-4, wd 1e-5, clip 1.0, cosine restarts), updated after
each critic step. No target network, no bootstrap through V_plan.

Why not TD for V_plan: rewards exist only for executed plans, so a TD plan critic trains on its own
argmax choices and extrapolates to the plans it rejects at ranking time; its bootstrap plan is
itself an argmax; it is a second bootstrapped critic with its own drift; its targets arrive only
after episode finalisation. Distillation follows DSRL's noise-aliased critic (`.cache/dsrl`:
Q^W regressed onto Q^A at fresh Gaussian noises, actor maximising Q^W); here plans play the role of
noises and V_plan is the ranker rather than an actor's objective.

**Residual actor**: unchanged, inputs (s, a_base), zero-init, β = 100, filter at ε.

## 8. Expert rows

Demos have no sampled plan but have the true future. For each demo chunk: encode the real next
4 frames, noise the latent to the step-k level, run one transformer pass against the demo's own
history cache, pool → p. This is the model's training augmentation, hence in-distribution. One-off
precompute in the expert featurisation (about 900 passes for 30 demos), stored beside
`expert_features.pt`. Expert rows keep z = 0 and a_base = demo action; their V_plan target is the
single candidate's Q.

## 9. Replay rows

Added per online row: plan_pool (executed), plan_pools_all (M × 3072), plan_index, plan_a_base_all
(M × K × 480), plan_z_all (M × K × 480), plan_pool_next (filled at finalisation). Expert rows carry
plan_pool only. Row size grows from about 17 KB to about 22 KB; 20k rows at 320k ≈ 0.45 GB.

## 10. Config

```json
{
  "task_ids": [9],
  "residual_input": "base",
  "epsilon": -0.3,
  "online_env_steps": 320000,
  "video_exec_step": "E from P1",
  "plan_candidates": 4,
  "plan_rank_step": "k from P3",
  "plan_random_prob": 0.1,
  "plan_selection_warmup_steps": 64000,
  "plan_reject_decode": "partial or full from P3"
}
```

`plan_candidates = 1` reproduces today's recipe; all plan fields and `video_exec_step` enter the
recipe fingerprint. The eval config gains `eval_plan_candidates` (default M) and `video_exec_step`
(default E), so every held-out result records its schedule.

## 11. Code changes (on the cleaned base; no comments or docstrings)

- `script/lingbot_rl_policy.py`: `_infer` takes M and k; loops plans through steps 1..k with the
  pooling hook active at step k, ranks, continues the winner to E; decodes K candidates for every
  plan via the K-expanded cache with snapshot/restore per plan (partial or full per config);
  `decode_candidates` returns pools, chosen index, per-plan candidates; `select_action` uses the
  early-exit path with the argmax only.
- `script/lingbot_rl_model.py`: `PlanCritic`; `CriticEnsemble` input 2·STATE_DIM + action;
  `n_step_target` takes p′; `update_plan_critic`; resume and inference states include the plan
  critic, its optimiser and LR schedule.
- `script/lingbot_rl_buffer.py`: new keys; finalisation fills plan_pool_next.
- `script/lingbot_rl_data.py`: expert plan pools from noised true futures.
- `script/lingbot_rl_train.py`: M-plan collection, random-plan schedule, the metrics of §6,
  train-eval and held-out eval with M plans at E.
- `script/lingbot_rl_config.py` and `script/lingbot_eval_config.py`: fields, validation,
  fingerprints.
- `brev/train.sh` (or its GB10 equivalent): `--plan-candidates`, `--plan-rank-step`,
  `--video-exec-step`; `script/lingbot_eval_modal.py`: the same for evals.
- Tests: fingerprints, buffer keys and finalisation, plan-critic update on a stub, policy M-loop on
  a mocked transformer with M = 1 reproducing today's actions exactly, eval option plumbing.

## 12. Costs

Video passes per chunk, M = 4, relative to today's single plan at 20 steps:

| E | k | rejected-plan route | collection | eval |
|---|---|---|---|---|
| 20 | 6 | partial | 38 (1.9×) | 38 (1.9×) |
| 10 | 6 | partial | 28 (1.4×) | 28 (1.4×) |
| 10 | 10 | partial (k = E) | 40 (2.0×) | 40 (2.0×) |
| 6 | 3 | partial | 15 (0.75×) | 15 (0.75×) |
| 10 | 6 | full (continue all 4) | 40 (2.0×) | 28 (1.4×) |

Wall-clock = 320k × (P0's s/env-step at E) × the collection factor. On an L40S at E = 10, k = 6,
partial: about 320k × 0.15 × 1.4 ≈ 19 h. A GB10 is expected to be slower per step; P0 measures it.

## 13. Validation and go/no-go

- Unit tests and the 300-step smoke; the smoke must show plan_agreement computed and the plan-critic
  loss decreasing.
- P2 must show across-plan spread well above 0.033 before anything is built.
- At 160k: plan_agreement ≥ 0.7 and plan_value_spread clearly above V_plan's regression error,
  else k is raised (a new fingerprint, hence a new run).
- Held-out at the end, all at E: the M-plan policy (the method), the same residual with M = 1
  (isolates plan selection), and the ε run's residual at E as the baseline.
- Contraction for the write-up: outcome variance across the M plans versus the executed plan,
  and V_plan spread per state; not ΔH on actions.

## 14. Risks

- Step-k features rank poorly at small k; mitigated by P3 before the run and P4 during it.
- Memory: M × CFG = batch 8 through the 5B stream at step k; today's batch 2 uses 13 GB; the GB10's
  128 GB unified memory covers it, an L40S (48 GB) likely does with the loop-over-M form.
- Selection bias in V_plan's training distribution; mitigated by the random plan and by training
  on rejected plans.
- Expert plan pools come from real futures, online pools from sampled plans; a gap the critic may
  key on. Diagnostic: V_plan on expert versus online rows at matched returns.
- Cache correctness with M plans against one history: covered by the M = 1 bit-for-bit test.
- Partial-denoise execution changes the prior; every baseline is re-run at E before comparison.

## 15. Open items

1. E and k (from P1 and P3).
2. Whether the re-post-train fallback is needed (from P1).
3. GB10 setup: aarch64 wheels for the LeRobot lock, EGL on the Blackwell driver, Modal client for
   syncs (verified transfers through `script/modal_volume_tools.py`).
