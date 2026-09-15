# Why DICE-RL matched, but did not beat, the SFT baseline

Analysis of `result/lingbot-eval/libero30-sft-step000600-eval` (SFT) and
`result/lingbot-rl/libero30-dice-baseline` + `result/lingbot-rl-eval/libero30-dice-baseline` (RL),
against the DICE-RL paper (arXiv 2603.10263v2, read in full).

## What the runs show

Both the SFT checkpoint and the RL policy after 100,000 env steps score exactly 69.0% macro
success on the pinned 200-episode protocol (20 episodes × 10 LIBERO-10 tasks, seed 42). The tie
hides real movement: RL improved tasks 0, 1, 3, 5 (e.g. task 0: 60→80%, task 1: 80→95%) and
regressed tasks 2, 6, 7, 8, 9 (task 7: 90→75%, task 8: 10→0%) by an exactly offsetting amount.

The training-time evals (1 episode/task, fixed seeds, `train_eval/step_*`) tell the more
important story:

| env steps | 0 | 25,204 | 50,291 | 75,359 | 100,000 |
|---|---|---|---|---|---|
| successes /10 | 1 | 7 | 9 | 9 | 5 |

Three observations follow. First, the run *was* learning: 9/10 at 50k–75k versus an expected
~6.9/10 for the base policy on these seeds. Second, the policy **degraded between 75k and 100k
steps**, and the 200-episode eval measured only the degraded final checkpoint — intermediate
`residual.pt` snapshots were not saved by that run (the per-step checkpointing in the current
uncommitted `lingbot_rl_train.py` changes was added afterwards). Third, the step-0 result of
1/10 is far below the base policy (P(≤1 success | SFT per-task rates) < 1e-4), meaning the
freshly initialized residual policy was substantially *worse* than the frozen prior it wraps.

So the correct framing is not "RL learned nothing" but "RL learned, destabilized late, and we
evaluated the endpoint of the instability".

## Implementation deviations from the paper, ranked by likely impact

**1. The frozen prior is never sampled inside the update loop, so multi-sample expectation
training (paper Eq. 4–5, K=16) is absent.** The paper forms the critic target by averaging
Q over K fresh candidates `π_pre(s_{t+h}, z'_k) + s_θ(s_{t+h}, z'_k)` drawn with the *current*
residual, and maximizes the actor's Q averaged over K fresh candidates at the current state.
Our `n_step_target` bootstraps from the single **stored executed next chunk** (`a_next` in
`lingbot_rl_buffer.py`), and `update_actor` optimizes the single stored `(z, a_base)` pair —
n-step SARSA on stale behavior actions rather than an expectation under the improving policy.
Expert rows additionally use `z = 0` with `a_base` set to the demo action, a pairing the actor
never sees at inference. This was a deliberate design choice (training without the 5B prior),
but it removes the paper's central mechanism: Fig. 15 shows K=1 is markedly slower and less
stable than K=16, and stale SARSA targets progressively mis-value the policy as the residual
drifts from the behavior that generated the buffer — consistent with the late-run regression.

**2. The BC-loss filter is anchored to the bootstrapped TD target instead of a Monte-Carlo
return.** Paper Eq. 6 relaxes the BC penalty only when the edited action improves on the base
action *and* its predicted value does not exceed a Monte-Carlo return estimate Ĝ(s) from
replay — explicitly to stop the actor exploiting critic overestimation. `update_actor` uses
`target_q` (the n-step target, itself bootstrapped from the target critic on stale `a_next`)
in place of Ĝ. When the target critic inflates at next states, the guard is checked against
the inflated quantity, so the BC penalty can be switched off exactly where overestimation is
worst. The buffer already computes discounted n-step returns; extending that to a
full-episode return-to-go per chunk would restore the paper's anchor.

**3. UTD is implemented as 10 critic gradient steps on one fixed minibatch with one fixed
target.** `_update_from_buffer` samples a single batch, computes the target once, then takes
10 critic steps toward it (with a Polyak update after each), plus one actor step, per collected
chunk. The paper's 10–20 gradient steps per update are standard fresh-minibatch updates.
Repeating one batch against a frozen target acts like a 10× critic learning rate with local
overfitting, and yields only ~6.4k actor updates over the whole run (one per chunk).

**4. The residual actor's output layer is not zero-initialized.** `ResidualActor` uses default
`nn.Linear` init, so at step 0 the policy adds a random residual of roughly 0.3–0.5 std to every
normalized action dimension, including the gripper. The 1/10 step-0 eval confirms the initial
policy was badly corrupted relative to the 69% prior. Consequences: the earliest replay data
comes from a broken policy, and part of the 100k-step budget is spent shrinking the random
residual (via β=100 BC) back to the prior instead of improving on it. Residual-RL practice
(and the paper's premise that finetuning starts *at* the prior) implies the initial residual
should be exactly zero.

**5. Value-guided best-of-N is used for collection (K=4) but not at evaluation.** All evals —
training-time and final — go through `predict_action_chunk(batch, k=1)`: one latent, no critic
ranking. The measured policy therefore omits one of DICE-RL's three components, which the paper
finds accelerates convergence and improves peak performance (Fig. 16). Collection also uses
K=4 candidates against the paper's K=16.

## Non-implementation factors

**Budget.** 100k env steps ≈ 6,418 chunks ≈ ~300 online episodes spread over 10 tasks. The
paper's LIBERO-10 runs use ~1.1M env steps (Fig. 10–11), and their curves show only small gains
by the 100k mark. Our mid-run 9/10 evals suggest budget was not the binding constraint yet —
stability was — but even a fully faithful implementation would be expected to show modest gains
at a tenth of the paper's budget with a single seed.

**Evaluation power.** With 200 episodes, the 95% CI on the macro rate is roughly ±6.5pp, and
per-task rates at 20 episodes move ±15pp by noise alone. The per-task "regressions" are
individually within noise; the 75k→100k drop across two independent eval protocols is the
signal that is not.

**Prior quality on the failing tasks (not our bug).** The paper's finetunability analysis
(§5.2) predicts exactly what task 8 shows: DICE-RL contracts the action distribution *within
the prior's support* and cannot create modes the prior essentially never samples. The SFT
policy succeeds at "put both moka pots on the stove" 2/20 times (all 18 failures hit the step
limit) — low good-mode coverage — and RL took it to 0/20. Task 9 (45% SFT) similarly stagnated.
Improving these two likely requires a better prior (more/better demos, different SFT
checkpoint epoch per the paper's Fig. 4) rather than more RL.

## Verdict

The implementation is a recognizable DICE-RL, and the mid-training evals prove the overall
loop (residual actor, chunk critic, RLPD mixing, sparse reward plumbing) can lift the policy
well above the prior. But the answer to "is it our implementation?" is largely yes, in the
specific sense that the three stabilizing mechanisms the paper leans on — multi-sample
expectation targets, the MC-anchored BC filter, and (at eval) best-of-N — are respectively
replaced by stale single-sample SARSA targets, a bootstrap-anchored filter, and disabled;
combined with the non-zero residual init and same-batch UTD, these plausibly produced the
late-training destabilization, and the eval protocol then measured only the destabilized
endpoint at one tenth of the paper's step budget.

## Recommended next steps, in order

1. Re-run with per-eval-point checkpoints (already in the uncommitted changes) and evaluate
   the 25k/50k/75k checkpoints on the full 200-episode protocol — cheapest test of the
   "peaked then collapsed" reading, since the old run's intermediate weights were not saved.
2. Zero-initialize the residual actor's final layer.
3. Sample a fresh minibatch (and recompute the target) for each of the 10 critic steps.
4. Anchor the BC filter to a Monte-Carlo return-to-go stored per chunk at episode
   finalization, per Eq. 6.
5. Evaluate with critic-ranked best-of-N (k=4, matching collection) so the measured policy
   includes value-guided selection.
6. If prior inference in the trainer stays off the table, approximate Eq. 4–5 by storing all
   K=4 collected candidates per chunk and averaging targets/actor objectives over them; that
   recovers most of the multi-sample benefit without new prior calls.
7. Check the W&B run (`dice-lingbot-va-rl`, `libero30-dice-baseline`): `residual_rms` and
   `bc_filter_rate` rising while `q_mean` inflates over the last quarter of training would
   directly confirm the drift mechanism in findings 1–3.
