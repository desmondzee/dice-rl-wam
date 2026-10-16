# Plan critic trained on counterfactual simulator rollouts (run A, online)

Status: approved design (2026-10-16). Run A (`dice-t9-plus-m4-pool`, Jarvis) is replaced by a run with one change to
its recipe: V_plan is trained on labels measured by executing every candidate plan in the simulator, instead of on a
target derived from Q. Run B (`dice-t9-plus-m4-struct`, Watson) keeps running unchanged; the pooled-feature baseline
with the Q target is re-run later if the result warrants it.

## 1. The problem this addresses

V_plan regresses onto the max over four candidates of the minimum over ten Q heads. On the 320k task 9 and 240k task 0
buffers two halves of the Q ensemble agree on the best plan 0.63 / 0.44 of the time, the best Q ranker's link to
outcomes is 0.42 / 0.39, V_plan's is 0.39 / 0.13, and the random-plan ablations put V_plan's contribution at 3–5 points
on 100 episodes. Nothing in the pipeline observes what the three plans that were not executed would have done.

## 2. The change

| item | decision |
|---|---|
| run | `dice-t9-plus-m4-pool-cf` on Jarvis: the run A recipe (task 9, prior frozen, thirds reset mix over the LIBERO-Plus training variants, pooled features + proprioception, M = 4, K = 4, 5 / 10 sampler, ε −0.3, 320k env steps, checkpoints every 80k) with `plan_target = "rollout"` |
| label, per chunk | after the policy decodes its M × K candidates and before it executes: a shadow environment of the same task and variant is set to the live simulator's state; each of the 16 candidate chunks (base actions, no residual) is executed there for the chunk's length with physics only; the score is φ(after) − φ(before) from the object poses; the plan label is the mean over its 4 candidates; labels (M, K) float32 are stored with the chunk |
| φ (task 9) | from the simulator poses the trainer records: −‖mug − interior target‖ while the mug is outside the microwave; +1 once inside; +(1 − door open fraction) once inside; +2 on the task's success predicate. One function per task in `script/lingbot_rl_progress.py`; interior target and door range measured once at startup from the base-task environment |
| critic | unchanged architecture, 10 heads on (s, p); the training target is the stored plan label; readout for ranking = mean over heads; the Q-derived target is still computed and logged for comparison, not trained on |
| what stays fixed | the residual actor, the Q ensemble and its targets, the sampler, M, K, the reset schedule, the buffer's other keys, both evaluation protocols |
| shadow environment | one per episode, constructed with the live environment's kwargs (same task or variant, same reset kind), reset once, then driven only by state assignment; the live environment is never restored to, so the episode is unaffected; a fidelity check at startup gates the run (section 4) |
| logged | per chunk: label agreement with the Q-mean ranking, label spread across the 4 candidates of a plan over the label spread across plans, label of the executed plan minus the mean; per episode as now |
| cost | about 16 × 16 physics steps per chunk, about one second, roughly 10% on the current 0.54 s per env step |

## 3. Why these choices

- Labels for every plan remove the two noise sources we measured: Q's estimation noise and the executed-only selection
  bias. What remains is the stochasticity after the chunk.
- Base actions without the residual make the label the plan's value under the prior's action noise, which is what
  V_plan is defined to estimate, and keep the label independent of the actor being trained.
- A shadow environment instead of restoring the live one: no fidelity risk to the episode and no controller state to
  put back; the shadow's own controller state is reset before each candidate.
- Object poses and simulator state enter the label only; the critic conditions on the policy's own features, so a
  critic trained this way runs without privileged information.
- Keeping the Q target computed but untrained gives a live comparison of the two signals at no cost.

## 4. Fidelity check (gates the launch)

Before collection: for 20 chunks of a warm-up episode, set the shadow to the live state, execute the live chunk's
actions in both, compare object poses after the chunk; require agreement within 1 mm and 1°. If the OSC controller's
internal state breaks this, the controller goal is copied along with the simulator state.

## 5. Measurements

Online: the logged agreement and spreads. Offline at 80k and 320k, on the stored labels: split-half agreement of the
new V_plan (reference 0.71 for the Q-mean ranker, 0.78 for V_plan's mean readout), correlation of the executed plan's
relative label with the chunk's Monte Carlo return (0.39 today), and the within-plan label spread. Held-out at 320k:
the standard 100 episodes with the V_plan ranker and with the random ranker, and the LIBERO-Plus 100, against the
existing rows (80 / 75; 53 / 50) and run B.

## 6. Code

- `script/lingbot_rl_progress.py`: `progress(task_id, before, after, success)` and the startup calibration of the
  interior target and door range.
- `script/lingbot_rl_config.py`: `plan_target` ("q" | "rollout"), in the protocol fingerprint.
- `script/lingbot_rl_train.py`: shadow environment per episode, `label_plans(shadow, state, candidates, norm, task_id)`,
  `plan_labels` stored per chunk, the logged diagnostics, the fidelity check at startup; `brev/train.sh --plan-target`.
- `script/lingbot_rl_buffer.py`: `plan_labels` (M, K) float32 with a legacy default of zeros.
- `script/lingbot_rl_model.py`: `update_plan_critic` takes the target from `plan_labels` when `plan_target == "rollout"`;
  `rank_plans` by the mean over heads.
- Tests: progress on hand-built poses; label_plans on a stub environment; the plan-critic update on stored labels;
  fingerprint; default recipe unchanged.

## 7. Risks

- Progress myopia: a chunk that scores well and leads somewhere bad. The offline Monte Carlo correlation measures it;
  a continuation calibration on 100 chunks is optional later.
- Shadow-state mismatch for variants with sampled objects: the shadow is built for the same variant and set to the
  live state, so object identities match; the fidelity check covers it.
- Label cost under the plus package: measured in the smoke.
- Simulator-only labels: the claim is "trained in simulation on counterfactual outcomes, conditioned only on the
  policy's own features".
