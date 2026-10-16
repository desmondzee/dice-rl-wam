# Trial: a plan critic trained on counterfactual simulator rollouts

Status: design for review. One change: V_plan's training signal. Everything else (prior, sampler, residual actor, Q
ensemble, recipe, evaluation protocols) is fixed, so a held-out difference is attributable to the critic.

## 1. The problem this addresses

V_plan regresses onto a target built from Q: the max over four candidates of the minimum over ten heads. Measured on
the 320k task 9 and 240k task 0 buffers (docs/heldout-results.md, plan-critic studies): two halves of the Q ensemble
agree on the best plan 0.63 / 0.44 of the time, the best Q ranker's link to outcomes is 0.42 / 0.39, V_plan's is 0.39 /
0.13, and random-plan ablations put V_plan's contribution at 3–5 points on 100 episodes. Nothing in the pipeline ever
observes what the three plans that were not executed would have done.

## 2. The change

Label every plan at a stored state by executing it in the simulator from that state, and train V_plan on those labels.

| item | decision |
|---|---|
| label | for each of the M = 4 plans at a stored chunk start: restore the simulator state, execute each of the plan's K = 4 decoded base-action candidates for 16 steps with physics only (no policy call), read the object poses, score progress; the plan's label is the mean over its 4 candidates |
| progress score (task 9) | φ(state) from the simulator poses the trainer already records: −‖mug − microwave target‖ while the mug is outside, a bonus once the mug is inside, −|door angle − closed| once inside, 1.0 on the task's success predicate; label = φ(after) − φ(before). One function per task in `script/lingbot_rl_progress.py`, task 9 for the trial |
| critic | unchanged architecture: 10 heads on (s, p), the policy's own pooled features only; object poses and simulator state never enter the critic's inputs, so the trained critic deploys without privileged information |
| loss | MSE onto the label, as today; readout = mean over heads (the aggregation study); a within-state pairwise ranking loss is reported offline as a secondary variant, not used live in this trial |
| what stays fixed | the 320k task 9 checkpoint's residual actor and Q ensemble; the 5 / 10 sampler; M = 4, K = 4; the evaluation protocols and seeds |
| no actor in this trial | the trial measures the critic; a noise-space pre-screen or noise actor trained against this critic is the next trial once the critic has a measured ceiling |

## 3. Data

The existing buffers do not store the simulator state, so the labels cannot be produced from them. Two routes:

- A. **Labelling collection**: run the 320k task 9 policy (V_plan ranker) on the standard task and on the LIBERO-Plus
  training variants with the thirds mix, storing per chunk `sim_state` (MuJoCo qpos, qvel and the controller goal, a
  few hundred floats), `s`, `plan_pools_all`, `plan_a_base_all`, `object_state`, `reset_kind`. Target 4,000 chunks
  (about 64k env steps, 10 h on one GB10 at full speed). No training during collection.
- B. Add `sim_state` to the buffer keys now and label the next training run's buffer; no extra collection, but the
  labels arrive only when that run exists.

Route A is the trial; B is the standing change so every later run can be labelled.

Restore fidelity check (gates labelling): for 100 stored chunks, restore the state, replay the executed chunk's actions,
compare the resulting object poses with the stored next-chunk `object_state`; require agreement within 1 mm and 1°.
The OSC controller carries internal state; if the check fails, the controller goal is added to `sim_state`.

Label cost: 16 physics rollouts of 16 steps per chunk, about 1 s on a CPU; 4,000 chunks in about an hour on one box.

Continuation calibration (optional, 100 chunks): roll each plan out to termination with the prior to compare the
short-horizon label with the true return; reports the horizon bias of the progress score.

## 4. Measurements

Offline, on an 80 / 20 split by time of the labelled chunks: split-half agreement of the new V_plan (ceiling to beat:
0.71 for the Q-mean ranker, 0.78 for V_plan's own mean readout); agreement with the Q-mean ranker; correlation of
the executed plan's relative score with the chunk's Monte Carlo return (0.39 today); label consistency across the
four candidates of a plan (how much of the label is plan, how much is action noise).

Held-out, the measurement that matters: the 320k task 9 checkpoint with the new V_plan as ranker, residual and Q
unchanged, on the standard 100 episodes (V_plan 80, random plan 75) and the LIBERO-Plus 100 (53, 50). Four hours
each. The LIBERO-Plus variants in the labelling collection are the training variants; the plus plan overlaps them,
so the held-out plan from the earlier spec is the clean test when it exists.

## 5. Code

- `script/lingbot_rl_progress.py`: `progress(task_id, object_state_before, object_state_after, success)`; task 9.
- `script/lingbot_rl_train.py`: `sim_state` read at each chunk start when `proprio` is set (both routes); the
  labelling collection is `train.sh --collect-only` (no updates, no checkpoints, buffer saved).
- `script/lingbot_rl_label.py`: offline labeller: loads a buffer, restores states, runs the rollouts, writes
  `labels.pt` (chunk id → M × K progress); the fidelity check as a flag.
- `script/lingbot_rl_plan_critic.py`: trains V_plan on labels, writes a residual checkpoint with the new plan critic
  and the unchanged actor / Q, so `evaluate` and the plus driver load it as today.
- Tests: progress score on hand-built poses; labeller on a stub environment; critic training on synthetic labels.

## 6. Risks

- Restore fidelity: controller state; gated by the check.
- Progress score myopia: a chunk that scores well and leads somewhere bad; the continuation calibration measures it.
- The label is a mean over four candidates, so it is a plan value under the prior's action noise, which is what V_plan
  is defined as; the residual is not applied in the rollouts, a deliberate simplification for the trial.
- Simulator-only labels: the deployment claim is "trained in simulation on counterfactual outcomes, conditioned only on
  the policy's own features"; transfer to a real robot rests on the visual domain, not on the labels.
