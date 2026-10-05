# Single-task DICE-RL on LingBot-VA: held-out results

Protocol: LIBERO-10, one task per run, 100 evaluation episodes per policy = the 50 canonical init states × 2 seeds (`EvalConfig(stage="heldout")`, seed 42). Identical states and seeds for every policy on a task. SFT prior = `libero30-sft` step 600 with best-of-1. RL = frozen prior + residual actor, best-of-4 by critic argmax (the method) or best-of-1 (residual only). 95% intervals are ±0.06–0.10 at n=100. Source: `docs/heldout-results.csv` (this table, with seed-pass splits and intervals) and the per-episode JSON under `result/heldout/<policy>/task_XX/`.

## Results

```mermaid
xychart-beta
    title "Held-out success rate, 100 episodes per bar"
    x-axis ["t0 SFT", "t0 RL k1", "t0 RL k4", "t0 base-probe k1", "t4 SFT", "t4 660k k1", "t4 480k k1", "t4 660k k4", "t4 480k k4", "t4 240k k4", "t4 base-probe k1", "t9 SFT", "t9 400k k1", "t9 400k k4", "t9 480k k4", "t9 660k k4", "t9 base-probe k1", "t9 base-probe k4", "t9 a_base-run k1", "t9 a_base-run k4", "t9 a_base-run 480k k4"]
    y-axis "success rate" 0 --> 1
    bar [0.63, 0.81, 0.90, 0.80, 0.76, 0.71, 0.75, 0.78, 0.83, 0.82, 0.84, 0.42, 0.49, 0.63, 0.55, 0.52, 0.57, 0.69, 0.63, 0.75, 0.74]
```

| task | policy | successes | rate | mean steps | vs SFT paired: gained / lost | states 2/2 · 1/2 · 0/2 |
|---|---|---|---|---|---|---|
| 0 | SFT, best-of-1 | 63 | 0.63 | 388 | — | 21 · 21 · 8 |
| 0 | RL 660k, best-of-1 | 81 | 0.81 | 355 | 26 / 8 | 32 · 17 · 1 |
| 0 | RL 660k, best-of-4 | 90 | 0.90 | 335 | 34 / 7 | 40 · 10 · 0 |
| 0 | base-conditioned actor (offline probe, 660k critic), best-of-1 | 80 | 0.80 | 347 | 23 / 6 | 34 · 12 · 4 |
| 4 | SFT, best-of-1 | 76 | 0.76 | 317 | — | 29 · 18 · 3 |
| 4 | RL 660k, best-of-1 | 71 | 0.71 | 325 | 8 / 13 | 28 · 15 · 7 |
| 4 | RL 480k, best-of-1 | 75 | 0.75 | 313 | 13 / 14 | 26 · 23 · 1 |
| 4 | RL 660k, best-of-4 | 78 | 0.78 | 300 | 12 / 10 | 32 · 14 · 4 |
| 4 | RL 480k, best-of-4 | 83 | 0.83 | 294 | 15 / 8 | 36 · 11 · 3 |
| 4 | RL 240k, best-of-4 | 82 | 0.82 | 290 | 16 / 10 | 36 · 10 · 4 |
| 4 | base-conditioned actor (offline probe, 660k critic), best-of-1 | 84 | 0.84 | 285 | 14 / 6 | 35 · 14 · 1 |
| 9 | SFT, best-of-1 | 42 | 0.42 | 437 | — | 11 · 20 · 19 |
| 9 | RL 400k, best-of-1 | 49 | 0.49 | 418 | 28 / 21 | 14 · 21 · 15 |
| 9 | RL 400k, best-of-4 | 63 | 0.63 | 382 | 33 / 12 | 21 · 21 · 8 |
| 9 | RL 480k, best-of-4 | 55 | 0.55 | 401 | 32 / 19 | 16 · 23 · 11 |
| 9 | RL 660k, best-of-4 | 52 | 0.52 | 404 | 27 / 17 | 10 · 32 · 8 |
| 9 | base-conditioned actor (offline probe, 400k critic), best-of-1 | 57 | 0.57 | 393 | 33 / 18 | 17 · 23 · 10 |
| 9 | base-conditioned actor (offline probe, 400k critic), best-of-4 | 69 | 0.69 | 365 | 39 / 12 | 25 · 19 · 6 |
| 9 | RL (s, a_base) trained from scratch, 660k, best-of-1 | 63 | 0.63 | 373 | 34 / 13 | 17 · 29 · 4 |
| 9 | RL (s, a_base) trained from scratch, 660k, best-of-4 | 75 | 0.75 | 343 | 43 / 10 | 26 · 23 · 1 |
| 9 | RL (s, a_base) trained from scratch, 480k, best-of-4 | 74 | 0.74 | 353 | 41 / 9 | 29 · 16 · 5 |

"Gained / lost" pairs each RL episode with the SFT episode on the same init state and seed. "States 2/2 · 1/2 · 0/2" counts init states succeeded on both seeds, one seed, neither. Every failure on every policy is a truncation at 520 steps; no policy fails by termination.

## Training runs

Identical recipe (`docs`: README, "DICE-RL residual training"), 660k env steps, one L40S each. Values below are means over 110k-env-step bins of the logged update metrics (10 gradient steps per 4 chunks) or over collection episodes.

| metric | task 0: 0k → 550k+ | task 4: 0k → 550k+ | task 9: 0k → 550k+ |
|---|---|---|---|
| collection success (procedural resets, best-of-4 after 64k) | 0.75 → 0.87 | 0.76 → 0.87 | 0.46 → 0.67 (400k) → 0.48 (640k) |
| train-eval, 10 procedural episodes (0 / 80k / 240k / 400k / 560k) | 6 / 7 / 9 / 8 / 9 | 6 / 6 / 10 / 9 / 9 | 4 / 8 / 5 / 6 / 3 |
| train-eval ΔV at 80k / 240k / 400k / 560k | 0.006 / 0.060 / 0.021 / 0.071 | 0.001 / 0.007 / 0.001 / 0.003 | 0.001 / 0.024 / 0.017 / 0.004 |
| better_than_base_rate | 0.72 → 0.87 | 0.73 → 0.88 | 0.63 → 0.88 |
| q_advantage | 0.017 → 0.040 | 0.018 → 0.064 | 0.019 → 0.102 |
| q_overestimation_online | +0.136 → −0.028 | +0.146 → +0.049 | +0.148 → −0.027 |
| q_overestimation (all rows) | 0.000 → −0.028 | 0.011 → +0.042 | +0.011 → −0.028 |
| residual_rms | 0.010 → 0.015 | 0.011 → 0.018 | 0.012 → 0.027 |
| actor_grad_norm | 0.20 → 0.22 | 0.19 → 0.33 | 0.35 → 0.72 |
| critic_loss | 0.019 → 0.018 (min 0.011 at 220k) | 0.019 → 0.018 (min 0.013 at 220k) | 0.031 → 0.028 (min 0.019 at 200k) |
| bc_filter_rate | ≥ 0.9994 | ≥ 0.9987 | ≥ 0.995 |

Metric definitions, from `script/lingbot_rl_model.py` and `script/lingbot_rl_train.py`:

- `better_than_base_rate`: fraction of (replay row, stored candidate) pairs with Q(s, a_base + residual) > Q(s, a_base), Q = min over the 10 critics. Computed on training-buffer states with the same critic the actor is optimised against.
- `q_advantage`: batch mean of Q(s, a_base + residual) − Q(s, a_base), same critic and states.
- `q_overestimation`: batch mean of Q(s, a_stored) − Monte-Carlo return, where a_stored is the action executed at collection time, not the actor's current output. `_online` restricts to online rows.
- `train-eval ΔV`: mean of Q(s, a_base + residual) − Q(s, a_base) over 8 fresh prior samples at one procedural reset state (`env.reset(seed=0)`), so a single-state measurement outside the replay buffer.
- `residual_rms`: RMS of the residual over the 7 live action channels × 16 steps, normalised action units.

## Task 9: gain at 400k, decline afterwards

Facts:

1. Held-out best-of-4: 63 at 400k, 55 at 480k, 52 at 660k, against SFT 42. Episodes gained versus SFT stayed at 27–33 across the three checkpoints; episodes lost rose from 12 to 17–19.
2. Collection success rose from 0.46 (first bin) to 0.67 at 400k and fell to 0.48 by 640k; mean episode length went 420 → 374 → 407 over the same range.
3. Train-eval ΔV was 0.024 at 240k, 0.017 at 400k and 0.004 at 560k; train-evals were 4 / 8 / 5 / 6 / 3 of 10.
4. From 400k to 640k: online overestimation went from +0.048 to −0.027 (critic Q on executed actions dropped below the Monte-Carlo return), the critic-predicted advantage of the actor's corrections held at 0.095–0.103, actor gradient norm rose 0.53 → 0.72, critic gradient norm 0.67 → 0.90, critic loss 0.020 → 0.028, residual RMS stayed at 0.026–0.027.
5. Task 9's residual (0.027) and actor gradient norm (0.72) are the largest of the three runs; its critic-predicted advantage (0.10) is the highest.

Conclusions that follow:

- The run's best measured policy is the 400k checkpoint; checkpoints after it lose more SFT successes than they gain. Selecting the final checkpoint would have reported +10 instead of +21.
- The decline coincides with the critic turning pessimistic on executed actions while rating the actor's unexecuted corrections ever higher, and with rising actor and critic gradients against a fixed-size residual. This is the task-4 pattern (its facts 2–4) arriving late, after a period in which the residual transferred.
- Across the three runs, the train-eval ΔV ordering at 240k–400k (task 0 > task 9 > task 4) matches the held-out gain ordering; `better_than_base_rate` and `q_advantage` do not.

6. Best-of-1 at 400k: 49/100 (SFT 42, best-of-4 63). The residual alone adds 7 points; the critic's selection adds a further 14.

Not established: the 240k checkpoint's held-out rate (ΔV peaked there).

## Task 4: why a high better-than-base rate did not carry to the held-out eval

Facts:

1. `better_than_base_rate` (0.88) and `q_advantage` (0.064) are the critic's assessment of the actor's outputs on buffer states, produced by the critic the actor maximises. Neither uses environment returns. Task 0 reached comparable values (0.87, 0.040) and its residual-only held-out gain was +18; task 4's was −5. Across the two runs these metrics do not predict realised gain.
2. The only logged quantity that compares the critic to realised returns, `q_overestimation`, is evaluated on the executed action a_stored, not on a_base + residual. It therefore cannot detect a critic that overrates the actor's residual specifically. On task 4 it stayed positive on online rows for the whole run (+0.146 → +0.049), whereas on task 0 it crossed to negative after 330k (−0.028 at the end).
3. Outside the buffer, the critic's predicted advantage for task 4 is near zero: train-eval ΔV was 0.001–0.007 at every checkpoint (task 0: 0.021–0.071), against a buffer-state `q_advantage` of 0.064. The task-4 advantage the critic reports is concentrated on replayed states.
4. Task 4's residual is larger than task 0's (RMS 0.018 vs 0.015) and its actor gradient norm grew through the run (0.19 → 0.33) while task 0's stayed flat (0.20 → 0.22).
5. Held-out best-of-1 at 660k loses 13 SFT successes and gains 8, and the number of init states failing on both seeds rises from 3 (SFT) to 7. Best-of-1 at 480k is 75/100 (12 gained / 13 lost). Best-of-4 recovers to 12 gained / 10 lost at 660k. The 240k and 480k checkpoints, best-of-4, are the only task-4 policies above the prior by more than their interval's half-width (82, 83 vs 76; ±0.08).
6. Collection success in the final bin (0.866, n=396 unseeded episodes, policy changing within the bin) exceeds the held-out best-of-4 rate at 660k (0.78, n=100). Both sample the same start distribution: 200 procedural resets versus the 50 canonical states give coincident per-object position ranges with all 50 canonical states inside them (tasks 0, 4, 9 checked). The two rates' intervals overlap.
7. All task-4 failures, for every policy, are 520-step truncations; successful episodes average 238–253 steps. RL successes are 6–15 steps faster than SFT successes.
8. The BC filter released BC on ≤0.13% of candidate rows in both runs; β = 100 BC applied throughout.

Conclusions that follow from the facts above:

- On task 4 the residual actor learned a correction the critic rates highly on training states (facts 1, 3) that does not improve, and slightly degrades, success on held-out states (fact 5). The critic's positive online overestimation throughout the run (fact 2) is consistent with, but does not by itself prove, the actor exploiting critic error; the logged overestimation metric cannot settle this because it scores the executed action rather than the actor's output (fact 2).
- The task-4 residual never exceeded the prior on its own at either checkpoint measured (75 at 480k, 71 at 660k, prior 76); the task-4 gain that does exist (best-of-4, 240k–480k, +6–7 points) comes entirely from selection. On task 0 the residual alone accounts for most of the gain (63 → 81 → 90); on task 9 for a third of it (42 → 49 → 63). The three tasks differ in whether the learned residual transfers, not in whether the critic learned to rank candidates.
- The train-eval ΔV, computed on a state outside the buffer, separated the two runs at every checkpoint (fact 3) while `better_than_base_rate` and `q_advantage` did not (fact 1). Of the logged signals, ΔV is the one that tracked the held-out outcome.
- Task 4's SFT prior (0.76) starts above the 40–70% band; task 0 (0.63) and task 9 (0.42) start inside it.

Not established by these data: whether the task-4 residual would transfer with a smaller β, a different critic state, or more seeds; whether the late decline from 480k to 660k (83 → 78, within the interval) is real.

## Sharpening analysis and the residual's input

`script/lingbot_rl_sharpen.py` computes, for every online replay row of a run, ΔV = mean over the 4 stored candidates of Q(s, a_base + r) − Q(s, a_base) and ΔH = entropy of the 4 corrected candidates minus entropy of the 4 base candidates (50-bin histogram over [−1, 1] on the 7 live channels × 16 steps, as in the paper's Figure 5; plus a log-std version). Figures: `result/analysis/figures/`.

| run, checkpoint | mean ΔV | mean ΔH | mean Δlog-std | r(ΔV, ΔH) per state |
|---|---|---|---|---|
| t0 240k / 480k / 660k | 0.022 / 0.033 / 0.047 | +0.037 / +0.034 / +0.032 | +0.14 / +0.14 / +0.13 | +0.28 / +0.49 / +0.56 |
| t4 240k / 480k / 660k | 0.025 / 0.056 / 0.063 | +0.053 / +0.051 / +0.048 | +0.18 / +0.17 / +0.16 | +0.35 / +0.43 / +0.60 |
| t9 240k / 400k / 660k | 0.042 / 0.066 / 0.111 | +0.072 / +0.069 / +0.067 | +0.23 / +0.22 / +0.21 | −0.03 / +0.10 / +0.43 |
| t9 (s, a_base) run, 240k / 400k / 660k | 0.163 / 0.177 / 0.192 | +0.013 / +0.009 / +0.008 | +0.07 / +0.06 / +0.06 | +0.12 / +0.09 / +0.09 |

Facts: mean ΔH is positive on all nine combinations (the residual spreads the candidates; the paper reports contraction with r = −0.18); r(ΔV, ΔH) rises to +0.43..+0.60 by 660k on every task. On the buffers, z adds nothing to predicting a_base beyond the state (ridge R² 0.47–0.55 with or without z), and the candidate-specific part of the residual is uncorrelated with the candidate's offset (−0.04 to −0.08). The residual's input (s, z) does not locate the candidate it corrects: with this base policy a_base is a 50-step denoise of the 5B model, unlike the paper's small flow.

`script/lingbot_rl_actor_probe.py` retrains fresh actors on a run's buffer against its frozen critic with inputs (s, z), (s, a_base), (s, z, a_base); held-out rows (`result/analysis/probe/summary.csv`):

| run | input | critic gain ΔV | Δlog-std | states sharpened | candidate-specific share | offset corr |
|---|---|---|---|---|---|---|
| t0 660k | z / base / z_base | 0.004 / 0.050 / 0.017 | +0.22 / +0.04 / +0.20 | 1% / 52% / 3% | 65% / 8% / 52% | −0.04 / −0.17 / −0.06 |
| t9 400k | z / base / z_base | 0.006 / 0.133 / 0.053 | +0.30 / +0.07 / +0.25 | 0% / 38% / 1% | 68% / 8% / 45% | −0.06 / −0.19 / −0.08 |
| t4 660k | z / base / z_base | 0.004 / 0.070 / 0.019 | +0.23 / +0.06 / +0.21 | 5% / 50% / 9% | 64% / 10% / 51% | −0.04 / −0.13 / −0.05 |

The (s, a_base) probe actors, each paired with its run's critic, scored on the held-out protocol (table above): task 9 at 400k, 57 best-of-1 and 69 best-of-4 (z-actor: 49 / 63); task 0 at 660k, 80 best-of-1 (z-actor: 81); task 4 at 660k, 84 best-of-1 (z-actor: 71; SFT 76). No new environment data was used for any of them. `residual_input` is now a recipe option (`z`, `base`, `z_base`); a training run with `base` is the pending test.

## Task 9 retrained with the base-conditioned residual

Run `dice-t9-base`: identical recipe, `residual_input: base`, actor and critic trained together from scratch for 660k env steps (W&B `dice-t9-base`). Held-out at 660k: best-of-4 75/100 (43 gained / 10 lost vs SFT; 35 / 12 vs the z-run's 660k; 26 / 14 vs the z-run's best checkpoint at 400k), best-of-1 63/100 (34 / 13 vs SFT; 30 / 16 vs the z-run's 400k best-of-1). Training signals over the same range as the z run: train-evals 4 / 5 / 6 / 6 / 8 (z: 4 / 8 / 5 / 6 / 3), ΔV 0.15–0.22 at every checkpoint (z: 0.001–0.024), collection success 0.77–0.78 at 400k–480k (z: 0.59–0.66), online overestimation negative throughout (−0.02 → −0.11), critic loss flat at 0.028 and actor gradient norm plateaued at 0.85 from 240k (z: both rising late), residual RMS 0.035 (z: 0.026). The late decline of the z run did not occur. The 480k checkpoint, best-of-4, scores 74/100 (41 gained / 9 lost vs SFT; 15 / 16 vs the 660k checkpoint), so the last 180k env steps changed which episodes succeed but not how many. On its own replay buffer the trained (s, a_base) actor leaves the candidate spread almost unchanged (mean ΔH +0.008, Δlog-std +0.06 at 660k, against +0.067 / +0.21 for the z run) while the critic-predicted gain is three times larger (0.19 vs 0.07), and the ΔV–ΔH coupling stays near zero at every checkpoint (r = 0.09–0.12) instead of rising to +0.43.

## Decomposition across tasks

| task | SFT | residual only (best-of-1) | residual + selection (best-of-4) | checkpoint |
|---|---|---|---|---|
| 0 | 63 | 81 (+18) | 90 (+9 more) | 660k |
| 9 | 42 | 49 (+7) | 63 (+14 more) | 400k, (s, z) |
| 9 | 42 | 63 (+21) | 75 (+12 more) | 660k, (s, a_base) |
| 4 | 76 | 75 (−1) / 71 (−5) | 83 / 78 (+7 / +2) | 480k / 660k |

Selection by the trained critic contributed on all three tasks (+9, +14, +7); the residual's own contribution ranged from +18 to −5 and ordered the tasks the same way the train-eval ΔV did.
