# Offline causal tests (plan, 2026-10-20)

Follows `2026-10-20-residual-and-critic.md`. Offline only, on Jarvis (both boxes idle; Watson released), on existing buffers; no rollouts. Scripts live in `~/Documents/dice/scratch/investigation/causal/` on Jarvis, not in the repo.

## Test A: does the critic's input representation decide whether it sees and ranks actions?

Data: the cf run buffer (`runs/dice-t9-plus-m4-pool-cf`), the only buffer with measured per-candidate outcome labels (16 base candidates per chunk, one-chunk Δφ + 2·success; 15,870 online chunks). It also stores structured features, proprioception and object state. Basis: observation (critic report §1a).

Split: the last 20% of online chunks by time held out, as in the earlier cf check. Basis: prior protocol, so numbers compare with the live critic's +0.004 Spearman.

Fit: the live critic exactly (`_mlp` 3 × 1024 with LayerNorm, 10 heads, Adam 1e-4, weight decay 1e-5, cosine restarts, clip 1.0, batch 256, τ 0.01 target network, 3-chunk n-step, γ 0.99, min over heads), on the buffer's own sparse rewards, with next actions the stored next base candidates. The steps run until the TD loss over the last 2,000 steps changes by less than 5% (cap 40,000). Basis: mirrors the live update, so differences come from the inputs; the earlier 3,000-step refits underfit (probe outcomes, cf check).

Variants:
- V0: pooled + proprio, the cf critic's own inputs. Control: calibrates the refit against the live critic.
- V1: structured, one global mean and std (the current recipe).
- V2: structured, per-dimension mean and std from expert rows. Removes the constant pattern measured at 93% of the state's energy.
- V3: V1 rescaled so the mean row norm equals V0's. Changes scale only, keeps the pattern. V1 vs V3 isolates scale, V1 vs V2 isolates the pattern.
- V4: V1 with the action entering at the second layer. Basis: the DDPG critic (Lillicrap et al. 2015, arXiv 1509.02971) inserts actions at the second hidden layer.
- V5: V2 plus the standardised object state (the privileged critic).

Metrics (definitions as in the critic report): action share of the first-layer pre-activation (V4: of the second); per-entry |dQ/da| median and mean at the stored action; within-state Q spread over the 16 candidates and within each plan's 4; held-out within-plan Spearman, label-best hit rate and captured value against the labels (rows with label spread > 1e-3); held-out correlation of Q with Monte-Carlo return.

Ceiling: for V0 and V2, a model of the same size regressed directly onto the within-state-centred per-candidate labels. If even this cannot rank held-out candidates, the inputs are not the limit and the outcome signal or data is. Basis: the open question of the investigation (inputs vs data).

## Test B: does decoupled weight decay prevent the actor's collapse?

Data: the ood-bc run (`runs/dice-t9-ood-bounded-bc`): the step-0 actor, the 80k critic frozen, the 80k buffer rows.

Fit: `update_actor` exactly as in the run (β 100, ε −0.3, bound 0.3, expert ratio 0.5, batch 256), for the number of actor updates the run had made by 80k (about 18,600), under three optimisers: Adam with weight decay 1e-5 (the run), AdamW with 1e-5, and no weight decay. Basis: decoupled weight decay (Loshchilov & Hutter 2017, arXiv 1711.05101) removes the coupling the audit identified.

Metrics: hidden-weight rms and LayerNorm gains against step 0 over the updates; residual rms; the share of the residual that varies across states (input dependence); the predicted equilibrium 0.56 · g / q.

## Result of Test B (2026-10-20)

The run made 18,600 actor updates by its 80k checkpoint (3,720 logged update calls of 5 actor steps; the resume's actor scheduler count confirms the rate). The Adam arm reproduces the run's weight shrinkage approximately (LayerNorm gains 0.26 / 0.21 / 0.19 vs the run's 0.19 / 0.18 / 0.16; hidden weights 0.11 / 0.09 vs 0.06 / 0.06), with residuals about 2× the run's because the frozen 80k critic is the one difference from the run (the run's own 80k actor continued against it moves the same way); the comparison between arms holds because the critic is shared.

| at 18,600 updates | Adam, weight decay 1e-5 (the run) | AdamW 1e-5 = no weight decay (bitwise identical) |
|---|---|---|
| hidden weights vs step 0 | 0.11 / 0.09 | 1.10 / 1.09 |
| LayerNorm gains | 0.26 / 0.21 / 0.19 | 1.00 / 1.00 / 0.91 |
| last hidden layer output rms (step 0: 0.65) | 0.027 | 0.033 |
| held-out residual rms / median | 2.4e-3 / 2.9e-4 | 2.0e-3 / 5.5e-4 |
| median vs predicted 0.56 · g / q (5.6e-4) | 0.54× | 1.05× |
| residual share that varies across states | 0.92 | 0.96 |

- AdamW at weight decay 1e-5 is a no-op in fp32 (1 − lr · wd rounds to 1), so it equals no decay; an AdamW decay large enough to act was not tested.
- Removing the coupled decay prevents the weight shrinkage completely but not the collapse of the last hidden layer's output (0.65 → 0.03 in every arm, a few units with large negative pre-activations dominating that LayerNorm; inferred mechanism, not tested).
- It does not change the residual's size in any useful sense (both below 1% of the 0.3 bound) or its input dependence.
- Conclusion: coupled weight decay causes the weight-norm collapse, not the small residual. The residual's size is set by the critic's g / q, as the equilibrium model says. Weight decay is not a lever; the critic is.

## Result of Test A (2026-10-20)

All fits converged (stop rule: 2,000-step mean TD loss changing < 5%, 12k–25k steps), two seeds each; held out: the last 20% of the cf buffer's online chunks (2,594 plan rows with label spread > 1e-3). Chance reference: 20 untrained critics give within-plan Spearman mean −0.002, sd 0.064, max +0.134 (the planned shuffled-label nulls were not run; untrained critics stand in). The seed-to-seed rank agreement of trained critics is 0.75–0.82 (nulls 0.06–0.17), so trained critics learn a reproducible ordering.

| fit | action share (layer 1; V4: layer 2) | per-entry \|dQ/da\| median | within-plan Q spread | held-out within-plan Spearman (two seeds) | hit vs 0.252 | value corr with MC, held out |
|---|---|---|---|---|---|---|
| V0 pooled + proprio + plan pool (the cf critic's inputs) | 0.58 / 0.57 | 0.0042 / 0.0039 | 0.0115 | +0.021 / +0.034 | 0.274 / 0.281 | 0.71 |
| V1 structured, global standardisation (current recipe) | 0.075 / 0.083 | 0.00014 / 0.00018 | 0.0039 | +0.063 / +0.059 | 0.300 / 0.294 | 0.66 |
| V2 structured, per-dimension stats from expert rows (keeps 78% of the pattern, norm 122) | 0.037 | 0.00005 | 0.0021 | +0.061 / +0.048 | 0.291 | 0.64 |
| V2o structured, per-dimension stats from online rows (pattern removed, norm 78) | 0.074 | 0.00008 | 0.0015 | +0.017 / +0.022 | 0.274 | 0.63 |
| V3 V1 rescaled to the pooled norm | 0.45 / 0.39 | 0.0024 / 0.0017 | 0.0105 | +0.040 / +0.051 | 0.283 / 0.277 | 0.67 |
| V4 V1 with the action at the second layer | 0.37 / 0.36 | 0.00067 / 0.00074 | 0.0036 | +0.056 / +0.055 | 0.293 | 0.66 |
| V5 V1 + standardised object state (object share 0.01) | 0.030 | 0.00004 | 0.0020 | +0.053 / +0.039 | 0.290 / 0.281 | 0.65 |
| live cf critic 80k / 160k / 240k / final | 0.57 / 0.53 / 0.50 / 0.49 | 0.0048 / 0.0028 / 0.0018 / 0.0017 | 0.0107–0.0086 | +0.014 / +0.034 / +0.026 / +0.021 | 0.268–0.285 | 0.62 / 0.69 (240k and final saw these rows) |
| Cv0 ceiling: V0 inputs regressed on centred labels | 0.44 / 0.41 | — | 0.0007 | +0.196 / +0.218 (train 0.22 / 0.24) | 0.367 / 0.371 | — |
| Cv2 ceiling: V2 inputs regressed on centred labels | 0.009 / 0.007 | — | 0.0001 | +0.288 / +0.302 (train 0.29 / 0.31) | 0.418 / 0.422 | — |
| linear ridge on the 112 action entries alone, no state | — | — | — | +0.330 ± 0.011 (train 0.338; fitted on half the rows: +0.320) | 0.437 | — |

- The V0 refit reproduces the live cf critic: chance-level ranking, action share, dQ/da, Q spread and value correlation all match.
- Input scale sets how strongly Q responds to the action (V1 → V3: share 5×, median dQ/da 12×; V4 gives the action a 0.36 share at its layer) but no representation changes the ranking: every converged TD critic lies inside the untrained-critic distribution (best +0.06 against a null sd of 0.064 and max +0.134). The constant pattern is not what suppresses the action (V2o, with it removed, has the same share as V1); the input norm is. The object state adds nothing.
- Trained TD critics learn a reproducible action tilt that is orthogonal to the label direction (cosine −0.10 to +0.09): a consistent ordering, but not the outcome ordering. Structured variants transiently reach +0.10–0.15 around 2,000 steps and fall back as the TD loss drops.
- The outcome ordering is learnable and needs no state: both label-supervised ceilings rank held-out candidates (+0.20–0.30, no overfitting), and a 112-parameter linear model on the action alone does better (+0.33, hit 0.44). Cv2's within-state Q is 96% a linear action tilt; the labels correlate with the candidate's summed y displacement at 0.29–0.31. No state-dependent ranking signal was found (a crude tilt-removed check gives negative ρ; weak evidence).
- Conclusion: the critic's inputs are not what stops it ranking actions. The one within-state signal in the measured labels is a fixed action direction that the TD critics never learn, because their sparse n-step targets do not contain it (inferred). The limit is the training signal, not the representation.
- Caveats: fits use online rows only (expert ratio 0); nulls are untrained critics; the Cv ceilings' MSE equals the predict-zero loss (64% of squared label mass sits in 1% of rows); the labels are one-chunk Δφ + 2·success, which correlate only 0.05–0.08 with episode return, so matching them is not the same as estimating the return.
