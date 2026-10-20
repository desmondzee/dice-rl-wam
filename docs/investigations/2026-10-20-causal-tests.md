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
