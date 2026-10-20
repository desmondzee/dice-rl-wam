# Why the residual is inert or collapses: investigation of 2026-10-20

Five independent investigations over all nine task 9/0 runs (W&B histories, every reachable checkpoint and replay buffer, the code at each run's commit, a local reproduction with the real model code, and the literature). Reports and scripts: the session scratchpad `investigation/` folders and `~/Documents/dice/scratch/investigation/` on both boxes. Numbers below are measured unless marked inferred.

## What sets the residual's size

- One model fits every run with a BC term. `update_actor` puts each online row's residual at r* = (16·7 / 2β) · dQ/da / q_scale = 0.56 · g / q at β = 100; measured medians are 0.42–1.28× this in 14 of 15 snapshots, and the logged rms follows within a factor of 2.4 from 0.0009 to 0.022. The actors sit at this balance from 4 actor steps after activation onward (Q-term and BC-term parameter gradients have cosine −0.68 to −0.98).
- q_scale is mean |Q| over online rows only: cf 0.29, plan-m4 0.43–0.53, std-bc 1.11, ood-bc 0.72–0.76. The W&B `q_mean` (1.45–1.6 in the BC runs) includes the 50% expert rows whose Q ≈ 2.2 and is not the normaliser.
- The BC runs' 15–45× smaller residual decomposes (against cf at 160k) into q_scale ×2.5–3.8, per-entry dQ/da ×2.2–4.4 and actor capacity ×1–2; against the in-distribution plan run, dQ/da (×4–8) dominates.
- Bound-only: with no BC term Adam's normalised steps move the zero-initialised output by lr · Σ|h| per step whatever the gradient's size (predicted 0.0060–0.0064 at the first update, logged 0.0061 / 0.0063), the residual saturates the tanh in 30–60 actor steps (under 3k env steps) before the ramp applies it to any chunk, and saturated tanh then freezes it.

## Why the critic barely sees the action (the common factor of run B and both new recipes)

- In every critic fed structured features the action supplies 2.7–4.0% of the first-layer pre-activation, against 31–62% in the pooled critics. Global scalar standardisation keeps the per-dimension means: the standardised state has norm 88, of which the mean vector is 85 (93–94% of the energy), and LayerNorm divides the action's small contribution by that constant pattern. Within a state the four candidates differ by 0.03 per entry against 0.44 across states, so the action moves the first layer about 1% as much as the state does. dQ/da orders with the action share across runs (correlation, not yet tested causally).
- The post-run-B decision was to standardise structured inputs "to the pooled scale"; the spec and code standardised to unit variance per element instead (norm 88, twice run B's unstandardised 39).
- The privileged object state adds nothing measurable: Q_p and Q_d generalise (0.58 / 0.58 out of sample) and extrapolate identically.

## The critics cannot rank actions within a state, old or new

- cf run, stored per-candidate outcome labels, out of sample: hit rate of the label-best candidate 0.269 vs 0.251 chance, Spearman +0.004 ± 0.006, value captured −0.006; the ranking is consistent across heads but changes across checkpoints. New runs: the selection critic hits the label-best candidate 0.23–0.34 of the time (chance 0.25); 48–64% of labelled chunks have identical one-chunk labels for all four candidates.
- The new critics are V(s) plus one learned, mostly state-independent action tilt whose direction differs between runs started from identical weights; V(s) is memorised (0.99 in sample, 0.54–0.63 out of sample). None penalises out-of-distribution actions: along the actor's direction Q rises linearly (bound-only) or convexly (BC) to 2.5× the bound; only the pooled cf critic is concave, and it still predicts a gain at the bound.
- The bound-only residual moved the end effector 1–3 cm per applied chunk in a direction that differed between boxes; the arm stopped reaching a grasp pose (gripper closed in 6–11% of chunks against 21%), and success went to 0 once 15–30% of chunks carried it. The causal effect of applying it was −0.03 ± 0.014 on return-to-go per chunk while every critic predicted +0.002 to +0.02.
- The BC runs' residual is a near-constant 0.1 mm-per-step drift with no measurable effect; training success equals the base policy's.

## Shaping: correctness bugs, not the cause of the small residual

- The critic learns exactly the defined return (identity checks to 2e-15): Q ≈ Q_sparse − Φ(s) + γ^τ Φ(s_T). Φ(s_T) is not zeroed at success or at the 520-step timeout (treated as terminal), so success pays about 2.95 instead of 1 and a timeout pays its final Φ; policy invariance is broken and the spec's "per-episode constant" is false.
- The inflation of q_scale comes from the kept terminal potential, which should scale dQ/da by about the same factor, so the normalised gradient is roughly unchanged by shaping (inferred). Normalising by Q + Φ(s) or by a batch-centred Q does not help.
- Further defects: the 0.12 m "inside" radius pays near-success value for 12–15% of failed episodes (mug 0.105 m from the microwave origin, door 5–12% open); the ±5 clamp with a 0.05 std floor hides the mug's orientation on 29% of online rows (median rotation 86°).

## Other findings

- Coupled Adam weight decay (1e-5, as in the reference) shrinks every actor weight whose loss gradient is negligible by about one learning-rate step per update: hidden weights −85–96% and LayerNorm gains 0.07–0.18 in run B and the BC runs, versus a healthy trunk in the pooled runs. It follows from the small residual and locks it in; it is not the cause (the small balance is reached before the trunk shrinks).
- The new recipe updates 4× more per env step than the old (not 2.5×); the expert ratio fixed at 0.5 halves the Q term's weight; several logged metrics now include expert rows or the bound factor and do not compare across recipes; selection uses the min over heads (the spec says mean).
- Ruled out by measurement: gradient incoherence, min-over-heads switching, curvature in the BC critics, the actor's own input share, learning-rate phase and cadence, the BC filter, any unused-channel floor, arithmetic bugs in the actor, critic, targets, selection or buffer.

## What this changes

- The proposed fix "normalise by Q + Φ(s)" is not supported: it would make the normalised gradient smaller, and shaping is not what shrank the residual.
- The earlier in-distribution gains did not come from a critic that ranks candidates; with pooled features the critic carried a coarse, state-dependent action tilt that the residual followed. The structured-feature recipes removed even that.
- Causal tests that settle the open mechanism cheaply, on existing buffers and without new rollouts: refit the critic with per-dimension mean-removed or pooled-scale structured features and measure the action share, dQ/da and within-state ranking against the cf labels; the same with an action pathway that bypasses the first LayerNorm; the actor with decoupled or no weight decay on a resume copy.
