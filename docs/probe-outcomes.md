# Probe and test outcomes

One line per measurement: what was run, the number, what it decided. Details and tables are in `docs/heldout-results.md`; raw files under `result/`. Task 9 unless stated; success rates are out of the standard 100 held-out episodes unless stated.

## Prior and sampler

- Released 20 / 50 sampler vs the authors' 5 / 10: SFT task 9 42 → 60, task 0 63 → 75; stopping the 5-step video sampler after pass 4 gives 39. Decision: 5 / 10 for everything since.
- SFT checkpoints 200 / 400 / 600 on task 0 (50 episodes): 38 / 33 / 38. No checkpoint puts task 0 in the 40–50 band; task 8 at step 600 is 12. The LIBERO-90 prior plan was drafted and put on hold.
- Video guidance 1 / 3 / 5 (SFT, 50 episodes): 25 / 34 / 31. Plan diversity at the same noise is identical at all three scales (across-plan action spread 0.051–0.052, pool spread 0.004). Decision: keep 5; guidance moves plans together, it does not spread them.
- Plan diversity in distribution: across-plan base-action spread 0.05–0.06 vs within-plan 0.013–0.019; plan pools differ by about 10% of their magnitude. The spread is set by the 16-action horizon and the demonstrations, not by the sampler.

## Plan selection with the Q-derived plan critic

- M = 4 plans, V_plan ranker, residual, Q best-of-4: task 9 79 at 160k, 80 at 320k (SFT 60); task 0 96 at 240k (SFT 75). Second half of task 9 flat.
- Q ranking over all 16 residual candidates instead of V_plan: 74 (14 / 19 paired vs 79), 190 s per episode. Not used.
- Random plan instead of V_plan, same checkpoint: 75 on the standard protocol (12 / 17 paired vs 80), 50 on LIBERO-Plus (17 / 20 vs 53; robot-init 8 / 12, layout 9 / 8). V_plan's ranking is worth 3–5 points, inside the noise; the residual with Q over 4 candidates carries the gain.
- LIBERO-Plus task 9 (100 episodes, 42 robot-init + 39 layout variants with seeded repeats): SFT 37 (robot-init 12 / 52, layout 25 / 48); plan policy 320k 53 (26 / 52, 27 / 48), paired 29 / 13.

## What limits the plan critic

- Live V_plan captures 0.73 (160k) / 0.81 (320k) of the Q target's value in sample; agreement with Q 0.51 / 0.55. Every fresh refit on the same target (MSE, Polyak target, centred, pairwise, regularised, HL-Gauss) is worse out of sample (0.41–0.62 captured).
- Q half-ensembles (heads 1–5 vs 6–10) agree on the best plan 0.63 (task 9) / 0.44 (task 0); they capture 0.89 of each other's value on task 9.
- Heads on bootstrap or disjoint rows (MC-regression refits): replica agreement 0.60 shared → 0.46 bootstrap → 0.42 disjoint; spread-vs-error rises only 0.43 → 0.53. Closed: splitting data across heads makes ranking less consistent.
- Ranking by the mean over heads instead of the minimum: task 9 split-half 0.63 → 0.71, task 0 0.44 → 0.56 with the outcome link 0.15 → 0.39; the min inflates within-state plan spread 2.5× on task 0 (pessimism noise). Max vs mean over the 4 candidates: no difference. V_plan's own mean readout: self-consistency 0.68 → 0.78 but outcome link unchanged. Decision: mean over heads for ranking; min stays in the TD target and actor loss.
- Fresh V_plan refits on the min-max target vs mean targets: 0.39–0.45 agreement with the Q-mean ranker out of sample on both tasks (live 0.55 / 0.48, chance 0.25), outcome link about 0 or negative, whichever target. The target statistic is not V_plan's limit; learnability from pooled (s, p) features is.
- Feature study (run B 80k buffer, 5,087 chunks): with structured plan features (per-camera, per-frame means) every state input ranks alike, agreement 0.56–0.58 and captured 0.77 (pooled state, structured state, attention over the 32 state tokens, ground-truth object poses); with the pooled plan vector even the object-pose oracle falls to 0.41 / 0.48. Outcome link 0.23 with structured plan features vs 0.11–0.13 pooled. Decision: plan-side features are the lever; the state side and the object-pose oracle are closed; the encoder ablation targets the 128 plan tokens.

## Does the method contract the distribution

- Plan level: pool dispersion of executed plans equals that of all four plans (0.0185) in both V_plan runs; ΔH within ±0.005 nats, no trend. Action level: ΔV rises (0.006 → 0.055 task 9, 0.005 → 0.022 task 0) while ΔH stays at zero and the executed actions' log-std ends 4–8% above the base candidates. Per state, ΔH vs ΔV: r² 0.05 (task 9) and 0.00 (task 0); earlier (s, z) runs had r² up to 0.31 in the spreading direction; the paper reports r = −0.18 contraction. No run contracts; the residual is a state-dependent translation plus selection.
- Actor-input probe (earlier): about 92% of the residual is common to all candidates of a state, 8% candidate-specific; the (s, z) input spreads candidates it cannot locate.

## Counterfactual plan labels (run `dice-t9-plus-m4-pool-cf`, about 140k steps)

- Shadow fidelity: 0.0 m and 0.0006 rad on the base task, 0.0 m and 0.0005 rad on a layout variant; door calibration closed 0.0, open −2.094, initial −1.43.
- Labels over 9,142 chunks: across-plan spread 0.014, within-plan 0.008, mean 0.07, success jumps in 2%; the same on all three reset kinds. The executed plan sits 0.003 above the plan mean; it is the label's best plan 28.5% of the time (chance 25).
- Live V_plan vs labels, held out: correlation 0.94 with the label value, agreement 0.33 with the label's best plan, captured 0.60; split-half 0.70; agreement flat 0.32–0.36 across the run.
- Live Q vs the same labels: agreement 0.28, captured −0.10. The Q-derived plan ranking is unrelated to measured plan outcomes.
- Short-horizon label vs episode return: 0.05–0.08 correlation.
- Fresh five-head refits on the labels (3,000 steps, held out by time): pooled plan features agreement 0.30 / captured 0.18 / correlation 0.71; structured plan features 0.28 / 0.30 / 0.71. Both below the live critic (0.33 / 0.60 / 0.94), which has had 140k steps of online updates; the refit is too weak a proxy to settle the feature question on these labels, and the within-state ordering is the hard part (plan-to-plan label differences of 0.014 against level variation an order of magnitude larger).

## LIBERO-Plus at 240k, both runs stopped (2026-10-18)

- Same 100-episode LIBERO-Plus plan as the baselines: counterfactual 240k 44 (robot-init 21 / 52, layout 23 / 48), run B 240k 48 (23 / 52, 25 / 48), against SFT 37, the in-distribution 320k plan policy 53, random plan 50. Training variants (67): 26 and 34 against 37; held-out variants (33): 18 and 14 against 16 and 17. Paired vs the 320k policy: 15 / 24 and 18 / 23. Decision: training on the perturbation mix, with either critic, did not improve LIBERO-Plus over the in-distribution policy, not even on the variants it trained on; the measured plan critic's per-chunk advantage (0.001–0.006 against a spread of about 0.01) is too small to move success.

## Structured features in the live model (run B, 240k)

- Run B's actor is inert: `residual_rms` 0.0015, `q_advantage` 0.0002, `actor_grad_norm` 0.017 at 240k against 0.02 / 0.03–0.05 / 0.5–0.7 in every pooled run. The structured state input has row norm 39 against 1.5 for the pooled vector (plan input 138 against 8.4); the critic's dQ/da median is 8× smaller (0.0028 vs 0.0235) and the first-layer action columns never left initialisation. Decision: any structured-feature recipe standardises the inputs to the pooled scale before the critic; run B measures structured features for the plan critic and Q selection only, not for the residual.
- Training-episode success (mixed reset kinds, 40k windows) is flat after 120k in both runs: counterfactual 0.38–0.51, run B 0.51–0.53; the counterfactual run's executed-plan label advantage is 0.001–0.006 against an across-plan spread of 0.007–0.017.

## Bounded-residual smokes (2026-10-19)

- At M = 1 with per-chunk updates the loop runs at 0.19 s per env step (0.56–0.61 with four plans), peak GPU 21.8 GB, so 240k steps take about 13 h.
- Within-chunk measured outcome differences between the 4 residual candidates are tiny (label spread 0.0005–0.0017); candidate-level label agreement is rarely measurable.
- With no BC term the residual reaches its bound quickly when the warm-up is short (W&B saturation 0.44–0.48 after 5,500 steps with a 1-episode warm-up); the real runs use a 50-episode warm-up and a 50k ramp.
- Standard resets in the earlier LIBERO-Plus runs used the stored initial states, not procedural placement (`stored = bool(plus_train) and kind != 1`); fixed to `kind == 2`.

## Bound-only residual collapses (first launch, 2026-10-19)

- With α · tanh at α = 0.3 and no BC term, the deterministic actor reached the bound at its first update (residual rms 0.26, 55–63% of entries within 5% of the bound) while the critic's advantage was 0.01; once the ramp applied it, training success fell from 0.55–0.60 to 0 by 40k env steps on both boxes, standard-reset control included, and stayed at 0 to 88k. Saturated tanh has near-zero gradient, so the actor cannot come back. Policy Decorator and PLD avoid this with a stochastic entropy-regularised actor (and PLD with a Cal-QL-pretrained critic); our deterministic actor needs the BC term. Decision: BC weight 100 with the filter at ε = −0.3 restored, bound kept as a cap.

## Residual and critic investigation (2026-10-20; `docs/investigations/2026-10-20-residual-and-critic.md`)

- The residual sits at r* = 0.56 · dQ/da / q_scale (β = 100) in every run with a BC term (14 of 15 snapshots within 0.42–1.28×); the BC runs' residual is 15–45× below cf's because q_scale is 2.5–3.8× larger and per-entry dQ/da 2–8× smaller.
- Structured features standardised with one global mean and std keep a constant per-dimension pattern (93% of the state's energy, norm 88), so the action supplies 2.7–4% of the critic's first-layer pre-activation against 31–62% with pooled features; the common factor of run B and both new recipes.
- No critic ranks actions within a state: cf out-of-sample Spearman against per-candidate labels +0.004, captured −0.006; new selection critics hit the label-best candidate 0.23–0.34 (chance 0.25). Critics extrapolate linearly or convexly along the actor's direction; only the pooled critic is concave.
- Shaping kept Φ(s_T) at success and timeout (success worth 2.95, invariance broken); it inflates q_scale but should scale dQ/da equally; normalising by Q + Φ would not help. The 0.12 m inside radius pays near-success value to 12–15% of failures; the ±5 clamp hides the mug's orientation on 29% of rows.
- Coupled Adam weight decay collapses actors whose residual is pinned small (hidden weights −85–96%); a consequence that locks the residual in, not its cause.
- W&B `q_mean` includes expert rows (the actor normaliser does not); earlier "dQ/da 0.025–0.036" were per-chunk norms over the last 300 rows (per-entry 0.001).

## Offline causal tests (2026-10-20; `docs/investigations/2026-10-20-causal-tests.md`)

- Weight decay: removing Adam's coupled decay keeps the actor's weights intact (hidden 1.10 vs 0.11 of start) but leaves the residual the same size (0.0020 vs 0.0024 rms, both at the critic-set equilibrium) and as input-dependent; the last hidden layer's output collapses either way. Not a lever.
- Critic inputs: refitting the cf critic on six representations moves the action's share of the first layer from 0.03 to 0.58 and dQ/da by 10–20× without changing held-out candidate ranking: every converged TD critic sits inside the untrained-critic distribution (best Spearman +0.06, null sd 0.064). Trained critics learn a reproducible action tilt orthogonal to the measured-outcome direction. The outcome ordering is learnable from the action alone (linear ridge on 112 action entries: +0.33, hit 0.44 vs 0.25) and needs no state. The limit is the TD training signal, not the representation.

## DICE-RL reproduction (2026-10-20; `docs/dice-rl-reproduction.md`)

- The authors' released Tool Hang checkpoint evaluates at 92.0% (fine-tuned) / 40.7% (pretrained) on their 300 held-out seeds under mujoco 2.3.7 (authors: 94% / 50% on 50 episodes; 49.7% pretrained in the 300-episode log). mujoco 3.8.1 lowers the fine-tuned policy to 85.3%: the simulator version shifts contact-rich success by 7 points, so the reproduction pins mujoco 2.3.7 on Python 3.11 with torch 2.11 (the GB10's build).

## Throughput and engineering facts

- Open-loop replay of the LeRobot demonstrations in the simulator does not reproduce them: with the initial state matched to about 0.01 per proprioception entry, only 7 of 30 task 9 demos reach success and 20 never place the mug (median max end-effector gap 0.065 m). The original LIBERO demonstration files (HF `yifengzhu-hf/LIBERO-datasets`, `libero_10/<task>_demo.hdf5`, 1.5 GB for task 9) store the flattened MuJoCo state (47 floats) at every frame; our 30 LeRobot episodes match `demo_k` exactly by actions (episode 450 + k ↔ demo_k), and loading those states reproduces the demos (progress −0.34 → 1.97, gripper site within 1.1 cm of the dataset proprioception). Expert object states come from those files; the files live under `~/Documents/dice/libero-demos/` on both boxes.
- Procedural arm poses: overriding the Panda model's `init_qpos` before a fresh environment build places the arm at the override exactly (joint error 0.0 at an l2 offset of 0.4 rad; end-effector moves 2–17 cm); the base task on the LIBERO-Plus package uses the same `MountedPanda` class. LIBERO-Plus robot-init variants are random joint offsets with l2 norm 0.1–0.3 rad at levels 1–2 and 0.2–0.5 at levels 3–5 (per-joint std 0.10–0.14).
- GB10 is GEMM-bound (cutlass kernels about 70% of 100 TFLOPS); batching 4 plans saves 10% over looping; 0.54 s per env step (pooled run), 0.56 (structured), peak memory 38.7 / 47.8 GB; evaluation 65 s per episode SFT, 120–130 s at M = 4, 155 s on LIBERO-Plus.
- Labelling: thread worker 0.72 s per step (interpreter lock defeats the overlap and slows the main loop); four worker processes 0.61 s per step with 0.35 s label time and 0.02 s wait per chunk.
- A second model process beside a rollout run with its workers exceeds the 121 GB unified memory and the kernel kills the run.
- Shadow environments must be built with the live episode's reset seed: LIBERO places fixtures at model build from the seeded sampling, and a shadow from another seed sits about 1 cm off. The live pose snapshot lags its qpos by one physics substep, so pose sanity checks need a 1 cm tolerance, not 1 mm.
- The LIBERO-Plus `libero_10` suite holds 2,519 variants and no base tasks; the base bddl and stored initial states exist in its checkout and are appended as task 2519. Its assets render the same physical state 0.36 / 255 differently per pixel.
- Proprioception from the environment matches the dataset's 8-float end-effector convention to 0.016 per entry (the settle steps account for the gap).
- The published parquet `observation.state` is end-effector position, axis-angle, gripper; robosuite's quaternion is x-y-z-w; MuJoCo's body quaternions are w-x-y-z.
- Resume files carry the protocol fingerprint; `label_workers` and `store_tokens` are outside it, `plan_target` inside; `rsync` drops the execute bit on shell scripts; the Brev tunnel drops long sessions, so long jobs run in box-local tmux.
