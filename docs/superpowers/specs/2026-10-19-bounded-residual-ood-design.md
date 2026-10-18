# Bounded residual for out-of-distribution performance: design

Status: reviewed 2026-10-19 (bound alone, α = 0.3, box 2 is the data control). Evidence: `docs/research-residual-ood-survey.md` (literature), `docs/probe-outcomes.md` (our measurements).

## Goal

Make the residual actor, not plan selection, the object of the method, and measure whether it can improve LIBERO-Plus performance on held-out perturbations when it is (a) large enough to act, (b) trained by a critic with a real action signal, and (c) trained on perturbed data. Two runs with one recipe: box 1 on the perturbed distribution, box 2 on standard resets, as the control that separates the recipe from the data.

## What changes against the current recipe

| component | current (`dice-t9-plus-m4-*`) | new | basis |
|---|---|---|---|
| plans | M = 4, V_plan ranker | M = 1, no plan critic | measured plan leverage 0.01 per chunk; random plan within noise |
| candidates | K = 4 noise samples, Q best-of-4 over residual actions | unchanged | DICE-RL core; the part of the gain that survived ablation |
| sampler | 5 / 10 | unchanged | 20 / 50 → 5 / 10 gave +18 / +12 |
| residual | unbounded, BC weight 100, DICE filter at ε = −0.3; settles at 8% of action rms | α·tanh(net), α = 0.3 in normalised action units (1.2 action rms), BC weight 0, filter off | Policy Decorator ablation (bound essential, set near demo action scale), PLD (clip 0.5, no BC term, 97 on LIBERO), fixed-bound sweeps insensitive over 0.1–0.5 |
| residual schedule | none | actor frozen and residual off for the first 50 episodes; then applied per chunk with probability min((steps − warm-up) / 50k, 1) | Silver critic burn-in, PLD 100-episode warm-up, Policy Decorator ramp (collapse when too short) |
| actor inputs | pooled 3072 + proprio 8 + base chunk | structured means (2 × 3072) standardised + proprio 8 + base chunk | pooled vector per-element spread 0.028; run B's actor died on unstandardised inputs (norm 39 vs 1.5) |
| critic inputs | same as actor + action | two critics: a privileged Q on the actor's standardised features + proprio + object state (22 floats) + action, driving the actor and the TD targets; a deployable Q on the actor's features + proprio + action, regressed onto the same targets, used for candidate selection in training and at test time | asymmetric critic (ReCoVLA, ResiP, Scaffolder); critic sees the actor's observation too (Baisero & Amato); every selection critic in use was trained on deployable inputs (V-GPS, DiVeR, SVA, DICE-RL); SimDist regresses an observation value head onto privileged targets |
| reward | sparse success per chunk | sparse success + γΦ(s') − Φ(s), Φ the calibrated progress potential without the success term | Ng, Harada, Russell 1999; our V_plan tracked Φ's level at 0.94 while sparse Q had no action signal |
| updates | 10 gradient steps every 4 chunks | 10 gradient steps every chunk | RLPD / IBRL update-to-data 5–20; critic is an MLP, chunk rollout is 8 s |
| expert ratio | 0.9 → 0.1 over 208k steps | 0.5 constant | RLPD symmetric sampling |
| reset mix (box 1) | thirds: standard / 28 robot-init variants / 26 layout variants | thirds: standard / procedural arm pose / 26 layout variants | RL4VLA (varied poses → held-out poses 0.34 → 0.80); LIBERO-Plus cross-axis transfer fails |
| curriculum (box 1) | none | arm-pose offset magnitude cap 0.3 → 0.5 rad on a success trigger | automatic domain randomisation |
| diagnostics | none on Q | shadow labeller scores the executed candidate set every 10th chunk; Q's agreement with measured outcomes logged | our cf run: Q's ranking had no signal and nobody could see it |

Unchanged: frozen prior, instruction conditioning, guidance 5, 10-head critic with LayerNorm, min over heads in the TD target, 3-chunk n-step, γ 0.99 per chunk, τ 0.01, batch 256, Adam 1e-4 with the cosine restarts, replay capacity 100k, 520-step horizon, resume fingerprint.

## Policy

- Plans: `plan_candidates = 1`. The plan critic, its buffer keys and the token sidecar are not built. K = 4 noise candidates per chunk; each gets the residual; Q (mean over heads at selection, as now) picks the executed candidate.
- Residual: `a = a_base + α · tanh(net(x))`, masked to the 7 used dof, α = `residual_bound` = 0.3. The final layer stays zero-initialised, so the residual starts at 0. `bc_weight` = 0 and `bc_filter` off; the DICE overestimation statistics stay logged.
- Actor input x: standardised structured features (per-camera, per-frame means of the last latent frame, 2 × 3072), proprioception (8), base chunk (16 × 30). Standardisation: subtract the global mean and divide by the global standard deviation of the expert features' structured means, computed once at run start and stored in `settings.json`; the same constants apply to the critic's copy of the features. Proprioception is not scaled (entries already of order 1).
- Schedule: `actor_warmup_episodes` = 50 (about 20k env steps) during which the actor is neither applied nor updated and the critic trains on base rollouts plus expert rows; then `residual_ramp_steps` = 50,000 over which the residual is applied per chunk with probability p = min((env_steps − warm-up end) / 50k, 1), drawn once per chunk for all K candidates. Rows store whether the residual was applied. Actor updates run every other critic update from the end of the warm-up, as now.

## Critic

- Two 10-head ensembles with the same architecture. The privileged critic Q_p takes [standardised structured features, proprio, standardised object state, action]; the deployable critic Q_d takes [standardised structured features, proprio, action]. Object state is the 22-float record the buffer already stores (three bodies' position and w-x-y-z quaternion, door joint), standardised per dimension with expert statistics; the record after each chunk is captured as `object_state_next`.
- Targets: r + γ³ · min-over-heads Q_p,target(s', a') over the 3-chunk window with a' = base candidates of s' plus the current actor's residual, as now, with the shaping term folded into r. Both critics regress onto this target; only Q_p has a target network. The actor maximises Q_p. Candidate selection (K = 4) uses Q_d in training and at test time, so the collected data reflects the deployed selector.
- Reward per chunk: `success` (1 on the first success) + γΦ(s_after) − Φ(s_before), where Φ(s) = −‖mug − microwave‖ + 1[inside] + (1 − door_open) · 1[inside] from `script/lingbot_rl_progress.py` with the success term removed. Φ at the terminal transition uses the real state; the telescoping sum over an episode is a per-episode constant. `shaping = True` on both boxes. Expert rows get the same shaping from their replayed object states.
- Updates: `gradient_steps` = 10 per chunk, `update_every_chunks` = 1, `rlpd_start = rlpd_end = 0.5`.
- Online checks: `select_agreement` (Q_d's and Q_p's argmax over the K candidates agree) every chunk; the labeller every 10th chunk scores the executed candidates and the base candidates, giving Q_d's agreement with measured outcomes and the residual's measured gain. The deployable-versus-privileged ranking gap is unpublished, so these two numbers are the run's own evidence.

## Data

- Box 1 (`dice-t9-ood-bounded`): per episode draw from `np.random.default_rng([seed, episode])`: kind 0 standard reset (base task on the plus package, `init_states = False`), kind 1 procedural arm pose, kind 2 one of the 26 layout training variants with a uniform stored state. Procedural arm pose: offset d = m · u with u uniform on the unit sphere in joint space (7 dof) and m uniform in [0.1, cap]; the Panda model's `init_qpos` is overridden with base + d before the environment is built for that episode (verified: zero joint error, fresh environment per episode). Objects are sampled by the simulator under the episode seed, as for the LIBERO-Plus robot-init variants. Curriculum: cap starts at 0.3 rad (the l2 range of LIBERO-Plus levels 1–2) and rises to 0.5 rad (levels 3–5) when the trailing success over the last 20 procedural episodes reaches 0.5; it never decreases. The cap and every episode's offset are logged.
- Box 2 (`dice-t9-std-bounded`): standard resets only, same recipe, same seed.
- Budget 320k env steps in the recipe so that a resume or an extension past 240k keeps the protocol fingerprint; launched with a 240k step cap; checkpoints every 80k; resume every 10 episodes. Expected rate about 0.45 s per env step at M = 1 (the four-plan runs cost 0.54–0.58), so about 30 h per run.

## Diagnostics

- Every 10th chunk the shadow labeller executes the K = 4 executed-candidate chunks (residual applied) and the K base chunks from the live state and records Δφ for each; logged: `q_label_agreement` (Q's argmax over the 4 residual candidates equals the label's), `residual_label_gain` (mean Δφ of residual candidates minus base candidates), per reset kind. Cost about 0.04 s per env step averaged.
- W&B per chunk: residual rms, bound saturation rate (|tanh| > 0.95), q_advantage, better-than-base rate, shaping reward, success per reset kind, curriculum cap, arm-pose offset magnitude; `train_eval` at checkpoints as now.
- `settings.json` records the standardisation constants, calibration, the object layout and the curriculum trace.

## Evaluation

1. The existing 100-episode LIBERO-Plus plan (`docs/libero-plus/task9-plan.json`), with its baselines SFT 37, in-distribution 320k plan policy 53, random plan 50. Box 1 trains on no robot-init variant (its poses are procedural), so all 52 robot-init episodes are unseen poses; the layout axis is reported split into the 32 episodes on the 26 training variants and the 16 on the 13 unseen variants. No new baselines.
2. Standard protocol, 100 episodes, for both boxes at 160k and 240k, against 79 / 80 and SFT 60.
3. Decisions: at 160k, continue if training success on perturbed episodes exceeds the cf run's 0.45 window mean or the residual's measured label gain is positive; the 240k LIBERO-Plus comparison of box 1 against box 2 and against 53 is the result.

## Smoke gates (2,100 env steps per box, on the box)

- Residual: zero during warm-up; after a forced ramp start, rms within [0.02, 0.25] and saturation rate below 0.5.
- Shaping: per-episode sum of shaping rewards equals Φ(s_T)-weighted minus Φ(s_0) to 1e-4; Φ(s_0) matches the calibration's `phi_initial` on standard resets.
- Procedural pose: the arm's joint offset after reset equals the drawn offset to 1e-6; eef position differs from the base pose.
- Privileged critic input width 2 × 3072 + 8 + 22 + 480 and deployable 2 × 3072 + 8 + 480; standardised feature norm of order sqrt(6144) ≈ 78 (unit per element), not 39 unstandardised.
- Labeller: diagnostic labels present for every 10th chunk; shadow fidelity gates as before.
- Throughput ≤ 0.5 s per env step; peak GPU memory below 60 GB.

## Decisions taken at review

1. Bound alone: the BC term and the DICE overestimation filter are dropped, as in Policy Decorator and PLD. What stays of DICE-RL: the frozen generative base, K noise candidates per chunk with the residual applied to each and Q choosing the executed one, the 10-head critic with the min target, n-step TD on a replay mixed with expert rows, and the actor maximising Q over the candidates. The hard bound replaces the BC penalty and filter as the constraint that keeps the residual near the base.
2. α = 0.3 (1.2 action rms), inside the range the fixed-bound sweeps found insensitive.
3. Box 2 is the data control, not an ablation. It is watched for the same signals as box 1: the labeller's Q agreement with measured outcomes and residual label gain on standard resets, residual rms and saturation, and training success against the in-distribution plan run's curve (0.8 window mean); its 160k standard-protocol result is compared with that run's 79 at 160k. Its held-out LIBERO-Plus number against box 1's is the result.
4. Test-time selection uses the deployable critic only; the privileged critic never leaves training (reviewed 2026-10-19 after the deployable-critic survey, section 5 of the survey doc).
5. Layout perturbations are trained on box 1 through the 26 stored training variants (a third of episodes) and evaluated on both boxes through the 13 held-out layout variants; procedural layout generation stays out of scope because the distractor variants need new scene objects and every policy already sits within 23–28 of 48 on this axis, against SFT's 25.

## Out of scope

LIBERO-90, other tasks, noise-space actors, token encoders, guidance changes, procedural layout generation (stored variants only), real-robot deployment.
