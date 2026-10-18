# Residual RL for out-of-distribution performance: literature survey (2026-10-18)

Three focused surveys run before designing the post-V_plan recipe. Numbers were read from the arXiv texts; items the surveys could not verify are marked. Our own measurements are in `docs/probe-outcomes.md`.

## 1. Residual RL on frozen policies

- Residual Policy Learning (Silver et al. 2018, 1812.06298): residual on state over a hand-coded controller, zero-init last layer, critic-only burn-in until TD loss < 1 before the actor moves. Base → residual: SlipperyPush ~0.45 → ~1.0, miscalibrated PickAndPlace ~0.5 → 1.0, ComplexHook ~0.55 → ~0.8. Randomisation shared by train and eval.
- Johannink et al. 2019 (1812.03201): TD3 residual on robot + object state over an impedance controller; real insertion 2/20 → 15/20 in ~8k samples; flat across block-rotation and bias sweeps.
- Policy Decorator (Yuan et al. 2024, 2412.13630): residual = α·tanh(·), α 0.03–0.3 state, 0.05–0.8 visual, set near the demo action scale; progressive exploration (base+residual with probability rising 0 → 1 over 30k–8M steps); critic on a_base + a_res. Ablations: bounded residual essential, too-small α caps the asymptote, too-fast schedule collapses. Residual on observation only; base action as input reported worse (figure only).
- ResiP (Ankile et al. 2024, 2407.16677): PPO residual on full sim state + a_base, scale 0.1, per-step closed loop over a chunked diffusion policy. one_leg 54 → 98 (low randomisation) / 26 → 76 (medium). Explicit: corrections are local and fail when the base strays far; vision students distilled from the state teacher lose ~20 points.
- DSRL (2506.15799): noise-space policy on state, box [−b, b], b 1–3; LIBERO π0 ~20% → ~100% in ~10k samples; ResiP and V-GPS did not significantly improve π0 there. No shift experiment.
- PLD (Xiao et al. 2025, 2511.00091): residual π(s, a_b), ResNet-10 encoder, δ clipped to ±0.5 on LIBERO (±0.1 SimplerEnv), no BC/KL term, 100-episode base-only warm-up, Cal-QL critic pretrain on ~50 base successes, SAC with 2 LayerNorm critics, 50/50 offline/online batches, sparse reward. π0 LIBERO 93.4 → 97.2, OpenVLA 91.8 → 99.2. Large δ hurts early, small δ lowers the asymptote.
- ReCoVLA (2606.09630): PPO residual on the frozen π0.5 latent only, asymmetric critic on simulator state, a = clip(a_b + 0.25·a_r). Sim 0.37 → 0.67, real 0.27 → 0.62, object-substitution OOD 10.0% → 53.3% (training distribution for that result uncertain).
- Object-Centric Residual RL (2606.18953): TD3 residual on object 6-DoF poses + proprio + base action, no images; pose noise ≤ 5 mm / 0.1 rad and 0.1 pose dropout in training. Real 42% → 76%. Object-centric vs image residual per task (/20): 17/8, 16/10, 15/9, 20/14, 8/6; removing noise + dropout: 17 → 12, 16 → 10, 15 → 9.
- RL Token (Physical Intelligence, 2604.23073): actor on a VLA token + proprio + reference chunk, loss −Q + β‖a − a_ref‖²; β = 0 is the largest ablation drop.
- DAWN (2602.10539): LayerNorm critics for residual RL, λ ~0.1, ~6× faster than Policy Decorator on PegInsertionSide.
- Warp RL (2606.31043): claims additive residuals cannot reshape the base distribution and can underperform the base under dynamics shift (numbers unverified).
- RL4VLA (Liu et al. 2025, 2505.19789): full LoRA PPO, the only train-on-randomised / test-held-out result: unseen robot initial pose SFT 0.339 → RL 0.797, unseen object positions 0.568 → 0.807, mid-episode reposition 0.286 → 0.745.

## 2. Privileged critics, shaping, sample efficiency

- Pinto et al. 2017 (1710.06542): critic on full state, actor on RGBD; real Pick / Push / Move 5/5 with asymmetric HER against mostly 0/5 symmetric; needs domain randomisation.
- Baisero & Amato 2022 (2105.11674): a state-only critic is biased when observations alias state; V(h, s) on history plus state is unbiased. Lambrechts et al. 2025 (2501.19116): asymmetric critic drops the aliasing term but its residual error can be higher. Ebi et al. 2025 (2509.26000): any state-derived signal plus history keeps the gradient unbiased; full state is not always best.
- Scaffolder (Hu et al. 2024, 2405.14853): privileged critic is decisive where the reward is object-centric (cube pose); model-free privileged-critic baselines are often below the non-privileged baseline; a privileged critic does not fix exploration.
- Shaping: Ng, Harada, Russell 1999 potential-based shaping preserves optimal policies. Popov et al. 2017 (1704.03073): sparse reward fails from scratch for stacking, composite shaping solves it; resets from success states with sparse reward learn in 1M transitions. Nair et al. 2018 (1709.10089): step rewards without a BC term produce pausing local optima. PLD reaches > 95% on LIBERO with sparse reward because the base supplies successes.
- RLPD (2302.02948): UTD 10–20, 10-critic ensemble, LayerNorm (collapse without it on Adroit), 50/50 demo/online batches. IBRL (2311.02198): 5 critics, UTD 5, actor dropout 0.5; real Lift 100% after 8k steps. SERL / HIL-SERL: RLPD backbone, 20–30 demos, 20 min to a few hours per task.
- OpenAI Rubik's cube (1910.07113), DeXtreme (2210.13702): value net on noise-free poses, velocities and randomisation parameters with automatic domain randomisation; no ablation isolating the critic.

## 3. VLA robustness to initial pose and layout

- LIBERO-Plus (2510.13626): robot initial state is the hardest non-camera axis for every model (OpenVLA 3.5, OpenVLA-OFT 31.9, π0 6.0, π0-FAST 21.6, UniVLA 46.2); layout 68.9–74.2 for OFT / π0. Robot + layout compounds (39.1 → 24.4). Their post-training on perturbed data for every axis except robot init moved robot init only 21.7 → 30.3 while camera reached 92.8 and layout 77.6. Mechanism offered: trajectory memorisation and missing kinematic understanding. Later: GR00T-N1.5 52.6, OFlow 58.5, ResVLA 57.5 on the robot axis.
- LIBERO-PRO (2510.03827): object displacement beyond ~0.2 units collapses OpenVLA / π0 to 0.00, π0.5 beyond ~0.4; same trajectory replayed with the target removed.
- RIPT-VLA (2505.17016): RL on in-distribution starts gives 31.2 on the LIBERO-Plus robot axis against 31.9 for its OFT base. SimpleVLA-RL, RLinf-VLA: held-out tasks or pooled OOD only.
- ReViP (2601.16667): π0 ignores vision when state and vision disagree (46/50 trials); State-free Policy (2509.18644): removing proprioception lifted height generalisation 0 → 85% (confounded with other changes); GAP (2602.12032): proprioception dominates early optimisation. No paper tests proprioception dropout on initial-pose perturbations.
- Bidirectional Decoding (2408.17355): open-loop chunks collapse under action noise (VQ-BeT Push-T 64.0 → 13.0 at noise 1.5, closed-loop 48.9 → 29.5); untested for initial-pose offsets.

## What the surveys support, against our data

1. A bounded residual with scale 0.1–0.5 of the action range and a schedule that grows it beats an unbounded one pinned by a BC penalty; our residual is 8% of action magnitude under BC weight 100 and the actor's grasp of a wrong initial pose is local by construction (ResiP). Nothing but full-policy PPO on varied poses has moved the robot-initial-pose axis in the literature; a bounded residual of 0.25–0.5 is the closest a frozen-base method gets.
2. Asymmetric critics on simulator state are standard where the reward is object-centric; condition the critic on the actor's observation as well as the state to keep it unbiased.
3. Actor conditioning: proprioception, the base chunk, and standardised visual features; object poses in the actor win on real tasks but are not deployable from images, and distilling them to vision loses ~20 points.
4. Train on the perturbed distribution with online rollouts from varied poses, with a difficulty curriculum; training on other axes does not transfer to arm pose.
5. Keep sparse success as the terminal term; potential-based shaping on object-state potentials is policy-invariant and cheap; avoid raw distance rewards.
6. Critic hygiene: LayerNorm, 5–10 critics, UTD 5–20, 50/50 batches, critic warm-up on base rollouts before the actor moves.

## 4. Scheduling the residual bound (follow-up search, 2026-10-19)

- No residual-RL paper reports a fixed-versus-growing bound ablation. Policy Decorator (2412.13630) fixes α per task and schedules only the probability of applying the residual (linear 0 → 1 over H steps); its α sweeps give similar success over 0.1–0.5 (PushChair) and 0.03–0.1 (StackCube) with sample efficiency varying; advice: set α near the demonstration action scale. ResiP (2407.16677): fixed 0.1, sweep 0.01–1.0 "very similar", 0.2 slightly better. DAWN (2602.10539): fixed λ 0.05–0.3, and Appendix F.2 shows the probability ramp becomes unnecessary once value learning is stabilised (LayerNorm critics): DAWN without a schedule beats Policy Decorator with its tuned one. PLD (2511.00091) says ξ is "tuned by a scheduler" but never gives the schedule; its only finding is qualitative (large ξ early hurts, small ξ caps the asymptote).
- Gating schedules with ablations: uncertainty-gated residual (2506.17564) decays the gate threshold exponentially and a constant threshold "restricts the residual from escaping the base"; DexFlyWheel (2509.23829) ramps the probability after 1,500 steps.
- The closest analogue with numbers is the decaying BC weight in offline-to-online TD3+BC (Beeson & Montana, 2211.11802): relaxing the constraint beats a fixed one on sub-optimal data (hopper-medium 100.1 vs 56.6) but relaxing from step 0 collapses learning; Zhao et al. (2210.13846) adapt the BC weight by a controller on returns because no fixed weight suits every task.
- Exploration-noise curricula (2206.03787) decrease noise over training; growing noise was not tested.

Verdict: a growing magnitude bound is unsupported; fixed bounds are insensitive over a broad range; the evidenced schedules are on the probability or gate of applying the residual, and a base-only warm-up before the actor moves. Decision: fixed α at the demonstration action scale, a probability ramp as insurance, critic warm-up first.
