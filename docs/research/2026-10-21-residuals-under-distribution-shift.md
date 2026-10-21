# Residuals on frozen priors under distribution shift (2026-10-21)

Question: does a small residual actor and critic on a frozen imitation prior, trained with RL over a wide distribution of shifts, give robustness on held-out shifts, and how does that compare with full fine-tuning? Items read from abstracts or secondary summaries are marked "(abstract)" or "(unverified)".

## 1. Residual policy lines

| Work | Residual input | Prior frozen | Shift tested | Gain |
|---|---|---|---|---|
| Residual Policy Learning, Silver et al. 2018 ([1812.06298](https://arxiv.org/abs/1812.06298)) | state (abstract) | yes | six MuJoCo tasks with partial observability, sensor noise, misspecification, miscalibration; train/eval share the randomisation | "consistently and substantially improves"; no held-out split |
| Residual RL for Robot Control, Johannink et al. 2019 ([1812.03201](https://arxiv.org/abs/1812.03201)) | robot state + object state, u = π_H(s_m) + π_θ(s_m, s_o) | yes (hand-engineered controller) | block rotation U[-r, r] up to 0.3 rad, control noise σ 0.01–0.1, control bias μ 0–0.2 | residual return "maintains a constant average performance over different variations" while the controller degrades; real 2/20 → 15/20. Train and eval distributions coincide |
| Residual RL from Demonstrations, Alakuijala et al. 2021 ([2106.08050](https://arxiv.org/abs/2106.08050)) | base action + proprio + BC bottleneck features; dropping robot state "consistently hurt" | yes | object positions/sizes, start pose, walls randomised per episode | abstract claims better generalisation to "unseen environment conditions" than BC or RL fine-tuning; no isolated held-out comparison reported |
| Policy Decorator, Yuan et al. 2025 ([2412.13630](https://arxiv.org/abs/2412.13630)) | observation only (base action as input was worse, figure only); critic on a_base + a_res; bounded α·tanh | yes | TurnFaucet/PushChair: base trained on source objects, residual learned online on 4 unseen faucets / 3 unseen chairs | near-perfect on all 8 tasks; no held-out test after training |
| ResiP, Ankile et al. 2024 ([2407.16677](https://arxiv.org/abs/2407.16677)) | full sim state + base action, per-step over a chunked diffusion policy | yes | object pose ±1.5/±5 cm (shared train/eval); per-step object perturbations "not seen during training" at eval only | one_leg 54 → 98; under unseen perturbations ResiP drops ~12 pts vs 19–26 for chunked baselines; vision students distilled from the state residual lose ~20 pts |
| DSRL, Wagenmaker et al. 2025 ([2506.15799](https://arxiv.org/abs/2506.15799)) | state → latent noise, box [−b, b] | yes | none; ablations on base size/data quality only | π0 LIBERO ≈20 → ≈100% in ≈10k samples; ResiP and V-GPS "not able to significantly improve" π0 |
| ResFiT, Ankile et al. 2025 ([2509.19301](https://arxiv.org/abs/2509.19301)) | observation + base action; critic Q(s, a_full) | yes | base trained on random start/object/tote; A/B eval with matched random scenes, no held-out split | real humanoid 14 → 64%, 23 → 64%; authors: behaviours "may remain constrained around the base policy" |
| PLD, 2025 ([2511.00091](https://arxiv.org/abs/2511.00091)) (abstract) | residual actors on the VLA | stage 1 frozen; stage 3 distills back by SFT | "seen and unseen tasks", no perturbation axes | LIBERO 99%, SimplerEnv +50% |
| DICE-RL, Sun & Song 2026 ([2603.10263](https://arxiv.org/abs/2603.10263)) | a = π_pre(s, z) + s_θ(s, z); frozen encoder latents; TD3+BC critic ensemble over chunks | yes | real eval randomises EE pose ±2 cm/±10°, board ±10 cm/±15°; action-noise injection p = 0.1–0.9 | Tool Hang 45 → >90%; RL "degrades more gracefully" than BC under action noise; explicit: "We do not claim that RL finetuning improves out-of-distribution generalization beyond the pretrained BC policy" |
| RFS, 2026 ([2602.01789](https://arxiv.org/abs/2602.01789)) | state → (initial noise, additive residual) | yes (sim) | 5 novel deformable objects in the real world (numbers in a figure) | sim avg 0.25 → 0.86 vs ResiP 0.43, DSRL 0.48; real PnP 50 → 90%; residual alone "struggles to induce global behavioral changes" |
| Object-Centric Residual RL, 2026 ([2606.18953](https://arxiv.org/abs/2606.18953)) | 6-DoF object poses + proprio + base action; no pixels | yes | object pose randomised in sim, pose noise ≤5 mm/0.1 rad, pose dropout 0.1; real robot is the held-out domain | real 42 → 76% zero-shot; image-based residual 47/100 vs pose 76/100; "cannot recover from states far outside the base VLA's training distribution" |
| TRANSIC, Jiang et al. 2024 ([2405.10315](https://arxiv.org/abs/2405.10315)) | point cloud, gated residual from human corrections | yes (fine-tuning the base on corrections "leads to large motions and even model collapse") | five artificial sim-to-real gaps (perception, underactuation, embodiment, dynamics, assets) | 77% vs 18% best baseline; zero-shot lamp 75%; gains came from correction data collected under that gap |
| Foresight Residual RL, 2026 ([2607.16506](https://arxiv.org/abs/2607.16506)) (abstract) | residual over frozen VLA | yes | none | chained sparse-reward residuals give "little or no gain"; 54.5 → 85.6% with terminal-state shaping |
| RL Token, Physical Intelligence 2026 ([2604.23073](https://arxiv.org/abs/2604.23073)) | bottleneck readout of a frozen VLA + small actor-critic | yes | none | up to 3× faster on the hardest segment of four real tasks |
| ZPRL, 2026 ([2605.19919](https://arxiv.org/abs/2605.19919)) (abstract) | residual on a VIB latent, not actions | yes | none | +33.7% real; action residuals give "noisy and poorly structured exploration" |

No residual paper trains on a wide shift distribution and reports a separate held-out-shift split. Nearest: Johannink (flat across its training sweep), ResiP (eval-only perturbations), Object-Centric (sim randomisation, real as held-out domain), TRANSIC (gap-specific data).

## 2. Capacity and conditioning

- Shift-dependent corrections require the shift to be observable. Johannink's residual sees object state; Object-Centric sees object poses and attributes its sim-to-real margin to that input; Alakuijala reports that removing proprioception hurts. None ablates a pose-blind residual against a pose-aware one on held-out poses.
- RMA ([2107.04034](https://arxiv.org/abs/2107.04034)) conditions on a latent "extrinsics" vector inferred from a 0.5 s state-action history, trained with privileged simulator factors; Learning by Cheating ([1912.12294](https://arxiv.org/abs/1912.12294)) and the asymmetric actor-critic ([1710.06542](https://arxiv.org/abs/1710.06542)) give privileged information to a teacher or critic only. OSCAR ([2110.00704](https://arxiv.org/abs/2110.00704)) adds latent extrinsics to a residual dynamics term for OOD dynamics (search summary, not read).
- Warp RL ([2606.31043](https://arxiv.org/abs/2606.31043)) (abstract): additive residuals "cannot change the distribution's shape, scale, or state-dependent geometry", and under dynamics shift "can even fall below the unadapted policy"; a state-conditioned invertible warp matches residuals when translation suffices and beats them otherwise.
- DAWN ([2602.10539](https://arxiv.org/abs/2602.10539)): critic bottlenecks are "cold start" and "structural scale mismatch" (residual "dwarfed by the base action"); fixes are base-only replay warm-up (20k transitions) and LayerNorm critics on the summed action; ≈5–6× faster than Policy Decorator; no generalisation study.
- Stability-Aligned Residual Adaptation ([2603.07775](https://arxiv.org/abs/2603.07775)) (abstract): bounded residual on a frozen RL policy recovers from mid-episode actuation/mass/contact changes 20–87% faster; held-out identities not stated.

## 3. RL post-training and generalisation

- Chu et al. 2025 ([2501.17161](https://arxiv.org/abs/2501.17161)): outcome-reward RL generalises to unseen rule and visual variants where SFT memorises; whole-model updates (parameterisation unverified).
- RL's Razor ([2509.04259](https://arxiv.org/abs/2509.04259)): forgetting tracks KL(fine-tuned ‖ base) on the new task; on-policy RL is "implicitly biased towards KL-minimal solutions"; OpenVLA in SimplerEnv with REINFORCE, results as Pareto plots only; LoRA vs full not stated.
- RL4VLA, Liu et al. 2025 ([2505.19789](https://arxiv.org/abs/2505.19789)): LoRA r = 32 PPO on OpenVLA; execution axis (unseen positions, unseen robot init pose, mid-episode moves) "significantly better" than SFT, semantics "noticeably better", vision "comparable". Prior survey records unseen robot init pose SFT 0.339 → RL 0.797 (figure only, not re-verified).
- SimpleVLA-RL ([2509.09674](https://arxiv.org/abs/2509.09674)): full-parameter RL on OpenVLA-OFT; on held-out LIBERO tasks SFT overfits to 0% on several while RL gains 5–36 pts; "pushcut" strategy discovery.
- πRL ([2510.25889](https://arxiv.org/abs/2510.25889)): VLM frozen, 300M action expert trained; ManiSkill π0 IND 38.4 → 78.8 but OOD 32.6 → 61.1; authors blame "freezing the VLM during the RL stage ... prevents the model from adapting its visual features".
- VLA-RFT ([2510.00406](https://arxiv.org/abs/2510.00406)): VLM frozen, action head updated; perturbed LIBERO gains are small: object pos ±5 cm 48.0 → 52.5, robot init ±50 63.5 → 67.0, combined-major 34.0 → 37.0.
- DPPO ([2409.00588](https://arxiv.org/abs/2409.00588)): full diffusion-weight RL yields policies "robust to perturbations in dynamics and the initial state distribution".
- RLinf-VLA ([2510.06710](https://arxiv.org/abs/2510.06710)) (abstract): preliminary real-Franka claim of stronger generalisation than SFT. VLA-RL ([2505.18719](https://arxiv.org/abs/2505.18719)): in-distribution only.

Pattern: the clearest OOD gains come with full or LoRA updates to the whole model; frozen-backbone RL (πRL, VLA-RFT) shows a larger IND/OOD gap. No paper compares a frozen-prior residual against full RL fine-tuning on the same held-out shifts.

## 4. Domain randomisation with a frozen base

- ExpertGen ([2603.15956](https://arxiv.org/abs/2603.15956)): frozen diffusion prior, state-conditioned initial-noise policy in parallel sim. Restricted demo coverage (banana left half only) 39.2 → 76.8 ("partially extrapolate"); drawer friction 0.2 → 0.5: 31.4 → 99.2; under external force Lift Banana DP 1.9, residual RL 46.7, ExpertGen 56.6. Visual randomisation only in the distillation stage.
- Object-Centric Residual RL: pose randomisation plus noise/dropout; prior survey records that removing noise + dropout lowers real successes (17 → 12, 16 → 10, 15 → 9 of 20).
- LIBERO-Plus post-training ([2510.13626](https://arxiv.org/abs/2510.13626)): supervised fine-tuning on perturbed data moved camera 55.6 → 92.8, layout 68.7 → 77.6, robot init only 21.7 → 30.3; camera angles differed by 5° from test, so only partly held out.
- DR theory ([2110.03239](https://arxiv.org/abs/2110.03239)): history-dependent policies are needed for DR to pay off; supports giving the residual history or proprioception.

## 5. Negative and cautionary results

- DICE-RL disclaims OOD generalisation; ResFiT: "cannot discover fundamentally different strategies"; RFS: residual is local; Object-Centric: fails far outside base support; Warp RL: residual can underperform the base under dynamics shift; Foresight: chained sparse-reward residuals give little gain.
- LP-DS ([2606.01151](https://arxiv.org/abs/2606.01151), alphaxiv summary): DSRL "collapses to one mode" on a four-goal toy; fix is a Lagrangian bound on the mean squared noise residual.
- Critic inputs: DAWN's scale mismatch; TRANSIC's residual gains were gap-specific.
- RIPT-VLA ([2505.17016](https://arxiv.org/abs/2505.17016)) on LIBERO-Plus Table 1: robot 36.7 vs 37.2 for its OpenVLA-OFT base, camera 58.3 vs 59.7: in-distribution RL post-training buys no robustness.
- No paper reports a residual collapsing to a constant offset; the failure mode is undocumented, not refuted (BRPO, [2002.05522](https://arxiv.org/abs/2002.05522), motivates state-dependent deviation bounds).

## 6. LIBERO-Plus and LIBERO-PRO

LIBERO-Plus (Table 1, zero-shot, %): OpenVLA-OFT original 97.1 / camera 59.7 / robot 37.2 / language 81.5 / light 85.8 / background 92.4 / noise 76.7 / layout 77.1; π0 94.2 / 15.8 / 6.6 / 61.0 / 79.6 / 78.5 / 79.4 / 70.4; UniVLA 95.2 / 4.3 / 50.3 / 71.8 / 59.1 / 80.0 / 25.3 / 34.3. Camera and robot init are the hardest axes; Finding 3: models "may be relying less on the language instruction than anticipated". The only RL-trained entry is RIPT-VLA (above); later zero-shot entries are supervised (VLA-GSE 81.2% average, [2605.06175](https://arxiv.org/abs/2605.06175), search summary; Afford-VLA [2605.24203](https://arxiv.org/abs/2605.24203)). LIBERO-PRO ([2510.03827](https://arxiv.org/abs/2510.03827)): object-position shifts send OpenVLA and π0 to 0.00 on every suite, π0.5 to 0.08–0.38; task changes 0.00–0.01 for all; no post-training evaluated. No RL post-training has reported per-axis gains on either benchmark.

## Verdict

Partially supported, mostly untested. Supported: a frozen prior plus a bounded, shift-observing residual trained under randomisation transfers across the randomised range (Johannink), to a new domain (Object-Centric, TRANSIC) and to eval-only perturbations (ResiP, ExpertGen). Untested: a held-out split over the same shift axes, and any head-to-head against full RL fine-tuning on those axes. Contrary evidence is indirect: frozen-backbone RL shows the largest IND/OOD gap (πRL, VLA-RFT), residual authors disclaim OOD gains (DICE-RL, ResFiT), and additive residuals have a shape limit (Warp RL).

Three ingredients the literature motivates most strongly:

1. Make the shift observable to the residual and critic: proprioception, object/pose state or an inferred shift latent, plus the base action; add pose noise and dropout during training (Object-Centric, Alakuijala, RMA, DAWN's summed-action LayerNorm critic with base-only warm-up).
2. Train on the robot-init and layout shifts themselves with online rollouts; gains on other axes do not transfer to robot init (LIBERO-Plus post-training, RIPT-VLA), and RL's execution-axis advantage appears only when those shifts are in the training distribution (RL4VLA, DPPO).
3. Keep the residual bounded and per-step closed-loop over the chunk, with a trust-region magnitude constraint and a shift-magnitude curriculum; add noise-space steering if global behaviour changes are needed (ResiP, LP-DS, RL's Razor, RFS). Expect no recovery far outside base support.

## What this adds beyond docs/research-residual-ood-survey.md

The prior survey already covers Silver, Johannink, Policy Decorator, ResiP, DSRL, PLD, ReCoVLA, Object-Centric, RL Token, DAWN, Warp RL, RL4VLA, asymmetric critics, LIBERO-Plus/PRO and RIPT-VLA. New here: the SFT-vs-RL mechanism papers (Chu et al., RL's Razor); frozen-backbone counter-evidence (πRL, VLA-RFT perturbed-LIBERO numbers, SimpleVLA-RL held-out tasks, DPPO); DICE-RL's no-OOD disclaimer and action-noise robustness; ExpertGen's coverage-extrapolation and dynamics-mismatch numbers; ResFiT, RFS, LP-DS, ZPRL, Foresight, Stability-Aligned, TRANSIC's five-gap result, Alakuijala's input ablation, and DR-needs-history theory. It also gives the RIPT-VLA robot-axis comparison from Table 1 (36.7 vs 37.2; the prior survey's 31.2 vs 31.9 come from a different table). Unchanged conclusion: no held-out-shift test of a residual exists, so the plan must measure it.
