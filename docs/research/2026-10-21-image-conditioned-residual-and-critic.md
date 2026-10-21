# Image-conditioned residual and critic: code audit, literature, feasibility (2026-10-21)

Every claim cites a file path or an arXiv id. Items marked (unverified) could not be confirmed from the primary text.

## 1. The DICE-RL image path as vendored

**Prior.** `VisionResNetFlowMatchingMLP` runs one `ResNetSpatialEncoder` per camera: torchvision `resnet18(weights=None)`, no ImageNet weights (`model/flow_matching/vision_resnet_mlp_flow_matching.py:216`), robomimic spatial softmax with 32 keypoints and a Linear-LayerNorm head to `spatial_emb` 64 (`:250-263`), concatenated across cameras and projected to `visual_feature_dim` 128 (`:572-580`, `extract_visual_features` at `:662`). Cameras are `agentview_image` and `robot0_eye_in_hand_image` at 96×96, channel-stacked by the wrapper (`env/gym_utils/wrapper/robomimic_image.py:147-167`; `cfg/robomimic/finetune/square/ft_distill_residual_flow_unet_img.yaml:48-51,71`). Proprioception is eef pos, quat and gripper qpos; no object poses (`...unet_img.yaml:52-54`). Random crop 96→84 at pretraining, centre crop at eval (`vision_resnet_mlp_flow_matching.py:42-70,274-276`; `cfg/robomimic/pretrain/square/pre_flow_matching_unet_img.yaml:102-104`). The prior integrates `flow_steps` 10 Euler steps from the merged vector (`model/flow_matching/flow_matching.py:173-196`).

**Residual.** `TrainDistillResidualFlowImgAgent` sets `obs_dim = visual_feature_dim + proprio_dim` (`agent/finetune/train_distill_residual_flow_img_agent.py:106-112`) and, on every environment observation, calls the frozen prior's `network.extract_visual_features` and concatenates `[state, visual_features]` (`:296-346`, feature call at `:334`). The prior is in `eval()`, so no augmentation at RL time (`:125`); expert demonstrations are featurised once at startup (`:182-294`). `DistilledActor` is an MLP over the flattened merged state plus noise (`model/rl/distill_residual_rl.py:64-89`); `get_action` feeds the same merged vector to prior and actor (`model/rl/distill_residual_rl_img.py:89-131`). The residual has no encoder and never sees pixels.

**Critic.** `DistilledCritic` is a 10-head MLP ensemble on `[merged state, action]`; noise enters only if `q_depends_on_noise` (default False) (`distill_residual_rl.py:91-176`; ensemble size `...unet_img.yaml:183`). Exploration scores 16 noise candidates, max over candidates of min over heads (`distill_residual_rl_img.py:133-243`; `...unet_img.yaml:219-221`). The critic also lives on the frozen features; the replay buffer stores 137-dim vectors, not images (`train_distill_residual_flow_img_agent.py:159-180`).

**Released image results.** The README gives only the launch command and a timing note: Transport (pixel) about 24 h and Tool Hang (pixel) 48 h on an RTX 4090, "the RL update ... around 1 second per batch" (`README.md:334-351`); no numeric pixel success. The paper reports pixel curves for Can/Square/Transport/Tool Hang (Fig. 2, 5 seeds, 300 held-out configs) and ">90% on Tool Hang from either state or pixel inputs" in "roughly 2000 online episodes" (2603.10263 §5.1; App. C Table 1: Transport 4 cams 96², Tool Hang 2 cams 240²→216, Square/Can 96²→84), with the encoder frozen and actor and critic on its latent features. Config `...unet_img.yaml:71` says "4 cameras" but Square uses 2.

## 2. RL fine-tuning of visuomotor diffusion/flow policies from pixels

| Method | Critic / residual visual input | Sample efficiency | Visual-shift generalisation |
|---|---|---|---|
| DPPO 2409.00588 | ViT (Hu et al.) + MLP policy, random-shift augmentation in pre-training and fine-tuning (App. E.6); critic encoder not specified (unverified) | 50 envs; Transport pixel iteration = 160k env steps in 871 s on an L40 + 50 CPU threads (Table A4) | none tested |
| DSRL 2506.15799 | WidowX: actor and critic take proprio plus "image features learned by πdp's pretrained ResNet encoder" (App. C.4.2); π0 sim: own 4-layer CNN at 64×64 trained with actor/critic (Table 11); π0-DROID: VLM 2048-d token + shallow CNN at 128×128 (App. C.5) | LIBERO π0 ≈20→≈100% in ≈10k samples; real Franka 3,500 steps (~40 episodes); π0-DROID 65–80 episodes | none tested |
| DICE-RL 2603.10263 | frozen prior features for actor and critic (App. C); π0 variant uses a 2048-d mean of π0 tokens (App. A) | Tool Hang ≈2,000 episodes; 4–8 envs, 10–20 gradient steps per env step (Table 2) | action-noise only (§5.3); authors say not OOD |
| ReinboT 2505.07395 | offline return-conditioned GPT, no critic, no online RL (§4.2) | n/a | real UR5 unseen backgrounds, distractors, objects; numbers only in Fig. 6 |
| πRL 2510.25889 | π0.5 critic on the VLM output; π0 value on obs + noisy action (§4.3.2); VLM frozen (§5.1) | LIBERO 57.6→97.6 (π0); 320 envs, 8 H100; env steps not reported | ManiSkill "vision" OOD (novel backgrounds/textures) gains "limited" (§5.2.2) |
| SimpleVLA-RL 2509.09674 / VLA-RL 2505.18719 | critic-free GRPO (§2.3) / PPO value model on image + prompt (Alg. 1) | 8×A800 / 48 GPU-h; interactions not reported | held-out LIBERO tasks only / none |
| RLinf-VLA 2510.06710 | value head on the VLA language model, shared with actor (§IV-C2) | 64–256 envs; ManiSkill OOD OpenVLA 39.1→81.9 after PPO (Table I) | OOD pooled over vision/language/action |
| Policy Decorator 2412.13630 | residual "MLP or CNN depending on modality" (§4.1); RGBD 64×64 to 128×128 (App. A.1); sharing with base encoder not stated (unverified) | StackCube 5M steps (Table 15) | object-level shift only |
| ResiP 2407.16677 | residual and critic on privileged sim state (§II-D); vision only in the distilled R3M student | 1024 envs, up to 500M steps | medium randomisation 98→76; part colour change breaks the real-only student (§IV-C1) |
| PLD 2511.00091 / ReCoVLA 2606.09630 | own ResNet-10 / residual on frozen π0.5 latent, critic on sim state (`docs/research-residual-ood-survey.md:10-12`) | see survey | ReCoVLA object substitution 10→53% |

Pattern: frozen-base methods with images either reuse the base's features (DSRL WidowX, DICE-RL, ReCoVLA, V-GPS, DiVeR per `docs/research-residual-ood-survey.md:58`) or train a small CNN from scratch (DSRL π0, PLD); none evaluates lighting, camera or background shift.

## 3. Critics from pixels

- Learning-from-scratch with augmentation matches frozen pretrained features: Adroit Pen 74.8 vs 81.3, Relocate 51.4 vs 47.5, Walker Walk 791.6 vs 788.0 (2212.05749 Table 2); under colour/video shift LfS with colour jitter 53.1/33.0 and overlay 39.3/46.6 against 37.2/9.2 for the best frozen representation (Table 3). Finetuning helps only with augmentation (R3M 64.0 frozen → 80.5, Table 5). Cost 0.263 s vs 0.003 s per BC iteration (Table 4).
- DrQ-v2 2107.09645: encoder updated by the critic loss only, actor gradient detached (§3.1), ±4 px random shift at 84×84, 96 FPS on a V100 vs 28 for DrQ (§4.2).
- SVEA 2107.00644: augment the current observation, bootstrap from the unaugmented successor; on DMC-GB color_hard SVEA 760 vs DrQ 520 (Walker Walk), video_hard 377 vs 104 (Tables 1, 4); "DrQ requires approx. 6× wall-time to match" (Fig. 6).
- Frozen DINOv2/VC/ResNet18 vs a CNN from scratch in online RL is task-dependent on Meta-World; a 1M-transition buffer is 21 GB at 84², 151 GB at 224², CLS tokens cut it >90% (2407.17238 §3.3). Pretrained-feature rankings under RL are unreliable (2304.04591).
- SERL/HIL-SERL: ImageNet ResNet-10 shared across cameras at 128×128, random crop, 20–30 demos, 10k–108k transitions, 1–6 h per task (2401.16013 §3.3; 2410.21845 Table 1, App. A); frozen vs finetuned not stated (unverified).
- Asymmetric critics on simulator state (Pinto 1710.06542, ReCoVLA) need domain randomisation and never ablate test-time selection (`docs/research-residual-ood-survey.md:50-62`).

## 4. Visual domain randomisation and transfer

- Tobin 2017 (1703.06907 §III-A, Table II): 0–10 distractors, procedural textures, camera position in a 10×5×10 cm box, angle ±0.1 rad, FOV ±5%, randomised light count/position/specular; ≥1,000 textures needed; removing distractors raises real error 1.8→7.2 cm, removing camera randomisation only 1.8→2.0–2.9 cm.
- Dactyl 1808.00177 (App. C.2 Table 9): camera ±1.5 mm, 0–3°, FOV ±1°; 4–6 lights, intensity 1–5; HSV ±1/15/15%; contrast 50–150%, ±10% pixel noise; real pose error 9.3 mm / 5.0° vs 3.1 mm / 2.7° in sim.
- RCAN 1812.07252 (§4.1, Table 1): >5,000 textures, random light position/direction/colour; zero-shot real grasp 33–37% with randomisation alone vs 70% with canonicalisation.
- Colosseum 2402.08191 (§III-C, V): light RGB 0–0.5, camera Euler ±0.1/±0.05, 20 colours, 213 textures, 78 distractors; policies lose 30–50% per factor and ≥75% combined; training on all perturbations recovers only +21.1 points (PerAct); no held-out-factor test.
- LIBERO-Plus 2510.13626: post-training on perturbed data lifts the camera axis to 92.8 and layout to 77.6 but robot-init only 21.7→30.3 (`docs/research-residual-ood-survey.md:38`); robot-init has moved only under on-policy RL on varied poses (RL4VLA 2505.19789: 0.339→0.797).

## 5. Feasibility on one GB10

Assumptions: LIBERO-10 at 128×128, two cameras, 4 envs, the §1 ResNet-18 prior, 10 Euler steps, 16 candidates, 10 critics, 10–20 gradient steps per 4-step chunk (`...unet_img.yaml:86-92`; `script/lingbot_rl_config.py` `gradient_steps` 10), bf16 at ~70 TFLOPS effective (`docs/probe-outcomes.md:86`).

- Inference per chunk: 2 × ResNet-18 at 128² ≈ 1.2 GFLOPs per env (0.25 at 84², 3.6 at 224²), 16×10 small UNet passes, 640 critic rows: a few GFLOPs, launch-bound at ~30–50 ms (estimate).
- Simulation: GPU-LIBERO reports ≈45 CPU steps/s per env with no cameras on x86 (github.com/Catigeart/GPU-LIBERO); DPPO's Transport pixel run reaches ≈184 env steps/s with 50 subprocess envs on an L40 (2409.00588 Table A4: 160k steps/871 s, derived). Our LeRobot stack renders 520 steps in 65 s including a 2B-parameter prior (`docs/heldout-results.md:196`). GB10 ARM EGL rate is unmeasured (unverified); assume 10–25 steps/s per env with two 128² cameras, 40–100 aggregate over 4 subprocess envs.
- Update with frozen features: the actor loss pushes B×K = 256×16 rows through the prior (10 steps) and critic, ≈1–1.5 TFLOPs per gradient step → 25–40 ms plus launch overhead, 0.4–0.8 s per chunk at UTD 10–20, i.e. 20–40 env steps/s update-bound. The LingBot runs measured 0.19–0.23 s per env step at M=1 with a far heavier prior (`docs/heldout-results.md:212,373`). The README's "1 second per batch" (`README.md:351`) does not reconcile with 150k iterations in 24 h (unverified).
- Expected: 10–25 env steps/s aggregate, 36–90k env steps/hour, 320k steps in 4–9 h; the update, not rendering, is the bottleneck.
- Critic with its own trained encoder (DrQ-style): fwd+bwd ResNet-18 on 1,024 images (s and s′, 2 cams) per gradient step ≈ 0.8 TFLOPs at 84², 1.8 at 128², 5.5 at 224² → +0.2–2 s per chunk; throughput falls to 5–15 env steps/s at 128² and ~3–7 at 224². Replay at 320k steps: 31 GB uint8 at 128², 96 GB at 224² (the box has 121 GB, `docs/heldout-results.md:337`), 18 GB at 84².

## Recommendation

Image-feature inputs from the start, but state-visible axes first in the RL stage.

1. The vendored image path already removes privileged state: residual and critic see `[proprio, frozen visual features]` (`train_distill_residual_flow_img_agent.py:344`), the design every published frozen-base method uses (DSRL App. C.4.2, DICE-RL App. C, ReCoVLA). The low-dim path with object poses would have to be rewritten anyway, so start on features; it needs no encoder training and keeps the update at 0.4–0.8 s per chunk (§5).
2. Visual axes (lighting, camera, background) are not fixable by RL on frozen features: no paper in §2 shows it, and robustness comes from augmentation during encoder training (2212.05749 Table 3, SVEA) or randomised data (§4; LIBERO-Plus camera 92.8 after perturbed SFT). The cheap route is re-rendering demonstrations with Tobin/Colosseum-scale randomisation (camera ±0.1 rad, 10 cm box; light colour 0–0.5; ≥1,000 textures) plus colour jitter and crop at prior pretraining, then checking held-out factors. A pixel-trained critic (DrQ/SVEA recipe) costs 2–5× update throughput, and the critic cannot yet rank candidates even on privileged state (`docs/investigations/2026-10-20-residual-and-critic.md:19-24`); a trainable encoder now would confound that unresolved problem.
3. Robot-initial-state and layout are the axes that need on-policy RL on varied starts (RL4VLA; LIBERO-Plus robot-init 21.7→30.3 with data alone) and are visible in proprioception plus features, so they are the first RL target. Measure on the held-out LIBERO-Plus variants already defined (`docs/heldout-results.md:322`), then add the visually randomised prior and re-run the same plan before deciding whether a trained image critic is needed.
