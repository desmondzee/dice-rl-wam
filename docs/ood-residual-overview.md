# Out-of-distribution residual over a frozen flow policy

Status as of 2026-10-23. Records: `docs/libero-prior.md`, `docs/libero-residual.md`, `docs/dice-rl-reproduction.md`, `docs/research/2026-10-21-residuals-under-distribution-shift.md`.

## Question

Does an MLP residual, trained with RL on a frozen flow-matching prior across a wide range of robot poses, succeed on held-out poses at less cost than post-training the prior? Published residuals that vary the scene train and evaluate on the same variation. DICE-RL (Sun and Song, ICML 2026, arXiv 2603.10263, §5.3) limits its robustness claim to small on-trajectory perturbations. The axis is the robot's initial pose. The test poses are held out of training.

## Prior

The prior is `VisionResNetFlowMatchingMLP`. Two 128×128 cameras, agent view and wrist, each go through a ResNet-18 trained from scratch, a spatial softmax with 32 keypoints, and a projection to 128-d visual features. Proprioception is 8-d: end-effector position, axis-angle orientation, and gripper, with no object pose. A 1D U-Net outputs 8-step chunks of 7-d OSC pose deltas, integrated for 10 Euler steps. Pretraining takes a random 116 crop. Evaluation uses the centre crop.

On Tool Hang, the state residual from this codebase scored 0.84 at 448k env steps, against 0.80 in the authors' log.

The task is LIBERO-10, put the yellow and white mug in the microwave and close it. The checkpoint is `state_100.pt` from run `2026-10-22_10-11-34_42`, demos 0–19, 9,400 updates. At that epoch a 50-demo prior scored 0.74 on 100 episodes, outside the 0.40–0.60 band. In distribution, 50 stored states and torch seeds 0–5 score 0.457 over 300 episodes. A 100-episode evaluation moves by about ±5 points. Seed pairs differ by about 7.

## Residual

The config is `brev/dicerl/cfg/libero/<task>/ft_distill_residual_flow_unet_img.yaml`, from the released Square image recipe. The prior stays frozen. Actor and critics receive the 8-d proprioception concatenated with the 128-d visual features. The action is the prior's chunk plus an MLP residual r(s, a_base) (`condition_residual_on_base_action: true`). The paper's residual is r(s, z). Both networks use hidden sizes 1024, 1024, 1024. Ten critics score a chunk by their minimum.

Evaluation draws 16 noise samples and executes the chunk with the highest minimum Q. Training exploration uses one sample through iteration 3,000 and 16 samples after that. The actor loss is −Q, averaged over 16 noise samples, plus weight 50 on the squared residual. The penalty is unconditional through iteration 3,000. Past that, the Tool Hang soft Q filter (`use_soft_q_filtering: true`, threshold −0.55, `always_retain_bc_loss_for_expert_data: false`) drops it when the corrected chunk outscores the prior and Q is underestimated. The 20 demonstrations mix into the replay at a ratio decaying from 0.5 to 0.1 over 10,000 iterations.

Training episodes load one of the 50 stored states and offset the seven arm joints: direction uniform on the sphere, norm uniform up to the current magnitude. The magnitude is 0 through iteration 3,000, linear to 0.6 rad at iteration 6,000, then held. A joint within 0.02 rad of its limit, or the arm in contact, is redrawn up to 20 times. The OSC nullspace target follows the loaded joints, as on a LIBERO-Plus robot class. The held-out set places the 42 LIBERO-Plus robot-init poses (suite indices 640–681, 0.1–0.5 rad) onto canonical stored states 0–6, 294 episodes. The control is the same config with `shift.end_magnitude: 0`, seed 42, on Watson. The shifted run is seed 42 on Jarvis.

## Results

One seed. Monitoring is 84 episodes, stored states 0 and 1 of each variant, seed 0, about ±0.05. In-distribution numbers for the two runs come from the in-training canonical evaluation.

| | held-out LIBERO-Plus poses | in distribution |
|---|---|---|
| frozen prior | 0.14 (40/294); 0/140 at 0.4 and 0.5 rad | 0.46 (300 episodes) |
| control, canonical starts, 512k env steps | 0.52 | 1.00 |
| shifted starts, 640k env steps | 0.76 | 1.00 |

Control monitoring from 128k to 512k env steps: 0.50, 0.55, 0.49, 0.52. Shifted monitoring: 0.49 at 128k (0 rad), 0.52 at 256k (0.2 rad), 0.65 at 384k (0.6 rad), 0.71 at 512k, 0.76 at 640k. Both runs are at 0.98–1.00 in distribution by 256k env steps. Each box is waiting on `model_step_13000.pth`, the last checkpoint at or under 850k env steps, for the 294-episode held-out set, 150 in-distribution episodes, and the procedural magnitudes, under hard reset.

## Next

- Object layout, on held-out LIBERO-Plus variants.
- Camera, lighting, and background, by re-rendering the demonstrations under visual randomisation at pretraining. The residual reads the frozen features.
- A frame-history policy (a world-action model), with the residual and critic on its latent state. The current runs use the flow prior.
- Resumable training state (`docs/superpowers/specs/2026-10-23-resumable-residual-training-design.md`). Restarting a current process resets the ramp, the warm-up, and the replay buffer.
