# Out-of-distribution robustness from a residual over a frozen flow policy

Status as of 2026-10-23. Records: `docs/libero-prior.md`, `docs/libero-residual.md`, `docs/dice-rl-reproduction.md`; research: `docs/research/2026-10-21-*.md`.

## The question

Can a small residual policy, trained with reinforcement learning on top of a frozen imitation prior, give out-of-distribution robustness (robot pose, object layout, and later camera, lighting, background) more cheaply than post-training the prior itself? The literature leaves this untested: every residual paper disclaims out-of-distribution gains, and the frozen-backbone RL results that exist show small gains on shifted conditions. So we measure it, one shift axis at a time, with a held-out split the training never sees.

## The policy we start from

A flow-matching visuomotor policy from the DICE-RL codebase (Sun and Song, ICML 2026, arXiv 2603.10263), whose released recipe we first reproduced on the paper's own Tool Hang benchmark (our 0.84 vs their 0.80 at 448k env steps). The prior sees two 128×128 cameras (agent view and wrist) through a ResNet-18 per camera trained from scratch, plus 8 numbers of proprioception (end-effector position, orientation, gripper), and outputs 8-step action chunks through a U-Net flow head with 10 integration steps. No object poses, no privileged state.

Task: LIBERO-10 `put the yellow and white mug in the microwave and close it`. The prior is trained on 20 demonstrations (the paper's own lever for a weaker prior; 50 demonstrations gave 74% at the first checkpoint) and selected at epoch 100 at 46% success on the in-distribution protocol (50 stored initial states, 300 episodes over three seed pairs). A single 100-episode evaluation carries about ±5 points; seed-to-seed spread is about 7 points, so comparisons use 300 episodes.

## What we train and how

The DICE-RL residual: the prior stays frozen; a small MLP residual r(s, a_base) adds a correction to the prior's sampled chunk (conditioned on the prior's own visual features, proprioception and the base action; the paper uses the flow noise instead of the base action, a code switch we flipped), a 10-critic ensemble Q(s, a) scores chunks, and both exploration and evaluation draw 16 candidate chunks and execute the one with the highest minimum Q. The actor loss is the critic's value plus a BC penalty (weight 50) that keeps the residual small unless a soft Q filter relaxes it; the 20 demonstrations are mixed into the replay at a ratio decaying from 0.5 to 0.1. Everything else is the released image recipe for the paper's Square task.

The shift axis under test is the robot's initial pose. Every training episode starts from a stored initial state whose seven arm joints are offset by a random direction with a norm drawn uniformly up to a magnitude that ramps from 0 (iterations 0–3,000, the recipe's warm-up) to 0.6 rad (iteration 6,000) and holds; starts that put a joint out of range or the arm in contact are resampled. The controller's reference pose follows the shifted start, as it does for a LIBERO-Plus variant. The held-out set is the 42 LIBERO-Plus robot-init variants of this task (absolute arm poses at offsets of 0.1–0.5 rad from the base pose), written onto the canonical stored layouts: 294 episodes, never trained on. A control run with the identical recipe on canonical starts only runs in parallel on the second box.

## Results so far (single seed; 84-episode monitoring set, ±5 points)

| | held-out LIBERO-Plus poses | in distribution |
|---|---|---|
| frozen prior | 0.14 (294 episodes); 0 at offsets ≥ 0.4 rad | 0.46 |
| control, RL on canonical starts, 512k env steps | 0.52, flat from 128k on | 1.00 |
| residual on shifted starts, 640k env steps, still rising | 0.76 | 1.00 |

In-distribution RL lifts the prior's held-out success to about 0.50 and stops; training on shifted starts adds 20–25 points on top and separates from the control only once the ramp reaches 0.6 rad. Final 294-episode evaluations of both runs at 850k env steps are queued. One mechanism finding from Tool Hang carries over: the residual raises the critic's value of the executed chunk everywhere but never sharpens the candidate distribution, so the gain comes from a value-increasing shift plus best-of-16 selection, not from contraction.

## What comes next

- Object layout: the second state-visible axis, on the same machinery, needing region-aware object placement with an object-contact validity check and its own held-out LIBERO-Plus variants.
- Camera, lighting, background: only visible to a policy that sees pixels, so the prior itself must be made robust to them (re-rendering the demonstrations under visual randomisation at pretraining) before a residual can be asked to correct what remains; the critic then works from the frozen prior's visual features, which published frozen-base methods also do.
- Video conditioning: the eventual target policy conditions on a history of frames (a world-action model) rather than a single frame plus proprioception; the residual and critic must then be conditioned on that model's latent state, and whether a residual can read the shift from those latents is the open question this line of work has to answer before moving off the flow prior.
- Resumable training state for the residual runs (`docs/superpowers/specs/2026-10-23-resumable-residual-training-design.md`), required before any extension.
