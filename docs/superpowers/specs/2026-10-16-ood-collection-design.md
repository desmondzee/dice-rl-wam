# Plan selection trained under LIBERO-Plus perturbations (task 9, two runs)

Status: approved design (2026-10-16). Two runs on task 9, one per GB10, collected under the robot-initial-state and
layout perturbations with the prior frozen. They differ in one thing: what the critics and the actor see. Run A uses
the image pools as today; run B uses per-camera, per-latent-frame means. Both add proprioception and both store the
full feature set for the offline ranking study. References: docs/heldout-results.md (LIBERO-Plus section),
docs/research-video-branch-ideas.md, docs/libero-plus/.

## 1. Decisions

| item | decision |
|---|---|
| runs | `dice-t9-plus-m4-pool` (run A) and `dice-t9-plus-m4-struct` (run B): task 9, SFT step 600 prior frozen, current recipe (M=4, K=4, 5 / 10 sampler, ε −0.3, base-conditioned residual, expert rows), same seed, same reset schedule |
| budget | `online_env_steps` 320k, checkpoints every 80k; the 240k checkpoint is a decision point (section 5) |
| collection distribution | each episode draws one of three resets with equal probability: a standard initial state (as today), a robot-init training variant, a layout training variant; the draw is logged per episode |
| robot-init reset | `LiberoEnv(task_suite=_get_suite("libero_10"), task_id=<variant>, init_states=False, is_libero_plus=True)`; the arm takes the variant pose and the simulator samples the object layout with the episode seed (`config.seed + env_steps`), the rule in docs/libero-plus/task9-plan.json |
| layout reset | the variant's stored state, `init_states=True`, uniform over its stored states |
| variant split | within each axis, variants ordered by index; every third one from the third is held out: robot-init test {642, 645, …, 681} (14 of 42), layout test {1984, 1987, …, 2020} (13 of 39); the rest train (28 and 26); the test sets cover every difficulty level present |
| proprioception | 8 floats in the published dataset's `observation.state` convention: end-effector position (3), orientation as axis-angle (3), gripper joint positions (2); from the environment with `obs_type="pixels_agent_pos"` (quaternion converted), from the parquet for expert rows; appended to the state of the Q ensemble, the plan critic and the residual actor in both runs |
| run A inputs | state pool (3072) + proprio; plan pool (3072) |
| run B inputs | state per-camera means (2 × 3072) + proprio; plan per-camera, per-latent-frame means (8 × 3072); text tokens excluded from the structured means |
| features stored in both runs | bf16, alongside the existing pools: `plan_struct_all` (M, 8, 3072); `state_struct` (2, 3072); `state_tokens` (32, 3072); `plan_latent_all` (M, 48, 4, 8, 16); `state_latent` (48, 1, 8, 16); `proprio` and `proprio_next` (8, float32); `object_state` (float32: position 3 and quaternion 4 for each of the base task's objects in a fixed order recorded in the run settings, plus the joint positions of articulated objects, read from the simulator; distractors ignored; offline use only, never a model input); `reset_kind` (0 standard, 1 robot-init, 2 layout) and `variant_index`. About 0.6 MB per chunk, 12 GB over a run; expert rows get the same keys from the published latents and parquet |
| W&B | project and entity as the previous runs; config records every new field; new per-episode metrics: collection success and length per reset kind, the reset draw, plan metrics per reset kind at the existing cadence |
| offline study | at the 80k checkpoints: refit V_plan and Q on pooled, structured, structured + proprio, attention-pooled state tokens, and an oracle with `object_state` + proprio as the upper bound, on both buffers; split-half agreement, replica agreement, MC-return fit |
| evaluation | deferred: the held-out LIBERO-Plus plan (section 4) and the standard 100-episode protocol, after the runs; GPUs are not shared with training |

The random-plan ablations already running (standard protocol and the existing LIBERO-Plus plan) are unaffected.

## 2. Why these choices

- Two runs, one difference: the paired held-out evaluation becomes a direct test of whether structured features let
  the policy adjust to layout shifts, in one cycle rather than two. The offline refits on both buffers test ranking
  consistency independently of the live choice.
- Both axes, one run each: the evaluation protocol pairs them, and the layout axis is where pooled features are most
  likely to be insufficient.
- One third standard resets: keeps in-distribution data in the buffer so the standard protocol can show any regression.
- Variants held out, not seeds: layout variants are stored states, so a seed split would train on test states.
- End-effector proprioception rather than joint angles: the actions are relative end-effector commands, so the
  correction lives in that space; the environment and the published expert data both provide it, with one conversion;
  joint angles exist in the environment and the raw LIBERO HDF5 demos but not in the published dataset the expert rows
  come from, and mapping episodes back to demos is plumbing with a mismatch risk. Alignment of expert and online rows
  is checked by resetting to a stored initial state and comparing against that demonstration's first-frame state.
- Attention pooling over stored tokens stays offline: untested, and the buffer holds the tokens for it.
- Object poses are stored for the oracle only: the critics rank plans and candidates at deployment, so a privileged
  input cannot be trained on and dropped; the oracle says whether the image features are the bottleneck.

## 3. Code changes

- `script/lingbot_rl_config.py`: `proprio` (bool), `state_features` ("pool" | "struct"), `plus_train` (axis →
  variant list; empty means today's collection), all in the protocol fingerprint; `collection_mix` fixed at thirds
  when `plus_train` is set.
- `script/lingbot_rl_train.py`: environment factory for plus variants (ENV_KWARGS from the plus driver, `_get_suite`),
  the per-episode reset draw, `obs_type="pixels_agent_pos"` when `proprio`, proprioception extraction and
  conversion, the new buffer keys, `proprio_next` at finalisation, per-reset-kind metrics; `brev/train.sh --plus`
  sourcing the plus activate script, `--state-features`, `--proprio`.
- `script/lingbot_rl_policy.py`: `_pool_from_latent` returns the structured means and the tokens alongside the pool
  (camera halves of the 8 × 16 grid; text tokens excluded from the structured means only); `select_plan` and
  `decode_candidates` carry them.
- `script/lingbot_rl_model.py`: state and plan dimensions from `state_features` and `proprio` for the critic, plan
  critic and actor; `critic_state` takes the structured inputs; recipe fingerprint; the inference state dict records
  both settings so evaluation reconstructs the model.
- `script/lingbot_rl_buffer.py`, `script/lingbot_rl_data.py`: new keys with legacy defaults; expert recipe v3 with
  structured features from the published latents and proprioception from the parquet.
- Evaluation: `evaluate` in `lingbot_rl_train.py` and the plus driver read `state_features` and `proprio` from the
  residual checkpoint, build the matching environment observation type and feed the policy.
- `docs/libero-plus/README.md`: the variant split.
- Tests: proprioception conversion against a stored initial state's first demo frame (GB10-only, skipped without the
  environment); structured means equal the pool on a synthetic token set; buffer round trip with the new keys; config
  fingerprint for both runs; the reset draw and variant sampling; expert featurisation v3 on the published-latent stub.

Unchanged: the SFT prior, the sampler, the plan-selection recipe, every result recorded so far.

## 4. Deferred: held-out plan and baselines

`docs/libero-plus/task9-heldout-plan.json` from the test variants: robot-init 14 × 4 seeds = 56, layout 13 × 3 seeds
+ a fourth seed for the first five = 44; 100 episodes, seeds from one RNG (seed 43), reset rules as the existing plan.
Policies on it: the SFT prior, the in-distribution `dice-t9-plan-m4` 320k, and the two new runs; the new runs also on
the standard protocol. Built and run after the runs are launched.

## 5. Budget

320k with the 240k checkpoint as a decision point. The in-distribution task 9 run gained nothing measurable in its
second half (79 at 160k, 80 at 320k), but its ranker kept improving with data (agreement 0.51 → 0.55, captured 0.73 →
0.81), and these runs split collection three ways, so 240k gives each distribution about 5k chunks against the 10k the
in-distribution run had by 160k. The learning-rate schedule is periodic, so stopping at 240k costs nothing; the call is
made from the per-reset-kind collection success and regret at 240k.

## 6. Sequence

1. Code changes with tests; a 300-step smoke per run under the plus environment on one box (all three reset kinds
   drawn, buffer keys present, memory and seconds per env step recorded).
2. Launch run A and run B, one per box; W&B as before.
3. At 80k: the offline feature study on the CPU of each box.
4. At 240k: decision on continuing to 320k.
5. After the runs: the held-out plan, its baselines, and the evaluations.

## 7. Risks

- Proprioception convention mismatch between the environment and the dataset silently misaligns expert and online
  rows: the unit check against a stored initial state gates the launch.
- The plus package must reproduce the standard task 9 initial states for the standard third of collection: checked by
  comparing a reset to initial state 0 under both installs before the smoke.
- Run B's first layers are about eight times larger (about 320 M parameters across the ensemble); memory and step time
  are read from the smoke.
- Resume files grow to about 16 GB; the save cadence stays at 80k.
- About 230 episodes per distribution, roughly eight per training variant; the held-out evaluation is the only
  measure of generalisation across variants, and a null result there is informative.
