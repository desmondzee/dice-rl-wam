# Plan selection trained under LIBERO-Plus perturbations (task 9)

Status: design for review. One run answers two questions: does training the critic, plan critic and residual under
the robot-initial-state and layout perturbations improve held-out LIBERO-Plus performance with the prior frozen; and,
from the same buffer, do structured pool features and proprioception make the plan ranking more consistent.
References: docs/heldout-results.md (LIBERO-Plus section), docs/research-video-branch-ideas.md, docs/libero-plus/.

## 1. Decisions

| item | decision |
|---|---|
| run | `dice-t9-plus-m4`: task 9, SFT step 600 prior frozen, current recipe (320k env steps, M=4, K=4, 5 / 10 sampler, ε −0.3, base-conditioned residual, expert rows as today) |
| collection distribution | each episode draws one of three resets with equal probability: a standard initial state (as today), a robot-init training variant, a layout training variant |
| robot-init reset | `LiberoEnv(task_suite=_get_suite("libero_10"), task_id=<variant>, init_states=False, is_libero_plus=True)`; the arm takes the variant pose and the simulator samples the object layout with the episode seed (`config.seed + env_steps`), the rule in docs/libero-plus/task9-plan.json |
| layout reset | the variant's stored state, `init_states=True`, uniform over its stored states |
| variant split | within each axis, variants ordered by index; every third one from the third is held out: robot-init test {642, 645, …, 681} (14 of 42), layout test {1984, 1987, …, 2020} (13 of 39); the rest train (28 and 26); the test sets cover every difficulty level present |
| held-out plan | `docs/libero-plus/task9-heldout-plan.json`: robot-init 14 variants × 4 seeds = 56, layout 13 variants × 3 seeds + a fourth seed for the first five = 44; 100 episodes, seeds from one RNG (seed 43), same reset rules as the existing plan |
| baselines on the held-out plan | the SFT prior; the in-distribution plan policy `dice-t9-plan-m4` 320k (V_plan ranker); then `dice-t9-plus-m4` at 160k and 320k; the new policy also runs the standard 100-episode protocol as the in-distribution check |
| live critic and actor inputs | image pools unchanged; proprioception appended to the state for the Q ensemble, the plan critic and the residual actor |
| proprioception | 8 floats in the published dataset's `observation.state` convention: end-effector position (3), orientation as axis-angle (3), gripper joint positions (2); from the environment with `obs_type="pixels_agent_pos"` (quaternion converted), from the parquet for expert rows |
| features stored for the offline study | bf16, alongside the existing pools: `plan_struct_all` (M, 8, 3072) camera × latent-frame means of the 128 plan tokens, text tokens excluded; `state_struct` (2, 3072) per-camera means of the 32 state tokens; `state_tokens` (32, 3072); `plan_latent_all` (M, 48, 4, 8, 16); `state_latent` (48, 1, 8, 16); `proprio` and `proprio_next` (8, float32). About 0.6 MB per chunk, 12 GB over the run; expert rows get the same keys from the published latents |
| offline study | at the 80k checkpoint: refit V_plan and Q on pooled, structured means, structured plus proprio, and attention-pooled state tokens; metrics as in the ranker studies (split-half agreement, replica agreement, MC-return fit); it decides the inputs of any follow-up run, not of this one |
| compute | the run on one GB10 (about 57 h); the two baselines on the held-out plan on the other (about 2 h and 4 h); evaluations of the new policy after 160k and 320k (about 4 h each on LIBERO-Plus, 3.5 h standard) |

The random-plan ablations already running (standard protocol and the existing LIBERO-Plus plan) are unaffected and
report before this run starts.

## 2. Why these choices

- Both axes, one run: the evaluation protocol already pairs them, the layout axis is where the pooled features are
  most likely to be insufficient, and the stored features let the offline study say so without a second collection.
- One third standard resets: keeps in-distribution data in the buffer so the standard protocol can show any regression.
- Variants held out, not just seeds: layout variants are stored states, so a seed split would train on test states.
- Proprioception live, structured features offline: proprioception is eight floats with a physical reason on the
  robot-init axis and no evidence against it; the structured features change the critic's input size by 8× and every
  larger refit so far ranked worse out of sample, so they are stored and tested before they enter a recipe.
- The convention of `observation.state` is used so expert rows and online rows carry the same proprioception.

## 3. Code changes

- `script/lingbot_rl_config.py`: `proprio` (bool, in the protocol), `plus_train` (dict of axis → variant list, in the
  protocol; empty means today's collection), `collection_mix` fixed at thirds when `plus_train` is set.
- `script/lingbot_rl_train.py`: environment factory for plus variants (ENV_KWARGS from the plus driver, `_get_suite`),
  the per-episode reset draw, `obs_type="pixels_agent_pos"` when `proprio`, proprioception extraction and conversion,
  the new buffer keys, `proprio_next` at finalisation, `brev/train.sh --plus` sourcing the plus activate script.
- `script/lingbot_rl_policy.py`: `_pool_from_latent` returns the structured means and the tokens alongside the pool
  (camera halves of the 8 × 16 grid; text tokens excluded from the structured means only).
- `script/lingbot_rl_model.py`: state dimension + 8 for critic, plan critic and actor when `proprio`; recipe fingerprint.
- `script/lingbot_rl_buffer.py`, `script/lingbot_rl_data.py`: new keys with legacy defaults; expert recipe v3 with
  structured features from the published latents and proprioception from the parquet.
- Evaluation: `evaluate` in `lingbot_rl_train.py` and the plus driver pass proprioception to the policy when the
  residual checkpoint records it; `make_env` takes the observation type.
- `docs/libero-plus/make_plan.py` writes the held-out plan; `README.md` records the split.
- Tests: proprioception conversion against a stored init state's first demo frame; structured means equal the pool on
  a synthetic token set; buffer round trip with the new keys; config fingerprint; collection draw.

Unchanged: the SFT prior, the sampler, the plan-selection recipe, every result recorded so far.

## 4. Sequence

1. Held-out plan, split and README; the SFT baseline on the held-out plan on the first free box.
2. Code changes with tests; a 300-step smoke under the plus environment (both axes drawn, buffer keys present,
   memory and seconds per env step recorded).
3. The in-distribution 320k baseline on the held-out plan.
4. Launch `dice-t9-plus-m4`; W&B as before.
5. At 80k: the offline feature study on the CPU of the same box.
6. At 160k and 320k: held-out plan and standard protocol evaluations.

## 5. Risks

- Proprioception convention mismatch between the environment and the dataset silently misaligns expert and online
  rows: the unit check against a stored init state gates the launch.
- The plus package must reproduce the standard task 9 initial states for the standard third of collection: checked by
  comparing a reset to init state 0 under both installs before the smoke.
- Resume files grow to about 16 GB; the save cadence stays at 80k.
- One third of the budget per distribution is about 230 episodes each, roughly eight per training variant; the
  held-out evaluation is the only measure of generalisation across variants, and a null result there is informative.
