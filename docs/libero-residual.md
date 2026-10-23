# LIBERO residual RL on shifted robot starts

## Task and prior

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10), the 20-demo flow-matching prior from `docs/libero-prior.md` (`$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain_20/`), 128x128 agentview and wrist cameras, horizon 520. Milestone 2 trains a residual on top of that prior from procedurally shifted robot initial joints. This needs resets that reload a state without rebuilding the simulator (soft reset) and a controller whose nullspace reference follows the loaded joints.

Wrapper changes (`env/gym_utils/wrapper/libero_image.py`, factory in `env/gym_utils/__init__.py`):

- `make_libero_env(..., hard_reset=True)` forwards `hard_reset` to `OffScreenRenderEnv`. The default keeps pretraining and prior evaluation unchanged. The factory pops `hard_reset` from the `libero_image` block and passes it both to `make_libero_env` and to `LiberoImageWrapper`, so the wrapper's soft-reset gripper handling follows the env's mode (until the fix recorded under gate A, only the env received it).
- `LiberoImageWrapper.load_state(state)` calls `env.set_init_state(state)` and then `controller.update_initial_joints(state[1:8])` when those joints differ from `init_states[0][1:8]` by more than 1e-9. `env.reset()` rebuilds the controller, so after a reset its reference is LIBERO's default (the robot's `init_qpos`), not the demo joints. Only stored state 0 matches the canonical joints, so it keeps that default; stored states 1 to 49 carry about 0.02 rad of per-joint noise around state 0, differ from it by more than 1e-9, and therefore get the controller-reference update. Every milestone-2 number (prior baselines, training, in-training evaluation) used this rule, so the internal comparison is consistent; an explicit canonical rule with re-measured baselines is deferred to the layout axis.

## Gate A: soft reset and controller reference

Gate A validated the directly constructed wrapper (`LiberoImageWrapper(env, ..., hard_reset=False)`). The vectorised factory path (`make_async`) dropped the flag before the wrapper, so training and evaluation through it ran soft resets with `LiberoImageWrapper.hard_reset` still True and without the gripper zeroing; this was found in the final review and is fixed (the factory passes `hard_reset` to the wrapper; a fake-backed test builds through `make_async` and asserts both the raw env and the wrapper receive it). Prior `state_100.pt`, 150 in-distribution episodes (canonical 50 states, seeds 0 1 2, 25 envs), Jarvis, 2026-10-22:

| reset mode | success | stderr | mean length |
|---|---|---|---|
| hard (`--hard-reset true`, run after the fix) | 0.467 (70 of 150) | 0.041 | 438.6 |
| soft with the gripper carry-over (`--hard-reset false`, before the factory fix) | 0.540 (81 of 150) | 0.041 | 430.4 |
| soft, fixed (`--hard-reset false`, after the fix) | 0.453 (68 of 150) | 0.041 | 438.8 |

The earlier recorded hard run of the same protocol gave 0.513 (77 of 150). The four runs disagree on 49 to 64 of the 150 episodes pairwise (including the two hard runs, 55), so per-run success varies by about one standard error and the carry-over effect (0.540 against 0.453 and 0.467) is not resolved at this sample size; the fixed soft mode and hard mode agree within noise. Factory-path re-check after the fix (2 soft envs through `make_async`, 20 random-action episodes each, then a reset to stored state 0): max abs proprioception difference against a fresh hard-reset build 0.0 for both envs (limit 1e-4); `robot.gripper.current_action` right after the soft `restart()` is zero in both envs (it was `[0.5, -0.5]` and `[-1, 1]` before, nonzero after all 20 episodes in each).

Script `script/dicerl_libero_gate_a.py`, output `$DICE_RL_DATA_DIR/libero/<task>/gate_a.json`. Run on Jarvis (GB10), 2026-10-22, latest run exit status 0.

| check | result |
|---|---|
| hard reset latency (`reset` + `set_init_state`, mean of 10) | 0.627 s |
| soft reset latency | 0.0020 s (318x faster) |
| async rate, 8 envs, chunk 8, 163 chunks (about 20 episodes), soft | 1212 env steps/s |
| demo_0 replay, 10 passes, hard | 6/10 in the latest run (9/10, 6/10, 8/10 and 5/10 in earlier runs) |
| demo_0 replay, 10 passes, soft | 5/10 in the latest run (6/10, 7/10, 7/10 and 9/10 in earlier runs); per-pass first-success steps are in `gate_a.json` |
| replay pass rule | soft at least 5 of 10: passed in the latest run (5). Hard and soft counts are both recorded but not compared in the exit code. Pooled parity: hard 28/40, soft 29/40 over the four earlier runs, hard 34/50, soft 34/50 including the latest |
| drift: 200 soft resets to state 0, 20 random steps between, max proprio spread | 0.0 (limit 1e-4) |

Open-loop replay of a demo is nondeterministic even with hard resets (5, 6, 8, 9 and 6 of 10 across runs), so the replay check counts successes instead of requiring a single pass. An earlier rule also required soft at least hard minus 2; it failed once (6 against 9) within binomial noise (the standard deviation of the difference of two 10-pass counts is about 2), so the parity comparison is now reported from the pooled counts instead of gated.

Equivalence of the first wrapper observation after `reset(options={"init_state": s})` against a fresh hard-reset build (max abs proprioception difference; mean abs image difference over 255 for agentview and wrist):

| state | hard after 20 random episodes | soft fresh | soft after 20 random episodes |
|---|---|---|---|
| 0 | 0.0; 0.0043, 0.017 | 0.0; 0.0062, 0.0068 | 0.0; 0.0086, 0.021 |
| 17 | 0.0; 0.013, 0.016 | 0.0; 0.0090, 0.014 | 0.0; 0.0047, 0.011 |
| 42 | 0.0; 0.0037, 0.017 | 0.0; 0.011, 0.021 | 0.0; 0.012, 0.028 |

Image differences of 0.004 to 0.03 are the same order for hard-after-random, so they are render jitter that exists with hard resets too (gate 0 reset jitter), not a soft-reset effect.

Controller reference (shifted joints are state 0 plus 0.3 rad with alternating signs, 50 NOOP steps, max joint deviation from the loaded joints):

| load | max drift (rad) |
|---|---|
| canonical state 0 (through `wrapper.reset` on a fresh soft env, same window as the restore row) | 0.042 |
| shifted, `set_init_state` only | 0.277 |
| shifted, with `update_initial_joints` | 5.5e-9 |
| shifted, through `wrapper.reset` | 2.7e-9 |
| shifted through the wrapper, then canonical state 0 through the wrapper, same soft env | 0.0424 (identical to the canonical row) |

The canonical row is nonzero because the controller's default reference is the robot's `init_qpos` (second joint -0.161), not the demo joints (-0.133); a hard-reset env gives the same 0.0424. Restoring the canonical reference is therefore checked as equality with the canonical row (within 1e-6), not as 2e-2; the with-update bound is 2e-2. `load_state` compares joints to the canonical ones with `rtol=0, atol=1e-9`.

Soft-reset gripper fix: the Panda gripper's `current_action` integrator survives a soft `env.reset()` (hard mode rebuilds it), which first gave a 3.0e-3 proprioception spread in the gripper dimensions (2.1e-4 in the equivalence check). The wrapper now sets `robot.gripper.current_action = np.zeros(gripper.dof)` after every `env.reset()` when `hard_reset` is false, which is the value of a fresh gripper.

## Gate B

Script `script/dicerl_libero_gate_b.py` (part 1), output `$DICE_RL_DATA_DIR/libero/<task>/gate_b.json`. Run on Jarvis (GB10), 2026-10-22, exit status 0. One soft-reset env, real init states, `set_shift_seed(1)` before each magnitude (the per-reset stream is `RandomState([seed, reset_count])`), 200 training-path resets cycling over the 50 canonical states. The offset is a uniform direction on the 7-sphere with norm uniform in `[0, m]`; a draw is rejected when a joint is within 0.02 rad of its limit after `sim.forward()` or a robot or gripper geom touches a geom that is not a gripper geom, resampled up to 20 times, then falls back to the unshifted state.

| magnitude (rad) | rejections / tries | rejection rate | fallbacks | joint l2 norm p10 / p50 / p90 | eef displacement p10 / p50 / p90 (m) |
|---|---|---|---|---|---|
| 0.1 | 0 / 200 | 0.000 | 0 | 0.011 / 0.054 / 0.095 | 0.0038 / 0.016 / 0.040 |
| 0.3 | 10 / 210 | 0.048 | 0 | 0.032 / 0.151 / 0.282 | 0.0093 / 0.046 / 0.116 |
| 0.5 | 14 / 214 | 0.065 | 0 | 0.054 / 0.247 / 0.467 | 0.0156 / 0.075 / 0.194 |
| 0.6 | 15 / 215 | 0.070 | 0 | 0.064 / 0.301 / 0.560 | 0.0187 / 0.091 / 0.231 |

Rejection rate is rejections over tries (rejections plus accepted draws); the gate fails if it exceeds 0.2 at 0.6 (it is 0.070). Joint norms are of accepted draws. Eef displacement is the norm of the unnormalised end-effector position after the shifted reset minus that after an unshifted reset of the same stored state (10 settle steps each); two unshifted resets of the same state give identical positions (jitter 0.0), so the displacement is entirely the shift. The earlier run with a summed seed gave the same statistics within noise (0.6 rad: 11 / 211, norm p50 0.275, eef p50 0.083). A diagnostic of 600 raw draws at 0.6 rad (summed-seed stream) found no joint-limit rejections for this task (the narrowest margin never binds) and 19 contact rejections, all gripper or hand geoms against microwave geoms (`microwave_1_g14`, `g15`, `g17`) plus two `robot0_link6` contacts, so the contact check fires on the real env.

Fallback resets reload the unshifted state after a fresh `env.reset()`, because `load_state` only updates the controller reference when the joints differ from the canonical ones and would otherwise leave the reference of the last rejected draw in place.

## Gate B part 2: evaluators, LIBERO-Plus poses and prior baselines

Evaluators (all take `--ckpt --envs --seeds (one or more) --states --shift --shift-seed --hard-reset {true,false} --out --device`; `--hard-reset` defaults to the config's `libero_image.hard_reset` (true when absent) and the resolved mode is written to `summary.json` as `hard_reset`):

- `script/dicerl_libero_eval.py` (prior, `state_100.pt`): `--states PATH` replaces the canonical init-state array with the rows of an npz (`states`, `labels`; the episode plan runs over those rows), `--shift M` passes `options["shift"] = {"magnitude": M, "seed": shift_seed + plan index}` on every reset (the `libero_image` block of the prior config has no `hard_reset` key, so these runs hard-reset unless `--hard-reset` says otherwise). `episodes.csv` carries `label` (the state index as a string without `--states`); `summary.json` records `states` (path or `canonical`), `shift`, `shift_seed`, `seeds`, `n_envs`, `weights`.
- `script/dicerl_libero_rl_eval.py` (residual, `model_step_<N>.pth` next to its `.hydra/config.yaml`): same flags, adds `iteration` to the summary. The visual features come from the prior (`pretrained_flow_policy.network.extract_visual_features`, state concatenated, `obs_dim = 8 + visual_feature_dim`), actions from `get_exploration_action(num_samples=16, "max_q_min", training_step=10**6)`.
- `script/dicerl_libero_plus_states.py`: `export` (LeRobot environment, LIBERO-Plus package) and `build` (LIBERO venv) write the held-out sets; `breakdown --episodes` prints success per variant.

LIBERO-Plus poses. Suite indices 640 to 681 of `libero_10` are tasks named `<task>_view_0_0_100_0_0_initstate_N`; `libero/envs/env_wrapper.py` turns `initstate_N` into the robot class `MountedPanda<N>` (the bddl files named so do not exist), and `MountedPanda<N>.init_qpos` is the variant's 7 joints. The stored init files are the canonical ones, so `export` reads the joints from the classes and `build` writes them into state indices 1 to 7 of canonical states 0 to 6 (variant-major, label `<variant>:<state>`, 294 rows; the monitoring set is states 0 and 1, 84 rows). The canonical stored states are the base `MountedPanda.init_qpos` plus noise (per-joint std 0.016 to 0.022), not a single vector. Check: 42 distinct poses, l2 offset from the base `MountedPanda.init_qpos` 0.1000 to 0.5000 (variants 640 to 647 at 0.1, 648 to 653 at 0.2, 654 to 661 at 0.3, 662 to 671 at 0.4, 672 to 681 at 0.5); from canonical state 0 (the vector `[-0.031, -0.133, 0.015, -2.439, 0.0, 2.207, 0.778]`) 0.0768 to 0.5264; the 294 rows against their own canonical state 0.0666 to 0.5629.

Prior baselines (`state_100.pt`, Jarvis GB10, 25 envs, 520 steps, one run at a time, wall time per run including model load and 25-env start-up):

| evaluation | episodes | success | stderr | mean length | wall |
|---|---|---|---|---|---|
| held-out LIBERO-Plus poses (`plus_robot_init.npz`, seed 0) | 294 | 0.136 (40) | 0.020 | 496.4 | 384 s |
| monitoring set (`plus_robot_init_monitor.npz`, seed 0) | 84 | 0.202 (17) | 0.044 | 483.9 | 137 s |
| in distribution (canonical 50 states, seeds 0 1 2) | 150 | 0.513 (77) | 0.041 | 435.3 | 210 s |
| procedural shift 0.1 (shift seed 1000, seeds 0 1) | 100 | 0.42 | 0.049 | 449.8 | 143 s |
| procedural shift 0.3 | 100 | 0.23 | 0.042 | 482.6 | 140 s |
| procedural shift 0.5 | 100 | 0.13 | 0.034 | 495.0 | 137 s |
| procedural shift 0.6 | 100 | 0.11 | 0.031 | 499.0 | 137 s |

The in-distribution rate is inside the earlier seed-pair spread (0.52, 0.47, 0.38; pooled 0.457 over 300) so the wrapper changes of tasks 1 and 2 did not move the canonical baseline (seeds 0, 1, 2: 25, 25, 27 of 50).

Held-out success by variant (mean over its 7 states) falls with the offset from the base pose:

| offset (variants) | held-out successes | monitoring successes |
|---|---|---|
| 0.1 (640 to 647) | 20 / 56 | 9 / 16 |
| 0.2 (648 to 653) | 13 / 42 | 4 / 12 |
| 0.3 (654 to 661) | 7 / 56 | 3 / 16 |
| 0.4 (662 to 671) | 0 / 70 | 1 / 20 |
| 0.5 (672 to 681) | 0 / 70 | 0 / 20 |

Per variant: 640 0.43, 641 0.57, 642 0.29, 643 0.29, 644 0.14, 645 0.14, 646 0.43, 647 0.57, 648 0.00, 649 0.43, 650 0.14, 651 0.14, 652 0.57, 653 0.57, 654 0.00, 655 0.43, 656 0.29, 657 0.00, 658 0.00, 659 0.14, 660 0.00, 661 0.14, and 0.00 for every variant from 662 to 681. Results are in `<prior run>/eval_plus_heldout`, `eval_plus_monitor`, `eval_indist`, `eval_shift_<m>` on Jarvis. Procedural curve and fallbacks: gate B measured the per-try rejection rate at 0.6 rad as 15/215 (0.070) with 0 fallbacks at every magnitude (a fallback needs 20 consecutive rejections), so the curve is not biased by fallbacks. Both evaluators now write `shift_stats` (`draws`, `rejections`, `fallbacks`, summed over the workers with `call_sync("shift_stats")`; draws include the padded duplicate envs of the last batch) to `summary.json`. A rerun of the 0.6 point with the same flags (`eval_shift_0.6_stats`, 137 s) recorded `shift_stats` draws 100, rejections 0, fallbacks 0 and success 0.13 (stderr 0.034; 0.11 in the first run, a difference within the evaluation noise). Zero rejections in 100 draws is a property of the shift seeds 1000 to 1099 used by the flags: the same draws through a bare wrapper gave 0 rejections (soft) and 1 (hard), seeds 1100 to 1199 gave 7 of 100, 1200 to 1399 gave 7 of 200, and 200 draws on other seeds gave 10 and 15, in line with the gate B rate.

## Gate C

Smoke run on Jarvis (GB10), 2026-10-22, seed 42, launched with `WATCH_EVERY=100 bash brev/dicerl/finetune_libero.sh <task> 42 train.num_train_steps=200 train.save_freq=100` (sessions `dicerl-rl-kitchenscene6-42` and `dicerl-rlwatch-42`, run directory `libero-finetune/<name>/2026-10-22_16-24-13_42`). Both sessions ended on their own (training at iteration 200, then the watcher saw its session gone and exited 0); no session or worker process was left.

| measurement | result |
|---|---|
| iteration-0 evaluation (before training) | 7 min 50 s: frozen prior 300 episodes 4 min 25 s (success 0.447), untrained residual 300 episodes 3 min 26 s (success 0.000, mean length 520) |
| iterations 0 to 100 | 372 s (3.7 s per iteration); no gradient updates until iteration 64, when the first episodes finish and the replay buffer holds a batch |
| iterations 100 to 200 | 1045 s (10.45 s per iteration, 344 iterations per hour), all with 20 gradient steps per iteration; includes the monitoring evaluation of iteration 100 sharing the GPU for about 93 s |
| in-training evaluation at iteration 200 | 3 min 31 s, success 0.487 (146 of 300), mean length 428.8 (canonical stored states) |
| monitoring evaluation, iteration 100 (`model_step_100.pth`, 84 episodes, seed 0) | success 0.083 (7 of 84, stderr 0.030), about 93 s sharing the GPU with training |
| monitoring evaluation, iteration 200 | success 0.119 (10 of 84, stderr 0.035), about 120 s; frozen prior on this set 0.202 |
| residual evaluator | loaded both checkpoints (`iteration` 100 and 200 in `summary.json`, `shift_stats` all zero), results in `heldout_100`, `heldout_200`, rows in `heldout_sweep.json` |
| W&B | `shift/magnitude`, `shift/draws`, `shift/rejections`, `shift/fallbacks` present on 135 of 136 logged rows (read back through the API); all zero in the smoke because the ramp starts at iteration 3,000 |

Evaluation times are derived from the 300 s watcher poll cycle (start) and the summary file time (end), so they are good to a few seconds only for the end and to the poll alignment for the start.

Projection to 13,000 iterations: 13,000 × 10.45 s = 37.7 h of training (34.4 h if the 93 s of overlap is removed from the 100 to 200 window), plus 13 in-training evaluations of 3.5 min (0.74 h), the iteration-0 evaluation (0.13 h) and 6 monitoring evaluations of about 2 min (0.2 h): about 38.8 h, under the 48 h limit. The smoke does not reach exploration after the flow warm-up (iteration 3,000 onward uses 16 samples through `max_q_min`; the smoke used the single-sample path), nor a non-zero shift (resets with draws and resamples), so the rate after iteration 3,000 is an extrapolation; the monitoring evaluator, which uses 16 samples on 25 envs, took about 100 s for 84 episodes, which suggests the extra rollout cost is small next to the 20 gradient steps.

A bare check of the training path with a non-zero shift (2 workers through `make_async` with the run's wrappers including `multi_step_full`, `set_shift_seed` 42 and 43, `set_shift(magnitude=0.6)`, ten no-option `reset()` calls): `shift_stats` per worker `draws` 10 and 10, `rejections` 1 and 0, `fallbacks` 0 and 0, joint proprioception spread across resets 0.76 (normalised units), so the vector-env calls and the auto-reset path reach the wrapper.

The step-0 success of 0.000 is the untrained residual (random actor initialisation on top of the base action, no behaviour-cloning updates yet), which is not the prior; the frozen prior in the same evaluation is 0.447.

Dependency: the vendored agent imports `plotly` at module level (`util/state_visualization.py`), which the LIBERO venv lacked. `plotly==5.24.1` (the root `pyproject.toml` pin; installs `tenacity==9.2.1` with it) was installed into `dicerl-libero-venv` and added to `brev/dicerl/setup_libero.sh`; no existing package changed.

## Config

`brev/dicerl/cfg/libero/<task>/ft_distill_residual_flow_unet_img.yaml`, the released Square image config (`cfg/robomimic/finetune/square/ft_distill_residual_flow_unet_img.yaml`) with comments removed and these differences (everything else is as released: `gradient_steps` 20, 10 critics, `bc_loss_weight` 50, `n_step` 3, expert ratio 0.5 to 0.1 over 10,000, `eval_freq` 1,000, `eval_n_envs` 10, `num_eval_episodes` 30, `max_q_min` with 16 samples, `q_filtering_warmup_steps` and `replay_flow_warmup_steps` 3,000):

| key | released Square image | LIBERO residual |
|---|---|---|
| `_target_` | `TrainDistillResidualFlowImgAgent` | `script.dicerl_libero_rl.ShiftedResidualAgent` |
| `logdir` | `log_dir/robomimic-finetune/...` | `$DICE_RL_LOG_DIR/libero-finetune/...` |
| `base_policy_path` | Square pretrain `state_2000.pt` | prior `state_100.pt` of `2026-10-22_10-11-34_42` |
| `normalization_path`, `expert_dataset.dataset_path` | `robomimic/square-img/ph_pretrain`, `ph_finetune/train.npz` | `libero/<task>/ph_pretrain_20/normalization.npz`, `train.npz` |
| `robomimic_env_cfg_path`, `env.specific` | set | removed |
| `env_name`, `env.env_type` | `square`, unset | `<task>`, `libero` |
| `obs_dim`, `horizon_steps`, `act_steps`, `noise_dim` | 9, 4, 4, 28 | 8, 8, 8, 56 |
| `env.max_episode_steps` | 400 | 520 |
| wrapper block | `robomimic_image` (image and low-dim keys, `use_6d_rot`, `success_steps_before_termination` 1) | `libero_image` (`normalization_path`, `shape_meta`, `success_steps_before_termination` 1, `hard_reset: false`); `multi_step` unchanged |
| `shape_meta` | rgb 96×96×6, state 9 | rgb 128×128×6, state 8 |
| `wandb` | entity null, `robomimic-...` project | entity `$DICE_RL_WANDB_ENTITY`, project `dice-rl-libero` |
| `train.num_train_steps`, `train.save_freq` | 150,000, unset (default 1,000) | 10,000,000, 1,000 |
| `model.condition_residual_on_base_action` | false | true |
| `model.use_soft_q_filtering` | false | true |
| `model.q_underestimation_threshold` | -0.25 | -0.55 |
| `model.always_retain_bc_loss_for_expert_data` | true | false |
| `expert_dataset.max_n_episodes` | 50 | 20 |
| `expert_dataset.success_steps_for_termination` | `robomimic_image` block | `libero_image` block |
| `shift` | absent | `end_magnitude: 0.6, ramp_start: 3000, ramp_end: 6000, seed: 42` |

The released config has no network block (the prior's network is rebuilt from the prior run's `.hydra/config.yaml`, crop 116 of 128), so none was copied. `script/dicerl_libero_rl.py` sets the per-worker shift seeds once (`shift.seed + worker index`), applies `ramp_magnitude` through `set_shift` when the magnitude changes, and logs the shift metrics to W&B once per iteration. `script/dicerl_libero_rl_watch.py` evaluates `model_step_<N>.pth` (N a multiple of `--every`) on the monitoring states with the residual evaluator (seed 0, 25 envs), appends `{iteration, env_steps, success_rate, stderr}` to `heldout_sweep.json`, never stops training, skips checkpoints younger than 30 s while training is alive, retries a failed evaluation on the next poll up to 3 times and then records `{iteration, env_steps, error}` and moves on, and exits only when the training session is gone and nothing is pending.

## Launch (2026-10-22 17:12 BST), stopped

This launch was stopped at iteration about 300 because of the factory defect recorded under gate A (its soft resets ran without the gripper zeroing). It is discarded and a relaunch on the fixed code follows. The text below describes the schedule, which the relaunch keeps.

`bash brev/dicerl/finetune_libero.sh KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it 42`: sessions `dicerl-rl-kitchenscene6-42` and `dicerl-rlwatch-42`, run directory `$DICE_RL_LOG_DIR/libero-finetune/..._ft_distill_residual_flow_unet_img_ta8_fl10/2026-10-22_17-11-57_42`, W&B run `02jssndy` in `dice-rl-libero`. Schedule: shift 0 until iteration 3,000, linear to 0.6 rad at 6,000, then held; checkpoints every 1,000 iterations; held-out monitoring every 2,000 on the 84-episode set; stop at 850k env steps (about 13,000 iterations) or after three consecutive monitoring rows without improvement once the ramp has ended. Projection from gate C: about 39 h, to be re-checked at iterations 3,500–4,000 once the 16-sample exploration and the shift are active. The run cannot be resumed: a restart restarts the ramp and the per-worker reset-count RNG.

## Relaunch (2026-10-22 17:44 BST)

Same command and schedule as the stopped launch, on the fixed code (commit 64f5593): run directory `.../2026-10-22_17-44-34_42`, W&B run `7t5af9s4`, log `finetune_kitchenscene6_42_20261022-174434.log`, sessions `dicerl-rl-kitchenscene6-42` and `dicerl-rlwatch-42`. Every environment built through the factory now soft-resets with the gripper integrator zeroed.

## Control run (Watson, 2026-10-22 22:57 BST)

The user's decision after the relaunch: the same recipe on canonical starts only, as the control for attribution. Watson was prepared over the boxes' LAN (repo, 20-demo dataset, held-out state files and the prior checkpoint copied from Jarvis; LIBERO venv built with `brev/dicerl/setup_libero.sh`). Launched with `bash brev/dicerl/finetune_libero.sh <task> 42 shift.end_magnitude=0` on the same commit (64f5593): the first two attempts (`2026-10-22_22-56-55_42`, W&B `uycoqpgp`; `2026-10-22_22-59-00_42`, W&B `rp1afksx`) died at environment creation: the configs' relative `data_dir/...` paths rely on the `data_dir` and `log_dir` symlinks in the repository root that the reproduction setup created on Jarvis and the LIBERO setup script did not (fixed in `setup_libero.sh`, commit 0678f4b; the concurrent LIBERO asset download that appeared in the first log was incidental). Third launch: run directory `.../2026-10-22_23-00-04_42` under Watson's `$DICE_RL_LOG_DIR/libero-finetune/`, W&B run `n42pshcu`, sessions `dicerl-rl-kitchenscene6-42` and `dicerl-rlwatch-42` on Watson. Identical config apart from `shift.end_magnitude: 0` (the ramp then holds the shift at 0 throughout, so every training start is a canonical stored state); same seed, same in-training and held-out monitoring protocols. Both runs use the fixed soft-reset path.

## Sweep

| iteration | env steps | shift (rad) | shifted run: in-training canonical | shifted run: held-out monitoring (84, ±0.05) | control: in-training canonical | control: held-out monitoring (84, ±0.05) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.45 (prior) | 0.20 (prior) | 0.45 (prior) | 0.20 (prior) |
| 2,000 | 128k | 0 | 0.86 | 0.49 | 0.85 | 0.50 |
| 4,000 | 256k | 0.20 | 0.98 | 0.52 | 0.98 | 0.55 |
| 6,000 | 384k | 0.60 | 1.00 | 0.65 | 0.99 | 0.49 |
| 8,000 | 512k | 0.60 | 1.00 | 0.71 | 1.00 | 0.52 |
| 10,000 | 640k | 0.60 | 1.00 | 0.76 | — | — |

Final-evaluation chains (2026-10-23 22:51 BST): a box-side tmux session `final-eval` on each box waits for `model_step_13000.pth` (the last checkpoint at or under 850k env steps) and runs the six final protocols on it with `--hard-reset true` (held-out 294, in-distribution 150, procedural 0.1/0.3/0.5/0.6 with 100 episodes each), writing under `<run>/final_13000/`. Training is deliberately not stopped at 13,000: the vendored agent has no resume path (a restart would reset the iteration counter, the warm-ups, the expert-ratio schedule, the ramp and the replay buffer), so keeping the process alive is the only way to extend a run; the decision to stop or extend is taken after the final numbers, and the held-out watcher keeps producing rows every 2,000 iterations meanwhile.

Read on 2026-10-23 22:45 BST (shifted run at iteration 10,000 after 29 h, 10.3 s per iteration over the last 2,000; control at 8,000 after 24 h). Both runs reach 0.98–1.00 in distribution by 256k env steps. On the held-out LIBERO-Plus poses the two runs are indistinguishable while the shift is zero (0.49 vs 0.50 at 128k) and small (0.52 vs 0.55 at 256k, magnitude 0.2), then separate once the ramp reaches 0.6 rad: 0.65 vs 0.49 at 384k, 0.71 vs 0.52 at 512k, with the shifted run still rising at 640k (0.76). The control's held-out number is flat at about 0.50 from 128k on: in-distribution RL transfers the prior's 0.20 to about 0.50 on these poses and no further. Residual norm and filter activity columns are read from W&B at the final evaluation.

## Final protocols

Prior and residual are evaluated in one matched reset mode, hard, matching the prior baselines above (`--hard-reset true` on both evaluators); the resolved mode is recorded as `hard_reset` in each `summary.json`. Monitoring rows during training come from the config's soft mode and are compared only with each other.
