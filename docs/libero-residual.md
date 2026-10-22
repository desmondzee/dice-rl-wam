# LIBERO residual RL on shifted robot starts

## Task and prior

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10), the 20-demo flow-matching prior from `docs/libero-prior.md` (`$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain_20/`), 128x128 agentview and wrist cameras, horizon 520. Milestone 2 trains a residual on top of that prior from procedurally shifted robot initial joints. This needs resets that reload a state without rebuilding the simulator (soft reset) and a controller whose nullspace reference follows the loaded joints.

Wrapper changes (`env/gym_utils/wrapper/libero_image.py`, factory in `env/gym_utils/__init__.py`):

- `make_libero_env(..., hard_reset=True)` forwards `hard_reset` to `OffScreenRenderEnv`. The default keeps pretraining and prior evaluation unchanged. The factory pops `hard_reset` from the `libero_image` block and builds the env with it.
- `LiberoImageWrapper.load_state(state)` calls `env.set_init_state(state)` and then `controller.update_initial_joints(state[1:8])` when those joints differ from `init_states[0][1:8]` by more than 1e-9. `env.reset()` rebuilds the controller, so its reference is canonical after every reset.

## Gate A: soft reset and controller reference

Script `script/dicerl_libero_gate_a.py`, output `$DICE_RL_DATA_DIR/libero/<task>/gate_a.json`. Run on Jarvis (GB10), 2026-10-22, latest run exit status 1, from the replay rule only (see the replay rows).

| check | result |
|---|---|
| hard reset latency (`reset` + `set_init_state`, mean of 10) | 0.610 s |
| soft reset latency | 0.0020 s (311x faster) |
| async rate, 8 envs, chunk 8, 163 chunks (about 20 episodes), soft | 1216 env steps/s |
| demo_0 replay, 10 passes, hard | 9/10 in the latest run (6/10, 8/10 and 5/10 in earlier runs) |
| demo_0 replay, 10 passes, soft | 6/10 in the latest run (7/10, 7/10 and 9/10 in earlier runs); per-pass first-success steps are in `gate_a.json` |
| replay pass rule | soft at least 5 of 10 and at least hard minus 2: failed in the latest run (6 against 9), passed in the two runs before. Pooled over four runs: hard 28/40, soft 29/40 |
| drift: 200 soft resets to state 0, 20 random steps between, max proprio spread | 0.0 (limit 1e-4) |

Open-loop replay of a demo is nondeterministic even with hard resets (5, 6, 8 and 9 of 10 across runs), so the replay check compares success counts instead of requiring a single pass.

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

## Gate C

## Config

## Launch

## Sweep

## Final results
