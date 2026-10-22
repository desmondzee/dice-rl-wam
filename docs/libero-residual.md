# LIBERO residual RL on shifted robot starts

## Task and prior

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10), the 20-demo flow-matching prior from `docs/libero-prior.md` (`$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain_20/`), 128x128 agentview and wrist cameras, horizon 520. Milestone 2 trains a residual on top of that prior from procedurally shifted robot initial joints. This needs resets that reload a state without rebuilding the simulator (soft reset) and a controller whose nullspace reference follows the loaded joints.

Wrapper changes (`env/gym_utils/wrapper/libero_image.py`, factory in `env/gym_utils/__init__.py`):

- `make_libero_env(..., hard_reset=True)` forwards `hard_reset` to `OffScreenRenderEnv`. The default keeps pretraining and prior evaluation unchanged. The factory pops `hard_reset` from the `libero_image` block and builds the env with it.
- `LiberoImageWrapper.load_state(state)` calls `env.set_init_state(state)` and then `controller.update_initial_joints(state[1:8])` when those joints differ from `init_states[0][1:8]` by more than 1e-9. `env.reset()` rebuilds the controller, so its reference is canonical after every reset.

## Gate A: soft reset and controller reference

Script `script/dicerl_libero_gate_a.py`, output `$DICE_RL_DATA_DIR/libero/<task>/gate_a.json`. Run on Jarvis (GB10), 2026-10-22. Exit status 1: the replay and drift checks failed (details below).

| check | result |
|---|---|
| hard reset latency (`reset` + `set_init_state`, mean of 10) | 0.608 s |
| soft reset latency | 0.0019 s (312x faster) |
| async rate, 8 envs, chunk 8, 163 chunks (about 20 episodes), soft | 1189 env steps/s |
| demo_0 replay, hard, two passes | pass 1 success (step 320), pass 2 failure |
| demo_0 replay, soft, two passes | pass 1 failure, pass 2 success (step 325) |
| demo_0 replay, ten passes each (ad hoc, not in the script) | hard 5/10, soft 9/10 |
| drift: 200 soft resets to state 0, 20 random steps between, max proprio spread | 3.04e-3 (gripper dims 6 and 7; every other dim at most 7.7e-6) |
| drift, hard resets, same protocol (ad hoc) | 0.0 |

Equivalence of the first wrapper observation after `reset(options={"init_state": s})` against a fresh hard-reset build (max abs proprioception difference; mean abs image difference over 255 for agentview and wrist):

| state | hard after 20 random episodes | soft fresh | soft after 20 random episodes |
|---|---|---|---|
| 0 | 0.0; 0.012, 0.027 | 0.0; 0.012, 0.020 | 1.80e-4; 0.0096, 0.027 |
| 17 | 0.0; 0.018, 0.018 | 0.0; 0.015, 0.017 | 2.13e-4; 0.0099, 0.015 |
| 42 | 0.0; 0.011, 0.023 | 0.0; 0.0045, 0.018 | 2.13e-4; 0.0025, 0.011 |

Image differences of 0.003 to 0.03 are the same order for hard-after-random, so they are render jitter that exists with hard resets too (gate 0 reset jitter), not a soft-reset effect.

Controller reference (shifted joints are state 0 plus 0.3 rad with alternating signs, 50 NOOP steps, max joint deviation from the loaded joints):

| load | max drift (rad) |
|---|---|
| canonical state 0 | 0.042 |
| shifted, `set_init_state` only | 0.277 |
| shifted, with `update_initial_joints` | 5.5e-9 |
| shifted, through `wrapper.reset` | 5.5e-9 |

Findings:

- The soft-reset gripper drift is the Panda gripper controller's `current_action` integrator, which survives `env.reset()` in soft mode (hard mode rebuilds it). Resetting it to its fresh value after `env.reset()` removed the drift in an ad hoc check (spread 0.0), but this is not applied in the wrapper.
- Demo replay is not deterministic in either mode, so the two-pass replay check is not a reliable pass or fail signal.

## Gate B

## Gate C

## Config

## Launch

## Sweep

## Final results
