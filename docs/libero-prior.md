# LIBERO prior: DICE-RL on LIBERO-10

## Task and data

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10), 128x128 agentview and wrist cameras, 20 Hz OSC_POSE, horizon 520.

- Demo file on Jarvis: `~/Documents/dice/libero-demos/libero_10/<task>_demo.hdf5`, 50 demos, 15,232 steps, lengths 224 to 449 (mean 304.6).
- Converted data: `$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain/` (`train.npz` 555 MB, `normalization.npz`), produced by `script/dataset/process_libero_dataset.py`, converted with `--flip vertical` (the default).
- Gate 0 output: `$DICE_RL_DATA_DIR/libero/<task>/gate0.json`.

## Software environment

A new uv-managed virtual environment at `~/Documents/dice/dicerl-libero-venv` on Jarvis, Python 3.11, built by `brev/dicerl/setup_libero.sh` from the reproduction pins plus the LIBERO stack:

| component | version | basis |
|---|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | the GB10 build used by the reproduction |
| hf-libero | 0.1.4 | the LIBERO package the LeRobot environment and all existing baselines use |
| robosuite | 1.4.0 | hf-libero's pin |
| mujoco | 3.8.1 | hf-libero requires ≥ 3.0; 3.8.1 is what every LIBERO baseline here ran on |
| numpy | 1.26.4 | DICE-RL's gym 0.22 is not numpy-2 clean; hf-libero leaves numpy unpinned |
| gym | 0.22.0 | DICE-RL's vector env |
| gymnasium | ≥ 0.29 | hf-libero's dependency; coexists with gym |
| robomimic | 0.2.0 | hf-libero's pin; DICE-RL imports robomimic only inside its robomimic branch, which this milestone never takes |
| hydra-core, einops, wandb 0.27.2, h5py, imageio, matplotlib, others | as in `brev/dicerl/setup.sh` | — |

`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` as in `brev/dicerl/env.sh`.

## Gate 0 results

Run on Jarvis (GB10, 20 cores, no other GPU jobs) with `python -m script.dicerl_libero_gate0`.

### Demo replay

`set_init_state(states[0])` of demo_0, then the 329 raw demo actions through the raw `OffScreenRenderEnv` (no wrapper, no normalisation): success reached, first at step 322 in the final run and step 326 in an earlier run (the scene fixtures are re-sampled at each `reset()`, so timing varies by a few steps). Success is still true after the last action.

### Image orientation

The demo file stores the raw MuJoCo orientation (agentview and wrist both upside down). Mean absolute pixel difference between the rendered and the stored frame 0, as a fraction of 255, max over the two cameras (8-reset mean per camera in brackets):

| flip | score | agentview | wrist |
|---|---|---|---|
| none | 0.0236 | 0.0101 (0.0093) | 0.0236 (0.0239) |
| vertical | 0.3700 | 0.3700 (0.3709) | 0.2521 (0.2539) |
| both | 0.4063 | 0.4063 (0.4068) | 0.2780 (0.2764) |

`none` is the unambiguous best match, 10 to 40 times closer than the others. It does not reach the 2/255 = 0.0078 criterion, so `pick_flip` raises and `flip` is null in `gate0.json`. The residual is reset-dependent scene jitter, not orientation: two renders of the same state from different `reset()` calls differ from each other by up to 0.9 and 2.8 percent (agentview, wrist), the same size as the difference to the stored frame, and the difference image shows only edges of the microwave and robot. The wrapper currently applies `vertical` and the converted data was written with `vertical`, so wrapper and data agree with each other but not with the demo file's own orientation.

### State match

The wrapper state (`normalization_path=None`) against `obs/ee_pos[0]`, `obs/ee_ori[0]`, `obs/gripper_states[0]`, maximum absolute difference per block:

| comparison point | ee_pos | ee_ori (axis-angle) | gripper |
|---|---|---|---|
| after `set_init_state`, before settling | 0.0033 | 0.0142 (equivalent representation 0.0141) | 0.0022 |
| after 10 settle steps | 0.0028 | 0.0568 | 0.0026 |

The pre-settle state matches the first demo observation; settling moves the orientation by about 0.05 rad. The 0.014 rad orientation offset is a near-constant bias along the first demo steps (0.013 at steps 1 to 3), small compared with the demo ranges but above the 1e-2 bar.

### Throughput

Random uniform chunks of 8 actions through `make_async` with `libero_image` and `multi_step` (n_obs_steps 1, n_action_steps 8, reset_within_step), 200 chunks each, images on:

| envs | env steps per second |
|---|---|
| 1 | 193.8 |
| 4 | 654.0 |
| 20 | 1638.3 |

At 20 envs one 8-step chunk across all envs takes 0.098 s. A 100-episode evaluation is five waves of 20 episodes of up to 65 chunks (520 steps): 5 x 65 x 0.098 s = about 32 s of simulation, excluding policy inference, resets and video.

### Findings that affect later tasks

- `libero_image` needs a real `shape_meta` (obs `rgb` [128, 128, 6], `state` [8]); with `None`, `multi_step` fails on the missing `observation_space`. Pass the same `shape_meta` to `make_async`.
- robosuite terminates an episode at its own horizon of 1000 steps and raises on the next step; `reset()` does not clear that because the wrapper only calls `env.reset()` once. `make_libero_env` now passes `ignore_done=True`; the wrapper enforces 520 steps itself.
- Rendering in the main process before forking the async workers makes a worker spin at 100 percent CPU forever. Create the vector env before any raw env in the same process, or run raw envs in a separate process.

## Pretraining

## Selection
