# LIBERO prior: DICE-RL on LIBERO-10

## Task and data

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (LIBERO-10), 128x128 agentview and wrist cameras, 20 Hz OSC_POSE, horizon 520.

- Demo file on Jarvis: `~/Documents/dice/libero-demos/libero_10/<task>_demo.hdf5`, 50 demos, 15,232 steps, lengths 224 to 449 (mean 304.6).
- Converted data: `$DICE_RL_DATA_DIR/libero/<task>/ph_pretrain/` (`train.npz` 555 MB, `normalization.npz`), produced by `script/dataset/process_libero_dataset.py`, converted with `--flip none` so the images keep the demo file's own orientation.
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

`set_init_state(states[0])` of demo_0, then the 329 raw demo actions through the raw `OffScreenRenderEnv` (no wrapper, no normalisation): success reached, first at step 324 in the final run (301, 322 and 326 in earlier runs; the replay timing varies between resets). Success is still true after the last action.

### Image orientation

The demo file stores the raw MuJoCo orientation (agentview and wrist both upside down). Mean absolute pixel difference between the rendered and the stored frame 0, as a fraction of 255, max over the two cameras (8-reset mean per camera in brackets):

| flip | score | agentview | wrist |
|---|---|---|---|
| none | 0.0204 | 0.0080 (0.0099) | 0.0204 (0.0280) |
| vertical | 0.3705 | 0.3705 (0.3706) | 0.2522 (0.2534) |
| both | 0.4066 | 0.4066 (0.4067) | 0.2772 (0.2766) |

Criterion: the best flip's score must be below 0.05 and at most half of the next-best score, otherwise `pick_flip` raises with the three scores. This replaces the design's 2/255 = 0.0078 bound, which this scene does not meet on the wrist camera (best of 8 resets 0.0197). The bound is too tight because renders of the same state already differ between resets. `gate0.json` records this as `reset_jitter`, the mean absolute pixel difference as a fraction of 255 between two renders of the same `set_init_state(states[0])` after separate `reset()` calls:

| measurement | agentview | wrist |
|---|---|---|
| render before the replay vs render after it | 0.0102 | 0.0183 |
| consecutive pairs of the 8 sweep renders, mean | 0.0099 | 0.0272 |
| consecutive pairs of the 8 sweep renders, max | 0.0152 | 0.0407 |

These are as large as the difference between a render and the stored frame (`none` row above), so a render-versus-stored difference of that size cannot be told apart from reset-to-reset variation. The cause of the variation was not isolated. `none` passes (0.0204 < 0.05, and about 18 times better than the next flip); the chosen flip is `none`.

The wrapper's `FLIP` is `none` and the data on Jarvis was reconverted with `--flip none` (frame 0 of `train.npz` equals the demo file's frame 0 exactly), so the wrapper, the data and the demo file agree.

### State match

The wrapper state (`normalization_path=None`) against `obs/ee_pos[0]`, `obs/ee_ori[0]`, `obs/gripper_states[0]`, maximum absolute difference per block:

| comparison point | ee_pos | ee_ori (axis-angle) | gripper |
|---|---|---|---|
| after `set_init_state`, before settling | 0.0033 | 0.0142 (equivalent representation 0.0141) | 0.0022 |
| after 10 settle steps | 0.0028 | 0.0568 | 0.0026 |

The pre-settle state matches the first demo observation; settling moves the orientation by about 0.05 rad. The 0.014 rad orientation offset is a near-constant bias along the first demo steps (0.013 at steps 1 to 3), accepted as reset jitter on the same axis-angle convention.

### Throughput

Random uniform chunks of 8 actions through `make_async` with `libero_image` and `multi_step` (n_obs_steps 1, n_action_steps 8, reset_within_step), 200 chunks each, images on:

| envs | env steps per second |
|---|---|
| 1 | 190.2 |
| 4 | 676.6 |
| 20 | 1612.6 |

At 20 envs one 8-step chunk across all envs takes 0.099 s. A 100-episode evaluation is five waves of 20 episodes of up to 65 chunks (520 steps): 5 x 65 x 0.099 s = about 32 s of simulation, excluding policy inference, resets and video.

### Findings that affect later tasks

- `libero_image` needs a real `shape_meta` (obs `rgb` [128, 128, 6], `state` [8]); with `None`, `multi_step` fails on the missing `observation_space`. Pass the same `shape_meta` to `make_async`.
- robosuite terminates an episode at its own horizon of 1000 steps and raises on the next step; `reset()` does clear it; the wrapper calls `env.reset()` before every `set_init_state`. `make_libero_env` now passes `ignore_done=True`; the wrapper enforces 520 steps itself.
- Rendering in the main process before forking the async workers makes a worker spin at 100 percent CPU forever. Create the vector env before any raw env in the same process, or run raw envs in a separate process.

## Pretraining

Config `brev/dicerl/cfg/libero/<task>/pre_flow_matching_unet_img.yaml` (Tool Hang recipe with 128x128 two-camera images, random crop 116, 8-d state, 10 flow steps, 10 envs, 520-step horizon). `brev/dicerl/pretrain_libero.sh <task> [seed] [--epochs N]` starts the training tmux session `dicerl-pre-<task-short>-<seed>` and the watcher session `dicerl-watch-<seed>`, which evaluates every 1,000th epoch checkpoint with 25 envs and stops training at the first success rate in [0.40, 0.60].

### Smoke run (100 epochs, seed 42, Jarvis GB10)

- 15,232 training samples (50 trajectories), batch 64, 238 updates per epoch: 21.20 s per epoch (mean of epochs 1 to 99; min 21.07, max 21.78). 8,000 epochs projects to 47.1 h, above the 24 h budget; the in-training rollouts add 80 x 12 s = 0.3 h.
- In-training rollout at epoch 100 (10 envs, 65 chunks): 11.8 s. Each checkpoint is 199 MB.
- Evaluating `state_100.pt` with `script.dicerl_libero_eval --envs 25` (100 episodes, 50 states x 2 seeds) after the reset fix: 144 s wall including model load. `summary.json`: success_rate 0.74, stderr 0.044, n 100, mean_length 358.2. Successes by batch of 25: seed 0 22 and 16, seed 1 18 and 18.
- The in-training rollout reported `Done envs: 1/10` and `success rate 0.9000` because its metric is the best summed reward over the rollout reaching 1 (any step with `check_success`), and 9 of 10 envs had succeeded, terminated and auto-reset before the last chunk, so only one was done at the end. It is not an inverted metric.

### Reset defect found and fixed by the smoke evaluation

The first evaluation gave 0.22 in 76 s. Per episode, the first reset of each worker succeeded 18 of 25, the second reset 0 of 25, and later resets (seed 1) 1 of 50. The wrapper called `env.reset()` only on a worker's first reset and then `set_init_state`; probing 4 workers on states 0 to 3 repeated four times gave `[1 1 1 1]` then `[0 0 0 0]` three times, while calling `env.reset()` before every `set_init_state` gave `[1 1 1 1]`, `[1 1 0 1]`, `[1 1 0 1]`, `[1 1 0 1]`. The wrapper now resets the raw env on every reset. Cost: the same evaluation takes 144 s instead of 76 s although episodes are shorter (mean length 358 instead of 471), about 17 s or more per batch of 25 parallel resets.

### Findings that affect later tasks

- `import huggingface_hub` in the `libero` branch of `make_async` is required: with `pass`, the 10 env workers print the OpenGL init line and then hang for more than 200 s (reproduced twice with `train.n_epochs=1`); with the import the same command trains. libero imports `huggingface_hub` from its `download_utils` inside each forked worker; the cause of the hang was not isolated further. The evaluator alone (no wandb, no training) works either way.
- `script/run.py` imports `pretty_errors` and `gdown`, so `setup_libero.sh` pins both.


## Selection
