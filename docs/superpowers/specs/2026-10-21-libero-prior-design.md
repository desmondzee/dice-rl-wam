# LIBERO-10 flow prior in the DICE-RL codebase: design

Date: 2026-10-21. Milestone 1 of the out-of-distribution residual programme. Predecessor: `docs/superpowers/specs/2026-10-20-dice-rl-reproduction-design.md` (gate 1 passed on Tool Hang, `docs/dice-rl-reproduction.md`). Research grounding: `docs/research/2026-10-21-dice-rl-paper-dive.md`, `docs/research/2026-10-21-residuals-under-distribution-shift.md`, `docs/research/2026-10-21-image-conditioned-residual-and-critic.md`.

## Intent

Bring LIBERO-10 into the DICE-RL codebase and pretrain an image-conditioned flow-matching prior on one LIBERO-10 task, selecting a checkpoint at about 50% success so a residual has room to improve it. Everything after this milestone (the residual on procedurally shifted poses and layouts, the held-out-shift evaluation, the visually randomised prior) reuses the environment wrapper, the dataset format and the evaluation protocol fixed here, so those are the deliverables that must be right.

The programme's thesis, agreed in chat on 2026-10-21: a residual trained with RL on a wide distribution of shifts over a frozen prior should give out-of-distribution robustness more cheaply than post-training a large model. The literature leaves this untested (second research report), so the programme measures it. This milestone builds the prior that the measurement needs.

## Goals

- A LIBERO environment usable through the DICE-RL `make_async` factory and wrapper chain, with pixels, proprioception, seeded resets to stored or supplied initial states, and the same success semantics as the robomimic path.
- The chosen task's 50 demonstrations in the DICE-RL pretraining format, with normalisation statistics.
- A flow-matching prior trained with the released image pretrain recipe, checkpoints evaluated on the held-out protocol, and one checkpoint selected in the 40–60% band.
- Measured throughput and environment correctness before anything long runs.

## Non-goals and anti-goals

- No residual training in this milestone, no procedural shifts, no LIBERO-Plus. Those are the next milestone and get their own spec.
- No visual randomisation of the demonstrations yet. The plain prior is the in-distribution baseline the later randomised prior is compared against.
- No multitask prior. One task.
- No changes to the vendored DICE-RL algorithm code (`agent/`, `model/`, `util/`, `script/run.py`, `script/eval_rl_checkpoint.py`, `cfg/robomimic/`). New code goes in new files; the only edits to existing files are registrations (the wrapper dictionary and, if needed, one `env_type` branch in `env/gym_utils/__init__.py`).
- No edits to `.cache/lerobot`, `script/lingbot_*`, the SFT environment, the Tool Hang reproduction environment or its run directories.
- Anti-goal: a prior whose evaluation numbers are not comparable to the LingBot baselines. The held-out protocol below is the one already used for those.

## Task

`KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` from LIBERO-10. The demonstration file is already on Jarvis at `~/Documents/dice/libero-demos/libero_10/<task>_demo.hdf5` (50 demos, lengths 224–449 steps, median 301), its stored MuJoCo states were validated last week to replay the demonstrations exactly, and the procedural-pose code was validated on it. Its bddl and 50 stored initial states ship inside the `hf-libero` package.

Demo file layout (verified 2026-10-21): per demo `obs/agentview_rgb` and `obs/eye_in_hand_rgb` (T, 128, 128, 3) uint8, `obs/ee_pos` (T, 3), `obs/ee_ori` (T, 3) axis-angle, `obs/gripper_states` (T, 2), `obs/joint_states` (T, 7), `actions` (T, 7) in [−1, 1] (OSC_POSE deltas and gripper, 20 Hz), `states` (T, 47) flattened MuJoCo state, `rewards`, `dones`.

## Environment

### Software environment

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

`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` as in `brev/dicerl/env.sh`; the LIBERO config file must point at the package's own bddl, init and asset folders (the LeRobot scripts already do this with `get_libero_path`; copy that convention, not the file).

The reproduction environment `dicerl-venv311` is left untouched. Platform risk is low: the LeRobot environment at `~/Documents/dice/lerobot/.venv` on the same aarch64 box already runs hf-libero 0.1.4, robosuite 1.4.0 and mujoco 3.8.1 under EGL (every LIBERO and LIBERO-Plus result in `result/heldout/` and the LingBot RL runs came from it); the new environment changes only the Python minor version and the training-side packages.

### Gate 0: environment correctness and throughput (before any training)

1. `OffScreenRenderEnv` for the task builds under EGL in the new environment, renders both cameras at 128×128, and `set_init_state` with the hdf5 `states[0]` of demo 0 followed by replaying that demo's actions reaches success. This repeats last week's validated replay inside the new stack.
2. The rendered agentview and wrist images, after the wrapper's flip, match the demo file's images for the same state to within the known asset-level difference (mean absolute pixel difference below 2/255 on the same frame). This fixes the orientation convention once; the LeRobot path needed a vertical-only flip and the wrapper must apply whatever makes this check pass.
3. Throughput: environment steps per second with 1, 4 and 20 parallel environments through `make_async` with the wrapper chain, images on, random actions, 200 steps each; and the same with the pretrained-prior evaluation loop once a checkpoint exists. These numbers set the evaluation and pretraining budgets. Anything projecting a single checkpoint evaluation past 30 minutes or pretraining past 24 hours is reported before launch.

## Wrapper: `env/gym_utils/wrapper/libero_image.py`

Registered as `libero_image` in `env/gym_utils/wrapper/__init__.py`. Constructed by `make_async` through a new `env_type == "libero"` branch that builds `OffScreenRenderEnv(bddl_file_name=<resolved from task name>, camera_heights=128, camera_widths=128, camera_names=["agentview", "robot0_eye_in_hand"])` and then applies the configured wrappers exactly as the robomimic branch does, so `multi_step` and `multi_step_full` work unchanged on top.

Behaviour, mirroring `RobomimicImageWrapper`:

- Observation `{"rgb": uint8 (128, 128, 6) with agentview then wrist stacked on the last axis, exactly as the robomimic wrapper emits (the pretrain rollout and evaluation code permute to channels-first themselves; only the npz on disk is channels-first), "state": float (8,)}`. State is `robot0_eef_pos` (3), `robot0_eef_quat` converted to axis-angle (3) with the same conversion LIBERO used to produce the demo `ee_ori`, and `robot0_gripper_qpos` (2), in that order, min–max normalised to [−1, 1] with `normalization.npz` exactly as the robomimic wrapper does. Images are oriented to match the demonstration file (gate 0, check 2).
- Actions arrive normalised in [−1, 1] and are un-normalised with the same statistics, so the policy's action space is the demonstrations' range per dimension, as on robomimic.
- Reward: 1.0 on the LIBERO success predicate, else 0.0. Termination once `success_steps_before_termination` success steps have accumulated (default 5, cumulative, mirroring the robomimic wrapper's counter) or at `max_episode_steps` = 520, the horizon used by every LIBERO baseline here. Episode success for every evaluation in this programme means at least one success step (reward sum ≥ 1), which is also how the LingBot held-out results were counted.
- `reset(options)`: `options["init_state"]` sets a supplied MuJoCo state; otherwise `options["seed"]` selects initial state `seed % 50` from the task's stored init file (the 50 canonical LIBERO states) and seeds numpy; with neither, a random stored state. The `seed` method seeds numpy only, as the robomimic wrapper does. After setting the state, 10 steps with the LeRobot no-op action `[0, 0, 0, 0, 0, 0, −1]` (gripper open) settle the scene before the first observation, matching the LeRobot protocol's `num_steps_wait=10`. Because `multi_step` auto-resets without options, evaluation never relies on auto-reset: every episode starts from an explicit `reset_arg` with its initial state.
- `get_sim_state` is exposed for later procedural-pose work.
- Video recording through `options["video_path"]` as in the robomimic wrapper, from the agentview camera, so evaluation videos work with the existing `render` config.

## Dataset: `script/dataset/process_libero_dataset.py`

Converts the hdf5 to the DICE-RL pretraining npz with the keys the robomimic converter writes: `states` (N, 8) proprioception in the wrapper's order, `actions` (N, 7), both stored min–max normalised to [−1, 1] as the robomimic `--normalize` path does, `images` (N, 6, 128, 128) uint8 channels-first in the wrapper's orientation and camera order, `rewards` (N,), `terminals` (N,), `traj_lengths` (50,), and `normalization.npz` with `obs_min`, `obs_max`, `action_min`, `action_max` as the true per-dimension minima and maxima over all 50 demos (not the robomimic converter's zero-initialised running extremes, which would force every range to straddle zero). No validation split (`val_split 0`, all 50 demos train, as the released robomimic pretraining does). Output under `~/Documents/dice/dicerl/data_dir/libero/<task>/ph_pretrain/`. The converter reads the demo file's own images rather than re-rendering, so gate 0 check 2 is what guarantees the environment matches them.

## Pretraining

Config `brev/dicerl/cfg/libero/<task>/pre_flow_matching_unet_img.yaml`, copied from `cfg/robomimic/pretrain/tool_hang/pre_flow_matching_unet_img.yaml` with only these changes:

- paths: dataset, normalisation, log directory under `DICE_RL_LOG_DIR/libero-pretrain/`, W&B project `dice-rl-libero`, entity from the environment as in the reproduction configs;
- `env_type: libero`, task name, wrapper `libero_image` with the two cameras, `max_episode_steps: 520`;
- `obs_dim: 8`, `model.network.cond_dim: 8`, `shape_meta.obs.rgb.shape: [128, 128, 6]`, `shape_meta.obs.state.shape: [8]`, network `img_height: 128`, `img_width: 128`, and the random-crop size scaled as Square/Can and Tool Hang are (84 of 96, 216 of 240): 116 of 128;
- `env.env_type: libero` (read as `cfg.env.env_type`), `robomimic_env_cfg_path` removed, and the libero branch of `make_async` placed before the robomimic and d4rl branches so neither is imported;
- `env.n_envs: 10` and `train.render.num: 1` for the in-training rollouts the pretrain agent runs every 100 epochs (`val_freq` is hard-coded to 100 in `agent/pretrain/`, so they cannot be disabled without editing vendored code); they are monitoring only and do not affect training;
- `horizon_steps: 8`, `act_steps: 8`, `flow_steps: 10`: the paper's longer-horizon setting; 10 flow steps is the Square/Can value and halves inference cost against Tool Hang's 20 (builder may not change these);
- `n_epochs: 8000`, `save_model_freq: 200`, batch 64, learning rate 1e-4, EMA as released.

Everything else, including the ResNet-18-per-camera architecture, spatial softmax, the U-Net flow head and the optimiser, stays as released. Launcher `brev/dicerl/pretrain_libero.sh <task> [seed]` in a box-local tmux session, log under the log directory, seed 42.

Before the full run: a 100-epoch smoke run (long enough to include one in-training rollout) must complete, and its epoch and rollout times project the 8,000-epoch cost (gate 0 item 3).

## Evaluation of the prior

`script/dicerl_libero_eval.py` (new) loads a pretraining checkpoint (`state_<epoch>.pt`, keys `epoch`, `model`, `ema`) with the run's `.hydra/config.yaml`, instantiates `cfg.model` and loads the raw `model` weights, not `ema`, because the residual stage loads `model`, so the 50% selected is the prior the residual sits on. It runs the held-out protocol: initial states 0–49 of the task's stored init file passed explicitly per episode through `reset_arg`, each with torch seeds 0 and 1 for the policy noise, 100 episodes, 520 steps maximum, standard flow sampling (one noise draw per chunk, no best-of-K), 8 actions executed per chunk. It records per-episode success, length and the initial-state id, writes `episodes.csv` and `summary.json` next to the checkpoint, and reports the success rate with its binomial standard error. This is the same initial-state and seed structure as `result/heldout/` for the LingBot baselines, so the prior's number sits next to those.

Evaluation runs during pretraining, not after it: `brev/dicerl/pretrain_libero.sh` starts a companion watcher in the same tmux session that, each time a checkpoint at a multiple of 1,000 epochs appears, evaluates it with the protocol above while training continues (the evaluation is 100 episodes and takes minutes; the brief GPU sharing is accepted), appends the row to `docs/libero-prior.md`'s table via a results file under the run directory, and stops the training session as soon as a checkpoint lands in the 40–60% band. The selected checkpoint is the in-band checkpoint closest to 50% among those evaluated, earlier epoch on ties (the paper's Fig. 4 shows intermediate checkpoints fine-tune best). If training reaches 8,000 epochs with every checkpoint above 60%, the fallback is to retrain on 20 demonstrations (the paper's Can setting), and the user is told before that run starts; if every checkpoint is below 40%, the user is told and the run is left to finish. The selected checkpoint is recorded with its epoch, success rate and standard error in `docs/libero-prior.md`, along with gate 0 numbers, throughput and the pretraining wall time.

## Builder decisions

Left to the builder: the resolution of the task name to the bddl and init files (use the package's own layout), the exact LIBERO observation keys for proprioception and how the axis-angle conversion is obtained (must match the demo file to 1e-3 on a sampled frame), the internals of the smoke run, how the eval script reuses the observation handling and per-episode seeding patterns of `script/eval_rl_checkpoint.py` (copy, do not edit; its checkpoint loading is for fine-tuned residuals and does not apply), and test coverage (`tests/test_dicerl_libero.py`: wrapper observation shapes and normalisation round-trip on a fake environment, converter output shapes and normalisation statistics on a synthetic hdf5, evaluation bookkeeping on a mocked policy; no simulator, no GPU, no cloud in tests).

Code rules: no comments or docstrings in new code, lean files, commit at each task without attribution trailers, nothing printed that could carry a secret.
