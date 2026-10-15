# LIBERO-90 prior for plan-selection RL

Status: design for review. Goal: a prior in the 40–50% band on an accepted benchmark, with every piece of the
RL and evaluation stack below the environment unchanged. Reference: docs/research-benchmarks.md.

## 1. Decisions

| item | decision |
|---|---|
| benchmark | LIBERO-90, five tasks, one per scene, the tasks FASTER (2604.19730) and DSRL (2506.15799) used |
| tasks | 2 KITCHEN_SCENE10 bowl into the top drawer; 29 KITCHEN_SCENE5 bowl into the top drawer; 57 LIVING_ROOM_SCENE3 cream cheese into the tray; 60 LIVING_ROOM_SCENE4 left bowl into the tray; 79 STUDY_SCENE2 book into the left caddy compartment |
| why one task per scene | the SFT prior is language-free (empty text embedding), so the scene must identify the task |
| demonstrations | all 50 per task from the released LIBERO-90 set (250 episodes); the SFT recipe otherwise unchanged (1,000 updates, effective batch 80, lr 1e-5, full transformer) |
| checkpoints | every 100 updates; evaluate 200, 400, 600, 800, 1,000 |
| prior selection | the checkpoint whose mean success over the five tasks at the 5 / 10 sampler sits in 40–50%, measured on shard 0 of 2 of the standard plan (every other episode, 50 per task), then completed to 100 |
| post-training compute | Modal, desmond-zee workspace, H100:8, the existing `lingbot_sft_modal` pipeline; a one-off |
| data build and evaluation compute | the GB10s |
| evaluation protocol | unchanged: 50 init states × 2 seeds per task, seed 42, 520-step horizon, same seeds across policies |

## 2. Data

Source: `yifengzhu-hf/LIBERO-datasets`, `libero_90/<task>_demo.hdf5` for the five tasks (50 demos each; 128 × 128
agentview and eye-in-hand RGB, 7-dim actions, 20 Hz; LIBERO stores frames rotated 180°).

Target: the layout the SFT loader reads today, copied from `robbyant/libero-long-lerobot` at revision 8c0313b1:

- `meta/info.json` (fps 60 as published, chunks_size, features: `observation.state` 8, `action` 7, two video features
  128 × 128 × 3, timestamp, frame_index, episode_index, index, task_index), `meta/episodes.jsonl`, `meta/tasks.jsonl`,
  `data/chunk-000/episode_XXXXXX.parquet` with the same columns.
- `latents/chunk-000/<camera>/episode_XXXXXX_0_<length>.pth` per camera with keys `latent` (tokens, 48) bf16 where
  tokens = latent_num_frames × 8 × 8, `latent_num_frames` = 1 + (video_num_frames − 1) / 4, `latent_height` 8,
  `latent_width` 8, `video_num_frames`, `video_height` 128, `video_width` 128, `text_emb` (512, 4096) bf16,
  `text`, `frame_ids`, `start_frame` 0, `end_frame` = length, `fps` 60, `ori_fps` 60.

Encoder: the frozen Wan VAE from the pinned model, run on a GB10, with the same frame orientation, normalisation and
temporal layout the published latents use. Validation before any training: re-encode one published LIBERO-10 episode
from its own `videos/` and compare with its published latent (expect agreement to bf16 rounding); re-encode the same
episode from the raw LIBERO-10 HDF5 to fix the orientation and frame mapping against the same reference. `text_emb` is
whatever the loader needs to accept the file; training uses the empty text embedding as today.

Output: `~/Documents/dice/sft-data/libero90-5task/` on the Watson store (about 300 MB of latents plus parquet and
videos), uploaded to the Modal SFT cache volume with the verified upload path.

## 3. Code changes

- `script/lingbot_sft_config.py`: `suite` ("libero_10" | "libero_90"), `task_ids`, `demos_per_task` validated against
  the dataset, `dataset_root` as an alternative to the HF repo (fingerprinted by the manifest it builds),
  `weight_steps` every 100; recipe version 2; `sft_config.json` records suite and task ids.
- `script/lingbot_sft_data.py`: `choose_episodes` / `validate_manifest` driven by the config's task ids and demo count
  instead of the 10 × 30 pin; local-root loading.
- `script/lingbot_sft_modal.py`: `prepare` takes the local dataset root from the volume; `train8` unchanged.
- `script/lingbot_eval_config.py`, `script/lingbot_rl_config.py`: `suite` field (default libero_10, in the protocol),
  task ids validated per suite (0–9 or 0–89), `episode_plan` identity carries the suite.
- `script/lingbot_eval.py`, `script/lingbot_rl_train.py`: `describe_suite` and `make_env` take the suite; checkpoint
  provenance check reads suite and task ids from `sft_config.json`.
- `brev/eval.sh`, `brev/train.sh`: `--suite`.
- Tests: fingerprints for both suites; task-id validation; loader on a two-episode synthetic local root; encoder
  round-trip on a published episode (GB10-only, skipped without the data).
- `AGENTS.md`: the LIBERO-90 data build and the Modal post-training command.

Unchanged: the RL recipe, plan selection, the plan critic, the residual, every held-out result on LIBERO-10.

## 4. Sequence

1. Encoder on Watson; round-trip validation against the published LIBERO-10 episode.
2. Build the five-task dataset; upload to the Modal volume; verify.
3. Code changes above with tests; commit.
4. Modal prepare (local root) and train8 for 1,000 updates with checkpoints every 100; W&B as before.
5. Download checkpoints 200–1,000 with the chunked fetch; ship to the Watson store; prepared indices on both boxes.
6. SFT baselines at 5 / 10, 50 episodes per task per checkpoint, split across the boxes (about 4.5 h per checkpoint
   per box); pick the prior; complete its baseline to 100 episodes per task.
7. RL on the chosen prior: `dice-l90-tX-plan-m4` runs, one task per box, the current recipe.

Meanwhile on the GB10s: the plan-critic studies (ceiling, regularised and classification refits, noise pre-screen
data pass).

## 5. Costs and risks

- Modal: prepare on CPU plus about 1–2 h of H100:8, tens of dollars.
- A prior that overshoots at every checkpoint: fall back to fewer demonstrations per task (30, then 10) with one more
  post-training run; undershoot at 1,000: the recipe's budget cap is the limit, extend to 2,000 updates.
- Encoder mismatch with the published latents would silently degrade the prior; the round-trip validation gates
  everything after it.
- Language-free prior on five scenes: no ambiguity by construction; evaluation keeps passing the instruction as today.
