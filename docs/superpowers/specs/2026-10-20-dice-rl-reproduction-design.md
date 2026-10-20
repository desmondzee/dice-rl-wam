# DICE-RL reproduction on Robomimic: design

Status: draft for review, 2026-10-20. Decision (user, 2026-10-20): pivot from the LingBot-VA residual recipes to the authors' own codebase and benchmarks; reproduce the paper's Square and Tool Hang results first, LIBERO-10 / LIBERO-Plus in a later spec, fast-WAM after that. Basis: the 2026-10-20 investigation (`docs/investigations/`) showed every recipe of the last month rested on a critic never checked against a known result.

## Goal

A verified, unmodified DICE-RL setup on Jarvis that reproduces the authors' released numbers for two state-based Robomimic tasks, so that later changes (LIBERO environments, a different prior) are measured against a known-correct baseline.

## Facts the design rests on

- The authors' code (DPPO-based) is vendored unmodified at this repo's top level (`agent/`, `model/`, `cfg/`, `env/`, `util/`, `script/run.py`, `script/eval_rl_checkpoint.py`, `script/download_hf.sh`); AGENTS.md forbids modifying it and this spec keeps that rule.
- Released on Hugging Face (`wintermelontree/robomimic-{pretrain,finetune}-{data,checkpoints}`): processed datasets (`{env}_low_dim/{train.npz, val.npz, normalization.npz}`), pretrained flow-matching BC checkpoints with their Hydra configs and run logs, fine-tuned residual checkpoints with their Hydra configs and 300-episode evaluation logs.
- The released fine-tune configs differ from the vendored `cfg/robomimic/finetune/*/ft_distill_residual_flow_mlp.yaml`: released `use_soft_q_filtering: true` (threshold −0.5 Square, −0.55 Tool Hang) and `expert_ratio: 0.3`; vendored has the filter off. The released config is the ground truth.
- Released fine-tune settings (both tasks): flow steps 10, 200 steps × 1,000 iterations, batch 256 (critic batch 2,048), γ 0.99, τ 0.01, 10 gradient steps per update, actor every 2nd, lr 1e-4 both, weight decay 1e-6, β (`bc_loss_weight`) 100, Q normalisation on, 16 exploration samples with `max_q_min`, 10-critic ensemble, residual not conditioned on the base action, flow warm-up 4,000 steps, eval every 1,000 iterations on 30 envs, seed 42. Square: horizon 4, act 4, n-step 3, 400-step episodes. Tool Hang: horizon 8, act 8, n-step 5, 700-step episodes. Pretraining: 8,000 epochs, batch 256, the same horizons.
- Released evaluation logs (300 held-out episodes, `success_rate` column; the `step` column is the log's own unit):

| task | pretrained flow BC | fine-tuned at matched steps |
|---|---|---|
| Square | 0.583 | 0.68 @32k, 0.69 @64k, 0.887 @160k, 0.937 @240k, 0.98 @448k |
| Tool Hang | 0.497 | 0.34 @64k, 0.553 @96k, 0.71 @160k, 0.797 @448k, 0.937 @1.09M (noisy: 0.47 @192k) |

- The paper (arXiv 2603.10263) averages five seeds on 300 held-out configurations and states Tool Hang passes 90% within about 2,000 online episodes; per-task numbers are in its figures only. The released single-seed logs above are the quantitative targets.
- Compute: Jarvis (GB10, aarch64, 20 cores, 121 GB). State tasks run the simulator on CPU and small MLPs on the GPU. The box already runs robosuite 1.4.0 with mujoco 3.8.1 (LeRobot environment) and torch 2.11 + CUDA 13; the reference pins Python 3.8 and torch 2.4, which do not support the GB10. Watson is in use by the user and is not part of this work.

## Environment (deviations recorded, never silent)

- A separate environment `~/Documents/dice/dicerl-venv` built with `uv`: the newest Python the reference's dependencies accept (3.12 if `gym==0.22.0`, `robomimic` v0.5.0 and `robosuite` v1.4.1 install there, else the newest that does, found by trying, not guessing), torch from the box's working CUDA 13 build, the reference's other pins unchanged, the `robomimic` extra only (no `gym`/`d4rl`, which need x86 mujoco-py). Every deviation from the reference's pins is listed in the reproduction record.
- Data and logs under `~/Documents/dice/dicerl/{data_dir,log_dir}` in the layout `download_hf.sh` expects (`data_dir/robomimic/{env}-low-dim/{ph_pretrain,ph_finetune}`, `log_dir/robomimic-{pretrain,finetune}/...`); `DICE_RL_DATA_DIR`, `DICE_RL_LOG_DIR`, `DICE_RL_WANDB_ENTITY` set in a box-local env file, never printed.
- Launch and download helpers live in `brev/dicerl/` in this repo (shell + minimal Python, no narrative comments); the reference code and its configs are not edited. Run-specific configs are copies of the released `.hydra/config.yaml` with only the dataset, normalisation, checkpoint and log paths changed, stored in `brev/dicerl/cfg/`.
- Long jobs run in box-local tmux with logs under `log_dir`; W&B project `dice-rl-repro`.

## Gates (each must pass before the next starts)

1. Environment gate. With the released checkpoints: run `script/eval_rl_checkpoint.py` on `square_low_dim` and `tool_hang_low_dim` (fine-tuned and pretrained), 300 episodes, 30 envs. Pass: each of the four success rates within 6 points of the released log's value (binomial 95% interval at n = 300 is ±3.4 points at 0.9 and ±5.7 at 0.5). Also records episodes per minute per task for budgeting. Fail: fix the environment, not the code; if the evaluator itself is at fault, stop and report.
2. Fine-tuning gate. Fine-tune from the released pretrained checkpoint with the released config, seed 42, evaluating on 300 episodes at the log's checkpoints. Pass: Square within 6 points of the log at 64k, 160k and 240k steps; Tool Hang within 8 points at 160k and 448k (its log is noisier) and ≥ 0.85 by 1.09M if the budget allows. Then seeds 43 and 44 to the shorter of the two budgets, reporting mean ± spread, since the paper averages five seeds.
3. Pretraining gate. Pretrain Square and Tool Hang flow policies with the released pretrain configs (8,000 epochs). Pass: 300-episode success within 6 points of 0.583 / 0.497. Then one fine-tune from our own Square checkpoint to the 240k mark, within 6 points of the log, which verifies the whole pipeline end to end.

Budgets are measured, not assumed: the environment gate's episodes-per-minute figure sets how long each fine-tune takes before it is launched; anything projected past 48 h per run is reported before launch.

## Record

`docs/dice-rl-reproduction.md`: environment versions and deviations, per-gate commands, numbers against the released logs, wall-clock per task, W&B ids; one line per outcome in `docs/probe-outcomes.md`.

## Out of scope

Image-based tasks, Can and Transport, LIBERO, any change to the reference code or recipe, fast-WAM. Transport's released log shows no gain through 160k steps; if Square and Tool Hang pass, Transport is a candidate third check, decided then.
