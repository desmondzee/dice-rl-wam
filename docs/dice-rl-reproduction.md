# DICE-RL reproduction record

Spec: `docs/superpowers/specs/2026-10-20-dice-rl-reproduction-design.md`. Box: Jarvis (GB10, aarch64). Helpers: `brev/dicerl/` (setup, download, eval, finetune); run configs `brev/dicerl/cfg/<env>/ft_distill_residual_flow_mlp.yaml` are the released `.hydra/config.yaml` files with only `base_policy_path`, `normalization_path`, the expert `dataset_path` and the W&B entity/project changed. The reference code is unmodified.

## Environment (2026-10-20)

| component | reference pin | used | basis |
|---|---|---|---|
| Python | 3.8 | 3.11 (uv-managed) | 3.8 lacks GB10 torch builds; 3.12 has no wheels for the reference's pandas 2.0.3 and matplotlib 3.7.5 |
| torch | 2.4.0 | 2.11.0+cu128 | the only builds supporting the GB10 (sm_121) |
| mujoco | unpinned (robosuite ≥ 2.3.0) | 2.3.7 | measured: 3.8.1 lowers the released Tool Hang checkpoint from 92.0% to 85.3% on the same 300 seeds (below); 3.16.0 fails robosuite 1.4.1's joint lookup |
| robosuite / robomimic / gym / hydra / others | v1.4.1 / v0.5.0 / 0.22.0 / 1.3.2 | as pinned | — |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | — | set | torch ≥ 2.6 refuses the released checkpoints' OmegaConf objects by default |

Data and checkpoints: `wintermelontree/robomimic-{pretrain,finetune}-{data,checkpoints}` (`square_low_dim`, `tool_hang_low_dim`) under `~/Documents/dice/dicerl/{data_dir,log_dir}`, symlinked as `data_dir/`, `log_dir/` in the repo root as the code expects. The pretrain and fine-tune normalisation files are identical per task. No fine-tuned Square checkpoint is released (its folder holds only the evaluation log).

## Gate 0: environment (released Tool Hang checkpoint `model_step_24000.pth`, 300 episodes, seeds 10000–10299, 30 envs, `max_q_min`)

| policy | authors (released log, 300 ep) | authors' own `eval_rl_checkpoint` run (50 ep) | ours, mujoco 3.8.1 | ours, mujoco 2.3.7 |
|---|---|---|---|---|
| fine-tuned residual | 0.86–0.94 at the log's later rows | 0.94 | 0.853 | 0.920 |
| pretrained flow BC | 0.497 | 0.50 | 0.397 | 0.407 |

Verdict: pass for the fine-tuned policy under mujoco 2.3.7 (within the 50-episode run's ±14 and the log's range); the pretrained policy is 9 points below the released 300-episode value (95% interval ±5.6), 1 point inside the authors' own 50-episode run. Open question carried forward: whether the released log's step-0 "pretrained" row was produced by the same evaluation path as `eval_rl_checkpoint.py` (the paper's held-out configurations are seeds 10000+). Throughput: 300 Tool Hang episodes in about 50 s with 30 envs.

## Configs: released versus public (2026-10-20)

The public repository (`zhanyisun/dice-rl`, 22 commits, head 98d936e of 2026-08-04) is byte-identical to the snapshot vendored here. The released checkpoints' `.hydra/config.yaml` files come from a later internal version: they carry about 70 keys the public code does not accept (`max_replay_diversity.*`, intrinsic reward, dynamics model, transformer actor, `discount_within_horizon`, `use_upsampling`, `sample_multi_z_for_actor_loss`, ...), all set to off or zero except the last two, whose public-code equivalents cannot be confirmed. Running a released config on the public code fails on the first unknown key. Among keys the public code supports, the released training differs from the public config in three values: `use_soft_q_filtering: true` (public false), `q_underestimation_threshold` −0.55 for Tool Hang (public −0.5; Square −0.5 in both), `always_retain_bc_loss_for_expert_data: false` (public true). Decision: the run configs are the public config files with only the paths, the W&B entity/project and those three values changed to the released ones, so the recipe is the one the released checkpoints were trained with as far as the public code can express it (`brev/dicerl/cfg/`).

Other environment notes: `wandb` 0.17.7 (reference pin) cannot initialise against the current W&B backend (`CommError: 'NoneType' object has no attribute 'get'` on a bare `wandb.init`); 0.27.2, the version the box's other environment uses, works, so the setup installs 0.27.2.

## Gate 1 runs (launched 2026-10-20 21:14 BST on Jarvis)

`brev/dicerl/finetune.sh square 42` and `brev/dicerl/finetune.sh tool_hang 42`: public code, released recipe values, released pretrained checkpoints, 4 envs each, 1,000 iterations of 200 steps, evaluation on 30 envs every 1,000 iterations as released. W&B project `dice-rl-repro`.
