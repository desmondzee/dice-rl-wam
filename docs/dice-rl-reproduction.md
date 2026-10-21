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

The public repository (`zhanyisun/dice-rl`, 22 commits, head 98d936e of 2026-08-04) is byte-identical to the snapshot vendored here. The released checkpoints' `.hydra/config.yaml` files come from a later internal version: they carry about 70 keys the public code does not accept (`max_replay_diversity.*`, intrinsic reward, dynamics model, transformer actor, `discount_within_horizon`, `use_upsampling`, `sample_multi_z_for_actor_loss`, ...), all set to off or zero except the last two, which the public code applies unconditionally: `actor_loss` always samples `num_multi_z_for_actor_loss` noises (16 in both the public and released configs), and the agent always swaps the `multi_step` wrapper for `multi_step_full` (upsampling) when the config has one, as these do. Running a released config on the public code fails on the first unknown key. Among keys the public code supports, the released training differs from the public config in three values: `use_soft_q_filtering: true` (public false), `q_underestimation_threshold` −0.55 for Tool Hang (public −0.5; Square −0.5 in both), `always_retain_bc_loss_for_expert_data: false` (public true). Decision: the run configs are the public config files with only the paths, the W&B entity/project and those three values changed to the released ones, so the recipe is the one the released checkpoints were trained with as far as the public code can express it (`brev/dicerl/cfg/`).

Other environment notes: `wandb` 0.17.7 (reference pin) cannot initialise against the current W&B backend (`CommError: 'NoneType' object has no attribute 'get'` on a bare `wandb.init`); 0.27.2, the version the box's other environment uses, works, so the setup installs 0.27.2.

## Gate 1 runs (2026-10-20, Jarvis)

`brev/dicerl/finetune.sh <env> 42`: public code, released recipe values, released pretrained checkpoints, 4 training envs, evaluation of 300 episodes on 30 envs every 1,000 steps (one step = one action chunk on each of the 4 envs, i.e. 16 env steps on Square and 32 on Tool Hang, which is the W&B x-axis). W&B project `dice-rl-repro`. Comparison points from the released logs, in steps: Square 2,000 (0.68), 4,000 (0.69), 10,000 (0.887), 15,000 (0.937), 28,000 (0.98); Tool Hang 3,000 (0.553), 5,000 (0.71), 14,000 (0.797), 34,000 (0.937).

### Step 0 (both runs, 21:14 BST)

| run | pretrained flow BC, ours | authors' log | random-init residual at step 0, ours | authors' log |
|---|---|---|---|---|
| Square | 0.587 | 0.583 | 0.007 (mean length 399.1) | 0.007 (399.6) |
| Tool Hang | 0.443 | 0.497 | 0.000 | 0.000 |

The training-time evaluation path reproduces the released step-0 rows to the third decimal on Square, which closes Square's gate 0 (no fine-tuned Square checkpoint is released). The Tool Hang pretrained value sits 5 points under the released log, consistent with the gate 0 figure above and inside the ±8 tolerance; the open question on that row stays open. The released Tool Hang CSV reports lengths in chunks (88 = 704 / 8), the released Square CSV in steps, a cosmetic difference between the authors' two code versions.

### Throughput: the two runs must not share the GPU

Measured with the W&B step counter (fresh API client per sample; the client caches run objects, so a reused one reports stale values):

| condition | Square | Tool Hang | authors' log |
|---|---|---|---|
| both running on the GB10 | 12.0 s/step | 33.4 s/step | — |
| alone (other process paused with SIGSTOP for 5 min) | 5.6 s/step | 6.4 s/step | Square 13–18 s/step, Tool Hang 4.7–7 s/step |

Both processes are single-threaded Python at about 85% of one core each; CUDA time-slicing between two contexts issuing many small kernels costs 2× on Square and 5× on Tool Hang, so concurrent runs are strictly worse than sequential ones (Tool Hang to 34,000 steps: 13 days concurrent, 60 h alone; Square to 28,000: 93 h concurrent, 44 h alone). Checkpoints: the agent reads `train.save_freq` (absent, default 1,000 steps), not `save_model_freq`, so weights are saved at every 1,000-step evaluation (32k env steps on Tool Hang, 16k on Square) under the run directory's `checkpoint/`. The Tool Hang step is heavier than Square's because the public replay buffer's n-step loop runs (n−1)·horizon+1 synchronised GPU iterations per sample: 33 with n = 5, horizon 8 versus 9 with n = 3, horizon 4. Decision: run one fine-tune at a time, Tool Hang first (it has the gate 0 checkpoint), Square after it reaches 34,000 steps; projected 2.5 days plus 1.8 days, the first comparison point of each inside 6 h of its launch.

### Launch record

- 21:14 BST: both launched concurrently; 22:02: Square stopped at step 339 after the measurement above (its W&B run `kp2dh169` ends there; its run directory keeps the step-0 CSV).
- 22:03: the `pkill` used to clear Square's leftover env workers matched the shared tmux server (its command line names the first session's command), which took the Tool Hang run (`epveqvjz`, step 230) down with it. Lesson: kill by pid, never by a pattern that the tmux server's or the shell's own command line can match.
- 22:06 BST: Tool Hang seed 42 relaunched alone (`dicerl-tool_hang-42`, W&B `hm8ciyax`). The box-side tmux session `dicerl-chain` (`~/Documents/dice/dicerl/chain.sh`, log `chain.log`) stops it by session name once its evaluation CSV holds the step-34,000 row and launches Square seed 42. Earlier failed launches (wandb 0.17.7, released-config unknown keys) remain in the same log files, appended by `tee -a`.

### Tool Hang seed 42, success on 300 held-out episodes (run `hm8ciyax`, 6.3 s/step, stopped at 448k on 2026-10-21 21:34 BST)

| env steps | 32k | 64k | 96k | 128k | 160k | 192k | 224k | 256k | 288k | 320k | 352k | 384k | 416k | 448k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | 0.330 | 0.287 | 0.520 | 0.473 | 0.553 | 0.617 | 0.553 | 0.750 | 0.727 | 0.787 | 0.803 | 0.853 | 0.850 | 0.823 |
| authors | 0.183 | 0.340 | 0.553 | 0.457 | 0.710 | 0.473 | 0.600 | 0.633 | 0.683 | 0.737 | 0.697 | 0.770 | 0.730 | 0.797 |

Gate 1 verdict for Tool Hang: pass. Mean over the fourteen points 0.638 ours against 0.597 theirs; the last seven points average 0.799 against 0.721; no point is below its counterpart by more than the ±6-point sampling noise, and the 448k comparison point is +2.6. The run was stopped at 448k (the spec's last Tool Hang comparison point before the 1.09M endpoint) to free the box. Square seed 42 was launched at 21:34 BST and stopped 25 minutes later: Tool Hang alone establishes that the environment, public code and released recipe reproduce the authors' curve, Square has no released fine-tuned checkpoint to compare against, and the next phase needs the GPU. Its step-0 rows (pretrained 0.587, residual 0.007) are recorded above. Gate 3 (pretraining) is deferred for the same reason; the LIBERO phase pretrains its own prior with the released pretrain config, which exercises that path.

### Evaluation noise (step-7,000 checkpoint, 2026-10-21)

The standalone evaluator (`script/eval_rl_checkpoint.py`) and the training loop use the same 300 reset seeds (10000–10299, applied through the robomimic wrapper's `np.random.seed` at every reset), the same 30 envs, the same max-Q-min selection with 16 samples and equivalent multi-step wrappers; the training loop additionally seeds torch with the run seed. Two standalone runs with torch seed 42 are identical, so both paths are deterministic given the seed and differ only by the policy's action noise. Repeated standalone evaluations of the same checkpoint on the same 300 initial states:

| torch seed | unseeded | unseeded | 42 | 43 | training loop (seed 42) |
|---|---|---|---|---|---|
| fine-tuned, step 7,000 | 0.463 | 0.450 | 0.570 | 0.543 | 0.553 |
| pretrained | 0.366 | 0.520 | 0.473 | 0.443 | 0.443 |

The spread across draws is about 6 points standard deviation, twice the binomial 2.9 for 300 independent episodes: on Tool Hang the outcome of an initial state depends more on the sampled action noise than on the state. The earlier "9-point gap" between the two paths was one such draw. Every 300-episode point on both our curve and the authors' therefore carries roughly ±6 points of sampling noise before any seed-to-seed training variance, and gate verdicts must average several evaluation points rather than read single rows.

### Contraction diagnostic (`script/dicerl_sharpen.py`, `result/dicerl-sharpen/tool_hang_42/`)

For each checkpoint: 60 max-Q-min rollouts on seeds 10000+ give 4,000–4,500 states; at each state K = 16 noises are drawn, the base action π_pre(s, z) and the corrected action π_pre(s, z) + r(s, z) are scored by the critic (min over the 10-ensemble), and the spread of the 16 corrected actions is compared with the spread of the 16 base actions (binned entropy ΔH with 50 bins over [−1, 1], and continuous Δlog-std).

| step (env steps) | rollout success | V = mean Q(base) | ΔV | ΔH | Δlog-std | residual norm | Δlog-std by V quartile (low → high) |
|---|---|---|---|---|---|---|---|
| 1,000 (32k) | 0.20 | 0.231 | +0.015 | +0.015 | +0.100 | 0.096 | 0.073, 0.121, 0.108, 0.098 |
| 4,000 (128k) | 0.60 | 0.313 | +0.027 | +0.001 | +0.086 | 0.119 | 0.083, 0.103, 0.098, 0.060 |
| 7,000 (224k) | 0.53 | 0.266 | +0.029 | −0.001 | +0.032 | 0.116 | 0.025, 0.050, 0.031, 0.023 |
| 8,000 (256k) | 0.70 | 0.338 | +0.037 | −0.007 | +0.021 | 0.116 | 0.031, 0.023, 0.011, 0.018 |

Reading: the residual raises the critic's value of the executed action at every state (ΔV > 0 in all quartiles) and the gain grows with training, but it does not contract the action distribution. Early on it widens the spread of the K candidates (Δlog-std +0.10, i.e. 10% wider), and the widening decays towards zero by 256k steps; the per-state correlation between ΔH and either ΔV or V is below 0.14 in magnitude at every checkpoint, and the highest-value quartile is not contracted more than the others. The residual is a value-increasing shift of the whole candidate set rather than a sharpening of it. The same measure on the LingBot bounded recipe is in `docs/investigations/2026-10-20-residual-and-critic.md`; later Tool Hang checkpoints (14,000 and 34,000) will be added as the run reaches them.

Rerun with the paper's measurement (256 candidates per state, 50-bin histogram entropy, demonstration anchor states from the expert dataset as well as rollout states; `result/dicerl-sharpen/k256_*`):

| step (env steps) | anchors | ΔV | ΔH | Δlog-std | r(ΔV, ΔH) | ΔH by ΔV quartile (low → high) |
|---|---|---|---|---|---|---|
| 1,000 (32k) | expert / rollout | +0.011 / +0.015 | +0.011 / +0.016 | +0.080 / +0.094 | 0.06 / 0.17 | 0.009, 0.010, 0.011, 0.016 / 0.004, 0.011, 0.019, 0.029 |
| 4,000 (128k) | expert / rollout | +0.023 / +0.027 | +0.003 / +0.001 | +0.043 / +0.080 | 0.12 / 0.03 | 0.000, −0.001, 0.004, 0.008 / 0.002, −0.001, 0.001, 0.001 |
| 8,000 (256k) | expert / rollout | +0.027 / +0.037 | −0.003 / −0.008 | +0.016 / +0.019 | 0.02 / 0.02 | −0.003, −0.003, −0.004, −0.001 / −0.007, −0.009, −0.005, −0.009 |
| 11,000 (352k) | expert / rollout | +0.037 / +0.035 | −0.005 / −0.008 | +0.003 / +0.001 | 0.00 / −0.01 | −0.003, −0.006, −0.005, −0.005 / −0.005, −0.007, −0.013, −0.009 |

With 256 candidates the picture is the same as with 16: the paper's Fig. 5 coupling ("larger value gains come with larger entropy drops") is absent through 352k; where the correlation is non-zero (step 1,000) it has the opposite sign, larger ΔV with slightly more spread. The residual's effect on the candidate spread goes from +8–9% wider at 32k to neutral at 352k, while ΔV triples. Entropy at the 448k checkpoint will be added when the run reaches it.
