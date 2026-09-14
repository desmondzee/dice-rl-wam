# DICE-RL on LingBot-VA

This repository explores DICE-RL post-training of LingBot-VA on **LIBERO-10 (LIBERO-Long)**. It starts from `robbyant/lingbot-va-base`, not the converged LIBERO checkpoint.

**Current scope: SFT, vanilla LingBot checkpoint evaluation, and a Modal DICE-RL residual trainer.** SFT uses **30 demonstrations per task, 300 demonstrations total**, with a **1,000-update cap**; the first run was stopped after saving step 600. Thirty refers to demonstrations, not tasks or optimizer steps. Evaluation loads the native transformer weights directly into a pinned LeRobot policy; no on-disk checkpoint conversion is required. RL freezes that step-600 prior and trains residual + critic MLPs with the same 20/50 full-video sampler that scored 69% on LIBERO-10.

References: [DICE-RL, including Appendix A](https://arxiv.org/html/2603.10263v2), [LingBot-VA, especially Sections 3–4](https://arxiv.org/html/2601.21998v2), [upstream LingBot-VA](https://github.com/Robbyant/lingbot-va), and [LeRobot integration](https://huggingface.co/docs/lerobot/lingbot_va).

## Initial SFT recipe

| Setting | Initial run |
| --- | --- |
| Suite/data | `robbyant/libero-long-lerobot`; 30 whole demonstrations from each of 10 tasks |
| Subset | Deterministic per-task SHA-256 ranking with seed 42; selected episode IDs saved in a manifest |
| Initial weights | `robbyant/lingbot-va-base`; full shared transformer fine-tuning, not LoRA |
| Frozen inputs | Published precomputed Wan VAE latents and UMT5 task embeddings; empty prompt encoded once during preparation |
| Cameras | Agentview then wrist, each 128×128; concatenate their latents along width |
| Actions | Seven LIBERO delta-action channels in channels 0–6 of a 30-dimensional vector; unused channels zero-masked |
| Normalization | q01/q99 from the selected 300 demos only; epsilon 1e-6, clipping to [-1.5, 1.5] |
| Optimizer | AdamW, LR 1e-5, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1, gradient clip 2.0 |
| Schedule | 10 warmup updates, then constant LR; stop at 1,000 updates |
| Distributed training | Native upstream FSDP2, activation checkpointing, BF16 compute, FP32 training parameters |
| Effective batch | 80 whole episodes per update: one per GPU microbatch, accumulation 10 on 8 GPUs or 20 on 4 |
| Training sequence | Full published latent episodes; random AR chunk size 1–4, attention window 4–64, text dropout 0.1, noisy video-history probability 0.5 |
| Weight checkpoints | Updates 100, 200, 400, 600, 800, 1,000; a shorter run also saves its final update |
| Resume state | One rolling full checkpoint every 100 updates and at weight checkpoints/final update |
| Tracking | W&B losses, gradient norm, LR, update time, throughput, GPU memory, and provenance |

The default is **8 H100s**, with a 4-H100 option at the same effective batch size. Faster wall-clock training is a hypothesis until measured; allocation delay, compilation, full-episode lengths, and checkpoint I/O affect runtime. No fixed runtime or success rate is promised.

No region, cloud, or routing region is pinned. Modal's published region multipliers are 1.15× for broad selection and 1.75× for narrow selection ([region selection](https://modal.com/docs/guide/region-selection)). As checked on 2026-09-13, published H100 pricing is $0.001097/second, approximately $3.95 per GPU-hour ([pricing](https://modal.com/pricing)): about $15.80/hour for four H100s or $31.59/hour for eight, before CPU, RAM, storage, and other charges. Four GPUs use accumulation 20 and eight use accumulation 10 at the same effective batch size; communication and checkpoint I/O remain, so the speed/cost tradeoff must be measured rather than assumed.

The adapter reuses the pinned upstream trainer's noising, dual-stream flow-matching loss, and update code. Two small checked patches select our restricted dataset loader and make the unused FlashAttention import optional; training still requires flex attention. It does not install the legacy root project's dependencies or LeRobot's conflicting training dependency stack.

## Run on Modal

Run all commands from this repository root using `uv`. Do **not** install the root `pyproject.toml` for LingBot SFT.

### 1. Authenticate and create access secrets

```bash
uv run --no-project --with modal==1.1.4 modal token new --profile james-j-carver2
uv run --no-project --with wandb==0.21.1 wandb login
uv run --no-project --with modal==1.1.4 python -m script.lingbot_sft_secret
uv run --no-project --with modal==1.1.4 python -m script.lingbot_sft_secret --service hf
```

Skip logins already completed. The Modal profile name is a local label; select the intended workspace in browser login. The W&B command prompts invisibly for your W&B key and creates the Modal Secret `dice-lingbot-wandb` containing `WANDB_API_KEY`. The `--service hf` command stores your read-only Hugging Face token in the Secret `dice-lingbot-hf` as `HF_TOKEN`; that token is supplied only to CPU preparation. Local Hugging Face or W&B login does not transfer credentials to Modal. The helper sends the key directly to the Modal SDK without writing a local credential file, and does not overwrite an existing secret. Never put keys in this repository, shell arguments, or chat.

### 2. Inspect the recipe locally

```bash
uv run --no-project python -m script.lingbot_sft_config --gpus 8
```

This is a local configuration preview: no cloud resources, weights, or dataset downloads.

### 3. Launch SFT

The following command **starts paid Modal compute**:

```bash
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_sft_modal \
  --run-name libero30-sft --gpus 8 --steps 1000
```

CPU preparation downloads the pinned subset/model assets, validates latent alignment, fits subset-only normalization, and creates the empty prompt embedding before GPU allocation. Use `--gpus 4` for four H100s. Add `--wandb-entity YOUR_ENTITY` if needed; the project defaults to `dice-lingbot-va-sft`. The Modal workspace name is not assumed to be your W&B entity.

For a first hardware smoke test, use a distinct run name and `--steps 30`. This still uses 30 demos/task but performs only 30 updates; it is a launch/throughput check, not a useful-prior claim. GPU execution and the container image build require validation on Modal; local tests cannot establish that they work on H100s.

If preparation reports a Hugging Face HTTP 429 from an anonymous shared IP, create `dice-lingbot-hf` with the read-only `HF_TOKEN` command above and rerun the original command without `--resume`; no training state exists when preparation fails. Preparation makes up to four attempts (three retries) with bounded backoff, uses two download workers, and commits the cache volume even on failure so valid partial cache contents can be reused. An NVIDIA warning in the CPU preparation container is expected when no GPU is allocated.

Keep the local command running for the automatic download. After successful training, only the final inference checkpoint is downloaded to:

```text
checkpoints/lingbot-sft/libero30-sft/step_001000/
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors
  norm_stats.json
  sft_config.json
  dataset_manifest.json
```

Use `--download-dir PATH` to choose a different local parent. Existing local checkpoint directories are not overwritten. Earlier milestones and the full resume state remain on Modal. The downloaded checkpoint is in **upstream LingBot-VA format**, not yet LeRobot format. Inference must also load the frozen VAE/text/tokenizer assets from the pinned base model and use the saved subset normalization.

### Resume or download manually

Resume an interrupted run with the same name, GPU count, and immutable recipe:

```bash
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_sft_modal \
  --run-name libero30-sft --gpus 8 --steps 1000 --resume
```

Resume restores model, optimizer, LR scheduler, per-rank RNG, and consumed data position. It requires `resume/latest.pt`; weights-only checkpoints cannot resume training. A short smoke run can be continued to 1,000 updates, but a completed 1,000-update run should only be downloaded/evaluated, not resumed.

If the client disconnected or local downloading failed, retrieve a saved checkpoint without rerunning training:

```bash
uv run --no-project --with modal==1.1.4 modal volume get dice-lingbot-sft-runs \
  libero30-sft/checkpoints/step_001000 ./libero30-sft-step_001000
```

The named volumes are `dice-lingbot-sft-cache` (dataset/model assets) and `dice-lingbot-sft-runs` (checkpoints/provenance). Large optimizer state is deliberately not downloaded. A persistent run lock prevents concurrent writers; after abrupt termination, a stale lock must be inspected and cleared only after confirming the old job has stopped.

## Evaluate step 600 on one H100

We fine-tuned only on **LIBERO-Long / LIBERO-10**, not Spatial, Object, or Goal. The default evaluation reads step 600 directly from `dice-lingbot-sft-runs`; there is no need to upload the local weight copy. The SFT checkpoint volume is mounted read-only.

The commands below start paid Modal compute when you run them. No evaluation job was launched during local verification. Keep the launch terminal running for the automatic result download. Click and Typer are pinned in the isolated CLI environment because newer versions made Modal 1.1.4's help command fail.

First run the **smoke test: one complete episode per task, 10 episodes total**:

```bash
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 \
  modal run -m script.lingbot_eval_modal --stage smoke
```

Inspect the rollout videos and confirm that camera orientation, control, and episode resets look correct. A smoke test is a plumbing check, not a reliable success-rate estimate. Then run **20 complete episodes per task, 200 total**:

```bash
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 \
  modal run -m script.lingbot_eval_modal --stage eval
```

Both stages use one H100 and one environment at a time, 128×128 agentview/wrist observations, the saved 300-demo action quantiles, BF16/SDPA, and the released 20-video/50-action-step sampler (CFG 5/1, four latent frames per chunk, no early video cutoff). Camera preprocessing explicitly preserves the native LingBot client's vertical-only flip rather than the generic LeRobot LIBERO processor's two-axis flip. Frozen text embeddings are cached across episodes without changing prompts or sampling.

The environment protocol is LeRobot's LIBERO-10 default: **520 policy actions maximum**, 10 settling actions, 20 Hz relative control, and hard resets. This differs from the native LingBot client's 800-step budget; it is not an exact reproduction of every upstream evaluation detail. Smoke uses initial-state index 0; evaluation uses indices 1–20. Initial-state IDs and per-episode seeds are recorded and held constant across checkpoints. These state IDs are separated between smoke/eval, but are not claimed to be disjoint from the SFT demonstrations.

CPU preparation checks native checkpoint tensor names/shapes, downloads the previously unneeded frozen VAE and pinned LIBERO simulation assets using `dice-lingbot-hf`, and commits the cache before GPU allocation. GPU evaluation gets only the W&B secret; model/cache mounts are read-only and Hub access is offline. Preparation downloads use two workers and bounded retries. No region/cloud is pinned. The GPU function has a six-hour invocation limit and no automatic retry.

Results are saved after every completed episode to `dice-lingbot-eval-results`. Successful runs download to:

```text
result/lingbot-eval/libero30-sft-step000600-smoke/
result/lingbot-eval/libero30-sft-step000600-eval/
```

Each contains `settings.json` (checkpoint hashes, dependency versions, protocol, and episode plan), per-episode JSON records, `summary.json` (success counts and per-task rates), `status.json`, and one actual rollout video per task. Imagined-video decoding is disabled. W&B uses project `dice-lingbot-va-eval`. Partial results are labeled incomplete; the final macro success rate is emitted only after all expected episodes finish.

After download, local reporting validates every planned episode, its identity and outcome fields, the aggregate/per-task counts, completed status, and all referenced videos. It adds `report.html` (offline, searchable episode table and per-task overview), `episodes.csv` (one row per episode), `tasks.csv` (per-task outcomes and timing), and `report_data.json` (the validated data snapshot, settings, and source-file SHA-256 hashes). Open `report.html` in a browser. These derived files are generated locally; the original JSON records and videos remain unchanged locally and on Modal.

To add reports to an already downloaded run without cloud compute or inference:

```bash
.cache/eval-venv/bin/python -m script.lingbot_eval_report \
  --result-dir result/lingbot-eval/libero30-sft-step000600-eval
```

The same command works for the `-smoke` directory. Identical existing reports are left unchanged; differing existing reports are not overwritten. Episode seconds include reset/inference/simulation and video-frame writes, but exclude model loading, between-episode persistence, and video finalization. Reported memory is a cumulative PyTorch allocated-memory high-water mark, not total GPU memory. Only the first episode per task was filmed; other videos and per-step action/observation traces cannot be recovered from these artifacts.

Use `--checkpoint-step 200` or `400` to evaluate an earlier saved prior. Add `--resume` to the same stage/run name after interruption; completed episodes are reused only when the checkpoint, protocol, dependencies, and harness match exactly. A stale persistent run lock requires explicit operator inspection/clearing after confirming shutdown. Use `--download-dir PATH` to choose a different local parent; existing outputs are not overwritten. `--stage prepare` performs CPU preparation only.

The later SFT-vs-DICE-RL benchmark should use 100 rollouts/task as reported in DICE-RL Appendix A; this initial 20-rollout evaluation is for selecting a useful prior, not the final paper-comparison result. Real Linux/EGL rendering, cloud image build, H100 throughput, and rollout success still require the smoke run.

## Paper comparison and checkpoint selection

- DICE-RL Appendix A uses **30 demos/task for its π0 experiment**. Its flow-policy comparison against DSRL uses 50/task. The exact π0 demo IDs, SFT update budget, and LIBERO chunk horizon are not published there; our subset is reproducible, not an exact reconstruction of theirs.
- LingBot-VA reports **4K LIBERO updates at 1e-5 and sequence length 100K**. The released shared-backbone config instead uses 5K updates and whole-episode microbatches. We follow the released trainer with an intentional 1K-update cap and reduced data budget, rather than claiming exact reproduction of the paper's 98.5% result.
- The published latent dataset is used as supplied. Its metadata has no independent per-demo success label; we cannot independently certify the paper's unsuccessful-demo filtering from that metadata.
- The intended prior has roughly **40–70% success**. This is a future rollout-based selection criterion, not an SFT metric or guarantee. If success is above about 80%, select an earlier checkpoint or revisit the demo budget before RL.
- The single-H100 evaluation adapter uses pinned LeRobot components and directly loads the native transformer checkpoint with strict tensor checks. Initial evaluation is 20 rollouts/task; final comparison remains 100 rollouts/task. Current LingBot streaming inference is single-environment; batched collection remains future work.

## DICE-RL residual training

The residual trainer freezes `libero30-sft` **step 000600** and uses the same released LIBERO sampler as the 69% eval (video 20, `video_exec_step=-1`, action 50, CFG 5.0/1.0). Do not switch to the paper real-time 3-step / s=0.6 decoder; that would be a different π_pre. Comparison eval is 20 rollouts/task, init-states 1–20, seed 42 — identical to `EvalConfig(stage="eval")`. Do not re-run the SFT 69% job.

The following command **starts paid Modal compute** (one H100, 12h train timeout). Use the active Modal profile (workspace `tiwariojas`). Add `--wandb-entity YOUR_ENTITY` if needed. W&B project is `dice-lingbot-va-rl`.

```bash
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_rl_modal \
  --stage train --run-name libero30-dice-baseline
```

Keep the client attached until download finishes. Local download is inference-only (`residual.pt`, summaries, train-eval rows, and after `--stage eval` the 20-rollout JSON/`report.html`). Resume state, replay, Adam, expert cache, and the 5B transformer stay on `dice-lingbot-rl-runs`. Volumes: `dice-lingbot-sft-cache` and `dice-lingbot-sft-runs` read-only on GPU; `dice-lingbot-rl-runs` writable. CPU prepare uses `dice-lingbot-hf`; GPU uses `dice-lingbot-wandb` with Hub offline.

`--stage eval` runs the 20-rollout comparison against the trained residual. Train-time eval is 1 rollout/task every 25k env steps and is not the 69% comparison.

## Local verification

See [AGENTS.md](AGENTS.md) for the separate pinned SFT and evaluation `uv` environments. `tests/test_lingbot_sft.py` covers SFT data restriction/alignment, normalization, checkpoint/resume paths, W&B setup, and mocked Modal commands. `tests/test_lingbot_eval.py` covers evaluation protocol, camera/action mapping, success accounting, result persistence/resume, asset retries, and checkpoint tensor schema. Real Linux/EGL rendering, CUDA inference/FSDP execution, and measured throughput are separate cloud checks.

## Original DICE-RL project

This fork retains the upstream Robomimic code and attribution below. Those installation/training instructions are separate from the LingBot-VA SFT path above.

<details>
<summary>Upstream DICE-RL documentation and attribution</summary>

# From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning (DICE-RL)
ICML 2026

[[Paper](https://arxiv.org/abs/2603.10263)]&nbsp;&nbsp;[[Website](https://zhanyisun.github.io/dice.rl.2026/)]&nbsp;&nbsp;[[Datasets](https://huggingface.co/datasets/wintermelontree/robomimic-pretrain-data)]&nbsp;&nbsp;[[Checkpoints](https://huggingface.co/wintermelontree/robomimic-pretrain-checkpoints)]&nbsp;&nbsp;[[Real Robot Code](https://github.com/zhanyisun/DICE-RL-Robot)]

[Zhanyi Sun](https://zhanyisun.github.io/), [Shuran Song](https://shurans.github.io/)

Stanford University

<img src="media/teaser.png" alt="Teaser" width="100%">

> DICE-RL is a sample-efficient and stable finetuning framework for diffusion- and flow-based Behavior Cloning policies. 

## Installation

1. Clone the repository.
```console
git clone git@github.com:real-stanford/dice-rl.git
cd dice-rl
```

2. Install core dependencies with a conda environment.
```console
conda create -n dice-rl python=3.8 -y
conda activate dice-rl
pip install -e .
```

3. Install Robomimic and dependencies.  We use `robomimic==0.5.0` and `robosuite==1.4.1` with `mujoco==3.2.3`. If you plan to use the provided checkpoints and data, please make sure the versions match.
```console
pip install -e .[robomimic]
```

4. [Install MuJoCo for Robomimic](installation/install_mujoco.md).

5. Set environment variables for data and logging directory (default is `data_dir/` and `log_dir/`), and set WandB entity.
```
source script/set_path.sh
```

## Download datasets and checkpoints from Hugging Face
Download all checkpoints and datasets from Hugging Face with the following command. This will download the datasets and checkpoints to the specified data and log directories respectively. If you only want to download specific datasets or checkpoints, you can find the links on the Hugging Face page and download them manually.

```console
bash script/download_hf.sh
```
The dowloaded datasets have the following structure:
```
data_dir/
├── robomimic
│   ├── {env_name}-low-dim
│   │   ├── ph_pretrain
│   │   └── ph_finetune
│   └── {env_name}-img
│       ├── ph_pretrain
│       └── ph_finetune
```

`data_dir/robomimic/{env_name}-low-dim/ph_pretrain` and 
`data_dir/robomimic/{env_name}-img/ph_pretrain`contain the datasets used for pretraining the BC policies, and `data_dir/robomimic/{env_name}-low-dim/ph_finetune` and `data_dir/robomimic/{env_name}-img/ph_finetune` contain the datasets used for finetuning the DICE-RL policies. `ph_finetune` is essentially the same as `ph_pretrain` with trajectories truncated to have exactly one success at the end to ensure the value learning between offline data and online data is consistent.. The datasets are in numpy format, and each dataset folder contains `train.npy` and `normalization.npz`. 

The checkpoints have the following structure:
```
log_dir/
├── robomimic-pretrain
│   ├── pretrained_bc_policy_{env_name}_low_dim
│   └── pretrained_bc_policy_{env_name}_img
└── robomimic-finetune
    ├── finetune_rl_policy_{env_name}_low_dim
    └── finetune_rl_policy_{env_name}_img
```
`log_dir/robomimic-pretrain/pretrained_bc_policy_{env_name}_low_dim` contains the pretrained BC checkpoints for state-based policies, and `log_dir/robomimic-pretrain/pretrained_bc_policy_{env_name}_img` contains the pretrained BC checkpoints for image-based policies.

### Generate your own data
You can optionally generate your own state and image datasets from the raw data downloaded from [this link](https://huggingface.co/datasets/wintermelontree/raw_robomimic_data/tree/main) or the official Robomimic repository. You can use `script/dataset/process_robomimic_dataset.py` to process raw datasets from Robomimic. See `script/dataset/README.md` for details. 

## Evaluating finetuned RL checkpoints and pretrained BC checkpoints
To directly evaluate the finetuned RL checkpoints and pretrained BC checkpoints and to get success rates for both, use the following commands. Make sure to change the pretrained checkpoint path in the config file of the finetuning checkpoint. The 
script/eval_rl_checkpoint.py script will automatically search for the corresponding pretrained BC checkpoint and evaluate it as well.

```console
python script/eval_rl_checkpoint.py  --ckpt_path  path_to_finetuned_checkpoint   --num_eval_episodes 10 --eval_n_envs 10
```
The output will include the success rates for both the finetuned RL checkpoint and the pretrained BC checkpoint, as well as the gain of the finetuned RL checkpoint over the pretrained BC checkpoint. You can specify `--num_eval_episodes` and `--eval_n_envs` to change the number of evaluation episodes and parallel evaluation environments respectively.

## Pretraining
**Note**: You may skip pre-training if you would like to use the default checkpoint (available for download at Hugging Face) for finetuning.

To pretrained state-based BC policies on the Robomimic dataset, use the following command. Make sure to change the paths to dataset and normalizer in the config file. You can optionally save eval videos during pretraining by changing the `save_video` flag in the config file to `True` an specifiy the number of eval envs for video saving.

```console
python script/run.py --config-name=pre_flow_matching_mlp --config-dir=cfg/robomimic/pretrain/{env_name}/
```

To pretrained image-based BC policies on the Robomimic dataset, use the following command. 

```console
python script/run.py --config-name=pre_flow_matching_unet_img --config-dir=cfg/robomimic/pretrain/{env_name}/
```

## Finetuning
To finetune the pretrained BC policies with DICE-RL, use the following command. Make sure to change the paths to finetuning dataset and normalizer in the config file, and also change the pretrained checkpoint path to the one you want to finetune from.

To finetune state-based policies, use the following command. You can optionally save eval videos during finetuning by changing the `save_video` flag in the config file to `True` an specifiy the number of eval envs for video saving.

```console
python script/run.py --config-name=ft_distill_residual_flow_mlp --config-dir=cfg/robomimic/finetune/{env_name}/
```

To finetune image-based policies, use the following command. 
```console
python script/run.py --config-name=ft_distill_residual_flow_unet_img --config-dir=cfg/robomimic/finetune/{env_name}/
```

### Key configurations for DICE-RL finetuning
Below we list some key configurations for DICE-RL finetuning that you can change in the config files. For more details on other configurations, please refer to the config files and the code.

* `bc_loss_weight`: the weight for BC loss. Setting it to 0 corresponds to pure online RL finetuning without distillation, and setting it to a large value corresponds to pure offline BC finetuning without online RL. In our experiments, we find that setting it to 50-100 works well across all tasks.

* `gradient_steps`: the number of gradient steps for each RL policy update. Used together with `n_envs` and `actor_update_freq`, it determines the UTD ratio for finetuning. In our experiments, we find that keeping the UTD ratio around 1 gives stable and sample-efficient training. 

* `n_step`: the number of steps for n-step return. In our experiments, we find that increasing this number to 3 or 5 works well for long-horizon tasks. 

* `critic_ensemble_size`: the number of critics in the critic ensemble. In our experiments, we find that using an ensemble of 10 critics works well for all tasks.

### Finetuning time
On an RTX 4090 GPU, finetuning the Transport (pixel) policy checkpoint takes about 24 hours to converge, while finetuning the Tool Hang (pixel) checkpoint takes about 48 hours. The main bottleneck is the RL update, which typically takes around 1 second per batch.


# Code Acknowledgements
Our code base is built on top of the following repositories. We thank the authors for open-sourcing their code.
- [DPPO](https://github.com/irom-princeton/dppo): our pretraining and finetuning stack is built on top of the DPPO codebase. 
- [Robomimic](https://github.com/ARISE-Initiative/robomimic) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): the encoder and policy architecture for image-based policies are adapted from the codebases of Robomimic and Diffusion Policy. 


If you find this codebase useful, consider citing:

```bibtex
@article{sun2026prior,
  title={From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning},
  author={Sun, Zhanyi and Song, Shuran},
  journal={arXiv preprint arXiv:2603.10263},
  year={2026}
}
```

# Contact
If you have any questions, please feel free to contact [Zhanyi Sun](mailto:zhanyis@stanford.edu). If you leave an issue, please send me an accompanying email!

</details>
