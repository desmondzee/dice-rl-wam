# LingBot-VA SFT

## Verified local commands

Use the pinned environment at `.cache/sft-venv/bin/python` and do not install the root project. The narrow verification commands are:

```text
.cache/sft-venv/bin/python -m pytest -q tests/test_lingbot_sft.py
.cache/sft-venv/bin/python -m compileall -q script/lingbot_sft_*.py
.cache/sft-venv/bin/python -m script.lingbot_sft_config --gpus 8
.cache/sft-venv/bin/python -m script.lingbot_sft_config --gpus 4
.cache/sft-venv/bin/python -m script.lingbot_sft_train --help
.cache/sft-venv/bin/python -m script.lingbot_sft_secret --help
```

Native checkout validation and patching use `.cache/lingbot-va` at revision `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`:

```text
.cache/sft-venv/bin/python -m script.lingbot_sft_patch --root .cache/lingbot-va
PYTHONPATH=$PWD:.cache/lingbot-va .cache/sft-venv/bin/python -c 'from wan_va.train import Trainer'
```

## Architecture

The SFT adapter uses the pinned native LingBot-VA Trainer and FSDP flow-loss math. `script/lingbot_sft_patch.py` makes two counted, idempotent upstream edits: it aliases the restricted `LatentDataset` and makes FlashAttention optional without changing flex attention. `script/lingbot_sft_data.py` supplies the precomputed-latent dataset, selected-demo manifest, action normalization, alignment, and empty text embedding.

`script/lingbot_sft_train.py` subclasses the patched native Trainer, controls deterministic episode cursors and main-process CFG dropout, counts optimizer updates, logs only rank zero metrics, and maintains one resumable full training state at `resume/latest.pt`. Milestone checkpoints contain only bf16 transformer weights plus JSON configuration, normalization, and manifest artifacts.

`script/lingbot_sft_modal.py` separates CPU cache preparation from 4-H100 and 8-H100 training functions and downloads only the successful final inference checkpoint locally. `script/lingbot_sft_secret.py` accepts a W&B key through a hidden prompt and calls `modal.Secret.objects.create` without writing a local credential file.

## Boundaries

The recipe is restricted to 30 demos per each of ten tasks, effective batch size 80, at most 1,000 optimizer updates, full-transformer training, and precomputed VAE/text inputs. VAE and text encoder parameters remain frozen by construction. There is no RL stage, no LeRobot installation, no flash-attention build, no weights-only resume fallback, and no automatic retry or new-run behavior after interruption.

Do not modify the root `pyproject.toml`, legacy RL code, lead-authored configuration/data algorithms, or published dataset probe. Do not run authentication, Modal cloud jobs, GPU allocations, model downloads, or the unrelated RL suite as local verification. W&B credentials must remain in the environment or named Modal secret and must never enter logs, command arguments, metadata, or artifacts. Modal run locks are persistent until the owning function releases them; a stale lock requires explicit operator confirmation and clearing, with no automatic takeover or TTL.

## Reproducible setup

From the repository root, verify the parent with `ls`, create the ignored cache, clone with GitHub CLI, and checkout the pinned upstream:

```text
mkdir -p .cache
gh repo clone Robbyant/lingbot-va .cache/lingbot-va
git -C .cache/lingbot-va checkout 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
uv venv --python 3.11 .cache/sft-venv
uv pip install --python .cache/sft-venv/bin/python torch==2.9.0 numpy==1.26.4 pyarrow==19.0.1 huggingface-hub==0.34.4 pytest==8.4.1 modal==1.1.4 wandb==0.21.1 diffusers==0.36.0 transformers==4.55.2 accelerate==1.10.1 einops==0.8.1 easydict==1.13 ftfy==6.3.1 sentencepiece==0.2.1 websockets==15.0.1 msgpack==1.1.1 imageio==2.37.0 imageio-ffmpeg==0.6.0 opencv-python-headless==4.11.0.86 matplotlib==3.10.5 safetensors==0.6.2
```

Do not install the root project or FlashAttention. The pinned dataset/model preparation is performed by the Modal CPU stage before selecting the GPU stage. CPU preparation requires the read-only `HF_TOKEN` in the named `dice-lingbot-hf` Modal secret; the GPU functions use only the prepared cache and receive only the W&B secret. No Modal region, cloud, or routing region is pinned.

## User launch commands

These commands are usage examples and were not executed during local verification. If CPU preparation fails with a Hugging Face 429, use the `--service hf` secret command and rerun the original non-resume command; bounded retries, two workers, and cache commits on failure preserve reusable partial downloads:

```text
uv run --no-project --with modal==1.1.4 modal token new --profile james-j-carver2
uv run --no-project --with wandb==0.21.1 wandb login
uv run --no-project --with modal==1.1.4 python -m script.lingbot_sft_secret
uv run --no-project --with modal==1.1.4 python -m script.lingbot_sft_secret --service hf
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_sft_modal --stage prepare --steps 1000
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_sft_modal --stage train --gpus 8 --steps 1000 --run-name libero-sft
```

The secret helper prompts invisibly and calls `modal.Secret.objects.create('dice-lingbot-wandb', {'WANDB_API_KEY': key})` directly. It never places the key in argv, logs, local files, or credential files, and it does not overwrite an existing secret.

## Downloading saved inference checkpoints

With Modal 1.1.4, pass an existing local parent directory to `modal volume get` when downloading a directory. Passing a nonexistent checkpoint leaf produced `IsADirectoryError`; the following user-authorized step-600 transfer succeeded, with all five local file sizes matching the remote files and JSON/Safetensors structure checks passing:

```text
ls -ld .
mkdir -p checkpoints/lingbot-sft/libero30-sft
.cache/sft-venv/bin/python -m modal volume get dice-lingbot-sft-runs libero30-sft/checkpoints/step_000600 checkpoints/lingbot-sft/libero30-sft
```

The resulting local directory is `checkpoints/lingbot-sft/libero30-sft/step_000600`. This copies only inference artifacts; `resume/latest.pt` and earlier weight checkpoints stay on the Modal volume. Do not use `--force` over an existing download. The current `download_checkpoint` wrapper still passes the checkpoint leaf, so use this manual parent-directory workaround until that wrapper is corrected.

## LingBot evaluation

Evaluation is separate from SFT: `script/lingbot_eval_config.py` defines the fixed LIBERO-10 smoke/20-rollout protocols, `script/lingbot_eval.py` validates and loads native checkpoint tensors into LeRobot and runs deterministic episode plans, and `script/lingbot_eval_modal.py` separates CPU asset preparation from one-H100 evaluation. The checkpoint/cache volumes are read-only on the GPU; only results are writable. Do not change the SFT environment or legacy RL code for evaluation.

Pinned LeRobot checkout: `.cache/lerobot` at `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`, Python 3.12. Its existing `uv.lock` supplies the `lingbot_va`, `libero`, and `evaluation` extras (Torch 2.11/CUDA 12.8 on Linux); no FlashAttention build is needed. The local macOS environment intentionally does not install the Linux-only `hf-libero` package.

Local setup (verify `.cache` exists first):

```text
UV_PROJECT_ENVIRONMENT=$PWD/.cache/eval-venv uv sync --project .cache/lerobot --python 3.12 --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation
uv export --project .cache/lerobot --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-emit-project --no-hashes --output-file $PWD/.cache/lingbot-eval-deps.txt
uv pip install --python .cache/eval-venv/bin/python --constraint .cache/lingbot-eval-deps.txt --exclude-newer 2026-09-05T00:00:00Z modal==1.1.4 wandb==0.27.2 pytest==8.4.1 imageio==2.37.4 imageio-ffmpeg==0.6.0
```

Narrow checks:

```text
.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_eval.py
.cache/eval-venv/bin/python -m compileall -q script/lingbot_eval.py script/lingbot_eval_config.py script/lingbot_eval_modal.py
.cache/eval-venv/bin/python -m script.lingbot_eval --help
.cache/eval-venv/bin/python -m script.lingbot_eval_config --stage eval
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 modal run -m script.lingbot_eval_modal --help
```

Do not run `.remote()`, cloud builds, model downloads, real rollouts, or GPU allocations as local verification. Local tests cover mocked rollout/persistence behavior and a header-only schema check of the saved step-600 transformer when present; they do not establish cloud/EGL/inference correctness. Keep the released 20/50 sampler, vertical-only camera flip, saved subset quantiles with native epsilon, explicit initial-state IDs/seeds, and 520-step LeRobot horizon identifiable in every result. Stop/error on missing or malformed success signals; do not count infrastructure errors as policy failures. Broader-suite evaluation remains unimplemented.

## LingBot DICE-RL

Residual RL lives in `script/lingbot_rl_*.py` on the **eval** LeRobot environment. Do not instantiate Hydra `DistillResidualRLModel`. Do not edit `.cache/lerobot` or `script/lingbot_eval.py` protocol fields. Local tests must not call `modal run` `.remote()`.

Narrow checks:

```text
.cache/eval-venv/bin/python -m pytest -q tests/test_lingbot_rl.py
.cache/eval-venv/bin/python -m compileall -q script/lingbot_rl_*.py
.cache/eval-venv/bin/python -m script.lingbot_rl_config
uv run --no-project --with modal==1.1.4 --with click==8.1.8 --with typer==0.16.0 modal run -m script.lingbot_rl_modal --help
```

User-run (paid) train, not executed as local verification:

```text
uv run --no-project --with modal==1.1.4 modal run -m script.lingbot_rl_modal --stage train --run-name libero30-dice-baseline
```

Modal image builders use an internal PyPI mirror by default. LeRobot's pinned lockfile records https://pypi.org/simple, so evaluation image uv sync, uv export, and constrained uv pip install commands explicitly pass --index-url https://pypi.org/simple. Without that flag, uv reports a missing remote index, re-resolves, and fails --locked. Keep --locked and the upstream lockfile unchanged; upgrading uv or regenerating the lock is not the fix.

## Local evaluation reports

`script/lingbot_eval_report.py` is stdlib-only postprocessing, separate from the inference harness so it does not change its provenance hash or resume behavior. `download_results` validates all expected episode JSON files, identities, outcomes, summary counts, completed status, and recorded videos, then generates local `report.html`, `episodes.csv`, `tasks.csv`, and the frozen `report_data.json` input snapshot. Renderers consume the snapshot data; they do not run inference. Raw result files are never changed, and differing existing reports are not overwritten.

Backfill existing downloads with `.cache/eval-venv/bin/python -m script.lingbot_eval_report --result-dir result/lingbot-eval/libero30-sft-step000600-eval` (or the `-smoke` directory). Reporting tests are included in `tests/test_lingbot_eval.py`; also compile `script/lingbot_eval_report.py` and check its `--help` entrypoint. Report timing is episode wall time, not pure GPU time or full application duration; memory is the cumulative PyTorch allocated-memory peak. Additional rollout videos and per-step trajectories remain unrecorded.
