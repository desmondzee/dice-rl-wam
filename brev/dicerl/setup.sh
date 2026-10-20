#!/usr/bin/env bash
set -euo pipefail
DICE_DATA="${DICE_DATA:-$HOME/Documents/dice}"
UV="${UV:-$HOME/.local/bin/uv}"
VENV="$DICE_DATA/dicerl-venv311"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$UV" python install 3.11
[ -d "$VENV" ] || "$UV" venv --python-preference only-managed --python 3.11 "$VENV"
PY="$VENV/bin/python"
"$UV" pip install --python "$PY" "av==12.3.0" "einops==0.8.0" "gdown==5.2.0" "gym==0.22.0" "huggingface_hub[cli]==0.23.4" "hydra-core==1.3.2" "imageio==2.35.1" "matplotlib==3.7.5" "omegaconf==2.3.0" "pretty_errors==1.2.25" "plotly==5.24.1" "pandas==2.0.3" "tqdm==4.66.5" "wandb==0.27.2" "cython<3" "patchelf" "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0" "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.4.1" "mujoco==2.3.7"
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" torchvision
mkdir -p "$DICE_DATA/dicerl/data_dir/robomimic" "$DICE_DATA/dicerl/log_dir/robomimic-pretrain" "$DICE_DATA/dicerl/log_dir/robomimic-finetune"
ln -sfn "$DICE_DATA/dicerl/data_dir" "$REPO/data_dir"
ln -sfn "$DICE_DATA/dicerl/log_dir" "$REPO/log_dir"
MUJOCO_GL=egl "$PY" -c 'import torch, gym, robosuite, robomimic, mujoco; print(torch.__version__, torch.cuda.is_available(), gym.__version__, robosuite.__version__, robomimic.__version__, mujoco.__version__)'
