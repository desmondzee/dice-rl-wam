#!/usr/bin/env bash
set -euo pipefail
DICE_DATA="${DICE_DATA:-$HOME/Documents/dice}"
UV="${UV:-$HOME/.local/bin/uv}"
VENV="$DICE_DATA/dicerl-libero-venv"
"$UV" python install 3.11
[ -d "$VENV" ] || "$UV" venv --python-preference only-managed --python 3.11 "$VENV"
PY="$VENV/bin/python"
"$UV" pip install --python "$PY" "numpy==1.26.4" "gym==0.22.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "einops==0.8.0" "imageio==2.35.1" "imageio-ffmpeg==0.6.0" "matplotlib==3.7.5" "pandas==2.0.3" "tqdm==4.66.5" "wandb==0.27.2" "h5py==3.16.0" "hf-libero==0.1.4" "robosuite==1.4.0" "mujoco==3.8.1" "robomimic==0.2.0" "pretty-errors==1.2.25" "gdown==5.2.0" "plotly==5.24.1"
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" torchvision
mkdir -p "$DICE_DATA/dicerl/data_dir/libero" "$DICE_DATA/dicerl/log_dir/libero-pretrain"
MUJOCO_GL=egl "$PY" -c 'import numpy, torch, gym, robosuite, mujoco, libero, h5py; print(torch.__version__, torch.cuda.is_available(), robosuite.__version__, mujoco.__version__, numpy.__version__)'
