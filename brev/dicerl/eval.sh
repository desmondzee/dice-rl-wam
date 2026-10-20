#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: eval.sh <finetuned checkpoint .pth> [episodes-per-env=10] [envs=30]}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PYTHONPATH"
WANDB_MODE=disabled "$PY" script/eval_rl_checkpoint.py --ckpt_path "$CKPT" --num_eval_episodes "${2:-10}" --eval_n_envs "${3:-30}"
