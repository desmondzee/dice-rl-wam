DICE_DATA="${DICE_DATA:-$HOME/Documents/dice}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO/brev/env.sh" ] && source "$REPO/brev/env.sh"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DICE_RL_DATA_DIR="$DICE_DATA/dicerl/data_dir" DICE_RL_LOG_DIR="$DICE_DATA/dicerl/log_dir"
export DICE_RL_WANDB_ENTITY="${DICE_RL_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
export PYTHONPATH="$REPO"
PY="$DICE_DATA/dicerl-venv/bin/python"
