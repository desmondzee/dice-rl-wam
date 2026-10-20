#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:?usage: finetune.sh <square|tool_hang> [seed=42] [--no-tmux]}"
SEED="${2:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${DICE_RL_WANDB_ENTITY:?set DICE_RL_WANDB_ENTITY}"
cd "$PYTHONPATH"
CMD="$PY script/run.py --config-name=ft_distill_residual_flow_mlp --config-dir=brev/dicerl/cfg/$ENV_NAME seed=$SEED"
if [ "${3:-}" = "--no-tmux" ]; then exec $CMD; fi
tmux has-session -t "dicerl-$ENV_NAME-$SEED" 2>/dev/null && { echo "session dicerl-$ENV_NAME-$SEED already running" >&2; exit 1; }
tmux new-session -d -s "dicerl-$ENV_NAME-$SEED" "$CMD 2>&1 | tee -a $DICE_RL_LOG_DIR/finetune_${ENV_NAME}_${SEED}.log"
echo "started dicerl-$ENV_NAME-$SEED; log $DICE_RL_LOG_DIR/finetune_${ENV_NAME}_${SEED}.log"
