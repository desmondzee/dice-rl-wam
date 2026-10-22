#!/usr/bin/env bash
set -euo pipefail
TASK="${1:?usage: finetune_libero.sh <task> [seed=42] [extra hydra overrides...]}"
SEED="${2:-42}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA="$*"
EVERY="${WATCH_EVERY:-2000}"
source "$(dirname "${BASH_SOURCE[0]}")/env_libero.sh"
: "${DICE_RL_WANDB_ENTITY:?set DICE_RL_WANDB_ENTITY}"
cd "$PYTHONPATH"
SHORT="$(echo "$TASK" | cut -d_ -f1,2 | tr -d _ | tr 'A-Z' 'a-z')"
TAG="$(date +%Y%m%d-%H%M%S)"
RL="dicerl-rl-$SHORT-$SEED"
WATCH="dicerl-rlwatch-$SEED"
LOG="$DICE_RL_LOG_DIR/finetune_${SHORT}_${SEED}_${TAG}.log"
STATES="$DICE_RL_DATA_DIR/libero/$TASK/plus_robot_init_monitor.npz"
for s in "$RL" "$WATCH"; do
  tmux has-session -t "$s" 2>/dev/null && { echo "session $s already running" >&2; exit 1; }
done
[ -f "$STATES" ] || { echo "missing monitoring states $STATES" >&2; exit 1; }
mkdir -p "$DICE_RL_LOG_DIR/libero-finetune"
STAMP="$(mktemp)"
tmux new-session -d -s "$RL" "$PY script/run.py --config-name=ft_distill_residual_flow_unet_img --config-dir=brev/dicerl/cfg/libero/$TASK seed=$SEED $EXTRA 2>&1 | tee -a $LOG"
echo "started $RL; log $LOG"
RUN_DIR=""
for _ in $(seq 120); do
  RUN_DIR="$(find "$DICE_RL_LOG_DIR/libero-finetune" -mindepth 2 -maxdepth 2 -type d -name "*_$SEED" -newer "$STAMP" 2>/dev/null | head -n 1)"
  [ -n "$RUN_DIR" ] && break
  tmux has-session -t "$RL" 2>/dev/null || { echo "$RL exited before creating a run directory; see $LOG" >&2; exit 1; }
  sleep 5
done
rm -f "$STAMP"
[ -n "$RUN_DIR" ] || { echo "run directory did not appear" >&2; exit 1; }
tmux new-session -d -s "$WATCH" "$PY -m script.dicerl_libero_rl_watch --run-dir $RUN_DIR --every $EVERY --states $STATES --envs 25 --session $RL 2>&1 | tee -a $DICE_RL_LOG_DIR/rlwatch_${SHORT}_${SEED}_${TAG}.log"
echo "started $WATCH on $RUN_DIR"
