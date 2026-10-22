#!/usr/bin/env bash
set -euo pipefail
TASK="${1:?usage: pretrain_libero.sh <task> [seed=42] [--epochs N]}"
SEED="${2:-42}"
EXTRA=""
if [ "${3:-}" = "--epochs" ]; then EXTRA="train.n_epochs=${4:?--epochs needs a value}"; fi
source "$(dirname "${BASH_SOURCE[0]}")/env_libero.sh"
: "${DICE_RL_WANDB_ENTITY:?set DICE_RL_WANDB_ENTITY}"
cd "$PYTHONPATH"
SHORT="$(echo "$TASK" | cut -d_ -f1,2 | tr -d _ | tr 'A-Z' 'a-z')"
PRE="dicerl-pre-$SHORT-$SEED"
WATCH="dicerl-watch-$SEED"
LOG="$DICE_RL_LOG_DIR/pretrain_${SHORT}_${SEED}.log"
for s in "$PRE" "$WATCH"; do
  tmux has-session -t "$s" 2>/dev/null && { echo "session $s already running" >&2; exit 1; }
done
mkdir -p "$DICE_RL_LOG_DIR"
STAMP="$(mktemp)"
tmux new-session -d -s "$PRE" "$PY script/run.py --config-name=pre_flow_matching_unet_img --config-dir=brev/dicerl/cfg/libero/$TASK seed=$SEED $EXTRA 2>&1 | tee -a $LOG"
echo "started $PRE; log $LOG"
RUN_DIR=""
for _ in $(seq 120); do
  RUN_DIR="$(find "$DICE_RL_LOG_DIR/libero-pretrain" -mindepth 2 -maxdepth 2 -type d -name "*_$SEED" -newer "$STAMP" 2>/dev/null | head -n 1)"
  [ -n "$RUN_DIR" ] && break
  tmux has-session -t "$PRE" 2>/dev/null || { echo "$PRE exited before creating a run directory; see $LOG" >&2; exit 1; }
  sleep 5
done
rm -f "$STAMP"
[ -n "$RUN_DIR" ] || { echo "run directory did not appear" >&2; exit 1; }
tmux new-session -d -s "$WATCH" "$PY -m script.dicerl_libero_watch --run-dir $RUN_DIR --every 1000 --envs 25 --session $PRE 2>&1 | tee -a $DICE_RL_LOG_DIR/watch_${SHORT}_${SEED}.log"
echo "started $WATCH on $RUN_DIR"
