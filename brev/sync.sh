#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: sync.sh <run-name> <output-dir>}"
OUT="${2:?usage: sync.sh <run-name> <output-dir>}"
: "${DICE_DATA:=/ephemeral/dice}"
: "${DICE_STEP:?DICE_STEP is set by the training hook}"
: "${MODAL_PROFILE:=desmond-zee}"
export MODAL_PROFILE
MODAL="$DICE_DATA/lerobot/.venv/bin/modal"
put() { "$MODAL" volume put dice-lingbot-rl-runs "$1" "$RUN/$2" --force; }
STEP_DIR="checkpoints/step_$(printf '%06d' "$DICE_STEP")"
[ -d "$OUT/$STEP_DIR" ] && put "$OUT/$STEP_DIR" "$STEP_DIR"
for name in settings.json resume/latest.pt residual.pt summary.json train.log; do
  [ -f "$OUT/$name" ] && put "$OUT/$name" "$name"
done
echo "sync done: step $DICE_STEP"
