#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
RUN="${1:?usage: sync.sh <run-name>}"
: "${DICE_DATA:=/ephemeral/dice}"
: "${DICE_OUTPUT_DIR:?DICE_OUTPUT_DIR is set by the training hook}"
: "${DICE_STEP:?DICE_STEP is set by the training hook}"
DRY="${DICE_DRY_RUN:-0}"
MODAL="$DICE_DATA/lerobot/.venv/bin/modal"
run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }
if [ -z "${MODAL_PROFILE:-}" ]; then
  echo "sync skip: MODAL_PROFILE unset (step $DICE_STEP)"
  exit 0
fi
export MODAL_PROFILE
put() { run "$MODAL" volume put dice-lingbot-rl-runs "$1" "$RUN/$2" --force; }
STEP_DIR="train_eval/step_$(printf '%06d' "$DICE_STEP")"
[ -d "$DICE_OUTPUT_DIR/$STEP_DIR" ] && put "$DICE_OUTPUT_DIR/$STEP_DIR" "$STEP_DIR"
[ -f "$DICE_OUTPUT_DIR/settings.json" ] && put "$DICE_OUTPUT_DIR/settings.json" settings.json
if [ -f "$DICE_OUTPUT_DIR/summary.json" ]; then
  for name in residual.pt resume/latest.pt summary.json train.log; do
    [ -f "$DICE_OUTPUT_DIR/$name" ] && put "$DICE_OUTPUT_DIR/$name" "$name"
  done
fi
echo "sync done: step $DICE_STEP"
