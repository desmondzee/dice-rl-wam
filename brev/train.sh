#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
RUN="${1:?usage: train.sh <run-name> <task-id> [--max-env-steps N] [--no-tmux] [--fresh]}"
TASK="${2:?usage: train.sh <run-name> <task-id> [--max-env-steps N] [--no-tmux] [--fresh]}"
shift 2
MAX_ENV_STEPS=""
NO_TMUX=0
FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-env-steps) MAX_ENV_STEPS="$2"; shift 2 ;;
    --no-tmux) NO_TMUX=1; shift ;;
    --fresh) FRESH=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/ephemeral/dice}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${MODAL_PROFILE:=desmond-zee}"
DRY="${DICE_DRY_RUN:-0}"
PY="$DICE_DATA/lerobot/.venv/bin/python"
MODAL="$DICE_DATA/lerobot/.venv/bin/modal"
PREPARED_INDEX="$DICE_DATA/prepared.json"
[ -f "$PREPARED_INDEX" ] || { echo "missing $PREPARED_INDEX; run brev/setup.sh first" >&2; exit 1; }
PREPARED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prepared_path", sys.argv[1]))' "$PREPARED_INDEX")"
DATASET_ROOT="$(cat "$DICE_DATA/dataset_root")"
OUT="$DICE_DATA/runs/$RUN"
mkdir -p "$OUT"
ENTITY="${WANDB_ENTITY:-}"
python3 -c 'import json,sys; print(json.dumps({"task_ids": [int(sys.argv[1])], "wandb_entity": sys.argv[2] or None}))' "$TASK" "$ENTITY" > "$OUT/config.json"
ARGS=(--prepared-path "$PREPARED" --output-dir "$OUT" --run-name "$RUN" --dataset-root "$DATASET_ROOT")
if [ ! -f "$OUT/resume/latest.pt" ] && [ "$FRESH" = 0 ] && [ -n "${MODAL_PROFILE:-}" ]; then
  if [ "$DRY" = "1" ]; then
    echo "env MODAL_PROFILE=$MODAL_PROFILE $MODAL volume ls dice-lingbot-rl-runs $RUN/resume"
  elif env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume ls dice-lingbot-rl-runs "$RUN/resume" >/dev/null 2>&1; then
    mkdir -p "$OUT/resume"
    env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume get dice-lingbot-rl-runs "$RUN/resume/latest.pt" "$OUT/resume/" --force
  fi
fi
[ -f "$OUT/resume/latest.pt" ] && ARGS+=(--resume)
[ -n "$MAX_ENV_STEPS" ] && ARGS+=(--max-env-steps "$MAX_ENV_STEPS")
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "[ -f '$DICE_REPO/brev/env.sh' ] && source '$DICE_REPO/brev/env.sh'"
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot' DICE_DATA='$DICE_DATA' MODAL_PROFILE='$MODAL_PROFILE' DICE_SYNC_CMD='$DICE_REPO/brev/sync.sh $RUN'"
  echo "cd '$DICE_REPO'"
  printf 'exec %q -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$PY" "$OUT/config.json"
  printf ' %q' "${ARGS[@]}"
  printf ' 2>&1 | tee -a %q\n' "$OUT/train.log"
} > "$OUT/run.sh"
chmod +x "$OUT/run.sh"
if [ "$DRY" = "1" ]; then
  cat "$OUT/config.json" "$OUT/run.sh"
  if [ "$NO_TMUX" = "1" ]; then
    echo "bash $OUT/run.sh"
  else
    echo "tmux new-session -d -s dice-$RUN bash $OUT/run.sh"
  fi
  exit 0
fi
if [ "$NO_TMUX" = "1" ]; then
  exec bash "$OUT/run.sh"
fi
if tmux has-session -t "dice-$RUN" 2>/dev/null; then
  echo "session dice-$RUN already running; attach with: tmux attach -t dice-$RUN" >&2
  exit 1
fi
tmux new-session -d -s "dice-$RUN" "bash $OUT/run.sh"
echo "started dice-$RUN; attach with: tmux attach -t dice-$RUN; log: $OUT/train.log"
