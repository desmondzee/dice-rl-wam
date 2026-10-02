#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] || { echo "copy brev/env.example to brev/env.sh and fill it in" >&2; exit 1; }
source "$ENV_FILE"
RUN="${1:?usage: train.sh <run-name> <task-id> [--residual-input z|base|z_base] [--max-env-steps N] [--no-tmux]}"
TASK="${2:?usage: train.sh <run-name> <task-id> [--residual-input z|base|z_base] [--max-env-steps N] [--no-tmux]}"
shift 2
MAX_ENV_STEPS=""
NO_TMUX=0
RESIDUAL_INPUT=z
while [ $# -gt 0 ]; do
  case "$1" in
    --residual-input) RESIDUAL_INPUT="$2"; shift 2 ;;
    --max-env-steps) MAX_ENV_STEPS="$2"; shift 2 ;;
    --no-tmux) NO_TMUX=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/ephemeral/dice}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${MODAL_PROFILE:=desmond-zee}"
PY="$DICE_DATA/lerobot/.venv/bin/python"
MODAL="$DICE_DATA/lerobot/.venv/bin/modal"
PREPARED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prepared_path", sys.argv[1]))' "$DICE_DATA/prepared.json")"
OUT="$DICE_DATA/runs/$RUN"
mkdir -p "$OUT"
python3 -c 'import json,sys; print(json.dumps({"task_ids": [int(sys.argv[1])], "wandb_entity": sys.argv[2], "residual_input": sys.argv[3]}))' "$TASK" "$WANDB_ENTITY" "$RESIDUAL_INPUT" > "$OUT/config.json"
ARGS=(--prepared-path "$PREPARED" --output-dir "$OUT" --run-name "$RUN" --dataset-root "$(cat "$DICE_DATA/dataset_root")")
if [ ! -f "$OUT/resume/latest.pt" ] && env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume ls dice-lingbot-rl-runs "$RUN/resume" >/dev/null 2>&1; then
  mkdir -p "$OUT/resume"
  env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume get dice-lingbot-rl-runs "$RUN/resume/latest.pt" "$OUT/resume/" --force
fi
[ -f "$OUT/resume/latest.pt" ] && ARGS+=(--resume)
[ -n "$MAX_ENV_STEPS" ] && ARGS+=(--max-env-steps "$MAX_ENV_STEPS")
ARGS+=(--checkpoint-hook "$DICE_REPO/brev/sync.sh $RUN $OUT")
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "source '$ENV_FILE'"
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot' MODAL_PROFILE='$MODAL_PROFILE'"
  echo "cd '$DICE_REPO'"
  printf 'exec %q -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$PY" "$OUT/config.json"
  printf ' %q' "${ARGS[@]}"
  printf ' 2>&1 | tee -a %q\n' "$OUT/train.log"
} > "$OUT/run.sh"
chmod +x "$OUT/run.sh"
if [ "$NO_TMUX" = "1" ]; then
  exec bash "$OUT/run.sh"
fi
tmux has-session -t "dice-$RUN" 2>/dev/null && { echo "session dice-$RUN already running" >&2; exit 1; }
tmux new-session -d -s "dice-$RUN" "bash $OUT/run.sh"
echo "started dice-$RUN; attach: tmux attach -t dice-$RUN; log: $OUT/train.log"
