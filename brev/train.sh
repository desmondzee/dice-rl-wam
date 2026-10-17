#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] || { echo "copy brev/env.example to brev/env.sh and fill it in" >&2; exit 1; }
source "$ENV_FILE"
RUN="${1:?usage: train.sh <run-name> <task-id> [--residual-input z|base|z_base] [--epsilon E] [--env-steps N] [--video-steps V] [--video-exec-step E] [--action-steps A] [--plan-candidates M] [--max-env-steps N] [--no-tmux] [--state-features pool|struct] [--proprio] [--plus] [--plan-target q|rollout] [--store-tokens] [--label-workers N]}"
TASK="${2:?usage: train.sh <run-name> <task-id> [--residual-input z|base|z_base] [--epsilon E] [--env-steps N] [--video-steps V] [--video-exec-step E] [--action-steps A] [--plan-candidates M] [--max-env-steps N] [--no-tmux] [--state-features pool|struct] [--proprio] [--plus] [--plan-target q|rollout] [--store-tokens] [--label-workers N]}"
shift 2
MAX_ENV_STEPS=""
NO_TMUX=0
RESIDUAL_INPUT=z
EPSILON=-0.5
ENV_STEPS=660000
VIDEO_STEPS=20
VIDEO_EXEC_STEP=-1
ACTION_STEPS=50
PLAN_CANDIDATES=1
STATE_FEATURES=pool
PROPRIO=0
PLUS=0
PLAN_TARGET=q
STORE_TOKENS=0
LABEL_WORKERS=4
while [ $# -gt 0 ]; do
  case "$1" in
    --residual-input) RESIDUAL_INPUT="$2"; shift 2 ;;
    --epsilon) EPSILON="$2"; shift 2 ;;
    --env-steps) ENV_STEPS="$2"; shift 2 ;;
    --video-steps) VIDEO_STEPS="$2"; shift 2 ;;
    --video-exec-step) VIDEO_EXEC_STEP="$2"; shift 2 ;;
    --action-steps) ACTION_STEPS="$2"; shift 2 ;;
    --plan-candidates) PLAN_CANDIDATES="$2"; shift 2 ;;
    --state-features) STATE_FEATURES="$2"; shift 2 ;;
    --proprio) PROPRIO=1; shift ;;
    --plus) PLUS=1; shift ;;
    --plan-target) PLAN_TARGET="$2"; shift 2 ;;
    --store-tokens) STORE_TOKENS=1; shift ;;
    --label-workers) LABEL_WORKERS="$2"; shift 2 ;;
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
PREPARED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prepared_path", sys.argv[1]))' "${DICE_PREPARED:-$DICE_DATA/prepared.json}")"
OUT="$DICE_DATA/runs/$RUN"
mkdir -p "$OUT"
PLUS_TRAIN='{}'
[ "$PLUS" = 1 ] && PLUS_TRAIN="$(PYTHONPATH="$DICE_REPO" "$PY" -c 'import json; from script.lingbot_rl_config import plus_train_variants; print(json.dumps({axis: list(variants) for axis, variants in plus_train_variants().items()}))')"
python3 -c 'import json,sys; print(json.dumps({"task_ids": [int(sys.argv[1])], "wandb_entity": sys.argv[2], "residual_input": sys.argv[3], "epsilon": float(sys.argv[4]), "online_env_steps": int(sys.argv[5]), "video_exec_step": int(sys.argv[6]), "video_steps": int(sys.argv[7]), "action_steps": int(sys.argv[8]), "plan_candidates": int(sys.argv[9]), "state_features": sys.argv[10], "proprio": sys.argv[11] == "1", "plus_train": json.loads(sys.argv[12]), "plan_target": sys.argv[13], "store_tokens": sys.argv[14] == "1", "label_workers": int(sys.argv[15])}))' "$TASK" "$WANDB_ENTITY" "$RESIDUAL_INPUT" "$EPSILON" "$ENV_STEPS" "$VIDEO_EXEC_STEP" "$VIDEO_STEPS" "$ACTION_STEPS" "$PLAN_CANDIDATES" "$STATE_FEATURES" "$PROPRIO" "$PLUS_TRAIN" "$PLAN_TARGET" "$STORE_TOKENS" "$LABEL_WORKERS" > "$OUT/config.json"
ARGS=(--prepared-path "$PREPARED" --output-dir "$OUT" --run-name "$RUN" --dataset-root "$(cat "$DICE_DATA/dataset_root")")
if [ "${DICE_NO_SYNC:-0}" != "1" ] && [ ! -f "$OUT/resume/latest.pt" ] && env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume ls dice-lingbot-rl-runs "$RUN/resume" >/dev/null 2>&1; then
  mkdir -p "$OUT/resume"
  env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume get dice-lingbot-rl-runs "$RUN/resume/latest.pt" "$OUT/resume/" --force
fi
[ -f "$OUT/resume/latest.pt" ] && ARGS+=(--resume)
[ -n "$MAX_ENV_STEPS" ] && ARGS+=(--max-env-steps "$MAX_ENV_STEPS")
[ "${DICE_NO_SYNC:-0}" = "1" ] || ARGS+=(--checkpoint-hook "$DICE_REPO/brev/sync.sh $RUN $OUT")
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "source '$ENV_FILE'"
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot' MODAL_PROFILE='$MODAL_PROFILE'"
  if [ "$PLUS" = 1 ]; then
    echo "source '$DICE_DATA/libero-plus-config/activate.sh'"
    echo "export PYTHONPATH='$DICE_REPO':\"\$PYTHONPATH\""
  fi
  echo "cd '$DICE_REPO'"
  if [ "$PLUS" = 1 ]; then
    printf 'exec "$PY" -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$OUT/config.json"
  else
    printf 'exec %q -m script.lingbot_rl_train train --config-json "$(cat %q)"' "$PY" "$OUT/config.json"
  fi
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
