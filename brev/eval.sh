#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] || { echo "copy brev/env.example to brev/env.sh and fill it in" >&2; exit 1; }
source "$ENV_FILE"
USAGE="usage: eval.sh <sft|run-name> <step> <candidates> <shard> <shards> [--video-steps V] [--video-exec-step E] [--action-steps A] [--video-guidance G] [--residual-input z|base|z_base] [--plans M] [--plan-ranker v_plan|q|random] [--task 9] [--no-tmux]"
RUN="${1:?$USAGE}"
STEP="${2:?$USAGE}"
K="${3:?$USAGE}"
SHARD="${4:?$USAGE}"
SHARDS="${5:?$USAGE}"
shift 5
VIDEO_STEPS=20
VIDEO_EXEC_STEP=-1
ACTION_STEPS=50
VIDEO_GUIDANCE=5
RESIDUAL_INPUT=z
PLANS=1
RANKER=v_plan
TASK=9
NO_TMUX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --video-steps) VIDEO_STEPS="$2"; shift 2 ;;
    --video-exec-step) VIDEO_EXEC_STEP="$2"; shift 2 ;;
    --action-steps) ACTION_STEPS="$2"; shift 2 ;;
    --video-guidance) VIDEO_GUIDANCE="$2"; shift 2 ;;
    --residual-input) RESIDUAL_INPUT="$2"; shift 2 ;;
    --plans) PLANS="$2"; shift 2 ;;
    --plan-ranker) RANKER="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --no-tmux) NO_TMUX=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
for STEPS in "$VIDEO_STEPS" "$ACTION_STEPS"; do
  case "$STEPS" in [1-9]|[1-4][0-9]|50) ;; *) echo "--video-steps and --action-steps must be in [1, 50]" >&2; exit 2 ;; esac
done
case "$VIDEO_EXEC_STEP" in
  -1) ;;
  [1-9]|[1-4][0-9]|50) [ "$VIDEO_EXEC_STEP" -le "$VIDEO_STEPS" ] || { echo "--video-exec-step must be -1 or in [1, --video-steps]" >&2; exit 2; } ;;
  *) echo "--video-exec-step must be -1 or in [1, --video-steps]" >&2; exit 2 ;;
esac
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/ephemeral/dice}"
PY="$DICE_DATA/lerobot/.venv/bin/python"
PREPARED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prepared_path", sys.argv[1]))' "${DICE_PREPARED:-$DICE_DATA/prepared.json}")"
STEP6="$(printf '%06d' "$STEP")"
TT="$(printf '%02d' "$TASK")"
SUFFIX=""
[ "$VIDEO_STEPS" = 20 ] && [ "$VIDEO_EXEC_STEP" = -1 ] && [ "$ACTION_STEPS" = 50 ] || SUFFIX="-v${VIDEO_STEPS}e${VIDEO_EXEC_STEP}a${ACTION_STEPS}"
[ "$VIDEO_GUIDANCE" = 5 ] || SUFFIX="$SUFFIX-g${VIDEO_GUIDANCE}"
if [ "$RUN" = "sft" ]; then
  POLICY="sft-step$STEP6$SUFFIX"
  [ -f "$DICE_DATA/sft/libero30-sft/checkpoints/step_$STEP6/norm_stats.json" ] || { echo "missing SFT checkpoint step_$STEP6 under $DICE_DATA/sft" >&2; exit 1; }
else
  POLICY="$RUN-step$STEP6-k$K$SUFFIX"
  [ "$PLANS" = 1 ] || POLICY="$RUN-step$STEP6-k$K-m$PLANS$SUFFIX"
  [ "$RANKER" = v_plan ] || POLICY="$POLICY-${RANKER}rank"
  RESIDUAL="$DICE_DATA/runs/$RUN/checkpoints/step_$STEP6/residual.pt"
  [ -f "$RESIDUAL" ] || { echo "missing $RESIDUAL" >&2; exit 1; }
fi
OUT="$DICE_DATA/heldout/$POLICY/task_$TT"
mkdir -p "$OUT"
if [ "$RUN" = "sft" ]; then
  CONFIG="$(python3 -c 'import json,sys; print(json.dumps({"source_run": "libero30-sft", "checkpoint_step": int(sys.argv[1]), "stage": "heldout", "task_ids": [int(sys.argv[2])], "shard": int(sys.argv[3]), "shards": int(sys.argv[4]), "video_exec_step": int(sys.argv[5]), "video_steps": int(sys.argv[6]), "action_steps": int(sys.argv[7]), "video_guidance": float(sys.argv[8])}))' "$STEP" "$TASK" "$SHARD" "$SHARDS" "$VIDEO_EXEC_STEP" "$VIDEO_STEPS" "$ACTION_STEPS" "$VIDEO_GUIDANCE")"
  ARGS=(-m script.lingbot_eval run --config-json "$CONFIG" --prepared-path "$PREPARED" --output-dir "$OUT" --run-name "$POLICY-task$TT")
else
  CONFIG="$(python3 -c 'import json,sys; print(json.dumps({"task_ids": [int(sys.argv[1])], "residual_input": sys.argv[2], "video_exec_step": int(sys.argv[3]), "video_steps": int(sys.argv[4]), "action_steps": int(sys.argv[5]), "plan_candidates": int(sys.argv[6])}))' "$TASK" "$RESIDUAL_INPUT" "$VIDEO_EXEC_STEP" "$VIDEO_STEPS" "$ACTION_STEPS" "$PLANS")"
  ARGS=(-m script.lingbot_rl_train eval --config-json "$CONFIG" --prepared-path "$PREPARED" --output-dir "$OUT" --residual-path "$RESIDUAL" --eval-candidates "$K" --shard "$SHARD" --shards "$SHARDS" --plan-ranker "$RANKER")
fi
[ -f "$OUT/settings/shard_$SHARD.json" ] && ARGS+=(--resume)
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "source '$ENV_FILE'"
  echo "export PYTHONPATH='$DICE_REPO' HF_HOME='$DICE_DATA/cache/hub' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH='$DICE_DATA/libero-config' TOKENIZERS_PARALLELISM=false"
  echo "export LEROBOT_SOURCE_ROOT='$DICE_DATA/lerobot'"
  echo "cd '$DICE_REPO'"
  printf 'exec %q' "$PY"
  printf ' %q' "${ARGS[@]}"
  printf ' 2>&1 | tee -a %q\n' "$OUT/shard_$SHARD.log"
} > "$OUT/run_shard_$SHARD.sh"
chmod +x "$OUT/run_shard_$SHARD.sh"
if [ "$NO_TMUX" = "1" ]; then
  exec bash "$OUT/run_shard_$SHARD.sh"
fi
SESSION="dice-eval-$POLICY-t$TT-s$SHARD"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session $SESSION already running" >&2; exit 1; }
tmux new-session -d -s "$SESSION" "bash $OUT/run_shard_$SHARD.sh"
echo "started $SESSION; attach: tmux attach -t $SESSION; log: $OUT/shard_$SHARD.log"
