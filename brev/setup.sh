#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
: "${DICE_REPO:=$(cd "$HERE/.." && pwd)}"
: "${DICE_DATA:=/ephemeral/dice}"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${MODAL_PROFILE:=desmond-zee}"
: "${MODAL_SFT_PROFILE:=source}"
DRY="${DICE_DRY_RUN:-0}"
run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }
LEROBOT="$DICE_DATA/lerobot"
PY="$LEROBOT/.venv/bin/python"
MODAL="$LEROBOT/.venv/bin/modal"
SFT="$DICE_DATA/sft/libero30-sft/checkpoints/step_000600"
REVISION="$(grep -o 'LEROBOT_REVISION = "[0-9a-f]*"' "$DICE_REPO/script/lingbot_eval_config.py" | cut -d'"' -f2)"
DATASET_REPO="$(grep -o 'DATASET_REPO = "[^"]*"' "$DICE_REPO/script/lingbot_sft_config.py" | cut -d'"' -f2)"
DATASET_REVISION="$(grep -o 'DATASET_REVISION = "[0-9a-f]*"' "$DICE_REPO/script/lingbot_sft_config.py" | cut -d'"' -f2)"
export PATH="$HOME/.local/bin:$PATH"
export LEROBOT_SOURCE_ROOT="$LEROBOT" HF_HOME="$DICE_DATA/cache/hub" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$DICE_DATA/libero-config" PYTHONPATH="$DICE_REPO" TOKENIZERS_PARALLELISM=false
run mkdir -p "$DICE_DATA/cache/hub" "$DICE_DATA/runs" "$DICE_DATA/sft/libero30-sft/checkpoints"
if command -v apt-get >/dev/null 2>&1; then
  run sudo apt-get update -y
  run sudo apt-get install -y git ffmpeg libgl1 libegl1 libegl1-mesa-dev libgl1-mesa-dev libglib2.0-0 libglvnd0 libgles2 build-essential cmake tmux rsync curl
fi
if ! command -v uv >/dev/null 2>&1; then
  run bash -c "curl -LsSf https://astral.sh/uv/0.8.18/install.sh | sh"
fi
if [ ! -d "$LEROBOT/.git" ]; then
  run git clone https://github.com/huggingface/lerobot.git "$LEROBOT"
fi
run git -C "$LEROBOT" checkout "$REVISION"
run uv sync --index-url https://pypi.org/simple --project "$LEROBOT" --python 3.12 --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-editable
run uv export --index-url https://pypi.org/simple --project "$LEROBOT" --locked --no-default-groups --extra lingbot_va --extra libero --extra evaluation --no-emit-project --no-hashes --output-file "$DICE_DATA/lingbot-eval-deps.txt"
run uv pip install --index-url https://pypi.org/simple --python "$PY" --constraint "$DICE_DATA/lingbot-eval-deps.txt" --exclude-newer 2026-09-05T00:00:00Z modal==1.1.4
for profile in "$MODAL_SFT_PROFILE" "$MODAL_PROFILE"; do
  if ! grep -q "^\[$profile\]" "$HOME/.modal.toml" 2>/dev/null; then
    echo "Modal profile '$profile' missing on this box; run: $MODAL token new --profile $profile --no-verify and approve it in the browser" >&2
    exit 1
  fi
done
if [ ! -f "$SFT/norm_stats.json" ]; then
  run env MODAL_PROFILE="$MODAL_SFT_PROFILE" "$MODAL" volume get dice-lingbot-sft-runs libero30-sft/checkpoints/step_000600 "$DICE_DATA/sft/libero30-sft/checkpoints/"
fi
run env MODAL_PROFILE="$MODAL_PROFILE" "$MODAL" volume create dice-lingbot-rl-runs || true
run "$PY" -m script.lingbot_eval prepare --config-json '{"source_run":"libero30-sft","checkpoint_step":600,"stage":"eval","seed":42}' --cache-root "$DICE_DATA/cache" --checkpoint "$SFT" --prepared-output "$DICE_DATA/prepared.json"
run "$PY" -c "import os; from script.lingbot_eval import download_snapshot; download_snapshot('$DATASET_REPO', repo_type='dataset', revision='$DATASET_REVISION', cache_dir='$DICE_DATA/cache/hub', token=os.environ['HF_TOKEN'])"
run bash -c "echo '$DICE_DATA/cache/hub/datasets--${DATASET_REPO//\//--}/snapshots/$DATASET_REVISION' > '$DICE_DATA/dataset_root'"
run "$PY" -c "from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy; import wandb, modal"
run "$PY" -m script.lingbot_rl_config
run "$PY" -m script.lingbot_rl_train check-inits --task-id 0 --prepared-path "$(cat "$DICE_DATA/prepared.json" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("prepared_path",""))' 2>/dev/null || true)"
echo "setup complete: $DICE_DATA"
