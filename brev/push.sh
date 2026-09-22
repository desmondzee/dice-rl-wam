#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
HOST="${1:?usage: push.sh <ssh-host>}"
: "${DICE_REMOTE_REPO:=dice-rl-wam}"
REPO="$(cd "$HERE/.." && pwd)"
run() { if [ "${DICE_DRY_RUN:-0}" = "1" ]; then echo "$*"; else "$@"; fi; }
run rsync -az --delete --exclude .venv --exclude .cache --exclude checkpoints --exclude result --exclude .git \
  --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache --exclude .superpowers --exclude brev/env.sh \
  "$REPO/" "$HOST:$DICE_REMOTE_REPO/"
