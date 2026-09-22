#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DICE_ENV_FILE:-$HERE/env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
RUN="${1:?usage: pull.sh <run-name>}"
: "${MODAL_PROFILE:=desmond-zee}"
DEST="result/brev"
run() { if [ "${DICE_DRY_RUN:-0}" = "1" ]; then echo "$*"; else "$@"; fi; }
run mkdir -p "$DEST"
run env MODAL_PROFILE="$MODAL_PROFILE" uv run --no-project --with modal==1.1.4 modal volume get dice-lingbot-rl-runs "$RUN" "$DEST/" --force
echo "pulled $RUN into $DEST/$RUN"
