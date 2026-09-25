#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: pull.sh <run-name>}"
: "${MODAL_PROFILE:=desmond-zee}"
mkdir -p result/runs
env MODAL_PROFILE="$MODAL_PROFILE" uv run --no-project --with modal==1.1.4 modal volume get dice-lingbot-rl-runs "$RUN" result/runs/ --force
echo "pulled $RUN into result/runs/$RUN"
