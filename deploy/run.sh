#!/usr/bin/env bash
# usage: [HEADLESS=1] ./deploy/run.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BODY_ARGS=(); BRAIN_ARGS=(--debug)
if [ "${HEADLESS:-0}" = "1" ]; then BODY_ARGS=(--headless); BRAIN_ARGS=(); fi
[ -z "${ANTHROPIC_API_KEY:-}" ] && echo "no ANTHROPIC_API_KEY, running in local mode"

uv run python -m body.main "${BODY_ARGS[@]}" &
BODY=$!
trap 'kill $BODY 2>/dev/null; wait $BODY 2>/dev/null' EXIT INT TERM
sleep 3      # the brain reconnects anyway, this just keeps the startup log tidy
uv run python -u -m brain.main "${BRAIN_ARGS[@]}"
