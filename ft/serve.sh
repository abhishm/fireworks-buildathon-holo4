#!/usr/bin/env bash
# Usage: ft/serve.sh <qwen|nemotron> [sampler_path|base] [port] [extra proxy.py flags...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PRESET="${1:?usage: serve.sh <qwen|nemotron> [sampler_path|base] [port] [extra flags]}"
SAMPLER="${2:-base}"
PORT="${3:-$([ "$PRESET" = qwen ] && echo 8301 || echo 8302)}"
shift $(( $# < 3 ? $# : 3 ))
set -a; . "$ROOT/.env"; set +a
# USE_OLD_KEY=1 -> use the original account that holds the full-data fine-tunes.
if [ -n "${USE_OLD_KEY:-}" ]; then export FIREWORKS_API_KEY="$FIREWORKS_API_KEY_OLD"; fi
ARGS=(--preset "$PRESET" --port "$PORT")
# Allow a key from ft/checkpoints.json (e.g. "qwen", "nemotron_dry") instead of a raw sampler path.
if [ "$SAMPLER" != "base" ] && [ -f "$HERE/checkpoints.json" ]; then
  RESOLVED="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); e=d.get(sys.argv[2]); print("" if e is None else (e.get("sampler_path") or "MISSING"))' "$HERE/checkpoints.json" "$SAMPLER")"
  if [ "$RESOLVED" = MISSING ]; then echo "checkpoints.json[$SAMPLER] has no sampler_path yet" >&2; exit 1; fi
  if [ -n "$RESOLVED" ]; then echo "checkpoints.json[$SAMPLER] -> $RESOLVED" >&2; SAMPLER="$RESOLVED"; fi
fi
if [ "$SAMPLER" != "base" ] && [ -n "$SAMPLER" ]; then
  ARGS+=(--sampler-path "$SAMPLER")
  NAME="${PRESET}_${SAMPLER##*/}"
else
  NAME="${PRESET}_base"
fi
ARGS+=(--served-name "$NAME")
exec "$HERE/.venv/bin/python" "$HERE/proxy.py" "${ARGS[@]}" "$@"
