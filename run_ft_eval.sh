#!/usr/bin/env bash
# Eval a proxy-served model on calib60 + test60.
# Usage: ./run_ft_eval.sh <name> <served_model> <port> <in_cost_per_tok> <out_cost_per_tok>
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME=$1; MODEL=$2; PORT=$3; IC=$4; OC=$5
set -a; . "$ROOT/.env"; set +a
cd "$ROOT/AutomationBench"
split() { python3 -c "import json;print(','.join(json.load(open('$ROOT/runs/splits.json'))['$1']))"; }
for S in test calib; do
  T=$(split ${S}60)
  AB_INCREMENTAL_JSONL=$ROOT/runs/${S}_${NAME}.jsonl uv run auto-bench --model "$MODEL" --base-url http://127.0.0.1:$PORT/v1 --api-key-var FIREWORKS_API_KEY \
    --toolset api --tasks "$T" --max-concurrent 10 --save-every 5 --input-cost $IC --output-cost $OC \
    --export-json ../runs/${S}_${NAME}.json > ../runs/${S}_${NAME}.log 2>&1 &
done
wait
echo DONE_$NAME
