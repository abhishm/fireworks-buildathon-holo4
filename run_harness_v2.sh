#!/usr/bin/env bash
# Harness v2 on a split. Usage: ./run_harness_v2.sh <calib60|test60> <KEY_VAR> [concurrency]. Resumable.
set -u
SPLIT=$1; KEYVAR=$2; CONC=${3:-50}
S=${SPLIT%60}; NAME=${S}_ds_harness_v2
ROOT="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$ROOT/.env"; set +a
cd "$ROOT/AutomationBench"
TASKS=$(python3 -c "
import json,os
t=json.load(open('$ROOT/runs/splits.json'))['$SPLIT']
p='$ROOT/runs/$NAME.jsonl'
done={json.loads(l)['task'] for l in open(p)} if os.path.exists(p) else set()
print(','.join(x for x in t if x not in done))")
[ -z "$TASKS" ] && { echo "NOTHING_TO_DO"; exit 0; }
HARNESS_VERSION=2 AB_INCREMENTAL_JSONL=$ROOT/runs/$NAME.jsonl uv run python ../harness/run.py \
  --model accounts/fireworks/models/deepseek-v4p1-flash --base-url https://api.fireworks.ai/inference/v1 \
  --api-key-var $KEYVAR --tasks "$TASKS" --max-concurrent $CONC --input-cost 0.0000003 --output-cost 0.0000012 \
  --export-json ../runs/$NAME.json >> ../runs/$NAME.log 2>&1
echo "DONE $NAME"
