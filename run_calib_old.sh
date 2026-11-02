#!/usr/bin/env bash
# calib_ds_harness on the OLD account key (separate rate limit). Resumable.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$ROOT/.env"; set +a
cd "$ROOT/AutomationBench"
TASKS=$(python3 -c "
import json,os
t=json.load(open('$ROOT/runs/splits.json'))['calib60']
p='$ROOT/runs/calib_ds_harness.jsonl'
done={json.loads(l)['task'] for l in open(p)} if os.path.exists(p) else set()
print(','.join(x for x in t if x not in done))")
AB_INCREMENTAL_JSONL=$ROOT/runs/calib_ds_harness.jsonl uv run python ../harness/run.py --model accounts/fireworks/models/deepseek-v4p1-flash \
  --base-url https://api.fireworks.ai/inference/v1 --api-key-var FIREWORKS_API_KEY_OLD \
  --tasks "$TASKS" --max-concurrent 50 --input-cost 0.0000003 --output-cost 0.0000012 \
  --export-json ../runs/calib_ds_harness.json >> ../runs/calib_ds_harness.log 2>&1
echo CALIB_DONE
