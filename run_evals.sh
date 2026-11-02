#!/usr/bin/env bash
# Launch main eval runs in parallel. Usage: ./run_evals.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$ROOT/.env"; set +a
cd "$ROOT/AutomationBench"
# split <name> [jsonl]: tasks in split, minus those already finished in jsonl (resume support)
split() { python3 -c "
import json,os,sys
t=json.load(open('$ROOT/runs/splits.json'))['$1']
done=set()
p='$ROOT/runs/'+'${2:-none}'+'.jsonl'
if os.path.exists(p): done={json.loads(l)['task'] for l in open(p)}
print(','.join(x for x in t if x not in done))"; }
BASE=https://api.fireworks.ai/inference/v1
DS=accounts/fireworks/models/deepseek-v4p1-flash
KIMI=accounts/fireworks/models/kimi-k3

AB_INCREMENTAL_JSONL=$ROOT/runs/test_ds_harness.jsonl uv run python ../harness/run.py --model $DS --base-url $BASE --api-key-var FIREWORKS_API_KEY \
  --tasks "$(split test60 test_ds_harness)" --max-concurrent 50 --input-cost 0.0000003 --output-cost 0.0000012 \
  --export-json ../runs/test_ds_harness.json >> ../runs/test_ds_harness.log 2>&1 &
AB_INCREMENTAL_JSONL=$ROOT/runs/calib_ds_harness.jsonl uv run python ../harness/run.py --model $DS --base-url $BASE --api-key-var FIREWORKS_API_KEY \
  --tasks "$(split calib60 calib_ds_harness)" --max-concurrent 50 --input-cost 0.0000003 --output-cost 0.0000012 \
  --export-json ../runs/calib_ds_harness.json >> ../runs/calib_ds_harness.log 2>&1 &
AB_INCREMENTAL_JSONL=$ROOT/runs/test_ds_base.jsonl uv run auto-bench --model $DS --base-url $BASE --api-key-var FIREWORKS_API_KEY \
  --tasks "$(split test60 test_ds_base)" --max-concurrent 50 --input-cost 0.0000003 --output-cost 0.0000012 \
  --export-json ../runs/test_ds_base.json >> ../runs/test_ds_base.log 2>&1 &
# Kimi K3 test60 already done: runs/test_kimi.json
wait
echo ALL_DONE
