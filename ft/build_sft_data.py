"""Convert Holo4 AutomationBench trajectories (train480 only) into chat SFT rows.

Stage 1 (model-agnostic): data/traj/*.json -> ft/data/all_convs.jsonl
Stage 2 (per model):      render with the cookbook renderer, count tokens,
                          select a budgeted subset -> ft/data/<key>.jsonl
Usage:
  python build_sft_data.py convert
  python build_sft_data.py select --key qwen --budget 3000000
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT = os.path.join(ROOT, "ft")
sys.path.insert(0, os.path.join(FT, "cookbook"))

AB_TOOLS = {"api_search", "api_fetch", "base64_encode"}
DROP_TOOLS = {"update_plan", "note"}

MODELS = {
    "qwen": {
        "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B",
        "renderer": "qwen3_6_interleaved",
        "max_seq_len": 32768,
    },
    "nemotron": {
        "base_model": "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16",
        "tokenizer": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "renderer": "nemotron35_vl",
        "max_seq_len": 32768,
    },
}


def _js(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def map_args(name: str, args: dict) -> dict | None:
    if name == "api_search":
        return {"query": str(args.get("query", "")), "top_k": int(args.get("top_k", 5))}
    if name == "base64_encode":
        return {"text": args.get("text", "")}
    if name == "api_fetch":
        params = args.get("params")
        extra = {k: v for k, v in args.items() if k not in ("method", "url", "params", "body", "method_note")}
        if extra:
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    return None
            params = {**(params or {}), **extra}
        out = {"method": str(args.get("method", "GET")).upper(), "url": args["url"]}
        # Omit null optionals (AB defaults them to None); rendering None would teach a literal "None".
        if params not in (None, "", {}):
            out["params"] = _js(params)
        if args.get("body") not in (None, ""):
            out["body"] = _js(args.get("body"))
        return out
    return None


def convert_one(traj: dict, prompt: list[dict], tools: list[dict]) -> dict | None:
    messages = [dict(m) for m in prompt]
    pending_reasoning: list[str] = []
    n_calls = 0
    final = None
    for si, step in enumerate(traj["steps"]):
        calls = step.get("calls") or []
        results = step.get("results") or []
        thought = (step.get("thought") or "").strip()
        answer = [c for c in calls if c["name"] == "answer"]
        if answer:
            reasoning = "\n\n".join(pending_reasoning + ([thought] if thought else []))
            final = {"role": "assistant", "content": str(answer[0]["args"].get("content", "")).strip(),
                     "reasoning_content": reasoning}
            break
        kept = []
        for ci, c in enumerate(calls):
            if c["name"] in DROP_TOOLS:
                continue
            if c["name"] not in AB_TOOLS:
                return None
            a = map_args(c["name"], c.get("args") or {})
            if a is None or ci >= len(results):
                return None
            kept.append((c["name"], a, results[ci]))
        if not kept:
            if thought:
                pending_reasoning.append(thought)
            continue
        reasoning = "\n\n".join(pending_reasoning + ([thought] if thought else []))
        pending_reasoning = []
        tcs = []
        for k, (name, a, _) in enumerate(kept):
            tcs.append({"id": f"call_{si}_{k}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(a, ensure_ascii=False)}})
        messages.append({"role": "assistant", "content": "", "reasoning_content": reasoning, "tool_calls": tcs})
        for k, (name, _, res) in enumerate(kept):
            messages.append({"role": "tool", "tool_call_id": f"call_{si}_{k}", "name": name,
                             "content": res if isinstance(res, str) else json.dumps(res)})
        n_calls += len(kept)
    if final is None or not final["content"] or n_calls == 0:
        return None
    messages.append(final)
    return {"messages": messages, "tools": tools,
            "meta": {"id": traj["id"], "task": traj["ab_task"], "model": traj["model"], "n_calls": n_calls}}


def cmd_convert(_args):
    sel = json.load(open(os.path.join(FT, "data/selected_index.json")))
    splits = json.load(open(os.path.join(ROOT, "runs/splits.json")))
    train, banned = set(splits["train480"]), set(splits["calib60"]) | set(splits["test60"])
    prompts = json.load(open(os.path.join(FT, "data/ab_prompts.json")))
    tools = json.load(open(os.path.join(FT, "data/ab_tools.json")))
    out, bad = [], 0
    for t in sel:
        assert t["ab_task"] in train and t["ab_task"] not in banned
        d = json.load(open(os.path.join(ROOT, f"data/traj/{t['id']}.json")))
        d["ab_task"] = t["ab_task"]
        row = convert_one(d, prompts[t["ab_task"]], tools)
        if row is None:
            bad += 1
            continue
        out.append(row)
    with open(os.path.join(FT, "data/all_convs.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"converted {len(out)} rows, skipped {bad}, tasks {len({r['meta']['task'] for r in out})}")


def load_renderer(key):
    from training.utils.supervised import build_renderer, load_tokenizer
    m = MODELS[key]
    tok = load_tokenizer(m["tokenizer"])
    return tok, build_renderer(tok, m["tokenizer"], m["renderer"])


def count_tokens(row, renderer):
    from training.utils.supervised import render_messages_to_datums
    rs = render_messages_to_datums(row["messages"], renderer=renderer, tools=row["tools"])
    total = sum(len(r.token_ids) for r in rs)
    trained = sum(sum(1 for w in r.token_weights if w > 0) for r in rs)
    return len(rs), total, trained


def cmd_select(args):
    key = args.key
    m = MODELS[key]
    tok, renderer = load_renderer(key)
    rows = [json.loads(l) for l in open(os.path.join(FT, "data/all_convs.jsonl"))]
    stats = []
    for r in rows:
        n, total, trained = count_tokens(r, renderer)
        stats.append((n, total, trained))
    nsplit = collections.Counter(s[0] for s in stats)
    print("datums-per-row histogram:", dict(nsplit))
    ok = [(r, s) for r, s in zip(rows, stats) if s[0] == 1 and s[1] <= m["max_seq_len"]]
    print(f"{len(ok)}/{len(rows)} rows fit in {m['max_seq_len']} tokens (single datum); "
          f"all-rows tokens={sum(s[1] for s in stats)}")
    # Diversify: per task, shortest first; take round-robin until budget.
    by_task = collections.defaultdict(list)
    for r, s in ok:
        by_task[r["meta"]["task"]].append((r, s))
    for v in by_task.values():
        v.sort(key=lambda x: x[1][1])
    rng = random.Random(0)
    chosen, tokens = [], 0
    for rnd in range(args.per_task):
        tasks = list(by_task)
        rng.shuffle(tasks)
        cand = [by_task[t][rnd] for t in tasks if len(by_task[t]) > rnd]
        cand.sort(key=lambda x: x[1][1])
        for r, s in cand:
            if tokens + s[1] > args.budget:
                continue
            chosen.append((r, s))
            tokens += s[1]
    rng.shuffle(chosen)
    path = os.path.join(FT, f"data/{key}.jsonl")
    with open(path, "w") as f:
        for r, s in chosen:
            r = dict(r)
            r["meta"] = {**r["meta"], "tokens": s[1], "trained_tokens": s[2]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    doms = collections.Counter(r["meta"]["task"].split(".")[0] for r, _ in chosen)
    summary = {"key": key, "rows": len(chosen), "tasks": len({r["meta"]["task"] for r, _ in chosen}),
               "tokens": tokens, "trained_tokens": sum(s[2] for _, s in chosen),
               "max_len": max(s[1] for _, s in chosen), "domains": dict(doms)}
    json.dump(summary, open(os.path.join(FT, f"data/{key}_summary.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    s = sub.add_parser("select")
    s.add_argument("--key", required=True)
    s.add_argument("--budget", type=int, default=3_000_000)
    s.add_argument("--per-task", type=int, default=2)
    a = p.parse_args()
    {"convert": cmd_convert, "select": cmd_select}[a.cmd](a)
