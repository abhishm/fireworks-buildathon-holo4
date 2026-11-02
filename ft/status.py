"""Regenerate ft/STATUS.md from logs + checkpoints.json."""

import json
import os
import time

FT = os.path.dirname(os.path.abspath(__file__))
PRICE = {"qwen": 2.14, "nemotron": 2.32}


def events(key, tag):
    p = os.path.join(FT, f"logs/train_{key}_{tag}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


lines = [f"# Holo4 → AutomationBench LoRA SFT status (updated {time.strftime('%Y-%m-%d %H:%M %Z')})", ""]
lines += [
    "Data: Hcompany/trajectories, AutomationBench, success==True, **train480 tasks only** (calib60/test60 asserted excluded).",
    "765 convertible trajectories / 289 tasks. Conversation = AB system+user prompt, AB `api` tool schemas, Holo steps as",
    "assistant(reasoning_content=thought, tool_calls) + tool results; update_plan/note dropped; final `answer` → plain assistant content.",
    "Null `params`/`body` omitted from api_fetch args (AB defaults them to None). Loss on assistant tokens only (reasoning + tool calls + final answer).",
    "Selection: <=32K tokens, 1–2 per task, shortest first, ~5M tokens/model.",
    "",
    "**Sampling integration:** base models `accounts/fireworks/models/qwen3p6-35b-a3b` (renderer `qwen3_6_interleaved`,",
    "tokenizer `Qwen/Qwen3.6-35B-A3B`) and `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` (renderer",
    "`nemotron35_vl`, tokenizer `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`). Build renderer with",
    "`training.utils.supervised.build_renderer(load_tokenizer(tok), tok, renderer_name)`; tools via `build_tool_prefixed_messages`.",
    "Training kept reasoning (`<think>`) on every assistant turn and used AB's system prompt + tools unchanged.",
    "",
    "Hyperparams: LoRA rank 32 (alpha 32), lr 2e-4 (3-step warmup, cosine to 4e-5), batch 8 conversations, 1 epoch, Adam(0.9,0.95,eps 1e-8, wd 0), cross_entropy (per-sequence mean).",
    "",
]
total_cost = 0.0
for key in ("qwen", "nemotron"):
    summ = os.path.join(FT, f"data/{key}_summary.json")
    s = json.load(open(summ)) if os.path.exists(summ) else {}
    ev = events(key, "run")
    steps = [e for e in ev if e.get("event") == "step"]
    plan = next((e for e in ev if e.get("event") == "plan"), {})
    cum = steps[-1]["cum_tokens"] if steps else 0
    cost = cum / 1e6 * PRICE[key]
    total_cost += cost
    lines.append(f"## {key}")
    lines.append(f"- dataset: data/{key}.jsonl — {s.get('rows')} convs, {s.get('tasks')} tasks, {s.get('tokens')} tokens "
                 f"({s.get('trained_tokens')} assistant/trained), max len {s.get('max_len')}")
    lines.append(f"- progress: step {len(steps)}/{plan.get('steps')} — {cum} tokens trained, est ${cost:.2f} (plan ${plan.get('est_cost')})")
    if steps:
        losses = [round(e["loss"], 4) for e in steps if e.get("loss") is not None]
        lines.append(f"- loss by step: {losses}")
    for e in ev:
        if e.get("event") in ("sampler_ckpt", "promoted", "promote_failed", "session"):
            e = {k: v for k, v in e.items() if k != "t"}
            lines.append(f"- {e}")
    lines.append("")
dry = 0.07
lines.append(f"**Spend estimate:** ${total_cost + dry:.2f} (incl. ~${dry} dry runs/sampling check). Budget $30.")
lines.append("")
ck = os.path.join(FT, "checkpoints.json")
if os.path.exists(ck):
    lines += ["## checkpoints.json", "```json", open(ck).read(), "```"]
open(os.path.join(FT, "STATUS.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-40:]))
