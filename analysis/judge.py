"""LLM judge: reads a harness-v2 trajectory (never the grader's assertions) and estimates P(task done correctly).
Usage: python3 analysis/judge.py calib|test  -> runs/judge_<split>.json (cached per task)."""
import json, os, re, sys, concurrent.futures as cf, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for l in open(ROOT / ".env"):
    if "=" in l and not l.startswith("#"): k, v = l.strip().split("=", 1); os.environ.setdefault(k, v.strip('"'))
MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
split = sys.argv[1]
src = json.load(open(ROOT / f"runs/{split}_ds_harness_v2.json"))["tasks"]
out_p = ROOT / f"runs/judge_{split}.json"
cache = json.load(open(out_p)) if out_p.exists() else {}

def tc_list(x):
    if not x or x == "None": return []
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception:
            import ast; x = ast.literal_eval(x)
    return [json.loads(c) if isinstance(c, str) else c for c in x]

def render(msgs):
    parts = []
    for m in msgs:
        r, c = m.get("role"), m.get("content") or ""
        if not isinstance(c, str): c = json.dumps(c)
        if r == "system": continue
        if r == "user": parts.append(f"[USER] {c[:3000]}")
        elif r == "tool": parts.append(f"[TOOL RESULT] {c[:700]}")
        elif r == "assistant":
            s = f"[AGENT] {c[:800]}" if c.strip() else "[AGENT]"
            for t in tc_list(m.get("tool_calls")):
                f = t.get("function", t); s += f"\n  -> {f.get('name')}({str(f.get('arguments'))[:500]})"
            parts.append(s)
    txt = "\n".join(parts)
    return txt if len(txt) < 60000 else txt[:12000] + "\n...[middle truncated]...\n" + txt[-46000:]

PROMPT = """You are a strict QA auditor for a business-workflow automation agent working in simulated SaaS apps (CRM, Slack, Sheets, email, ...).
Below is the full transcript: the user's request, every tool call the agent made, truncated tool results, and its final message.
A hidden grader checks the final app state. Estimate the probability that the task was completed fully and correctly.

Be skeptical. Common failure modes:
- the request implies company policy/eligibility/compliance rules or reference data, and the agent never found or applied them
- a required write (create/update/send/post) is missing, went to the wrong record/channel/recipient, or was done for too many or too few records
- the agent claims success but tool results show errors, empty results, or values that do not match what it wrote
- values were altered (numbers reformatted, names/IDs changed) instead of copied exactly
- the agent stopped early, asked a question, or only partly handled multi-part requests
- the agent did extra, unrequested destructive actions

Transcript:
<<<
{T}
>>>

Reply with ONLY a JSON object: {{"p_success": <0..1>, "main_risk": "<one sentence>"}}"""

def call(t):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.replace("{T}", render(t["messages"])).replace("{{", "{").replace("}}", "}")}],
            "temperature": 0, "max_tokens": 6000, "reasoning_effort": "low"}
    import time, random
    for attempt in range(8):
        if attempt: time.sleep(min(30, 2 * 2 ** attempt) * random.uniform(0.5, 1.0))
        try:
            req = urllib.request.Request("https://api.fireworks.ai/inference/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            txt = r["choices"][0]["message"].get("content") or ""
            if not txt.strip(): body["reasoning_effort"] = "none"; raise ValueError("empty content")
            j = json.loads(re.findall(r"\{[^{}]*\"p_success\"[^{}]*\}", txt, re.S)[-1])
            u = r.get("usage", {})
            return t["name"], {"p": float(j["p_success"]), "risk": j.get("main_risk", ""),
                               "cost": u.get("prompt_tokens", 0) * 0.30e-6 + u.get("completion_tokens", 0) * 1.20e-6}
        except Exception as e:
            err = repr(e)[:200]
    return t["name"], {"p": None, "err": err}

todo = [t for t in src if t["name"] not in cache or cache[t["name"]].get("p") is None]
print(f"{split}: {len(src)} tasks, {len(todo)} to judge", flush=True)
with cf.ThreadPoolExecutor(int(os.environ.get('J', 4))) as ex:
    for i, (n, res) in enumerate(ex.map(call, todo)):
        cache[n] = res; print(i, n, res.get("p"), res.get("err", ""), flush=True)
        json.dump(cache, open(out_p, "w"), indent=1)
