"""Regenerate slides/story_data.js from runs/ (test60 is the held-out split)."""
import json, sys, datetime
sys.path.insert(0, "analysis")
from router import load_run, boot_ci, mcnemar, route_knn, ROOT
import router as RT

cfg = json.load(open("analysis/systems.json"))
emb = json.load(open(ROOT / "runs/task_embeddings.json"))
have = lambda p: (ROOT / p).exists() or (ROOT / p).with_suffix(".jsonl").exists()
calib, test = {}, {}
for n, sc in cfg["systems"].items():
    if sc.get("calib") and have(sc["calib"]): calib[n] = load_run(sc["calib"], sc["price"])
    if sc.get("test") and have(sc["test"]): test[n] = load_run(sc["test"], sc["price"])
S = cfg["router_systems"]
TT = sorted(set.intersection(*[set(test[s]) for s in S + ["kimi_k3"]]))
CT = sorted(set.intersection(*[set(calib[s]) for s in S]))
knn = {t: route_knn(t, CT, calib, S, emb) for t in TT}
rows = {s: [test[s][t] for t in TT] for s in S + ["kimi_k3"]}
rows["router_knn"] = [test[knn[t]][t] for t in TT]
rows["oracle"] = [test[max(S, key=lambda s: (test[s][t]["pass"], -test[s][t]["cost"]))][t] for t in TT]
def stat(rs):
    p = [1.0 if x["pass"] else 0.0 for x in rs]; lo, hi = boot_ci(p)
    return dict(pass_=round(sum(p) / len(p), 4), lo=round(lo, 4), hi=round(hi, 4), cost=round(sum(x["cost"] for x in rs) / len(rs), 4))
st = {k: stat(v) for k, v in rows.items()}
HERO = "ds_harness_v2"
pv = lambda a, b: mcnemar([x["pass"] for x in rows[a]], [x["pass"] for x in rows[b]])[2]
mix = {}
for s in knn.values(): mix[s] = mix.get(s, 0) + 1
cp = lambda s: sum(calib[s][t]["pass"] for t in CT) / len(CT)

# spend: every eval run we have (both splits) + LoRA training
evals = sum(v["cost"] for D in (calib, test) for r in D.values() for v in r.values())
spend_total = round(evals + 22.27, 2)

LABEL = {"kimi_k3": ("Kimi K3 (frontier, effort=high)", "frontier"), "ds_base": ("DeepSeek V4.1 Flash, no harness", "open"),
         "ds_harness": ("DeepSeek + Holo harness v1", "harness"), "ds_harness_v2": ("DeepSeek + Holo harness v2 (ours)", "harness"),
         "qwen_ft": ("Qwen 3.6 35B-A3B + Holo4 LoRA", "finetune"), "nemo_ft": ("Nemotron 3.5 Super 120B + Holo4 LoRA", "finetune"),
         "router_knn": ("kNN router over all five", "router"), "oracle": ("Oracle (perfect routing, upper bound)", "oracle")}
systems = []
for k, (lab, kind) in LABEL.items():
    d = st[k]; e = {"id": k, "label": lab, "kind": kind, "pass": d["pass_"], "lo": d["lo"], "hi": d["hi"], "cost": d["cost"]}
    if kind == "finetune": e["note"] = "uncached sampler"
    if k == HERO: e["hero"] = True
    systems.append(e)

h, kk = st[HERO], st["kimi_k3"]
data = {
  "asof": datetime.datetime.now().strftime("%b %-d, %Y %-I:%M %p PT"), "n_test": len(TT),
  "hero": {"system": "DeepSeek V4.1 Flash + our Holo-style harness v2", "router_pass": h["pass_"], "kimi_pass": kk["pass_"],
           "router_cost": h["cost"], "kimi_cost": kk["cost"], "cost_ratio": None, "pass_delta_pts": None,
           "p_vs_kimi": round(pv(HERO, "kimi_k3"), 3), "base_pass": st["ds_base"]["pass_"], "base_cost": st["ds_base"]["cost"],
           "p_base_vs_kimi": round(pv("ds_base", "kimi_k3"), 3)},
  "systems": systems,
  "published": [{"label": "Claude Opus 5", "pass": 0.503}, {"label": "Kimi K3", "pass": 0.4667}, {"label": "GPT-5.6 Sol", "pass": 0.4583}],
  "router": {"mix": mix, "loo_pass": None, "beats": f"5 systems, kNN (k=7) fit on {len(CT)} calib tasks",
             "pass": st["router_knn"]["pass_"], "oracle": st["oracle"]["pass_"], "p_vs_hero": round(pv("router_knn", HERO), 3)},
  "harness": {"calib_base": round(cp("ds_base"), 4), "calib_v1": round(cp("ds_harness"), 4), "calib_v2": round(cp("ds_harness_v2"), 4),
              "test_base": st["ds_base"]["pass_"], "test_v1": st["ds_harness"]["pass_"], "test_v2": st["ds_harness_v2"]["pass_"],
              "n_calib": len(CT), "turns_v1": 25, "turns_base": 13},
  "distill": {"holo_solved": 289, "train_tasks": 480, "sft_rows": 259, "sft_tasks": 257, "lora_rank": 32, "steps_qwen": 33, "steps_nemo": 32,
              "train_cost_qwen": 10.68, "train_cost_nemo": 11.59, "train_minutes": 5,
              "ft_union_test": sum(test["qwen_ft"][t]["pass"] or test["nemo_ft"][t]["pass"] for t in TT)},
  "spend": {"training": 22.27, "kimi_eval": round(sum(x["cost"] for x in rows["kimi_k3"]), 2), "evals": round(evals, 2), "total": spend_total},
  "example": {"task": "sales.add_contact_to_campaign",
              "request": "\u201cAdd senior contacts to the Q1 Product Launch Webinar campaign. Focus on director-level and above. Make sure everyone meets our campaign eligibility and compliance requirements.\u201d",
              "kimi": "Searched the CRM for senior titles and added <b>all 22</b> director-level contacts in 29 tool calls. It checked opt-out flags but never looked for the company's eligibility rules. Its summary claims \u201celigibility & compliance checks applied\u201d. <b>4 of 8 checks failed</b>: ineligible contacts are now in the campaign.",
              "ours": "Searched for the eligibility policy first and found the <code>ss_campaign_rules</code> sheet (industry restrictions, compliance, legal and regulatory holds). It filtered the candidates and added <b>only the 4</b> who qualify. <b>All 4 grader checks passed</b>.",
              "system": "DeepSeek + harness v2", "kimi_short": "adds all 22 senior contacts, including ones the compliance sheet excludes"},
}
js = "// Generated by analysis/fill_story.py from runs/. Do not edit by hand.\nwindow.STORY = " + json.dumps(data, indent=1).replace('"pass_"', '"pass"') + ";\n"
open("slides/story_data.js", "w").write(js)
print(json.dumps({k: data[k] for k in ["hero", "router", "harness", "spend"]}, indent=1))
