"""Cascade: run harness v2; an LLM judge reads its trajectory; if P(success) < th, fall back to FALLBACK.
Fallback and threshold are selected on calib only, then reported once on test."""
import json, sys
sys.path.insert(0, "analysis")
from router import load_run, boot_ci, mcnemar, ROOT
cfg = json.load(open(ROOT / "analysis/systems.json"))
R = {sp: {n: load_run(sc[sp], sc["price"]) for n, sc in cfg["systems"].items() if sc.get(sp)} for sp in ["calib", "test"]}
J = {sp: json.load(open(ROOT / f"runs/judge_{sp}.json")) for sp in ["calib", "test"]}

def run(sp, fb, th):
    T = [t for t in R[sp]["ds_harness_v2"] if J[sp].get(t, {}).get("p") is not None and t in R[sp][fb]]
    out = []
    for t in T:
        v, j = R[sp]["ds_harness_v2"][t], J[sp][t]
        esc = j["p"] < th
        r = R[sp][fb][t] if esc else v
        out.append(dict(t=t, pass_=r["pass"], cost=v["cost"] + j.get("cost", 0) + (R[sp][fb][t]["cost"] if esc else 0), esc=esc))
    return out
summ = lambda o: (sum(x["pass_"] for x in o), len(o), sum(x["cost"] for x in o) / max(1, len(o)), sum(x["esc"] for x in o))

# judge quality
for sp in ["calib", "test"]:
    T = [t for t in R[sp]["ds_harness_v2"] if J[sp].get(t, {}).get("p") is not None]
    ps = sorted((J[sp][t]["p"], R[sp]["ds_harness_v2"][t]["pass"]) for t in T)
    pos = [p for p, y in ps if y]; neg = [p for p, y in ps if not y]
    auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
    print(f"[{sp}] judged {len(T)}  judge AUC (v2 pass vs fail) = {auc:.2f}  mean p | pass {sum(pos)/max(1,len(pos)):.2f}  | fail {sum(neg)/max(1,len(neg)):.2f}")

THS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
FBS = ["nemo_ft", "qwen_ft", "ds_base", "ds_harness"]
best = None
print("\ncalib grid (pass, n, $/task, escalated):")
for fb in FBS:
    row = []
    for th in THS:
        p, n, c, e = summ(run("calib", fb, th)); row.append(f"{th:.1f}:{p}/{n}({e})")
        key = (p / n, -c)
        if th > 0 and (best is None or key > best[0]): best = (key, fb, th)
    print(f"  {fb:12s}", " ".join(row))
_, FB, TH = best
print(f"\nSELECTED on calib: fallback={FB}, threshold={TH}")
res = {}
for fb in FBS:
    o = run("test", fb, TH); p, n, c, e = summ(o)
    v2 = [R["test"]["ds_harness_v2"][x["t"]]["pass"] for x in o]; casc = [x["pass_"] for x in o]
    k = [R["test"]["kimi_k3"][x["t"]]["pass"] for x in o if x["t"] in R["test"]["kimi_k3"]]
    lo, hi = boot_ci([float(x) for x in casc])
    res[fb] = dict(pass_=p / n, lo=lo, hi=hi, cost=c, esc=e, n=n, gained=sum(a and not b for a, b in zip(casc, v2)), lost=sum(b and not a for a, b in zip(casc, v2)),
                   p_vs_kimi=mcnemar(casc, k)[2] if len(k) == n else None, rescued=[x["t"] for x, b in zip(o, v2) if x["pass_"] and not b])
    tag = "  <== selected" if fb == FB else ""
    print(f"test  v2 -> judge(th={TH}) -> {fb:10s}: {p}/{n} = {p/n:.1%}  [{lo:.1%},{hi:.1%}]  ${c:.4f}/task  escalated {e}  +{res[fb]['gained']}/-{res[fb]['lost']} vs v2  p_vs_kimi={res[fb]['p_vs_kimi']}{tag}")
json.dump({"fallback": FB, "threshold": TH, "test": res}, open(ROOT / "runs/cascade.json", "w"), indent=1)
