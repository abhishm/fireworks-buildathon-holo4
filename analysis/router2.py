"""Router v2 search. Hyperparameters chosen ONLY by leave-one-out CV on calib; test touched once at the end."""
import json, itertools, sys
sys.path.insert(0, "analysis")
import router as R
from router import load_run, sim, mcnemar, boot_ci, ROOT

cfg = json.load(open("analysis/systems.json"))
emb = json.load(open(ROOT / "runs/task_embeddings.json"))
calib, test = {}, {}
have = lambda p: (ROOT / p).exists() or (ROOT / p).with_suffix(".jsonl").exists()
for n, sc in cfg["systems"].items():
    if sc.get("calib") and have(sc["calib"]): calib[n] = load_run(sc["calib"], sc["price"])
    if sc.get("test") and have(sc["test"]): test[n] = load_run(sc["test"], sc["price"])
ALL = cfg["router_systems"]
ct = sorted(set.intersection(*[set(calib[s]) for s in ALL]))
tt = sorted(set.intersection(*[set(test[s]) for s in ALL + ["kimi_k3"]]))

def route(task, train, D, pool, k, margin, prior, metric):
    # default = best pool system on the training tasks; deviate only if neighbours favour another by > margin
    default = max(pool, key=lambda s: sum(D[s][u][metric] for u in train))
    nb = sorted(train, key=lambda u: -sim(task, u, emb))[:k]
    def est(s):  # shrink neighbour estimate toward that system's global rate
        g = sum(D[s][u][metric] for u in train) / len(train)
        return (sum(D[s][u][metric] for u in nb) + prior * g) / (k + prior)
    e = {s: est(s) for s in pool}
    best = max(pool, key=lambda s: (e[s], -sum(D[s][u]["cost"] for u in nb)))
    return best if e[best] - e[default] > margin else default

for D in (calib, test):
    for s in D:
        for v in D[s].values(): v["passf"] = 1.0 if v["pass"] else 0.0

pools = {"all5": ALL, "deepseek3": ["ds_base", "ds_harness", "ds_harness_v2"], "base+v2": ["ds_base", "ds_harness_v2"]}
grid = list(itertools.product(pools, [3, 5, 7, 11, 15], [0.0, 0.1, 0.2], [0, 3, 10], ["score", "passf"]))
res = []
for pn, k, m, pr, met in grid:
    cv = sum(calib[route(t, [u for u in ct if u != t], calib, pools[pn], k, m, pr, met)][t]["pass"] for t in ct) / len(ct)
    res.append((cv, pn, k, m, pr, met))
res.sort(key=lambda r: (-r[0], r[2] == "all5"))
print("calib LOO-CV top 8:"); [print(f"  {r[0]:.1%}  pool={r[1]} k={r[2]} margin={r[3]} prior={r[4]} metric={r[5]}") for r in res[:8]]
print(f"  baseline best-single on calib CV: {max(sum(calib[s][t]['pass'] for t in ct) for s in ALL)/len(ct):.1%}")

cv, pn, k, m, pr, met = res[0]
ch = {t: route(t, ct, calib, pools[pn], k, m, pr, met) for t in tt}
rp = [test[ch[t]][t]["pass"] for t in tt]
print(f"\nSELECTED (by calib CV only): pool={pn} k={k} margin={m} prior={pr} metric={met}")
print(f"TEST router2: {sum(rp)}/{len(tt)} = {sum(rp)/len(tt):.1%}  cost ${sum(test[ch[t]][t]['cost'] for t in tt)/len(tt):.4f}  CI {boot_ci([float(x) for x in rp])}")
mix = {}
for s in ch.values(): mix[s] = mix.get(s, 0) + 1
print("mix", mix)
for s in ALL + ["kimi_k3"]:
    a, b, p = mcnemar(rp, [test[s][t]["pass"] for t in tt]); print(f"  vs {s}: {sum(test[s][t]['pass'] for t in tt)}/{len(tt)}  wins {a}/{b} p={p:.3f}")
print("\nOracle upper bounds on test:")
for pn2, pool in pools.items():
    o = sum(any(test[s][t]["pass"] for s in pool) for t in tt)
    print(f"  oracle {pn2}: {o}/{len(tt)} = {o/len(tt):.1%}")
o = sum(any(test[s][t]["pass"] for s in ["ds_harness_v2", "qwen_ft", "nemo_ft"]) for t in tt); print(f"  oracle v2+finetunes: {o/len(tt):.1%}")
print("\nTop 5 configs evaluated on test (for disclosure, NOT used for selection):")
for r in res[:5]:
    _, pn, k, m, pr, met = r
    print(f"  {r}: test {sum(test[route(t, ct, calib, pools[pn], k, m, pr, met)][t]['pass'] for t in tt)/len(tt):.1%}")
