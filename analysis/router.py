"""Per-task router over N systems + paired statistics.

Usage:
  python3 analysis/router.py analysis/systems.json

price = [$ per 1M uncached input, $ per 1M cached input, $ per 1M output].
"""
import json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = 7


_RATIO = {}  # price tuple -> exact/naive cost ratio, learned from .json exports (jsonl lacks cache counts)


def _json_cost(t, price):
    cached = t.get("cached_input_tokens") or 0
    uncached = t.get("uncached_input_tokens")
    if uncached is None:
        uncached = max(0, (t.get("input_tokens") or 0) - cached)
    return (uncached * price[0] + cached * price[1] + (t.get("output_tokens") or 0) * price[2]) / 1e6


def load_run(path, price):
    """Pass/score from .jsonl when present (covers resumed runs), cost from the .json export.

    The incremental jsonl has no cached-token counts. Tasks present in the export get the
    exact cached-aware cost; the rest get naive cost x the run's exact/naive ratio."""
    p = ROOT / path
    js = json.load(open(p)) if p.exists() else None
    exact = {t["name"]: (_json_cost(t, price), t) for t in js["tasks"]} if js else {}
    if exact:
        naive = sum((t.get("input_tokens") or 0) * price[0] + (t.get("output_tokens") or 0) * price[2]
                    for _, t in exact.values()) / 1e6
        if naive > 0:
            _RATIO[tuple(price)] = sum(c for c, _ in exact.values()) / naive
    if p.with_suffix(".jsonl").exists():
        out = load_jsonl(p.with_suffix(".jsonl"), price)
        r = _RATIO.get(tuple(price), 1.0)
        for k, v in out.items():
            v["cost"] = exact[k][0] if k in exact else v["cost"] * r
        return out
    return {n: {"pass": bool(t["passed"]), "score": float(t["score"] or 0), "cost": c}
            for n, (c, t) in exact.items()}


def load_jsonl(path, price):
    out = {}
    for line in open(path):
        o = json.loads(line)
        u = o.get("_usage") or {}
        cached = u.get("cached_input_tokens") or 0
        uncached = (u.get("input_tokens") or 0) - cached
        cost = (uncached * price[0] + cached * price[1] + (u.get("output_tokens") or 0) * price[2]) / 1e6
        m = o.get("metrics") or {}
        passed = m.get("task_completed_correctly")
        passed = float(o.get("reward") or 0) == 1.0 if passed is None else float(passed) == 1.0
        out[o["task"]] = {"pass": passed, "score": float(o.get("reward") or 0), "cost": cost}
    return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans. Returns (a_only, b_only, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def boot_ci(xs, iters=5000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def domain(t):
    return t.split(".")[0]


_sim_cache = {}


def sim(a, b, emb):
    key = (a, b) if a < b else (b, a)
    if key not in _sim_cache:
        _sim_cache[key] = cos(emb[a], emb[b])
    return _sim_cache[key]


def route_knn(task, train_tasks, R, systems, emb, k=K):
    nbrs = sorted(train_tasks, key=lambda u: -sim(task, u, emb))[:k]
    best = None
    for s in systems:
        sc = sum(R[s][u]["score"] for u in nbrs) / k
        cst = sum(R[s][u]["cost"] for u in nbrs) / k
        key = (round(sc, 6), -cst)
        if best is None or key > best[0]:
            best = (key, s)
    return best[1]


def route_domain(task, train_tasks, R, systems):
    us = [u for u in train_tasks if domain(u) == domain(task)]
    return max(systems, key=lambda s: (sum(R[s][u]["score"] for u in us), -sum(R[s][u]["cost"] for u in us)))


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    emb = json.load(open(ROOT / "runs/task_embeddings.json"))
    S = cfg["router_systems"]
    calib, test = {}, {}
    def have(p):
        return (ROOT / p).exists() or (ROOT / p).with_suffix(".jsonl").exists()

    for name, sc in cfg["systems"].items():
        if sc.get("calib") and have(sc["calib"]):
            calib[name] = load_run(sc["calib"], sc["price"])
        if sc.get("test") and have(sc["test"]):
            test[name] = load_run(sc["test"], sc["price"])
    refs = [r for r in cfg.get("reference", []) if r in test]

    test_tasks = sorted(set.intersection(*[set(test[s]) for s in S + refs]))
    calib_tasks = sorted(set.intersection(*[set(calib[s]) for s in S]))
    print(f"test tasks with all systems: {len(test_tasks)}; calib tasks: {len(calib_tasks)}")

    choices = {
        "router_knn": {t: route_knn(t, calib_tasks, calib, S, emb) for t in test_tasks},
        "router_domain": {t: route_domain(t, calib_tasks, calib, S) for t in test_tasks},
    }
    best_single = max(S, key=lambda s: sum(calib[s][u]["score"] for u in calib_tasks))
    choices["best_single_on_calib"] = {t: best_single for t in test_tasks}
    choices["oracle_upper_bound"] = {
        t: max(S, key=lambda s: (test[s][t]["pass"], test[s][t]["score"], -test[s][t]["cost"])) for t in test_tasks
    }

    rows = {s: [test[s][t] for t in test_tasks] for s in S + refs}
    for r, ch in choices.items():
        rows[r] = [test[ch[t]][t] for t in test_tasks]

    lines = [f"Held-out test set: {len(test_tasks)} tasks (router fit on {len(calib_tasks)} calib tasks)", "",
             "| System | Pass rate | 95% bootstrap CI | Avg partial score | Real $/task |", "|---|---|---|---|---|"]
    for name, rs in rows.items():
        p = [1.0 if x["pass"] else 0.0 for x in rs]
        lo, hi = boot_ci(p)
        lines.append(f"| {name} | {sum(p)/len(p):.1%} | [{lo:.1%}, {hi:.1%}] | "
                     f"{sum(x['score'] for x in rs)/len(rs):.1%} | ${sum(x['cost'] for x in rs)/len(rs):.4f} |")

    lines += ["", "Paired exact McNemar, router_knn vs X (router-only wins / X-only wins / p):", ""]
    rk = [x["pass"] for x in rows["router_knn"]]
    for name in S + refs + ["router_domain", "best_single_on_calib"]:
        a, b, p = mcnemar(rk, [x["pass"] for x in rows[name]])
        lines.append(f"- vs {name}: {a} / {b} / p={p:.3f}")

    mix = {}
    for s in choices["router_knn"].values():
        mix[s] = mix.get(s, 0) + 1
    lines += ["", f"router_knn selection mix on test: {mix}", f"best single system on calib: {best_single}"]

    pooled = {s: {**calib.get(s, {}), **test[s]} for s in S}
    pool_tasks = sorted(set.intersection(*[set(pooled[s]) for s in S]))
    loo = [pooled[route_knn(t, [u for u in pool_tasks if u != t], pooled, S, emb)][t]["pass"] for t in pool_tasks]
    lines += ["", f"Leave-one-out kNN router over {len(pool_tasks)} pooled calib+test tasks: {sum(loo)/len(loo):.1%}"]
    for s in S:
        ps = [pooled[s][t]["pass"] for t in pool_tasks]
        a, b, p = mcnemar(loo, ps)
        lines.append(f"- {s}: {sum(ps)/len(ps):.1%} (router-only {a} / {s}-only {b}, p={p:.3f})")

    out = "\n".join(lines)
    print(out)
    (ROOT / "runs/results.md").write_text(out + "\n")
    json.dump(choices, open(ROOT / "runs/router_choices.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "analysis/systems.json")
