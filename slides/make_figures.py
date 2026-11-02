"""Build slide figures + numbers from runs/. Re-run any time; partial runs are labeled.

Usage: python3 slides/make_figures.py        (needs matplotlib)
Writes slides/img/*.png and slides/data.js.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from router import boot_ci, load_run, mcnemar, route_knn  # noqa: E402

IMG = ROOT / "slides/img"
IMG.mkdir(parents=True, exist_ok=True)

BG, PANEL, FG, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
C = {"kimi_k3": "#a371f7", "ds_base": "#58a6ff", "ds_harness": "#1f6feb", "ds_harness_v2": "#39c5cf", "qwen_ft": "#f0883e",
     "nemo_ft": "#76b900", "router": "#e3b341"}
LABEL = {"kimi_k3": "Kimi K3", "ds_base": "DeepSeek V4.1 Flash", "ds_harness": "DeepSeek + Holo harness",
         "ds_harness_v2": "DeepSeek + harness v2",
         "qwen_ft": "Qwen3.6-35B-A3B + LoRA", "nemo_ft": "Nemotron 3.5 Super + LoRA", "router": "kNN router"}
DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
# Share of input tokens served from prompt cache, measured on the complete DeepSeek screen run. Used only when
# an in-progress .jsonl has no cached-token counts (it would otherwise be priced as fully uncached).
DS_CACHE_FRAC = 0.995

plt.rcParams.update({"axes.axisbelow": True,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13, "axes.titleweight": "bold",
    "axes.titlesize": 17, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(IMG / name, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", IMG / name)


# ---------------------------------------------------------------- data
cfg = json.load(open(ROOT / "analysis/systems.json"))
splits = json.load(open(ROOT / "runs/splits.json"))
TEST, CALIB = splits["test60"], splits["calib60"]


def load(split, name):
    sc = cfg["systems"][name]
    rel = sc.get(split)
    if not rel:
        return {}, False
    p = ROOT / rel
    final = p.exists()
    if not final and not p.with_suffix(".jsonl").exists():
        return {}, False
    runs = load_run(rel, sc["price"])
    if not final and sc["price"][1] != sc["price"][0]:
        jl = [json.loads(x) for x in open(p.with_suffix(".jsonl"))]
        if not any((o.get("_usage") or {}).get("cached_input_tokens") for o in jl):
            pr = sc["price"]
            for o in jl:
                u = o.get("_usage") or {}
                i, out = u.get("input_tokens") or 0, u.get("output_tokens") or 0
                runs[o["task"]]["cost"] = (i * (1 - DS_CACHE_FRAC) * pr[0] + i * DS_CACHE_FRAC * pr[1] + out * pr[2]) / 1e6
    keep = set(TEST if split == "test" else CALIB)
    return {k: v for k, v in runs.items() if k in keep}, final


SYS = ["kimi_k3", "ds_base", "ds_harness", "ds_harness_v2", "qwen_ft", "nemo_ft"]
test, calib, done = {}, {}, {}
for s in SYS:
    test[s], done[s] = load("test", s)
    calib[s], _ = load("calib", s)


def stats(rs):
    p = [1.0 if r["pass"] else 0.0 for r in rs]
    lo, hi = boot_ci(p) if p else (0, 0)
    n = len(p)
    return {"n": n, "pass": sum(p) / n if n else 0, "lo": lo, "hi": hi,
            "score": sum(r["score"] for r in rs) / n if n else 0, "cost": sum(r["cost"] for r in rs) / n if n else 0}


S = {s: stats(list(test[s].values())) for s in SYS if test[s]}
for s in S:
    S[s]["complete"] = S[s]["n"] == len(TEST)
    S[s]["kimi_same"] = sum(test["kimi_k3"][t]["pass"] for t in test[s]) / S[s]["n"]

# Router: only when every router system has full calib60 + test60.
R = cfg["router_systems"]
router_ready = all(len(calib[s]) == len(CALIB) and len(test[s]) == len(TEST) for s in R)
router_mix = {}
if router_ready:
    emb = json.load(open(ROOT / "runs/task_embeddings.json"))
    choice = {t: route_knn(t, sorted(CALIB), calib, R, emb) for t in TEST}
    S["router"] = stats([test[choice[t]][t] for t in TEST])
    S["router"]["complete"] = True
    for v in choice.values():
        router_mix[v] = router_mix.get(v, 0) + 1
    rk = [test[choice[t]][t]["pass"] for t in TEST]
    S["router"]["vs_kimi"] = mcnemar(rk, [test["kimi_k3"][t]["pass"] for t in TEST])

paired = {}
for s in ["qwen_ft", "nemo_ft", "ds_base", "ds_harness", "ds_harness_v2"]:
    common = [t for t in TEST if t in test[s] and t in test["kimi_k3"]]
    if common:
        paired[s] = dict(zip(["sys_only", "kimi_only", "p"],
                             mcnemar([test[s][t]["pass"] for t in common], [test["kimi_k3"][t]["pass"] for t in common])),
                         n=len(common))

# ---------------------------------------------------------------- 1. pipeline diagram
fig, ax = plt.subplots(figsize=(16, 7.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, sub, color, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=PANEL, ec=color, lw=2.2))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=fs, weight="bold",
            color=FG)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=18))


box(0.2, 5.2, 3.2, 1.6, "Holo4 trajectories", "H Company, Apache-2.0\n289 / 480 train tasks solved", "#8b949e")
box(0.2, 2.8, 3.2, 1.6, "AutomationBench", "600 tasks, 6 domains\n47 simulated apps", "#8b949e")
box(0.2, 0.4, 3.2, 1.6, "Fixed splits", "train480 / calib60 / test60\nno task overlap", "#8b949e")

box(4.6, 5.2, 3.4, 1.6, "LoRA SFT", "259 successful runs, ~5M tok\nFireworks serverless training", C["qwen_ft"])
box(4.6, 2.8, 3.4, 1.6, "Holo-style harness", "operating procedure +\nverify-before-finish gate", C["ds_harness"])

box(9.2, 6.0, 3.0, 1.0, "Qwen3.6-35B-A3B + LoRA", "", C["qwen_ft"], fs=12)
box(9.2, 4.8, 3.0, 1.0, "Nemotron 3.5 Super + LoRA", "", C["nemo_ft"], fs=12)
box(9.2, 2.9, 3.0, 1.4, "DeepSeek V4.1 Flash", "+ harness", C["ds_harness"], fs=12)

box(13.2, 3.9, 2.6, 2.0, "kNN router", "task embedding ->\n7 nearest calib tasks ->\nbest score, then cheapest", C["router"])
box(13.2, 0.9, 2.6, 1.9, "test60 report", "pass rate, $/task,\nMcNemar vs Kimi K3,\nbootstrap CI", "#a371f7")

arrow(3.4, 6.0, 4.6, 6.0)
arrow(3.4, 3.6, 4.6, 3.6)
arrow(8.0, 6.2, 9.2, 6.45)
arrow(8.0, 5.8, 9.2, 5.3)
arrow(8.0, 3.6, 9.2, 3.6)
for y in (6.5, 5.3, 3.6):
    arrow(12.2, y, 13.2, 4.9, C["router"])
arrow(14.5, 3.9, 14.5, 2.8, C["router"])
arrow(3.4, 1.2, 13.2, 1.6, "#484f58")
ax.text(8.3, 1.0, "calib60 fits the router  |  test60 is touched only for the final report",
        ha="center", fontsize=11, color=MUTED, style="italic")
save(fig, "pipeline.png")

# ---------------------------------------------------------------- 2. data: splits + SFT funnel
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
parts = [("train480", 480, "#484f58"), ("calib60", 60, C["router"]), ("test60", 60, "#a371f7")]
left = 0
for name, n, col in parts:
    a1.barh(0, n, left=left, color=col, height=0.5)
    a1.text(left + n / 2, 0, f"{name}\n{n}", ha="center", va="center", fontsize=12, weight="bold",
            color=BG if col != "#484f58" else FG)
    left += n
a1.set_xlim(0, 600)
a1.set_ylim(-0.6, 0.9)
a1.set_yticks([])
a1.set_xlabel("AutomationBench public tasks (stratified, 10 per domain in calib/test)")
a1.set_title("600 tasks, three disjoint splits", loc="left")
a1.text(0, 0.55, "SFT data comes only from train480; router learns on calib60; test60 is held out",
        fontsize=11, color=MUTED)

fun = [("train tasks", 480), ("solved by Holo4 (either model)", 289), ("used for SFT (<=32K tok)", 257)]
cols = ["#484f58", "#6e7681", C["qwen_ft"]]
for i, ((lab, n), col) in enumerate(zip(fun, cols)):
    a2.barh(-i, n, color=col, height=0.62)
    a2.text(n + 6, -i, f"{n}", va="center", fontsize=14, weight="bold")
    a2.text(6, -i, lab, va="center", fontsize=12, color=FG if col != C["qwen_ft"] else BG, weight="bold")
a2.set_xlim(0, 560)
a2.set_yticks([])
a2.set_title("SFT coverage: 54% of train tasks have a usable demo", loc="left")
a2.set_xlabel("tasks   (259 conversations, ~5.0M tokens per model, loss on assistant tokens only)")
save(fig, "data.png")

# ---------------------------------------------------------------- 3. training loss
fig, ax = plt.subplots(figsize=(12, 5))
for m, key in [("qwen", "qwen_ft"), ("nemotron", "nemo_ft")]:
    f = ROOT / f"ft/logs/train_{m}_run.jsonl"
    if not f.exists():
        continue
    L = [e["loss"] for e in map(json.loads, open(f)) if e.get("event") == "step"]
    sm = [sum(L[max(0, i - 4):i + 1]) / len(L[max(0, i - 4):i + 1]) for i in range(len(L))]
    ax.plot(range(1, len(L) + 1), L, color=C[key], alpha=0.3, lw=1.2)
    ax.plot(range(1, len(L) + 1), sm, color=C[key], lw=2.8, label=f"{LABEL[key]} (5-step avg)")
ax.set_xlabel("optimizer step (batch = 8 conversations, 1 epoch)")
ax.set_ylabel("cross-entropy (assistant tokens)")
ax.set_title("LoRA SFT on Holo4 trajectories: rank 32, lr 2e-4, ~$22 total", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, loc="upper right")
save(fig, "loss.png")

# ---------------------------------------------------------------- 4. test60 pass-rate bars
order = [s for s in ["router", "ds_harness_v2", "ds_base", "ds_harness", "kimi_k3", "qwen_ft", "nemo_ft"] if s in S]
order.sort(key=lambda s: S[s]["pass"])
fig, ax = plt.subplots(figsize=(14, 0.9 * len(order) + 1.8))
for i, s in enumerate(order):
    st = S[s]
    ax.barh(i, st["pass"] * 100, color=C[s], height=0.62, alpha=1 if st["complete"] else 0.45,
            hatch=None if st["complete"] else "//", edgecolor=C[s])
    ax.plot([st["lo"] * 100, st["hi"] * 100], [i, i], color=FG, lw=1.6, alpha=0.8)
    ax.plot([st["score"] * 100], [i], marker="D", color=FG, ms=7, mec=BG)
    tag = ""
    if not st["complete"]:
        tag = f"   {st['n']}/60 done; Kimi on same tasks: {st['kimi_same'] * 100:.0f}%"
    ax.text(101, i, f"{st['pass']*100:.1f}%{tag}", va="center", fontsize=12.5, weight="bold")
ax.set_yticks(range(len(order)), [LABEL[s] for s in order])
ax.set_xlim(0, 100)
ax.set_xlabel("held-out test60: pass rate %  (bar)  |  95% bootstrap CI (line)  |  avg partial credit (diamond)")
ax.set_title("Test60 results", loc="left")
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "test60_pass.png")

# ---------------------------------------------------------------- 5. pass vs cost
fig, ax = plt.subplots(figsize=(13, 7))
published = [("Holo4 27B", 45.4, 0.05), ("Holo4 35B-A3B", 34.5, 0.02), ("Claude Opus 5", 50.3, 3.05),
             ("Kimi K3 (H Co. report)", 46.7, 0.43)]
for name, p, c in published:
    ax.scatter(c, p, s=140, facecolors="none", edgecolors=MUTED, lw=1.6, zorder=2)
    ax.annotate(name, (c, p), xytext=(8, -14), textcoords="offset points", fontsize=10.5, color=MUTED)
for s, st in S.items():
    ax.errorbar(st["cost"], st["pass"] * 100, yerr=[[st["pass"] * 100 - st["lo"] * 100], [st["hi"] * 100 - st["pass"] * 100]],
                fmt="o", color=C[s], ms=13, lw=1.5, capsize=4, alpha=1 if st["complete"] else 0.5, zorder=3)
    off, ha = {"ds_harness": ((14, -16), "left"), "ds_harness_v2": ((14, 14), "left"), "router": ((14, -2), "left"), "kimi_k3": ((12, 10), "left"), "ds_base": ((-14, -4), "right"),
               "qwen_ft": ((12, 8), "left"), "nemo_ft": ((12, -22), "left")}.get(s, ((12, 6), "left"))
    ax.annotate(LABEL[s] + ("" if st["complete"] else f" ({st['n']}/60)"), (st["cost"], st["pass"] * 100),
                xytext=off, textcoords="offset points", fontsize=11.5, color=C[s], weight="bold", ha=ha, zorder=5,
                bbox=dict(fc=BG, ec="none", alpha=0.85, pad=1.5))
ax.set_xscale("log")
ax.set_xlim(0.01, 6)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
ax.set_ylim(0, 80)
ax.set_xlabel("$ per task (log scale; our runs priced from real token usage incl. prompt-cache discounts)")
ax.set_ylabel("pass rate %")
ax.set_title("Pass rate vs cost: filled = our test60 runs, hollow = published (public 600 tasks)", loc="left")
ax.grid(color=GRID, lw=0.5, which="major")
ax.text(0.011, 76, "better ->  up and to the left", fontsize=11, color=MUTED, style="italic")
save(fig, "pass_vs_cost.png")

# ---------------------------------------------------------------- 6. per-domain heatmap
rows = [s for s in ["kimi_k3", "ds_harness_v2", "ds_harness", "ds_base", "qwen_ft", "nemo_ft"] if test[s]]
fig, ax = plt.subplots(figsize=(13, 0.8 * len(rows) + 1.6))
M = []
for s in rows:
    by = defaultdict(list)
    for t, r in test[s].items():
        by[t.split(".")[0]].append(r["pass"])
    M.append([(sum(by[d]) / len(by[d]) if by[d] else float("nan"), sum(by[d]), len(by[d])) for d in DOMAINS])
vals = [[c[0] for c in row] for row in M]
ax.imshow(vals, cmap="viridis", vmin=0, vmax=1, aspect="auto")
for i, row in enumerate(M):
    for j, (v, k, n) in enumerate(row):
        ax.text(j, i, f"{k}/{n}" if n else "-", ha="center", va="center", fontsize=13, weight="bold",
                color=BG if v == v and v > 0.55 else FG)
ax.set_xticks(range(len(DOMAINS)), DOMAINS)
ax.set_yticks(range(len(rows)), [LABEL[s] + ("" if S[s]["complete"] else " *") for s in rows])
partial = any(not S[s]["complete"] for s in rows)
ax.set_title("Where each system wins: test60 tasks passed / run per domain" + ("   (* still running)" if partial else ""),
             loc="left")
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
save(fig, "domains.png")

# ---------------------------------------------------------------- 7. complementarity
q, n_, k = test["qwen_ft"], test["nemo_ft"], test["kimi_k3"]
common = [t for t in TEST if t in q and t in n_ and t in k]
sets = [("Nemotron LoRA", lambda t: n_[t]["pass"], C["nemo_ft"]),
        ("Qwen LoRA", lambda t: q[t]["pass"], C["qwen_ft"]),
        ("either LoRA (oracle)", lambda t: q[t]["pass"] or n_[t]["pass"], C["router"]),
        ("Kimi K3", lambda t: k[t]["pass"], C["kimi_k3"]),
        ("any of the three (oracle)", lambda t: q[t]["pass"] or n_[t]["pass"] or k[t]["pass"], "#e6edf3")]
fig, ax = plt.subplots(figsize=(13, 4.8))
for i, (lab, f, col) in enumerate(sets):
    v = sum(1 for t in common if f(t))
    ax.barh(-i, v, color=col, height=0.6)
    ax.text(v + 0.6, -i, f"{v}/{len(common)}  ({v/len(common)*100:.0f}%)", va="center", fontsize=12.5, weight="bold")
ax.set_yticks([-i for i in range(len(sets))], [s[0] for s in sets])
ax.set_xlim(0, len(common) + 9)
ax.set_xlabel("test60 tasks passed")
ax.set_title("The two LoRAs solve different tasks, but even together they trail Kimi K3", loc="left")
save(fig, "overlap.png")

# ---------------------------------------------------------------- numbers for the deck
data = {"stats": S, "paired_vs_kimi": paired, "router_ready": router_ready, "router_mix": router_mix,
        "oracle": {lab: sum(1 for t in common if f(t)) for lab, f, _ in sets}, "n_common": len(common)}
(ROOT / "slides/data.js").write_text("window.RESULTS = " + json.dumps(data, indent=1) + ";\n")
print("wrote slides/data.js")
for s, st in S.items():
    print(f"  {s:11} n={st['n']:2} pass={st['pass']:.3f} CI=[{st['lo']:.2f},{st['hi']:.2f}] score={st['score']:.3f} $/task={st['cost']:.4f}")
print("  paired vs kimi:", paired)
