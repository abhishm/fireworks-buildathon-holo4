# Fireworks: Deploy (Microsoft × NVIDIA): winning playbook

Nov 2, 2026, TERRA, 511 Harrison St, SF. Build window: **10:30 AM to 4:30 PM PT**. **Start submitting at 3:30 PM**; aim to be done by 4:00 PM. The video is due by 4:00 PM.

Sources: [Builder Guide](https://hackersquad.io/events/fireworks-deploy-with-microsoft-nvidia/resources), [Rules](https://hackersquad.io/events/fireworks-deploy-with-microsoft-nvidia/resources?tab=rules), [Submission & judging](https://hackersquad.io/events/fireworks-deploy-with-microsoft-nvidia/resources?tab=submission), the [official sample submission](https://app.notion.com/p/Forge-Deploy-Sample-Submission-4093715e8b9b80e3beefcd833c37fa90) written by Sinan Ozdemir (a judge), and the [Fireworks Serverless Training docs](https://docs.fireworks.ai/fine-tuning/training-api/serverless).

---

## 0. The ten facts that decide this event

1. **Benchmark evidence is worth 40%.** "A strong presentation or complex implementation cannot substitute for benchmark evidence."
2. **A frontier-model baseline is REQUIRED by the rules (§3).** Name the frontier model and run it on the *same* eval cases. The sample submission says closed models are "optional" and compares open-model before vs after. **The rules override the sample, so do both:** frontier baseline + base open model + your improved open model.
3. **"Improvement" can be quality, cost or latency.** Cost and latency wins must keep *useful* quality, and you must disclose the tradeoffs. Being cheaper or faster at near-equal quality counts.
4. **Prizes require Fireworks** for the overall prizes ($10K / $5K / $3K) and for the NVIDIA DGX Spark award. Fireworks is technically "optional" under the rules, but you need it to win anything.
5. **NVIDIA Best-Use Award: use Nemotron on Fireworks.** *NVIDIA Nemotron 3.5 Super VL 120B* is in the serverless training pool, and it is the cheapest large model there (50% promo). Sagar Singh (NVIDIA) is a judge and gave the talk "Nemotron LoRA on Fireworks." You can win a placement prize **and** a sponsor prize.
6. **Team of 2–5 members, all present in person.** Solo entries cannot compete. Every member must be listed. Finalists present live, so someone must stay for the awards.
7. **Public GitHub repo plus a commit hash.** If the repo is private or unreachable, the project is **unscored** and gets no prize. Test the link in an incognito window.
8. **Demo video under 120 s.** Aim for under 60 s.
9. **Six finalists are chosen**, then all judges re-score them live with the same rubric.
10. **Honesty is rewarded:** report your losses, caveats and failed ideas. Fabricated results, cherry-picking or a deliberately weakened baseline get you disqualified.

---

## 1. Rubric → what to build

| Criterion | Weight | How to max it |
|---|---|---|
| Benchmark evaluation | **40%** | 100+ held-out, varied cases (aim for 300–1,000). Same cases, same prompt and temperature 0 for every system. Frontier baseline plus base open model plus your system. **McNemar test + cluster bootstrap 95% CI.** Report cost and latency per system too. |
| Business value | **25%** | A named user and their headcount, with a source (for example BLS). Dollar value with the assumptions written out. Published evidence that frontier models fail at this task. A reason open weights matter (cost, privacy, data residency, export control). |
| Technical execution | 20% | A working pipeline: label audit, then baseline, then ≥2 ideas (harness/prompt **and** LoRA fine-tune), with the failures reported. Cached per-case results. |
| Presentation & demo | 10% | Video: problem → frontier/base model fails on one example → yours gets it right → the chart → one number + one caveat. |
| Documentation | 5% | README with run order, cost per step, pinned deps, `.env.example`, saved results judges can read without rerunning. |

The sample submission scores itself against this rubric with a "where it would lose points" column. **Copy that table into your README**; judges like seeing the self-assessment.

---

## 2. Picking the task (do this in under 30 minutes)

A winning task has all of these properties:

- **A clear right answer**, so it can be scored automatically: exact match, numeric, JSON field F1, yes/no, or a classification label. Avoid LLM-as-judge if you can. If you must use one, validate it against human labels.
- **Real, trustworthy labels** from a published dataset or paper. **Audit the labels first:** the sample found that the Hugging Face copy of FloorPlanCAD had scrambled class names.
- **Frontier models are measurably mediocre** at it (ideally a cited benchmark shows this). Look for headroom in small-detail perception, domain jargon, strict output formats, long structured extraction, or niche classification.
- **Enough data:** about 1–5K training examples and 300–1,000 held-out ones. Split by **group** (document, building, user, company), not by row, to avoid leakage.
- **A named, paid user** whose time costs money: estimators, paralegals, claims adjusters, coders, analysts or support agents.
- **An open-weights advantage** you can argue: cost at scale, latency, data cannot leave the premises, export control.

Don't use FloorPlanCAD or engineering-drawing extraction: those are the sample projects, and judges have already seen them.

Good shapes for a ~5-hour build:
- **Vision extraction/counting** on documents, charts, receipts, forms or diagrams. This uses Nemotron 3.5 Super VL, which also targets the NVIDIA award.
- **Structured extraction to JSON** from domain text (contracts, filings, clinical notes), scored by per-field F1.
- **Domain classification or routing** (a support ticket queue, a legal clause type, a prompt router), scored by accuracy or macro-F1. Fireworks has an official [prompt-router serverless notebook](https://github.com/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb) you can adapt.
- **Cost/latency play:** a small fine-tuned open model matches the frontier model (within the CI) at 10–40× lower cost, with a distribution of latencies.

Write the one-paragraph business case **before** writing code: who the user is, how many there are, how many hours they spend, the dollar value, why frontier models fail, and why an open model fits.

---

## 3. Time plan (adjusted, it is already about 10:50 AM)

| Time | Do |
|---|---|
| 10:50–11:20 | Lock in team (2–5 members). Pick the task and dataset, write the business case, audit the labels. Create the **public** GitHub repo now. |
| 11:20–12:00 | Freeze the test split (by group). Write `eval.py`. Run the **base open model + frontier baseline** with one saved JSONL row per case. |
| **By 12:00** | **Launch the LoRA fine-tune** in the background. It takes about 1–1.5 h including scoring. |
| 12:00–1:30 | Harness ideas on the base model while training runs: list-then-answer, few-shot, self-consistency vote, schema-constrained JSON. |
| 1:30–2:30 | Score every system on the same test set. Run McNemar + cluster bootstrap. Make two charts: the leaderboard and the disagreements/CI. Measure cost and latency per system. |
| 2:30–3:30 | README, rubric self-assessment table, disclosure section, 60-s video, incognito check of the repo. |
| **3:30–4:00** | **Submit** on HackerSquad (project form → save → upload video → **Submit for judging** → confirm the status shows "submitted"). Saving a draft is not submitting. |
| 4:30 | Hard stop. No new features or results after this. |

Split the team: (A) data, labels and eval harness; (B) fine-tune; (C) frontier baseline, cost and latency, then the README and video. With 2 people, combine A+C.

---

## 4. Fireworks technical cheat-sheet

### Inference (baseline and harness), OpenAI-compatible
```python
from openai import AsyncOpenAI
import os
client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1",
                     api_key=os.environ["FIREWORKS_API_KEY"])
resp = await client.chat.completions.create(
    model="accounts/fireworks/models/<model-id>",
    messages=msgs, temperature=0.0, max_tokens=4096)
# resp.usage → prompt/completion tokens → compute $ per case
```
Use an `asyncio.Semaphore` (about 16–32) for concurrency. Cache by case id so reruns are free. Record `time.perf_counter()` around each call to get latency.

### Serverless LoRA training (no provisioning, per-token billing)
```bash
pip install "fireworks-ai[training]>=1.2.29"   # Python ≥ 3.11
```
```python
from fireworks.training.sdk import FiretitanServiceClient, AdamParams
service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
    base_url="https://api.fireworks.ai/training/v1/serverless")   # MUST end in /training/v1/serverless
tc = service.create_lora_training_client(base_model=BASE, rank=16)  # session id starts with ts-
for batch in batches(datums, 16):
    tc.forward_backward(batch, "cross_entropy").result()           # ALWAYS .result()
    tc.optim_step(AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)).result()
path = tc.save_weights_for_sampler("final").result().path         # names ≤ 17 chars, unique
sampler = service.create_sampling_client(model_path=path, tokenizer=tok)
```
Settings from the sample: rank 16, alpha 32, LR 1e-4, 1 epoch, batch 16, ~3,000 examples. SFT target format: a short reasoning sentence followed by the answer.

**Gotchas from the docs:**
- Call `.result()` on every future, or failures pass silently.
- The **tokenizer must match the base model**, or results are silently corrupted.
- Keep checkpoint names to **17 characters or fewer** and make each unique. A name that is too long fails only at *promote* time.
- **Promote before you delete the session.** After deletion, checkpoints return NOT_FOUND.
- The default limit is 8 concurrent runs per account, and some models allow 1 run per model. Delete finished sessions to free slots.
- An `out of capacity` error means the shared pool is full. Retry or pick another base model.
- The sampler is not a chat endpoint. To measure **serving cost and latency**, promote the checkpoint and run `firectl deployment create accounts/<acct>/models/<id> --deployment-shape default`, then call it through chat completions. (The sample lost points for not measuring this.) Delete the deployment afterward.

### Serverless training pool (USD per 1M tokens; verify in the [catalog](https://docs.fireworks.ai/fine-tuning/models))
| Model | Ctx | Prefill | Sample | Train |
|---|---|---|---|---|
| **NVIDIA Nemotron 3.5 Super VL 120B** (vision, 50% promo) | 256K | 0.57 | 1.44 | 2.32 |
| Qwen 3.6 35B A3B | 256K | 0.45 | 1.335 | 2.14 |
| Qwen 3.8 27B | 128K | 1.86 | 5.595 | 4.103 |
| DeepSeek V4.1 Flash 1017 | 256K | 0.60 | 3.60 | 4.98 |
| GLM 5.3 Flash (used in the sample, vision) | 195K | 2.96 | 7.41 | 8.89 |
| Muse Glimmer 30B | 128K | 1.96 | 4.88 | 5.86 |
| GLM 5.3 / Kimi K3 / Muse Spark Open | — | pricier | | |

**Recommendation:** use Nemotron 3.5 Super VL as the main fine-tune target. It's cheap, it handles vision, and it qualifies for the NVIDIA award. Add Qwen 3.6 35B A3B as a cheap second open model if time allows.

### Other references
- Cookbook: https://github.com/fw-ai/cookbook (`training/recipes`, `training/examples`, `case-studies`), plus agent skills (research/configure/debug) you can add to Cursor, Claude Code or Codex.
- Model IDs: check https://app.fireworks.ai/models for the exact `accounts/fireworks/models/...` string.
- Frontier baseline: call the provider directly or use OpenRouter, as the sample did. **Microsoft Foundry** works too and suits the Microsoft judge (Jay Sen). The sample's frontier models (Gemini 4 Argon, Claude Opus 5.5, GPT-6.1 Terra Pro) scored 70.5–76.8% on its task.

---

## 5. Evaluation protocol (where 40% is won)

1. **Freeze the test set first** (by group, ideally 300–1,000 cases), and commit `test_ids.json` to the repo.
2. **Keep it fair:** the same prompt, temperature 0, max_tokens, parsing and scorer for every system. Never tune on test data. Use a separate dev split for harness and hyperparameter choices.
3. **Save one JSONL row per (system, case):**
   `{"id","group","system","answer","pred","score","latency_ms","prompt_tokens","completion_tokens","cost_usd"}`
4. **Run these systems:** frontier baseline(s) · base open model · harness variants · LoRA fine-tune (· fine-tune + best harness).
5. **Statistics:** run both tests, and claim a gain only when they agree.

```python
import numpy as np, pandas as pd
from scipy.stats import binomtest

def mcnemar(a, b):                    # a,b: aligned 0/1 arrays (same case order)
    b_only = int(((a == 0) & (b == 1)).sum()); a_only = int(((a == 1) & (b == 0)).sum())
    p = binomtest(b_only, b_only + a_only, 0.5).pvalue if b_only + a_only else 1.0
    return b_only, a_only, p          # "new wins / old wins / exact p"

def cluster_bootstrap(df, sys_a, sys_b, n=5000, seed=0):
    # df: columns id, group, system, score; resample GROUPS, not rows
    rng = np.random.default_rng(seed)
    w = df.pivot_table(index=["group", "id"], columns="system", values="score").reset_index()
    groups = w["group"].unique(); by_g = {g: w[w.group == g] for g in groups}
    gaps = []
    for _ in range(n):
        s = pd.concat([by_g[g] for g in rng.choice(groups, len(groups), replace=True)])
        gaps.append(s[sys_b].mean() - s[sys_a].mean())
    gaps = np.array(gaps)
    return gaps.mean(), np.percentile(gaps, [2.5, 97.5]), (gaps > 0).mean()
```
- For **pass/fail** scores, report both tests. For a **continuous** score (F1, cost, latency), use the bootstrap only, or run McNemar on per-field pass/fail pieces.
- With **few groups** (under 20), lean on McNemar over many small items.
- **Never use McNemar alone on grouped data.** It is overconfident there.

6. **Headline table:** system | accuracy (95% CI) | Δ vs frontier | Δ vs base | McNemar p | $/1K cases | p50/p95 latency.
7. **Two charts:** (a) the leaderboard with CIs, including the frontier models; (b) the disagreement bars (wins vs losses per change) or the bootstrap gap distribution.
8. **Error analysis:** report the slice where the base model fails (the sample's "sinks at 38%") and how the fix moved it ("38% → 85%"). This story sells the result.
9. **Report the losers too.** In the sample, both harnesses *lost* to the plain prompt, and the report says why. That shows you tested alternatives.

---

## 6. Submission checklist (HackerSquad project form)

- [ ] Project name and **overview**: task, user, approach, business value
- [ ] **Public GitHub repo** link and the **commit hash** of the judged version, tested in incognito
- [ ] README: problem, user and value math (with assumptions), dataset source and size and license, split method, models and configs, results table, charts, statistical tests, **limitations and tradeoffs**, run order, cost per step, `requirements.txt` pinned, `.env.example` (no keys committed!)
- [ ] **Disclosure section:** pre-existing code or models, AI coding assistants used (rules §4), and what was built today
- [ ] Rubric self-assessment table ("what we show / where we'd lose points")
- [ ] **Demo video under 120 s** (target under 60 s), uploaded and *processed*, by 4:00 PM
- [ ] All team members listed. Technologies: Fireworks (serverless training plus inference), NVIDIA Nemotron
- [ ] **Sponsor track:** state that you are entering the NVIDIA Best-Use Award and explain exactly how Nemotron and Fireworks are used
- [ ] Click **Submit for judging** and confirm the status shows "submitted"

### 60-second video script
1. (0–10 s) "<User> spends <N> hours on <task>; frontier models get it right only <X>%."
2. (10–30 s) One example: the frontier/base model misses it, and our fine-tuned Nemotron gets it right.
3. (30–50 s) The leaderboard chart: "+Y points over <frontier>, p = …, 95% CI [a, b], at 1/Z the cost."
4. (50–60 s) One caveat and the repo link.

### Finals (if you're one of the six)
Expect questions about **leakage** (how you split), **baseline fairness** (prompt parity), **statistical significance**, **serving cost**, and **what you built today vs. what existed before**. Prepare a single slide that shows the headline table.

---

## 7. Judges worth knowing (18 total)
- **Sinan Ozdemir** (Fireworks, Head of AI Education): wrote the sample and gave the talk "Making your own frontier on Fireworks." Optimize for *his* playbook: label audit, group split, McNemar plus bootstrap, honest losses.
- **Sagar Singh** (NVIDIA, Senior Solutions Architect): Nemotron LoRA training, evaluation and tradeoffs. Show Nemotron LoRA results and their tradeoffs.
- **Jayanta (Jay) Sen** (Microsoft, Principal Solution Engineer): Microsoft Foundry and trusted agentic AI. Trust, governance and deployment framing helps here.
- Others: nborenszteinn, rohrao, murkum, Sai Krishna, Sunny Bakhda, Jacob Peake, Shahul Es, Kshitiz Parashar, Jetashree Ravi, Gana Chandrasekaran, Eldar Hasanov, Prosenjit Das, Adam Chan.

## 8. Mistakes to avoid
- Skipping the frontier baseline (it is required by the rules).
- Testing on data the model trained on, or splitting by row when examples are grouped.
- Running a weaker prompt or different settings for the baseline.
- Leaving the repo private or with a wrong link (the submission is unscored), committing API keys, or forgetting the commit hash.
- Uploading the video at 4:25 when processing isn't finished; that does not count.
- Choosing a task without a ground-truth scorer, which means no credible 40%.
- Starting the fine-tune after 1 PM; it won't finish with time left to evaluate.
- Leaving deployments or sessions running, which burns credits. Clean up after promoting.
