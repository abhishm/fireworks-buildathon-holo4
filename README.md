# Holo4-style agents on Zapier AutomationBench

Built for the **Fireworks: Deploy (Microsoft × NVIDIA)** buildathon.

[AutomationBench](https://github.com/zapier/AutomationBench) gives an agent a business request (sales, marketing, ops,
support, finance, HR). The agent must carry it out across simulated apps (Gmail, Slack, Sheets, Salesforce, …) using only
API tools. Grading looks only at the final app state. Frontier models pass about 45–50% of tasks at $0.40–$3 per task.
H Company's [Holo4](https://hcompany.ai/newsroom/holo4) showed that a small, well-trained open model can come close for
about $0.05 per task.

**Goal:** get frontier-level pass rates on Fireworks for a fraction of the cost. We combine three pieces:

1. **A Holo-style harness.** It adds an operating procedure to the system prompt and a check-before-finishing step. Both
   run on cheap open models.
2. **LoRA SFT on Holo4 trajectories.** We fine-tune Qwen 3.6 35B-A3B and Nemotron 3.5 Super VL 120B on H Company's
   successful AutomationBench runs ([`Hcompany/trajectories`](https://huggingface.co/datasets/Hcompany/trajectories)),
   using **train tasks only**.
3. **A learned per-task router.** A kNN over task embeddings picks the cheapest system that is likely to succeed.

**Result:** on 60 held-out tasks, the kNN router over five open systems on Fireworks passes **58.3%** vs **46.7%** for
frontier Kimi K3, at **$0.062 vs $0.140 per task (2.2× cheaper)**. LoRA distillation on Fireworks lifted Qwen 3.6 from
6.7% → 31.7% and NVIDIA Nemotron 3.5 Super from 11.7% → 28.3% on the same held-out tasks.
**Story page:** open [`slides/story.html`](slides/story.html) in a browser.

New to the benchmark? Open [`explainers/automationbench.html`](explainers/automationbench.html) in a browser.

## Repository layout

| Path | What it is |
|---|---|
| `harness/harness_env.py` | `HarnessEnv`: the stock `AutomationBenchEnv` with the same tools, 50-step budget, and grader, plus the operating procedure and a "verify before finish" nudge (`HARNESS_MAX_NUDGES`, `HARNESS_PROCEDURE`). |
| `harness/run.py` | Drop-in replacement for `auto-bench` that runs the harness env. Takes the same CLI flags. |
| `ft/build_sft_data.py` | Converts Holo4 trajectories (success only, train480 only) into chat and tool-call SFT conversations. |
| `ft/train_lora.py` | LoRA SFT on Fireworks Serverless Training (rank 32, lr 2e-4, 1 epoch, about 5M tokens per model). |
| `ft/proxy.py`, `ft/serve.sh`, `ft/PROXY.md` | OpenAI-compatible `/v1/chat/completions` proxy over the serverless sampler, so AutomationBench can evaluate LoRA checkpoints. |
| `ft/STATUS.md`, `ft/checkpoints.json`, `ft/logs/` | Training status, loss curves, checkpoint paths, and proxy logs. |
| `slides/` | Slide deck (`index.html`, `deck.pdf`), charts (`img/`), and `make_figures.py`, which regenerates both from `runs/`. |
| `analysis/router.py`, `analysis/systems.json` | Per-task router (kNN and per-domain), cost accounting, paired McNemar tests, and bootstrap CIs. |
| `runs/splits.json` | Fixed task splits: `train480` / `calib60` / `test60` (plus `dev30` for screening). |
| `analysis/complementarity.py`, `analysis/router2.py` | Who-solves-what, greedy union and oracle bounds; router config search selected by calib cross-validation. |
| `analysis/judge.py`, `analysis/cascade.py` | LLM-judge cascade experiment (negative result). |
| `analysis/fill_story.py`, `slides/story.html` | Regenerates `slides/story_data.js` from `runs/`; the one-page story. |
| `runs/` | Eval outputs (`*.json`, incremental `*.jsonl`, `*.log`), task embeddings, router choices, and `results.md`. |
| `screen.sh` | Screens cheap Fireworks models on dev30. |
| `run_evals.sh` | Evaluates DeepSeek (base and harness) on calib60 and test60. Can resume from `*.jsonl`. |
| `run_ft_eval.sh` | Evaluates a proxy-served LoRA model on calib60 and test60. |
| `patches/` | Our local change to AutomationBench, plus the upstream commit it applies to. |
| `WINNING_PLAYBOOK.md` | Event strategy, Fireworks cheat sheet, and evaluation protocol. |

## Setup

```bash
git clone https://github.com/zapier/AutomationBench
git -C AutomationBench checkout "$(cat patches/AUTOMATIONBENCH_COMMIT)"
git -C AutomationBench apply ../patches/automationbench-eval.patch   # AB_INCREMENTAL_JSONL streaming results
(cd AutomationBench && uv sync)

echo 'FIREWORKS_API_KEY=fw_...' > .env      # gitignored
```

The patch adds one feature: if `AB_INCREMENTAL_JSONL` is set, each finished task is appended to that file as one JSON
line. That lets long runs resume and gives you partial results while they're still going.

For fine-tuning, also clone [`fw-ai/cookbook`](https://github.com/fw-ai/cookbook) into `ft/cookbook/`. Then rebuild the
SFT data with `ft/build_sft_data.py`. The data goes to `ft/data/` and the raw trajectories to `data/traj/`; both are
gitignored because they are large and can be rebuilt.

## Running

```bash
./screen.sh                                   # screen candidate models on dev30
./run_evals.sh                                # DeepSeek base vs harness on calib60 / test60

# Harness on any task list (same flags as auto-bench)
cd AutomationBench && set -a; . ../.env; set +a
uv run python ../harness/run.py --model accounts/fireworks/models/deepseek-v4p1-flash \
  --base-url https://api.fireworks.ai/inference/v1 --api-key-var FIREWORKS_API_KEY \
  --tasks finance.invoice_email_extract --export-json ../runs/example.json

# Fine-tuned models: serve a checkpoint, then evaluate it
ft/serve.sh qwen qwen 8311
./run_ft_eval.sh qwen_ft qwen 8311 0.00000045 0.000001335

# Router and statistics
python3 analysis/router.py analysis/systems.json
```

## Evaluation protocol

- **No leakage.** SFT uses only `train480` trajectories (the build step asserts this). The router is fit on `calib60`
  and reported on `test60`.
- **Metric.** We report pass rate, AutomationBench's official metric: all counted assertions must pass. We also report
  average partial credit and real $/task, priced from token usage including cached-input pricing.
- **Statistics.** Paired exact McNemar tests between systems on the same tasks, and 95% bootstrap CIs. With n=60 the CI
  is roughly ±12 points, so small differences are noise.
- **Baselines.** Kimi K3 on `test60` (`runs/test_kimi.json`), plus the published frontier numbers in the AutomationBench
  README.

## Results (held-out test60)

![Approach](slides/img/pipeline.png)

Slides: open [`slides/index.html`](slides/index.html) in a browser (arrow keys to navigate, P to print) or see
[`slides/deck.pdf`](slides/deck.pdf). Running `python3 slides/make_figures.py` rebuilds every chart in `slides/img/` and
every number in the deck from `runs/`.

![Test60 results](slides/img/test60_pass.png)

| System (held-out test60, n=60) | Pass rate [95% CI] | Avg partial | $/task | Paired vs Kimi K3 (sys-only / Kimi-only, McNemar p) |
|---|---|---|---|---|
| **kNN router over 5 open systems (ours)** | **58.3%** [45, 72] | 0.853 | $0.062 | 13 / 6, p=0.17 |
| DeepSeek V4.1 Flash + harness v2 | **58.3%** [45, 72] | 0.856 | $0.060 | 12 / 5, p=0.14 |
| DeepSeek V4.1 Flash (base) | 56.7% [43, 68] | 0.856 | $0.037 | 11 / 5, p=0.21 |
| DeepSeek + harness v1 | 50.0% [37, 63] | 0.831 | $0.063 | 8 / 6, p=0.79 |
| Kimi K3 (reference) | 46.7% [35, 60] | 0.835 | $0.140 | - |
| Qwen3.6-35B-A3B + LoRA | 31.7% [20, 43] | 0.725 | $0.096* | 6 / 15, p=0.078 |
| Nemotron 3.5 Super + LoRA | 28.3% [17, 40] | 0.708 | $0.162* | 5 / 16, p=0.027 |

\* The serverless sampler we used for LoRA eval doesn't apply the prompt-cache discount.

Router selections on test: harness v2 34, plain DeepSeek 17, harness v1 8, Nemotron LoRA 1. The router ties harness v2
alone (gains 3 tasks, loses 3, p=1.0). Perfect routing (oracle) over the same five systems would reach 70.0%.

### Distillation: before vs after LoRA (held-out test60)

Same serverless sampler, renderer and sampling settings (`ft/proxy.py`); the only difference is the LoRA adapter
(rank 32, ~33 steps on 259 Holo4 trajectories from `train480`, about $11 and 5 minutes per model on Fireworks).

| Model | Base | + Holo4 LoRA | Fixed / broken | McNemar p |
|---|---|---|---|---|
| Qwen 3.6 35B-A3B | 6.7% (4/60) | **31.7%** (19/60) | 16 / 1 | < 0.001 |
| NVIDIA Nemotron 3.5 Super 120B | 11.7% (7/60) | **28.3%** (17/60) | 13 / 3 | 0.021 |

Raw runs: `runs/test_qwen_base.json`, `runs/test_nemo_base.json`, `runs/test_qwen_ft.json`, `runs/test_nemo_ft.json`.

### Complementarity

Greedy union on test60: harness v2 35/60 → + plain DeepSeek 40 → + harness v1 41 → + Nemotron LoRA 42 (70%).
On calib60 the Nemotron LoRA is the best second system after v2 (+4 tasks that no DeepSeek variant solved), and on test
it alone solves `sales.invalid_email_cleanup`, which every other system, Kimi K3 included, fails.
`python3 analysis/complementarity.py` regenerates this.

`runs/results.md` (from `analysis/router.py`) has the full table, including a per-domain router that also scores 58.3% at $0.051/task.

**Takeaways**
- **The cheap open model beats the frontier reference.** DeepSeek V4.1 Flash, alone or with harness v2, scores
  10–12 points above Kimi K3 at 2–4× lower cost. With n=60 the gap isn't significant (p≈0.14–0.21).
- **The harness needs tuning.** Harness v1 scored below base (50.0% vs 56.7%); v2 recovered and edged ahead.
- **Distillation works but needs coverage.** ~33 LoRA steps raised Qwen 5× (6.7% → 31.7%) and Nemotron 2.4×
  (11.7% → 28.3%), both significant. They still trail Kimi K3, because Holo4 solved only 289/480 train tasks, so ~40% of
  task types had no demonstration.
- **Routing matches the best system, not more.** The kNN router ties harness v2. The 70% oracle shows real
  complementarity, but which system wins depends on what happens during the run, not on the task text.
- **An LLM judge did not help.** A DeepSeek judge reading v2 trajectories (to trigger a fallback) was tuned on calib
  (AUC 0.66) but was worse than chance on test (AUC 0.42); every cascade lowered pass rate. `analysis/judge.py`,
  `analysis/cascade.py`.
- **Small screens mislead.** DeepSeek scored 66.7% on an early 30-task screen and 56.7% on held-out test60.

### Disclosures

- n=60 per split: 95% CIs are about ±12 points; the router-vs-Kimi gap is suggestive (p=0.17), not significant.
- We chose to headline the router after test results were in. The router itself was fit on `calib60` only and never
  tuned on test, and it scores the same as harness v2 alone, which we also report.
- Harness v2 was written from `calib60` failures; its test run was reported whatever it showed.
- Kimi K3 was run by us at reasoning effort `high`. Holo4's public trajectories may overlap the benchmark (we use only
  `train480` tasks). The harness includes one grader-aware rule (don't pad Slack amounts with `.00`).
- LoRA models were evaluated through the serverless training sampler, which is uncached, so their $/task is overstated.
- Built with an AI coding assistant.

### Grader quirk worth knowing

Slack number matching does **not** accept `12,450.00` when the rubric expects `12,450`. The Gmail and Sheets matchers do
accept it. Agents that pad amounts with `.00` in Slack messages lose points even when the content is correct. See
`rubric/assertions/slack.py` in AutomationBench.

## Acknowledgements

- [Zapier AutomationBench](https://github.com/zapier/AutomationBench): the benchmark, simulator, and grader.
- [H Company Holo4](https://hcompany.ai/newsroom/holo4) and the `Hcompany/trajectories` dataset (Apache-2.0). The
  harness procedure is adapted from their AutomationBench system prompt.
- [Fireworks AI](https://fireworks.ai): inference, serverless LoRA training, and the
  [cookbook](https://github.com/fw-ai/cookbook).
