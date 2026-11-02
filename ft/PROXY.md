# Serverless-sampler Chat Completions proxy (`ft/proxy.py`)

LoRA checkpoints from Fireworks Serverless Training have no chat endpoint. This proxy gives AutomationBench (or any
OpenAI client) an OpenAI-compatible `POST /v1/chat/completions` endpoint, with tool calling, backed by the
Serverless Training SDK sampler (`FiretitanServiceClient.create_sampling_client`). A sampler-only client uses no
training slot.

Each request goes through these steps:
1. The OpenAI messages and tools are normalized and rendered by the cookbook renderer (`build_tool_prefixed_messages`
   plus `renderer.build_generation_prompt`). This is the same path SFT data takes, so prompt and training formats match.
2. `DeploymentSampler.sample_with_prompt_tokens_result` samples the completion, using a per-task `prompt_cache_key`.
3. `renderer.parse_response` turns the output into `content`, `reasoning_content`, and `tool_calls`.

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

## Launch

```bash
# from the project root (sources .env for FIREWORKS_API_KEY)
ft/serve.sh qwen                      # Qwen 3.6 35B-A3B base         -> :8301, model name qwen_base
ft/serve.sh nemotron                  # Nemotron 3.5 Super VL base    -> :8302, model name nemotron_base
ft/serve.sh qwen qwen 8311            # key from ft/checkpoints.json  -> sampler_path of "qwen"
ft/serve.sh nemotron <acct>/<run>/<ckpt> 8312   # raw save_weights_for_sampler(...).path
ft/serve.sh qwen base 8301 --max-tokens 4096 --temperature 0.7   # extra flags pass through to proxy.py
```

Startup takes about 15–30 s while the tokenizer loads and the sampler route is created. Wait until `curl localhost:PORT/health` responds.
If you pass a checkpoints.json key whose entry has no `sampler_path` yet, `serve.sh` exits with an error.

Run AutomationBench against the proxy. The model name is informational; per-token prices come from the serverless sampling table.
```bash
cd AutomationBench && set -a; . ../.env; set +a
uv run auto-bench --model qwen_base --base-url http://127.0.0.1:8301/v1 --api-key-var FIREWORKS_API_KEY \
  --toolset api --tasks <...> --max-concurrent 8 --export-json ../runs/<name>.json \
  --input-cost 0.00000045 --output-cost 0.000001335        # Qwen 3.6 35B A3B
#  --input-cost 0.00000057 --output-cost 0.00000144        # Nemotron 3.5 Super VL
```

## Presets / flags

| preset | base model | tokenizer | renderer | sampling |
|---|---|---|---|---|
| `qwen` | `accounts/fireworks/models/qwen3p6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | `qwen3_6_interleaved` | T=0.6, top_p=0.95, top_k=20 |
| `nemotron` | `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | `nemotron35_vl` | T=0.6, top_p=0.95 |

Both presets have a 262,144-token context. The renderers match what `ft/train_lora.py` uses for SFT, as recorded in `checkpoints.json`.
`qwen3_6_interleaved` renders byte-exact against the HF chat template and keeps the reasoning of assistant turns after
the last user message. The legacy `qwen3_6` renderer drops that reasoning.

Flags for `proxy.py`. Each one can also be set with a `PROXY_*` env var, as shown in `--help`.
- `--preset`, `--base-model`, `--tokenizer`, `--renderer`: override preset values.
- `--sampler-path`: the checkpoint to sample. Omit it to sample the bare base model (`create_sampling_client(base_model=...)`).
- `--port`, `--host`, `--served-name`, `--log-name`: server and logging settings.
- `--max-tokens` (default 8192): the default and cap for generated tokens per turn. A request's `max_tokens`/`max_completion_tokens` is honored if smaller.
- `--temperature`, `--top-p`, `--top-k`: sampling settings. By default the proxy ignores the request's temperature; pass `--honor-request-temperature` to use it.
- `--concurrency` (default 16): the in-flight sampler requests allowed by `FixedConcurrencyController`.
- `--retries` (default 3): retries with backoff on transient sampler errors.
- `--no-prompt-cache-key`: disables the prompt cache key.
- `--no-coerce-args`: disables tool-argument fix-ups (see below).

## Behaviour details

- **Thinking**: thinking stays on, since it's the native mode for both models. Thinking text is returned in `message.reasoning_content`, never in `content`.
  AutomationBench and verifiers send it back in history, and the renderer places it as the native template does.
  `usage.completion_tokens_details.reasoning_tokens` is filled in.
- **Tool calls**: each call gets a fresh `call_<24hex>` id, `type=function`, and `arguments` as a JSON string. `finish_reason` is `tool_calls`.
  If parsing fails, the proxy returns the raw text as `content` (AutomationBench treats that as the end of the task) and logs `parse_error` plus `raw_completion`.
- **Nemotron argument coercion**: Nemotron writes tool calls in XML with raw text per parameter, so it sometimes sends literal
  `body: "None"`. AutomationBench rejects that with `Invalid JSON`. The proxy now drops `"None"`/`"null"` values when the param is nullable or not a string,
  and JSON-parses string values when the schema doesn't allow strings. Before this fix, one Nemotron smoke run hit 43 `Invalid JSON` tool errors in a single task.
  Qwen emits JSON arguments, so the fix-up is a no-op there.
- **Context limit**: if the prompt is longer than `max_context - 256`, the proxy returns HTTP 400 with "maximum context length … context length exceeded".
  verifiers maps that to `OverlongPromptError`, and AutomationBench ends the rollout gracefully. Otherwise `max_tokens` is clamped to the remaining context.
  The proxy never truncates messages.
- **Usage**: `prompt_tokens` is the rendered prompt length and `completion_tokens` is the number of sampled tokens. `prompt_tokens_details.cached_tokens` comes from sampler server metrics.
- **Logging**: each request appends one line to `ft/logs/proxy_<served_name>.jsonl`. A line holds the token counts, cached tokens, latency, finish reason, tool names, parse errors,
  coerced-arg count, and two cost fields: `cost_usd` at list prefill price and `cost_usd_with_cache` at the cached-prefill price.

## Smoke results (base models, 2 tasks, `--max-concurrent 2`)

Tasks: `finance.overdue_invoice_followup` and `sales.zoom_meeting_followup`. Exports are in `runs/proxy_smoke_{qwen,nemotron}.json`.

| | Qwen 3.6 35B-A3B base | Nemotron 3.5 Super VL base |
|---|---|---|
| avg score (partial) / pass | 19% / 0% (zoom 3/8, finance 0/5) | 38% / 0% (zoom 2/8, finance 4/8) |
| model calls / tool calls | 35 / 51 | 20 / 38 |
| wall time per task | 36 s, 100 s | 25 s, 32 s |
| per-turn latency median / p90 / max | 2.8 s / 4.9 s / 41.7 s* | 1.8 s / 7.5 s / 10.1 s |
| mean prompt / completion tokens per turn | 15.9K / 0.5K | 13.9K / 0.64K |
| prefix-cache hit rate (proxy log) | 75% | 85% |
| cost per task, AB estimate (no cache discount) | $0.04 – $0.18 | $0.05 – $0.08 |
| cost per task, with cached-prefill price (proxy log) | ≈ $0.06 avg | ≈ $0.03 avg |

\* The 41.7 s turn was one runaway-thinking turn that hit the 8,192-token cap (`finish_reason=length`). AutomationBench ends the rollout on a length finish.
An earlier Nemotron run without arg coercion scored 6% and used 1.2M input tokens. It looped on `Invalid JSON` errors, which is what led to the fix.
The second run had no `"None"` args, so part of the difference is sampling variance.

Other checks:
- curl simple chat and tool-call requests work for both models.
- The `qwen_dry` sampler checkpoint loads and returns a correct `api_search` tool call.
- 8 concurrent tool-call requests all returned 200 in 3.5–5.9 s, with 6 s total wall time.

Total testing spend was about $0.40 at cache-aware prices, or about $1.10 at list prices.

## Known limitations

- No streaming (`stream=true` returns HTTP 400; AutomationBench does not stream), `n=1` only, text-only (image parts are not supported), and logprobs are always null.
- The proxy uses the SDK's private `FiretitanSamplingClient._submit` to run the async sampler call on the client's own loop. If you upgrade the SDK, recheck this.
- AutomationBench drops usage for `finish_reason=length` turns, so its token totals can undercount. Use `ft/logs/*.jsonl` as the source of truth for cost.
- AutomationBench's cost estimate uses list prefill price. With prefix caching, serverless bills cached prefill at 20% of list, so real cost is lower; see `cost_usd_with_cache`.
- Base Qwen sometimes thinks past 8K tokens. Raise `--max-tokens` if this matters, at some extra latency and cost.
- The Nemotron renderer (`nemotron35_vl`) was not diffed byte-for-byte against the HF template. It is the same renderer used for SFT, so training and eval formats are consistent.
- Sampling is non-deterministic (T=0.6), so a 2-task smoke test is noisy. Use the full calib/test splits for comparisons.
