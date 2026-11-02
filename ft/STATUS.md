# Holo4 → AutomationBench LoRA SFT status (updated 2026-11-02 12:28 PST)

Data: Hcompany/trajectories, AutomationBench, success==True, **train480 tasks only** (calib60/test60 asserted excluded).
765 convertible trajectories / 289 tasks. Conversation = AB system+user prompt, AB `api` tool schemas, Holo steps as
assistant(reasoning_content=thought, tool_calls) + tool results; update_plan/note dropped; final `answer` → plain assistant content.
Null `params`/`body` omitted from api_fetch args (AB defaults them to None). Loss on assistant tokens only (reasoning + tool calls + final answer).
Selection: <=32K tokens, 1–2 per task, shortest first, ~5M tokens/model.

**Sampling integration:** base models `accounts/fireworks/models/qwen3p6-35b-a3b` (renderer `qwen3_6_interleaved`,
tokenizer `Qwen/Qwen3.6-35B-A3B`) and `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` (renderer
`nemotron35_vl`, tokenizer `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`). Build renderer with
`training.utils.supervised.build_renderer(load_tokenizer(tok), tok, renderer_name)`; tools via `build_tool_prefixed_messages`.
Training kept reasoning (`<think>`) on every assistant turn and used AB's system prompt + tools unchanged.

Hyperparams: LoRA rank 32 (alpha 32), lr 2e-4 (3-step warmup, cosine to 4e-5), batch 8 conversations, 1 epoch, Adam(0.9,0.95,eps 1e-8, wd 0), cross_entropy (per-sequence mean).

## qwen
- dataset: data/qwen.jsonl — 259 convs, 257 tasks, 4991770 tokens (1528425 assistant/trained), max len 31477
- progress: step 33/33 — 4991770 tokens trained, est $10.68 (plan $10.68)
- loss by step: [0.3305, 0.319, 0.3225, 0.2706, 0.2436, 0.3312, 0.2132, 0.3119, 0.2796, 0.2469, 0.3145, 0.2218, 0.4444, 0.2343, 0.24, 0.2849, 0.2346, 0.2182, 0.2619, 0.2987, 0.2772, 0.3322, 0.3755, 0.2541, 0.2028, 0.2508, 0.2077, 0.2622, 0.2204, 0.2167, 0.2585, 0.2881, 0.4286]
- {'event': 'session', 'session': 'accounts/FW_ACCOUNT_A/trainingSessions/ts-db61efadbaef4623a7501daa0d6068c9', 'run_id': 'run-bfc18cdcb47a4873a000bda68f0b861d', 'base_model': 'accounts/fireworks/models/qwen3p6-35b-a3b'}
- {'event': 'sampler_ckpt', 'name': 'qrun-s12-1212', 'path': 'FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-s12-1212-d07d5db4'}
- {'event': 'sampler_ckpt', 'name': 'qrun-s24-1212', 'path': 'FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-s24-1212-d07d5db4'}
- {'event': 'sampler_ckpt', 'name': 'qrun-fin-1212', 'path': 'FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-fin-1212-d07d5db4'}
- {'event': 'promoted', 'checkpoint': 'accounts/FW_ACCOUNT_A/trainingSessions/ts-db61efadbaef4623a7501daa0d6068c9/checkpoints/run-bfc18cdcb47a4873a000bda68f0b861d-qrun-fin-1212-d07d5db4', 'model': 'accounts/FW_ACCOUNT_A/models/holo-ab-qwen-sft-11021227'}

## nemotron
- dataset: data/nemotron.jsonl — 254 convs, 253 tasks, 4996653 tokens (1498953 assistant/trained), max len 31414
- progress: step 32/32 — 4996653 tokens trained, est $11.59 (plan $11.59)
- loss by step: [0.2722, 0.4139, 0.3256, 0.2898, 0.1802, 0.2676, 0.2487, 0.2819, 0.3133, 0.2841, 0.1909, 0.2686, 0.3017, 0.2592, 0.2142, 0.2579, 0.2789, 0.2661, 0.2496, 0.2888, 0.2065, 0.2503, 0.2223, 0.2654, 0.3113, 0.2615, 0.292, 0.1739, 0.2585, 0.2216, 0.2094, 0.2106]
- {'event': 'session', 'session': 'accounts/FW_ACCOUNT_A/trainingSessions/ts-a4a19ca23742418aa9c0d2bfa23c036b', 'run_id': 'run-85f0368aeace4c9082d05b789d0c167f', 'base_model': 'accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16'}
- {'event': 'sampler_ckpt', 'name': 'nrun-s12-1212', 'path': 'FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-s12-1212-db88363c'}
- {'event': 'sampler_ckpt', 'name': 'nrun-s24-1212', 'path': 'FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-s24-1212-db88363c'}
- {'event': 'sampler_ckpt', 'name': 'nrun-fin-1212', 'path': 'FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-fin-1212-db88363c'}
- {'event': 'promoted', 'checkpoint': 'accounts/FW_ACCOUNT_A/trainingSessions/ts-a4a19ca23742418aa9c0d2bfa23c036b/checkpoints/run-85f0368aeace4c9082d05b789d0c167f-nrun-fin-1212-db88363c', 'model': 'accounts/FW_ACCOUNT_A/models/holo-ab-nemotron-sft-11021221'}

**Spend estimate:** $22.34 (incl. ~$0.07 dry runs/sampling check). Budget $30.

## checkpoints.json
```json
{
  "nemotron_dry": {
    "base_model": "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16",
    "renderer": "nemotron35_vl",
    "tokenizer": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "session": "accounts/FW_ACCOUNT_A/trainingSessions/ts-fa3b9652465d4fabb2d9df822cf03fe9",
    "run_id": "run-4b9481cc8edc4dd2af25bf2f69895e70",
    "status": "done",
    "sampler_path": "FW_ACCOUNT_A/run-4b9481cc8edc4dd2af25bf2f69895e70/ndry-fin-1211-aa6b5192",
    "steps": 1,
    "train_tokens": 15611,
    "est_cost_usd": 0.04
  },
  "qwen_dry": {
    "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
    "renderer": "qwen3_6_interleaved",
    "tokenizer": "Qwen/Qwen3.6-35B-A3B",
    "session": "accounts/FW_ACCOUNT_A/trainingSessions/ts-86057b21d7a34a189fe28acd4e829fb4",
    "run_id": "run-a4d32e5f14094c1385cd99cfc407a9c0",
    "status": "done",
    "sampler_path": "FW_ACCOUNT_A/run-a4d32e5f14094c1385cd99cfc407a9c0/qdry-fin-1211-ef93dad9",
    "steps": 1,
    "train_tokens": 15219,
    "est_cost_usd": 0.03
  },
  "nemotron": {
    "base_model": "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16",
    "renderer": "nemotron35_vl",
    "tokenizer": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "session": "accounts/FW_ACCOUNT_A/trainingSessions/ts-a4a19ca23742418aa9c0d2bfa23c036b",
    "run_id": "run-85f0368aeace4c9082d05b789d0c167f",
    "status": "done",
    "sampler_path_step12": "FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-s12-1212-db88363c",
    "sampler_path_step24": "FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-s24-1212-db88363c",
    "sampler_path": "FW_ACCOUNT_A/run-85f0368aeace4c9082d05b789d0c167f/nrun-fin-1212-db88363c",
    "steps": 32,
    "train_tokens": 4996653,
    "est_cost_usd": 11.59,
    "promoted_model": "accounts/FW_ACCOUNT_A/models/holo-ab-nemotron-sft-11021221"
  },
  "qwen": {
    "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
    "renderer": "qwen3_6_interleaved",
    "tokenizer": "Qwen/Qwen3.6-35B-A3B",
    "session": "accounts/FW_ACCOUNT_A/trainingSessions/ts-db61efadbaef4623a7501daa0d6068c9",
    "run_id": "run-bfc18cdcb47a4873a000bda68f0b861d",
    "status": "done",
    "sampler_path_step12": "FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-s12-1212-d07d5db4",
    "sampler_path_step24": "FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-s24-1212-d07d5db4",
    "sampler_path": "FW_ACCOUNT_A/run-bfc18cdcb47a4873a000bda68f0b861d/qrun-fin-1212-d07d5db4",
    "steps": 33,
    "train_tokens": 4991770,
    "est_cost_usd": 10.68,
    "promoted_model": "accounts/FW_ACCOUNT_A/models/holo-ab-qwen-sft-11021227"
  }
}
```

## Caveats
- Loss is noisy (batch=8 long trajectories); mean loss fell roughly 0.30 → 0.25 (qwen) and 0.30 → 0.21 (nemotron) over 1 epoch. Base models were already close to Holo's style (start loss ~0.3).
- Only trajectories <=32K tokens were used (shortest per task), so very long tasks are under-represented.
- Training sessions for the real runs are left open (deleting a session removes checkpoint access via session APIs); dry-run sessions were deleted.
- Promoted models: accounts/FW_ACCOUNT_A/models/holo-ab-qwen-sft-11021227, accounts/FW_ACCOUNT_A/models/holo-ab-nemotron-sft-11021221 (not deployed).
- Sampling sanity-check of final Nemotron checkpoint on an unseen task produced Holo-style parallel api_search calls in the native tool format.
