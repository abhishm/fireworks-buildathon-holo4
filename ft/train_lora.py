"""Serverless LoRA SFT on Holo4 AutomationBench trajectories.

Usage:
  python train_lora.py --key qwen --data data/qwen.jsonl --epochs 1 --batch-size 8 --lr 2e-4
  python train_lora.py --key qwen --data data/qwen.jsonl --max-steps 1 --tag dry   # dry run
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import sys
import time

FT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FT, "cookbook"))
sys.path.insert(0, FT)

from build_sft_data import MODELS  # noqa: E402

PRICE = {"qwen": 2.14, "nemotron": 2.32}  # USD per 1M train tokens
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"


def log(fh, **kw):
    kw["t"] = round(time.time(), 1)
    line = json.dumps(kw)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def update_json(path, key, value):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.load(open(path)) if os.path.exists(path) else {}
        data.setdefault(key, {}).update(value)
        json.dump(data, open(path, "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True, choices=list(MODELS))
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--token-cap", type=int, default=7_000_000, help="abort before exceeding this many train tokens")
    p.add_argument("--tag", default="run")
    p.add_argument("--promote", action="store_true")
    p.add_argument("--checkpoints-file", default=os.path.join(FT, "checkpoints.json"))
    a = p.parse_args()

    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.utils.supervised import build_renderer, load_tokenizer, render_messages_to_datums

    m = MODELS[a.key]
    os.makedirs(os.path.join(FT, "logs"), exist_ok=True)
    fh = open(os.path.join(FT, f"logs/train_{a.key}_{a.tag}.jsonl"), "a")

    tok = load_tokenizer(m["tokenizer"])
    renderer = build_renderer(tok, m["tokenizer"], m["renderer"])
    rows = [json.loads(l) for l in open(a.data)]
    if a.max_rows:
        rows = rows[: a.max_rows]
    datums, ntoks = [], []
    for r in rows:
        rs = render_messages_to_datums(r["messages"], renderer=renderer, tools=r["tools"], reduction="mean")
        assert len(rs) == 1 and len(rs[0].token_ids) <= m["max_seq_len"]
        datums.append(rs[0].datum)
        ntoks.append(len(rs[0].token_ids))
    log(fh, event="data", rows=len(datums), tokens=sum(ntoks), max_len=max(ntoks))

    rng = random.Random(0)
    batches = []
    for ep in range(a.epochs):
        idx = list(range(len(datums)))
        rng.shuffle(idx)
        for i in range(0, len(idx), a.batch_size):
            batches.append((ep, idx[i : i + a.batch_size]))
    if a.max_steps:
        batches = batches[: a.max_steps]
    total_steps = len(batches)
    planned = sum(ntoks[i] for _, b in batches for i in b)
    log(fh, event="plan", steps=total_steps, planned_tokens=planned, est_cost=round(planned / 1e6 * PRICE[a.key], 2))
    if planned > a.token_cap:
        raise SystemExit(f"planned tokens {planned} exceed cap {a.token_cap}")

    def lr_at(step):  # step is 1-based
        if step <= a.warmup:
            return a.lr * step / max(1, a.warmup)
        frac = (step - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=SERVERLESS_URL)
    tc = service.create_lora_training_client(base_model=m["base_model"], rank=a.rank)
    session = getattr(service, "training_session_name", None) or getattr(service, "training_session_id", None)
    log(fh, event="session", session=str(session), run_id=str(tc.run_id), base_model=m["base_model"])
    update_json(a.checkpoints_file, f"{a.key}" if a.tag == "run" else f"{a.key}_{a.tag}", {
        "base_model": m["base_model"], "renderer": m["renderer"], "tokenizer": m["tokenizer"],
        "session": str(session), "run_id": str(tc.run_id), "status": "training"})

    stamp = time.strftime("%H%M")
    short = {"qwen": "q", "nemotron": "n"}[a.key]

    def save(name):
        assert len(name) <= 17, name
        path = tc.save_weights_for_sampler(name).result().path
        log(fh, event="sampler_ckpt", name=name, path=path)
        return path

    cum_tokens, t0 = 0, time.time()
    for s, (ep, b) in enumerate(batches, start=1):
        batch = [datums[i] for i in b]
        btoks = sum(ntoks[i] for i in b)
        weight = sum(float(sum(d.loss_fn_inputs["weights"].data)) for d in batch)
        lr = lr_at(s)
        fb = tc.forward_backward(batch, "cross_entropy")
        opt = tc.optim_step(sdk.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0))
        res = fb.result()
        opt.result()
        cum_tokens += btoks
        loss_sum = res.metrics.get("loss:sum")
        loss = (loss_sum / weight) if (loss_sum is not None and weight) else None
        log(fh, event="step", step=s, total=total_steps, epoch=ep, lr=lr, loss=loss, metrics=res.metrics,
            batch_tokens=btoks, cum_tokens=cum_tokens, est_cost=round(cum_tokens / 1e6 * PRICE[a.key], 3),
            elapsed=round(time.time() - t0, 1))
        if a.save_every and s % a.save_every == 0 and s < total_steps:
            path = save(f"{short}{a.tag[:4]}-s{s}-{stamp}")
            update_json(a.checkpoints_file, f"{a.key}" if a.tag == "run" else f"{a.key}_{a.tag}",
                        {f"sampler_path_step{s}": path})

    path = save(f"{short}{a.tag[:4]}-fin-{stamp}")
    key = f"{a.key}" if a.tag == "run" else f"{a.key}_{a.tag}"
    update_json(a.checkpoints_file, key, {"sampler_path": path, "steps": total_steps, "train_tokens": cum_tokens,
                                          "est_cost_usd": round(cum_tokens / 1e6 * PRICE[a.key], 2), "status": "done"})

    if a.promote:
        try:
            from datetime import datetime
            from fireworks.training.sdk import FireworksClient

            fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
            rows_ = fw.list_training_session_checkpoints(service.training_session_name)
            target = max((r for r in rows_ if r.get("promotable")),
                         key=lambda r: datetime.fromisoformat(r["createTime"].replace("Z", "+00:00")))
            out_id = f"holo-ab-{a.key}-sft-{time.strftime('%m%d%H%M')}"
            model = fw.promote_session_checkpoint(name=target["name"], output_model_id=out_id, base_model=m["base_model"])
            name = getattr(model, "name", None) or (model.get("name") if isinstance(model, dict) else str(model))
            log(fh, event="promoted", checkpoint=target["name"], model=str(name))
            update_json(a.checkpoints_file, key, {"promoted_model": str(name)})
        except Exception as e:  # promotion is best-effort
            log(fh, event="promote_failed", error=repr(e)[:500])
    log(fh, event="done", sampler_path=path, cum_tokens=cum_tokens)


if __name__ == "__main__":
    main()
