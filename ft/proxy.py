#!/usr/bin/env python3
"""OpenAI-compatible Chat Completions proxy over the Fireworks Serverless Training sampler.

Serverless-training LoRA checkpoints (``save_weights_for_sampler(...).path``) have no
chat endpoint, only a token-in/token-out sampler. This proxy renders OpenAI chat
requests (messages + tools) with the cookbook renderer (same code the SFT data
uses), samples tokens through ``FiretitanServiceClient.create_sampling_client``,
and parses the sampled tokens back into an assistant message with tool calls.

    python ft/proxy.py --preset qwen [--sampler-path <acct>/<run>/<ckpt>] --port 8301
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cookbook"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

log = logging.getLogger("proxy")

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"

PRESETS: dict[str, dict[str, Any]] = {
    "qwen": {
        "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B",
        # HF-template-exact for Qwen3.6 (keeps reasoning of assistant turns after
        # the last user message). Legacy "qwen3_6" drops it.
        "renderer": "qwen3_6_interleaved",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_context": 262144,
        "served_name": "qwen3p6-35b-a3b-sampler",
        "prefill_per_m": 0.45,
        "sample_per_m": 1.335,
        "cached_per_m": 0.09,
    },
    "nemotron": {
        "base_model": "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16",
        "tokenizer": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "renderer": "nemotron35_vl",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": None,
        "max_context": 262144,
        "served_name": "nemotron-3p5-super-sampler",
        "prefill_per_m": 0.57,
        "sample_per_m": 1.44,
        "cached_per_m": 0.114,
    },
}

THINK_TAG_RE = re.compile(r"</?think>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default=os.environ.get("PROXY_PRESET", "qwen"))
    p.add_argument("--base-model", default=os.environ.get("PROXY_BASE_MODEL"))
    p.add_argument("--tokenizer", default=os.environ.get("PROXY_TOKENIZER"), help="HF tokenizer id")
    p.add_argument("--renderer", default=os.environ.get("PROXY_RENDERER"), help="cookbook renderer name")
    p.add_argument(
        "--sampler-path",
        default=os.environ.get("PROXY_SAMPLER_PATH") or None,
        help="save_weights_for_sampler(...).path; omit to sample the base model",
    )
    p.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "8301")))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("PROXY_MAX_TOKENS", "8192")),
                   help="default/cap for generated tokens per turn")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--honor-request-temperature", action="store_true",
                   help="use the request's temperature/top_p when provided (default: proxy settings win)")
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--served-name", default=os.environ.get("PROXY_SERVED_NAME"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("PROXY_CONCURRENCY", "16")))
    p.add_argument("--no-prompt-cache-key", action="store_true")
    p.add_argument("--no-coerce-args", action="store_true", help="disable schema-aware \"None\"->null tool-arg fix-ups")
    p.add_argument("--log-name", default=None, help="logs/proxy_<name>.jsonl")
    p.add_argument("--retries", type=int, default=3, help="sampler retries per request on transient errors")
    args = p.parse_args(argv)
    preset = PRESETS[args.preset]
    for key in ("base_model", "tokenizer", "renderer", "max_context", "served_name"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    if args.temperature is None:
        args.temperature = preset["temperature"]
    if args.top_p is None:
        args.top_p = preset["top_p"]
    if args.top_k is None:
        args.top_k = preset["top_k"]
    args.prefill_per_m = preset["prefill_per_m"]
    args.sample_per_m = preset["sample_per_m"]
    args.cached_per_m = preset["cached_per_m"]
    if args.log_name is None:
        args.log_name = args.served_name
    return args


class Backend:
    def __init__(self, args: argparse.Namespace):
        from fireworks.training.sdk import FiretitanServiceClient
        from fireworks.training.sdk.concurrency import FixedConcurrencyController
        from training.renderer import get_renderer
        from training.utils.tokenizers import load_tokenizer

        self.args = args
        self.tokenizer = load_tokenizer(args.tokenizer)
        self.renderer = get_renderer(args.renderer, self.tokenizer)
        stops = self.renderer.get_stop_sequences()
        self.stop_token_ids = [s for s in stops if isinstance(s, int)]
        self.stop_strings = [
            self.tokenizer.decode([s], skip_special_tokens=False) if isinstance(s, int) else str(s) for s in stops
        ]
        api_key = os.environ["FIREWORKS_API_KEY"]
        self.service = FiretitanServiceClient(api_key=api_key, base_url=SERVERLESS_URL)
        controller = FixedConcurrencyController(args.concurrency)
        if args.sampler_path:
            self.sampler = self.service.create_sampling_client(
                model_path=args.sampler_path, tokenizer=self.tokenizer, concurrency_controller=controller
            )
        else:
            self.sampler = self.service.create_sampling_client(
                base_model=args.base_model, tokenizer=self.tokenizer, concurrency_controller=controller
            )
        log.info("sampler model=%s", self.sampler.deployment_sampler.model)

    def render(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        from training.utils.supervised import build_tool_prefixed_messages

        msgs = build_tool_prefixed_messages(_clean_messages(messages), renderer=self.renderer, tools=tools or None)
        return list(self.renderer.build_generation_prompt(msgs).to_ints())

    async def sample(self, prompt_ids: list[int], *, max_tokens: int, temperature: float, top_p: float | None,
                     top_k: int | None, cache_key: str | None):
        kwargs: dict[str, Any] = {}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        coro = self.sampler.deployment_sampler.sample_with_prompt_tokens_result(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.stop_strings or None,
            prompt_cache_key=cache_key,
            **kwargs,
        )
        # Run on the sampling client's own event loop (it owns the HTTP session).
        return await asyncio.wrap_future(self.sampler._submit(coro))

    def parse(self, tokens: list[int], finish_reason: str) -> tuple[dict, str | None]:
        """Return (openai_message, parse_error)."""
        toks = list(tokens)
        if finish_reason != "length" and self.stop_token_ids and (not toks or toks[-1] not in self.stop_token_ids):
            toks.append(self.stop_token_ids[0])
        parse_error = None
        try:
            message, termination = self.renderer.parse_response(toks)
        except Exception as exc:  # noqa: BLE001
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return {"role": "assistant", "content": _strip_think(text)}, f"parse_exception: {exc!r}"
        content = message.get("content")
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        else:
            for part in content or []:
                if part.get("type") == "thinking":
                    reasoning_parts.append(part.get("thinking", ""))
                elif part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts).strip()
        # Fallback: unparsed <think> blocks left in text.
        if "</think>" in text:
            head, _, tail = text.rpartition("</think>")
            reasoning = (reasoning + "\n" + head.replace("<think>", "")).strip()
            text = tail
        text = _strip_think(text).strip()
        out: dict[str, Any] = {"role": "assistant", "content": text or None}
        if reasoning:
            out["reasoning_content"] = reasoning
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            args = tc.function.arguments
            if not isinstance(args, str):
                args = json.dumps(args)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": tc.function.name, "arguments": args},
            })
        unparsed = message.get("unparsed_tool_calls") or []
        if unparsed:
            parse_error = "unparsed_tool_calls: " + "; ".join(u.error for u in unparsed)[:500]
            if not tool_calls:
                raw = "\n".join(u.raw_text for u in unparsed)
                out["content"] = ((text + "\n") if text else "") + raw
        if tool_calls:
            out["tool_calls"] = tool_calls
        if termination is not None and not getattr(termination, "is_clean", True) and parse_error is None:
            if finish_reason != "length":
                parse_error = f"termination={termination}"
        if not out["content"] and not tool_calls and not reasoning:
            out["content"] = ""
        return out, parse_error


_NULL_STRINGS = {"None", "null", "NULL", "none"}


def _schema_types(schema: dict) -> set[str]:
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            if isinstance(sub, dict):
                types |= _schema_types(sub)
    return types


def coerce_tool_arguments(message: dict, tools: list[dict] | None) -> int:
    """Schema-aware fix-ups for XML-style tool calls (Nemotron emits raw text per parameter).

    - literal "None"/"null" for a nullable or non-string param -> drop it (== default None)
    - JSON-looking string for a param whose schema doesn't allow string -> parsed value
    Returns number of arguments changed.
    """
    if not message.get("tool_calls"):
        return 0
    props_by_tool = {}
    for t in tools or []:
        fn = t.get("function") or {}
        props_by_tool[fn.get("name")] = ((fn.get("parameters") or {}).get("properties") or {})
    changed = 0
    for tc in message["tool_calls"]:
        fn = tc["function"]
        try:
            args = json.loads(fn["arguments"])
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(args, dict):
            continue
        props = props_by_tool.get(fn["name"], {})
        new_args = {}
        for k, v in args.items():
            types = _schema_types(props.get(k) or {})
            if isinstance(v, str):
                vs = v.strip()
                if vs in _NULL_STRINGS and (not types or "null" in types or "string" not in types):
                    changed += 1
                    continue
                if types and "string" not in types and vs[:1] in "[{0123456789-tf":
                    try:
                        v = json.loads(vs)
                        changed += 1
                    except Exception:  # noqa: BLE001
                        pass
            new_args[k] = v
        if new_args != args:
            fn["arguments"] = json.dumps(new_args)
    return changed


def _strip_think(text: str) -> str:
    return THINK_TAG_RE.sub("", text)


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Coerce OpenAI wire messages into what the cookbook normalizer accepts."""
    cleaned = []
    for m in messages:
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            m["role"] = "system"
        content = m.get("content")
        if content is None:
            m["content"] = ""
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            m["content"] = "".join(texts)
        if m.get("reasoning_content") is None:
            m.pop("reasoning_content", None)
        if m.get("reasoning") is None or not isinstance(m.get("reasoning"), str):
            m.pop("reasoning", None)
        if m.get("tool_calls") == []:
            m.pop("tool_calls")
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        json.loads(fn["arguments"])
                    except Exception:  # noqa: BLE001
                        fn["arguments"] = json.dumps({"raw": fn["arguments"]})
                tc["function"] = fn
                tc.setdefault("type", "function")
                tcs.append(tc)
            m["tool_calls"] = tcs
        for k in ("refusal", "annotations", "audio", "function_call"):
            m.pop(k, None)
        cleaned.append(m)
    return cleaned


def _cache_key(messages: list[dict], tools: list[dict] | None) -> str:
    head = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in messages[:2]
    ]
    blob = json.dumps({"h": head, "t": [t.get("function", {}).get("name") for t in tools or []]}, sort_keys=True,
                      default=str)
    return "abproxy-" + hashlib.sha256(blob.encode()).hexdigest()[:32]


def _error(status: int, message: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"
                                                              if status < 500 else "server_error", "code": code}})


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    backend = Backend(args)
    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"proxy_{args.log_name}.jsonl"
    log_lock = asyncio.Lock()
    inflight = {"n": 0}

    async def write_log(rec: dict) -> None:
        async with log_lock:
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "proxy",
                                            "base_model": args.base_model, "sampler_path": args.sampler_path}]}

    @app.get("/health")
    async def health():
        return {"ok": True, "inflight": inflight["n"], "model": args.served_name}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0 = time.time()
        body = await request.json()
        if body.get("stream"):
            return _error(400, "stream=true is not supported by this proxy")
        messages = body.get("messages") or []
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        req_max = body.get("max_completion_tokens") or body.get("max_tokens")
        max_tokens = min(int(req_max), args.max_tokens) if req_max else args.max_tokens
        temperature, top_p = args.temperature, args.top_p
        if args.honor_request_temperature:
            if body.get("temperature") is not None:
                temperature = float(body["temperature"])
            if body.get("top_p") is not None:
                top_p = float(body["top_p"])
        try:
            prompt_ids = await asyncio.to_thread(backend.render, messages, tools)
        except Exception as exc:  # noqa: BLE001
            log.exception("render failed")
            await write_log({"ts": t0, "event": "render_error", "error": repr(exc)[:1000]})
            return _error(400, f"failed to render messages: {exc!r}"[:2000])
        n_prompt = len(prompt_ids)
        if n_prompt + 256 > args.max_context:
            await write_log({"ts": t0, "event": "context_overflow", "prompt_tokens": n_prompt})
            return _error(400, f"This model's maximum context length is {args.max_context} tokens, but the prompt "
                               f"has {n_prompt} tokens (context length exceeded).", "context_length_exceeded")
        max_tokens = max(1, min(max_tokens, args.max_context - n_prompt))
        cache_key = None if args.no_prompt_cache_key else _cache_key(messages, tools)
        inflight["n"] += 1
        result = None
        last_exc: Exception | None = None
        try:
            for attempt in range(args.retries + 1):
                try:
                    result = await backend.sample(prompt_ids, max_tokens=max_tokens, temperature=temperature,
                                                  top_p=top_p, top_k=args.top_k, cache_key=cache_key)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    text = repr(exc).lower()
                    log.warning("sample attempt %d failed: %r", attempt, exc)
                    if "context" in text and ("length" in text or "exceed" in text):
                        break
                    if "prompt_cache_key" in text and cache_key is not None:
                        cache_key = None
                        continue
                    await asyncio.sleep(min(30, 2 ** attempt))
        finally:
            inflight["n"] -= 1
        if result is None or not result.completions:
            await write_log({"ts": t0, "event": "sample_error", "prompt_tokens": n_prompt,
                             "latency_s": round(time.time() - t0, 3), "error": repr(last_exc)[:1000]})
            text = repr(last_exc)
            if "context" in text.lower() and ("length" in text.lower() or "exceed" in text.lower()):
                return _error(400, f"context length exceeded: {text}"[:2000], "context_length_exceeded")
            return _error(502, f"sampler error: {text}"[:2000])
        comp = result.completions[0]
        out_tokens = list(comp.full_tokens[comp.prompt_len:])
        finish = "length" if comp.finish_reason == "length" else "stop"
        message, parse_error = await asyncio.to_thread(backend.parse, out_tokens, finish)
        n_coerced = 0 if args.no_coerce_args else coerce_tool_arguments(message, tools)
        if message.get("tool_calls"):
            finish_reason = "tool_calls" if finish != "length" else "length"
        else:
            finish_reason = finish
        cached = None
        sm = result.server_metrics
        if sm is not None and sm.cached_prompt_tokens is not None:
            cached = int(sm.cached_prompt_tokens)
        latency = time.time() - t0
        usage: dict[str, Any] = {
            "prompt_tokens": n_prompt,
            "completion_tokens": len(out_tokens),
            "total_tokens": n_prompt + len(out_tokens),
        }
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        if message.get("reasoning_content"):
            usage["completion_tokens_details"] = {
                "reasoning_tokens": len(backend.tokenizer.encode(message["reasoning_content"],
                                                                 add_special_tokens=False))
            }
        rec = {
            "ts": t0, "event": "ok", "model": args.served_name, "sampler_path": args.sampler_path,
            "prompt_tokens": n_prompt, "completion_tokens": len(out_tokens), "cached_tokens": cached,
            "latency_s": round(latency, 3), "sampler_wall_s": round(result.wall_seconds, 3),
            "finish_reason": finish_reason, "n_tool_calls": len(message.get("tool_calls") or []),
            "tool_names": [tc["function"]["name"] for tc in message.get("tool_calls") or []],
            "parse_error": parse_error, "n_coerced_args": n_coerced, "n_messages": len(messages), "cache_key": cache_key,
            "cost_usd": round(n_prompt * args.prefill_per_m / 1e6 + len(out_tokens) * args.sample_per_m / 1e6, 6),
            "cost_usd_with_cache": round(((n_prompt - (cached or 0)) * args.prefill_per_m + (cached or 0) * args.cached_per_m
                                          + len(out_tokens) * args.sample_per_m) / 1e6, 6),
        }
        if parse_error:
            rec["raw_completion"] = backend.tokenizer.decode(out_tokens, skip_special_tokens=False)[-4000:]
            log.warning("parse issue: %s", parse_error)
        await write_log(rec)
        log.info("ok prompt=%d completion=%d tools=%s finish=%s %.1fs", n_prompt, len(out_tokens),
                 rec["tool_names"], finish_reason, latency)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(t0),
            "model": body.get("model") or args.served_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
            "usage": usage,
        }

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    log.info("config: preset=%s base_model=%s renderer=%s tokenizer=%s sampler_path=%s max_tokens=%d temp=%s "
             "top_p=%s top_k=%s ctx=%d port=%d", args.preset, args.base_model, args.renderer, args.tokenizer,
             args.sampler_path, args.max_tokens, args.temperature, args.top_p, args.top_k, args.max_context,
             args.port)
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
