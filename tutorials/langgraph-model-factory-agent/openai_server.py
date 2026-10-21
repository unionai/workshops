"""A small OpenAI-compatible chat server on top of `transformers`, for whatever device torch has.

Enough of the API for the OpenAI SDK, LangChain and a Gradio chat UI to treat it as a provider:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions      messages, tools, stream, max_tokens, temperature, chat_template_kwargs

Tool calls are rendered by the model's own chat template and parsed back out of the Hermes-style
`<tool_call>{...}</tool_call>` blocks Qwen emits, so a LangGraph agent can drive it. Qwen3's thinking
is off unless the request asks for it. Generation is serialized with a lock: one GPU, one stream.

No Flyte in here. `serve.py` wraps `build_app()` in an app environment; you can also run it bare:

    uvicorn openai_server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ── Devices ───────────────────────────────────────────────────────────────────


def pick_device(accelerator: str):
    """The torch device for an ACCELERATOR value. Fails loudly rather than quietly serving from the CPU."""
    import torch

    if accelerator == "apple":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "ACCELERATOR=apple but torch has no mps device. Run this on an Apple Silicon devbox "
                "(flyte start devbox --apple-silicon), or set ACCELERATOR=cpu."
            )
        return torch.device("mps")
    if accelerator in ("T4", "L40s"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"ACCELERATOR={accelerator} but torch has no cuda device.")
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device):
    import torch

    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# ── Messages in, text out ─────────────────────────────────────────────────────


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI wire format -> what the chat template expects (tool-call arguments as objects)."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):  # content parts -> plain text
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        if m.get("content") is None:
            m["content"] = ""
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    pass
        out.append(m)
    return out


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split generated text into (content, tool_calls) in OpenAI's shape."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": payload.get("name", ""), "arguments": json.dumps(payload.get("arguments", {}))},
            }
        )
    content = TOOL_CALL_RE.sub("", text).strip()
    return content, calls


class Engine:
    """One model on one device, generating for one request at a time."""

    def __init__(self, model_name: str, accelerator: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = pick_device(accelerator)
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        self.lock = threading.Lock()
        self.supports_thinking = "enable_thinking" in (self.tokenizer.chat_template or "")
        log.info(f"{model_name} on {self.device} ({self.dtype})")
        self.generate([{"role": "user", "content": "Hello"}], max_tokens=4)  # compile kernels before traffic

    def prompt(self, messages: list[dict], tools: list[dict] | None, template_kwargs: dict) -> str:
        kwargs = dict(template_kwargs)
        if self.supports_thinking:
            kwargs.setdefault("enable_thinking", False)
        return self.tokenizer.apply_chat_template(
            _normalize_messages(messages), tools=tools or None, add_generation_prompt=True, tokenize=False, **kwargs
        )

    def _inputs(self, text: str):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _gen_kwargs(self, max_tokens: int, temperature: float, top_p: float) -> dict:
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def generate(self, messages, tools=None, max_tokens=512, temperature=0.0, top_p=1.0, template_kwargs=None) -> dict:
        import torch

        enc = self._inputs(self.prompt(messages, tools, template_kwargs or {}))
        with self.lock, torch.inference_mode():
            start = time.perf_counter()
            out = self.model.generate(**enc, **self._gen_kwargs(max_tokens, temperature, top_p))
            if self.device.type != "cpu":
                getattr(torch, self.device.type).synchronize()
            elapsed = time.perf_counter() - start
        new = out[0, enc["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return {
            "text": text,
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "completion_tokens": int(len(new)),
            "seconds": elapsed,
            "finished": bool(len(new) < max_tokens),
        }

    def stream(self, messages, tools=None, max_tokens=512, temperature=0.0, top_p=1.0, template_kwargs=None):
        """Yield text pieces as they are generated; the last item is the usage dict."""
        import torch
        from transformers import TextIteratorStreamer

        enc = self._inputs(self.prompt(messages, tools, template_kwargs or {}))
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {**enc, **self._gen_kwargs(max_tokens, temperature, top_p), "streamer": streamer}
        result: dict[str, Any] = {}

        def run():
            with self.lock, torch.inference_mode():
                start = time.perf_counter()
                out = self.model.generate(**kwargs)
                result["completion_tokens"] = int(out.shape[1] - enc["input_ids"].shape[1])
                result["seconds"] = time.perf_counter() - start

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()
        yield {"prompt_tokens": int(enc["input_ids"].shape[1]), **result}


# ── The FastAPI app ───────────────────────────────────────────────────────────


def build_app(model_name: str, accelerator: str, model_id: str | None = None) -> FastAPI:
    """`model_name` is what transformers loads (a Hub repo or a local directory, such as a mounted
    prefetch output); `model_id` is the name the API reports and clients ask for (default: the same)."""
    load_from, model_name = model_name, model_id or model_name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = await asyncio.to_thread(Engine, load_from, accelerator)
        yield

    app = FastAPI(title="OpenAI-compatible LLM", description=f"{model_name} via transformers on {accelerator}", lifespan=lifespan)

    def engine() -> Engine:
        eng = getattr(app.state, "engine", None)
        if eng is None:
            raise HTTPException(status_code=503, detail="model loading")
        return eng

    @app.get("/health")
    async def health():
        eng = getattr(app.state, "engine", None)
        return {"status": "ok" if eng else "loading", "model": model_name, "device": str(eng.device) if eng else None}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        eng = engine()
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")
        tools = body.get("tools")
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
        temperature = float(body.get("temperature") or 0.0)
        top_p = float(body.get("top_p") or 1.0)
        template_kwargs = body.get("chat_template_kwargs") or {}
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if usage:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        # Streaming plain text: pieces as they come. Tool calls are parsed from the full text, so a
        # request with tools streams one final chunk instead.
        if body.get("stream") and not tools:

            async def events():
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def producer():
                    try:
                        for item in eng.stream(messages, None, max_tokens, temperature, top_p, template_kwargs):
                            loop.call_soon_threadsafe(queue.put_nowait, item)
                    except Exception as e:  # surface generation errors to the client
                        loop.call_soon_threadsafe(queue.put_nowait, e)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=producer, daemon=True).start()
                yield chunk({"role": "assistant", "content": ""})
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        yield chunk({}, "error")
                        break
                    if isinstance(item, dict):
                        usage = {"prompt_tokens": item["prompt_tokens"], "completion_tokens": item["completion_tokens"], "total_tokens": item["prompt_tokens"] + item["completion_tokens"]}
                        yield chunk({}, "stop", usage)
                    else:
                        yield chunk({"content": item})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        result = await asyncio.to_thread(eng.generate, messages, tools, max_tokens, temperature, top_p, template_kwargs)
        content, tool_calls = parse_tool_calls(result["text"]) if tools else (result["text"], [])
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else ("stop" if result["finished"] else "length")
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        }
        log.info(f"{result['completion_tokens']} tokens in {result['seconds']:.1f}s = {result['completion_tokens'] / max(result['seconds'], 1e-6):.0f} tok/s on {eng.device}")

        if body.get("stream"):

            async def one_chunk():
                yield chunk({"role": "assistant", **{k: v for k, v in message.items() if k != "role"}}, finish, usage)
                yield "data: [DONE]\n\n"

            return StreamingResponse(one_chunk(), media_type="text/event-stream")

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
            # Not part of the OpenAI schema; handy for benchmarks.
            "timing": {"seconds": round(result["seconds"], 3), "tokens_per_second": round(result["completion_tokens"] / max(result["seconds"], 1e-6), 1), "device": str(eng.device)},
        }

    return app


# `uvicorn openai_server:app` for a bare local server, no Flyte.
app = build_app(os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B"), os.getenv("ACCELERATOR", "apple"))
