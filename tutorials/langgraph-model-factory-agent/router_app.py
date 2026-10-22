"""The ticket router as a Union app: the factory's output, actually serving.

A FastAPI app that mounts the latest `ticket-router` artifact at startup and classifies
tickets. `promote` deploys it from inside the agent's tool call; you can also deploy it
by hand, or redeploy after a new promotion, with:

    flyte deploy router_app.py router_app

Then:

    curl -X POST "$URL/classify" -H 'content-type: application/json' \
         -d '{"text": "I was charged twice this month, please refund one"}'

The artifact version is resolved and pinned when the app is deployed, so a later
promotion does not swap weights under a running app: redeploy to pick it up.
"""

from __future__ import annotations

import os
import threading
import time

import flyte
import flyte.app
import flyte.io
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel

from config import ON_MAC, PROPAGATED, SHAPES, ml_image, tagged
from factory import (  # top-level on purpose: the app's code bundle only carries modules loaded at deploy time
    CANDIDATES,
    half_dtype,
    kind_of,
)
from tickets import DATASET_VERSIONS, parse_label, system_prompt

MODEL_DIR = "/tmp/model"

app = FastAPI(title=tagged("ticket-router"), description="Classifies support tickets with the promoted model")


def _billions(params: str) -> float:
    """`"360M"` -> 0.36, `"1.5B"` -> 1.5."""
    n, unit = float(params[:-1]), params[-1].upper()
    return n / 1000 if unit == "M" else n


def serving_resources(base: str | None) -> flyte.Resources:
    """Size the serving pod to the model being promoted.

    The encoder runs on a small CPU pod; a chat model up to 0.5B on a bigger one; anything
    larger gets a T4, so the latency the request was judged on is the latency production
    sees. ROUTER_CPU, ROUTER_MEMORY and ROUTER_GPU override for an unusual cluster (they
    travel with the run). An unknown base is treated as large.
    """
    if ON_MAC:
        # On an Apple Silicon devbox the app runs natively on the Mac, on its GPU: a devbox
        # container cannot run this stack, and the Mac is the production box anyway.
        return SHAPES["apple"]
    spec = CANDIDATES.get(base or "", {})
    if spec.get("kind") == "encoder":
        # 2 CPUs, not 1: the encoder is CPU-bound, and on one core it answers in ~600 ms
        # instead of ~200. Memory is what small nodes lack, and 2Gi is plenty for 149M params.
        cpu, memory, gpu = 2, "2Gi", ""
    elif _billions(spec.get("params", "9B")) <= 0.6:
        cpu, memory, gpu = 2, "4Gi", ""
    else:
        cpu, memory, gpu = 3, "12Gi", "T4:1"
    cpu = int(os.environ.get("ROUTER_CPU", cpu))
    memory = os.environ.get("ROUTER_MEMORY", memory)
    gpu = os.environ.get("ROUTER_GPU", gpu)
    return flyte.Resources(cpu=cpu, memory=memory, gpu=gpu) if gpu else flyte.Resources(cpu=cpu, memory=memory)


def make_router_app(
    version: str | None = None, base: str | None = None, weights: flyte.io.Dir | None = None
) -> FastAPIAppEnvironment:
    """The app environment, mounting the latest `ticket-router` artifact or a pinned version.

    `promote` deploys the latest; `rollback` deploys a pinned earlier version. Same app,
    same name, different artifact version resolved at deploy time, and a pod sized to
    `base`, the candidate the artifact was made from. On a platform with no artifact
    registry, `promote` passes the output directory itself as `weights` and the app mounts
    that instead.
    """
    model = (
        weights
        if weights is not None
        else flyte.app.ArtifactValue(name=tagged("ticket-router"), type="directory", version=version)
    )
    return FastAPIAppEnvironment(
        name=tagged("ticket-router"),
        app=app,
        image=ml_image,
        resources=serving_resources(base),
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
        timeouts=flyte.app.Timeouts(request=600),  # a classify may be waiting on the model load
        requires_auth=False,
        env_vars={**PROPAGATED},
        parameters=[
            flyte.app.Parameter(name="model", value=model, mount=MODEL_DIR),
        ],
    )


router_app = make_router_app()

_state: dict = {}


@router_app.on_startup
async def start_loading():
    """Answer `/` at once and load the model in the background.

    The platform's readiness check wants the port open promptly. A 1.5B model read from the
    mounted artifact takes a while, and a pod that is still loading when the check gives up
    is restarted, forever. So the pod becomes ready immediately, `/` reports `loaded`, and
    `/classify` waits for the model. `promote` polls `/` until the pod reports the right
    model *and* `loaded`.
    """
    import asyncio
    import json
    import pathlib

    meta = pathlib.Path(MODEL_DIR) / "factory.json"
    _state["factory"] = json.loads(meta.read_text()) if meta.exists() else {}
    _state["dataset"] = _state["factory"].get("dataset", "v1")
    router = pathlib.Path(MODEL_DIR) / "router.json"  # written by publish_router: which version this is, and why
    _state["router"] = json.loads(router.read_text()) if router.exists() else {}
    _state["version"] = _state["router"].get("version")
    _state["loaded"] = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _done(fut):
        if fut.exception():
            _state["load_error"] = repr(fut.exception())
        _state["loaded"].set()

    loop.run_in_executor(None, _load_model).add_done_callback(_done)


class Router:
    """The promoted model, loaded and answering.

    The app wraps one of these behind `/classify`. A run with no platform (`flyte run
    --local`, Colab) has no app, so the support agent and `test_deployment` use one
    in-process, loaded from the local registry: the same weights, the same code, no HTTP.
    """

    def __init__(self, model_dir: str):
        import json
        import pathlib

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        self.model_dir = model_dir
        meta = pathlib.Path(model_dir) / "factory.json"
        self.factory = json.loads(meta.read_text()) if meta.exists() else {}
        self.dataset = self.factory.get("dataset", "v1")
        router = pathlib.Path(model_dir) / "router.json"  # written by publish_router: which version this is, and why
        self.router = json.loads(router.read_text()) if router.exists() else {}
        self.version = self.router.get("version")
        self.kind = self.factory.get("kind") or kind_of(model_dir)
        # A T4 if the pod has one (see serving_resources), the Mac's GPU on a devbox, else CPU in fp32.
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = half_dtype(self.device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # On a GPU, stream the weights straight there (device_map) instead of staging the full
        # fp32 checkpoint in CPU memory and casting: for a 1.5B model that staging alone is
        # 6GB plus the fp16 copy, and the pod gets OOM-killed before `.to(cuda)` ever runs.
        load_kwargs = {"dtype": dtype, "device_map": self.device} if self.device == "cuda" else {"dtype": dtype}
        if self.kind == "encoder":
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(model_dir, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        self.model = model.to(self.device).eval()
        self.loaded_in = time.perf_counter() - t0
        # One GPU, one stream: callers (the app's workers, the support agent's tickets) may
        # arrive on several threads at once, and MPS aborts the process on concurrent use.
        self._lock = threading.Lock()

    def classify(self, text: str) -> dict:
        with self._lock:
            return self._classify(text)

    def _classify(self, text: str) -> dict:
        import torch

        tok, model, version = self.tok, self.model, self.dataset
        if self.kind == "encoder":
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, -1)[0]
            idx = int(probs.argmax())
            label = model.config.id2label[idx]
            return {
                "label": label,
                "confidence": round(float(probs[idx]), 4),
                "raw": label,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "dataset": version,
                "kind": "encoder",
            }
        msgs = [{"role": "system", "content": system_prompt(version)}, {"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
            )
        reply = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return {
            "label": parse_label(reply, DATASET_VERSIONS[version]),
            "confidence": None,
            "raw": reply.strip(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "dataset": version,
            "kind": "causal",
        }


def _load_model():
    r = Router(MODEL_DIR)
    _state.update(router_obj=r, kind=r.kind, device=r.device, loaded_in=r.loaded_in)


class Ticket(BaseModel):
    text: str


async def _wait_loaded(timeout_s: float = 590) -> None:
    import asyncio

    event = _state.get("loaded")
    if event is not None:
        await asyncio.wait_for(event.wait(), timeout=timeout_s)
    if _state.get("load_error"):
        raise RuntimeError(f"model failed to load: {_state['load_error']}")


@app.get("/")
async def info() -> dict:
    event = _state.get("loaded")
    return {
        "model_dir": MODEL_DIR,
        "version": _state.get("version"),
        "loaded": bool(event and event.is_set()) and not _state.get("load_error"),
        "load_error": _state.get("load_error"),
        "loaded_in_s": _state.get("loaded_in"),
        "device": _state.get("device"),
        "factory": _state.get("factory"),
        "router": _state.get("router"),
        "dataset": _state.get("dataset"),
    }


@app.post("/classify")
async def classify(ticket: Ticket) -> dict:
    import asyncio

    await _wait_loaded()
    return await asyncio.to_thread(_state["router_obj"].classify, ticket.text)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(router_app)
    print(handle.url)
