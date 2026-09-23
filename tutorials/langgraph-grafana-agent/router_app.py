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
import time

import flyte
import flyte.app
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel

from config import GRAFANA_ENV_VARS, GRAFANA_SECRETS, PROPAGATED, ml_image, tagged
from factory import (  # top-level on purpose: the app's code bundle only carries modules loaded at deploy time
    CANDIDATES,
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
    spec = CANDIDATES.get(base or "", {})
    if spec.get("kind") == "encoder":
        cpu, memory, gpu = 1, "2Gi", ""
    elif _billions(spec.get("params", "9B")) <= 0.6:
        cpu, memory, gpu = 2, "4Gi", ""
    else:
        cpu, memory, gpu = 3, "12Gi", "T4:1"
    cpu = int(os.environ.get("ROUTER_CPU", cpu))
    memory = os.environ.get("ROUTER_MEMORY", memory)
    gpu = os.environ.get("ROUTER_GPU", gpu)
    return flyte.Resources(cpu=cpu, memory=memory, gpu=gpu) if gpu else flyte.Resources(cpu=cpu, memory=memory)


def make_router_app(version: str | None = None, base: str | None = None) -> FastAPIAppEnvironment:
    """The app environment, mounting the latest `ticket-router` artifact or a pinned version.

    `promote` deploys the latest; `rollback` deploys a pinned earlier version. Same app,
    same name, different artifact version resolved at deploy time, and a pod sized to
    `base`, the candidate the artifact was made from.
    """
    return FastAPIAppEnvironment(
        name=tagged("ticket-router"),
        app=app,
        image=ml_image,
        resources=serving_resources(base),
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
        requires_auth=False,
        # The app reports what it serves: predictions per label, confidence, latency, as
        # OpenTelemetry metrics to the same Grafana stack. That is the drift dashboard.
        env_vars={**PROPAGATED, **GRAFANA_ENV_VARS},
        secrets=[*GRAFANA_SECRETS],
        parameters=[
            flyte.app.Parameter(
                name="model",
                value=flyte.app.ArtifactValue(name=tagged("ticket-router"), type="directory", version=version),
                mount=MODEL_DIR,
            )
        ],
    )


router_app = make_router_app()

_state: dict = {}


def _metrics():
    """OpenTelemetry instruments, exported over OTLP if the endpoint is configured; no-ops otherwise."""
    import os

    if "_meter" in _state:
        return _state["_meter"]
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        headers = dict(h.split("=", 1) for h in os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "").split(",") if "=" in h)
        exporter = OTLPMetricExporter(endpoint=endpoint.rstrip("/") + "/v1/metrics", headers=headers or None)
        provider = MeterProvider(
            resource=Resource.create({"service.name": "ticket-router"}),
            metric_readers=[PeriodicExportingMetricReader(exporter, export_interval_millis=15000)],
        )
        metrics.set_meter_provider(provider)
        meter = provider.get_meter("ticket-router")
    else:
        from opentelemetry import metrics

        meter = metrics.get_meter("ticket-router")  # the no-op meter
    _state["_meter"] = {
        "predictions": meter.create_counter(
            "ticket_router_predictions", description="Tickets routed, by predicted label"
        ),
        "confidence": meter.create_histogram(
            "ticket_router_confidence", description="Router confidence in its prediction (0-1)"
        ),
        "latency": meter.create_histogram("ticket_router_latency_ms", description="Router latency per ticket, ms"),
    }
    return _state["_meter"]


@router_app.on_startup
async def load_model():
    import json
    import pathlib

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    meta = pathlib.Path(MODEL_DIR) / "factory.json"
    factory = json.loads(meta.read_text()) if meta.exists() else {}
    kind = factory.get("kind") or kind_of(MODEL_DIR)
    # A T4 if the pod has one (see serving_resources), else CPU in fp32.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if kind == "encoder":
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype)
    model = model.to(device).eval()
    # The model knows the label set it was trained on; the prompt has to match it.
    _state.update(
        tok=tok,
        model=model,
        kind=kind,
        device=device,
        loaded_in=time.perf_counter() - t0,
        dataset=factory.get("dataset", "v1"),
        factory=factory,
    )


class Ticket(BaseModel):
    text: str


@app.get("/")
async def info() -> dict:
    return {
        "model_dir": MODEL_DIR,
        "loaded_in_s": _state.get("loaded_in"),
        "device": _state.get("device"),
        "factory": _state.get("factory"),
        "dataset": _state.get("dataset"),
    }


@app.post("/classify")
async def classify(ticket: Ticket) -> dict:
    import torch

    tok, model = _state["tok"], _state["model"]
    version = _state.get("dataset", "v1")
    m = _metrics()
    if _state.get("kind") == "encoder":
        inputs = tok(ticket.text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, -1)[0]
        idx = int(probs.argmax())
        label = model.config.id2label[idx]
        confidence = float(probs[idx])
        latency = round((time.perf_counter() - t0) * 1000, 1)
        attrs = {"label": label, "model": str((_state.get("factory") or {}).get("base", "?")), "dataset": version}
        m["predictions"].add(1, attrs)
        m["confidence"].record(confidence, attrs)
        m["latency"].record(latency, attrs)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw": label,
            "latency_ms": latency,
            "dataset": version,
            "kind": "encoder",
        }
    msgs = [{"role": "system", "content": system_prompt(version)}, {"role": "user", "content": ticket.text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    label = parse_label(reply, DATASET_VERSIONS[version])
    latency = round((time.perf_counter() - t0) * 1000, 1)
    attrs = {"label": label, "model": str((_state.get("factory") or {}).get("base", "?")), "dataset": version}
    m["predictions"].add(1, attrs)
    m["latency"].record(latency, attrs)
    return {
        "label": label,
        "confidence": None,
        "raw": reply.strip(),
        "latency_ms": latency,
        "dataset": version,
        "kind": "causal",
    }


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(router_app)
    print(handle.url)
