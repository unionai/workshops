"""Serve the fine-tuned adapter as a text-to-SQL API, on the same GPU it was trained on.

The app asks for the same ACCELERATOR shape as the training tasks (config.py), so on an Apple Silicon
devbox it runs natively on the Mac and generates on `mps`. Flyte mounts the adapter the pipeline
produced (`RunOutput`) before the server starts.

    python serve.py                       # latest pipeline output
    python serve.py --run-name <run>      # a specific run
    ACCELERATOR=T4 python serve.py        # the same app on a cluster GPU

    curl -s "$URL/health"
    curl -s -X POST "$URL/generate" -H "Content-Type: application/json" -d '{
      "schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
      "question": "What is the average salary by department?"
    }'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

# App helpers are imported at module top on purpose: the app bundle only carries modules
# that are imported when the app is serialized.
import sql
from config import ACCELERATOR, MODEL_NAME, PIP, PROPAGATED, SHAPES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ADAPTER_MOUNT = "/tmp/sql-adapter"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = sql.pick_device(ACCELERATOR)
    dtype = sql.pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    adapter = Path(ADAPTER_MOUNT)
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        trained_on = json.loads((adapter / "train_metrics.json").read_text()).get("device") if (adapter / "train_metrics.json").exists() else None
        log.info(f"Adapter loaded from {adapter} (trained on {trained_on})")
    else:
        log.warning(f"No adapter at {adapter}; serving the base model")
    model.eval()
    app.state.model, app.state.tokenizer, app.state.device = model, tokenizer, device
    log.info(f"{MODEL_NAME} ready on {device} ({dtype})")
    yield


app = FastAPI(title="Text-to-SQL", description=f"{MODEL_NAME} with a LoRA adapter, on {ACCELERATOR}", lifespan=lifespan)

env = FastAPIAppEnvironment(
    name="apple-silicon-sql-api",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(*PIP, "fastapi", "uvicorn"),
    resources=SHAPES[ACCELERATOR],
    requires_auth=False,
    env_vars=PROPAGATED,
    parameters=[
        flyte.app.Parameter(
            name="adapter",
            # The second output of `pipeline` is the adapter directory.
            value=flyte.app.RunOutput(task_name="apple-silicon-ft-cpu.pipeline", type="directory", getter=(1,)),
            mount=ADAPTER_MOUNT,
        )
    ],
)


class SQLRequest(BaseModel):
    model_config = {"populate_by_name": True}

    schema_: str = Field("", alias="schema")
    question: str
    max_new_tokens: int = 128


@app.get("/health")
async def health():
    return {"status": "ok" if getattr(app.state, "model", None) is not None else "loading", "device": str(getattr(app.state, "device", None))}


@app.post("/generate")
async def generate(request: SQLRequest):
    import torch

    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    tokenizer, model, device = app.state.tokenizer, app.state.model, app.state.device
    prompt = sql.render_prompt(tokenizer, {"context": request.schema_, "question": request.question})
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=request.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return {"sql": sql.normalize_sql(raw), "raw_output": raw, "device": str(device)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the text-to-SQL API")
    parser.add_argument("--run-name", help="A specific pipeline run (default: the latest successful one)")
    args = parser.parse_args()

    flyte.init_from_config()
    if args.run_name:
        env.parameters[0].value = flyte.app.RunOutput(run_name=args.run_name, type="directory", getter=(1,))
    deployed = flyte.serve(env)
    print(f"Console:  {deployed.url}")
    print(f"Endpoint: {deployed.endpoint}")
