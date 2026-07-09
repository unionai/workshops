"""
Deploy the GRPO-trained Countdown model as a FastAPI endpoint.

Give it numbers and a target; it returns an arithmetic expression that (hopefully)
uses each number once and equals the target — and the server *verifies* the answer
with the same safe AST evaluator used for the training reward.

Usage:
    # Deploy the model from the latest pipeline run
    python serve.py

    # Deploy from a specific run (from the Flyte UI)
    python serve.py --run-name <run-name>

    # Test it
    curl -X POST https://your-app-url/solve \
      -H "Content-Type: application/json" \
      -d '{"numbers": [3, 7, 9], "target": 20}'
"""

import argparse
import ast
import logging
import operator
import re
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import flyte
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MOUNT_PATH = "/tmp/finetuned_model"

SYSTEM_PROMPT = (
    "You are solving a Countdown puzzle. You are given a list of numbers and a "
    "target. Using each number exactly once and the operators + - * / (and "
    "parentheses), write an arithmetic expression that equals the target. "
    "Give ONLY the expression, nothing else. "
    "Example: for numbers [3, 4, 2] and target 14, answer: 3 * 4 + 2"
)

# --- the same safe verifier used for the training reward ---
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str):
    try:
        node = ast.parse(expr, mode="eval").body

        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](ev(n.operand))
            raise ValueError
        return ev(node)
    except Exception:
        return None


def numbers_in_expr(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return []
    return sorted(int(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                  and float(n.value).is_integer())


def extract_answer(text: str):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m[-1].strip()
    eqs = re.findall(r"([0-9][0-9+\-*/()\s]*?)\s*=", text)
    if eqs:
        return eqs[-1].strip()
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and re.fullmatch(r"[0-9+\-*/()\s]+", l.strip())]
    return lines[-1] if lines else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading Countdown model...")
    model_path = Path(MODEL_MOUNT_PATH)
    if model_path.exists():
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        app.state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        app.state.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, device_map="auto",
        )
        app.state.model.eval()
        logger.info("Model loaded.")
    else:
        logger.warning(f"Model not found at {model_path}")
        app.state.model = None
        app.state.tokenizer = None
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="GRPO Countdown Solver",
    description="Solve Countdown number puzzles with a GRPO-fine-tuned model",
    version="1.0.0",
    lifespan=lifespan,
)

env = FastAPIAppEnvironment(
    name="grpo-countdown-api",
    app=app,
    description="GRPO-trained Countdown puzzle solver",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi", "uvicorn", "torch", "transformers", "accelerate",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
    requires_auth=False,
    parameters=[
        Parameter(
            name="model",
            # the pipeline returns the trained model directory
            value=flyte.app.RunOutput(
                task_name="grpo-countdown-cpu.pipeline",
                type="directory",
            ),
            mount=MODEL_MOUNT_PATH,
        ),
    ],
)


class SolveRequest(BaseModel):
    numbers: list[int]
    target: int


class SolveResponse(BaseModel):
    expression: str | None
    value: float | None
    correct: bool
    raw_output: str


@app.get("/health")
async def health():
    return {"status": "healthy" if app.state.model is not None else "not_ready",
            "model_loaded": app.state.model is not None}


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer = app.state.tokenizer
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Numbers: {request.numbers}. Target: {request.target}."},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(app.state.model.device)
    with torch.no_grad():
        outputs = app.state.model.generate(
            **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    expr = extract_answer(raw)
    value = safe_eval(expr) if expr else None
    correct = bool(
        expr
        and sorted(int(x) for x in request.numbers) == numbers_in_expr(expr)
        and value is not None
        and abs(value - request.target) < 1e-6
    )
    return SolveResponse(expression=expr, value=value, correct=correct, raw_output=raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the GRPO Countdown model")
    parser.add_argument("--run-name", help="Run name of a specific pipeline run (Flyte UI). "
                                            "Default: latest train output.")
    args = parser.parse_args()

    flyte.init_from_config()

    if args.run_name:
        for p in env.parameters:
            if p.name == "model":
                p.value = flyte.app.RunOutput(
                    run_name=args.run_name,
                    task_name="grpo-countdown-cpu.pipeline",
                    type="directory",
                )
                break

    deployed = flyte.serve(env)
    print(f"Deployed: {deployed.url}")
