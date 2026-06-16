"""
Deploy the fine-tuned emotion classifier as a FastAPI endpoint.

The server loads the ModernBERT model on startup and exposes a /predict
endpoint that returns emotion predictions with confidence scores and
attention weights for visualization.

Usage:
    # Deploy the latest trained model
    python serve.py

    # Deploy from a specific pipeline run
    python serve.py --run-name <run-name>

    # Test the endpoint
    curl -X POST https://your-app-url/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "I am so happy today!"}'
"""

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import flyte
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MOUNT_PATH = "/tmp/finetuned_model"
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Loading emotion classifier...")
    model_path = Path(MODEL_MOUNT_PATH)

    if model_path.exists():
        app.state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        # Use eager attention to extract attention weights
        app.state.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            output_attentions=True,
            attn_implementation="eager",
        )
        app.state.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app.state.model = app.state.model.to(device)
        logger.info(f"Model loaded on {device}")
    else:
        logger.warning(f"Model not found at {model_path}")
        app.state.model = None
        app.state.tokenizer = None

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Emotion Classifier",
    description="Classify emotions in text using a fine-tuned ModernBERT model",
    version="1.0.0",
    lifespan=lifespan,
)

env = FastAPIAppEnvironment(
    name="emotion-classifier-api",
    app=app,
    description="Fine-tuned ModernBERT emotion classification",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi", "uvicorn", "torch", "transformers", "accelerate",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,
    ),
    requires_auth=False,
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.RunOutput(
                task_name="bert-emotion-cpu.pipeline",
                type="directory",
            ),
            mount=MODEL_MOUNT_PATH,
        ),
    ],
)


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str


class EmotionScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    scores: list[EmotionScore]
    tokens: list[str]
    attention_weights: list[float]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy" if app.state.model is not None else "not_ready",
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = app.state.model
    tokenizer = app.state.tokenizer

    inputs = tokenizer(
        request.text, return_tensors="pt", truncation=True, max_length=128,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Probabilities
    probs = torch.softmax(outputs.logits[0], dim=-1).cpu().tolist()
    pred_idx = int(torch.argmax(outputs.logits[0]).item())

    # Attention: CLS token attention from last layer, averaged across heads
    last_attention = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    cls_attention = last_attention.mean(dim=0)[0].cpu().tolist()  # CLS row

    # Clean tokens for display
    token_ids = inputs["input_ids"][0]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    clean_tokens = []
    clean_attention = []
    for i, tok in enumerate(raw_tokens):
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
            continue
        if tok == tokenizer.pad_token:
            continue
        # Strip tokenizer prefixes
        display_tok = tok.replace("##", "").replace("Ġ", "").replace("▁", "")
        if not display_tok.strip():
            continue
        clean_tokens.append(display_tok)
        clean_attention.append(cls_attention[i])

    # Normalize attention to 0-1 for visualization
    if clean_attention:
        max_att = max(clean_attention)
        min_att = min(clean_attention)
        att_range = max_att - min_att or 1
        clean_attention = [(a - min_att) / att_range for a in clean_attention]

    scores = [
        EmotionScore(label=EMOTION_LABELS[i], score=round(probs[i], 4))
        for i in range(len(EMOTION_LABELS))
    ]

    return PredictResponse(
        predicted_emotion=EMOTION_LABELS[pred_idx],
        confidence=round(probs[pred_idx], 4),
        scores=scores,
        tokens=clean_tokens,
        attention_weights=clean_attention,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy emotion classifier")
    parser.add_argument(
        "--run-name",
        help="Run name of a specific pipeline or train task run (from Flyte UI). "
             "Default: latest pipeline output.",
    )
    args = parser.parse_args()

    flyte.init_from_config()

    if args.run_name:
        for p in env.parameters:
            if p.name == "model":
                p.value = flyte.app.RunOutput(
                    run_name=args.run_name,
                    type="directory",
                )
                break

    deployed = flyte.serve(env)
    print(f"Deployed: {deployed.url}")
