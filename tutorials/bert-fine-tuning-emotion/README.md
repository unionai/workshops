# ModernBERT Emotion Classification

Fine-tune ModernBERT to classify emotions in text, then explore *how* the model makes decisions with attention heatmaps and gradient-based token attribution.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/bert-fine-tuning-emotion/bert-emotion-tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The dataset is [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — ~20k English Twitter messages labeled with one of 6 emotions:

| Label | Emotion | Example |
|-------|---------|---------|
| 0 | sadness | "i feel so empty inside" |
| 1 | joy | "i am so happy right now" |
| 2 | love | "i feel blessed to have you" |
| 3 | anger | "i am furious about this" |
| 4 | fear | "i feel so scared and anxious" |
| 5 | surprise | "i cant believe this just happened" |

## What's in the Pipeline

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────────┐
│ Get Data │───▶│   Train    │───▶│  Evaluate  │───▶│    Explore      │
│  (CPU)   │    │   (GPU)    │    │   (GPU)    │    │   Inference     │
└──────────┘    └────────────┘    └────────────┘    │    (GPU)        │
 emotion         ModernBERT        Confusion         └─────────────────┘
 dataset         fine-tuning       matrix +           Attention heatmaps
                 with live         per-class           + token importance
                 loss/eval         metrics             + misclassification
                 charts                                analysis
```

1. **Get data** — Downloads the emotion dataset, shuffles, and splits into train/eval.
2. **Train** — Fine-tunes ModernBERT (or any HuggingFace encoder) for 6-class classification. Live report shows loss curve, eval accuracy/F1, and a progress bar.
3. **Evaluate** — Compares the base model (random classifier head) vs fine-tuned. Produces a confusion matrix heatmap, per-class precision/recall/F1, and a grouped bar chart of per-class accuracy.
4. **Explore inference** — The interesting part. For a set of examples, produces:
   - **Confidence distribution** — Softmax probabilities across all 6 emotions, not just the argmax
   - **Attention heatmap** — CLS token attention from the last transformer layer, averaged across heads. Shows which words the model "looks at" when classifying
   - **Token importance** — Gradient-based attribution (gradient x embedding norm) showing which tokens most influence the prediction. Green = supports prediction, red = opposes
   - **Misclassification spotlight** — The model's most confident wrong predictions, revealing blind spots

## Files

| File | What it does |
|------|-------------|
| `workflow.py` | Full pipeline — get_data, train, evaluate, explore_inference |
| `config.py` | Flyte task environments (CPU/GPU), image config, secrets |
| `report_helpers.py` | SVG charts, confusion matrix, attention/importance visualization |
| `serve.py` | FastAPI model server — serves predictions with attention weights |
| `app_gradio.py` | Gradio frontend — interactive UI with attention heatmap |
| `requirements.txt` | Python dependencies |

## Setup

```bash
cd tutorials/bert-fine-tuning-emotion
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your HuggingFace token (needed for gated models):

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

## Run

### Quick local test

```bash
flyte run --local --tui workflow.py pipeline \
  --max_train_samples 200 \
  --max_eval_samples 50 \
  --epochs 1 \
  --num_eval_examples 30 \
  --num_explore_examples 6
```

Small dataset, one epoch — finishes in a few minutes on CPU. Good for verifying the pipeline works.

### Standard run

```bash
flyte run workflow.py pipeline \
  --model_name "answerdotai/ModernBERT-base" \
  --epochs 3 \
  --lr 2e-5 \
  --batch_size 16 \
  --max_train_samples 10000 \
  --max_eval_samples 2000 \
  --num_eval_examples 200 \
  --num_explore_examples 12
```

### With classic BERT

```bash
flyte run workflow.py pipeline --model_name "bert-base-uncased"
```

### Longer training

```bash
flyte run workflow.py pipeline \
  --epochs 5 \
  --max_train_samples 16000 \
  --num_eval_examples 500 \
  --num_explore_examples 18
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `answerdotai/ModernBERT-base` | HuggingFace encoder model to fine-tune |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch_size` | `16` | Batch size for training |
| `--warmup_steps` | `100` | Warmup steps for the scheduler |
| `--max_train_samples` | `10000` | Number of training examples |
| `--max_eval_samples` | `2000` | Number of held-out eval examples |
| `--num_eval_examples` | `200` | Examples used in the base vs fine-tuned comparison |
| `--num_explore_examples` | `12` | Examples for attention/attribution deep-dive |

## Model Serving

After training, you can deploy the model as a live API with a Gradio frontend.

### Step 1: Deploy the FastAPI server

```bash
python serve.py
```

This deploys the fine-tuned model as a `/predict` endpoint that returns:
- Predicted emotion and confidence
- Full probability distribution across all 6 emotions
- Attention weights per token (for heatmap visualization)

To deploy from a specific training run:

```bash
python serve.py --run-name <run-name-from-flyte-ui>
```

Test the endpoint:

```bash
curl -X POST https://your-app-url/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```



```bash
curl -X POST https://empty-snowflake-f34fa.apps.demo.hosted.unionai.cloud/predict  \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```

### Step 2: Deploy the Gradio frontend

```bash
python app_gradio.py
```

This auto-discovers the FastAPI server and deploys a Gradio UI where users can:
- Type text and see emotion predictions with confidence bars
- View an attention heatmap showing which words the model focuses on
- Try pre-loaded example texts

For local development (server already running):

```bash
SERVER_URL=https://your-app-url python app_gradio.py
```

## Why ModernBERT?

[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (2024) is a drop-in replacement for BERT with several improvements:

- **8192 token context** (vs BERT's 512) — handles longer text without truncation
- **Rotary positional embeddings** — better position encoding for variable-length inputs
- **Flash Attention** — faster training and inference
- **Better pretraining** — trained on more data with modern techniques
- **Same API** — works with `AutoModelForSequenceClassification` just like BERT

For this tutorial, the practical benefit is better classification accuracy with the same code. The attention visualization works the same way since it's still a multi-head attention transformer.

## Understanding the Visualizations

### Attention heatmap

The attention heatmap shows what the [CLS] token "looks at" in the final transformer layer. In BERT-style models, the [CLS] token is used for classification — its representation is fed to the classifier head. So the [CLS] attention pattern reveals which tokens the model considers most relevant for its emotion prediction.

For example, on "i am so happy right now":
- High attention on "happy" and "so" → the model correctly focuses on the emotional content
- Low attention on "i" and "right" → function words get less attention

### Token importance (gradient attribution)

This uses gradient-based attribution: for each token, we compute how much the token's embedding influences the predicted class score. Specifically:

```
importance(token) = ||gradient(prediction, embedding(token)) * embedding(token)||
```

Green tokens **support** the prediction, red tokens **oppose** it. This is complementary to attention — attention shows where the model looks, while attribution shows what actually drives the decision.

### Negation (a dataset gap)

Try "this does not make me angry". The model predicts **anger** with 99.8% confidence because it latches onto the word "angry" without properly handling "not" as a negation. The attention heatmap reveals this: "angry" gets high attention (0.44) and "not" gets some (0.41), but the model doesn't use the negation to flip the meaning.

This isn't a limitation of the model architecture. The training data (Twitter messages) consists mostly of direct emotional statements. Negated emotions like "I'm not sad anymore" or "this doesn't scare me" are rare in the dataset, so the model simply pattern-matches on emotional keywords. A dataset with more negated examples would teach the model to handle these correctly.

Try these in the Gradio app to explore the effect:
- "this does not make me angry" → predicts anger (wrong)
- "I used to be sad but now I'm fine" → may still predict sadness
- "I'm not surprised at all" → may still predict surprise

This is exactly why the attention visualization is valuable. You can *see* what the model latches onto and understand where the training data has gaps.

### Misclassification spotlight

The most confident wrong predictions are the most informative errors. A model that says "95% anger" on a text that's actually "fear" reveals something about the model's confusion boundary between those emotions. These often involve ambiguous text where emotions overlap (e.g., "i can't believe they did that" could be anger or surprise).
