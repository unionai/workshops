# BERT Fine-Tuning: Sentiment Classification

Fine-tune ModernBERT (or any HuggingFace encoder) on IMDB movie reviews for binary sentiment classification. The pipeline trains the model and evaluates accuracy/F1 with before/after comparison.

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: prepare data → train → evaluate before/after |

## Setup

```bash
cd tutorials/bert-fine-tuning-sentiment

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

### Default (ModernBERT on IMDB)

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test

```bash
flyte run --local --tui workflow.py pipeline --max_train_samples 200 --max_eval_samples 50 --epochs 1
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline --epochs 3
```

The task image now copies this tutorial directory into the container explicitly.
Run the command from `tutorials/bert-fine-tuning-sentiment` so Flyte builds the image with the local workflow source and `requirements.txt`.

### Swap model

```bash
# Classic BERT
flyte run workflow.py pipeline --model_name "bert-base-uncased"

# DistilBERT (smaller, faster)
flyte run workflow.py pipeline --model_name "distilbert-base-uncased"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `answerdotai/ModernBERT-base` | HuggingFace encoder model |
| `--dataset_name` | `imdb` | HuggingFace dataset |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch_size` | `16` | Per-device batch size |
| `--max_train_samples` | `10000` | Max training examples |
| `--max_eval_samples` | `2000` | Max evaluation examples |
| `--num_eval_examples` | `100` | Examples for before/after comparison |

## Evaluation

The evaluate step runs the same test examples through both the base model (random classifier head) and the fine-tuned model, then compares:

- **Accuracy** and **F1 score**
- **Side-by-side predictions** showing base vs fine-tuned per review

The base model predicts essentially at random (~50%) since its classification head is untrained. The fine-tuned model should reach 85-90%+ accuracy.

flyte run workflow.py pipeline --max_train_samples 500 --max_eval_samples 100 --num_eval_examples 50 --epochs 5   
