# LLM Fine-Tuning: Text-to-SQL

Fine-tune a language model on text-to-SQL with full fine-tuning, LoRA, or QLoRA — all in one Flyte pipeline. Then deploy the result as a FastAPI endpoint with a Gradio UI.

**Default model:** [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — a tiny 135M parameter model. Small enough to train quickly on a single GPU, large enough to learn the SQL pattern and demonstrate the difference between fine-tuning methods.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/llm-fine-tuning-lora-qlora/llm-fine-tune-tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: prepare data → train (full/LoRA/QLoRA) → evaluate before/after |
| `report_helpers.py` | Report CSS, SVG chart generators (line/bar), and HTML helpers |
| `serve.py` | Deploy the fine-tuned model as a FastAPI endpoint |
| `app_gradio.py` | Gradio UI for interactive text-to-SQL queries |

## Setup

```bash
cd tutorials/llm-fine-lora-qlora

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional — set a HuggingFace token for gated models:

```bash
export HF_TOKEN=your-token
# or add HF_TOKEN=your-token to .env
```

## How LoRA Works

**Full fine-tuning** updates every weight in the model — effective but expensive. For a 7B model that's billions of parameters to train, store, and deploy.

**LoRA (Low-Rank Adaptation)** takes a different approach: freeze the entire base model and inject small trainable adapters alongside the original weights. Instead of modifying a large weight matrix `W` directly, LoRA adds a low-rank decomposition `A × B` that learns a small correction:

```
                    ┌─────────────────────────┐
                    │   Original Weight W      │
input ────────────→ │   (frozen, e.g. 768×768) │──→ main output
    │               └─────────────────────────┘         │
    │               ┌───────────┐ ┌───────────┐         │
    └─────────────→ │ A (768×16) │→│ B (16×768) │→ × α/r ──→ + ──→ combined output
                    └───────────┘ └───────────┘
                    (LoRA adapter, trainable)
```

The original weight `W` stays completely frozen. The adapter matrices `A` and `B` are tiny — for a 768×768 layer with rank `r=16`, LoRA adds only 24,576 params vs the original 589,824 (~4%).

**Key parameters:**
- **`r` (rank)** — size of the adapter matrices. Higher = more capacity but more params
- **`alpha`** — scaling factor. The adapter output is multiplied by `alpha/r` before being added. Controls how strongly the adapter influences the output. Common practice: `alpha = 2 × r`
- **`alpha` adds zero extra parameters** — it's just a scalar multiplier

These small corrections are applied at every key layer in every transformer block — which is enough to steer the model's behavior significantly while training only 2-4% of total parameters.

**QLoRA** goes further: it quantizes the frozen base model to 4-bit precision, reducing memory even more. The LoRA adapters still train in full precision. This lets you fine-tune models that wouldn't otherwise fit in GPU memory.

## Fine-Tuning Methods

| Method | What happens | Memory | Best for |
|--------|-------------|--------|----------|
| `full` | Train all model parameters | High | Small models, maximum quality |
| `lora` | Freeze base, train low-rank adapters | Medium | Good balance of quality and efficiency |
| `qlora` | 4-bit quantized base + LoRA adapters | Low | Larger models on limited GPU memory |

> **Note on QLoRA:** With a small model like SmolLM2-135M, QLoRA is overkill — the model already fits easily in GPU memory, and 4-bit quantization just hurts quality. QLoRA shines when you need to fine-tune a model that's too large to fit in VRAM otherwise (e.g., a 7B+ model on a single T4). It's included here to show *how* it works so you can apply it when you need it.

## Run

### LoRA (default)

```bash
flyte run workflow.py pipeline --method lora
```

### QLoRA (requires CUDA)

```bash
flyte run --local --tui workflow.py pipeline --method qlora
```

### Full fine-tuning

```bash
flyte run --local --tui workflow.py pipeline --method full
```

### Quick test (small subset)

```bash
flyte run workflow.py pipeline \
  --max_train_samples 100 --max_eval_samples 20 --epochs 3
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline \
  --method lora --model_name "Qwen/Qwen2.5-0.5B" --epochs 3
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model to fine-tune |
| `--dataset_name` | `b-mc2/sql-create-context` | HuggingFace dataset |
| `--method` | `lora` | Fine-tuning method: `full`, `lora`, or `qlora` |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--batch_size` | `4` | Per-device batch size |
| `--max_train_samples` | `5000` | Max training examples |
| `--max_eval_samples` | `500` | Max evaluation examples |
| `--num_eval_examples` | `50` | Examples for before/after comparison |
| `--lora_r` | `16` | LoRA rank (for lora/qlora) |
| `--lora_alpha` | `32` | LoRA alpha (for lora/qlora) |

## Evaluation

The evaluate step runs the same prompts through both the base model and the fine-tuned model, then compares:

- **Exact match accuracy** on generated SQL
- **Side-by-side examples** showing base vs fine-tuned output
- **Improvement** in percentage points

The report shows the **full raw output** from each model. This is intentional — one of the clearest effects of fine-tuning is that the base model tends to ramble (repeating the prompt template, generating extra text after the SQL), while the fine-tuned model learns to stop cleanly after the answer thanks to the EOS token in the training data.

For scoring, `normalize_sql` extracts just the first SQL statement (truncating at `###` or newline) so the accuracy comparison is fair even when the base model keeps generating.

Results appear as interactive Flyte reports with stat grids, SVG charts, and side-by-side comparisons.

## Deploy the Fine-Tuned Model

After training, deploy as an OpenAI-compatible API:

```bash
# Deploy the latest trained model
python serve.py

# Deploy a specific run (e.g. your best LoRA run)
python serve.py --run-name rk2zfpk6x49c5vs45652

# python serve.py --run-name <run-name>
```

This deploys a FastAPI endpoint that loads the fine-tuned model and serves SQL generation. `RunOutput` pulls the model directory from your training pipeline.

Test the endpoint:

```bash
curl -X POST https://steep-fog-ad9c2.apps.tryv2.hosted.unionai.cloud/generate \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
    "question": "What is the average salary by department?"
  }'
```

Response:

```json
{
  "sql": "SELECT department, AVG(salary) FROM employees GROUP BY department",
  "raw_output": "SELECT department, AVG(salary) FROM employees GROUP BY department"
}
```

## Gradio UI

Deploy an interactive frontend for the model:

```bash
# Auto-discovers the deployed serve.py endpoint
python app_gradio.py

# Or connect to a specific server
SERVER_URL=https://your-app-url python app_gradio.py
```

Includes example schemas and questions to try out.

## Swapping Models and Datasets

Everything is HuggingFace-based, so swapping is just changing a string:

```bash
# Different model
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"

# Different dataset (must have similar structure or update format_example in workflow.py)
flyte run workflow.py pipeline --dataset_name "your-org/your-dataset"
```

### LoRA target modules

When swapping models, be aware that **LoRA target module names vary between architectures**. The default targets in `workflow.py` are set for LLaMA-style models (SmolLM2, Qwen, Mistral, etc.):

```python
target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

These are the layers inside each transformer block where LoRA injects low-rank adapters:

**Attention layers** — how the model decides what to focus on:
| Module | Name | What it does |
|--------|------|-------------|
| `q_proj` | Query | What to look for in the context |
| `k_proj` | Key | What each token offers to match against |
| `v_proj` | Value | What information to extract once matched |
| `o_proj` | Output | Combines multi-head attention results |

**MLP (feed-forward) layers** — how the model processes information after attention:
| Module | Name | What it does |
|--------|------|-------------|
| `gate_proj` | Gate | Controls how much information flows through (SwiGLU activation) |
| `up_proj` | Up | Projects to a higher dimension for richer representations |
| `down_proj` | Down | Projects back down to the model's hidden size |

By targeting all seven layers, LoRA can adapt both *what the model pays attention to* and *how it processes that information* — without retraining all the weights.

Other architectures use different naming conventions:

| Architecture | Attention modules | Example models |
|-------------|------------------|----------------|
| LLaMA-style | `q_proj`, `k_proj`, `v_proj`, `o_proj` | SmolLM2, Qwen, Mistral, LLaMA |
| GPT-2 / GPT-Neo | `q_proj`, `k_proj`, `v_proj`, `out_proj` | GPT-2, GPT-Neo, GPT-J |
| BLOOM | `query_key_value`, `dense` | BLOOM, BLOOMZ |
| Falcon | `query_key_value`, `dense` | Falcon |
| Phi | `q_proj`, `k_proj`, `v_proj`, `dense` | Phi-1, Phi-2 |

If you see LoRA training with 0 trainable parameters or an error about missing modules, check the model's attention layer names:

```python
# Quick way to find the right module names
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-model")
print([n for n, _ in model.named_modules() if "proj" in n or "query" in n or "dense" in n])
```

### Good models to try

| Model | Params | Notes |
|-------|--------|-------|
| `HuggingFaceTB/SmolLM2-135M` | 135M | Default — fast training, good for demos |
| `Qwen/Qwen2.5-0.5B` | 500M | Better quality, still fits easily on a T4 |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Good quality, may benefit from QLoRA on smaller GPUs |
| `meta-llama/Llama-3.2-1B` | 1B | Strong base model, requires HF token (gated) |
