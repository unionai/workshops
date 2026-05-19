# LLM Fine-Tuning: Text-to-SQL

Fine-tune a language model on text-to-SQL with full fine-tuning, LoRA, or QLoRA — all in one Flyte pipeline. Then deploy the result as an OpenAI-compatible API via vLLM.

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: prepare data → train (full/LoRA/QLoRA) → evaluate before/after |
| `report_helpers.py` | Report CSS, SVG chart generators (line/bar), and HTML helpers |
| `serve.py` | Deploy the fine-tuned model as a vLLM endpoint |

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

## Fine-Tuning Methods

| Method | What happens | Memory | Best for |
|--------|-------------|--------|----------|
| `full` | Train all model parameters | High | Small models, maximum quality |
| `lora` | Freeze base, train low-rank adapters | Medium | Good balance of quality and efficiency |
| `qlora` | 4-bit quantized base + LoRA adapters | Low | Larger models on limited GPU memory |

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
python serve.py --run-name <run-name>
```

This uses vLLM to serve the fine-tuned model with `RunOutput` pulling the model artifact from your training run.

Test the endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-app-url/v1", api_key="na")

response = client.chat.completions.create(
    model="finetuned-sql",
    messages=[{
        "role": "user",
        "content": (
            "### Task: Generate a SQL query to answer the question.\n"
            "### Schema:\n"
            "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)\n"
            "### Question:\n"
            "What is the average salary by department?\n"
            "### SQL:\n"
        ),
    }],
)
print(response.choices[0].message.content)
```

## Swapping Models and Datasets

Everything is HuggingFace-based, so swapping is just changing a string:

```bash
# Different model
flyte run --local --tui workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"

# Different dataset (must have similar structure or update format_example in workflow.py)
flyte run --local --tui workflow.py pipeline --dataset_name "your-org/your-dataset"
```
