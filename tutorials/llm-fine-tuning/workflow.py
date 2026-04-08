"""
LLM Fine-Tuning Pipeline — Full, LoRA, and QLoRA in one workflow.

One pipeline, three fine-tuning methods. The `method` flag switches between:
  - full:  train all parameters (best quality, most compute)
  - lora:  freeze base model, train low-rank adapters (good balance)
  - qlora: 4-bit quantized base + LoRA adapters (least memory)

Usage:
    # LoRA fine-tuning (default)
    flyte run --local --tui workflow.py pipeline

    # QLoRA (requires CUDA)
    flyte run --local --tui workflow.py pipeline --method qlora

    # Full fine-tuning with a small model
    flyte run --local --tui workflow.py pipeline --method full

    # Quick local test (few samples)
    flyte run --local --tui workflow.py pipeline --max-train-samples 100 --max-eval-samples 20 --epochs 1

    # Remote (GPU cluster)
    flyte run workflow.py pipeline --method lora --epochs 3
"""

import json
import logging
import os
import tempfile

import flyte
import flyte.report
import markdown
from config import cpu_env, gpu_env, HF_TOKEN

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "b-mc2/sql-create-context",
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
) -> flyte.io.Dir:
    """Download dataset from HuggingFace and format for instruction fine-tuning."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    def format_example(ex):
        return {
            "text": (
                "### Task: Generate a SQL query to answer the question.\n"
                f"### Schema:\n{ex['context']}\n"
                f"### Question:\n{ex['question']}\n"
                f"### SQL:\n{ex['answer']}"
            )
        }

    ds = ds.map(format_example)

    # Split into train and eval
    total = len(ds)
    train_end = min(max_train_samples, total - max_eval_samples)
    eval_start = train_end
    eval_end = min(eval_start + max_eval_samples, total)

    processed = DatasetDict({
        "train": ds.select(range(train_end)),
        "eval": ds.select(range(eval_start, eval_end)),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(processed['train'])} train, {len(processed['eval'])} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune a model using full, LoRA, or QLoRA method."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    log.info(f"Training: model={model_name}, method={method}")

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Load tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Load model (method determines how) --
    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2><p>Method: {method}</p>")
    await flyte.report.flush.aio()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if method == "qlora":
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
        )

    # -- Apply LoRA adapters --
    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable, total_params = model.get_nb_trainable_parameters()
        pct = trainable / total_params * 100
        log.info(f"Trainable params: {trainable:,} / {total_params:,} ({pct:.1f}%)")

    # -- Flyte report callback --
    class FlyteReportCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            loss = logs["loss"]
            step = state.global_step
            total = state.max_steps
            pct = step / total * 100 if total else 0
            epoch = logs.get("epoch", 0)
            lr_val = logs.get("learning_rate", 0)

            flyte.report.replace(
                f"<h2>Training — {method.upper()}</h2>"
                f"<p><b>Model:</b> {model_name}</p>"
                f"<p><b>Step:</b> {step}/{total} ({pct:.0f}%) | "
                f"<b>Epoch:</b> {epoch:.1f} | "
                f"<b>Loss:</b> {loss:.4f} | "
                f"<b>LR:</b> {lr_val:.2e}</p>"
            )
            flyte.report.flush()

    # -- Train --
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_seq_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        callbacks=[FlyteReportCallback()],
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete.")

    # -- Merge LoRA weights and save --
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")

    if method in ("lora", "qlora"):
        log.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # Final report
    train_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
    await flyte.report.replace.aio(
        f"<h2>Training Complete — {method.upper()}</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Final loss:</b> {train_loss}</p>"
        f"<p><b>Epochs:</b> {epochs} | <b>LR:</b> {lr} | <b>Batch size:</b> {batch_size}</p>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate — before/after comparison
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 50,
) -> str:
    """Compare base model vs fine-tuned model on test examples."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Load eval data
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_sql(model, prompt, max_new_tokens=128):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def normalize_sql(sql):
        """Normalize SQL for comparison — lowercase, collapse whitespace, strip."""
        return " ".join(sql.lower().split()).strip().rstrip(";")

    def build_prompt(example):
        return (
            "### Task: Generate a SQL query to answer the question.\n"
            f"### Schema:\n{example['context']}\n"
            f"### Question:\n{example['question']}\n"
            "### SQL:\n"
        )

    # -- Run base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running base model inference...</p>")
    await flyte.report.flush.aio()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, torch_dtype=dtype, device_map="auto",
    )

    base_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(base_model, prompt)
        base_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Base model: {i + 1}/{len(eval_ds)}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Run fine-tuned model --
    log.info("Loading fine-tuned model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running fine-tuned model inference...</p>")
    await flyte.report.flush.aio()

    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_path, torch_dtype=dtype, device_map="auto",
    )

    ft_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(ft_model, prompt)
        ft_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Fine-tuned model: {i + 1}/{len(eval_ds)}")

    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    base_correct = 0
    ft_correct = 0
    comparisons = []

    for i, example in enumerate(eval_ds):
        expected = example["answer"]
        base_gen = base_results[i]
        ft_gen = ft_results[i]

        base_match = normalize_sql(base_gen) == normalize_sql(expected)
        ft_match = normalize_sql(ft_gen) == normalize_sql(expected)

        if base_match:
            base_correct += 1
        if ft_match:
            ft_correct += 1

        comparisons.append({
            "question": example["question"],
            "schema": example["context"],
            "expected": expected,
            "base": base_gen,
            "finetuned": ft_gen,
            "base_correct": base_match,
            "ft_correct": ft_match,
        })

    total = len(eval_ds)
    base_acc = base_correct / total * 100
    ft_acc = ft_correct / total * 100

    log.info(f"Base model accuracy: {base_acc:.1f}% ({base_correct}/{total})")
    log.info(f"Fine-tuned accuracy: {ft_acc:.1f}% ({ft_correct}/{total})")

    # -- Build report --
    examples_html = ""
    for c in comparisons[:10]:  # Show first 10 examples
        base_color = "green" if c["base_correct"] else "red"
        ft_color = "green" if c["ft_correct"] else "red"
        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p><b>Q:</b> {c['question']}</p>
<p style="font-size:0.85em; color:#666;"><b>Schema:</b> {c['schema'][:200]}...</p>
<p><b>Expected:</b> <code>{c['expected']}</code></p>
<p><b>Base model:</b> <code style="color:{base_color};">{c['base'][:200]}</code></p>
<p><b>Fine-tuned:</b> <code style="color:{ft_color};">{c['finetuned'][:200]}</code></p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results</h2>
<table>
<tr><th></th><th>Exact Match Accuracy</th><th>Correct</th></tr>
<tr><td><b>Base model</b></td><td>{base_acc:.1f}%</td><td>{base_correct}/{total}</td></tr>
<tr><td><b>Fine-tuned</b></td><td>{ft_acc:.1f}%</td><td>{ft_correct}/{total}</td></tr>
</table>
<p><b>Improvement:</b> {ft_acc - base_acc:+.1f} percentage points</p>
<hr/>
<h3>Example Comparisons</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_accuracy": round(base_acc, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "improvement": round(ft_acc - base_acc, 1),
        "num_examples": total,
        "comparisons": comparisons[:10],
    })


# ------------------------------------------------------------------
# Pipeline: orchestrate everything
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "b-mc2/sql-create-context",
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    End-to-end LLM fine-tuning pipeline.

    1. Download and format dataset
    2. Fine-tune model (full / LoRA / QLoRA)
    3. Evaluate: before/after comparison on test set
    """
    log.info(f"Pipeline: {model_name} | method={method} | dataset={dataset_name}")

    await flyte.report.replace.aio(
        f"<h2>LLM Fine-Tuning Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Method:</b> {method} | <b>Dataset:</b> {dataset_name}</p>"
        f"<p>Step 1/3: Preparing data...</p>"
    )
    await flyte.report.flush.aio()

    # Step 1: Prepare data
    data_dir = await prepare_data(dataset_name, max_train_samples, max_eval_samples)

    # Step 2: Train
    await flyte.report.replace.aio(
        f"<h2>LLM Fine-Tuning Pipeline</h2>"
        f"<p><b>Model:</b> {model_name} | <b>Method:</b> {method}</p>"
        f"<p>Step 2/3: Training...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(
        model_name, data_dir, method, epochs, lr, batch_size, lora_r, lora_alpha,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        f"<h2>LLM Fine-Tuning Pipeline</h2>"
        f"<p><b>Model:</b> {model_name} | <b>Method:</b> {method}</p>"
        f"<p>Step 3/3: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    # Final report
    await flyte.report.replace.aio(
        f"<h2>Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name} | <b>Method:</b> {method}</p>"
        f"<p><b>Base accuracy:</b> {metrics['base_accuracy']}%</p>"
        f"<p><b>Fine-tuned accuracy:</b> {metrics['finetuned_accuracy']}%</p>"
        f"<p><b>Improvement:</b> {metrics['improvement']:+.1f} percentage points</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return result
