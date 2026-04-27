"""
BERT Sentiment Classification — Fine-tune ModernBERT on IMDB reviews.

Fine-tune a ModernBERT (or any HuggingFace encoder) model on binary sentiment
classification. The pipeline downloads the IMDB dataset, trains the model,
and evaluates accuracy/F1 with example predictions.

Usage:
    # Default (ModernBERT-base on IMDB)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --max_train_samples 200 --max_eval_samples 50 --epochs 1

    # Remote
    flyte run workflow.py pipeline --epochs 3

    # Swap model
    flyte run workflow.py pipeline --model_name "bert-base-uncased"
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
    dataset_name: str = "imdb",
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
) -> flyte.io.Dir:
    """Download IMDB dataset and save splits to disk."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    train_ds = ds["train"].shuffle(seed=42).select(range(min(max_train_samples, len(ds["train"]))))
    eval_ds = ds["test"].shuffle(seed=42).select(range(min(max_eval_samples, len(ds["test"]))))

    processed = DatasetDict({"train": train_ds, "eval": eval_ds})

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
) -> flyte.io.Dir:
    """Fine-tune a BERT-style model for sentiment classification."""
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    log.info(f"Training: model={model_name}")

    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Tokenize --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # -- Load model --
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=HF_TOKEN,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

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

            flyte.report.replace(
                f"<h2>Training — {model_name}</h2>"
                f"<p><b>Step:</b> {step}/{total} ({pct:.0f}%) | "
                f"<b>Epoch:</b> {epoch:.1f} | "
                f"<b>Loss:</b> {loss:.4f}</p>"
            )
            flyte.report.flush()

    # -- Train --
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_steps=50,
        report_to="none",
    )

    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[FlyteReportCallback()],
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete.")

    # -- Save --
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # Final report
    metrics = trainer.evaluate()
    await flyte.report.replace.aio(
        f"<h2>Training Complete — {model_name}</h2>"
        f"<p><b>Eval Accuracy:</b> {metrics.get('eval_accuracy', 0):.1%}</p>"
        f"<p><b>Eval F1:</b> {metrics.get('eval_f1', 0):.1%}</p>"
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
    num_examples: int = 100,
) -> str:
    """Compare base model vs fine-tuned model on test examples."""
    import numpy as np
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    # Load eval data (raw text + labels)
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    texts = eval_ds["text"]
    labels = eval_ds["label"]

    def predict_batch(model, tokenizer, texts, batch_size=32):
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
        return preds

    # -- Base model (untrained, random classifier head) --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running base model inference...</p>")
    await flyte.report.flush.aio()

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=HF_TOKEN, num_labels=2,
    )
    base_model.eval()
    if torch.cuda.is_available():
        base_model = base_model.cuda()

    base_preds = predict_batch(base_model, base_tokenizer, texts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Fine-tuned model --
    log.info("Loading fine-tuned model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running fine-tuned model inference...</p>")
    await flyte.report.flush.aio()

    ft_path = await finetuned_dir.download()
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
    ft_model = AutoModelForSequenceClassification.from_pretrained(ft_path)
    ft_model.eval()
    if torch.cuda.is_available():
        ft_model = ft_model.cuda()

    ft_preds = predict_batch(ft_model, ft_tokenizer, texts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    label_names = {0: "negative", 1: "positive"}

    base_acc = accuracy_score(labels, base_preds) * 100
    base_f1 = f1_score(labels, base_preds, average="binary") * 100
    ft_acc = accuracy_score(labels, ft_preds) * 100
    ft_f1 = f1_score(labels, ft_preds, average="binary") * 100

    log.info(f"Base model — Accuracy: {base_acc:.1f}%, F1: {base_f1:.1f}%")
    log.info(f"Fine-tuned — Accuracy: {ft_acc:.1f}%, F1: {ft_f1:.1f}%")

    # -- Build report --
    comparisons = []
    examples_html = ""
    for i in range(min(10, len(texts))):
        text_preview = texts[i][:300] + "..." if len(texts[i]) > 300 else texts[i]
        true_label = label_names[labels[i]]
        base_label = label_names[base_preds[i]]
        ft_label = label_names[ft_preds[i]]

        base_color = "green" if base_preds[i] == labels[i] else "red"
        ft_color = "green" if ft_preds[i] == labels[i] else "red"

        comparisons.append({
            "text": text_preview,
            "true_label": true_label,
            "base_pred": base_label,
            "ft_pred": ft_label,
            "base_correct": base_preds[i] == labels[i],
            "ft_correct": ft_preds[i] == labels[i],
        })

        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p style="font-size:0.9em;">{text_preview}</p>
<p><b>True:</b> {true_label} |
<b>Base:</b> <span style="color:{base_color};">{base_label}</span> |
<b>Fine-tuned:</b> <span style="color:{ft_color};">{ft_label}</span></p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results</h2>
<table>
<tr><th></th><th>Accuracy</th><th>F1</th></tr>
<tr><td><b>Base model</b></td><td>{base_acc:.1f}%</td><td>{base_f1:.1f}%</td></tr>
<tr><td><b>Fine-tuned</b></td><td>{ft_acc:.1f}%</td><td>{ft_f1:.1f}%</td></tr>
</table>
<p><b>Accuracy improvement:</b> {ft_acc - base_acc:+.1f} percentage points</p>
<hr/>
<h3>Example Predictions</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_accuracy": round(base_acc, 1),
        "base_f1": round(base_f1, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "finetuned_f1": round(ft_f1, 1),
        "accuracy_improvement": round(ft_acc - base_acc, 1),
        "num_examples": len(texts),
        "comparisons": comparisons,
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "answerdotai/ModernBERT-base",
    dataset_name: str = "imdb",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 100,
) -> str:
    """
    End-to-end BERT fine-tuning pipeline for sentiment classification.

    1. Download and prepare IMDB dataset
    2. Fine-tune ModernBERT (or any encoder model) for binary classification
    3. Evaluate: before/after comparison on test set
    """
    log.info(f"Pipeline: {model_name} | dataset={dataset_name}")

    await flyte.report.replace.aio(
        f"<h2>Sentiment Classification Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Dataset:</b> {dataset_name}</p>"
        f"<p>Step 1/3: Preparing data...</p>"
    )
    await flyte.report.flush.aio()

    # Step 1: Prepare data
    data_dir = await prepare_data(dataset_name, max_train_samples, max_eval_samples)

    # Step 2: Train
    await flyte.report.replace.aio(
        f"<h2>Sentiment Classification Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/3: Training...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(model_name, data_dir, epochs, lr, batch_size)

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        f"<h2>Sentiment Classification Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/3: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    # Final report
    await flyte.report.replace.aio(
        f"<h2>Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Base accuracy:</b> {metrics['base_accuracy']}%</p>"
        f"<p><b>Fine-tuned accuracy:</b> {metrics['finetuned_accuracy']}%</p>"
        f"<p><b>Improvement:</b> {metrics['accuracy_improvement']:+.1f} percentage points</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Improvement: {metrics['accuracy_improvement']:+.1f}pp")
    return result
