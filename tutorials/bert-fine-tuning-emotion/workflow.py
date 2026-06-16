"""
ModernBERT Emotion Classification — Fine-tune, evaluate, and explore.

Fine-tune ModernBERT on emotion classification (6 classes: sadness, joy,
love, anger, fear, surprise) from Twitter text. The pipeline trains the model,
evaluates with confusion matrix and per-class metrics, then explores inference
with attention heatmaps and token importance visualization.

Usage:
    # Quick local test
    flyte run --local --tui workflow.py pipeline \
        --max_train_samples 200 --max_eval_samples 50 --epochs 1

    # Default (ModernBERT-base on emotion dataset)
    flyte run workflow.py pipeline

    # Remote with more data
    flyte run workflow.py pipeline --epochs 3 --max_train_samples 10000

    # Swap model
    flyte run workflow.py pipeline --model_name "bert-base-uncased"
"""

import json
import logging
import os
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env, HF_TOKEN
from report_helpers import (
    make_attention_text,
    make_bar_chart,
    make_confidence_bars,
    make_confusion_matrix,
    make_line_chart,
    make_token_importance_text,
    pipeline_step_indicator,
    wrap_report,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_DATASET = "dair-ai/emotion"


# ------------------------------------------------------------------
# Task 1: Get data
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def get_data(
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
) -> flyte.io.Dir:
    """Download the emotion dataset and save train/eval splits.

    The dair-ai/emotion dataset contains ~20k English Twitter messages labeled
    with one of 6 emotions: sadness, joy, love, anger, fear, surprise.
    """
    from datasets import DatasetDict, load_dataset

    log.info("Loading emotion dataset...")
    ds = load_dataset(EMOTION_DATASET)

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
    warmup_steps: int = 100,
) -> flyte.io.Dir:
    """Fine-tune a BERT-style model for 6-class emotion classification."""
    import numpy as np
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    log.info(f"Training: model={model_name}")

    id2label = {i: l for i, l in enumerate(EMOTION_LABELS)}
    label2id = {l: i for i, l in enumerate(EMOTION_LABELS)}

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Loading Model...</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="card"><p>Preparing for emotion classification training...</p></div>'
        ),
        do_flush=True,
    )

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Tokenize --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # -- Load model --
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=HF_TOKEN,
        num_labels=6,
        id2label=id2label,
        label2id=label2id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # -- Metrics tracking for live report --
    training_log: list[dict] = []
    eval_log: list[dict] = []

    def _build_training_report(max_steps: int) -> str:
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(dataset['train']):,}</div><div class="label">Train Samples</div></div>
          <div class="stat"><div class="value">{len(dataset['eval']):,}</div><div class="label">Eval Samples</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>
        </div>
        """

        charts_html = ""

        if training_log:
            current = training_log[-1]
            progress_pct = current["step"] / max_steps * 100 if max_steps else 0
            loss_display = f"Loss: <span class=\"highlight\">{current['loss']:.4f}</span>" if current.get("loss") else ""
            charts_html += f"""
            <div class="card">
              <b>Step {current['step']}/{max_steps}</b>
              ({progress_pct:.0f}%) |
              Epoch {current['epoch']:.2f}/{epochs}
              {f' | {loss_display}' if loss_display else ''}
              <div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#0f3460;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_entries = [e for e in training_log if "loss" in e]
            if len(loss_entries) >= 2:
                loss_chart = make_line_chart(
                    data=loss_entries,
                    x_key="epoch",
                    y_keys=["loss"],
                    title="Training Loss",
                    x_label="Epoch",
                    y_label="Loss",
                    colors=["#5a7db5"],
                )
                charts_html += f'<div class="chart-container">{loss_chart}</div>'

        if eval_log:
            latest_eval = eval_log[-1]
            best_acc = max(e.get("accuracy", 0) for e in eval_log)
            best_f1 = max(e.get("f1", 0) for e in eval_log)
            charts_html += f"""
            <div class="stat-grid" style="margin-top:16px;">
              <div class="stat"><div class="value">{latest_eval.get('accuracy', 0):.1%}</div><div class="label">Eval Accuracy</div></div>
              <div class="stat"><div class="value">{latest_eval.get('f1', 0):.1%}</div><div class="label">Eval F1</div></div>
              <div class="stat"><div class="value">{best_acc:.1%}</div><div class="label">Best Accuracy</div></div>
              <div class="stat"><div class="value">{latest_eval.get('eval_loss', 0):.4f}</div><div class="label">Eval Loss</div></div>
            </div>
            """

            if len(eval_log) >= 2:
                eval_chart = make_line_chart(
                    data=eval_log,
                    x_key="epoch",
                    y_keys=["accuracy", "f1"],
                    title="Eval Metrics Over Training",
                    x_label="Epoch",
                    y_label="Score",
                    colors=["#0f3460", "#06d6a0"],
                    y_max_cap=1.05,
                    y_display_names={"accuracy": "Accuracy", "f1": "Weighted F1"},
                )
                charts_html += f'<div class="chart-container">{eval_chart}</div>'

                eval_loss_chart = make_line_chart(
                    data=[e for e in eval_log if "eval_loss" in e],
                    x_key="epoch",
                    y_keys=["eval_loss"],
                    title="Eval Loss",
                    x_label="Epoch",
                    y_label="Loss",
                    colors=["#e63946"],
                )
                if any("eval_loss" in e for e in eval_log):
                    charts_html += f'<div class="chart-container">{eval_loss_chart}</div>'

        return wrap_report(stats_html + charts_html)

    # -- Callbacks --
    class ReportCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
            }
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 4)
            if "eval_accuracy" in logs:
                eval_log.append({
                    "epoch": entry["epoch"],
                    "accuracy": logs["eval_accuracy"],
                    "f1": logs.get("eval_f1", 0),
                    "eval_loss": logs.get("eval_loss", 0),
                })
            if "loss" in entry:
                training_log.append(entry)

            flyte.report.replace(
                _build_training_report(state.max_steps),
                do_flush=True,
            )

    # -- Compute metrics --
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # -- Training --
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_steps=warmup_steps,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ReportCallback()],
    )

    log.info("Starting training...")
    await flyte.report.replace.aio(
        _build_training_report(0),
        do_flush=True,
    )

    trainer.train()
    log.info("Training complete.")

    # -- Save model --
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # -- Final eval + report --
    metrics = trainer.evaluate()
    final_acc = metrics.get("eval_accuracy", 0)
    final_f1 = metrics.get("eval_f1", 0)

    final_charts = ""
    loss_entries = [e for e in training_log if "loss" in e]
    if len(loss_entries) >= 2:
        loss_chart = make_line_chart(
            data=loss_entries,
            x_key="epoch",
            y_keys=["loss"],
            title="Training Loss",
            x_label="Epoch",
            y_label="Loss",
            colors=["#5a7db5"],
        )
        final_charts += f'<div class="chart-container">{loss_chart}</div>'

    if len(eval_log) >= 2:
        eval_chart = make_line_chart(
            data=eval_log,
            x_key="epoch",
            y_keys=["accuracy", "f1"],
            title="Eval Metrics Over Training",
            x_label="Epoch",
            y_label="Score",
            colors=["#0f3460", "#06d6a0"],
            y_max_cap=1.05,
            y_display_names={"accuracy": "Accuracy", "f1": "Weighted F1"},
        )
        final_charts += f'<div class="chart-container">{eval_chart}</div>'

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Training Complete</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{final_acc:.1%}</div><div class="label">Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{final_f1:.1%}</div><div class="label">Weighted F1</div></div>'
            f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
            f'  <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>'
            f'</div>'
            f"{final_charts}"
        ),
        do_flush=True,
    )

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 200,
) -> str:
    """Compare base model (random head) vs fine-tuned on emotion classification.

    Produces confusion matrix, per-class precision/recall/F1, and overall metrics.
    """
    import numpy as np
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Loading models...</p>"),
        do_flush=True,
    )

    # -- Load eval data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))
    texts = eval_ds["text"]
    labels = eval_ds["label"]

    def predict_batch(model, tokenizer, texts, batch_size=32):
        preds = []
        probs_all = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_preds = torch.argmax(batch_probs, dim=-1).tolist()
            preds.extend(batch_preds)
            probs_all.extend(batch_probs.tolist())
        return preds, probs_all

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running base model (random classifier head)...</p>"),
        do_flush=True,
    )

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=HF_TOKEN, num_labels=6,
    )
    base_model.eval()
    if torch.cuda.is_available():
        base_model = base_model.cuda()

    base_preds, base_probs = predict_batch(base_model, base_tokenizer, texts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Fine-tuned model --
    log.info("Loading fine-tuned model...")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running fine-tuned model...</p>"),
        do_flush=True,
    )

    ft_path = await finetuned_dir.download()
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
    ft_model = AutoModelForSequenceClassification.from_pretrained(ft_path)
    ft_model.eval()
    if torch.cuda.is_available():
        ft_model = ft_model.cuda()

    ft_preds, ft_probs = predict_batch(ft_model, ft_tokenizer, texts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Compute metrics --
    base_acc = accuracy_score(labels, base_preds) * 100
    base_f1 = f1_score(labels, base_preds, average="weighted") * 100
    ft_acc = accuracy_score(labels, ft_preds) * 100
    ft_f1 = f1_score(labels, ft_preds, average="weighted") * 100

    log.info(f"Base:      Accuracy={base_acc:.1f}%, F1={base_f1:.1f}%")
    log.info(f"Fine-tuned: Accuracy={ft_acc:.1f}%, F1={ft_f1:.1f}%")

    # -- Confusion matrix --
    ft_cm = sk_confusion_matrix(labels, ft_preds, labels=list(range(6)))
    cm_list = ft_cm.tolist()
    cm_svg = make_confusion_matrix(cm_list, EMOTION_LABELS, title="Fine-tuned Model — Confusion Matrix")

    # -- Per-class metrics --
    report_dict = classification_report(labels, ft_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    per_class_html = "<table><tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
    for label_name in EMOTION_LABELS:
        if label_name in report_dict:
            m = report_dict[label_name]
            per_class_html += (
                f"<tr><td><b>{label_name}</b></td>"
                f"<td>{m['precision']:.1%}</td>"
                f"<td>{m['recall']:.1%}</td>"
                f"<td>{m['f1-score']:.1%}</td>"
                f"<td>{int(m['support'])}</td></tr>"
            )
    per_class_html += "</table>"

    # -- Bar chart: base vs fine-tuned --
    per_class_base_acc = []
    per_class_ft_acc = []
    for cls_idx in range(6):
        cls_mask = [i for i, l in enumerate(labels) if l == cls_idx]
        if cls_mask:
            base_cls_acc = sum(1 for i in cls_mask if base_preds[i] == cls_idx) / len(cls_mask) * 100
            ft_cls_acc = sum(1 for i in cls_mask if ft_preds[i] == cls_idx) / len(cls_mask) * 100
        else:
            base_cls_acc = 0
            ft_cls_acc = 0
        per_class_base_acc.append(base_cls_acc)
        per_class_ft_acc.append(ft_cls_acc)

    bar_chart = make_bar_chart(
        labels=EMOTION_LABELS,
        series={"Base": per_class_base_acc, "Fine-tuned": per_class_ft_acc},
        title="Per-Class Accuracy — Base vs Fine-tuned",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=105.0,
    )

    # -- Example predictions --
    improvement = ft_acc - base_acc
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    examples_html = ""
    for i in range(min(10, len(texts))):
        true_label = EMOTION_LABELS[labels[i]]
        ft_label = EMOTION_LABELS[ft_preds[i]]
        base_label = EMOTION_LABELS[base_preds[i]]
        ft_correct = ft_preds[i] == labels[i]
        base_correct = base_preds[i] == labels[i]
        text_preview = texts[i][:200]

        ft_badge = "badge-success" if ft_correct else "badge-danger"
        base_badge = "badge-success" if base_correct else "badge-danger"

        examples_html += f"""
<div class="card">
  <p style="font-size:0.95em;">"{text_preview}"</p>
  <p>True: <b>{true_label}</b> |
  Base: <span class="badge {base_badge}">{base_label}</span> |
  Fine-tuned: <span class="badge {ft_badge}">{ft_label}</span></p>
</div>"""

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation Results — Emotion Classification</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{base_acc:.1f}%</div><div class="label">Base Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{ft_acc:.1f}%</div><div class="label">Fine-tuned Accuracy</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'  <div class="stat"><div class="value">{ft_f1:.1f}%</div><div class="label">Fine-tuned F1</div></div>'
            f'</div>'
            f'<div class="chart-container">{bar_chart}</div>'
            f'<div class="chart-container">{cm_svg}</div>'
            f"<h3>Per-Class Metrics (Fine-tuned)</h3>"
            f"{per_class_html}"
            f"<h3>Example Predictions</h3>"
            f"{examples_html}"
        ),
        do_flush=True,
    )

    return json.dumps({
        "base_accuracy": round(base_acc, 1),
        "base_f1": round(base_f1, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "finetuned_f1": round(ft_f1, 1),
        "improvement": round(improvement, 1),
        "num_examples": len(texts),
        "confusion_matrix": cm_list,
        "per_class": {k: report_dict[k] for k in EMOTION_LABELS if k in report_dict},
    })


# ------------------------------------------------------------------
# Task 4: Explore inference
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def explore_inference(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 8,
) -> str:
    """Deep-dive into model behavior with attention and token importance.

    For a set of examples, this task produces:
    1. Predictions with full confidence distribution across all 6 emotions
    2. Attention heatmaps — which tokens the model focuses on for classification
       (CLS token attention from the last layer, averaged across heads)
    3. Token importance via gradient-based attribution — which tokens most
       influence the predicted class (gradient x embedding norm)
    4. Misclassification analysis — confident wrong predictions with explanations
    """
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("Starting explore_inference...")
    await flyte.report.replace.aio(
        wrap_report(
            "<h2>Explore Inference</h2>"
            "<p>Loading model for attention and attribution analysis...</p>"
        ),
        do_flush=True,
    )

    # -- Load model (with eager attention for weight extraction) --
    ft_path = await finetuned_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(ft_path)

    # Need eager attention to extract attention weights (flash attention doesn't return them)
    model = AutoModelForSequenceClassification.from_pretrained(
        ft_path,
        output_attentions=True,
        attn_implementation="eager",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -- Load eval data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"]

    # Pick a diverse set of examples — try to get some from each class
    examples_per_class = max(1, num_examples // 6)
    selected_indices = []
    for cls_idx in range(6):
        cls_indices = [i for i in range(len(eval_ds)) if eval_ds[i]["label"] == cls_idx]
        selected_indices.extend(cls_indices[:examples_per_class])
    # Fill remaining with random
    remaining = num_examples - len(selected_indices)
    if remaining > 0:
        other_indices = [i for i in range(len(eval_ds)) if i not in selected_indices]
        selected_indices.extend(other_indices[:remaining])
    selected_indices = selected_indices[:num_examples]

    # -- Analyze each example --
    analyses = []
    for idx_num, ds_idx in enumerate(selected_indices):
        text = eval_ds[ds_idx]["text"]
        true_label = eval_ds[ds_idx]["label"]

        await flyte.report.replace.aio(
            wrap_report(
                f"<h2>Explore Inference</h2>"
                f"<p>Analyzing example {idx_num + 1}/{len(selected_indices)}...</p>"
                f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
                f'<div style="background:#0f3460;width:{(idx_num + 1) / len(selected_indices) * 100:.1f}%;height:100%;border-radius:4px;"></div>'
                f'</div>'
            ),
            do_flush=True,
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        token_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Forward pass with attention
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(logits).item())

        # -- Attention: CLS token attention from last layer --
        # attentions shape: (num_layers, batch, num_heads, seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
        # Average across heads, take CLS row (index 0)
        cls_attention = last_layer_attention.mean(dim=0)[0].cpu().numpy()  # (seq_len,)

        # Remove [CLS] and [SEP] and padding from visualization
        real_token_mask = []
        clean_tokens = []
        clean_attention = []
        for i, tok in enumerate(tokens):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
                continue
            if tok == tokenizer.pad_token:
                continue
            clean_tokens.append(tok)
            clean_attention.append(float(cls_attention[i]))
            real_token_mask.append(i)

        # -- Token importance via gradient attribution --
        # Re-run with gradients enabled on embeddings
        embedding_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "word" in name.lower():
                embedding_layer = module
                break
        if embedding_layer is None:
            # Fallback: find the first large embedding
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding) and module.weight.shape[0] > 1000:
                    embedding_layer = module
                    break

        importance_scores = [0.0] * len(clean_tokens)
        if embedding_layer is not None:
            inputs_grad = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs_grad = {k: v.to(device) for k, v in inputs_grad.items()}

            embeddings = embedding_layer(inputs_grad["input_ids"])
            embeddings.retain_grad()

            # Run model with embeddings instead of input_ids
            # We need to hook into the model to replace the embedding output
            embedding_output = [None]

            def hook_fn(module, input, output):
                embedding_output[0] = output
                return embeddings.requires_grad_(True)

            handle = embedding_layer.register_forward_hook(hook_fn)

            outputs_grad = model(**inputs_grad)
            handle.remove()

            # Gradient of predicted class w.r.t. embeddings
            pred_score = outputs_grad.logits[0, pred_idx]
            pred_score.backward()

            if embeddings.grad is not None:
                # Token importance = L2 norm of (gradient * embedding) per token
                token_importance = (embeddings.grad[0] * embeddings[0]).norm(dim=-1).detach().cpu().numpy()
                for clean_idx, orig_idx in enumerate(real_token_mask):
                    if orig_idx < len(token_importance):
                        importance_scores[clean_idx] = float(token_importance[orig_idx])

            model.zero_grad()

        analyses.append({
            "text": text,
            "true_label": true_label,
            "pred_idx": pred_idx,
            "probs": probs,
            "tokens": clean_tokens,
            "attention": clean_attention,
            "importance": importance_scores,
            "correct": pred_idx == true_label,
        })

    # -- Build report --
    log.info("Building explore_inference report...")

    # Overall summary
    correct = sum(1 for a in analyses if a["correct"])
    total = len(analyses)

    # Separate correct vs wrong
    correct_analyses = [a for a in analyses if a["correct"]]
    wrong_analyses = [a for a in analyses if not a["correct"]]

    # -- Build example cards --
    examples_html = ""
    for a in analyses:
        true_name = EMOTION_LABELS[a["true_label"]]
        pred_name = EMOTION_LABELS[a["pred_idx"]]
        status_badge = "badge-success" if a["correct"] else "badge-danger"
        status_text = "Correct" if a["correct"] else "Wrong"

        # Confidence bars
        conf_bars = make_confidence_bars(
            labels=EMOTION_LABELS,
            probabilities=a["probs"],
            predicted_idx=a["pred_idx"],
            true_idx=a["true_label"],
        )

        # Attention heatmap
        attention_viz = make_attention_text(
            tokens=a["tokens"],
            weights=a["attention"],
            title="Attention (what the model looks at for its prediction — darker = more attention)",
        )

        # Token importance
        importance_viz = make_token_importance_text(
            tokens=a["tokens"],
            importance=a["importance"],
            title="Token importance (gradient attribution — green = supports prediction, red = opposes)",
        )

        text_preview = a["text"][:300]
        examples_html += f"""
<div class="card">
  <p style="font-size:1em;"><b>"{text_preview}"</b></p>
  <p>True: <b>{true_name}</b> | Predicted: <span class="badge {status_badge}">{pred_name} ({status_text})</span>
     | Confidence: <b>{a['probs'][a['pred_idx']]:.1%}</b></p>
  <div style="margin:12px 0;">{conf_bars}</div>
  <div style="margin:12px 0;">{attention_viz}</div>
  <div style="margin:12px 0;">{importance_viz}</div>
</div>"""

    # -- Misclassification spotlight --
    misclass_html = ""
    if wrong_analyses:
        # Sort by confidence (most confident wrong first)
        wrong_sorted = sorted(wrong_analyses, key=lambda a: a["probs"][a["pred_idx"]], reverse=True)

        misclass_html = "<h3>Misclassification Spotlight</h3>"
        misclass_html += '<div class="note">These are the model\'s most confident wrong predictions — cases where the model is sure but incorrect. These reveal the model\'s blind spots.</div>'

        for a in wrong_sorted[:3]:
            true_name = EMOTION_LABELS[a["true_label"]]
            pred_name = EMOTION_LABELS[a["pred_idx"]]
            conf = a["probs"][a["pred_idx"]]
            true_conf = a["probs"][a["true_label"]]

            misclass_html += f"""
<div class="card" style="border-left:4px solid #e63946;">
  <p><b>"{a['text'][:200]}"</b></p>
  <p>Predicted <span class="badge badge-danger">{pred_name}</span> ({conf:.1%})
     but true label is <span class="badge badge-info">{true_name}</span> ({true_conf:.1%})</p>
  <p style="font-size:0.85em;color:#6c757d;">
     The model assigned {conf:.1%} confidence to {pred_name} vs {true_conf:.1%} to {true_name}.
     {"The model was very sure here — this is a genuine blind spot." if conf > 0.7 else "The model was uncertain — the true class was a close second."}
  </p>
</div>"""

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Explore Inference — Attention &amp; Attribution</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{correct}/{total}</div><div class="label">Correct</div></div>'
            f'  <div class="stat"><div class="value">{correct/total:.0%}</div><div class="label">Accuracy (sample)</div></div>'
            f'  <div class="stat"><div class="value">{len(wrong_analyses)}</div><div class="label">Errors to Analyze</div></div>'
            f'</div>'
            f'<div class="note">'
            f'<b>How to read the visualizations below:</b><br/>'
            f'<b>Attention heatmap:</b> Shows which tokens the [CLS] token attends to in the final layer '
            f'(averaged across all attention heads). Darker = more attention. This reveals what the model "looks at" when making its classification decision.<br/>'
            f'<b>Token importance:</b> Gradient-based attribution showing which tokens most influence the prediction. '
            f'Green = supports the prediction, Red = opposes it. Computed as gradient &times; embedding norm.'
            f'</div>'
            f"<h3>Example Analysis</h3>"
            f"{examples_html}"
            f"{misclass_html}"
        ),
        do_flush=True,
    )

    return json.dumps({
        "num_examples": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1),
        "num_misclassifications": len(wrong_analyses),
        "analyses": [
            {
                "text": a["text"][:200],
                "true_label": EMOTION_LABELS[a["true_label"]],
                "predicted": EMOTION_LABELS[a["pred_idx"]],
                "confidence": round(a["probs"][a["pred_idx"]], 3),
                "correct": a["correct"],
            }
            for a in analyses
        ],
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 200,
    num_explore_examples: int = 12,
) -> flyte.io.Dir:
    """
    ModernBERT emotion classification pipeline.

    Returns the fine-tuned model directory (used by serve.py for deployment).

    1. Download emotion dataset (6 classes from Twitter text)
    2. Fine-tune ModernBERT for sequence classification
    3. Evaluate: base vs fine-tuned with confusion matrix
    4. Explore inference: attention heatmaps + token importance

    Args:
        model_name: HuggingFace encoder model to fine-tune.
        num_explore_examples: Number of examples for attention/attribution analysis.
    """
    log.info(f"Pipeline: {model_name} | emotion classification")
    steps = ["Get Data", "Train", "Evaluate", "Explore Inference"]

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Emotion Classification Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(0, steps)}"
            f'<div class="card"><p>Downloading emotion dataset...</p></div>'
        ),
        do_flush=True,
    )

    # Step 1: Get data
    data_dir = await get_data(max_train_samples, max_eval_samples)

    # Step 2: Train
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Emotion Classification Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(1, steps)}"
            f'<div class="card"><p>Fine-tuning for emotion classification...</p></div>'
        ),
        do_flush=True,
    )

    finetuned_dir = await train(model_name, data_dir, epochs, lr, batch_size, warmup_steps)

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Emotion Classification Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(2, steps)}"
            f'<div class="card"><p>Evaluating base vs fine-tuned model...</p></div>'
        ),
        do_flush=True,
    )

    eval_result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    eval_metrics = json.loads(eval_result)

    # Step 4: Explore inference
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Emotion Classification Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(3, steps)}"
            f'<div class="card"><p>Analyzing attention patterns and token importance...</p></div>'
        ),
        do_flush=True,
    )

    explore_result = await explore_inference(finetuned_dir, data_dir, num_explore_examples)

    # -- Final report --
    improvement = eval_metrics["improvement"]
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Emotion Classification Pipeline Complete</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(4, steps)}"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{eval_metrics["base_accuracy"]}%</div><div class="label">Base Accuracy</div></div>'
            f'  <div class="stat"><div class="value">{eval_metrics["finetuned_accuracy"]}%</div><div class="label">Fine-tuned Accuracy</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'  <div class="stat"><div class="value">{eval_metrics["finetuned_f1"]}%</div><div class="label">Weighted F1</div></div>'
            f'</div>'
        ),
        do_flush=True,
    )

    log.info(f"Pipeline complete. Accuracy improvement: {improvement:+.1f}pp")
    return finetuned_dir
