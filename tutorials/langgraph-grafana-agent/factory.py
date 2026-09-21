"""The factory floor: candidate models, evaluation, and LoRA fine-tuning.

Plain functions over Hugging Face `transformers`, `peft` and `trl`. `tools.py` wraps
them as Flyte tasks; nothing here knows about Flyte or the agent. Runs on a T4 on the
cluster, on Colab's free T4 under `--local`, or (slowly, for smoke tests) on a laptop CPU.

A model reference is either a candidate id from CANDIDATES or a directory produced by
`fine_tune` (a local path, or a remote `s3://...` path the tools download first).
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tickets import DATASET_VERSIONS, as_chat, load_split, parse_label

CANDIDATES: dict[str, dict] = {
    "smollm2-360m": {
        "hf": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "license": "Apache-2.0",
        "cpu_ok": True,
        "notes": "Smallest chat model here. Instruction-tuned, but weak on multi-way classification.",
    },
    "qwen2.5-0.5b": {
        "hf": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "license": "Apache-2.0",
        "notes": "Strong for its size; follows the label-only instruction most of the time.",
    },
    "qwen2.5-1.5b": {
        "hf": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "license": "Apache-2.0",
        "notes": "Most accurate base model here; three times the weights of the 0.5B.",
    },
    "smollm2-1.7b": {
        "hf": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "license": "Apache-2.0",
        "notes": "The biggest SmolLM2. Same family as the 360M, a different league.",
    },
    "modernbert-base": {
        "hf": "answerdotai/ModernBERT-base",
        "params": "149M",
        "license": "Apache-2.0",
        "kind": "encoder",
        "cpu_ok": True,
        "notes": (
            "Not a chat model: an encoder with a classification head. Cannot be used zero-shot (the head is "
            "untrained until you fine-tune it; give it 2-3 epochs, which take seconds), returns a label rather "
            "than text, and is very fast."
        ),
    },
}


def kind_of(path_or_id: str, model_ref: str | None = None) -> str:
    """'encoder' for a sequence-classification model, 'causal' for a chat model."""
    if model_ref in CANDIDATES:
        return CANDIDATES[model_ref].get("kind", "causal")
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(path_or_id)
        archs = " ".join(getattr(cfg, "architectures", None) or [])
        if "SequenceClassification" in archs or getattr(cfg, "model_type", "") in (
            "modernbert",
            "bert",
            "distilbert",
            "roberta",
            "deberta-v2",
        ):
            return "encoder"
    except Exception:
        pass
    return "causal"


MAX_NEW_TOKENS = 8  # a label is 1-4 tokens; base models that ramble pay for it in latency


@dataclass
class EvalResult:
    model: str
    n: int
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    load_seconds: float
    device: str
    per_category: dict[str, float]
    mistakes: list[dict]

    def summary(self) -> str:
        cats = ", ".join(f"{c} {a:.0%}" for c, a in self.per_category.items())
        return (
            f"{self.model}: accuracy {self.accuracy:.1%} on {self.n} tickets; "
            f"latency p50 {self.latency_p50_ms:.0f} ms, p95 {self.latency_p95_ms:.0f} ms per ticket "
            f"({self.device}, batch size 1); model load {self.load_seconds:.0f}s.\n"
            f"per category: {cats}"
        )


def device() -> str:
    """cuda if there is one, else cpu. MPS is skipped on purpose: it is unreliable for these sizes."""
    forced = os.environ.get("FACTORY_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve(model_ref: str) -> tuple[str, str]:
    """(display name, path or hf id) for a candidate id or a fine-tuned directory."""
    if model_ref in CANDIDATES:
        return model_ref, CANDIDATES[model_ref]["hf"]
    p = Path(model_ref)
    if p.exists() and (p / "config.json").exists():
        return p.name, str(p)
    raise ValueError(
        f"unknown model {model_ref!r}; candidates: {', '.join(CANDIDATES)}, or a fine-tuned model directory"
    )


def _load(path: str, dev: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
    return tok, model


def _has_trained_head(path: str) -> bool:
    """True for a checkpoint that already carries a trained classification head."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    labels = getattr(cfg, "id2label", None) or {}
    return "SequenceClassification" in archs and any(not str(v).startswith("LABEL_") for v in labels.values())


def _load_encoder(path: str, dev: str, categories: list[str]):
    """A sequence-classification model.

    A base encoder gets a fresh, untrained head sized to the dataset's labels. A fine-tuned
    checkpoint keeps its own head and label set, whatever dataset it is being scored on:
    that is how a v1 model can be honestly measured on v2 (it simply cannot say the new
    label), instead of having its head silently re-initialized to random.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if _has_trained_head(path):
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(dev).eval()
    else:
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                path,
                num_labels=len(categories),
                id2label=dict(enumerate(categories)),
                label2id={c: i for i, c in enumerate(categories)},
                ignore_mismatched_sizes=True,
                dtype=torch.float32,
            )
            .to(dev)
            .eval()
        )
    return tok, model


def evaluate(model_ref: str, n: int = 120, version: str = "v1") -> EvalResult:
    """Classify `n` held-out tickets one at a time; report accuracy and per-ticket latency."""
    import torch

    dev = device()
    name, path = resolve(model_ref)
    categories = DATASET_VERSIONS[version]
    kind = kind_of(path, model_ref)
    t0 = time.perf_counter()
    if kind == "encoder":
        tok, model = _load_encoder(path, dev, categories)
        # A fine-tuned encoder carries its own label set; trust it over the dataset's.
        trained_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    else:
        tok, model = _load(path, dev)
    load_seconds = time.perf_counter() - t0

    test = load_split("test", version)
    tickets = test[: max(1, min(n, len(test)))]
    latencies, correct = [], 0
    per_cat_hits = {c: [0, 0] for c in categories}
    mistakes = []
    with torch.inference_mode():
        for i, t in enumerate(tickets):
            if kind == "encoder":
                inputs = tok(t.text, return_tensors="pt", truncation=True, max_length=256).to(dev)
            else:
                prompt = tok.apply_chat_template(
                    as_chat(t, with_answer=False, version=version), tokenize=False, add_generation_prompt=True
                )
                inputs = tok(prompt, return_tensors="pt").to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if kind == "encoder":
                logits = model(**inputs).logits
            else:
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id
                )
            if dev == "cuda":
                torch.cuda.synchronize()
            if i > 0:  # the first call includes kernel warm-up
                latencies.append((time.perf_counter() - t1) * 1000)
            if kind == "encoder":
                reply = str(trained_labels[int(logits.argmax(-1))])
                pred = reply if reply in categories else "other"
            else:
                reply = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                pred = parse_label(reply, categories)
            per_cat_hits[t.label][1] += 1
            if pred == t.label:
                correct += 1
                per_cat_hits[t.label][0] += 1
            elif len(mistakes) < 8:
                mistakes.append({"ticket": t.text[:90], "expected": t.label, "got": pred, "raw": reply.strip()[:40]})

    latencies = latencies or [0.0]
    return EvalResult(
        model=name,
        n=len(tickets),
        accuracy=correct / len(tickets),
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        load_seconds=load_seconds,
        device=dev,
        per_category={c: (h / t if t else 0.0) for c, (h, t) in per_cat_hits.items()},
        mistakes=mistakes,
    )


@dataclass
class TrainResult:
    base: str
    output_dir: str
    steps: int
    epochs: float
    train_loss_start: float
    train_loss_end: float
    seconds: float
    device: str
    history: list[dict] = None  # [{"step", "epoch", "loss"}, ...] every logging step

    def summary(self) -> str:
        how = "full fine-tune (encoder)" if "modernbert" in self.base.lower() or "bert" in self.base.lower() else "LoRA"
        return (
            f"fine-tuned {self.base} with {how} for {self.steps} steps ({self.epochs:g} epoch(s)) in {self.seconds:.0f}s on {self.device}; "
            f"loss {self.train_loss_start:.2f} -> {self.train_loss_end:.2f}. Weights saved to {self.output_dir}"
        )


def fine_tune(
    model_ref: str,
    output_dir: str,
    epochs: float = 1.0,
    max_steps: int = -1,
    lr: float = 2e-4,
    batch_size: int = 8,
    lora_r: int = 16,
    version: str = "v1",
    callbacks: list | None = None,
) -> TrainResult:
    """LoRA SFT on the synthetic train split; merge the adapter and save a plain HF model.

    Encoders take a different road: a full fine-tune of the classification head and body
    with the plain HF Trainer (they are small enough that LoRA buys nothing).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dev = device()
    base, path = resolve(model_ref)
    if kind_of(path, model_ref) == "encoder":
        return _fine_tune_encoder(
            base,
            path,
            output_dir,
            epochs=epochs,
            max_steps=max_steps,
            lr=5e-5,
            batch_size=16,
            version=version,
            dev=dev,
            callbacks=callbacks,
        )
    train = load_split("train", version)
    ds = Dataset.from_list([{"messages": as_chat(t, with_answer=True, version=version)} for t in train])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 weights with fp16 autocast: the T4 has no bf16, and fp16 master weights are
    # a bad idea for training. At these sizes fp32 fits with room to spare.
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=os.path.join(tempfile.mkdtemp(prefix="trainer-"), "state"),  # not inside the artifact
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=(dev == "cuda"),
        max_length=256,
        use_cpu=(dev == "cpu"),
        seed=7,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    history = loss_history(trainer.state.log_history)
    losses = [h["loss"] for h in history]
    if not losses:  # fewer steps than logging_steps: fall back to the run's mean loss
        losses = [h["train_loss"] for h in trainer.state.log_history if "train_loss" in h]

    merged = trainer.model.merge_and_unload()
    merged = merged.to(torch.float16) if dev == "cuda" else merged
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "factory.json"), "w") as f:
        json.dump(
            {"base": base, "epochs": epochs, "steps": trainer.state.global_step, "lora_r": lora_r, "dataset": version},
            f,
        )

    return TrainResult(
        base=base,
        output_dir=output_dir,
        steps=trainer.state.global_step,
        epochs=epochs,
        train_loss_start=losses[0] if losses else float("nan"),
        train_loss_end=losses[-1] if losses else float("nan"),
        seconds=seconds,
        device=dev,
        history=history,
    )


def loss_history(log_history: list[dict]) -> list[dict]:
    """The (step, epoch, loss) points the Trainer logged, for a chart."""
    return [
        {"step": h.get("step", 0), "epoch": round(h.get("epoch", 0.0), 3), "loss": h["loss"]}
        for h in log_history
        if "loss" in h
    ]


def _fine_tune_encoder(
    base, path, output_dir, *, epochs, max_steps, lr, batch_size, version, dev, callbacks=None
) -> TrainResult:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    categories = DATASET_VERSIONS[version]
    tok = AutoTokenizer.from_pretrained(path)
    # Training always sizes the head to the dataset's labels (re-initialized if the count
    # changed), which is what "fine-tune on v2" means for a model trained on v1.
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=len(categories),
        id2label=dict(enumerate(categories)),
        label2id={c: i for i, c in enumerate(categories)},
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    )
    train = load_split("train", version)
    ds = Dataset.from_list([{"text": t.text, "label": categories.index(t.label)} for t in train])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
    args = TrainingArguments(
        output_dir=os.path.join(tempfile.mkdtemp(prefix="trainer-"), "state"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=(dev == "cuda"),
        use_cpu=(dev == "cpu"),
        seed=7,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok, callbacks=callbacks or None)
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    history = loss_history(trainer.state.log_history)
    losses = [h["loss"] for h in history] or [h["train_loss"] for h in trainer.state.log_history if "train_loss" in h]
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "factory.json"), "w") as f:
        json.dump(
            {"base": base, "kind": "encoder", "epochs": epochs, "steps": trainer.state.global_step, "dataset": version},
            f,
        )
    return TrainResult(
        base=base,
        output_dir=output_dir,
        steps=trainer.state.global_step,
        epochs=epochs,
        train_loss_start=losses[0] if losses else float("nan"),
        train_loss_end=losses[-1] if losses else float("nan"),
        seconds=seconds,
        device=dev,
        history=history,
    )


def result_to_json(r) -> str:
    return json.dumps(asdict(r))
