"""Fine-tune a small LLM for text-to-SQL on your Mac's GPU, with Flyte.

    flyte run workflow.py gpu_check                 # step 0: is the task really on the Mac's GPU?
    flyte run workflow.py baseline                  # step 1: score the untouched model
    flyte run workflow.py pipeline                  # steps 1-3: data -> baseline -> LoRA -> evaluate
    ACCELERATOR=T4 flyte run workflow.py pipeline   # the same pipeline on a cluster GPU

Every GPU task asks for `gpu="AppleSilicon"` (see config.py), so on an Apple Silicon devbox it runs
natively on the Mac and PyTorch uses the `mps` device. The pipeline task itself is a plain CPU task
and stays in its container: it only orchestrates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.report

import sql
from config import ACCELERATOR, MODEL_NAME, cpu_env, gpu_env
from report_helpers import make_bar_chart, make_line_chart, wrap_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET = "b-mc2/sql-create-context"


def _stat(value, label: str) -> str:
    return f"<div class='stat'><div class='value'>{value}</div><div class='label'>{label}</div></div>"


# ── Step 0: prove the GPU ─────────────────────────────────────────────────────


@gpu_env.task(report=True)
async def gpu_check(n: int = 2048, seconds: float = 2.0) -> dict:
    """Where does this task run, and how fast is a matmul there versus the CPU? No downloads."""
    import platform

    import torch

    device = sql.pick_device(ACCELERATOR)

    def tflops(dev) -> float:
        a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
        (a @ b).sum().item()  # warm up (compiles Metal kernels on mps)
        runs, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            (a @ b).sum().item()
            runs += 1
        return round(2 * n**3 * runs / (time.perf_counter() - start) / 1e12, 2)

    accel, cpu = tflops(device), tflops(torch.device("cpu"))
    result = {
        "platform": f"{platform.system()} {platform.machine()}",
        "chip": sql.device_name(device),
        "device": str(device),
        "dtype": str(sql.pick_dtype(device)).replace("torch.", ""),
        "accelerator_tflops": accel,
        "cpu_tflops": cpu,
        "speedup": round(accel / cpu, 1) if cpu else None,
        "torch": torch.__version__,
    }
    for k, v in result.items():
        log.info(f"{k:>20}: {v}")

    body = (
        "<h2>Where this task ran</h2><div class='stats'>"
        + _stat(result["platform"], "platform")
        + _stat(result["chip"], "chip")
        + _stat(result["device"], "torch device")
        + _stat(f"{result['speedup']}x", "GPU vs CPU")
        + "</div>"
        + "<div class='chart-container'>"
        + make_bar_chart(["CPU", str(device)], {"TFLOPS": [cpu, accel]}, title=f"{n}x{n} fp32 matmul", colors=["#adb5bd", "#0f3460"])
        + "</div>"
        "<div class='card'>The devbox container is a Linux VM with no Metal. A <code>gpu=\"AppleSilicon\"</code> task runs "
        "natively on the Mac instead, so <code>platform</code> says Darwin and torch has an <code>mps</code> device.</div>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return result


# ── Step 1: data ─────────────────────────────────────────────────────────────


@cpu_env.task(cache="auto")
async def prepare_data(train_samples: int = 500, eval_samples: int = 100, seed: int = 42) -> tuple[flyte.io.File, flyte.io.File]:
    """Download the text-to-SQL corpus and split it. Returns two JSONL files."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train").shuffle(seed=seed)
    train, evaluation = ds.select(range(train_samples)), ds.select(range(train_samples, train_samples + eval_samples))
    out = Path(tempfile.mkdtemp())
    for name, split in (("train", train), ("eval", evaluation)):
        with open(out / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps({"context": ex["context"], "question": ex["question"], "answer": ex["answer"]}) + "\n")
    log.info(f"{len(train)} train / {len(evaluation)} eval examples from {DATASET}")
    return await flyte.io.File.from_local(out / "train.jsonl"), await flyte.io.File.from_local(out / "eval.jsonl")


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_model(adapter_dir: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = sql.pick_device(ACCELERATOR)
    dtype = sql.pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    log.info(f"{MODEL_NAME} on {device} ({dtype}){' + adapter' if adapter_dir else ''}")
    return tokenizer, model, device


# ── Step 2 / 4: evaluate ─────────────────────────────────────────────────────


@gpu_env.task(report=True)
async def evaluate(
    eval_file: flyte.io.File, adapter: flyte.io.Dir | None = None, num_examples: int = 100, max_new_tokens: int = 128
) -> dict:
    """Greedy-decode SQL for held-out questions and score exact match. With `adapter`, the fine-tuned model."""
    import torch

    examples = _read_jsonl(await eval_file.download())[:num_examples]
    adapter_dir = await adapter.download() if adapter else None
    tokenizer, model, device = _load_model(adapter_dir)
    label = "fine-tuned" if adapter else "base"

    rows, correct, start = [], 0, time.perf_counter()
    for i, ex in enumerate(examples):
        enc = tokenizer(sql.render_prompt(tokenizer, ex), return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        pred = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        ok = sql.exact_match(pred, ex["answer"])
        correct += ok
        rows.append({"question": ex["question"], "gold": ex["answer"], "prediction": pred, "correct": ok})
        if (i + 1) % 10 == 0:
            log.info(f"[{label}] {i + 1}/{len(examples)}: {correct}/{i + 1} exact match")
    seconds = time.perf_counter() - start
    accuracy = correct / len(examples)

    table = "".join(
        f"<tr><td>{'✅' if r['correct'] else '❌'}</td><td>{r['question']}</td>"
        f"<td><code>{r['gold']}</code></td><td><code>{r['prediction'][:300]}</code></td></tr>"
        for r in rows
    )
    body = (
        f"<h2>{label.title()} model: {MODEL_NAME}</h2><div class='stats'>"
        + _stat(f"{accuracy:.0%}", "exact match")
        + _stat(f"{correct}/{len(examples)}", "correct")
        + _stat(str(device), "device")
        + _stat(f"{seconds / len(examples):.2f}s", "per question")
        + "</div>"
        "<table><tr><th></th><th>Question</th><th>Gold</th><th>Prediction</th></tr>" + table + "</table>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return {"label": label, "exact_match": round(accuracy, 4), "correct": correct, "n": len(examples), "seconds": round(seconds, 1), "device": str(device)}


# ── Step 3: train ────────────────────────────────────────────────────────────


@gpu_env.task(report=True)
async def train(
    train_file: flyte.io.File,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 2,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """LoRA fine-tune on the GPU this task landed on. Plain PyTorch loop, live loss chart, adapter out."""
    import torch
    from peft import LoraConfig, get_peft_model

    examples = _read_jsonl(await train_file.download())
    tokenizer, model, device = _load_model()
    tokenizer.padding_side = "right"
    model.train()

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    texts = [sql.render_train(tokenizer, ex) for ex in examples]
    batches_per_epoch = (len(texts) + batch_size - 1) // batch_size
    max_steps = max(1, epochs * batches_per_epoch // grad_accum)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.1, anneal_strategy="cos")

    history: list[dict] = []
    t0 = time.time()

    def head(status: str) -> str:
        return (
            f"<h2>LoRA: {MODEL_NAME}</h2><p>{status}</p><div class='stats'>"
            + _stat(str(device), "device")
            + _stat(str(sql.pick_dtype(device)).replace("torch.", ""), "dtype")
            + _stat(f"{trainable / 1e6:.1f}M", f"trainable ({trainable / total:.2%})")
            + _stat(f"{len(texts):,}", "train examples")
            + _stat(f"{max_steps}", "optimizer steps")
            + "</div>"
            f"<div class='card'><b>LoRA targets:</b> <code>{', '.join(targets)}</code> &middot; rank {lora_r}, alpha {lora_alpha} "
            f"&middot; batch {batch_size} x {grad_accum} accumulation &middot; lr {lr}</div>"
        )

    def live(status: str) -> str:
        body = head(status)
        if history:
            cur = history[-1]
            pct = cur["step"] / max_steps * 100
            body += (
                f"<div class='card'><b>Step {cur['step']}/{max_steps}</b> ({pct:.0f}%) &middot; loss "
                f"<span class='highlight'>{cur['loss']:.4f}</span> &middot; {(time.time() - t0) / 60:.1f} min"
                "<div style='background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;'>"
                f"<div style='background:#0f3460;width:{pct:.1f}%;height:100%;border-radius:4px;'></div></div></div>"
                "<div class='chart-container'>"
                + make_line_chart(history, x_key="step", y_keys=["loss"], title="Training loss", x_label="optimizer step", y_label="loss")
                + "</div>"
            )
        return wrap_report(body)

    await flyte.report.replace.aio(live("Training…"), do_flush=True)

    step, micro, running = 0, 0, 0.0
    for epoch in range(epochs):
        order = torch.randperm(len(texts)).tolist()
        for b in range(batches_per_epoch):
            idx = order[b * batch_size : (b + 1) * batch_size]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            loss = model(**enc, labels=labels).loss / grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step + 1 < max_steps:
                    schedule.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                history.append({"step": step, "loss": round(running, 4), "epoch": epoch + 1})
                running = 0.0
                if step % 5 == 0 or step == max_steps:
                    log.info(f"step {step}/{max_steps} loss {history[-1]['loss']:.4f} ({(time.time() - t0):.0f}s)")
                    await flyte.report.replace.aio(live("Training…"), do_flush=True)
                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    seconds = time.time() - t0
    if device.type != "cpu":
        getattr(torch, device.type).synchronize()
    peak_gb = sql.peak_memory_gb(device)
    final_loss = history[-1]["loss"] if history else None
    log.info(f"done: {step} steps in {seconds / 60:.1f} min, final loss {final_loss}, peak memory {peak_gb:.1f} GB")

    out = Path(tempfile.mkdtemp()) / "adapter"
    model.save_pretrained(out)
    metrics = {
        "model": MODEL_NAME,
        "device": str(device),
        "steps": step,
        "train_examples": len(texts),
        "train_seconds": round(seconds, 1),
        "final_loss": final_loss,
        "first_loss": history[0]["loss"] if history else None,
        "peak_memory_gb": round(peak_gb, 2),
        "trainable_params": trainable,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "history": history,
    }
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    await flyte.report.replace.aio(
        live(f"Done: {step} steps in {seconds / 60:.1f} min on {device}, {peak_gb:.1f} GB peak, loss {history[0]['loss']:.3f} → {final_loss:.3f}"),
        do_flush=True,
    )
    return await flyte.io.Dir.from_local(out)


# ── Entrypoints ──────────────────────────────────────────────────────────────


@cpu_env.task
async def baseline(eval_samples: int = 100, num_eval_examples: int = 100) -> dict:
    """Score the untouched model. Fine-tuning needs a gap to close; this measures it."""
    _, eval_file = await prepare_data(train_samples=500, eval_samples=eval_samples)
    return await evaluate(eval_file, num_examples=num_eval_examples)


@cpu_env.task(report=True)
async def pipeline(
    train_samples: int = 500,
    eval_samples: int = 100,
    num_eval_examples: int = 100,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 2,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> tuple[dict, flyte.io.Dir]:
    """Data -> baseline -> LoRA -> evaluate. Returns the summary and the adapter directory."""
    train_file, eval_file = await prepare_data(train_samples=train_samples, eval_samples=eval_samples)
    base = await evaluate(eval_file, num_examples=num_eval_examples)
    adapter = await train(train_file, epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum, max_length=max_length, lora_r=lora_r, lora_alpha=lora_alpha)
    tuned = await evaluate(eval_file, adapter=adapter, num_examples=num_eval_examples)
    metrics = json.loads(Path(await adapter.download()).joinpath("train_metrics.json").read_text())

    summary = {
        "model": MODEL_NAME,
        "accelerator": ACCELERATOR,
        "device": tuned["device"],
        "base_exact_match": base["exact_match"],
        "finetuned_exact_match": tuned["exact_match"],
        "train_seconds": metrics["train_seconds"],
        "train_steps": metrics["steps"],
        "base_eval_seconds": base["seconds"],
        "finetuned_eval_seconds": tuned["seconds"],
        "peak_memory_gb": metrics["peak_memory_gb"],
        "final_loss": metrics["final_loss"],
    }
    log.info(json.dumps(summary, indent=2))
    body = (
        f"<h2>{MODEL_NAME} on {tuned['device']}</h2><div class='stats'>"
        + _stat(f"{base['exact_match']:.0%}", "base exact match")
        + _stat(f"{tuned['exact_match']:.0%}", "fine-tuned exact match")
        + _stat(f"{metrics['train_seconds'] / 60:.1f} min", f"training ({metrics['steps']} steps)")
        + _stat(f"{metrics['peak_memory_gb']:.1f} GB", "peak GPU memory (driver, incl. cache)")
        + "</div><div class='chart-container'>"
        + make_bar_chart(["base", "fine-tuned"], {"exact match %": [base["exact_match"] * 100, tuned["exact_match"] * 100]}, title=f"Exact match on {tuned['n']} held-out questions", colors=["#adb5bd", "#0f3460"], y_max_cap=100)
        + "</div><div class='chart-container'>"
        + make_line_chart(metrics["history"], x_key="step", y_keys=["loss"], title="Training loss", x_label="optimizer step", y_label="loss")
        + "</div>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return summary, adapter


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline)
    print(run.url)
    run.wait()
    print(run.outputs())
