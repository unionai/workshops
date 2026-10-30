"""Fine-tune Qwen3.5-4B on support tickets three ways, and measure each the same way.

    flyte run workflow.py baseline                      # score the untouched model first
    flyte run workflow.py pipeline --method lora        # BF16 base + LoRA adapter (default)
    flyte run workflow.py pipeline --method qlora       # 4-bit NF4 base + LoRA adapter
    flyte run workflow.py pipeline --method full        # every weight, bf16, 8-bit AdamW
    flyte run workflow.py compare                       # all three, in parallel, one report

The task (support.py): a customer message in, `{"intent": ..., "reply": ...}` out. Intent is one of
27 labels the base model has never seen named, which is what gives fine-tuning a gap to close.

Every evaluation reports the same two recomputable numbers per row (intent exact match, format
kept) and, with `JUDGE` set, a pairwise position-swapped judge for the reply's voice. Base and
fine-tuned are always scored on the same held-out rows, in the same task, so the comparison
isolates what training changed.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import random
import tempfile
import time

import flyte
import flyte.io
import flyte.report

import support
from config import ARTIFACTS, DATASET, JUDGE, MODEL, MODEL_ID, SHAPES, cpu_env, gpu_env
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METHODS = ("full", "lora", "qlora")
ARTIFACT_NAME = f"{MODEL_ID}-support"
ARTIFACT_DESCRIPTION = "Qwen3.5-4B fine-tuned for support-ticket intent + reply"
ARTIFACT_TASK = "text-generation"
ARTIFACT_ATTRS = lambda r: {"method": r["method"], "intent_accuracy": r["finetuned"]["intent_accuracy"], "base_intent_accuracy": r["base"]["intent_accuracy"], "format_rate": r["finetuned"]["format_rate"], "train_examples": r.get("train", {}).get("train_examples", "")}


def _esc(text) -> str:
    return html.escape(str(text))


def _badge(method: str) -> str:
    return {
        "full": '<span class="badge badge-danger">Full fine-tune (bf16)</span>',
        "qlora": '<span class="badge badge-success">QLoRA (4-bit NF4)</span>',
    }.get(method, '<span class="badge badge-info">LoRA (16-bit)</span>')


# ------------------------------------------------------------------
# Task 1: data
# ------------------------------------------------------------------


@cpu_env.task(cache="auto")
async def prepare_data(dataset_name: str = DATASET, train_samples: int = 800, eval_samples: int = 200) -> flyte.io.Dir:
    """Tickets, shuffled and split. The eval split is held out from every training run."""
    from datasets import DatasetDict, load_dataset

    ds = load_dataset(dataset_name, split="train").shuffle(seed=42)
    needed = train_samples + eval_samples
    if len(ds) < needed:
        raise ValueError(f"{dataset_name} has {len(ds)} rows, need {needed}")
    keep = ["instruction", "intent", "category", "response"]
    ds = ds.select_columns(keep)
    out = DatasetDict({"train": ds.select(range(train_samples)), "eval": ds.select(range(train_samples, needed))})
    path = os.path.join(tempfile.mkdtemp(), "tickets")
    out.save_to_disk(path)
    log.info(f"tickets: {train_samples} train / {eval_samples} eval, {len(set(ds['intent']))} intents")
    return await flyte.io.Dir.from_local(path)


# ------------------------------------------------------------------
# Shared: loading, generation, scoring
# ------------------------------------------------------------------


def _load(model_name: str, method: str, weights_dir: str | None = None):
    """Load the base the way `method` needs it: 16-bit for full and lora, 4-bit NF4 for qlora.

    `weights_dir` replaces the base with a full fine-tune's saved weights.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = weights_dir or model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = _device()
    kwargs = dict(dtype=torch.bfloat16)
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    if method == "qlora":
        if device.type != "cuda":
            raise RuntimeError("QLoRA needs bitsandbytes, which is CUDA-only. On a Mac use --method lora.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    return model, tokenizer


def _device():
    """cuda on a GPU pod, mps on an Apple Silicon devbox, else cpu (slow; only for a smoke test)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_name() -> str:
    import platform

    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        chip = platform.processor() or "Apple Silicon"
        try:
            import subprocess

            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or chip
        except Exception:
            pass
        return f"{chip} (mps)"
    return "cpu"


def _mem_gb(peak: bool = False) -> float:
    import torch

    if torch.cuda.is_available():
        return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 2**30
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0


def _generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 160, batch_size: int = 8) -> list[str]:
    import torch

    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs += tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    return outs


def score_model(model, tokenizer, examples: list[dict], batch_size: int = 8) -> dict:
    prompts = [support.generation_prompt(tokenizer, ex) for ex in examples]
    outputs = _generate(model, tokenizer, prompts, batch_size=batch_size)
    rows = [support.score(out, ex) for ex, out in zip(examples, outputs)]
    return {**support.summarize(rows), "per_example": rows}


# ------------------------------------------------------------------
# Task 2: train
# ------------------------------------------------------------------


@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    epochs: int = 1,
    lr: float = 0.0,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 768,
    lora_r: int = 32,
    lora_alpha: int = 64,
) -> flyte.io.Dir:
    """Train with the chosen method and return what it produced: an adapter (lora, qlora) or the full weights (full).

    Learning rates differ by an order of magnitude between the methods, so `lr=0` means "the usual
    default for this method": 1e-5 for full, 1e-4 for the adapters.
    """
    import torch
    from datasets import Dataset, load_from_disk
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    lr = lr or (1e-5 if method == "full" else 1e-4)
    cuda = torch.cuda.is_available()
    if method == "full" and not cuda:
        log.warning("Full fine-tuning off CUDA uses plain AdamW (fp32 state, 8 bytes per parameter): expect ~37 GB on top of the weights.")
    gpu_name = _device_name()
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0.0

    await flyte.report.replace.aio(
        wrap_report(f"<h2>Loading model</h2><h3>{_esc(model_name)}</h3><div class='card'><p>{_badge(method)} on <b>{_esc(gpu_name)}</b>" + (f" ({gpu_total_gb:.1f} GiB)" if cuda else "") + "</p></div>"),
        do_flush=True,
    )
    model, tokenizer = _load(model_name, method)
    weights_gb = _mem_gb()

    data = load_from_disk(await data_dir.download())["train"]
    pairs = [support.training_pair(tokenizer, ex) for ex in data]
    train_ds = Dataset.from_dict({"prompt": [p for p, _ in pairs], "completion": [c for _, c in pairs]})

    targets: list[str] = []
    peft_config = None
    if method in ("lora", "qlora"):
        from peft import LoraConfig

        targets = support.discover_lora_targets(model)
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=targets, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        log.info(f"LoRA targets: {targets}")
    else:
        # Full fine-tune: every weight trains. 8-bit AdamW keeps the optimizer state at 2 bytes per
        # parameter instead of 8, which is what lets a 4B model train on a 48 GB card.
        for p in model.parameters():
            p.requires_grad_(True)
        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                p.requires_grad_(False)

    training_log: list[dict] = []
    loop = asyncio.get_running_loop()
    t0 = time.time()
    header_stats: dict = {}

    def _head(title: str) -> str:
        return (
            f"<h2>{title}</h2><h3>{_esc(model_name)}</h3><div class='stat-grid'>"
            f"<div class='stat'><div class='value'>{_esc(gpu_name)}</div><div class='label'>GPU</div></div>"
            f"<div class='stat'><div class='value'>{method.upper()}</div><div class='label'>Method</div></div>"
            f"<div class='stat'><div class='value'>{weights_gb:.1f} GiB</div><div class='label'>Weights on GPU</div></div>"
            f"<div class='stat'><div class='value'>{header_stats.get('trainable_pct', '?')}%</div><div class='label'>Trainable</div></div>"
            f"<div class='stat'><div class='value'>{len(train_ds):,}</div><div class='label'>Train rows</div></div>"
            f"<div class='stat'><div class='value'>{lr}</div><div class='label'>Learning rate</div></div></div>"
            + (f"<div class='card'><b>LoRA targets ({len(targets)}):</b> <code>{', '.join(targets)}</code> &middot; rank {lora_r}, alpha {lora_alpha}</div>" if targets else "")
        )

    def _charts() -> str:
        if len(training_log) < 2:
            return ""
        return "<div class='chart-container'>" + make_line_chart(
            data=training_log, x_key="step", y_keys=["loss"], title="Training loss", x_label="Step", y_label="Loss", colors=["#5a7db5"]
        ) + "</div>"

    def _live(max_steps: int) -> str:
        body = _head("Training in progress")
        if training_log:
            cur = training_log[-1]
            pct = cur["step"] / max_steps * 100 if max_steps else 0
            body += (
                f"<div class='card'><b>Step {cur['step']}/{max_steps}</b> ({pct:.0f}%) | loss <span class='highlight'>{cur['loss']:.4f}</span> | "
                f"{(time.time() - t0) / 60:.1f} min<div style='background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;'>"
                f"<div style='background:#0f3460;width:{pct:.1f}%;height:100%;border-radius:4px;'></div></div></div>"
            )
        return wrap_report(body + _charts())

    class Live(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                training_log.append({"step": state.global_step, "loss": round(float(logs["loss"]), 4)})
                log.info(f"step={state.global_step}/{state.max_steps} loss={logs['loss']:.4f}")
                asyncio.run_coroutine_threadsafe(flyte.report.replace.aio(_live(state.max_steps), do_flush=True), loop)

    args = SFTConfig(
        output_dir=os.path.join(tempfile.mkdtemp(), "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,
        logging_steps=5,
        save_strategy="no",
        # Mixed precision on CUDA. On mps the weights are already bf16 and the adapters train in fp32
        # with no autocast, which is what works there today.
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=5,
        lr_scheduler_type="cosine",
        # 8-bit AdamW is what lets every weight of a 4B model train on one 48 GB card; it is CUDA-only.
        optim="adamw_bnb_8bit" if method == "full" and cuda else "adamw_torch",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, processing_class=tokenizer, peft_config=peft_config, callbacks=[Live()])
    if peft_config is not None:
        trainable, total = trainer.model.get_nb_trainable_parameters()
        if trainable == 0:
            raise RuntimeError(f"LoRA attached to nothing. Discovered targets: {targets}")
    else:
        total = sum(p.numel() for p in trainer.model.parameters())
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    header_stats["trainable_pct"] = round(trainable / total * 100, 2)
    log.info(f"Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    await flyte.report.replace.aio(_live(trainer.state.max_steps or 0), do_flush=True)
    await asyncio.to_thread(trainer.train)
    train_seconds = time.time() - t0
    peak_gb = _mem_gb(peak=True)
    final_loss = training_log[-1]["loss"] if training_log else None
    if final_loss is not None and final_loss != final_loss:
        raise RuntimeError("Training loss is NaN.")
    log.info(f"Done in {train_seconds / 60:.1f} min. Peak VRAM {peak_gb:.2f} GiB. Final loss {final_loss}")

    out_dir = os.path.join(tempfile.mkdtemp(), "adapter" if peft_config is not None else "weights")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {
        "method": method, "gpu": gpu_name, "gpu_total_gb": round(gpu_total_gb, 1), "weights_gb": round(weights_gb, 2),
        "peak_gb": round(peak_gb, 2), "train_seconds": round(train_seconds), "final_loss": final_loss,
        "trainable_params": trainable, "total_params": total, "lora_targets": targets, "train_examples": len(train_ds),
        "epochs": epochs, "lr": lr, "loss_curve": training_log,
        "artifact_mb": round(sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 2**20),
    }
    with open(os.path.join(out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f)

    await flyte.report.replace.aio(
        wrap_report(
            _head("Training complete") + "<div class='stat-grid'>"
            f"<div class='stat'><div class='value'>{final_loss}</div><div class='label'>Final loss</div></div>"
            f"<div class='stat'><div class='value'>{peak_gb:.1f} GiB</div><div class='label'>Peak VRAM</div></div>"
            f"<div class='stat'><div class='value'>{train_seconds / 60:.1f} min</div><div class='label'>Train time</div></div>"
            f"<div class='stat'><div class='value'>{metrics['artifact_mb']} MB</div><div class='label'>What came out</div></div></div>"
            + _charts()
            + "<div class='note'>The output is an adapter for lora and qlora (tens of MB) and the full weights for full "
            "(about 9 GB). The evaluate task scores base vs fine-tuned on the held-out rows.</div>"
        ),
        do_flush=True,
    )
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 3: evaluate, with an optional pairwise judge
# ------------------------------------------------------------------

JUDGE_PROMPT = """You are judging two customer-support replies to the same message. Judge ONLY the voice and
usefulness of the reply: warm, specific to the message, actionable, no filler, keeps {{placeholders}} intact.
Ignore which intent label was chosen.

Customer message:
{ticket}

Reply A:
{a}

Reply B:
{b}

Answer with exactly one letter: A if A is better, B if B is better, T if they are equally good."""


async def _judge_pairs(ticket_rows: list[dict], a_replies: list[str], b_replies: list[str]) -> dict:
    """Pairwise, blind, position-swapped: each pair is judged twice with A and B swapped, and only
    verdicts that agree across the swap count. Position bias shows up as disagreement, not as a win."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(8)

    async def ask(ticket: str, a: str, b: str) -> str:
        async with sem:
            msg = await client.messages.create(
                model=JUDGE, max_tokens=5,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(ticket=ticket, a=a or "(empty)", b=b or "(empty)")}],
            )
        text = "".join(getattr(b_, "text", "") for b_ in msg.content).strip().upper()
        return text[:1] if text[:1] in ("A", "B", "T") else "T"

    async def both(row: dict, a: str, b: str) -> str:
        v1, v2 = await asyncio.gather(ask(row["instruction"], a, b), ask(row["instruction"], b, a))
        v2 = {"A": "B", "B": "A"}.get(v2, v2)  # undo the swap
        return v1 if v1 == v2 else "T"  # disagreement across the swap is a tie

    verdicts = await asyncio.gather(*(both(r, a, b) for r, a, b in zip(ticket_rows, a_replies, b_replies)))
    n = max(len(verdicts), 1)
    return {
        "model": JUDGE, "pairs": len(verdicts),
        "base_wins": verdicts.count("A"), "finetuned_wins": verdicts.count("B"), "ties": verdicts.count("T"),
        "finetuned_win_rate": round(verdicts.count("B") / n * 100, 1),
        "verdicts": list(verdicts),
    }


@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    trained_dir: flyte.io.Dir | None,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    num_examples: int = 100,
    judge_pairs: int = 40,
) -> str:
    """Score the base, then the fine-tuned model, on the same held-out tickets. Base loads the way it was trained
    (4-bit for qlora) so the comparison isolates the training."""
    import torch
    from datasets import load_from_disk

    examples = list(load_from_disk(await data_dir.download())["eval"])[:num_examples]
    base_label = "Base (4-bit)" if method == "qlora" else "Base"
    await flyte.report.replace.aio(wrap_report(f"<h2>Evaluation</h2><div class='card'><p>Scoring {base_label} on {len(examples)} held-out tickets…</p></div>"), do_flush=True)

    model, tokenizer = _load(model_name, method)
    model.eval()
    base = await asyncio.to_thread(score_model, model, tokenizer, examples)
    log.info(f"{base_label}: intent {base['intent_accuracy']}%, format {base['format_rate']}%")
    result = {"model": model_name, "method": method, "num_examples": len(examples), "base": base}
    ft = None
    train_metrics = None

    if trained_dir is not None:
        await flyte.report.replace.aio(
            wrap_report(f"<h2>Evaluation</h2><div class='card'><p>{base_label}: intent {base['intent_accuracy']}%. Loading the fine-tuned model…</p></div>"), do_flush=True
        )
        path = await trained_dir.download()
        if method == "full":
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, tokenizer = _load(model_name, method, weights_dir=path)
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, path)
        model.eval()
        ft = await asyncio.to_thread(score_model, model, tokenizer, examples)
        log.info(f"Fine-tuned: intent {ft['intent_accuracy']}%, format {ft['format_rate']}%")
        result["finetuned"] = ft
        result["improvement"] = round(ft["intent_accuracy"] - base["intent_accuracy"], 1)
        mp = os.path.join(path, "train_metrics.json")
        if os.path.exists(mp):
            with open(mp) as f:
                train_metrics = json.load(f)
            train_metrics.pop("loss_curve", None)
            result["train"] = train_metrics
        if JUDGE and judge_pairs > 0:
            await flyte.report.replace.aio(wrap_report(f"<h2>Evaluation</h2><div class='card'><p>Judging {judge_pairs} reply pairs with {JUDGE}…</p></div>"), do_flush=True)
            k = min(judge_pairs, len(examples))
            result["judge"] = await _judge_pairs(examples[:k], [r["reply"] for r in base["per_example"][:k]], [r["reply"] for r in ft["per_example"][:k]])
            log.info(f"judge: {result['judge']}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Report --
    cats = sorted({ex["category"] for ex in examples})

    def by_cat(scored: dict) -> list[float]:
        out = []
        for c in cats:
            idx = [i for i, ex in enumerate(examples) if ex["category"] == c]
            out.append(round(sum(scored["per_example"][i]["intent_ok"] for i in idx) / max(len(idx), 1) * 100, 1))
        return out

    series = {base_label: [base["intent_accuracy"], base["format_rate"]]}
    colors = ["#adb5bd"]
    cat_series = {base_label: by_cat(base)}
    if ft:
        series["Fine-tuned"] = [ft["intent_accuracy"], ft["format_rate"]]
        colors.append("#0f3460")
        cat_series["Fine-tuned"] = by_cat(ft)
    stats = (
        "<div class='stat-grid'>"
        f"<div class='stat'><div class='value'>{base['intent_accuracy']}%</div><div class='label'>{base_label} intent</div></div>"
    )
    if ft:
        imp = result["improvement"]
        badge = "badge-success" if imp > 0 else "badge-danger" if imp < 0 else "badge-info"
        stats += (
            f"<div class='stat'><div class='value'>{ft['intent_accuracy']}%</div><div class='label'>Fine-tuned intent</div></div>"
            f"<div class='stat'><div class='value'><span class='badge {badge}'>{imp:+.1f}pp</span></div><div class='label'>Improvement</div></div>"
            f"<div class='stat'><div class='value'>{base['format_rate']}% &rarr; {ft['format_rate']}%</div><div class='label'>Format kept</div></div>"
        )
        if "judge" in result:
            j = result["judge"]
            stats += f"<div class='stat'><div class='value'>{j['finetuned_win_rate']}%</div><div class='label'>Judge: fine-tuned wins ({j['finetuned_wins']}/{j['base_wins']}/{j['ties']} W/L/T)</div></div>"
    stats += f"<div class='stat'><div class='value'>{len(examples)}</div><div class='label'>Held-out tickets</div></div></div>"
    if train_metrics:
        stats += (
            "<div class='stat-grid'>"
            f"<div class='stat'><div class='value'>{_esc(train_metrics['gpu'])}</div><div class='label'>Trained on</div></div>"
            f"<div class='stat'><div class='value'>{train_metrics['peak_gb']} GiB</div><div class='label'>Peak VRAM</div></div>"
            f"<div class='stat'><div class='value'>{train_metrics['train_seconds'] // 60} min</div><div class='label'>Train time</div></div>"
            f"<div class='stat'><div class='value'>{train_metrics['artifact_mb']} MB</div><div class='label'>Artifact</div></div></div>"
        )
    rows = ""
    for i, ex in enumerate(examples[:8]):
        b = base["per_example"][i]
        cells = (
            f"<tr><th>Gold</th><td><code>{_esc(ex['intent'])}</code></td><td>{_esc(ex['response'][:220])}</td><td></td></tr>"
            f"<tr><th>{base_label}</th><td><code>{_esc(b['intent'] or '?')}</code></td><td>{_esc((b['reply'] or b['output'])[:220])}</td>"
            f"<td><span class='badge {'badge-success' if b['intent_ok'] else 'badge-danger'}'>{'intent ok' if b['intent_ok'] else 'wrong'}</span></td></tr>"
        )
        if ft:
            f_ = ft["per_example"][i]
            cells += (
                f"<tr><th>Fine-tuned</th><td><code>{_esc(f_['intent'] or '?')}</code></td><td>{_esc((f_['reply'] or f_['output'])[:220])}</td>"
                f"<td><span class='badge {'badge-success' if f_['intent_ok'] else 'badge-danger'}'>{'intent ok' if f_['intent_ok'] else 'wrong'}</span></td></tr>"
            )
        rows += f"<div class='card'><p><b>Ticket:</b> {_esc(ex['instruction'])}</p><table>{cells}</table></div>"

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation results</h2><h3>{_esc(model_name)} {_badge(method)}</h3>" + stats
            + "<div class='chart-container'>" + make_bar_chart(["Intent exact match", "Format kept"], series, title="Held-out tickets (%)", colors=colors, y_max_cap=100.0) + "</div>"
            + "<div class='chart-container'>" + make_bar_chart(cats, cat_series, title="Intent accuracy by category (%)", colors=colors, y_max_cap=100.0, width=900) + "</div>"
            + "<h3>What the models wrote</h3>" + rows
            + "<div class='note'><b>Intent</b> is exact match against the gold label. <b>Format</b> is valid JSON with both keys and every "
            "{{placeholder}} from the ticket kept in the reply. The judge, when on, compares replies pairwise and blind, with A and B "
            "swapped on a second call; a verdict that flips with the swap is a tie. Greedy decoding, thinking off.</div>"
        ),
        do_flush=True,
    )
    for key in ("base", "finetuned"):
        if key in result:
            result[key] = {k: v for k, v in result[key].items() if k != "per_example"}
    if "judge" in result:
        result["judge"].pop("verdicts", None)
    return json.dumps(result)


# ------------------------------------------------------------------
# Union artifacts, with an open-source fallback
# ------------------------------------------------------------------

_REGISTRY: str | None = None


async def registry() -> str:
    """`artifacts` on a platform with an artifact registry (Union), `outputs` everywhere else.

    Decided once per process, by asking. A devbox or open-source Flyte accepts `artifacts.new(...)`
    without complaint and then has nothing to find, so one lookup is the only reliable test. In
    `outputs` mode the trained weights are passed around as task outputs: the same files, no
    registry, no card, no lineage tab.
    """
    global _REGISTRY
    if _REGISTRY is None:
        ctx = flyte.ctx()
        if not ARTIFACTS or ctx is None or ctx.mode == "local":
            _REGISTRY = "outputs"
        else:
            from flyte.remote import Artifact

            try:
                async for _ in Artifact.listall.aio(name=ARTIFACT_NAME):
                    break
                _REGISTRY = "artifacts"
            except Exception as e:  # noqa: BLE001
                if "not found" not in str(e).lower() and "unimplemented" not in str(e).lower():
                    raise
                _REGISTRY = "outputs"
    return _REGISTRY


@cpu_env.task(produces_artifacts=True)
async def publish(trained: flyte.io.Dir, result_json: str) -> tuple[flyte.io.Dir, str]:
    """Register the trained weights as a new version of the `{MODEL_ID}-support` model artifact, with the numbers on its card.

    A task output wrapped with `artifacts.new` is how a run registers a version: Union stamps the producing
    action on it, so the artifact's lineage leads back to this pipeline and the data it trained on.
    """
    import dataclasses

    import flyte.artifacts as artifacts

    result = json.loads(result_json)
    version = f"v{int(time.time())}"
    numbers = {k: v for k, v in result.items() if k not in ("model",)}
    card = artifacts.Card.create_from(
        content=f"# {ARTIFACT_NAME} {version}\n\nTrained from `{result.get('model', MODEL)}`.\n\n```json\n{json.dumps(numbers, indent=2)}\n```",
        format="md",
        card_type="model",
    )
    meta = artifacts.Metadata.create_model_metadata(
        name=ARTIFACT_NAME, version=version, description=ARTIFACT_DESCRIPTION, framework="transformers",
        architecture="qwen3_5", task=ARTIFACT_TASK, serial_format="safetensors", card=card,
    )
    meta = dataclasses.replace(meta, attrs={**(meta.attrs or {}), **{k: str(v) for k, v in ARTIFACT_ATTRS(result).items()}})
    return artifacts.new(trained, meta), version


# ------------------------------------------------------------------
# Entrypoints
# ------------------------------------------------------------------


@cpu_env.task(report=True)
async def pipeline(
    model_name: str = MODEL,
    method: str = "lora",
    gpu: str = "",
    epochs: int = 1,
    lr: float = 0.0,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 768,
    train_samples: int = 800,
    eval_samples: int = 200,
    num_eval_examples: int = 100,
    judge_pairs: int = 40,
    lora_r: int = 32,
    lora_alpha: int = 64,
) -> flyte.io.Dir:
    """Data, train with one method, score base vs fine-tuned. Returns the adapter or the full weights."""
    if method not in METHODS:
        raise ValueError(f"--method must be one of {METHODS}, got {method!r}")
    resources = SHAPES[gpu] if gpu else None
    override = (lambda t: t.override(resources=resources)) if resources else (lambda t: t)
    steps = ["Prepare data", "Train", "Evaluate"]
    header = f"<h2>Support-ticket fine-tune</h2><h3>{_esc(model_name)} {_badge(method)}</h3>"

    await flyte.report.replace.aio(wrap_report(header + pipeline_step_indicator(0, steps)), do_flush=True)
    data_dir = await prepare_data(DATASET, train_samples, eval_samples)
    await flyte.report.replace.aio(wrap_report(header + pipeline_step_indicator(1, steps) + "<div class='card'><p>Training. Open the <b>train</b> task for the live loss curve.</p></div>"), do_flush=True)
    trained = await override(train)(model_name, data_dir, method, epochs, lr, batch_size, grad_accum, max_length, lora_r, lora_alpha)
    await flyte.report.replace.aio(wrap_report(header + pipeline_step_indicator(2, steps) + "<div class='card'><p>Scoring base vs fine-tuned on held-out tickets…</p></div>"), do_flush=True)
    result_json = await override(evaluate)(model_name, trained, data_dir, method, num_eval_examples, judge_pairs)
    result = json.loads(result_json)
    version = ""
    if await registry() == "artifacts":
        trained, version = await publish(trained, result_json)
        log.info(f"published {ARTIFACT_NAME} {version}")

    base, ft, imp = result["base"], result["finetuned"], result["improvement"]
    badge = "badge-success" if imp > 0 else "badge-danger" if imp < 0 else "badge-info"
    tm = result.get("train", {})
    judge = result.get("judge")
    await flyte.report.replace.aio(
        wrap_report(
            header + pipeline_step_indicator(3, steps) + "<div class='stat-grid'>"
            f"<div class='stat'><div class='value'>{base['intent_accuracy']}%</div><div class='label'>Base intent</div></div>"
            f"<div class='stat'><div class='value'>{ft['intent_accuracy']}%</div><div class='label'>Fine-tuned intent</div></div>"
            f"<div class='stat'><div class='value'><span class='badge {badge}'>{imp:+.1f}pp</span></div><div class='label'>Improvement</div></div>"
            f"<div class='stat'><div class='value'>{base['format_rate']}% &rarr; {ft['format_rate']}%</div><div class='label'>Format kept</div></div>"
            + (f"<div class='stat'><div class='value'>{judge['finetuned_win_rate']}%</div><div class='label'>Judge win rate</div></div>" if judge else "")
            + f"<div class='stat'><div class='value'>{tm.get('peak_gb', '?')} GiB</div><div class='label'>Peak VRAM</div></div>"
            f"<div class='stat'><div class='value'>{tm.get('train_seconds', 0) // 60} min</div><div class='label'>Train time</div></div></div>"
            "<div class='note'>Open <b>train</b> for the loss curve and <b>evaluate</b> for per-category bars and side-by-side replies. "
            "The output is the adapter (or the full weights): point <code>serve.py</code> at this run to deploy it."
            + (f" Published as artifact <b>{ARTIFACT_NAME}</b> version <b>{version}</b>: its card carries these numbers, and <code>serve.py --artifact</code> binds to it by name." if version else "")
            + "</div>"
        ),
        do_flush=True,
    )
    log.info(f"{method}: intent {base['intent_accuracy']}% -> {ft['intent_accuracy']}% ({imp:+.1f}pp)")
    return trained


@cpu_env.task
async def baseline(model_name: str = MODEL, gpu: str = "", train_samples: int = 800, eval_samples: int = 200, num_eval_examples: int = 100) -> str:
    """Score the untouched model on the same held-out split `pipeline` uses. Run this first."""
    data_dir = await prepare_data(DATASET, train_samples, eval_samples)
    task = evaluate.override(resources=SHAPES[gpu]) if gpu else evaluate
    result = await task(model_name, None, data_dir, "lora", num_eval_examples, 0)
    log.info(result)
    return result


@cpu_env.task(report=True)
async def compare(
    model_name: str = MODEL,
    methods: list[str] = ["lora", "qlora", "full"],
    epochs: int = 1,
    train_samples: int = 800,
    eval_samples: int = 200,
    num_eval_examples: int = 100,
    judge_pairs: int = 40,
) -> str:
    """All methods on the same data, in parallel (one GPU each), scored the same way, one report."""
    data_dir = await prepare_data(DATASET, train_samples, eval_samples)

    async def one(method: str) -> dict:
        with flyte.group(method):
            trained = await train(model_name, data_dir, method, epochs)
            return json.loads(await evaluate(model_name, trained, data_dir, method, num_eval_examples, judge_pairs))

    results = await asyncio.gather(*(one(m) for m in methods))
    labels = [r["method"] for r in results]
    base_intent = results[0]["base"]["intent_accuracy"]
    body = (
        f"<h2>Three methods, one measurement</h2><h3>{_esc(model_name)} on {results[0]['num_examples']} held-out tickets</h3>"
        "<table><tr><th>Method</th><th>Base intent</th><th>Fine-tuned intent</th><th>Format kept</th><th>Judge wins</th><th>Trainable</th><th>Peak VRAM</th><th>Train time</th><th>Artifact</th></tr>"
        + "".join(
            f"<tr><td>{_badge(r['method'])}</td><td>{r['base']['intent_accuracy']}%</td><td><b>{r['finetuned']['intent_accuracy']}%</b></td>"
            f"<td>{r['finetuned']['format_rate']}%</td><td>{r['judge']['finetuned_win_rate'] if 'judge' in r else '—'}{'%' if 'judge' in r else ''}</td>"
            f"<td>{r['train']['trainable_params'] / r['train']['total_params'] * 100:.2f}%</td><td>{r['train']['peak_gb']} GiB</td>"
            f"<td>{r['train']['train_seconds'] // 60} min</td><td>{r['train']['artifact_mb']} MB</td></tr>"
            for r in results
        )
        + "</table><div class='chart-container'>"
        + make_bar_chart(labels, {"base": [r["base"]["intent_accuracy"] for r in results], "fine-tuned": [r["finetuned"]["intent_accuracy"] for r in results]}, title="Intent exact match (%)", colors=["#adb5bd", "#0f3460"], y_max_cap=100.0)
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"peak VRAM GiB": [r["train"]["peak_gb"] for r in results]}, title="Peak VRAM while training (GiB)", colors=["#ef476f"])
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"minutes": [round(r["train"]["train_seconds"] / 60, 1) for r in results]}, title="Train time (minutes)", colors=["#06d6a0"])
        + f"</div><div class='note'>Same {train_samples} training rows, same {num_eval_examples} held-out rows, same scorer. The base scores "
        f"{base_intent}% before any of them. What differs is what you pay in memory, time and bytes for the lift.</div>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return json.dumps({r["method"]: {k: v for k, v in r.items() if k != "model"} for r in results})
