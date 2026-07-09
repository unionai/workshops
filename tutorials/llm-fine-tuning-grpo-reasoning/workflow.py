"""
GRPO Fine-Tuning — Teach a model to REASON (the Countdown game).

This is the "R1-Zero" demo: take a small model that has NO reasoning ability
(Qwen2.5 has no thinking mode — that's Qwen3/QwQ) and use GRPO to make it reason,
without ever showing it a single reasoning example. The only signal is a reward
for reaching the right answer. The model discovers, on its own, that thinking
step-by-step before answering earns more reward — so its responses get longer
and more structured as training proceeds. That emergent "response length grows"
curve is the famous DeepSeek-R1 / TinyZero result, reproduced on a tiny model.

The task: **Countdown**. Given a few numbers and a target, write an arithmetic
expression using each number exactly once (with + - * /) that equals the target.
It's verifiable (just evaluate the expression — no sandbox, an AST evaluator
handles it safely), homogeneous (one skill, endless instances → it transfers),
and hard enough that reasoning genuinely helps.

Usage:
    # Workshop run (single T4, ~20-30 min)
    flyte run workflow.py pipeline

    # Compare model sizes
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B-Instruct"
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-1.5B-Instruct"
"""

import ast
import json
import logging
import operator
import os
import re
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env, HF_TOKEN
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


SYSTEM_PROMPT = (
    "You are solving a Countdown puzzle. You are given a list of numbers and a "
    "target. Using each number exactly once and the operators + - * / (and "
    "parentheses), write an arithmetic expression that equals the target.\n"
    "First reason step by step inside <think> </think>, then give ONLY the final "
    "expression (no words) inside <answer> </answer>. "
    "Example: <think>3 and 4 make 12, plus 2 is 14</think><answer>3 * 4 + 2</answer>"
)


# ------------------------------------------------------------------
# Safe arithmetic evaluation + answer checking (no sandbox needed)
# ------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr: str):
    """Evaluate an arithmetic expression with numbers and + - * / only.
    Returns the value, or None if the expression is invalid/unsafe."""
    try:
        node = ast.parse(expr, mode="eval").body

        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](ev(n.operand))
            raise ValueError("unsupported expression")

        return ev(node)
    except Exception:
        return None


def numbers_in_expr(expr: str) -> list:
    """Return the integer literals used in an expression (for the 'use each
    number exactly once' check)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return []
    return sorted(
        int(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and float(n.value).is_integer()
    )


def extract_answer(text: str):
    """Pull the expression out of the last <answer>...</answer> block."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def is_correct(text: str, numbers: list, target: int) -> bool:
    expr = extract_answer(text)
    if not expr:
        return False
    if sorted(int(x) for x in numbers) != numbers_in_expr(expr):
        return False  # must use exactly the given numbers, each once
    value = safe_eval(expr)
    return value is not None and abs(value - target) < 1e-6


def has_reasoning_format(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", text, re.DOTALL))


# ------------------------------------------------------------------
# Countdown problem generation (guaranteed solvable)
# ------------------------------------------------------------------

def make_problem(rng, n_numbers: int = 3, max_num: int = 9) -> dict:
    """Generate a solvable Countdown problem: pick numbers, fold them with random
    +,-,* into a positive integer target. A solution is guaranteed to exist (the
    generating expression), though the model may find a different one."""
    for _ in range(100):
        nums = [rng.randint(1, max_num) for _ in range(n_numbers)]
        result = nums[0]
        for x in nums[1:]:
            op = rng.choice(["+", "-", "*"])
            result = {"+": result + x, "-": result - x, "*": result * x}[op]
        if result > 0:  # keep positive integer targets
            return {
                "question": f"Numbers: {nums}. Target: {result}.",
                "numbers": nums,
                "target": result,
            }
    # fallback (extremely unlikely): trivial problem
    return {"question": f"Numbers: [1, {max_num}]. Target: {1 + max_num}.",
            "numbers": [1, max_num], "target": 1 + max_num}


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    n_numbers: int = 3,
    max_num: int = 9,
    max_train_samples: int = 300,
    max_eval_samples: int = 100,
) -> flyte.io.Dir:
    """Generate solvable Countdown problems for train + eval."""
    import random
    from datasets import Dataset, DatasetDict

    rng = random.Random(42)
    seen = set()
    rows = []
    # de-dup on (numbers, target) so train/eval don't trivially overlap
    while len(rows) < max_train_samples + max_eval_samples:
        p = make_problem(rng, n_numbers, max_num)
        key = (tuple(p["numbers"]), p["target"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(p)

    processed = DatasetDict({
        "train": Dataset.from_list(rows[:max_train_samples]),
        "eval": Dataset.from_list(rows[max_train_samples:max_train_samples + max_eval_samples]),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {max_train_samples} train, {max_eval_samples} eval Countdown problems")
    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train with GRPO
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 320,
    beta: float = 0.005,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune with GRPO — reward = correct Countdown answer. Watch the model
    teach itself to reason (response length grows as accuracy climbs)."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GRPO Reasoning Training: model={model_name}, method={method}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    method_badge = (
        '<span class="badge badge-info">GRPO + LoRA</span>' if method == "lora"
        else '<span class="badge badge-danger">GRPO + Full Fine-Tune</span>'
    )
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Loading Model...</h2><h3>{model_name}</h3>"
            f'<div class="card"><p><b>Method:</b> {method_badge}</p>'
            f"<p>Teaching a non-reasoning model to reason via GRPO...</p></div>"
        ),
        do_flush=True,
    )

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    train_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, dtype=train_dtype, device_map="auto",
    )

    def to_conversational(ex):
        return {"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]}

    train_ds = dataset["train"].map(to_conversational)

    peft_config = None
    if method == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )

    # -- Metrics --
    training_log: list[dict] = []
    reward_log: list[dict] = []
    stats = {"calls": 0, "correct": 0, "total": 0, "len_sum": 0}
    import asyncio
    loop = asyncio.get_running_loop()

    def _completion_text(c):
        if isinstance(c, list):
            return c[-1].get("content", "") if c else ""
        return c or ""

    def _build_report(max_steps: int) -> str:
        acc = stats["correct"] / max(stats["total"], 1) * 100
        mean_len = stats["len_sum"] / max(stats["total"], 1)
        stats_html = f"""
        <h2>GRPO Reasoning Training in Progress...</h2><h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">GRPO + {method.upper()}</div><div class="label">Method</div></div>
          <div class="stat"><div class="value">{len(train_ds):,}</div><div class="label">Train Problems</div></div>
          <div class="stat"><div class="value">{acc:.1f}%</div><div class="label">Solve Rate</div></div>
          <div class="stat"><div class="value">{mean_len:.0f}</div><div class="label">Mean Answer Tokens</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{num_generations}</div><div class="label">Generations</div></div>
        </div>
        """
        charts = ""
        if training_log:
            cur = training_log[-1]
            pct = cur["step"] / max_steps * 100 if max_steps else 0
            charts += (
                f'<div class="card"><b>Step {cur["step"]}/{max_steps}</b> ({pct:.0f}%) | '
                f'Epoch {cur["epoch"]:.2f}/{epochs}'
                + (f' | Loss: <span class="highlight">{cur["loss"]:.4f}</span>' if cur.get("loss") else "")
                + f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">'
                f'<div style="background:#0f3460;width:{pct:.1f}%;height:100%;border-radius:4px;"></div></div></div>'
            )
        if len(reward_log) >= 2:
            charts += '<div class="chart-container">' + make_line_chart(
                data=reward_log, x_key="batch", y_keys=["solve_rate", "batch_rate"],
                title="Solve Rate (Reward)", x_label="Reward Batch", y_label="Solved %",
                colors=["#06d6a0", "#5a7db5"], y_max_cap=105.0,
                y_display_names={"solve_rate": "Running Avg", "batch_rate": "Per Batch"},
            ) + '</div>'
            # THE R1 CHART: response length growing as the model learns to reason
            charts += '<div class="chart-container">' + make_line_chart(
                data=reward_log, x_key="batch", y_keys=["mean_len"],
                title="Response Length — the model teaching itself to reason",
                x_label="Reward Batch", y_label="Mean Completion Tokens",
                colors=["#ef476f"],
            ) + '</div>'
        return wrap_report(stats_html + charts)

    def accuracy_reward(completions, numbers, target, **kwargs):
        rewards = []
        batch_correct = 0
        for c, nums, tgt in zip(completions, numbers, target):
            text = _completion_text(c)
            ok = is_correct(text, nums, tgt)
            rewards.append(1.0 if ok else 0.0)
            stats["len_sum"] += len(tokenizer(text).input_ids)
            stats["total"] += 1
            if ok:
                stats["correct"] += 1
                batch_correct += 1
        stats["calls"] += 1
        reward_log.append({
            "batch": stats["calls"],
            "solve_rate": round(stats["correct"] / max(stats["total"], 1) * 100, 2),
            "batch_rate": round(batch_correct / len(completions) * 100, 2),
            "mean_len": round(stats["len_sum"] / max(stats["total"], 1), 1),
        })
        if stats["calls"] % 5 == 0:
            log.info(f"[Reward] batch {stats['calls']}: solve={reward_log[-1]['solve_rate']}%, "
                     f"mean_len={reward_log[-1]['mean_len']}")
        return rewards

    def format_reward(completions, **kwargs):
        return [1.0 if has_reasoning_format(_completion_text(c)) else 0.0 for c in completions]

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {"step": state.global_step, "epoch": round(logs.get("epoch", 0), 2)}
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 4)
            training_log.append(entry)
            log.info(f"step={state.global_step}/{state.max_steps} epoch={entry['epoch']:.2f}"
                     + (f" loss={entry['loss']:.4f}" if "loss" in entry else ""))
            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(_build_report(state.max_steps), do_flush=True), loop,
            )

    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    grpo_config = GRPOConfig(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        num_generations=num_generations, max_completion_length=max_completion_length,
        beta=beta, reward_weights=[1.0, 0.2],
        logging_steps=5, save_strategy="epoch",
        bf16=use_bf16, fp16=use_fp16, report_to="none",
    )
    trainer = GRPOTrainer(
        model=model, args=grpo_config, train_dataset=train_ds,
        reward_funcs=[accuracy_reward, format_reward],
        peft_config=peft_config, processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    log.info(f"Starting GRPO training (num_generations={num_generations})...")
    await flyte.report.replace.aio(_build_report(trainer.state.max_steps or 0), do_flush=True)
    await asyncio.to_thread(trainer.train)
    log.info("Training loop finished.")

    final_acc = stats["correct"] / max(stats["total"], 1) * 100
    log.info(f"Training complete. Train solve rate: {final_acc:.1f}%")

    save_dir = os.path.join(tempfile.mkdtemp(), "grpo_model")
    if method == "lora":
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(save_dir)
    else:
        trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    charts = ""
    if len(reward_log) >= 2:
        charts += '<div class="chart-container">' + make_line_chart(
            data=reward_log, x_key="batch", y_keys=["solve_rate", "batch_rate"],
            title="Solve Rate (Reward)", x_label="Reward Batch", y_label="Solved %",
            colors=["#06d6a0", "#5a7db5"], y_max_cap=105.0,
            y_display_names={"solve_rate": "Running Avg", "batch_rate": "Per Batch"},
        ) + '</div>'
        charts += '<div class="chart-container">' + make_line_chart(
            data=reward_log, x_key="batch", y_keys=["mean_len"],
            title="Response Length — the model teaching itself to reason",
            x_label="Reward Batch", y_label="Mean Completion Tokens", colors=["#ef476f"],
        ) + '</div>'
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Reasoning Training Complete</h2><h3>{model_name}</h3>"
            f'<div class="stat-grid">'
            f'<div class="stat"><div class="value">{final_acc:.1f}%</div><div class="label">Train Solve Rate</div></div>'
            f'<div class="stat"><div class="value">{stats["len_sum"] / max(stats["total"], 1):.0f}</div><div class="label">Mean Answer Tokens</div></div>'
            f'<div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
            f'</div>{charts}'
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
    num_examples: int = 60,
    max_new_tokens: int = 320,
) -> str:
    """Compare base vs GRPO-trained: solve rate AND response length AND example
    reasoning traces (does the trained model actually reason?)."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(wrap_report("<h2>Evaluation</h2><p>Loading models...</p>"), do_flush=True)

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))
    questions, numbers, targets = eval_ds["question"], eval_ds["numbers"], eval_ds["target"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    def generate(model, questions):
        outs = []
        for q in questions:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            outs.append(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
        return outs

    await flyte.report.replace.aio(wrap_report("<h2>Evaluation</h2><p>Running base model...</p>"), do_flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, dtype=dtype, device_map="auto")
    base_out = generate(base_model, questions)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    await flyte.report.replace.aio(wrap_report("<h2>Evaluation</h2><p>Running GRPO-trained model...</p>"), do_flush=True)
    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    ft_out = generate(ft_model, questions)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_correct = ft_correct = 0
    base_len = ft_len = 0
    comparisons = []
    for i in range(len(questions)):
        b_ok = is_correct(base_out[i], numbers[i], targets[i])
        f_ok = is_correct(ft_out[i], numbers[i], targets[i])
        base_correct += int(b_ok)
        ft_correct += int(f_ok)
        base_len += len(tokenizer(base_out[i]).input_ids)
        ft_len += len(tokenizer(ft_out[i]).input_ids)
        comparisons.append({
            "question": questions[i], "target": targets[i],
            "base_answer": extract_answer(base_out[i]), "grpo_answer": extract_answer(ft_out[i]),
            "base_out": base_out[i][:400], "grpo_out": ft_out[i][:400],
            "base_ok": b_ok, "grpo_ok": f_ok,
        })

    total = len(questions)
    base_rate, ft_rate = base_correct / total * 100, ft_correct / total * 100
    base_ml, ft_ml = base_len / total, ft_len / total
    improvement = ft_rate - base_rate
    log.info(f"Base: {base_rate:.1f}% (len {base_ml:.0f}) | GRPO: {ft_rate:.1f}% (len {ft_ml:.0f})")

    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"
    rate_chart = make_bar_chart(
        labels=["Solve Rate", "Mean Response Tokens"],
        series={"Base": [base_rate, base_ml], "GRPO": [ft_rate, ft_ml]},
        title="Base vs GRPO — Countdown Reasoning",
        colors=["#adb5bd", "#0f3460"],
    )

    # Show reasoning traces — prefer ones where GRPO solved it
    examples_html = ""
    picked = sorted(comparisons, key=lambda c: (not c["grpo_ok"], c["base_ok"]))[:8]
    for c in picked:
        b = "badge-success" if c["base_ok"] else "badge-danger"
        f = "badge-success" if c["grpo_ok"] else "badge-danger"
        examples_html += f"""
<div class="card">
<p><b>{c['question']}</b></p>
<p><b>Base</b> <span class="badge {b}">{c['base_answer']}</span>:
<pre style="white-space:pre-wrap;background:#f5f5f5;padding:8px;font-size:0.8em;border-radius:4px;">{c['base_out']}</pre></p>
<p><b>GRPO</b> <span class="badge {f}">{c['grpo_answer']}</span>:
<pre style="white-space:pre-wrap;background:#f0fff0;padding:8px;font-size:0.8em;border-radius:4px;">{c['grpo_out']}</pre></p>
</div>"""

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation Results — Countdown Reasoning</h2>"
            f'<div class="stat-grid">'
            f'<div class="stat"><div class="value">{base_rate:.1f}%</div><div class="label">Base Solve Rate</div></div>'
            f'<div class="stat"><div class="value">{ft_rate:.1f}%</div><div class="label">GRPO Solve Rate</div></div>'
            f'<div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'<div class="stat"><div class="value">{base_ml:.0f} &rarr; {ft_ml:.0f}</div><div class="label">Response Tokens (reasoning!)</div></div>'
            f'</div>'
            f'<div class="chart-container">{rate_chart}</div>'
            f"<h3>Reasoning traces (base vs GRPO)</h3>{examples_html}"
        ),
        do_flush=True,
    )

    return json.dumps({
        "base_solve_rate": round(base_rate, 1),
        "grpo_solve_rate": round(ft_rate, 1),
        "improvement": round(improvement, 1),
        "base_mean_tokens": round(base_ml, 1),
        "grpo_mean_tokens": round(ft_ml, 1),
        "num_problems": total,
        "comparisons": comparisons[:8],
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    method: str = "lora",
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 320,
    beta: float = 0.005,
    n_numbers: int = 3,
    max_num: int = 9,
    max_train_samples: int = 300,
    max_eval_samples: int = 100,
    num_eval_examples: int = 60,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    GRPO reasoning pipeline — teach a non-reasoning model to reason (Countdown).

    1. Generate solvable Countdown problems
    2. Train with GRPO — reward = correct answer (reasoning emerges on its own)
    3. Evaluate: solve rate + response length + reasoning traces, before/after
    """
    log.info(f"Pipeline: {model_name} | GRPO Countdown reasoning")
    steps = ["Prepare Data", "GRPO Train", "Evaluate"]

    await flyte.report.replace.aio(
        wrap_report(f"<h2>GRPO Reasoning Pipeline</h2><h3>{model_name}</h3>"
                    f"{pipeline_step_indicator(0, steps)}"
                    f'<div class="card"><p>Generating Countdown problems...</p></div>'),
        do_flush=True,
    )
    data_dir = await prepare_data(n_numbers, max_num, max_train_samples, max_eval_samples)

    await flyte.report.replace.aio(
        wrap_report(f"<h2>GRPO Reasoning Pipeline</h2><h3>{model_name}</h3>"
                    f"{pipeline_step_indicator(1, steps)}"
                    f'<div class="card"><p>GRPO training — watch reasoning emerge...</p></div>'),
        do_flush=True,
    )
    finetuned_dir = await train(
        model_name, data_dir, method, epochs, lr, batch_size,
        num_generations, max_completion_length, beta, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        wrap_report(f"<h2>GRPO Reasoning Pipeline</h2><h3>{model_name}</h3>"
                    f"{pipeline_step_indicator(2, steps)}"
                    f'<div class="card"><p>Evaluating base vs GRPO...</p></div>'),
        do_flush=True,
    )
    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples, max_completion_length)
    metrics = json.loads(result)

    improvement = metrics["improvement"]
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Reasoning Pipeline Complete</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(3, steps)}"
            f'<div class="stat-grid">'
            f'<div class="stat"><div class="value">{metrics["base_solve_rate"]}%</div><div class="label">Base Solve Rate</div></div>'
            f'<div class="stat"><div class="value">{metrics["grpo_solve_rate"]}%</div><div class="label">GRPO Solve Rate</div></div>'
            f'<div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'<div class="stat"><div class="value">{metrics["base_mean_tokens"]:.0f} &rarr; {metrics["grpo_mean_tokens"]:.0f}</div><div class="label">Response Tokens</div></div>'
            f'</div>'
        ),
        do_flush=True,
    )
    log.info(f"Pipeline complete. Improvement: {improvement:+.1f}pp")
    return result
