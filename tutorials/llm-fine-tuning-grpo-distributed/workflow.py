"""
Level 1 — Distributed GRPO: fan out verification onto a warm pool.

The single-GPU tutorial (../llm-fine-tuning-grpo-code) runs generation,
verification, and the gradient step in one task. Its reward function is a serial
loop over one in-process sandbox: at batch_size=8 x num_generations=8 that is 64
sequential sandbox executions per step, each up to 5s. Verification, not the
gradient, is what you wait for.

This file changes exactly one thing: the reward function shards its completions
and dispatches them to a reusable verifier pool (verify.py), then reassembles the
rewards in order. TRL's GRPOTrainer, the binary reward, the KL anchor, and every
hyperparameter behave identically — so any speedup is attributable to the fan-out
and nothing else.

Level 2 (distributed.py) goes further and moves generation off the learner too.

Usage:
    # Standard run — 14B on one L40s
    flyte run workflow.py pipeline

    # Quick wiring check — NOT a learning signal (a 0.5B produces dead groups)
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-Coder-0.5B" \\
        --max_candidate_samples 20 --epochs 1 --num_generations 2 \\
        --batch_size 2 --num_eval_examples 4

    # Bigger groups — more mixed pass/fail groups, and more load on the pool
    flyte run workflow.py pipeline --batch_size 16 --num_generations 16 --shard_size 8
"""

import asyncio
import html
import json
import logging
import os
import tempfile
import time

import flyte
import flyte.io
import flyte.report

from common import (
    build_code_prompt,
    download_model,
    extract_test_list,
    func_def_from_prompt,
    prepare_data,
)
from config import cpu_env, learner_env
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report
from verify import VerifyItem, chunk, verify_shard

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _shard_timeout(shard_size: int, per_test_timeout: float, n_shards: int, capacity: int) -> float:
    """Wall-clock budget for one fanned-out verification round.

    A shard is `shard_size` completions run sequentially, so its own cost is
    `shard_size x per_test_timeout` — not `per_test_timeout`. Getting this wrong is
    the most likely way to turn a slow verifier into a mysterious training crash.

    On top of that, if there are more shards than pool slots they queue, so budget
    for `ceil(n_shards / capacity)` waves plus generous slack for cold starts.
    """
    waves = max(1, -(-n_shards // max(1, capacity)))
    return shard_size * per_test_timeout * waves + 120.0


# ------------------------------------------------------------------
# Train — TRL GRPOTrainer, reward fanned out to the verifier pool
# ------------------------------------------------------------------

@learner_env.task(report=True)
async def train(
    model_name: str,
    model_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 192,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.04,
    use_chat_template: bool = False,
    shard_size: int = 4,
    verify_timeout_s: float = 5.0,
    pool_capacity: int = 160,
) -> flyte.io.Dir:
    """GRPO fine-tune with LoRA; reward = do ALL tests pass, scored off-box.

    Args:
        shard_size: completions per verifier task call. Smaller = more parallelism
            and more dispatch overhead; larger = fewer, longer calls. With 64
            completions per step, shard_size=4 gives 16 concurrent shards.
        pool_capacity: `replicas x concurrency` of verify_env, used only to size
            the timeout budget. Keep it in sync with config.py.
    """
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GRPO (Level 1, distributed verification): model={model_name}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")

    await flyte.report.replace.aio(
        wrap_report(f"<h2>Loading Model...</h2><h3>{model_name}</h3>"), do_flush=True
    )

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    model_path = await model_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    train_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    log.info(f"Training precision: {train_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=train_dtype,
        device_map="auto",
        # SDPA's fused kernels can emit nan logits during left-padded GRPO
        # generation, which crashes sampling with "probability tensor contains
        # inf/nan". Eager is slower but numerically safe.
        attn_implementation="eager",
    )
    # Gradient checkpointing on a PEFT model: the frozen base produces activations
    # with requires_grad=False, so the checkpointed segment has nothing to backprop
    # through. Forcing input embeddings to emit grad-requiring activations is the fix.
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -- Metrics --
    training_log: list[dict] = []
    reward_log: list[dict] = []
    stats = {"calls": 0, "total_reward": 0.0, "all_pass": 0, "total": 0,
             "verify_seconds": 0.0, "workers": set()}
    train_state = {"max_steps": 0}
    loop = asyncio.get_running_loop()

    def _build_training_report(max_steps: int) -> str:
        avg = stats["total_reward"] / stats["total"] if stats["total"] else 0.0
        pass_pct = stats["all_pass"] / stats["total"] * 100 if stats["total"] else 0.0
        html_out = f"""
        <h2>GRPO Training — Distributed Verification</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{num_generations}</div><div class="label">Generations</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{shard_size}</div><div class="label">Shard Size</div></div>
          <div class="stat"><div class="value">{len(stats['workers'])}</div><div class="label">Verifier Workers</div></div>
          <div class="stat"><div class="value">{pass_pct:.1f}%</div><div class="label">Pass Rate</div></div>
          <div class="stat"><div class="value">{avg:.3f}</div><div class="label">Avg Reward</div></div>
        </div>
        """
        if training_log:
            cur = training_log[-1]
            pct = cur["step"] / max_steps * 100 if max_steps else 0
            html_out += (
                f'<div class="card"><b>Step {cur["step"]}/{max_steps}</b> ({pct:.0f}%) — '
                f'loss {cur.get("loss", 0):.4f}</div>'
            )
        if len(reward_log) > 1:
            html_out += (
                '<div class="chart-container">'
                + make_line_chart(
                    reward_log,
                    x_key="batch",
                    y_keys=["avg_reward", "batch_reward"],
                    title="Reward (running vs per-batch)",
                    colors=["#0f3460", "#adb5bd"],
                )
                + "</div>"
            )
        # The distinct-worker count above is the load-bearing number for this
        # tutorial: it is direct evidence the reusable pool is engaging. One worker
        # per shard would mean every call cold-started and reuse bought nothing.
        html_out += (
            f'<div class="card"><p><b>Verification</b>: {stats["calls"]} fan-out rounds, '
            f'{stats["verify_seconds"]:.0f}s total, across '
            f'{len(stats["workers"])} distinct pool workers.</p></div>'
        )
        return wrap_report(html_out)

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                training_log.append({"step": state.global_step, **logs})
                train_state["max_steps"] = state.max_steps

    # Dispatching the shards has to happen *inside* a coroutine body, not while
    # building the argument to run_coroutine_threadsafe.
    #
    # `verify_shard(shard)` is a Flyte task invocation, not a plain `async def`.
    # Calling a plain async function off-loop is harmless — it just builds an inert
    # coroutine object. Calling a task template is not: it touches the Flyte run
    # context and the running loop at call time, and TRL's reward function runs in
    # a worker thread (trainer.train is wrapped in asyncio.to_thread). Building the
    # generator there fails with:
    #
    #     RuntimeError: There is no current event loop in thread 'asyncio_N'.
    #
    # Wrapping the calls in this coroutine means creating it is inert, and every
    # task invocation happens on the event-loop thread where the context lives.
    async def _verify_all(shards: list[list[VerifyItem]]) -> list[list]:
        return await asyncio.gather(*(verify_shard(s, verify_timeout_s) for s in shards))

    # -- Reward: the fan-out --
    def code_reward(completions, func_prompt, tests, setup_code, **kwargs) -> list[float]:
        """Binary reward, computed on the verifier pool instead of in-process.

        TRL calls this synchronously from the trainer thread, so we bridge to the
        task's event loop with run_coroutine_threadsafe — the same bridge the
        single-GPU tutorial uses. The difference is what gets scheduled: one
        `gather` over N shard tasks, instead of one sandbox call per completion.
        """
        items = [
            VerifyItem(
                func_def=func_def_from_prompt(p),
                completion=c,
                setup=s or "",
                tests=t,
            )
            for c, p, t, s in zip(completions, func_prompt, tests, setup_code)
        ]
        shards = chunk(items, shard_size)
        timeout = _shard_timeout(shard_size, verify_timeout_s, len(shards), pool_capacity)

        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(_verify_all(shards), loop)
        try:
            shard_results = future.result(timeout=timeout)
        except Exception as e:
            # Verification failing should not kill a multi-hour run. Zero rewards
            # make this step a no-op gradient (a flat group has no advantage), and
            # the run continues. A persistent failure shows as reward pinned at 0.
            log.warning(f"[Reward] fan-out failed ({type(e).__name__}: {e}); scoring batch as 0")
            return [0.0] * len(completions)
        elapsed = time.monotonic() - started

        # Flatten in shard order — chunk() preserves order, so this realigns with
        # `completions`. GRPO computes advantages *within* each prompt's group, so
        # a misalignment here silently trains on the wrong rewards rather than
        # raising. Do not reorder shards.
        results = [r for shard in shard_results for r in shard]
        rewards = [r.reward for r in results]

        stats["calls"] += 1
        stats["verify_seconds"] += elapsed
        stats["total"] += len(rewards)
        stats["total_reward"] += sum(rewards)
        stats["all_pass"] += sum(1 for r in rewards if r > 0)
        stats["workers"].update(r.worker_id for r in results)

        running_avg = stats["total_reward"] / stats["total"]
        reward_log.append({
            "batch": stats["calls"],
            "avg_reward": round(running_avg, 4),
            "batch_reward": round(sum(rewards) / len(rewards), 4),
            "pass_rate": round(stats["all_pass"] / stats["total"] * 100, 2),
        })

        if stats["calls"] % 5 == 0:
            log.info(
                f"[Reward] round {stats['calls']}: avg={running_avg:.3f} "
                f"batch={sum(rewards) / len(rewards):.3f} "
                f"verify={elapsed:.1f}s over {len(shards)} shards "
                f"({len(stats['workers'])} workers seen)"
            )
            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(_build_training_report(train_state["max_steps"]), do_flush=True),
                loop,
            )

        return rewards

    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    train_dataset = dataset["train"].map(
        lambda ex: {"prompt": build_code_prompt(tokenizer, ex["func_prompt"], use_chat_template)}
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=code_reward,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    log.info(f"Starting GRPO (num_generations={num_generations}, shard_size={shard_size})")
    await flyte.report.replace.aio(_build_training_report(trainer.state.max_steps or 0), do_flush=True)

    await asyncio.to_thread(trainer.train)
    log.info("Training loop finished.")

    save_dir = os.path.join(tempfile.mkdtemp(), "final")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    await flyte.report.replace.aio(_build_training_report(train_state["max_steps"]), do_flush=True)
    log.info(
        f"Done. {stats['calls']} verification rounds, {stats['verify_seconds']:.0f}s "
        f"in the pool, {len(stats['workers'])} distinct workers."
    )
    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Evaluate — generation on the learner, scoring on the pool
# ------------------------------------------------------------------

@learner_env.task(report=True)
async def evaluate(
    model_name: str,
    model_dir: flyte.io.Dir,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 50,
    use_chat_template: bool = False,
    eval_k: int = 4,
    eval_temperature: float = 0.8,
    shard_size: int = 8,
) -> str:
    """Compare base vs GRPO-trained model, scored as mean pass@1 over `eval_k` samples.

    Sampling k completions rather than one greedy draw matters: GRPO raises
    P(correct) — it turns a problem solved 1-in-4 times into one solved 3-in-4. A
    single greedy sample collapses that back into one pass/fail bit and throws most
    of the improvement away, which is how a real training gain shows up as a flat
    eval. Averaging k samples estimates P(correct) directly.

    Scoring fans out to the same verifier pool the trainer uses — 50 problems x 4
    samples x 2 models is 400 sandbox runs, which is exactly the workload the pool
    exists for.
    """
    import torch
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Loading models...</p>"), do_flush=True
    )

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    prompts = eval_ds["prompt"]
    tests_list = eval_ds["tests"]
    setup_codes = eval_ds["setup_code"]
    names = eval_ds["name"]

    model_path = await model_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32

    def generate_code(model, max_new_tokens=192):
        greedy = eval_k <= 1
        results = []
        for prompt in prompts:
            model_input = build_code_prompt(tokenizer, prompt, use_chat_template)
            inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    temperature=None if greedy else eval_temperature,
                    top_p=None if greedy else 0.95,
                    num_return_sequences=1 if greedy else eval_k,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = outputs[:, inputs.input_ids.shape[1]:]
            results.append([tokenizer.decode(g, skip_special_tokens=True) for g in gen])
        return results

    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running base model...</p>"), do_flush=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    base_results = generate_code(base_model)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running GRPO-trained model...</p>"), do_flush=True
    )
    ft_path = await finetuned_dir.download()
    ft_base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    ft_model = PeftModel.from_pretrained(ft_base, ft_path)
    ft_results = generate_code(ft_model)
    del ft_model, ft_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score everything on the pool in one fan-out --
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Scoring on the verifier pool...</p>"), do_flush=True
    )

    items: list[VerifyItem] = []
    index: list[tuple[str, int]] = []  # (which model, problem index), parallel to `items`
    for tag, results in (("base", base_results), ("ft", ft_results)):
        for i, samples in enumerate(results):
            func_def = func_def_from_prompt(prompts[i])
            for sample in samples:
                items.append(VerifyItem(func_def, sample, setup_codes[i] or "", tests_list[i]))
                index.append((tag, i))

    shards = chunk(items, shard_size)
    log.info(f"Scoring {len(items)} completions across {len(shards)} shards")
    shard_results = await asyncio.gather(*(verify_shard(s) for s in shards))
    flat = [r for shard in shard_results for r in shard]

    n = len(prompts)
    k = max(eval_k, 1)
    solved = {"base": [0] * n, "ft": [0] * n}
    tests_passed = {"base": 0, "ft": 0}
    tests_total = {"base": 0, "ft": 0}
    # A passing sample to display per (model, problem), so the snippet in the report
    # matches the badge beside it. Showing samples[0] unconditionally would print
    # broken code next to a "3/4 passed" badge.
    shown: dict[tuple[str, int], tuple[str, bool]] = {}

    for (tag, i), res, it in zip(index, flat, items):
        if res.reward > 0:
            solved[tag][i] += 1
        tests_passed[tag] += res.passed
        tests_total[tag] += res.total
        prev = shown.get((tag, i))
        if prev is None or (res.reward > 0 and not prev[1]):
            shown[(tag, i)] = (it.completion, res.reward > 0)

    base_rate = sum(solved["base"]) / (n * k) * 100
    ft_rate = sum(solved["ft"]) / (n * k) * 100
    improvement = ft_rate - base_rate

    # GRPO can only act on problems the base solves *sometimes* — never-solved and
    # always-solved problems have no within-group advantage and no gradient.
    movable = sum(1 for c in solved["base"] if 0 < c < k)
    base_never = sum(1 for c in solved["base"] if c == 0)
    ft_never = sum(1 for c in solved["ft"] if c == 0)

    log.info(f"Base: {base_rate:.1f}% pass@1 | GRPO: {ft_rate:.1f}% pass@1 (k={k}, n={n})")
    log.info(f"Movable band: {movable}/{n}; never-solved {base_never} -> {ft_never}")

    comparisons = []
    for i in range(n):
        comparisons.append({
            "name": names[i],
            "base_frac": round(solved["base"][i] / k, 2),
            "grpo_frac": round(solved["ft"][i] / k, 2),
            "base_passed": f"{solved['base'][i]}/{k} samples",
            "grpo_passed": f"{solved['ft'][i]}/{k} samples",
            "base_code": shown.get(("base", i), ("", False))[0][:300],
            "grpo_code": shown.get(("ft", i), ("", False))[0][:300],
            "base_code_ok": shown.get(("base", i), ("", False))[1],
            "grpo_code_ok": shown.get(("ft", i), ("", False))[1],
        })

    # Lead with problems GRPO actually fixed, rather than the first 15 in dataset
    # order — that ordering is arbitrary and can contain no successes at all.
    gains = sorted([c for c in comparisons if c["grpo_frac"] > c["base_frac"]],
                   key=lambda c: c["grpo_frac"] - c["base_frac"], reverse=True)
    both_ok = [c for c in comparisons if c["grpo_frac"] == c["base_frac"] and c["base_frac"] > 0]
    both_bad = [c for c in comparisons if c["grpo_frac"] == 0 and c["base_frac"] == 0]
    regress = sorted([c for c in comparisons if c["grpo_frac"] < c["base_frac"]],
                     key=lambda c: c["grpo_frac"] - c["base_frac"])

    examples_html = ""
    for label, badge, group in (
        ("GRPO improved it", "badge-success", gains[:5]),
        ("Both solved it", "badge-info", both_ok[:3]),
        ("Neither solved it", "badge-danger", both_bad[:3]),
        ("GRPO regressed", "badge-danger", regress[:2]),
    ):
        if not group:
            continue
        examples_html += f'<h4><span class="badge {badge}">{label}</span> ({len(group)} shown)</h4>'
        for c in group:
            bb = "badge-success" if c["base_frac"] > 0 else "badge-danger"
            fb = "badge-success" if c["grpo_frac"] > 0 else "badge-danger"
            bn = "a passing sample" if c["base_code_ok"] else "a failing sample"
            fn = "a passing sample" if c["grpo_code_ok"] else "a failing sample"
            # escape: generated Python contains <, > and & (comparisons, generics)
            examples_html += f"""
<div class="card">
<p><b>{html.escape(str(c['name']))}</b> —
  Base: <span class="badge {bb}">{c['base_passed']}</span> |
  GRPO: <span class="badge {fb}">{c['grpo_passed']}</span></p>
<p><b>Base</b> <i>({bn})</i>:<pre style="white-space:pre-wrap;background:#f5f5f5;padding:8px;font-size:0.85em;border-radius:4px;">{html.escape(c['base_code'])}</pre></p>
<p><b>GRPO</b> <i>({fn})</i>:<pre style="white-space:pre-wrap;background:#f0fff0;padding:8px;font-size:0.85em;border-radius:4px;">{html.escape(c['grpo_code'])}</pre></p>
</div>"""

    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"
    bar_chart = make_bar_chart(
        labels=[f"pass@1 (k={k})", "Individual Tests"],
        series={
            "Base": [base_rate, tests_passed["base"] / max(tests_total["base"], 1) * 100],
            "GRPO": [ft_rate, tests_passed["ft"] / max(tests_total["ft"], 1) * 100],
        },
        title="Base vs GRPO — Code Generation",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=105.0,
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation Results</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{base_rate:.1f}%</div><div class="label">Base pass@1</div></div>'
            f'  <div class="stat"><div class="value">{ft_rate:.1f}%</div><div class="label">GRPO pass@1</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'  <div class="stat"><div class="value">{n}&times;{k}</div><div class="label">Problems &times; Samples</div></div>'
            f"</div>"
            f'<div class="chart-container">{bar_chart}</div>'
            f'<div class="card"><p>{len(items)} completions scored across {len(shards)} '
            f"shards on the reusable verifier pool.</p>"
            f"<p>GRPO can only move problems the base solves <i>sometimes</i>: "
            f"<b>{movable}/{n}</b> were in that band. Never-solved went "
            f"<b>{base_never} &rarr; {ft_never}</b>.</p></div>"
            f"<h3>Example problems (base vs GRPO)</h3>{examples_html}"
        ),
        do_flush=True,
    )

    return json.dumps({
        "base_pass_rate": round(base_rate, 1),
        "grpo_pass_rate": round(ft_rate, 1),
        "improvement": round(improvement, 1),
        "eval_k": k,
        "num_problems": n,
        "movable_band": movable,
        "base_never_solved": base_never,
        "grpo_never_solved": ft_never,
        "completions_scored": len(items),
        "shards": len(shards),
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "Qwen/Qwen2.5-Coder-14B",
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 192,
    max_candidate_samples: int = 300,
    max_eval_samples: int = 50,
    num_eval_examples: int = 50,
    eval_k: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.04,
    use_chat_template: bool = False,
    shard_size: int = 4,
    verify_timeout_s: float = 5.0,
) -> str:
    """Level 1 pipeline: prepare data -> GRPO train (distributed verify) -> evaluate.

    The base model is the load-bearing choice, not the hyperparameters. GRPO's
    gradient comes only from groups where *some* completions pass and some fail, so
    a base that almost never passes produces all-zero groups and no gradient at all.
    Qwen2.5-Coder-14B is the default here because it lands in that middle band often
    enough on MBPP to produce a real held-out gain on an L40s.
    """
    steps = ["Prepare Data", "GRPO Train", "Evaluate"]
    log.info(f"Level 1 pipeline: {model_name}")

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Distributed GRPO — Level 1</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(0, steps)}"
            f'<div class="card"><p>Preparing coding problems...</p></div>'
        ),
        do_flush=True,
    )

    data_dir = await prepare_data(max_candidate_samples, max_eval_samples)
    model_dir = await download_model(model_name)

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Distributed GRPO — Level 1</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(1, steps)}"
            f'<div class="card"><p>GRPO training — verification fanned out to the pool...</p></div>'
        ),
        do_flush=True,
    )

    finetuned_dir = await train(
        model_name, model_dir, data_dir, epochs, lr, batch_size, num_generations,
        max_completion_length, lora_r, lora_alpha, beta, use_chat_template,
        shard_size, verify_timeout_s,
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Distributed GRPO — Level 1</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(2, steps)}"
            f'<div class="card"><p>Evaluating base vs GRPO...</p></div>'
        ),
        do_flush=True,
    )

    result = await evaluate(
        model_name, model_dir, finetuned_dir, data_dir, num_eval_examples,
        use_chat_template, eval_k,
    )
    metrics = json.loads(result)
    improvement = metrics["improvement"]
    badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Distributed GRPO — Level 1 Complete</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(3, steps)}"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{metrics["base_pass_rate"]}%</div><div class="label">Base pass@1</div></div>'
            f'  <div class="stat"><div class="value">{metrics["grpo_pass_rate"]}%</div><div class="label">GRPO pass@1</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f"</div>"
        ),
        do_flush=True,
    )

    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline)
    print(run.url)
