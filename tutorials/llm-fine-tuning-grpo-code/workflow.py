"""
GRPO Fine-Tuning — Teach a model to write correct Python functions.

GRPO (Group Relative Policy Optimization) generates multiple completions per
prompt, scores them with a reward function, and reinforces the best ones.

Here the reward is simple: does the generated code actually pass the test cases?
Multiple valid implementations exist for each problem, so GRPO can explore
different solutions — unlike SFT which teaches a single "correct" answer.

This is the same technique DeepSeek used for R1, applied to code generation.
Generated code runs in a sandboxed environment (union.sandbox) for safe execution.

Before training, a learnability filter samples the base model over a candidate
pool and keeps only the problems it solves *sometimes but not always* — the zone
where GRPO's within-group advantage is non-zero. Combined with a binary
(all-or-nothing) reward, this stops the model from collapsing onto degenerate
constants like `return True` that hack partial credit on impossible problems.
The filter task is cached, so it runs once and later runs reuse the filtered set.

The base model is the load-bearing choice. GRPO's gradient comes only from groups
where *some* completions pass and some fail, so a base that almost never passes
produces all-zero groups and no gradient. Qwen2.5-Coder-7B is the default because
it is the smallest model here that yields a real held-out gain on MBPP; smaller
models run to completion and produce charts without actually learning.

Usage:
    # Standard run — the documented default (large-memory GPU, ~2-3 hours)
    flyte run workflow.py pipeline

    # Quick sanity check — verifies the pipeline end to end, NOT a learning signal
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-Coder-0.5B" \\
        --max_candidate_samples 20 --max_train_samples 6 \\
        --epochs 1 --num_generations 2 --batch_size 2 --num_eval_examples 4

    # Larger groups — more mixed pass/fail groups, proportionally more generation time
    flyte run workflow.py pipeline --batch_size 16 --num_generations 16 \\
        --max_candidate_samples 800

    # Full fine-tuning (no LoRA)
    flyte run workflow.py pipeline --method full --model_name "Qwen/Qwen2.5-Coder-0.5B"
"""

import asyncio
import html
import json
import logging
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


MBPP_DATASET = "google-research-datasets/mbpp"

# For instruct models, a bare "def foo():" completion prompt makes them ramble
# (explanations, example usage, prose) which breaks the sandbox. A chat prompt
# with an explicit "code only" instruction gives a fair baseline — the base
# model produces clean code, so the GRPO delta reflects skill, not just format.
CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a problem description and a "
    "function signature, respond with ONLY the complete Python function that "
    "solves it — the signature and its body. Do not include any explanation, "
    "prose, example usage, test code, or text outside the function."
)


def build_code_prompt(tokenizer, raw_prompt: str, use_chat_template: bool = True) -> str:
    """Wrap the raw (problem + signature) prompt in the chat template with a
    code-only instruction, when the tokenizer supports it. Falls back to the raw
    completion-style prompt (correct for base models with no chat template)."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return raw_prompt


def assemble_code(func_def: str, completion: str, setup: str = "") -> str:
    """Turn a model completion into a runnable script.

    Handles both styles: a completion that is just the function body (prepend the
    signature) and one that already includes `def name(...)` (use as-is). Strips
    markdown code fences that instruct models often add.
    """
    m = re.search(r"def\s+(\w+)", func_def)
    fname = m.group(1) if m else None
    completion = re.sub(r"^\s*```(?:python)?\n?", "", completion)
    completion = re.sub(r"\n?```\s*$", "", completion)
    if fname and re.search(rf"\bdef\s+{re.escape(fname)}\b", completion):
        code = completion
    else:
        code = func_def + "\n" + completion
    if setup:
        code = setup + "\n" + code
    return code


async def run_tests_sandboxed(
    sbx, code: str, test_list: list[str],
) -> tuple[bool, int, int]:
    """Run test cases against generated code inside a sandbox.

    Uses union.sandbox to execute untrusted LLM-generated code in an isolated
    environment with no network access. Each test assertion is checked
    individually so we can report partial credit (passed/total).

    Args:
        sbx: An open sandbox session.
        code: The complete generated code (full function definition).
        test_list: List of assert strings to run against the code.
    """
    total = len(test_list)

    # Build a script that runs each test and prints PASS/FAIL per line
    test_script = code + "\n\n"
    for i, test in enumerate(test_list):
        test_script += (
            f"try:\n"
            f"    {test}\n"
            f"    print('PASS:{i}')\n"
            f"except Exception:\n"
            f"    print('FAIL:{i}')\n"
        )

    proc = await sbx.run(
        test_script,
        script_type="python",
        stdout=True,
        stderr=True,
        network_mode="blocked",
        timeout_s=5.0,
    )
    out, err = await proc.communicate_text()

    if not out or proc.returncode != 0:
        log.debug(f"[Sandbox] returncode={proc.returncode} stderr={err[:200] if err else 'None'}")

    passed = out.count("PASS:") if out else 0
    return passed == total, passed, total


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    max_candidate_samples: int = 250,
    max_eval_samples: int = 50,
) -> flyte.io.Dir:
    """Load MBPP coding problems and prepare a candidate pool + eval split.

    MBPP columns: text (problem description), code (solution), test_list (assert strings).
    We build a prompt that includes the problem description and the function signature
    extracted from the reference solution, then let the model complete the body.

    The "train" split here is a *candidate pool* — `filter_learnable` samples the
    base model over it and keeps only the problems in the learnable zone (solved
    sometimes but not always). GRPO only learns where the generation group has
    reward variance, so we start with more candidates than we'll ultimately train on.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    log.info(f"Loading MBPP dataset...")

    # MBPP has train (374), test (500), validation (90), prompt (10)
    mbpp = load_dataset(MBPP_DATASET, "full")

    # Combine train + validation + test for a larger pool, then re-split
    all_rows = []
    for split in ["train", "validation", "test"]:
        for row in mbpp[split]:
            # Extract function name from first test assertion
            # e.g. "assert min_cost(...)" -> "min_cost"
            first_test = row["test_list"][0]
            match = re.search(r"assert\s+(\w+)\s*\(", first_test)
            if not match:
                continue
            func_name = match.group(1)

            # Extract the function signature from the reference solution
            func_sig = None
            for line in row["code"].split("\n"):
                if line.strip().startswith(f"def {func_name}"):
                    func_sig = line.rstrip()
                    break

            if not func_sig:
                continue

            # Build prompt: problem description + function signature for the model to complete
            prompt_text = f"{row['text']}\n\n{func_sig}"

            # Join test_list + challenge_test_list into a single newline-separated string
            tests = "\n".join(row["test_list"])

            # Setup code (some problems need imports like `import math`)
            setup = row.get("test_setup_code", "").strip()

            all_rows.append({
                "prompt": prompt_text,
                "func_prompt": prompt_text,  # duplicate — "prompt" is reserved by GRPOTrainer
                "tests": tests,
                "setup_code": setup,
                "name": func_name,
            })

    log.info(f"Loaded {len(all_rows)} valid MBPP problems")

    # Shuffle and split
    import random
    rng = random.Random(42)
    rng.shuffle(all_rows)

    n_train = min(max_candidate_samples, len(all_rows) - max_eval_samples)
    n_eval = min(max_eval_samples, len(all_rows) - n_train)

    processed = DatasetDict({
        "train": Dataset.from_list(all_rows[:n_train]),  # candidate pool (pre-filter)
        "eval": Dataset.from_list(all_rows[n_train:n_train + n_eval]),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {n_train} candidates, {n_eval} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task: Download the base model (cached)
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def download_model(model_name: str) -> flyte.io.Dir:
    """Download the base model weights once and cache them, so `filter_learnable`,
    `train`, and `evaluate` don't each re-fetch from HuggingFace every run.
    Marginal for a 0.5B; a real win (and a dodge of HF rate-limits) for a 7B."""
    from huggingface_hub import snapshot_download

    local = os.path.join(tempfile.mkdtemp(), "model")
    snapshot_download(
        repo_id=model_name,
        local_dir=local,
        token=HF_TOKEN,
        ignore_patterns=["*.pth", "*.onnx", "*.gguf", "*.msgpack", "*.h5"],
    )
    log.info(f"Downloaded {model_name} -> {local}")
    return await flyte.io.Dir.from_local(local)


# ------------------------------------------------------------------
# Task 2: Filter for learnable problems (RLVR data curation)
# ------------------------------------------------------------------

@gpu_env.task(cache="auto", report=True)
async def filter_learnable(
    model_name: str,
    data_dir: flyte.io.Dir,
    max_train_samples: int = 120,
    filter_samples: int = 4,
    temperature: float = 0.8,
    max_completion_length: int = 128,
) -> flyte.io.Dir:
    """Keep only the *learnable* training problems for GRPO.

    GRPO computes advantages *within* a group of completions, so it only learns
    where the group has reward variance. Two kinds of problems give zero signal:

      - Impossible (base model never solves it) — every completion scores 0.
        These are also exactly where partial-credit reward hacking takes over:
        `return True` / `return -1` grab a fraction of the asserts and become the
        highest-advantage completion in an otherwise all-zero group.
      - Trivial (base model always solves it) — every completion scores 1.

    This pre-pass samples the BASE model `filter_samples` times on each candidate
    and keeps only problems it solves *sometimes but not always*
    (1 <= all_pass_count < filter_samples) — the learnable middle. Cached, so the
    first workshop run pays for it once and later runs reuse the filtered set.
    """
    import torch
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from union import sandbox as sb

    log.info(f"Filtering candidates for learnability with {model_name} (N={filter_samples})")

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Filtering for Learnable Problems...</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="card"><p>Sampling the base model {filter_samples}× per '
            f"candidate — keeping only problems it solves <i>sometimes</i>.</p></div>"
        ),
        do_flush=True,
    )

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    candidates = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, dtype=dtype, device_map="auto",
    )
    model.eval()

    def sample_completions(prompt: str) -> list[str]:
        """Draw `filter_samples` sampled completions for one prompt."""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=filter_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = outputs[:, inputs.input_ids.shape[1]:]
        return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    kept: list[dict] = []
    n_impossible = 0
    n_trivial = 0

    async with sb.on_device.session(network_mode="blocked", backend="bubblewrap") as sbx:
        for idx in range(len(candidates)):
            if len(kept) >= max_train_samples:
                break

            row = candidates[idx]
            func_def = row["func_prompt"].strip().split("\n")[-1]
            setup = row["setup_code"].strip() if row["setup_code"] else ""
            test_list = [
                l.strip() for l in row["tests"].strip().split("\n")
                if l.strip().startswith("assert")
            ]

            completions = sample_completions(row["prompt"])

            all_pass_count = 0
            for completion in completions:
                if not completion.strip():
                    continue
                full_code = func_def + "\n" + completion
                if setup:
                    full_code = setup + "\n" + full_code
                all_passed, _, _ = await run_tests_sandboxed(sbx, full_code, test_list)
                if all_passed:
                    all_pass_count += 1

            if all_pass_count == 0:
                n_impossible += 1
            elif all_pass_count == filter_samples:
                n_trivial += 1
            else:
                kept.append(row)

            if (idx + 1) % 10 == 0 or len(kept) >= max_train_samples:
                log.info(
                    f"[Filter] scanned {idx + 1}/{len(candidates)}: "
                    f"kept={len(kept)} impossible={n_impossible} trivial={n_trivial}"
                )
                await flyte.report.replace.aio(
                    wrap_report(
                        f"<h2>Filtering for Learnable Problems...</h2>"
                        f"<h3>{model_name}</h3>"
                        f'<div class="stat-grid">'
                        f'  <div class="stat"><div class="value">{len(kept)}</div><div class="label">Learnable (kept)</div></div>'
                        f'  <div class="stat"><div class="value">{n_impossible}</div><div class="label">Impossible (all fail)</div></div>'
                        f'  <div class="stat"><div class="value">{n_trivial}</div><div class="label">Trivial (all pass)</div></div>'
                        f'  <div class="stat"><div class="value">{idx + 1}/{len(candidates)}</div><div class="label">Scanned</div></div>'
                        f'</div>'
                        f'<div class="card"><p>Target: {max_train_samples} learnable problems. '
                        f"Sampling base model {filter_samples}× each.</p></div>"
                    ),
                    do_flush=True,
                )

    log.info(
        f"Filter complete: kept {len(kept)} learnable "
        f"(dropped {n_impossible} impossible, {n_trivial} trivial)"
    )

    if not kept:
        raise RuntimeError(
            "No learnable problems found — every candidate was impossible or trivial. "
            "Try a stronger model, more candidates, or a higher filter_samples."
        )

    filtered = DatasetDict({
        "train": Dataset.from_list(kept),
        "eval": dataset["eval"],  # pass the held-out eval split through unchanged
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "filtered")
    filtered.save_to_disk(output_dir)

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Learnability Filter Complete</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{len(kept)}</div><div class="label">Learnable (kept)</div></div>'
            f'  <div class="stat"><div class="value">{n_impossible}</div><div class="label">Impossible</div></div>'
            f'  <div class="stat"><div class="value">{n_trivial}</div><div class="label">Trivial</div></div>'
            f'</div>'
            f'<div class="card"><p>GRPO now trains only on problems with reward '
            f"variance — the learnable zone where advantages are non-zero.</p></div>"
        ),
        do_flush=True,
    )

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 3: Train with GRPO
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    model_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    method: str = "lora",
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.04,
    use_chat_template: bool = True,
    precision: str = "auto",
) -> flyte.io.Dir:
    """Fine-tune a model with GRPO — reward = do the tests pass?

    Args:
        method: "lora" (default), "full" for full fine-tuning, or "qlora" —
            4-bit quantized base + LoRA, which lets a 7B fit on a 16GB T4
            (generation is slow, but memory fits). qlora saves the adapter
            (the 4-bit base can't be merged on a small GPU).
        beta: KL penalty coefficient. Anchors the policy to the base model so it
            can't drift far and overfit the small training set (train reward ↑
            but held-out eval ↓). 0 disables the KL term; 0.04 is a safe default.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from union import sandbox as sb

    is_qlora = method == "qlora"

    log.info(f"GRPO Code Training: model={model_name}, method={method}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # On unified-memory parts (GB10/Grace-Blackwell) the GPU allocates from the
        # same DRAM the OS lives in, so an over-large batch has nothing to OOM into:
        # the driver spins on NV_ERR_NO_MEMORY, the kernel reclaims from everything
        # else, and the whole box locks up instead of raising OutOfMemoryError. Cap
        # the process so a bad config fails loudly and the host survives.
        torch.cuda.set_per_process_memory_fraction(0.85)

    method_badge = (
        '<span class="badge badge-info">GRPO + LoRA</span>' if method == "lora"
        else '<span class="badge badge-danger">GRPO + Full Fine-Tune</span>'
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Loading Model...</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="card">'
            f"<p><b>Method:</b> {method_badge}</p>"
            f"<p>Setting up code generation training...</p>"
            f"</div>"
        ),
        do_flush=True,
    )

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Load model + tokenizer (from the cached download) --
    model_path = await model_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Precision selection. "auto": bf16 where supported, else fp16 (T4/Turing has
    # no bf16), else fp32. Override with precision="fp32"/"bf16"/"fp16" — fp32 is
    # the escape hatch when low-precision generation produces nan/inf logits and
    # crashes GRPO sampling ("probability tensor contains inf/nan"), which we've
    # seen on larger models (3B+) on new (Blackwell) GPUs even with eager
    # attention. fp32 is ~2x memory and slower (generation is the GRPO bottleneck)
    # but numerically bulletproof.
    if precision == "auto":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
    else:
        use_bf16 = precision == "bf16"
        use_fp16 = precision == "fp16"
    train_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    )
    log.info(f"Training precision: {train_dtype} (requested={precision}, qlora={is_qlora})")

    quant_config = None
    if is_qlora:
        # 4-bit NF4 — a 7B base drops from ~14GB to ~4GB so it fits on a T4.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=train_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=train_dtype,
        device_map="auto",
        quantization_config=quant_config,
        # Use eager attention: SDPA's fused kernels can emit nan logits during
        # left-padded GRPO generation (fully-masked rows, and a head_dim-128 path
        # that's unstable on new Blackwell GPUs), which crashes sampling with
        # "probability tensor contains inf/nan". Hit on 1.5B (head_dim 128) but not
        # 0.5B (head_dim 64). Eager is slower but numerically safe.
        attn_implementation="eager",
    )

    # -- LoRA / QLoRA (optional) --
    peft_config = None
    if method in ("lora", "qlora"):
        from peft import LoraConfig, prepare_model_for_kbit_training

        if is_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            # Gradient checkpointing on a PEFT model: the frozen base produces
            # activations with requires_grad=False, so the checkpointed segment has
            # nothing to backprop through and autograd errors with "element 0 of
            # tensors does not require grad". Forcing the input embeddings to emit
            # grad-requiring activations is the standard fix (prepare_model_for_kbit_training
            # already does this on the qlora path).
            model.enable_input_require_grads()

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # -- Metrics tracking --
    training_log: list[dict] = []
    reward_log: list[dict] = []
    reward_stats = {"calls": 0, "total_reward": 0, "all_pass": 0, "total": 0}
    train_state = {"max_steps": 0}  # updated by callback, read by reward fn
    loop = asyncio.get_running_loop()

    def _build_training_report(max_steps: int) -> str:
        """Build the live training report HTML from current metrics."""
        stats_html = f"""
        <h2>GRPO Training in Progress...</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">GRPO + {method.upper()}</div><div class="label">Method</div></div>
          <div class="stat"><div class="value">{len(dataset['train']):,}</div><div class="label">Train Examples</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{num_generations}</div><div class="label">Generations</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
        </div>
        """

        charts_html = ""

        # Progress bar from trainer logs
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

            # Training loss chart
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

        # Reward charts from reward function
        if len(reward_log) >= 2:
            avg_reward = reward_stats["total_reward"] / max(reward_stats["total"], 1)
            pass_rate = reward_stats["all_pass"] / max(reward_stats["total"], 1) * 100

            charts_html += f"""
            <div class="stat-grid" style="margin-top:16px;">
              <div class="stat"><div class="value">{avg_reward:.3f}</div><div class="label">Avg Reward</div></div>
              <div class="stat"><div class="value">{pass_rate:.1f}%</div><div class="label">Pass Rate</div></div>
              <div class="stat"><div class="value">{reward_stats['total']}</div><div class="label">Samples Scored</div></div>
            </div>
            """

            reward_chart = make_line_chart(
                data=reward_log,
                x_key="batch",
                y_keys=["avg_reward", "batch_reward"],
                title="Reward (Tests Passed)",
                x_label="Reward Batch",
                y_label="Reward",
                colors=["#0f3460", "#5a7db5"],
                y_max_cap=1.05,
                y_display_names={"avg_reward": "Running Avg", "batch_reward": "Per Batch"},
                faint_keys={"batch_reward"},
            )
            charts_html += f'<div class="chart-container">{reward_chart}</div>'

            pass_chart = make_line_chart(
                data=reward_log,
                x_key="batch",
                y_keys=["pass_rate", "batch_pass_rate"],
                title="All Tests Pass Rate",
                x_label="Reward Batch",
                y_label="Pass %",
                colors=["#06d6a0", "#5a7db5"],
                y_max_cap=105.0,
                y_display_names={"pass_rate": "Running Avg", "batch_pass_rate": "Per Batch"},
                faint_keys={"batch_pass_rate"},
            )
            charts_html += f'<div class="chart-container">{pass_chart}</div>'

        return wrap_report(stats_html + charts_html)

    # -- Trainer callback for loss/progress updates --
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            train_state["max_steps"] = state.max_steps
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
            }
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 4)
            if "learning_rate" in logs:
                entry["lr"] = logs["learning_rate"]
            if "grad_norm" in logs:
                entry["grad_norm"] = round(float(logs["grad_norm"]), 4)
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f}"
                + (f" loss={entry['loss']:.4f}" if "loss" in entry else "")
            )

            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(
                    _build_training_report(state.max_steps),
                    do_flush=True,
                ),
                loop,
            )

    # -- Sandbox session for safe code execution --
    # Opens an isolated sandbox where LLM-generated code runs with no network
    # access. The reward function (sync, in trainer thread) uses
    # run_coroutine_threadsafe to call into the async sandbox.
    async with sb.on_device.session(network_mode="blocked", backend="bubblewrap") as sbx:

        # -- Reward function --
        def code_reward(completions: list[str], func_prompt: list[str], tests: list[str], setup_code: list[str], **kwargs) -> list[float]:
            """Reward = 1.0 only if ALL tests pass, else 0.0 (binary / all-or-nothing).

            Binary reward is deliberate. Partial credit (passed/total) is easy to
            hack on this task: a constant like `return True` or `return -1` grabs a
            fraction of the asserts and, in a group where the genuine attempts all
            score 0, becomes the highest-advantage completion — so GRPO reinforces
            degenerate constants. All-or-nothing removes that gradient. It only works
            because `filter_learnable` guarantees each problem's group still has
            variance (some completions fully pass, some don't).
            """
            rewards = []
            batch_passes = 0

            for completion, p, t, setup in zip(completions, func_prompt, tests, setup_code):
                func_def = p.strip().split("\n")[-1]
                full_code = assemble_code(func_def, completion, setup)
                test_list = [l.strip() for l in t.strip().split("\n") if l.strip().startswith("assert")]

                # Run tests in sandbox from trainer thread
                future = asyncio.run_coroutine_threadsafe(
                    run_tests_sandboxed(sbx, full_code, test_list),
                    loop,
                )
                try:
                    all_passed, passed, total = future.result(timeout=10)
                except Exception:
                    all_passed, passed, total = False, 0, len(test_list)

                reward = 1.0 if all_passed else 0.0
                rewards.append(reward)

                if all_passed:
                    reward_stats["all_pass"] += 1
                    batch_passes += 1
                reward_stats["total"] += 1

            reward_stats["calls"] += 1
            reward_stats["total_reward"] += sum(rewards)

            # Track metrics
            batch_avg = sum(rewards) / len(rewards)
            batch_pass_pct = batch_passes / len(completions) * 100
            running_avg = reward_stats["total_reward"] / reward_stats["total"]
            running_pass = reward_stats["all_pass"] / reward_stats["total"] * 100

            reward_log.append({
                "batch": reward_stats["calls"],
                "avg_reward": round(running_avg, 4),
                "pass_rate": round(running_pass, 2),
                "batch_reward": round(batch_avg, 4),
                "batch_pass_rate": round(batch_pass_pct, 2),
            })

            if reward_stats["calls"] % 5 == 0:
                log.info(
                    f"[Reward] batch {reward_stats['calls']}: "
                    f"avg_reward={running_avg:.3f}, pass_rate={running_pass:.1f}%, "
                    f"batch_reward={batch_avg:.3f}"
                )
                # Push report update from trainer thread
                asyncio.run_coroutine_threadsafe(
                    flyte.report.replace.aio(
                        _build_training_report(train_state["max_steps"]),
                        do_flush=True,
                    ),
                    loop,
                )

            return rewards

        # -- GRPO config --
        output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            beta=beta,  # KL anchor to the base model — prevents overfit-drift
            # Recompute activations instead of storing them. Costs ~25-30% backward
            # time; without it a 7B LoRA run holds every layer's activations for the
            # whole group and larger num_generations won't fit.
            gradient_checkpointing=method in ("lora", "qlora"),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            save_strategy="epoch",
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
        )

        # Prompt the model the way we'll evaluate it: for instruct models, wrap
        # the raw problem in the chat template with a "code only" instruction.
        # `func_prompt` stays raw so the reward can reconstruct the signature.
        train_dataset = dataset["train"].map(
            lambda ex: {"prompt": build_code_prompt(tokenizer, ex["func_prompt"], use_chat_template)}
        )
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            log.info("Using chat template for training prompts (code-only instruction).")

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_funcs=code_reward,
            peft_config=peft_config,
            processing_class=tokenizer,
            callbacks=[MetricsCallback()],
        )

        log.info(f"Starting GRPO training (num_generations={num_generations})...")
        await flyte.report.replace.aio(
            _build_training_report(trainer.state.max_steps or 0),
            do_flush=True,
        )

        await asyncio.to_thread(trainer.train)
        log.info("Training loop finished.")

    log.info("Saving model...")
    final_avg = reward_stats["total_reward"] / max(reward_stats["total"], 1)
    final_pass = reward_stats["all_pass"] / max(reward_stats["total"], 1) * 100
    log.info(f"GRPO training complete. Avg reward: {final_avg:.3f}, pass rate: {final_pass:.1f}%")

    # -- Save model --
    save_dir = os.path.join(tempfile.mkdtemp(), "grpo_model")
    if method == "lora":
        log.info("Merging LoRA weights...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_dir)
    elif is_qlora:
        # can't merge into a 4-bit base on a small GPU — save just the adapter;
        # eval reloads the 4-bit base + this adapter.
        log.info("Saving QLoRA adapter...")
        trainer.model.save_pretrained(save_dir)
    else:
        log.info("Saving full model...")
        trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info("Model saved.")

    # -- Final report --
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

    if len(reward_log) >= 2:
        reward_chart = make_line_chart(
            data=reward_log,
            x_key="batch",
            y_keys=["avg_reward", "batch_reward"],
            title="Reward (Tests Passed)",
            x_label="Reward Batch",
            y_label="Reward",
            colors=["#0f3460", "#5a7db5"],
            y_max_cap=1.05,
            y_display_names={"avg_reward": "Running Avg", "batch_reward": "Per Batch"},
            faint_keys={"batch_reward"},
        )
        final_charts += f'<div class="chart-container">{reward_chart}</div>'

        pass_chart = make_line_chart(
            data=reward_log,
            x_key="batch",
            y_keys=["pass_rate", "batch_pass_rate"],
            title="All Tests Pass Rate",
            x_label="Reward Batch",
            y_label="Pass %",
            colors=["#06d6a0", "#5a7db5"],
            y_max_cap=105.0,
            y_display_names={"pass_rate": "Running Avg", "batch_pass_rate": "Per Batch"},
            faint_keys={"batch_pass_rate"},
        )
        final_charts += f'<div class="chart-container">{pass_chart}</div>'

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Training Complete</h2>"
            f"<h3>{model_name}</h3>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{final_avg:.3f}</div><div class="label">Avg Reward</div></div>'
            f'  <div class="stat"><div class="value">{final_pass:.1f}%</div><div class="label">Pass Rate</div></div>'
            f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
            f'  <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>'
            f'</div>'
            f"{final_charts}"
        ),
        do_flush=True,
    )

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 4: Evaluate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    model_dir: flyte.io.Dir,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 30,
    use_chat_template: bool = True,
    method: str = "lora",
    eval_k: int = 4,
    eval_temperature: float = 0.8,
) -> str:
    """Compare base vs GRPO-trained model on code generation.

    Scored as mean pass@1 over `eval_k` sampled completions per problem, not a
    single greedy draw. This matters: GRPO raises P(correct) — it turns a problem
    the model solves 1-in-4 times into one it solves 3-in-4. A single greedy
    sample collapses that probability back into one pass/fail bit and throws most
    of the improvement away, which is why a real training gain can show up as a
    flat eval. Averaging k samples estimates P(correct) directly, and cuts the
    noise floor on a 50-problem set from roughly +/-7pp to +/-3pp.

    Set eval_k=1 to recover the old greedy behaviour.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from union import sandbox as sb

    is_qlora = method == "qlora"

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Loading models...</p>"),
        do_flush=True,
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

    # Match training precision: T4 has no bf16, so use fp16 rather than fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    quant_config = None
    if is_qlora:
        # a 7B base won't fit in fp16 on a T4 for generation — load it 4-bit too
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )

    def generate_code(model, prompts, max_new_tokens=192):
        """Return `eval_k` completions per prompt (list-of-lists).

        Sampled at eval_temperature so the pass fraction estimates P(correct).
        eval_k=1 falls back to greedy, matching the original single-draw eval.
        """
        greedy = eval_k <= 1
        results = []
        for prompt in prompts:
            # Prompt exactly as in training: chat template + code-only instruction
            # for instruct models, raw completion otherwise.
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

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running base model...</p>"),
        do_flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="auto", quantization_config=quant_config,
    )
    base_results = generate_code(base_model, prompts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GRPO model --
    log.info("Loading GRPO-trained model...")
    await flyte.report.replace.aio(
        wrap_report("<h2>Evaluation</h2><p>Running GRPO-trained model...</p>"),
        do_flush=True,
    )

    ft_path = await finetuned_dir.download()
    if is_qlora:
        # qlora saved just the adapter — reload the 4-bit base and attach it
        from peft import PeftModel
        ft_base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto", quantization_config=quant_config,
        )
        ft_model = PeftModel.from_pretrained(ft_base, ft_path)
    else:
        ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    ft_results = generate_code(ft_model, prompts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score (sandboxed) --
    # Every one of the eval_k samples per problem is scored. `*_solved` counts how
    # many of the k passed, so its mean over problems is pass@1 = P(correct).
    base_solved: list[int] = []
    ft_solved: list[int] = []
    base_total_tests = 0
    base_passed_tests = 0
    ft_total_tests = 0
    ft_passed_tests = 0
    comparisons = []

    async def score_samples(sbx, samples, func_def, setup, test_list):
        """Run every sample for one problem.

        Returns (n_all_pass, asserts_passed, asserts_total, shown, shown_ok) where
        `shown` is the sample to display in the report. We pick the first sample
        that actually passed, falling back to the first non-empty one. Showing
        samples[0] unconditionally would be misleading: on a problem scored "3/4
        samples" the first draw can be one of the failures, so the report would
        print broken code next to a passing badge.
        """
        n_all_pass = 0
        passed_tests = 0
        total_tests = 0
        shown = ""
        shown_ok = False
        for sample in samples:
            if not sample.strip():
                total_tests += len(test_list)
                continue
            code = assemble_code(func_def, sample, setup)
            all_pass, p, t = await run_tests_sandboxed(sbx, code, test_list)
            if all_pass:
                n_all_pass += 1
            passed_tests += p
            total_tests += t
            if (all_pass and not shown_ok) or not shown:
                shown, shown_ok = sample, all_pass
        return n_all_pass, passed_tests, total_tests, shown, shown_ok

    async with sb.on_device.session(network_mode="blocked", backend="bubblewrap") as sbx:
        for i in range(len(prompts)):
            func_def = prompts[i].strip().split("\n")[-1]
            setup = setup_codes[i] if setup_codes[i] else ""
            test_list = [l.strip() for l in tests_list[i].strip().split("\n") if l.strip().startswith("assert")]

            base_n, base_p, base_t, base_show, base_show_ok = await score_samples(
                sbx, base_results[i], func_def, setup, test_list)
            ft_n, ft_p, ft_t, ft_show, ft_show_ok = await score_samples(
                sbx, ft_results[i], func_def, setup, test_list)

            base_solved.append(base_n)
            ft_solved.append(ft_n)
            base_passed_tests += base_p
            base_total_tests += base_t
            ft_passed_tests += ft_p
            ft_total_tests += ft_t

            k = max(len(base_results[i]), 1)
            comparisons.append({
                "name": names[i],
                "prompt": prompts[i][:200],
                # The displayed snippet is a *passing* sample where one exists, so
                # the code shown matches the badge beside it.
                "base_code": base_show[:300],
                "grpo_code": ft_show[:300],
                "base_code_ok": base_show_ok,
                "grpo_code_ok": ft_show_ok,
                "base_passed": f"{base_n}/{k} samples",
                "grpo_passed": f"{ft_n}/{k} samples",
                "base_all_pass": base_n == k,
                "grpo_all_pass": ft_n == k,
                "base_solve_frac": round(base_n / k, 2),
                "grpo_solve_frac": round(ft_n / k, 2),
            })

    total = len(prompts)
    k = max(eval_k, 1)
    # pass@1 = mean over problems of (samples passed / k). The headline metric.
    base_rate = sum(base_solved) / (total * k) * 100
    ft_rate = sum(ft_solved) / (total * k) * 100

    # How the movable mass shifted: GRPO can only act on problems the base solves
    # *sometimes* — never-solved and always-solved problems have no gradient.
    movable = sum(1 for c in base_solved if 0 < c < k)
    base_never = sum(1 for c in base_solved if c == 0)
    ft_never = sum(1 for c in ft_solved if c == 0)

    log.info(f"Base model: {base_rate:.1f}% pass@1 (k={k}, {total} problems)")
    log.info(f"GRPO model: {ft_rate:.1f}% pass@1 (k={k}, {total} problems)")
    log.info(f"Movable band (base solves sometimes): {movable}/{total}; "
             f"never-solved {base_never} -> {ft_never}")

    # -- Report --
    improvement = ft_rate - base_rate
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    bar_chart = make_bar_chart(
        labels=[f"pass@1 (k={k})", "Individual Tests"],
        series={
            "Base": [base_rate, base_passed_tests / max(base_total_tests, 1) * 100],
            "GRPO": [ft_rate, ft_passed_tests / max(ft_total_tests, 1) * 100],
        },
        title="Base vs GRPO — Code Generation",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=105.0,
    )

    # Show a REPRESENTATIVE mix, not the first 15 problems in dataset order — that
    # ordering is arbitrary and can easily contain no successes at all. Bucketing
    # guarantees the report leads with problems GRPO actually fixed, while still
    # showing what both models already solved and what neither could do.
    gains = sorted(  # GRPO solves it more often than base
        [c for c in comparisons if c["grpo_solve_frac"] > c["base_solve_frac"]],
        key=lambda c: c["grpo_solve_frac"] - c["base_solve_frac"], reverse=True)
    both_ok = [c for c in comparisons  # both solve it, at least sometimes
               if c["grpo_solve_frac"] == c["base_solve_frac"] and c["base_solve_frac"] > 0]
    both_bad = [c for c in comparisons  # neither ever solves it
                if c["grpo_solve_frac"] == 0 and c["base_solve_frac"] == 0]
    regress = sorted(  # base solved it more often than GRPO
        [c for c in comparisons if c["grpo_solve_frac"] < c["base_solve_frac"]],
        key=lambda c: c["grpo_solve_frac"] - c["base_solve_frac"])
    log.info(f"Buckets — gained:{len(gains)} both_ok:{len(both_ok)} "
             f"both_bad:{len(both_bad)} regressed:{len(regress)}")

    buckets = [
        ("GRPO improved it (passes more often after training)", "badge-success", gains[:5]),
        ("Both solved it", "badge-info", both_ok[:3]),
        ("Neither solved it", "badge-danger", both_bad[:3]),
        ("GRPO regressed (base passed more often)", "badge-danger", regress[:2]),
    ]
    examples_html = ""
    for label, badge, items in buckets:
        if not items:
            continue
        examples_html += f'<h4><span class="badge {badge}">{label}</span> ({len(items)} shown)</h4>'
        for c in items:
            base_badge = "badge-success" if c["base_solve_frac"] > 0 else "badge-danger"
            ft_badge = "badge-success" if c["grpo_solve_frac"] > 0 else "badge-danger"
            # The snippet is a passing sample when one exists; say so, otherwise a
            # reader can't tell whether the code shown is the code that worked.
            base_note = "a passing sample" if c["base_code_ok"] else "a failing sample"
            ft_note = "a passing sample" if c["grpo_code_ok"] else "a failing sample"
            # escape: generated Python contains <, > and & (comparisons, generics)
            examples_html += f"""
<div class="card">
<p><b>{html.escape(str(c['name']))}</b> —
  Base: <span class="badge {base_badge}">{c['base_passed']}</span> |
  GRPO: <span class="badge {ft_badge}">{c['grpo_passed']}</span></p>
<p><b>Base</b> <i>({base_note})</i>:<pre style="white-space:pre-wrap;background:#f5f5f5;padding:8px;font-size:0.85em;border-radius:4px;">{html.escape(c['base_code'])}</pre></p>
<p><b>GRPO</b> <i>({ft_note})</i>:<pre style="white-space:pre-wrap;background:#f0fff0;padding:8px;font-size:0.85em;border-radius:4px;">{html.escape(c['grpo_code'])}</pre></p>
</div>"""

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Evaluation Results — Code Generation</h2>"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{base_rate:.1f}%</div><div class="label">Base pass@1</div></div>'
            f'  <div class="stat"><div class="value">{ft_rate:.1f}%</div><div class="label">GRPO pass@1</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'  <div class="stat"><div class="value">{total}&times;{k}</div><div class="label">Problems &times; Samples</div></div>'
            f'</div>'
            f'<div class="chart-container">{bar_chart}</div>'
            f'<div class="card"><p>Scored as <b>mean pass@1 over {k} samples</b> per problem '
            f"(temperature {eval_temperature}) — GRPO raises <i>P(correct)</i>, which a single "
            f"greedy draw would round away to a pass/fail bit.</p>"
            f"<p>GRPO can only move problems the base model solves <i>sometimes</i>: "
            f"<b>{movable}/{total}</b> were in that band. Never-solved went "
            f"<b>{base_never} &rarr; {ft_never}</b>.</p></div>"
            f"<table>"
            f"<tr><th></th><th>pass@1 (k={k})</th><th>Individual Tests</th></tr>"
            f"<tr><td><b>Base model</b></td><td>{base_rate:.1f}% ({sum(base_solved)}/{total * k})</td><td>{base_passed_tests}/{base_total_tests}</td></tr>"
            f"<tr><td><b>GRPO-trained</b></td><td>{ft_rate:.1f}% ({sum(ft_solved)}/{total * k})</td><td>{ft_passed_tests}/{ft_total_tests}</td></tr>"
            f"</table>"
            f"<h3>Example problems (base vs GRPO)</h3>"
            f"{examples_html}"
        ),
        do_flush=True,
    )

    return json.dumps({
        "base_pass_rate": round(base_rate, 1),   # mean pass@1 over k samples
        "grpo_pass_rate": round(ft_rate, 1),
        "improvement": round(ft_rate - base_rate, 1),
        "eval_k": k,
        "base_tests": f"{base_passed_tests}/{base_total_tests}",
        "grpo_tests": f"{ft_passed_tests}/{ft_total_tests}",
        "num_problems": total,
        "movable_band": movable,          # problems the base solved sometimes-but-not-always
        "base_never_solved": base_never,
        "grpo_never_solved": ft_never,
        "buckets": {"gained": len(gains), "both_ok": len(both_ok),
                    "both_bad": len(both_bad), "regressed": len(regress)},
        "comparisons": comparisons[:15],
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    # Defaults are the smallest configuration that produces a real held-out gain on
    # MBPP. A 7B *code-pretrained base* is the load-bearing choice: GRPO's gradient
    # comes only from groups with mixed pass/fail, and sub-2B models almost never
    # pass, so their groups are all-zero and teach nothing. See the README's
    # "Choosing a Base Model". use_chat_template is off to match the base model —
    # Qwen ships a ChatML template in its base tokenizer, and applying it makes the
    # model ramble instead of completing the MBPP stub.
    model_name: str = "Qwen/Qwen2.5-Coder-7B",
    method: str = "lora",
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 192,
    max_train_samples: int = 100,
    max_candidate_samples: int = 300,
    filter_samples: int = 4,
    max_eval_samples: int = 50,
    num_eval_examples: int = 50,
    eval_k: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.04,
    use_chat_template: bool = False,
    use_filter: bool = False,
    precision: str = "auto",
) -> str:
    """
    GRPO fine-tuning pipeline — teach a model to write correct Python.

    Args:
        method: "lora" for LoRA adapters (default), "full" for full fine-tuning.

    1. Prepare a candidate pool of MBPP problems with test cases
    2. Filter for learnability — keep problems the base model solves *sometimes*
    3. Train with GRPO — binary reward = do ALL tests pass?
    4. Evaluate: pass rate before/after
    """
    log.info(f"Pipeline: {model_name} | GRPO code generation (filter={use_filter})")
    steps = (
        ["Prepare Data", "Filter", "GRPO Train", "Evaluate"] if use_filter
        else ["Prepare Data", "GRPO Train", "Evaluate"]
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Code Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(0, steps)}"
            f'<div class="card"><p>Preparing coding problems...</p></div>'
        ),
        do_flush=True,
    )

    data_dir = await prepare_data(max_candidate_samples, max_eval_samples)
    model_dir = await download_model(model_name)  # cached base-model weights

    # The learnability filter concentrates training on problems the base solves
    # *sometimes*. With use_filter=False we train on the whole candidate pool —
    # more data, but many zero-gradient (all-pass/all-fail) groups. Binary reward
    # keeps this safe (impossible problems just contribute no gradient, no hacking).
    if use_filter:
        await flyte.report.replace.aio(
            wrap_report(
                f"<h2>GRPO Code Pipeline</h2>"
                f"<h3>{model_name}</h3>"
                f"{pipeline_step_indicator(1, steps)}"
                f'<div class="card"><p>Filtering candidates for learnability...</p></div>'
            ),
            do_flush=True,
        )
        train_dir = await filter_learnable(
            model_name, data_dir, max_train_samples, filter_samples,
            max_completion_length=max_completion_length,
        )
    else:
        log.info("Skipping learnability filter — training on the full candidate pool.")
        train_dir = data_dir

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Code Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(len(steps) - 2, steps)}"
            f'<div class="card"><p>GRPO training — binary reward = all tests pass...</p></div>'
        ),
        do_flush=True,
    )

    finetuned_dir = await train(
        model_name, model_dir, train_dir, method, epochs, lr, batch_size,
        num_generations, max_completion_length, lora_r, lora_alpha, beta,
        use_chat_template, precision,
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Code Pipeline</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(len(steps) - 1, steps)}"
            f'<div class="card"><p>Evaluating base vs GRPO model...</p></div>'
        ),
        do_flush=True,
    )

    result = await evaluate(
        model_name, model_dir, finetuned_dir, train_dir, num_eval_examples,
        use_chat_template, method, eval_k,
    )
    metrics = json.loads(result)

    improvement = metrics["improvement"]
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>GRPO Code Pipeline Complete</h2>"
            f"<h3>{model_name}</h3>"
            f"{pipeline_step_indicator(4, steps)}"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{metrics["base_pass_rate"]}%</div><div class="label">Base Pass Rate</div></div>'
            f'  <div class="stat"><div class="value">{metrics["grpo_pass_rate"]}%</div><div class="label">GRPO Pass Rate</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f'</div>'
        ),
        do_flush=True,
    )

    log.info(f"Pipeline complete. Improvement: {improvement:+.1f}pp")
    return result
