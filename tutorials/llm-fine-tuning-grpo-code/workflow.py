"""
GRPO Fine-Tuning — Teach a model to write correct Python functions.

GRPO (Group Relative Policy Optimization) generates multiple completions per
prompt, scores them with a reward function, and reinforces the best ones.

Here the reward is simple: does the generated code actually pass the test cases?
Multiple valid implementations exist for each problem, so GRPO can explore
different solutions — unlike SFT which teaches a single "correct" answer.

This is the same technique DeepSeek used for R1, applied to code generation.

Usage:
    # Default (SmolLM2-135M)
    flyte run --local --tui workflow.py pipeline

    # Quick test
    flyte run --local --tui workflow.py pipeline --max_train_samples 30 --epochs 1

    # Remote
    flyte run workflow.py pipeline --epochs 3

    # Bigger model
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
"""

import json
import logging
import os
import re
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
# Coding problems — simple Python functions with test cases
# ------------------------------------------------------------------

PROBLEMS = [
    {
        "prompt": "Write a Python function called `double` that takes a number and returns it multiplied by 2.\n\ndef double(x):",
        "tests": "assert double(3) == 6\nassert double(0) == 0\nassert double(-5) == -10",
        "name": "double",
    },
    {
        "prompt": "Write a Python function called `is_even` that returns True if a number is even, False otherwise.\n\ndef is_even(n):",
        "tests": "assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True",
        "name": "is_even",
    },
    {
        "prompt": "Write a Python function called `add` that takes two numbers and returns their sum.\n\ndef add(a, b):",
        "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
        "name": "add",
    },
    {
        "prompt": "Write a Python function called `max_of_two` that returns the larger of two numbers.\n\ndef max_of_two(a, b):",
        "tests": "assert max_of_two(3, 5) == 5\nassert max_of_two(10, 2) == 10\nassert max_of_two(4, 4) == 4",
        "name": "max_of_two",
    },
    {
        "prompt": "Write a Python function called `absolute` that returns the absolute value of a number.\n\ndef absolute(x):",
        "tests": "assert absolute(-5) == 5\nassert absolute(3) == 3\nassert absolute(0) == 0",
        "name": "absolute",
    },
    {
        "prompt": "Write a Python function called `square` that returns the square of a number.\n\ndef square(x):",
        "tests": "assert square(3) == 9\nassert square(0) == 0\nassert square(-4) == 16",
        "name": "square",
    },
    {
        "prompt": "Write a Python function called `is_positive` that returns True if a number is greater than 0.\n\ndef is_positive(n):",
        "tests": "assert is_positive(5) == True\nassert is_positive(-3) == False\nassert is_positive(0) == False",
        "name": "is_positive",
    },
    {
        "prompt": "Write a Python function called `celsius_to_fahrenheit` that converts Celsius to Fahrenheit.\n\ndef celsius_to_fahrenheit(c):",
        "tests": "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\nassert celsius_to_fahrenheit(-40) == -40",
        "name": "celsius_to_fahrenheit",
    },
    {
        "prompt": "Write a Python function called `reverse_string` that reverses a string.\n\ndef reverse_string(s):",
        "tests": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
        "name": "reverse_string",
    },
    {
        "prompt": "Write a Python function called `string_length` that returns the length of a string.\n\ndef string_length(s):",
        "tests": "assert string_length('hello') == 5\nassert string_length('') == 0\nassert string_length('abc') == 3",
        "name": "string_length",
    },
    {
        "prompt": "Write a Python function called `first_element` that returns the first element of a list.\n\ndef first_element(lst):",
        "tests": "assert first_element([1, 2, 3]) == 1\nassert first_element(['a', 'b']) == 'a'\nassert first_element([42]) == 42",
        "name": "first_element",
    },
    {
        "prompt": "Write a Python function called `last_element` that returns the last element of a list.\n\ndef last_element(lst):",
        "tests": "assert last_element([1, 2, 3]) == 3\nassert last_element(['a', 'b']) == 'b'\nassert last_element([42]) == 42",
        "name": "last_element",
    },
    {
        "prompt": "Write a Python function called `list_sum` that returns the sum of all numbers in a list.\n\ndef list_sum(lst):",
        "tests": "assert list_sum([1, 2, 3]) == 6\nassert list_sum([]) == 0\nassert list_sum([-1, 1]) == 0",
        "name": "list_sum",
    },
    {
        "prompt": "Write a Python function called `count_items` that returns the number of items in a list.\n\ndef count_items(lst):",
        "tests": "assert count_items([1, 2, 3]) == 3\nassert count_items([]) == 0\nassert count_items(['a']) == 1",
        "name": "count_items",
    },
    {
        "prompt": "Write a Python function called `to_upper` that converts a string to uppercase.\n\ndef to_upper(s):",
        "tests": "assert to_upper('hello') == 'HELLO'\nassert to_upper('') == ''\nassert to_upper('ABC') == 'ABC'",
        "name": "to_upper",
    },
    {
        "prompt": "Write a Python function called `to_lower` that converts a string to lowercase.\n\ndef to_lower(s):",
        "tests": "assert to_lower('HELLO') == 'hello'\nassert to_lower('') == ''\nassert to_lower('abc') == 'abc'",
        "name": "to_lower",
    },
    {
        "prompt": "Write a Python function called `multiply` that takes two numbers and returns their product.\n\ndef multiply(a, b):",
        "tests": "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0\nassert multiply(-2, 3) == -6",
        "name": "multiply",
    },
    {
        "prompt": "Write a Python function called `is_empty` that returns True if a list is empty.\n\ndef is_empty(lst):",
        "tests": "assert is_empty([]) == True\nassert is_empty([1]) == False\nassert is_empty([1, 2, 3]) == False",
        "name": "is_empty",
    },
    {
        "prompt": "Write a Python function called `clamp` that clamps a number between a minimum and maximum value.\n\ndef clamp(x, low, high):",
        "tests": "assert clamp(5, 0, 10) == 5\nassert clamp(-5, 0, 10) == 0\nassert clamp(15, 0, 10) == 10",
        "name": "clamp",
    },
    {
        "prompt": "Write a Python function called `contains` that returns True if an item is in a list.\n\ndef contains(lst, item):",
        "tests": "assert contains([1, 2, 3], 2) == True\nassert contains([1, 2, 3], 4) == False\nassert contains([], 1) == False",
        "name": "contains",
    },
    {
        "prompt": "Write a Python function called `factorial` that returns the factorial of a non-negative integer.\n\ndef factorial(n):",
        "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
        "name": "factorial",
    },
    {
        "prompt": "Write a Python function called `power` that returns base raised to the exponent.\n\ndef power(base, exp):",
        "tests": "assert power(2, 3) == 8\nassert power(5, 0) == 1\nassert power(3, 2) == 9",
        "name": "power",
    },
    {
        "prompt": "Write a Python function called `min_of_list` that returns the smallest number in a list.\n\ndef min_of_list(lst):",
        "tests": "assert min_of_list([3, 1, 2]) == 1\nassert min_of_list([5]) == 5\nassert min_of_list([-1, -5, 0]) == -5",
        "name": "min_of_list",
    },
    {
        "prompt": "Write a Python function called `max_of_list` that returns the largest number in a list.\n\ndef max_of_list(lst):",
        "tests": "assert max_of_list([3, 1, 2]) == 3\nassert max_of_list([5]) == 5\nassert max_of_list([-1, -5, 0]) == 0",
        "name": "max_of_list",
    },
    {
        "prompt": "Write a Python function called `remove_duplicates` that returns a list with duplicates removed, preserving order.\n\ndef remove_duplicates(lst):",
        "tests": "assert remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == [1]",
        "name": "remove_duplicates",
    },
    {
        "prompt": "Write a Python function called `flatten` that flattens a list of lists into a single list.\n\ndef flatten(lst):",
        "tests": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\nassert flatten([[], [1]]) == [1]\nassert flatten([]) == []",
        "name": "flatten",
    },
    {
        "prompt": "Write a Python function called `is_palindrome` that returns True if a string reads the same forwards and backwards.\n\ndef is_palindrome(s):",
        "tests": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True",
        "name": "is_palindrome",
    },
    {
        "prompt": "Write a Python function called `count_vowels` that returns the number of vowels (a, e, i, o, u) in a string.\n\ndef count_vowels(s):",
        "tests": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\nassert count_vowels('aeiou') == 5",
        "name": "count_vowels",
    },
    {
        "prompt": "Write a Python function called `fibonacci` that returns the nth Fibonacci number (0-indexed).\n\ndef fibonacci(n):",
        "tests": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8",
        "name": "fibonacci",
    },
    {
        "prompt": "Write a Python function called `intersection` that returns elements common to two lists.\n\ndef intersection(a, b):",
        "tests": "assert sorted(intersection([1, 2, 3], [2, 3, 4])) == [2, 3]\nassert intersection([1, 2], [3, 4]) == []\nassert sorted(intersection([1, 1, 2], [1, 2, 2])) == [1, 2]",
        "name": "intersection",
    },
]


def extract_function_body(completion: str, func_name: str) -> str | None:
    """Extract just the indented function body from the model's completion."""
    lines = completion.split("\n")
    body_lines = []
    found_body = False

    for line in lines:
        # Empty lines within the body are ok
        if not line.strip():
            if found_body:
                body_lines.append(line)
            continue

        # Indented lines are part of the function body
        if line.startswith("    ") or line.startswith("\t"):
            found_body = True
            body_lines.append(line)
        elif found_body:
            # First non-indented, non-empty line = end of function body
            break

    # Strip trailing empty lines
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    body = "\n".join(body_lines)
    if not body.strip():
        return None
    return body


def run_tests(func_def: str, func_body: str, tests: str, timeout: float = 2.0) -> tuple[bool, int, int]:
    """Run test cases against generated code. Returns (all_passed, passed_count, total_count)."""
    import signal

    # Build the full function
    code = func_def + "\n" + func_body

    test_lines = [t.strip() for t in tests.strip().split("\n") if t.strip().startswith("assert")]
    total = len(test_lines)
    passed = 0

    # Set up timeout
    def timeout_handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(int(timeout))
        namespace = {}
        exec(code, namespace)

        for test in test_lines:
            try:
                exec(test, namespace)
                passed += 1
            except Exception:
                pass

        signal.alarm(0)
    except (TimeoutError, Exception):
        signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    return passed == total, passed, total


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    max_train_samples: int = 200,
    max_eval_samples: int = 50,
) -> flyte.io.Dir:
    """Create coding problem dataset by repeating and shuffling the problem set."""
    import random
    from datasets import Dataset, DatasetDict

    log.info(f"Preparing {max_train_samples + max_eval_samples} coding problems...")

    rng = random.Random(42)

    # Repeat problems to fill the requested size
    all_examples = []
    while len(all_examples) < max_train_samples + max_eval_samples:
        for p in PROBLEMS:
            all_examples.append({
                "prompt": p["prompt"],
                "func_prompt": p["prompt"],  # duplicate — "prompt" is reserved by GRPOTrainer
                "tests": p["tests"],
                "name": p["name"],
            })
            if len(all_examples) >= max_train_samples + max_eval_samples:
                break

    rng.shuffle(all_examples)

    train_examples = all_examples[:max_train_samples]
    eval_examples = all_examples[max_train_samples:max_train_samples + max_eval_samples]

    processed = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "eval": Dataset.from_list(eval_examples),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(train_examples)} train, {len(eval_examples)} eval ({len(PROBLEMS)} unique problems)")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train with GRPO
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune a model with GRPO — reward = do the tests pass?"""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GRPO Code Training: model={model_name}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2><p>Setting up GRPO for code gen...</p>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Load model + tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    # -- LoRA --
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -- Metrics tracking --
    metrics_history = {
        "batch": [], "avg_reward": [], "pass_rate": [],
        "batch_reward": [], "batch_pass_rate": [],
    }
    reward_stats = {"calls": 0, "total_reward": 0, "all_pass": 0, "total": 0}

    def build_charts() -> str:
        import base64
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(metrics_history["batch"], metrics_history["avg_reward"], "b-", linewidth=1.5, label="Running avg")
        axes[0].plot(metrics_history["batch"], metrics_history["batch_reward"], "b.", alpha=0.3, markersize=4, label="Per batch")
        axes[0].set_xlabel("Batch")
        axes[0].set_ylabel("Avg Reward")
        axes[0].set_title("Reward (Tests Passed)")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(metrics_history["batch"], metrics_history["pass_rate"], "g-", linewidth=1.5, label="Running avg")
        axes[1].plot(metrics_history["batch"], metrics_history["batch_pass_rate"], "g.", alpha=0.3, markersize=4, label="Per batch")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("All Tests Pass %")
        axes[1].set_title("Pass Rate")
        axes[1].set_ylim(0, 105)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def update_report():
        avg_reward = reward_stats["total_reward"] / max(reward_stats["total"], 1)
        pass_rate = reward_stats["all_pass"] / max(reward_stats["total"], 1) * 100

        chart_html = ""
        if len(metrics_history["batch"]) >= 2:
            chart_b64 = build_charts()
            chart_html = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;max-width:800px;" />'

        flyte.report.replace(
            f"<h2>GRPO Code Training — {model_name}</h2>"
            f"<table>"
            f"<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Batches</td><td>{reward_stats['calls']}</td></tr>"
            f"<tr><td>Avg Reward</td><td>{avg_reward:.3f}</td></tr>"
            f"<tr><td>All Tests Pass</td><td>{pass_rate:.1f}%</td></tr>"
            f"<tr><td>Total Samples</td><td>{reward_stats['total']}</td></tr>"
            f"</table>"
            f"{chart_html}"
        )
        flyte.report.flush()

    # -- Reward function --
    def code_reward(completions: list[str], func_prompt: list[str], tests: list[str], name: list[str], **kwargs) -> list[float]:
        """Reward = fraction of test cases passed. All pass = 1.0."""
        rewards = []
        batch_passes = 0

        for completion, p, t, n in zip(completions, func_prompt, tests, name):
            func_body = extract_function_body(completion, n)
            if func_body is None:
                rewards.append(0.0)
                reward_stats["total"] += 1
                continue

            # Build the full function def line from the prompt
            func_def = p.strip().split("\n")[-1]  # "def func_name(...):"
            all_passed, passed, total = run_tests(func_def, func_body, t)

            reward = passed / total if total > 0 else 0.0
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

        metrics_history["batch"].append(reward_stats["calls"])
        metrics_history["avg_reward"].append(running_avg)
        metrics_history["pass_rate"].append(running_pass)
        metrics_history["batch_reward"].append(batch_avg)
        metrics_history["batch_pass_rate"].append(batch_pass_pct)

        if reward_stats["calls"] % 5 == 0:
            log.info(
                f"[Reward] batch {reward_stats['calls']}: "
                f"avg_reward={running_avg:.3f}, pass_rate={running_pass:.1f}%, "
                f"batch_reward={batch_avg:.3f}"
            )
            update_report()

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
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset["train"],
        reward_funcs=code_reward,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info(f"Starting GRPO training (num_generations={num_generations})...")
    update_report()

    trainer.train()

    final_avg = reward_stats["total_reward"] / max(reward_stats["total"], 1)
    final_pass = reward_stats["all_pass"] / max(reward_stats["total"], 1) * 100
    log.info(f"GRPO training complete. Avg reward: {final_avg:.3f}, pass rate: {final_pass:.1f}%")
    update_report()

    # -- Merge LoRA and save --
    save_dir = os.path.join(tempfile.mkdtemp(), "grpo_model")
    log.info("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    chart_html = ""
    if len(metrics_history["batch"]) >= 2:
        chart_b64 = build_charts()
        chart_html = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;max-width:800px;" />'

    await flyte.report.replace.aio(
        f"<h2>GRPO Training Complete — {model_name}</h2>"
        f"<p><b>Epochs:</b> {epochs} | <b>LR:</b> {lr}</p>"
        f"<p><b>Avg reward:</b> {final_avg:.3f} | <b>Pass rate:</b> {final_pass:.1f}%</p>"
        f"{chart_html}"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 30,
) -> str:
    """Compare base vs GRPO-trained model on code generation."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    prompts = eval_ds["prompt"]
    tests_list = eval_ds["tests"]
    names = eval_ds["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    def generate_code(model, prompts, max_new_tokens=128):
        results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append(generated)
        return results

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running base model...</p>")
    await flyte.report.flush.aio()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, dtype=dtype, device_map="auto")
    base_results = generate_code(base_model, prompts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GRPO model --
    log.info("Loading GRPO-trained model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running GRPO-trained model...</p>")
    await flyte.report.flush.aio()

    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    ft_results = generate_code(ft_model, prompts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    base_pass = 0
    ft_pass = 0
    base_total_tests = 0
    base_passed_tests = 0
    ft_total_tests = 0
    ft_passed_tests = 0
    comparisons = []

    for i in range(len(prompts)):
        func_def = prompts[i].strip().split("\n")[-1]

        base_body = extract_function_body(base_results[i], names[i])
        ft_body = extract_function_body(ft_results[i], names[i])

        base_all = False
        base_p = 0
        base_t = 0
        if base_body:
            base_all, base_p, base_t = run_tests(func_def, base_body, tests_list[i])
        if base_all:
            base_pass += 1
        base_total_tests += base_t
        base_passed_tests += base_p

        ft_all = False
        ft_p = 0
        ft_t = 0
        if ft_body:
            ft_all, ft_p, ft_t = run_tests(func_def, ft_body, tests_list[i])
        if ft_all:
            ft_pass += 1
        ft_total_tests += ft_t
        ft_passed_tests += ft_p

        comparisons.append({
            "name": names[i],
            "prompt": prompts[i][:200],
            "base_code": base_results[i][:300],
            "grpo_code": ft_results[i][:300],
            "base_passed": f"{base_p}/{base_t}",
            "grpo_passed": f"{ft_p}/{ft_t}",
            "base_all_pass": base_all,
            "grpo_all_pass": ft_all,
        })

    total = len(prompts)
    base_rate = base_pass / total * 100
    ft_rate = ft_pass / total * 100

    log.info(f"Base model: {base_rate:.1f}% all-pass ({base_pass}/{total})")
    log.info(f"GRPO model: {ft_rate:.1f}% all-pass ({ft_pass}/{total})")

    # -- Report --
    examples_html = ""
    for c in comparisons[:15]:
        base_color = "green" if c["base_all_pass"] else "red"
        ft_color = "green" if c["grpo_all_pass"] else "red"
        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p><b>{c['name']}</b> — Tests: base <span style="color:{base_color};">{c['base_passed']}</span> | GRPO <span style="color:{ft_color};">{c['grpo_passed']}</span></p>
<p><b>Base:</b><pre style="background:#f5f5f5;padding:8px;font-size:0.85em;">{c['base_code'][:200]}</pre></p>
<p><b>GRPO:</b><pre style="background:#f0fff0;padding:8px;font-size:0.85em;">{c['grpo_code'][:200]}</pre></p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results — Code Generation</h2>
<table>
<tr><th></th><th>All Tests Pass</th><th>Individual Tests</th></tr>
<tr><td><b>Base model</b></td><td>{base_rate:.1f}% ({base_pass}/{total})</td><td>{base_passed_tests}/{base_total_tests}</td></tr>
<tr><td><b>GRPO-trained</b></td><td>{ft_rate:.1f}% ({ft_pass}/{total})</td><td>{ft_passed_tests}/{ft_total_tests}</td></tr>
</table>
<p><b>Improvement:</b> {ft_rate - base_rate:+.1f} percentage points</p>
<hr/>
<h3>Examples</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_pass_rate": round(base_rate, 1),
        "grpo_pass_rate": round(ft_rate, 1),
        "improvement": round(ft_rate - base_rate, 1),
        "base_tests": f"{base_passed_tests}/{base_total_tests}",
        "grpo_tests": f"{ft_passed_tests}/{ft_total_tests}",
        "num_problems": total,
        "comparisons": comparisons[:15],
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 128,
    max_train_samples: int = 200,
    max_eval_samples: int = 50,
    num_eval_examples: int = 30,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    GRPO fine-tuning pipeline — teach a model to write correct Python.

    1. Generate coding problems with test cases
    2. Train with GRPO — reward = fraction of tests passed
    3. Evaluate: pass rate before/after
    """
    log.info(f"Pipeline: {model_name} | GRPO code generation")

    await flyte.report.replace.aio(
        f"<h2>GRPO Code Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 1/3: Preparing problems...</p>"
    )
    await flyte.report.flush.aio()

    data_dir = await prepare_data(max_train_samples, max_eval_samples)

    await flyte.report.replace.aio(
        f"<h2>GRPO Code Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/3: GRPO training (reward = tests pass)...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(
        model_name, data_dir, epochs, lr, batch_size,
        num_generations, max_completion_length, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        f"<h2>GRPO Code Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/3: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    await flyte.report.replace.aio(
        f"<h2>GRPO Code Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Base pass rate:</b> {metrics['base_pass_rate']}%</p>"
        f"<p><b>GRPO pass rate:</b> {metrics['grpo_pass_rate']}%</p>"
        f"<p><b>Improvement:</b> {metrics['improvement']:+.1f} percentage points</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return result
