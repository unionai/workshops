"""
GRPO Fine-Tuning — Teach a model to count letters in words.

GRPO (Group Relative Policy Optimization) generates multiple completions per
prompt, scores them with a reward function, and uses the relative performance
within the group to update the policy.

LLMs famously struggle with counting letters in words. Here we use GRPO to
teach a small model this skill — the reward is simply how close the model's
answer is to the correct count.

Usage:
    # Default (SmolLM2-135M)
    flyte run --local --tui workflow.py pipeline

    # Quick test
    flyte run --local --tui workflow.py pipeline --max_train_samples 50 --epochs 1

    # Remote
    flyte run workflow.py pipeline --epochs 3

    # Bigger model
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
"""

import json
import logging
import os
import random
import re
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env, HF_TOKEN

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Common words with repeated letters — good for letter counting
WORDS = [
    "banana", "apple", "strawberry", "blueberry", "pineapple", "watermelon",
    "raspberry", "blackberry", "pomegranate", "cantaloupe", "tangerine",
    "persimmon", "avocado", "broccoli", "cauliflower", "asparagus",
    "mushroom", "zucchini", "jalapeño", "pepperoni", "mozzarella",
    "cinnamon", "chocolate", "cappuccino", "espresso", "croissant",
    "baguette", "spaghetti", "fettuccine", "macaroni", "ravioli",
    "tagliatelle", "prosciutto", "bruschetta", "tiramisu", "amaretto",
    "pistachio", "butterscotch", "marshmallow", "gingerbread",
    "peppermint", "spearmint", "rosemary", "oregano", "coriander",
    "cardamom", "turmeric", "chamomile", "dandelion", "sunflower",
    "chrysanthemum", "rhinoceros", "hippopotamus", "crocodile",
    "caterpillar", "grasshopper", "dragonfly", "butterfly", "jellyfish",
    "salamander", "chameleon", "armadillo", "chinchilla", "porcupine",
    "woodpecker", "hummingbird", "nightingale", "mockingbird",
    "mississippi", "tennessee", "connecticut", "massachusetts",
    "philadelphia", "indianapolis", "minneapolis", "jacksonville",
    "abracadabra", "onomatopoeia", "encyclopedia", "communication",
    "environment", "temperature", "celebration", "entertainment",
    "professional", "international", "extraordinary", "sophisticated",
]


def generate_examples(n: int, seed: int = 42) -> list[dict]:
    """Generate letter-counting examples."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        word = rng.choice(WORDS).lower()
        # Pick a letter that appears in the word
        letters_in_word = list(set(word))
        # Mix in some letters that might not appear too
        all_letters = letters_in_word + list("aeiourst")
        letter = rng.choice(all_letters)
        count = word.count(letter)

        examples.append({
            "prompt": f"How many times does the letter '{letter}' appear in the word '{word}'? Answer with just the number.",
            "word": word,
            "letter": letter,
            "count": count,
        })
    return examples


def extract_number(text: str) -> int | None:
    """Extract the first number from model output."""
    # Look for standalone digits first
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return int(match.group(1))
    # Try number words
    word_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for word, num in word_to_num.items():
        if word in text.lower():
            return num
    return None


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    max_train_samples: int = 500,
    max_eval_samples: int = 100,
) -> flyte.io.Dir:
    """Generate letter-counting examples."""
    from datasets import Dataset, DatasetDict

    log.info(f"Generating {max_train_samples + max_eval_samples} letter-counting examples...")

    all_examples = generate_examples(max_train_samples + max_eval_samples, seed=42)
    train_examples = all_examples[:max_train_samples]
    eval_examples = all_examples[max_train_samples:max_train_samples + max_eval_samples]

    processed = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "eval": Dataset.from_list(eval_examples),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(train_examples)} train, {len(eval_examples)} eval")

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
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 64,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune a model with GRPO to count letters in words."""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    log.info(f"GRPO Training: model={model_name}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        log.info("No GPU available, running on CPU")

    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2><p>Setting up GRPO...</p>")
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

    # -- LoRA config --
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
        "batch": [],
        "avg_reward": [],
        "exact_match_pct": [],
        "batch_reward": [],
        "batch_exact_pct": [],
    }
    reward_stats = {"calls": 0, "total_reward": 0, "exact_matches": 0, "total": 0}

    def build_charts() -> str:
        """Generate training charts as base64 PNG."""
        import base64
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Reward chart
        axes[0].plot(metrics_history["batch"], metrics_history["avg_reward"], "b-", linewidth=1.5, label="Running avg")
        axes[0].plot(metrics_history["batch"], metrics_history["batch_reward"], "b.", alpha=0.3, markersize=4, label="Per batch")
        axes[0].set_xlabel("Batch")
        axes[0].set_ylabel("Avg Reward")
        axes[0].set_title("Reward")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Accuracy chart
        axes[1].plot(metrics_history["batch"], metrics_history["exact_match_pct"], "g-", linewidth=1.5, label="Running avg")
        axes[1].plot(metrics_history["batch"], metrics_history["batch_exact_pct"], "g.", alpha=0.3, markersize=4, label="Per batch")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("Exact Match %")
        axes[1].set_title("Accuracy")
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
        """Update the Flyte report with current charts and stats."""
        avg_reward = reward_stats["total_reward"] / max(reward_stats["total"], 1)
        exact_pct = reward_stats["exact_matches"] / max(reward_stats["total"], 1) * 100

        chart_html = ""
        if len(metrics_history["batch"]) >= 2:
            chart_b64 = build_charts()
            chart_html = f'<img src="data:image/png;base64,{chart_b64}" alt="Training charts" style="width:100%;max-width:800px;" />'

        flyte.report.replace(
            f"<h2>GRPO Training — {model_name}</h2>"
            f"<p><b>Task:</b> Count letters in words | "
            f"<b>Generations per prompt:</b> {num_generations}</p>"
            f"<table>"
            f"<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Batches</td><td>{reward_stats['calls']}</td></tr>"
            f"<tr><td>Avg Reward</td><td>{avg_reward:.3f}</td></tr>"
            f"<tr><td>Exact Match</td><td>{exact_pct:.1f}%</td></tr>"
            f"<tr><td>Total Samples</td><td>{reward_stats['total']}</td></tr>"
            f"</table>"
            f"{chart_html}"
        )
        flyte.report.flush()

    # -- Reward function --
    def counting_reward(completions: list[str], count: list[int], **kwargs) -> list[float]:
        """Binary reward: 1.0 only if the exact letter count is correct, else 0.0.

        Binary is deliberate. A graded reward (off-by-1 = 0.5, off-by-2 = 0.25) is
        trivially hackable here: since most counts are small, always answering "1"
        lands within 1-2 of the truth on most words and racks up partial credit —
        so the model collapses to a constant instead of learning to count.
        All-or-nothing removes that shortcut.
        """
        rewards = []
        batch_exact = 0
        for completion, expected in zip(completions, count):
            predicted = extract_number(completion)
            if predicted is not None and predicted == expected:
                rewards.append(1.0)
                reward_stats["exact_matches"] += 1
                batch_exact += 1
            else:
                rewards.append(0.0)
            reward_stats["total"] += 1

        reward_stats["calls"] += 1
        reward_stats["total_reward"] += sum(rewards)

        # Track per-batch metrics
        batch_avg = sum(rewards) / len(rewards)
        batch_exact_pct = batch_exact / len(completions) * 100
        running_avg = reward_stats["total_reward"] / reward_stats["total"]
        running_exact = reward_stats["exact_matches"] / reward_stats["total"] * 100

        metrics_history["batch"].append(reward_stats["calls"])
        metrics_history["avg_reward"].append(running_avg)
        metrics_history["exact_match_pct"].append(running_exact)
        metrics_history["batch_reward"].append(batch_avg)
        metrics_history["batch_exact_pct"].append(batch_exact_pct)

        if reward_stats["calls"] % 5 == 0:
            log.info(
                f"[Reward] batch {reward_stats['calls']}: "
                f"avg_reward={running_avg:.3f}, exact_match={running_exact:.1f}%, "
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

    # -- Train --
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset["train"],
        reward_funcs=counting_reward,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info(f"Starting GRPO training (num_generations={num_generations})...")
    update_report()

    trainer.train()

    # Final report with charts
    final_avg = reward_stats["total_reward"] / max(reward_stats["total"], 1)
    final_acc = reward_stats["exact_matches"] / max(reward_stats["total"], 1) * 100
    log.info(f"GRPO training complete. Final avg reward: {final_avg:.3f}, exact match: {final_acc:.1f}%")
    update_report()

    # -- Merge LoRA and save --
    save_dir = os.path.join(tempfile.mkdtemp(), "grpo_model")
    log.info("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # Final report with charts
    chart_html = ""
    if len(metrics_history["batch"]) >= 2:
        chart_b64 = build_charts()
        chart_html = f'<img src="data:image/png;base64,{chart_b64}" alt="Training charts" style="width:100%;max-width:800px;" />'

    await flyte.report.replace.aio(
        f"<h2>GRPO Training Complete — {model_name}</h2>"
        f"<p><b>Epochs:</b> {epochs} | <b>LR:</b> {lr}</p>"
        f"<p><b>Final avg reward:</b> {final_avg:.3f} | <b>Exact match:</b> {final_acc:.1f}%</p>"
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
    num_examples: int = 50,
) -> str:
    """Compare base vs GRPO-trained model on letter counting."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    prompts = eval_ds["prompt"]
    words = eval_ds["word"]
    letters = eval_ds["letter"]
    counts = eval_ds["count"]

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    def generate_answers(model, prompts, max_new_tokens=96):
        results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append(generated)
        return results

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running base model...</p>")
    await flyte.report.flush.aio()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, dtype=dtype, device_map="auto")
    base_results = generate_answers(base_model, prompts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GRPO model --
    log.info("Loading GRPO-trained model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running GRPO-trained model...</p>")
    await flyte.report.flush.aio()

    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    ft_results = generate_answers(ft_model, prompts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    base_exact = 0
    base_close = 0
    ft_exact = 0
    ft_close = 0
    comparisons = []

    for i in range(len(prompts)):
        expected = counts[i]
        base_pred = extract_number(base_results[i])
        ft_pred = extract_number(ft_results[i])

        base_match = base_pred is not None and base_pred == expected
        ft_match = ft_pred is not None and ft_pred == expected
        base_near = base_pred is not None and abs(base_pred - expected) <= 1
        ft_near = ft_pred is not None and abs(ft_pred - expected) <= 1

        if base_match:
            base_exact += 1
        if base_near:
            base_close += 1
        if ft_match:
            ft_exact += 1
        if ft_near:
            ft_close += 1

        comparisons.append({
            "word": words[i],
            "letter": letters[i],
            "expected": expected,
            "base_answer": base_pred,
            "base_output": base_results[i][:200],
            "ft_answer": ft_pred,
            "ft_output": ft_results[i][:200],
            "base_correct": base_match,
            "ft_correct": ft_match,
        })

    total = len(prompts)
    base_acc = base_exact / total * 100
    ft_acc = ft_exact / total * 100
    base_close_pct = base_close / total * 100
    ft_close_pct = ft_close / total * 100

    log.info(f"Base model: exact={base_acc:.1f}%, within 1={base_close_pct:.1f}%")
    log.info(f"GRPO model: exact={ft_acc:.1f}%, within 1={ft_close_pct:.1f}%")

    # -- Report --
    examples_html = ""
    for c in comparisons[:15]:
        base_color = "green" if c["base_correct"] else "red"
        ft_color = "green" if c["ft_correct"] else "red"
        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p><b>How many '{c['letter']}'s in '{c['word']}'?</b> Answer: {c['expected']}</p>
<p><b>Base:</b> <span style="color:{base_color};">{c['base_output'][:100]}</span> (extracted: {c['base_answer']})</p>
<p><b>GRPO:</b> <span style="color:{ft_color};">{c['ft_output'][:100]}</span> (extracted: {c['ft_answer']})</p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results — Letter Counting</h2>
<table>
<tr><th></th><th>Exact Match</th><th>Within ±1</th></tr>
<tr><td><b>Base model</b></td><td>{base_acc:.1f}%</td><td>{base_close_pct:.1f}%</td></tr>
<tr><td><b>GRPO-trained</b></td><td>{ft_acc:.1f}%</td><td>{ft_close_pct:.1f}%</td></tr>
</table>
<p><b>Exact match improvement:</b> {ft_acc - base_acc:+.1f} percentage points</p>
<hr/>
<h3>Examples</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_exact_match": round(base_acc, 1),
        "base_within_1": round(base_close_pct, 1),
        "grpo_exact_match": round(ft_acc, 1),
        "grpo_within_1": round(ft_close_pct, 1),
        "improvement": round(ft_acc - base_acc, 1),
        "num_examples": total,
        "comparisons": comparisons[:15],
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 8,
    num_generations: int = 8,
    max_completion_length: int = 64,
    max_train_samples: int = 200,
    max_eval_samples: int = 60,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    GRPO fine-tuning pipeline — teach a model to count letters.

    1. Generate letter-counting examples (e.g. "How many 'a's in 'banana'?")
    2. Train with GRPO — reward = closeness to correct count
    3. Evaluate: accuracy before/after
    """
    log.info(f"Pipeline: {model_name} | GRPO letter counting")

    await flyte.report.replace.aio(
        f"<h2>GRPO Pipeline — Letter Counting</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 1/3: Generating examples...</p>"
    )
    await flyte.report.flush.aio()

    data_dir = await prepare_data(max_train_samples, max_eval_samples)

    await flyte.report.replace.aio(
        f"<h2>GRPO Pipeline — Letter Counting</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/3: GRPO training...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(
        model_name, data_dir, epochs, lr, batch_size,
        num_generations, max_completion_length, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        f"<h2>GRPO Pipeline — Letter Counting</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/3: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    await flyte.report.replace.aio(
        f"<h2>GRPO Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Base accuracy:</b> {metrics['base_exact_match']}%</p>"
        f"<p><b>GRPO accuracy:</b> {metrics['grpo_exact_match']}%</p>"
        f"<p><b>Improvement:</b> {metrics['improvement']:+.1f} percentage points</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return result
