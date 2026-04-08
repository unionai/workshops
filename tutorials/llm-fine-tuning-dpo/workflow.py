"""
DPO Fine-Tuning — Align a model with human preferences.

DPO (Direct Preference Optimization) trains a model to prefer "chosen" responses
over "rejected" ones, without needing a separate reward model. It directly
optimizes the policy using preference pairs.

Uses the Anthropic HH-RLHF dataset — pairs of helpful/harmless (chosen) vs
unhelpful/harmful (rejected) assistant responses.

Usage:
    # Default (SmolLM2-135M on HH-RLHF)
    flyte run --local --tui workflow.py pipeline

    # Quick test
    flyte run --local --tui workflow.py pipeline --max_train_samples 100 --max_eval_samples 20 --epochs 1

    # Remote
    flyte run workflow.py pipeline --epochs 2

    # Bigger model
    flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
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


def extract_last_turn(conversation: str) -> dict:
    """Extract the last human question and assistant response from HH-RLHF format."""
    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    parts = conversation.split("\n\nAssistant: ")
    if len(parts) < 2:
        return {"prompt": conversation.strip(), "response": ""}

    response = parts[-1].strip()
    prompt_parts = "\n\nAssistant: ".join(parts[:-1])

    # Get the last human turn as the prompt
    human_parts = prompt_parts.split("\n\nHuman: ")
    if len(human_parts) >= 2:
        last_human = human_parts[-1].strip()
        # Include prior context if it exists
        context = "\n\nHuman: ".join(human_parts[:-1]).strip()
        if context:
            prompt = context + "\n\nHuman: " + last_human
        else:
            prompt = last_human
    else:
        prompt = prompt_parts.strip()

    return {"prompt": prompt, "response": response}


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "Anthropic/hh-rlhf",
    max_train_samples: int = 2000,
    max_eval_samples: int = 500,
) -> flyte.io.Dir:
    """Download HH-RLHF and format as DPO preference pairs."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    def format_preference(ex):
        chosen = extract_last_turn(ex["chosen"])
        rejected = extract_last_turn(ex["rejected"])
        return {
            "prompt": chosen["prompt"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
        }

    train_ds = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    eval_ds = ds["test"].select(range(min(max_eval_samples, len(ds["test"]))))

    train_ds = train_ds.map(format_preference)
    eval_ds = eval_ds.map(format_preference)

    # Filter out any examples where chosen == rejected
    train_ds = train_ds.filter(lambda x: x["chosen"] != x["rejected"])
    eval_ds = eval_ds.filter(lambda x: x["chosen"] != x["rejected"])

    processed = DatasetDict({"train": train_ds, "eval": eval_ds})

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train with DPO
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 2,
    lr: float = 5e-6,
    batch_size: int = 2,
    beta: float = 0.1,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Fine-tune a model with DPO using human preference pairs."""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    log.info(f"DPO Training: model={model_name}, beta={beta}")

    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2><p>Setting up DPO...</p>")
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

    # -- DPO config --
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        beta=beta,
        max_length=max_length,
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    # -- Train --
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info(f"Starting DPO training (beta={beta})...")

    await flyte.report.replace.aio(
        f"<h2>DPO Training — {model_name}</h2>"
        f"<p><b>Beta:</b> {beta} (higher = stay closer to reference policy)</p>"
        f"<p>Training...</p>"
    )
    await flyte.report.flush.aio()

    trainer.train()
    log.info("DPO training complete.")

    # -- Merge LoRA and save --
    save_dir = os.path.join(tempfile.mkdtemp(), "dpo_model")
    log.info("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    await flyte.report.replace.aio(
        f"<h2>DPO Training Complete — {model_name}</h2>"
        f"<p><b>Epochs:</b> {epochs} | <b>LR:</b> {lr} | <b>Beta:</b> {beta}</p>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate — preference win rate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 50,
) -> str:
    """Compare base vs DPO-trained model — does it prefer chosen over rejected?"""
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
    chosen_responses = eval_ds["chosen"]
    rejected_responses = eval_ds["rejected"]

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    def compute_log_probs(model, prompt, response):
        """Compute average log probability of response given prompt."""
        text = prompt + "\n\n" + response
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        prompt_len = prompt_ids.input_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Only score the response tokens
        response_logits = logits[:, prompt_len - 1 : -1, :]
        response_labels = inputs.input_ids[:, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.mean().item()

    def compute_win_rate(model, prompts, chosen, rejected):
        """How often does the model assign higher probability to chosen vs rejected?"""
        wins = 0
        for i in range(len(prompts)):
            chosen_lp = compute_log_probs(model, prompts[i], chosen[i])
            rejected_lp = compute_log_probs(model, prompts[i], rejected[i])
            if chosen_lp > rejected_lp:
                wins += 1
            if (i + 1) % 10 == 0:
                log.info(f"Evaluated {i + 1}/{len(prompts)}")
        return wins, len(prompts)

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Scoring with base model...</p>")
    await flyte.report.flush.aio()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, dtype=dtype, device_map="auto",
    )
    base_model.eval()
    base_wins, total = compute_win_rate(base_model, prompts, chosen_responses, rejected_responses)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- DPO model --
    log.info("Loading DPO-trained model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Scoring with DPO-trained model...</p>")
    await flyte.report.flush.aio()

    ft_path = await finetuned_dir.download()
    ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    ft_model.eval()
    ft_wins, _ = compute_win_rate(ft_model, prompts, chosen_responses, rejected_responses)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_win_rate = base_wins / total * 100
    ft_win_rate = ft_wins / total * 100

    log.info(f"Base model win rate: {base_win_rate:.1f}% ({base_wins}/{total})")
    log.info(f"DPO model win rate: {ft_win_rate:.1f}% ({ft_wins}/{total})")

    # -- Generate sample responses for comparison --
    log.info("Generating sample responses...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Generating comparison responses...</p>")
    await flyte.report.flush.aio()

    ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=dtype, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, dtype=dtype, device_map="auto",
    )

    comparisons = []
    examples_html = ""
    for i in range(min(8, len(prompts))):
        prompt = prompts[i]

        # Generate from both models
        inputs = tokenizer(prompt + "\n\n", return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            base_out = base_model.generate(
                **inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        base_gen = tokenizer.decode(base_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        inputs = {k: v.to(ft_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            ft_out = ft_model.generate(
                **inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        ft_gen = tokenizer.decode(ft_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
        comparisons.append({
            "prompt": prompt_preview,
            "base_response": base_gen[:300],
            "dpo_response": ft_gen[:300],
            "chosen": chosen_responses[i][:300],
        })

        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p><b>Prompt:</b> {prompt_preview}</p>
<p><b>Chosen (ground truth):</b> <span style="color:#666;">{chosen_responses[i][:200]}</span></p>
<p><b>Base model:</b> {base_gen[:200]}</p>
<p><b>DPO model:</b> {ft_gen[:200]}</p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results — Preference Win Rate</h2>
<p>Win rate = how often the model assigns higher probability to the <b>chosen</b> response over the <b>rejected</b> one.</p>
<table>
<tr><th></th><th>Win Rate</th><th>Wins</th></tr>
<tr><td><b>Base model</b></td><td>{base_win_rate:.1f}%</td><td>{base_wins}/{total}</td></tr>
<tr><td><b>DPO-trained</b></td><td>{ft_win_rate:.1f}%</td><td>{ft_wins}/{total}</td></tr>
</table>
<p><b>Improvement:</b> {ft_win_rate - base_win_rate:+.1f} percentage points</p>
<hr/>
<h3>Generated Responses</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_win_rate": round(base_win_rate, 1),
        "dpo_win_rate": round(ft_win_rate, 1),
        "improvement": round(ft_win_rate - base_win_rate, 1),
        "num_examples": total,
        "comparisons": comparisons,
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "Anthropic/hh-rlhf",
    epochs: int = 2,
    lr: float = 5e-6,
    batch_size: int = 2,
    beta: float = 0.1,
    max_length: int = 512,
    max_train_samples: int = 2000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    DPO fine-tuning pipeline for preference alignment.

    1. Download and format Anthropic HH-RLHF preference pairs
    2. Train with DPO — model learns to prefer chosen over rejected responses
    3. Evaluate: win rate (chosen vs rejected) before/after
    """
    log.info(f"Pipeline: {model_name} | DPO | dataset={dataset_name}")

    await flyte.report.replace.aio(
        f"<h2>DPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Dataset:</b> {dataset_name}</p>"
        f"<p>Step 1/3: Preparing data...</p>"
    )
    await flyte.report.flush.aio()

    data_dir = await prepare_data(dataset_name, max_train_samples, max_eval_samples)

    await flyte.report.replace.aio(
        f"<h2>DPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/3: DPO training...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(
        model_name, data_dir, epochs, lr, batch_size,
        beta, max_length, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        f"<h2>DPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/3: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    await flyte.report.replace.aio(
        f"<h2>DPO Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Base win rate:</b> {metrics['base_win_rate']}%</p>"
        f"<p><b>DPO win rate:</b> {metrics['dpo_win_rate']}%</p>"
        f"<p><b>Improvement:</b> {metrics['improvement']:+.1f} percentage points</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return result
