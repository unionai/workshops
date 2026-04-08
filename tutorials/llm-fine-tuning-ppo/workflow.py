"""
PPO Fine-Tuning — Classic RLHF with a reward model in the loop.

PPO (Proximal Policy Optimization) is the original RLHF technique used to
train ChatGPT. It's the most complex post-training method — requiring a
policy model, reference model, value head, and reward model all running
simultaneously.

The pipeline:
1. Train a reward model on preference data (chosen > rejected)
2. Use PPO to optimize the policy model against the reward model
3. Evaluate: compare base vs PPO-trained model

Uses the Anthropic HH-RLHF dataset for reward model training, then optimizes
the policy to generate more helpful, harmless responses.

Usage:
    # Default (SmolLM2-135M)
    flyte run --local --tui workflow.py pipeline

    # Quick test
    flyte run --local --tui workflow.py pipeline --max_train_samples 100 --max_eval_samples 20 --epochs 1 --ppo_epochs 1

    # Remote
    flyte run workflow.py pipeline --epochs 2 --ppo_epochs 2
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


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "Anthropic/hh-rlhf",
    max_train_samples: int = 2000,
    max_eval_samples: int = 500,
) -> flyte.io.Dir:
    """Download HH-RLHF and format for reward model training + PPO prompts."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    def extract_turns(conversation: str) -> dict:
        parts = conversation.split("\n\nAssistant: ")
        if len(parts) < 2:
            return {"prompt": conversation.strip(), "response": ""}
        response = parts[-1].strip()
        prompt_parts = "\n\nAssistant: ".join(parts[:-1])
        return {"prompt": prompt_parts.strip(), "response": response}

    def format_example(ex):
        chosen = extract_turns(ex["chosen"])
        rejected = extract_turns(ex["rejected"])
        return {
            "prompt": chosen["prompt"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
        }

    train_ds = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    eval_ds = ds["test"].select(range(min(max_eval_samples, len(ds["test"]))))

    train_ds = train_ds.map(format_example)
    eval_ds = eval_ds.map(format_example)

    train_ds = train_ds.filter(lambda x: x["chosen"] != x["rejected"])
    eval_ds = eval_ds.filter(lambda x: x["chosen"] != x["rejected"])

    processed = DatasetDict({"train": train_ds, "eval": eval_ds})

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    return await flyte.io.Dir.from_local(output_dir)


# ------------------------------------------------------------------
# Task 2: Train reward model
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train_reward_model(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Train a reward model on preference pairs using RewardTrainer."""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    log.info(f"Training reward model from: {model_name}")

    await flyte.report.replace.aio(f"<h2>Training Reward Model</h2><p>Base: {model_name}</p>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Reward model (classification head on top of causal LM) --
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=HF_TOKEN,
        num_labels=1,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # -- LoRA --
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    # -- Train --
    output_dir = os.path.join(tempfile.mkdtemp(), "reward_checkpoints")
    reward_config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        max_length=512,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info("Starting reward model training...")
    trainer.train()
    log.info("Reward model training complete.")

    # -- Save --
    save_dir = os.path.join(tempfile.mkdtemp(), "reward_model")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    await flyte.report.replace.aio(
        f"<h2>Reward Model Complete</h2>"
        f"<p><b>Base:</b> {model_name}</p>"
        f"<p><b>Epochs:</b> {epochs}</p>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: PPO training
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train_ppo(
    model_name: str,
    reward_model_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    ppo_epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> flyte.io.Dir:
    """Optimize the policy model with PPO against the reward model."""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    log.info(f"PPO Training: model={model_name}")

    await flyte.report.replace.aio(f"<h2>PPO Training</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    # -- Load data (just prompts for PPO) --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # -- Policy model (with value head for PPO) --
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        token=HF_TOKEN,
        dtype=dtype,
        peft_config=lora_config,
        device_map="auto",
    )

    # -- Reward model --
    reward_path = await reward_model_dir.download()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path, num_labels=1, dtype=dtype, device_map="auto",
    )
    reward_model.eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path)

    # -- PPO config --
    ppo_config = PPOConfig(
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        log_with=None,
    )

    # -- Prepare prompts --
    prompts = dataset["train"]["prompt"]

    # Tokenize prompts
    def tokenize_prompt(prompt):
        return tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt")

    # -- PPO Trainer --
    trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
    )

    # -- Training loop --
    log.info(f"Starting PPO training loop ({len(prompts)} prompts)...")

    await flyte.report.replace.aio(
        f"<h2>PPO Training — {model_name}</h2>"
        f"<p>Running PPO optimization loop...</p>"
    )
    await flyte.report.flush.aio()

    total_reward = 0
    num_batches = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        if len(batch_prompts) < batch_size:
            continue

        # Tokenize prompts
        query_tensors = [
            tokenizer(p, truncation=True, max_length=256, return_tensors="pt").input_ids.squeeze().to(model.pretrained_model.device)
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode responses
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Score with reward model
        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            text = prompt + "\n\n" + response
            inputs = reward_tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(reward_model.device)
            with torch.no_grad():
                score = reward_model(**inputs).logits.squeeze().item()
            rewards.append(torch.tensor(score))

        # PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        batch_reward = sum(r.item() for r in rewards) / len(rewards)
        total_reward += batch_reward
        num_batches += 1

        if (num_batches) % 5 == 0:
            avg_reward = total_reward / num_batches
            log.info(f"Batch {num_batches}: avg reward = {avg_reward:.4f}")
            flyte.report.replace(
                f"<h2>PPO Training — {model_name}</h2>"
                f"<p><b>Batch:</b> {num_batches} | "
                f"<b>Avg reward:</b> {avg_reward:.4f}</p>"
            )
            flyte.report.flush()

    log.info("PPO training complete.")

    # -- Save merged model --
    save_dir = os.path.join(tempfile.mkdtemp(), "ppo_model")
    log.info("Saving PPO-trained model...")

    # Merge LoRA and save
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    avg_reward = total_reward / max(num_batches, 1)
    await flyte.report.replace.aio(
        f"<h2>PPO Training Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Batches:</b> {num_batches} | <b>Avg reward:</b> {avg_reward:.4f}</p>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 4: Evaluate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    ppo_model_dir: flyte.io.Dir,
    reward_model_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    num_examples: int = 30,
) -> str:
    """Compare base vs PPO-trained model using reward model scores."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading models...</p>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))
    prompts = eval_ds["prompt"]

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # -- Reward model --
    reward_path = await reward_model_dir.download()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path, num_labels=1, dtype=dtype, device_map="auto",
    )
    reward_model.eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path)

    def generate_and_score(model, prompts):
        responses = []
        scores = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(response)

            # Score with reward model
            text = prompt + "\n\n" + response
            reward_inputs = reward_tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(reward_model.device)
            with torch.no_grad():
                score = reward_model(**reward_inputs).logits.squeeze().item()
            scores.append(score)

            if (i + 1) % 10 == 0:
                log.info(f"Generated {i + 1}/{len(prompts)}")

        return responses, scores

    # -- Base model --
    log.info(f"Evaluating base model: {model_name}")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running base model...</p>")
    await flyte.report.flush.aio()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, dtype=dtype, device_map="auto")
    base_responses, base_scores = generate_and_score(base_model, prompts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- PPO model --
    log.info("Evaluating PPO-trained model...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Running PPO-trained model...</p>")
    await flyte.report.flush.aio()

    ppo_path = await ppo_model_dir.download()
    ppo_model = AutoModelForCausalLM.from_pretrained(ppo_path, dtype=dtype, device_map="auto")
    ppo_responses, ppo_scores = generate_and_score(ppo_model, prompts)
    del ppo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Results --
    base_avg = sum(base_scores) / len(base_scores)
    ppo_avg = sum(ppo_scores) / len(ppo_scores)
    ppo_wins = sum(1 for b, p in zip(base_scores, ppo_scores) if p > b)
    win_rate = ppo_wins / len(prompts) * 100

    log.info(f"Base avg reward: {base_avg:.4f}")
    log.info(f"PPO avg reward: {ppo_avg:.4f}")
    log.info(f"PPO win rate: {win_rate:.1f}%")

    # -- Report --
    examples_html = ""
    for i in range(min(8, len(prompts))):
        prompt_preview = prompts[i][:300] + "..." if len(prompts[i]) > 300 else prompts[i]
        base_color = "green" if base_scores[i] > 0 else "red"
        ppo_color = "green" if ppo_scores[i] > 0 else "red"

        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p><b>Prompt:</b> {prompt_preview}</p>
<p><b>Base model</b> (reward: <span style="color:{base_color};">{base_scores[i]:.3f}</span>): {base_responses[i][:200]}</p>
<p><b>PPO model</b> (reward: <span style="color:{ppo_color};">{ppo_scores[i]:.3f}</span>): {ppo_responses[i][:200]}</p>
</div>"""

    await flyte.report.replace.aio(f"""
<h2>Evaluation Results — PPO vs Base</h2>
<table>
<tr><th></th><th>Avg Reward Score</th></tr>
<tr><td><b>Base model</b></td><td>{base_avg:.4f}</td></tr>
<tr><td><b>PPO-trained</b></td><td>{ppo_avg:.4f}</td></tr>
</table>
<p><b>PPO win rate:</b> {win_rate:.1f}% ({ppo_wins}/{len(prompts)} prompts scored higher)</p>
<hr/>
<h3>Example Responses</h3>
{examples_html}
""")
    await flyte.report.flush.aio()

    return json.dumps({
        "base_avg_reward": round(base_avg, 4),
        "ppo_avg_reward": round(ppo_avg, 4),
        "ppo_win_rate": round(win_rate, 1),
        "num_examples": len(prompts),
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "Anthropic/hh-rlhf",
    epochs: int = 2,
    ppo_epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    max_train_samples: int = 2000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 30,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """
    PPO fine-tuning pipeline — classic RLHF.

    1. Prepare preference data from Anthropic HH-RLHF
    2. Train a reward model on chosen/rejected pairs
    3. Optimize the policy with PPO against the reward model
    4. Evaluate: reward scores and generated responses before/after
    """
    log.info(f"Pipeline: {model_name} | PPO | dataset={dataset_name}")

    await flyte.report.replace.aio(
        f"<h2>PPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 1/4: Preparing data...</p>"
    )
    await flyte.report.flush.aio()

    data_dir = await prepare_data(dataset_name, max_train_samples, max_eval_samples)

    await flyte.report.replace.aio(
        f"<h2>PPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/4: Training reward model...</p>"
    )
    await flyte.report.flush.aio()

    reward_model_dir = await train_reward_model(
        model_name, data_dir, epochs, lr, batch_size, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        f"<h2>PPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/4: PPO training...</p>"
    )
    await flyte.report.flush.aio()

    ppo_model_dir = await train_ppo(
        model_name, reward_model_dir, data_dir,
        ppo_epochs, lr, batch_size, max_new_tokens, lora_r, lora_alpha,
    )

    await flyte.report.replace.aio(
        f"<h2>PPO Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 4/4: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    result = await evaluate(model_name, ppo_model_dir, reward_model_dir, data_dir, num_eval_examples)
    metrics = json.loads(result)

    await flyte.report.replace.aio(
        f"<h2>PPO Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Base avg reward:</b> {metrics['base_avg_reward']}</p>"
        f"<p><b>PPO avg reward:</b> {metrics['ppo_avg_reward']}</p>"
        f"<p><b>PPO win rate:</b> {metrics['ppo_win_rate']}%</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. PPO win rate: {metrics['ppo_win_rate']}%")
    return result
