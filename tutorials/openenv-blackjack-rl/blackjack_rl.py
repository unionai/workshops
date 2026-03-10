"""Train an LLM to play Blackjack via GRPO using OpenEnv + Flyte.

Demonstrates:
- OpenEnv's Gymnasium-style API (reset / step / state)
- Group Relative Policy Optimization (GRPO) training loop
- Eval tracking that shows win-rate improvement over training
- Baseline comparison (random policy vs. heuristic policy)
- Flyte HTML reports for visualization

Run locally:
  flyte run --local --tui blackjack_rl.py pipeline --training_steps 3

Run on a cluster:
  flyte run blackjack_rl.py pipeline --training_steps 10
"""

import base64
import io
import json
import os
import random
import shutil

import flyte
import flyte.report
from flyte.io import File

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="blackjack_rl",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch",
        "transformers",
        "openenv-core",
        "git+https://huggingface.co/spaces/openenv/openspiel_env",
        "matplotlib",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENV_URL = "https://openenv-openspiel-env.hf.space"

SYSTEM_PROMPT = (
    "You are playing Blackjack. You will see your hand information and legal actions.\n"
    "You MUST respond with exactly one word: HIT or STAND.\n"
    "- HIT means take another card.\n"
    "- STAND means keep your current hand.\n"
    "Basic strategy: HIT when your hand total is below 17, STAND on 17 or above."
)

ACTION_MAP = {0: "HIT", 1: "STAND"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


def format_observation(obs) -> str:
    """Convert an OpenSpiel observation into a natural-language prompt."""
    actions = ", ".join(ACTION_MAP.get(a, str(a)) for a in obs.legal_actions)
    return (
        f"Hand state: {obs.info_state}\n"
        f"Game phase: {obs.game_phase}\n"
        f"Legal actions: {actions}\n"
        "What do you do? Reply with HIT or STAND."
    )


def parse_action(text: str) -> int:
    """Return 1 (STAND) if the model says STAND, else 0 (HIT)."""
    return 1 if "STAND" in text.strip().upper() else 0


def create_env_client(env_url: str):
    """Create a connected OpenSpiel client with generous timeouts."""
    from openspiel_env import OpenSpielEnv

    client = OpenSpielEnv(
        base_url=env_url,
        connect_timeout_s=30.0,
        message_timeout_s=300.0,  # 5 min — CPU inference is slow
    )
    client.connect()
    return client


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


@env.task(cache="auto")
async def eval_baselines(env_url: str = DEFAULT_ENV_URL, num_episodes: int = 200) -> str:
    """Play blackjack with random and heuristic policies to set baselines."""
    from openspiel_env import OpenSpielAction

    results = {}
    for policy in ["random", "heuristic"]:
        wins = losses = draws = 0
        client = create_env_client(env_url)
        try:
            for _ in range(num_episodes):
                try:
                    result = client.reset()
                except Exception:
                    client.connect()
                    result = client.reset()
                while not result.done:
                    obs = result.observation
                    if policy == "random":
                        action_id = random.choice(obs.legal_actions)
                    else:
                        # Stand on 17+: use sum of info_state as a rough proxy
                        hand_signal = sum(obs.info_state)
                        action_id = 1 if hand_signal >= 17 else 0
                        if action_id not in obs.legal_actions:
                            action_id = obs.legal_actions[0]
                    result = client.step(
                        OpenSpielAction(action_id=action_id, game_name="blackjack")
                    )
                reward = result.reward or 0
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
        finally:
            client.close()

        results[policy] = {
            "win_rate": wins / num_episodes,
            "loss_rate": losses / num_episodes,
            "draw_rate": draws / num_episodes,
        }
        print(f"  {policy}: win={results[policy]['win_rate']:.3f}")

    return json.dumps(results)


# ---------------------------------------------------------------------------
# Model interaction helpers
# ---------------------------------------------------------------------------


def play_episode(client, model, tokenizer, device, temperature=0.7):
    """Play one blackjack episode with the LLM. Returns (trajectory, shaped_reward)."""
    import torch
    from openspiel_env import OpenSpielAction

    # Reconnect if the WebSocket was dropped (HF Space idle timeout)
    try:
        result = client.reset()
    except Exception:
        client.connect()
        result = client.reset()
    trajectory = []

    while not result.done:
        obs = result.observation
        user_prompt = format_observation(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Log-prob of generated tokens
        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(gen_ids):
                lp = torch.log_softmax(score[0], dim=-1)
                log_probs.append(lp[gen_ids[i]].item())

        action_id = parse_action(gen_text)
        if action_id not in obs.legal_actions:
            action_id = obs.legal_actions[0]

        result = client.step(
            OpenSpielAction(action_id=action_id, game_name="blackjack")
        )

        trajectory.append(
            {
                "prompt_ids": inputs["input_ids"][0].tolist(),
                "completion_ids": gen_ids.tolist(),
                "log_probs": log_probs,
                "action": gen_text.strip(),
            }
        )

    # Reward shaping
    raw = result.reward or 0
    if raw > 0:
        shaped = 2.0
    elif raw == 0:
        shaped = 0.5
    else:
        shaped = -1.0

    return trajectory, shaped


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------


@env.task
async def train_step(
    model_path: str,
    env_url: str = DEFAULT_ENV_URL,
    num_rollouts: int = 16,
    group_size: int = 4,
    lr: float = 1e-5,
    step_idx: int = 0,
) -> tuple[File, str]:
    """One GRPO iteration: collect rollouts, compute group advantages, update policy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    print(f"  Step {step_idx} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    client = create_env_client(env_url)

    all_rewards = []
    total_loss = 0.0
    num_groups = max(num_rollouts // group_size, 1)

    try:
        for g in range(num_groups):
            group_trajs = []
            group_rewards = []

            for _ in range(group_size):
                traj, reward = play_episode(client, model, tokenizer, device)
                group_trajs.append(traj)
                group_rewards.append(reward)

            all_rewards.extend(group_rewards)

            # Group-relative advantages
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (
                sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)
            ) ** 0.5
            std_r = max(std_r, 1e-8)
            advantages = [(r - mean_r) / std_r for r in group_rewards]

            # Policy gradient
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for traj, adv in zip(group_trajs, advantages):
                for step_data in traj:
                    prompt_t = torch.tensor(
                        [step_data["prompt_ids"]], device=device
                    )
                    comp_t = torch.tensor(
                        [step_data["completion_ids"]], device=device
                    )
                    full_ids = torch.cat([prompt_t, comp_t], dim=1)

                    out = model(full_ids)
                    logits = out.logits[0, prompt_t.shape[1] - 1 : -1]
                    lp = torch.log_softmax(logits, dim=-1)
                    token_lp = lp.gather(
                        1, comp_t[0].unsqueeze(1)
                    ).squeeze(1)
                    batch_loss = batch_loss - token_lp.sum() * adv

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
    finally:
        client.close()

    # Metrics
    win_rate = sum(1 for r in all_rewards if r > 1.0) / len(all_rewards)
    avg_reward = sum(all_rewards) / len(all_rewards)

    # Save checkpoint
    save_dir = f"checkpoint_step_{step_idx}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Tar the directory for Flyte File transport
    tar_path = f"{save_dir}.tar.gz"
    shutil.make_archive(save_dir, "gztar", ".", save_dir)
    checkpoint_file = await File.from_local(tar_path)

    metrics = json.dumps(
        {
            "step": step_idx,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "loss": total_loss / num_groups,
        }
    )
    print(f"  Step {step_idx} | win_rate={win_rate:.3f} avg_reward={avg_reward:.2f}")
    return checkpoint_file, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@env.task
async def eval_model(
    model_path: str,
    env_url: str = DEFAULT_ENV_URL,
    num_episodes: int = 100,
    step_idx: int = 0,
) -> str:
    """Evaluate the model on blackjack, tracking win rate and action distribution."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    client = create_env_client(env_url)
    wins = losses = draws = 0
    action_counts = {"HIT": 0, "STAND": 0}

    try:
        for _ in range(num_episodes):
            traj, reward = play_episode(
                client, model, tokenizer, device, temperature=0.0
            )
            if reward > 1.0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
            for s in traj:
                key = "STAND" if "STAND" in s["action"].upper() else "HIT"
                action_counts[key] += 1
    finally:
        client.close()

    return json.dumps(
        {
            "step": step_idx,
            "win_rate": wins / num_episodes,
            "loss_rate": losses / num_episodes,
            "draw_rate": draws / num_episodes,
            "action_distribution": action_counts,
        }
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@env.task(report=True)
async def pipeline(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    env_url: str = DEFAULT_ENV_URL,
    training_steps: int = 5,
    rollouts_per_step: int = 16,
    group_size: int = 4,
    eval_episodes: int = 100,
    lr: float = 1e-5,
    open_report: bool = False,
) -> tuple[str, File]:
    """Full RL pipeline: baselines -> GRPO training -> eval at each checkpoint -> report."""

    device = get_device()
    print(f"Device: {device}")
    print(f"Model:  {model_id}\n")

    # 1. Baselines
    print("=== Evaluating baselines ===")
    baselines = json.loads(await eval_baselines(env_url, num_episodes=200))
    print(f"  Random:    {baselines['random']['win_rate']:.3f}")
    print(f"  Heuristic: {baselines['heuristic']['win_rate']:.3f}")

    # 2. Evaluate untrained model
    print("\n=== Evaluating untrained model ===")
    eval_results = [json.loads(await eval_model(model_id, env_url, eval_episodes, 0))]
    print(f"  Untrained: {eval_results[0]['win_rate']:.3f}")

    # 3. Training loop
    current_model = model_id
    train_metrics = []

    for step in range(1, training_steps + 1):
        print(f"\n=== Training step {step}/{training_steps} ===")

        checkpoint_file, metrics_json = await train_step(
            current_model,
            env_url,
            num_rollouts=rollouts_per_step,
            group_size=group_size,
            lr=lr,
            step_idx=step,
        )
        train_metrics.append(json.loads(metrics_json))

        # Download checkpoint, extract, and evaluate
        local_tar = await checkpoint_file.download()
        extract_dir = f"eval_checkpoint_{step}"
        shutil.unpack_archive(local_tar, extract_dir)
        checkpoint_dir = os.path.join(extract_dir, f"checkpoint_step_{step}")

        eval_json = await eval_model(checkpoint_dir, env_url, eval_episodes, step)
        eval_results.append(json.loads(eval_json))
        current_model = checkpoint_dir
        print(f"  Eval win rate: {eval_results[-1]['win_rate']:.3f}")

    # 4. Generate report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in eval_results]
    win_rates = [e["win_rate"] for e in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Win rate over training
    ax = axes[0]
    ax.plot(steps, win_rates, "b-o", markersize=6, linewidth=2, label="GRPO Agent")
    ax.axhline(
        baselines["random"]["win_rate"],
        color="r", linestyle="--",
        label=f"Random ({baselines['random']['win_rate']:.3f})",
    )
    ax.axhline(
        baselines["heuristic"]["win_rate"],
        color="g", linestyle="--",
        label=f"Heuristic ({baselines['heuristic']['win_rate']:.3f})",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    # Chart 2: Training reward
    ax = axes[1]
    if train_metrics:
        ax.plot(
            [m["step"] for m in train_metrics],
            [m["avg_reward"] for m in train_metrics],
            "m-o", markersize=6,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Shaped Reward")
    ax.set_title("Training Reward")
    ax.grid(True, alpha=0.3)

    # Chart 3: Action distribution
    ax = axes[2]
    hit_fracs = []
    for e in eval_results:
        dist = e.get("action_distribution", {})
        total = dist.get("HIT", 0) + dist.get("STAND", 0)
        hit_fracs.append(dist.get("HIT", 0) / max(total, 1))
    ax.plot(steps, hit_fracs, "c-o", markersize=6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("HIT Fraction")
    ax.set_title("Action Distribution")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Blackjack RL Training Report</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Win Rate</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['win_rate']:.3f}</td></tr>"
        f"<tr><td>Heuristic (stand on 17)</td><td>{baselines['heuristic']['win_rate']:.3f}</td></tr>"
        f"<tr><td><b>GRPO Agent (step 0)</b></td><td>{eval_results[0]['win_rate']:.3f}</td></tr>"
        f"<tr><td><b>GRPO Agent (final)</b></td><td><b>{final['win_rate']:.3f}</b></td></tr>"
        f"</table>"
        f"<h3>Training Progress</h3>{charts_html}"
        f"<h3>Config</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td>Model</td><td>{model_id}</td></tr>"
        f"<tr><td>Training Steps</td><td>{training_steps}</td></tr>"
        f"<tr><td>Rollouts/Step</td><td>{rollouts_per_step}</td></tr>"
        f"<tr><td>Group Size</td><td>{group_size}</td></tr>"
        f"<tr><td>Learning Rate</td><td>{lr}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    task_ctx = flyte.ctx()
    if task_ctx:
        from flyte._internal.runtime import io as flyte_io

        report_path = flyte_io.report_path(task_ctx.output_path)
        print(f"\nReport: {report_path}")
        if open_report:
            import webbrowser

            webbrowser.open(f"file://{report_path}")

    summary = (
        f"Final win rate: {final['win_rate']:.3f} "
        f"(random: {baselines['random']['win_rate']:.3f}, "
        f"heuristic: {baselines['heuristic']['win_rate']:.3f})"
    )
    return summary, checkpoint_file


# Local:  flyte run --local --tui blackjack_rl.py pipeline --training_steps 3
# Remote: flyte run blackjack_rl.py pipeline --training_steps 10
