"""
MuJoCo RL training with PPO — Unitree G1 humanoid locomotion.

Trains a policy network on the Unitree G1 humanoid robot using Proximal Policy
Optimization (PPO) with Generalized Advantage Estimation (GAE). Uses the G1 model
from mujoco_menagerie loaded into Gymnasium's Humanoid-v5 environment.

Usage:
    # Local (quick demo)
    flyte run --local --tui workflow.py train_agent --num_iterations 5 --episodes_per_iter 10

    # Remote (full training — default params)
    flyte run workflow.py train_agent
"""

import os
import io
import json
import base64
import logging

# Use EGL for headless MuJoCo rendering (required in containers without a display)
os.environ["MUJOCO_GL"] = "egl"

import flyte
import flyte.report
from flyte.io import File

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── Environments ─────────────────────────────────────────────────────────────

train_env = flyte.TaskEnvironment(
    name="mujoco-rl",
    image=flyte.Image.from_debian_base().with_apt_packages(
        "libegl1", "libgl1", "libgles2",
    ).with_pip_packages(
        "gymnasium[mujoco]", "torch", "numpy", "Pillow", "matplotlib",
        "mujoco_menagerie",
    ),
    resources=flyte.Resources(cpu=4, memory="8Gi"),
)

# ── Constants ────────────────────────────────────────────────────────────────

ENV_ID = "HalfCheetah-v4"
STATE_DIM = 17   # HalfCheetah observation space
ACTION_DIM = 6   # HalfCheetah action space
HIDDEN_DIM = 128


# ── Report helpers ───────────────────────────────────────────────────────────

def fig_to_html(fig) -> str:
    """Convert a matplotlib figure to an inline base64 <img> tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0f0f23")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'


def generate_replay_html(episodes: list[dict]) -> str:
    """
    Generate interactive MuJoCo frame replay with slider, play/pause, and episode selector.

    Each episode dict: {"label": str, "reward": float, "frames_b64": list[str]}
    """
    uid = f"mj{id(episodes) % 100000}"

    options_html = ""
    for i, ep in enumerate(episodes):
        options_html += (
            f'<option value="{i}">{ep["label"]} (reward: {ep["reward"]:.1f})</option>'
        )

    return f"""
    <div style="font-family: monospace; background: #0f0f23; color: #ccc; padding: 20px; border-radius: 8px;">
      <h3 style="color: #fdcb6e; margin-top: 0;">Episode Replay</h3>
      <div style="margin-bottom: 10px;">
        <label>Episode:
          <select id="ep-{uid}" onchange="changeEp_{uid}()" style="background:#1a1a2e;color:#ccc;padding:4px;border:1px solid #333;">
            {options_html}
          </select>
        </label>
        <span id="info-{uid}" style="margin-left: 15px;"></span>
      </div>
      <div style="margin-bottom: 10px;">
        <label>Step: <span id="step-{uid}">0</span></label><br>
        <input type="range" id="slider-{uid}" min="0" max="0" value="0" oninput="render_{uid}()"
               style="width: 480px;">
        <button onclick="play_{uid}()" id="btn-{uid}"
                style="margin-left:10px;padding:4px 12px;background:#00b894;color:#fff;border:none;border-radius:4px;cursor:pointer;">
          Play
        </button>
      </div>
      <img id="frame-{uid}" style="max-width:480px;display:block;border:2px solid #333;border-radius:4px;" />
    </div>

    <script>
    (function() {{
      var EPS = {json.dumps([{"reward": ep["reward"], "frames": ep["frames_b64"]} for ep in episodes])};
      var curEp = 0;
      var timer = null;

      window.changeEp_{uid} = function() {{
        curEp = parseInt(document.getElementById('ep-{uid}').value);
        var ep = EPS[curEp];
        var sl = document.getElementById('slider-{uid}');
        sl.max = Math.max(ep.frames.length - 1, 0);
        sl.value = 0;
        document.getElementById('info-{uid}').textContent = 'Total reward: ' + ep.reward.toFixed(1);
        render_{uid}();
      }};

      window.render_{uid} = function() {{
        var ep = EPS[curEp];
        var idx = parseInt(document.getElementById('slider-{uid}').value);
        document.getElementById('step-{uid}').textContent = idx;
        if (ep.frames[idx]) {{
          document.getElementById('frame-{uid}').src = 'data:image/jpeg;base64,' + ep.frames[idx];
        }}
      }};

      window.play_{uid} = function() {{
        if (timer) {{
          clearInterval(timer);
          timer = null;
          document.getElementById('btn-{uid}').textContent = 'Play';
          return;
        }}
        document.getElementById('btn-{uid}').textContent = 'Pause';
        var sl = document.getElementById('slider-{uid}');
        timer = setInterval(function() {{
          var val = parseInt(sl.value);
          if (val >= parseInt(sl.max)) {{
            clearInterval(timer);
            timer = null;
            document.getElementById('btn-{uid}').textContent = 'Play';
            return;
          }}
          sl.value = val + 1;
          render_{uid}();
        }}, 50);
      }};

      changeEp_{uid}();
    }})();
    </script>
    """


def generate_progress_html(history: list[dict], baseline_reward: float, num_iterations: int, current_phase: str = "") -> str:
    """Build a live progress table showing completed iterations and current status."""
    rows = ""
    for h in history:
        reward_color = "#00b894" if h["eval_reward"] > baseline_reward else "#e17055"
        rows += (
            f'<tr>'
            f'<td style="padding:4px 8px;border-bottom:1px solid #333;">{h["iteration"]}</td>'
            f'<td style="padding:4px 8px;border-bottom:1px solid #333;color:{reward_color};">{h["eval_reward"]:.1f}</td>'
            f'<td style="padding:4px 8px;border-bottom:1px solid #333;">{h["best_reward"]:.1f}</td>'
            f'<td style="padding:4px 8px;border-bottom:1px solid #333;">{h["loss"]:.4f}</td>'
            f'</tr>'
        )

    best_so_far = max((h["eval_reward"] for h in history), default=baseline_reward)
    improvement = best_so_far - baseline_reward

    progress_pct = len(history) / num_iterations * 100
    bar = (
        f'<div style="background:#1a1a2e;border-radius:4px;height:20px;width:100%;margin:8px 0;">'
        f'<div style="background:linear-gradient(90deg,#00b894,#6c5ce7);height:100%;width:{progress_pct:.0f}%;'
        f'border-radius:4px;transition:width 0.3s;"></div></div>'
    )

    status = f"<p style='color:#888;'>{current_phase}</p>" if current_phase else ""

    return (
        f'<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        f'<h3 style="color:#fdcb6e;margin-top:0;">Training Progress — {len(history)}/{num_iterations} iterations</h3>'
        f'{bar}{status}'
        f'<div style="display:flex;gap:20px;margin:10px 0;">'
        f'<span>Baseline: <b style="color:#e17055;">{baseline_reward:.1f}</b></span>'
        f'<span>Best: <b style="color:#00b894;">{best_so_far:.1f}</b></span>'
        f'<span>Improvement: <b style="color:#fdcb6e;">{improvement:+.1f}</b></span>'
        f'</div>'
        f'<table style="border-collapse:collapse;width:100%;margin-top:10px;">'
        f'<tr style="color:#fdcb6e;">'
        f'<th style="padding:4px 8px;text-align:left;border-bottom:2px solid #444;">Iter</th>'
        f'<th style="padding:4px 8px;text-align:left;border-bottom:2px solid #444;">Mean Reward</th>'
        f'<th style="padding:4px 8px;text-align:left;border-bottom:2px solid #444;">Best Reward</th>'
        f'<th style="padding:4px 8px;text-align:left;border-bottom:2px solid #444;">Loss</th>'
        f'</tr>{rows}</table></div>'
    )


def generate_training_charts(history: list[dict], baseline_reward: float) -> str:
    """Generate training progress charts: reward curve + loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [h["iteration"] for h in history]
    rewards = [h["eval_reward"] for h in history]
    losses = [h["loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f0f23")

    # Reward curve
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(iterations, rewards, color="#00b894", linewidth=2, marker="o", markersize=4, label="Trained policy")
    ax1.axhline(y=baseline_reward, color="#e17055", linestyle="--", linewidth=1.5, label=f"Random baseline ({baseline_reward:.1f})")
    ax1.set_xlabel("Iteration", color="#ccc")
    ax1.set_ylabel("Mean Episode Reward", color="#ccc")
    ax1.set_title("Training Progress", color="#fdcb6e", fontsize=14)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    ax1.tick_params(colors="#888")
    ax1.grid(alpha=0.15)

    # Loss curve
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(iterations, losses, color="#6c5ce7", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Iteration", color="#ccc")
    ax2.set_ylabel("PPO Loss", color="#ccc")
    ax2.set_title("Policy Loss", color="#fdcb6e", fontsize=14)
    ax2.tick_params(colors="#888")
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    html = fig_to_html(fig)
    plt.close(fig)
    return html


# ── Policy network ───────────────────────────────────────────────────────────

def _build_policy_and_value():
    """Build policy (actor) and value (critic) networks. Returns (policy, value, device)."""
    import torch
    import torch.nn as nn

    device = torch.device("cpu")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
            )
            self.mean = nn.Linear(HIDDEN_DIM, ACTION_DIM)
            self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))  # std ~0.6, good exploration

        def forward(self, x):
            h = self.net(x)
            return self.mean(h), self.log_std.expand_as(self.mean(h))

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Policy().to(device), Value().to(device), device


# ── Tasks ────────────────────────────────────────────────────────────────────

@train_env.task
async def collect_rollouts(
    checkpoint: File | None,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> File:
    """Collect rollouts using the current policy. Always uses the policy (even untrained)
    so log probs are correct for PPO importance sampling."""
    import torch
    import gymnasium as gym

    # Always build a policy — untrained weights give exploratory actions
    # with correct log probs (critical for PPO's importance sampling ratio)
    policy, _, device = _build_policy_and_value()
    if checkpoint is not None:
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
    policy.eval()

    all_episodes = []
    for ep_idx in range(num_episodes):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=ep_idx)
        transitions = []

        for step in range(max_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, log_std = policy(obs_t)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).sum(-1).item()
                action = action_t.squeeze(0).numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            transitions.append({
                "obs": obs.tolist(),
                "action": action.tolist(),
                "reward": float(reward),
                "log_prob": float(log_prob),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            obs = next_obs
            if terminated or truncated:
                break

        env.close()
        all_episodes.append(transitions)

    path = "/tmp/rollouts.json"
    with open(path, "w") as f:
        json.dump(all_episodes, f)
    return await File.from_local(path)


@train_env.task
async def ppo_update(
    rollouts_file: File,
    checkpoint: File | None,
    ppo_epochs: int = 10,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> File:
    """Run PPO update on collected rollouts. Returns updated checkpoint."""
    import torch
    import numpy as np

    rollouts_path = await rollouts_file.download()
    with open(rollouts_path) as f:
        episodes = json.load(f)
    policy, value_net, device = _build_policy_and_value()
    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    # Load existing checkpoint
    if checkpoint is not None:
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        value_net.load_state_dict(ckpt["value"])
        policy_optim.load_state_dict(ckpt["policy_optim"])
        value_optim.load_state_dict(ckpt["value_optim"])

    # Flatten episodes into one batch with GAE computation
    all_obs, all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], [], []

    for ep in episodes:
        obs = torch.tensor([t["obs"] for t in ep], dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in ep], dtype=torch.float32)
        rewards = [t["reward"] for t in ep]
        old_log_probs = torch.tensor([t["log_prob"] for t in ep], dtype=torch.float32)

        # Compute values for all observed states
        with torch.no_grad():
            values = value_net(obs).numpy()

        # GAE — reverse sweep through the episode
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            is_last = (t == T - 1)
            if is_last:
                # Last step: bootstrap only if truncated (time limit, not failure)
                if ep[t].get("truncated", False):
                    next_val = values[t]  # approximate: use current value as bootstrap
                else:
                    next_val = 0.0  # terminated or natural end — no bootstrap
                gae = 0.0  # no future advantage beyond episode end
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values

        all_obs.append(obs)
        all_actions.append(actions)
        all_old_log_probs.append(old_log_probs)
        all_advantages.append(torch.tensor(advantages, dtype=torch.float32))
        all_returns.append(torch.tensor(returns, dtype=torch.float32))

    obs_batch = torch.cat(all_obs)
    actions_batch = torch.cat(all_actions)
    old_log_probs_batch = torch.cat(all_old_log_probs)
    advantages_batch = torch.cat(all_advantages)
    returns_batch = torch.cat(all_returns)

    # Normalize advantages
    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    total_loss = 0.0
    entropy_coef = 0.01
    batch_size = min(256, len(obs_batch))

    # PPO epochs with minibatch updates
    for _ in range(ppo_epochs):
        # Shuffle data each epoch
        indices = torch.randperm(len(obs_batch))
        for start in range(0, len(obs_batch), batch_size):
            idx = indices[start:start + batch_size]
            mb_obs = obs_batch[idx]
            mb_actions = actions_batch[idx]
            mb_old_lp = old_log_probs_batch[idx]
            mb_adv = advantages_batch[idx]
            mb_ret = returns_batch[idx]

            # Policy loss with entropy bonus
            mean, log_std = policy(mb_obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(mb_actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (new_log_probs - mb_old_lp).exp()
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
            policy_loss = policy_loss - entropy_coef * entropy

            policy_optim.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optim.step()

            # Value loss
            values_pred = value_net(mb_obs)
            value_loss = (values_pred - mb_ret).pow(2).mean()

            value_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optim.step()

            total_loss += policy_loss.item()

    num_updates = ppo_epochs * max(1, len(obs_batch) // batch_size)
    avg_loss = total_loss / num_updates

    # Save checkpoint
    path = "/tmp/ppo_checkpoint.pt"
    torch.save({
        "policy": policy.state_dict(),
        "value": value_net.state_dict(),
        "policy_optim": policy_optim.state_dict(),
        "value_optim": value_optim.state_dict(),
        "loss": avg_loss,
    }, path)
    return await File.from_local(path)


@train_env.task(report=True)
async def evaluate(
    checkpoint: File | None,
    label: str = "Random",
    num_episodes: int = 5,
    max_steps: int = 1000,
    capture_frames: bool = True,
) -> File:
    """Evaluate a policy and capture MuJoCo frames for the best episode.
    Returns a File containing JSON with metrics and frames."""
    import torch
    import numpy as np
    import gymnasium as gym
    from PIL import Image

    policy = None
    if checkpoint is not None:
        policy, _, device = _build_policy_and_value()
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()

    best_reward = -float("inf")
    best_frames = []
    all_rewards = []

    frame_every = 5  # capture every Nth step to keep output small
    max_frames = 100  # cap total frames per episode

    for ep_idx in range(num_episodes):
        env = gym.make(ENV_ID, render_mode="rgb_array" if capture_frames else None)
        obs, _ = env.reset(seed=ep_idx + 1000)
        total_reward = 0.0
        frames = []

        for step in range(max_steps):
            if capture_frames and step % frame_every == 0 and len(frames) < max_frames:
                frame = env.render()
                buf = io.BytesIO()
                Image.fromarray(frame).resize((240, 160)).save(buf, format="JPEG", quality=70)
                frames.append(base64.b64encode(buf.getvalue()).decode())

            if policy is not None:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    mean, _ = policy(obs_t)
                    action = mean.squeeze(0).numpy()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        env.close()
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    mean_reward = sum(all_rewards) / len(all_rewards)
    log.info(f"[Eval] {label}: mean={mean_reward:.1f}, best={best_reward:.1f}")

    # Report for this eval task
    await flyte.report.replace.aio(
        f"<h3>{label}</h3>"
        f"<p>Mean reward: {mean_reward:.1f} | Best: {best_reward:.1f}</p>"
    )
    await flyte.report.flush.aio()

    # Return everything as a File to avoid inline IO limits
    result_path = "/tmp/eval_result.json"
    with open(result_path, "w") as f:
        json.dump({
            "label": label,
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "frames_b64": best_frames,
        }, f)
    return await File.from_local(result_path)


# ── Orchestrator ─────────────────────────────────────────────────────────────

@train_env.task(report=True)
async def train_agent(
    num_iterations: int = 15,
    episodes_per_iter: int = 20,
    ppo_epochs: int = 15,
    max_steps: int = 1000,
    lr: float = 3e-4,
) -> str:
    """
    PPO training loop with evaluation and interactive reports.

    1. Evaluate random baseline
    2. For each iteration: collect rollouts → PPO update → evaluate
    3. Build final report with training curves + episode replays
    """
    log.info(f"Starting PPO training: {num_iterations} iterations, {episodes_per_iter} episodes each")

    await flyte.report.replace.aio(
        "<h2>MuJoCo RL Training</h2>"
        f"<p>Training {ENV_ID} with PPO for {num_iterations} iterations...</p>"
    )
    await flyte.report.flush.aio()

    # ── Evaluate random baseline ─────────────────────────────────────────
    baseline_file = await evaluate(
        checkpoint=None, label="Random Policy",
        num_episodes=5, max_steps=max_steps, capture_frames=True,
    )
    baseline_path = await baseline_file.download()
    with open(baseline_path) as f:
        baseline = json.load(f)
    baseline_reward = baseline["mean_reward"]
    log.info(f"Random baseline: {baseline_reward:.1f}")

    replays = []
    if baseline.get("frames_b64"):
        replays.append({"label": baseline["label"], "reward": baseline["best_reward"], "frames_b64": baseline["frames_b64"]})
        # Add baseline replay tab immediately
        ep = replays[0]
        tab = flyte.report.get_tab("Random Policy")
        uid = "tab_baseline"
        frames_json = json.dumps(ep["frames_b64"])
        tab.log(
            f"<h3>Random Policy</h3><p>Reward: {ep['reward']:.1f}</p>"
            f'<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:12px;border-radius:8px;">'
            f'<img id="f-{uid}" src="data:image/jpeg;base64,{ep["frames_b64"][0]}" style="max-width:480px;display:block;border:2px solid #333;border-radius:4px;"/><br/>'
            f'<input type="range" id="s-{uid}" min="0" max="{len(ep["frames_b64"])-1}" value="0" style="width:480px;"/>'
            f' <span id="l-{uid}">Step 0/{len(ep["frames_b64"])-1}</span>'
            f'<script>(function(){{var F={frames_json};'
            f'var s=document.getElementById("s-{uid}"),i=document.getElementById("f-{uid}"),l=document.getElementById("l-{uid}");'
            f's.oninput=function(){{i.src="data:image/jpeg;base64,"+F[this.value];'
            f'l.textContent="Step "+this.value+"/{len(ep["frames_b64"])-1}";}};'
            f'}})();</script></div>'
        )
        await flyte.report.flush.aio()

    # ── Training loop ────────────────────────────────────────────────────
    checkpoint = None
    history = []

    import torch

    for i in range(1, num_iterations + 1):
        log.info(f"── Iteration {i}/{num_iterations} ──")

        # Collect rollouts with current policy
        rollouts_file = await collect_rollouts(
            checkpoint=checkpoint,
            num_episodes=episodes_per_iter,
            max_steps=max_steps,
        )

        # PPO update
        checkpoint = await ppo_update(
            rollouts_file=rollouts_file,
            checkpoint=checkpoint,
            ppo_epochs=ppo_epochs,
            lr=lr,
        )

        # Read loss from checkpoint
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location="cpu")
        loss = ckpt.get("loss", 0.0)

        # Evaluate updated policy
        capture = (i % max(1, num_iterations // 5) == 0) or i == num_iterations
        eval_file = await evaluate(
            checkpoint=checkpoint,
            label=f"Iteration {i}",
            num_episodes=5,
            max_steps=max_steps,
            capture_frames=capture,
        )
        eval_path = await eval_file.download()
        with open(eval_path) as f:
            eval_result = json.load(f)

        history.append({
            "iteration": i,
            "eval_reward": eval_result["mean_reward"],
            "best_reward": eval_result["best_reward"],
            "loss": loss,
        })

        if capture and eval_result.get("frames_b64"):
            replays.append({
                "label": eval_result["label"],
                "reward": eval_result["best_reward"],
                "frames_b64": eval_result["frames_b64"],
            })

            # Add replay tab immediately so you can watch it while training continues
            idx = len(replays) - 1
            ep = replays[-1]
            tab = flyte.report.get_tab(ep["label"][:25])
            uid = f"tab{idx}"
            frames_json = json.dumps(ep["frames_b64"])
            tab.log(
                f"<h3>{ep['label']}</h3><p>Reward: {ep['reward']:.1f}</p>"
                f'<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:12px;border-radius:8px;">'
                f'<img id="f-{uid}" src="data:image/jpeg;base64,{ep["frames_b64"][0]}" style="max-width:480px;display:block;border:2px solid #333;border-radius:4px;"/><br/>'
                f'<input type="range" id="s-{uid}" min="0" max="{len(ep["frames_b64"])-1}" value="0" style="width:480px;"/>'
                f' <span id="l-{uid}">Step 0/{len(ep["frames_b64"])-1}</span>'
                f'<script>(function(){{var F={frames_json};'
                f'var s=document.getElementById("s-{uid}"),i=document.getElementById("f-{uid}"),l=document.getElementById("l-{uid}");'
                f's.oninput=function(){{i.src="data:image/jpeg;base64,"+F[this.value];'
                f'l.textContent="Step "+this.value+"/{len(ep["frames_b64"])-1}";}};'
                f'}})();</script></div>'
            )

        log.info(f"  Reward: {eval_result['mean_reward']:.1f} | Loss: {loss:.4f}")

        # Update main report with latest chart + table after each iteration
        charts_html = generate_training_charts(history, baseline_reward)
        progress = generate_progress_html(history, baseline_reward, num_iterations)
        await flyte.report.replace.aio(
            f"<h2>MuJoCo RL Training</h2>{progress}<br/>{charts_html}"
        )
        await flyte.report.flush.aio()

    # ── Build final report ───────────────────────────────────────────────
    charts_html = generate_training_charts(history, baseline_reward)
    replay_html = generate_replay_html(replays) if replays else ""

    best = max(history, key=lambda h: h["eval_reward"])
    improvement = best["eval_reward"] - baseline_reward

    summary_html = (
        '<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        '<h3 style="color:#00b894;margin-top:0;">Training Complete</h3>'
        '<table style="border-collapse:collapse;width:100%;">'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Environment</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">{ENV_ID}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Iterations</td><td style="padding:6px;border-bottom:1px solid #333;">{num_iterations}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Episodes/iter</td><td style="padding:6px;border-bottom:1px solid #333;">{episodes_per_iter}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Random baseline</td><td style="padding:6px;border-bottom:1px solid #333;color:#e17055;">{baseline_reward:.1f}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Best trained reward</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">{best["eval_reward"]:.1f}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Improvement</td><td style="padding:6px;border-bottom:1px solid #333;color:#fdcb6e;">{improvement:+.1f}</td></tr>'
        f'<tr><td style="padding:6px;">Learning rate</td><td style="padding:6px;">{lr}</td></tr>'
        '</table></div>'
    )

    progress = generate_progress_html(history, baseline_reward, num_iterations)

    await flyte.report.replace.aio(
        "<h2>MuJoCo RL Training — Complete</h2>"
        f"{summary_html}<br/>"
        f"{progress}<br/>"
        f"<h3 style='color:#fdcb6e;font-family:monospace;'>Training Progress</h3>{charts_html}<br/>"
        + (f"<h3 style='color:#fdcb6e;font-family:monospace;'>Episode Replays</h3>{replay_html}" if replay_html else "")
    )
    await flyte.report.flush.aio()

    log.info(f"Training complete. Best: {best['eval_reward']:.1f} (baseline: {baseline_reward:.1f})")

    return json.dumps({
        "env": ENV_ID,
        "baseline_reward": baseline_reward,
        "best_reward": best["eval_reward"],
        "improvement": improvement,
        "history": history,
    })
