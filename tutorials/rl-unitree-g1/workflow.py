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
import sys
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

import flyte
import flyte.report
from flyte.io import File

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── Environments ─────────────────────────────────────────────────────────────

train_env = flyte.TaskEnvironment(
    name="unitree-g1-rl",
    image=flyte.Image.from_debian_base().with_apt_packages(
        "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libgles2", "libglx-mesa0", "git",
    ).with_pip_packages(
        "gymnasium[mujoco]", "torch", "numpy", "Pillow", "matplotlib", "unionai-reuse",
    ).with_commands([
        "git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie",
    ]),
    resources=flyte.Resources(cpu=8, memory="16Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=10,
        scaledown_ttl=60,
    ),
)

# ── Constants ────────────────────────────────────────────────────────────────

HIDDEN_DIM = 256

# G1 env config — passed to Humanoid-v5 with the G1 XML
G1_ENV_KWARGS = {
    "healthy_z_range": (0.6, 1.4),      # G1 pelvis at ~0.79m; tight range prevents "almost fallen = healthy"
    "healthy_reward": 0.0,               # disable built-in alive bonus — custom wrapper handles it
    "forward_reward_weight": 0.0,        # disable built-in forward reward — custom wrapper handles it
    "ctrl_cost_weight": 0.0,             # disable built-in ctrl cost — custom wrapper handles it
    "contact_cost_weight": 0.0,          # disable built-in contact cost
    "reset_noise_scale": 0.005,          # very small noise so it starts near standing
    "terminate_when_unhealthy": True,
    "exclude_current_positions_from_observation": True,
    "include_cfrc_ext_in_observation": True,
    "frame_skip": 4,                     # finer control for complex humanoid
}

# Target walking speed in m/s
TARGET_VELOCITY = 0.8


class G1RewardWrapper:
    """Custom reward wrapper for Unitree G1 walking.

    Replaces the default Gymnasium reward with terms from MuJoCo Playground
    and legged_gym that produce real walking:
    - Velocity tracking (exponential kernel at target speed)
    - Alive bonus (small)
    - Termination penalty (large)
    - Vertical velocity penalty (don't fall)
    - Angular velocity penalty (don't tumble)
    - Action rate penalty (smooth movements)
    - Orientation penalty (stay upright)
    """

    def __init__(self, env):
        self.env = env
        # Expose gym.Env interface for SyncVectorEnv compatibility
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = getattr(env, "spec", None)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.np_random = getattr(env, "np_random", None)
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        import numpy as np
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        # Access MuJoCo state for custom rewards
        data = self.env.unwrapped.data

        # 1. Velocity tracking — exponential kernel peaked at target speed
        x_vel = data.qvel[0]
        vel_error = (x_vel - TARGET_VELOCITY) ** 2
        tracking_reward = 1.5 * np.exp(-vel_error / 0.25)

        # 2. Small alive bonus
        alive_bonus = 0.5

        # 3. Termination penalty — make falling very costly
        death_penalty = -50.0 if terminated else 0.0

        # 4. Vertical velocity penalty — don't bounce or fall
        z_vel = data.qvel[2]
        z_vel_penalty = -2.0 * z_vel ** 2

        # 5. Angular velocity penalty — don't tumble (roll/pitch)
        ang_vel_xy = data.qvel[3:5]
        ang_vel_penalty = -0.15 * np.sum(ang_vel_xy ** 2)

        # 6. Action rate penalty — smooth actions = smooth walking
        if self.prev_action is not None:
            action_rate_penalty = -0.01 * np.sum((action - self.prev_action) ** 2)
        else:
            action_rate_penalty = 0.0
        self.prev_action = action.copy()

        # 7. Orientation penalty — penalize torso tilt from upright
        #    qpos[3:7] is torso quaternion (w, x, y, z); upright means w~1, x~0, y~0
        quat = data.qpos[3:7]
        # Tilt = 1 - w^2 (0 when upright, ~1 when sideways/fallen)
        tilt = 1.0 - quat[0] ** 2
        orientation_penalty = -2.0 * tilt

        # 8. Control cost — moderate penalty for large torques
        ctrl_penalty = -0.01 * np.sum(action ** 2)

        reward = (
            tracking_reward
            + alive_bonus
            + death_penalty
            + z_vel_penalty
            + ang_vel_penalty
            + action_rate_penalty
            + orientation_penalty
            + ctrl_penalty
        )

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


def _make_env(render_mode=None):
    """Create the Unitree G1 environment with custom reward shaping."""
    import gymnasium as gym

    xml_path = "/opt/mujoco_menagerie/unitree_g1/scene.xml"
    env = gym.make(
        "Humanoid-v5",
        xml_file=xml_path,
        render_mode=render_mode,
        **G1_ENV_KWARGS,
    )
    return G1RewardWrapper(env)


def _get_dims():
    """Get observation and action dimensions from the G1 environment."""
    env = _make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    return state_dim, action_dim


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
    ax1.set_title("Unitree G1 — Training Progress", color="#fdcb6e", fontsize=14)
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


# ── Observation normalizer ────────────────────────────────────────────────────

class ObsNormalizer:
    """Running mean/std normalizer for observations. Stabilizes training by
    ensuring all inputs to the policy network are roughly zero-mean, unit-variance."""

    def __init__(self, dim: int):
        import numpy as np
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4  # avoid division by zero

    def update(self, batch):
        """Update running stats with a batch of observations (N x dim)."""
        import numpy as np
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        # Welford's online algorithm for combining stats
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, obs):
        """Normalize observations using running stats."""
        import numpy as np
        return (np.asarray(obs) - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, d):
        import numpy as np
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = d["count"]


# ── Policy network ───────────────────────────────────────────────────────────

def _build_policy_and_value(state_dim: int, action_dim: int):
    """Build policy (actor) and value (critic) networks. Returns (policy, value, device)."""
    import torch
    import torch.nn as nn

    device = torch.device("cpu")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
            )
            self.mean = nn.Linear(HIDDEN_DIM, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

        def forward(self, x):
            h = self.net(x)
            return self.mean(h), self.log_std.expand_as(self.mean(h))

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
                nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Policy().to(device), Value().to(device), device


# ── Training helpers (run in-memory inside train_agent) ──────────────────────

NUM_PARALLEL_ENVS = 32  # number of environments to run simultaneously


def collect_rollouts(policy, obs_norm, num_episodes, max_steps):
    """Collect rollouts using vectorized parallel environments. Returns list of episodes."""
    import torch
    import numpy as np
    import gymnasium as gym

    policy.eval()

    n_envs = min(NUM_PARALLEL_ENVS, num_episodes)
    envs = gym.vector.SyncVectorEnv([lambda: _make_env() for _ in range(n_envs)])

    all_episodes = []
    ep_transitions = [[] for _ in range(n_envs)]

    obs, _ = envs.reset()
    episodes_completed = 0

    while episodes_completed < num_episodes:
        obs_normed = obs_norm.normalize(obs).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_normed)
            mean, log_std = policy(obs_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            actions_t = dist.sample()
            log_probs = dist.log_prob(actions_t).sum(-1)
            actions = actions_t.numpy()

        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        obs_norm.update(obs)

        for i in range(n_envs):
            ep_transitions[i].append({
                "obs": obs[i].tolist(),
                "action": actions[i].tolist(),
                "reward": float(rewards[i]),
                "log_prob": float(log_probs[i].item()),
                "terminated": bool(terminateds[i]),
                "truncated": bool(truncateds[i]),
            })

            if terminateds[i] or truncateds[i]:
                all_episodes.append(ep_transitions[i])
                ep_transitions[i] = []
                episodes_completed += 1
                if episodes_completed >= num_episodes:
                    break

        obs = next_obs

    envs.close()
    return all_episodes


def ppo_update(episodes, policy, value_net, policy_optim, value_optim, obs_norm,
               ppo_epochs=5, clip_eps=0.2, gamma=0.95, lam=0.9):
    """Run PPO update on collected episodes. Returns average loss."""
    import torch
    import numpy as np

    all_obs, all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], [], []

    for ep in episodes:
        raw_obs = np.array([t["obs"] for t in ep], dtype=np.float32)
        obs = torch.tensor(obs_norm.normalize(raw_obs).astype(np.float32))
        actions = torch.tensor([t["action"] for t in ep], dtype=torch.float32)
        rewards = [t["reward"] for t in ep]
        old_log_probs = torch.tensor([t["log_prob"] for t in ep], dtype=torch.float32)

        with torch.no_grad():
            values = value_net(obs).numpy()

        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            is_last = (t == T - 1)
            if is_last:
                if ep[t].get("truncated", False):
                    next_val = values[t]
                else:
                    next_val = 0.0
                gae = 0.0
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

    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    total_loss = 0.0
    entropy_coef = 0.002
    batch_size = min(512, len(obs_batch))

    for _ in range(ppo_epochs):
        indices = torch.randperm(len(obs_batch))
        for start in range(0, len(obs_batch), batch_size):
            idx = indices[start:start + batch_size]
            mb_obs = obs_batch[idx]
            mb_actions = actions_batch[idx]
            mb_old_lp = old_log_probs_batch[idx]
            mb_adv = advantages_batch[idx]
            mb_ret = returns_batch[idx]

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
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            policy_optim.step()

            values_pred = value_net(mb_obs)
            value_loss = (values_pred - mb_ret).pow(2).mean()

            value_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 2.0)
            value_optim.step()

            total_loss += policy_loss.item()

    num_updates = ppo_epochs * max(1, len(obs_batch) // batch_size)
    return total_loss / num_updates


def run_eval(policy, obs_norm, num_episodes, max_steps):
    """Quick evaluation — returns mean and best reward. No rendering."""
    import torch
    import numpy as np

    if policy is not None:
        policy.eval()
    all_rewards = []

    for ep_idx in range(num_episodes):
        env = _make_env()
        obs, _ = env.reset(seed=ep_idx + 1000)
        total_reward = 0.0

        for _ in range(max_steps):
            if policy is not None:
                with torch.no_grad():
                    obs_normed = obs_norm.normalize(obs).astype(np.float32)
                    obs_t = torch.tensor(obs_normed).unsqueeze(0)
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

    return sum(all_rewards) / len(all_rewards), max(all_rewards)


@train_env.task(report=True)
async def evaluate(
    checkpoint: File | None,
    label: str = "Random",
    num_episodes: int = 5,
    max_steps: int = 1000,
    capture_frames: bool = True,
    show_report: bool = True,
) -> File:
    """Evaluate a policy and capture MuJoCo frames for the best episode."""
    import torch
    from PIL import Image

    import numpy as np

    state_dim, action_dim = _get_dims()

    policy = None
    obs_norm = ObsNormalizer(state_dim)
    if checkpoint is not None:
        policy, _, device = _build_policy_and_value(state_dim, action_dim)
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        if "obs_norm" in ckpt:
            obs_norm.load_state_dict(ckpt["obs_norm"])
        policy.eval()

    best_reward = -float("inf")
    best_frames = []
    all_rewards = []

    frame_every = 5
    max_frames = 100

    for ep_idx in range(num_episodes):
        env = _make_env(render_mode="rgb_array" if capture_frames else None)
        obs, _ = env.reset(seed=ep_idx + 1000)
        total_reward = 0.0
        frames = []

        for step in range(max_steps):
            if capture_frames and step % frame_every == 0 and len(frames) < max_frames:
                # Track the robot with the camera
                base_env = env.unwrapped
                robot_pos = base_env.data.qpos[:3].copy()
                cam = base_env.mujoco_renderer._get_viewer("rgb_array").cam
                cam.lookat[0] = robot_pos[0]
                cam.lookat[1] = robot_pos[1]
                cam.lookat[2] = robot_pos[2]
                frame = env.render()
                buf = io.BytesIO()
                Image.fromarray(frame).resize((320, 240)).save(buf, format="JPEG", quality=70)
                frames.append(base64.b64encode(buf.getvalue()).decode())

            if policy is not None:
                with torch.no_grad():
                    obs_normed = obs_norm.normalize(obs).astype(np.float32)
                    obs_t = torch.tensor(obs_normed).unsqueeze(0)
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

    if show_report:
        report_html = (
            f"<h3>{label}</h3>"
            f"<p>Mean reward: {mean_reward:.1f} | Best: {best_reward:.1f}</p>"
        )
        if capture_frames and best_frames:
            replay_html = generate_replay_html([
                {"label": label, "reward": best_reward, "frames_b64": best_frames},
            ])
            report_html += replay_html
        await flyte.report.replace.aio(report_html)
        await flyte.report.flush.aio()

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
    ppo_epochs: int = 5,
    max_steps: int = 1000,
    lr: float = 3e-5,
    eval_every: int = 20,
) -> str:
    """
    PPO training loop for the Unitree G1 humanoid.

    1. Evaluate random baseline
    2. For each iteration: collect rollouts -> PPO update -> evaluate
    3. Build final report with training curves + episode replays
    """
    import torch
    import numpy as np

    log.info(f"Starting PPO training: {num_iterations} iterations, {episodes_per_iter} episodes each")

    await flyte.report.replace.aio(
        "<h2>Unitree G1 — RL Training</h2>"
        f"<p>Training Unitree G1 humanoid with PPO for {num_iterations} iterations...</p>"
    )
    await flyte.report.flush.aio()

    # ── Setup — all state lives in memory ─────────────────────────────
    state_dim, action_dim = _get_dims()
    policy, value_net, device = _build_policy_and_value(state_dim, action_dim)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)
    obs_norm = ObsNormalizer(state_dim)

    # Evaluate random baseline
    baseline_mean, baseline_best = run_eval(None, obs_norm, num_episodes=5, max_steps=max_steps)
    baseline_reward = baseline_mean
    log.info(f"Random baseline: {baseline_reward:.1f}")

    # ── Traced training steps (visible in Flyte UI, enable crash recovery) ──

    # ── Training loop (all in-memory — no inter-task overhead) ────────
    best_reward = float("-inf")
    best_state = None
    history = []

    for i in range(1, num_iterations + 1):
        log.info(f"── Iteration {i}/{num_iterations} ──")

        # Linear LR decay — full LR at start, 10% at end
        iter_lr = lr * max(0.1, 1.0 - 0.9 * (i - 1) / max(1, num_iterations - 1))
        for pg in policy_optim.param_groups:
            pg["lr"] = iter_lr
        for pg in value_optim.param_groups:
            pg["lr"] = iter_lr

        # Collect → Update → Eval (all in-memory, no serialization)
        episodes = collect_rollouts(policy, obs_norm, episodes_per_iter, max_steps)
        loss = ppo_update(
            episodes, policy, value_net, policy_optim, value_optim, obs_norm,
            ppo_epochs=ppo_epochs,
        )
        eval_mean, eval_best = run_eval(policy, obs_norm, num_episodes=5, max_steps=max_steps)

        history.append({
            "iteration": i,
            "eval_reward": eval_mean,
            "best_reward": eval_best,
            "loss": loss,
        })

        # Track best model
        if eval_mean > best_reward:
            best_reward = eval_mean
            best_state = {
                "policy": {k: v.clone() for k, v in policy.state_dict().items()},
                "value": {k: v.clone() for k, v in value_net.state_dict().items()},
                "obs_norm": obs_norm.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
            }

        log.info(f"  Reward: {eval_mean:.1f} | Loss: {loss:.4f}")

        # Periodic checkpoint + replay via evaluate task
        if i % eval_every == 0 or i == num_iterations:
            ckpt_path = f"/tmp/ppo_checkpoint_iter{i}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "value": value_net.state_dict(),
                "policy_optim": policy_optim.state_dict(),
                "value_optim": value_optim.state_dict(),
                "obs_norm": obs_norm.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
            }, ckpt_path)
            iter_checkpoint = await File.from_local(ckpt_path)
            log.info(f"  Checkpoint saved — running replay for iteration {i}...")

            replay_file = await evaluate(
                checkpoint=iter_checkpoint,
                label=f"Iter {i}",
                num_episodes=3,
                max_steps=max_steps,
                capture_frames=True,
            )
            replay_path = await replay_file.download()
            with open(replay_path) as f:
                replay_data = json.load(f)
            if replay_data.get("frames_b64"):
                tab = flyte.report.get_tab(f"Iter {i}")
                frames_json = json.dumps(replay_data["frames_b64"])
                uid = f"iter{i}"
                tab.log(
                    f"<h3>Iteration {i}</h3><p>Reward: {replay_data['best_reward']:.1f}</p>"
                    f'<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:12px;border-radius:8px;">'
                    f'<img id="f-{uid}" src="data:image/jpeg;base64,{replay_data["frames_b64"][0]}" style="max-width:480px;display:block;border:2px solid #333;border-radius:4px;"/><br/>'
                    f'<input type="range" id="s-{uid}" min="0" max="{len(replay_data["frames_b64"])-1}" value="0" style="width:480px;"/>'
                    f' <span id="l-{uid}">Step 0/{len(replay_data["frames_b64"])-1}</span>'
                    f'<script>(function(){{var F={frames_json};'
                    f'var s=document.getElementById("s-{uid}"),i=document.getElementById("f-{uid}"),l=document.getElementById("l-{uid}");'
                    f's.oninput=function(){{i.src="data:image/jpeg;base64,"+F[this.value];'
                    f'l.textContent="Step "+this.value+"/{len(replay_data["frames_b64"])-1}";}};'
                    f'}})();</script></div>'
                )

        # Update live report
        charts_html = generate_training_charts(history, baseline_reward)
        progress = generate_progress_html(history, baseline_reward, num_iterations)
        await flyte.report.replace.aio(
            f"<h2>Unitree G1 — RL Training</h2>{progress}<br/>{charts_html}"
        )
        await flyte.report.flush.aio()

    # ── Save best checkpoint to disk for replay ──────────────────────
    best_checkpoint = None
    if best_state is not None:
        ckpt_path = "/tmp/ppo_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        best_checkpoint = await File.from_local(ckpt_path)

    # ── Replay best model and baseline with frames ─────────────────────
    log.info("Generating replay for best model and baseline...")
    replays = []

    # Baseline replay
    baseline_replay_file = await evaluate(
        checkpoint=None, label="Random Policy",
        num_episodes=5, max_steps=max_steps, capture_frames=True, show_report=False,
    )
    baseline_replay_path = await baseline_replay_file.download()
    with open(baseline_replay_path) as f:
        baseline_replay = json.load(f)
    if baseline_replay.get("frames_b64"):
        replays.append({"label": "Random Policy", "reward": baseline_replay["best_reward"], "frames_b64": baseline_replay["frames_b64"]})

    # Best model replay — run 5 episodes, keep frames from the best one
    if best_checkpoint is not None:
        best_iter = max(history, key=lambda h: h["eval_reward"])
        best_replay_file = await evaluate(
            checkpoint=best_checkpoint, label=f"Best (Iter {best_iter['iteration']})",
            num_episodes=5, max_steps=max_steps, capture_frames=True, show_report=False,
        )
        best_replay_path = await best_replay_file.download()
        with open(best_replay_path) as f:
            best_replay = json.load(f)
        if best_replay.get("frames_b64"):
            replays.append({"label": f"Best (Iter {best_iter['iteration']})", "reward": best_replay["best_reward"], "frames_b64": best_replay["frames_b64"]})

    # Add replay tabs
    for idx, ep in enumerate(replays):
        tab = flyte.report.get_tab(ep["label"][:25])
        uid = f"final_tab{idx}"
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

    # ── Build final report ───────────────────────────────────────────────
    charts_html = generate_training_charts(history, baseline_reward)
    replay_html = generate_replay_html(replays) if replays else ""

    best = max(history, key=lambda h: h["eval_reward"])
    improvement = best["eval_reward"] - baseline_reward

    summary_html = (
        '<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        '<h3 style="color:#00b894;margin-top:0;">Training Complete</h3>'
        '<table style="border-collapse:collapse;width:100%;">'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Robot</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">Unitree G1 Humanoid</td></tr>'
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
        "<h2>Unitree G1 — Training Complete</h2>"
        f"{summary_html}<br/>"
        f"{progress}<br/>"
        f"<h3 style='color:#fdcb6e;font-family:monospace;'>Training Progress</h3>{charts_html}<br/>"
        + (f"<h3 style='color:#fdcb6e;font-family:monospace;'>Episode Replays</h3>{replay_html}" if replay_html else "")
    )
    await flyte.report.flush.aio()

    log.info(f"Training complete. Best: {best['eval_reward']:.1f} (baseline: {baseline_reward:.1f})")

    return json.dumps({
        "robot": "Unitree G1",
        "baseline_reward": baseline_reward,
        "best_reward": best["eval_reward"],
        "improvement": improvement,
        "history": history,
    })
