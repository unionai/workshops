"""
MuJoCo RL training with PPO — Unitree Go2 quadruped locomotion.

Trains a policy network on the Unitree Go2 robot dog using Proximal Policy
Optimization (PPO) with Generalized Advantage Estimation (GAE). Uses the Go2 model
from mujoco_menagerie loaded into Gymnasium's Ant-v5 environment.

Quadrupeds are much easier to train than humanoids — 4 legs provide inherent stability,
12 DOF (vs 29 for G1), and forward locomotion typically converges in 10-20 iterations.

Usage:
    # Local (quick demo)
    flyte run --local workflow.py train_agent --num_iterations 5 --steps_per_iter 1000

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

go2_image = flyte.Image.from_debian_base().with_apt_packages(
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libgles2", "libglx-mesa0", "git",
).with_pip_packages(
    "gymnasium[mujoco]", "torch", "numpy", "Pillow", "matplotlib", "unionai-reuse",
).with_commands([
    "git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie",
])

# Workers — lightweight, many run in parallel for rollout collection + evaluation
worker_env = flyte.TaskEnvironment(
    name="unitree-go2-worker",
    image=go2_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
)

# GPU — PPO update only (allocated briefly each iteration, not held during rollouts)
gpu_env = flyte.TaskEnvironment(
    name="unitree-go2-gpu",
    image=go2_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
)

# Orchestrator — lightweight, just coordinates workers and GPU task
train_env = flyte.TaskEnvironment(
    name="unitree-go2-train",
    image=go2_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    depends_on=[worker_env, gpu_env],
)

# ── Constants ────────────────────────────────────────────────────────────────

HIDDEN_DIM = 256  # Larger network for 42-dim custom observation

# Go2 env config — passed to Ant-v5 with the Go2 XML
GO2_ENV_KWARGS = {
    "healthy_z_range": (0.18, 0.6),        # Match MJX termination bounds
    "healthy_reward": 0.0,                  # disable built-in — custom wrapper handles it
    "forward_reward_weight": 0.0,           # disable built-in — custom wrapper handles it
    "ctrl_cost_weight": 0.0,               # disable built-in — custom wrapper handles it
    "contact_cost_weight": 0.0,             # disable built-in contact cost
    "reset_noise_scale": 0.01,              # small noise so it starts near standing
    "terminate_when_unhealthy": True,
    "exclude_current_positions_from_observation": True,
    "include_cfrc_ext_in_observation": True,
    "frame_skip": 5,                        # Match MJX n_frames=5
}

# Target walking speed in m/s (Go2 walks ~0.5-1.0 m/s)
TARGET_VELOCITY = 0.6

# Lazy import — gymnasium only available in container
try:
    import gymnasium
    _GymnasiumWrapper = gymnasium.Wrapper
except ImportError:
    _GymnasiumWrapper = object


class Go2ActionScaler(_GymnasiumWrapper):
    """Wraps the Go2 env with proper action scaling and custom observation.

    Go2 uses position-control motors (PD controllers), not torque. Policy outputs
    [-1, 1] are scaled to small offsets from the default standing pose:
        motor_targets = default_pose + action * action_scale

    Custom observation matches the MJX version: projected gravity, angular velocity,
    joint deltas from default pose, joint velocities, and last action (42 dims).

    Action space is normalized to [-1, 1] — the policy outputs in this range and
    the wrapper handles the scaling to joint positions.
    """

    ACTION_SCALE = 0.3  # Policy output [-1,1] → ±0.3 rad offset from default pose
    ACTION_REPEAT = 4   # Hold each action for 4 env steps (0.04s) — gives PD controller
                        # time to actually move legs, producing real forward motion.
                        # Without this, per-step random noise just makes the robot jiggle.
    OBS_DIM = 42  # 3 + 3 + 12 + 12 + 12

    def __init__(self, env):
        import numpy as np
        import gymnasium.spaces as spaces
        super().__init__(env)

        # Custom 42-dim observation space matching MJX version
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # Normalize action space to [-1, 1] — matches what Brax PPO expects
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_space.shape[0],), dtype=np.float32
        )

        # Get default standing pose from model
        mj_model = env.unwrapped.model
        if mj_model.nkey > 0:
            self._default_pose = mj_model.key_qpos[0][7:].copy()
        else:
            self._default_pose = mj_model.qpos0[7:].copy()

        # Joint limits: default ± offset (abduction, hip, knee per leg)
        offset = np.array([0.2, 0.8, 0.8] * 4)
        self._lower = self._default_pose - offset
        self._upper = self._default_pose + offset
        self._last_action = np.zeros(env.action_space.shape[0], dtype=np.float32)

        # Foot site indices for air time reward
        import mujoco
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ])
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contact = np.ones(4, dtype=bool)
        self._dt = mj_model.opt.timestep * GO2_ENV_KWARGS.get("frame_skip", 5)

    def _get_obs(self, data):
        """Build custom observation matching the MJX version."""
        import numpy as np

        # Projected gravity: rotate [0,0,-1] by inverse body quaternion
        quat = data.qpos[3:7]
        w, x, y, z = quat[0], -quat[1], -quat[2], -quat[3]
        gx = 2.0 * (x * z + w * y) * (-1.0)
        gy = 2.0 * (y * z - w * x) * (-1.0)
        gz = (1.0 - 2.0 * (x * x + y * y)) * (-1.0)
        gravity_body = np.array([gx, gy, gz], dtype=np.float32)

        # Angular velocity (scaled)
        ang_vel = data.qvel[3:6].astype(np.float32) * 0.25

        # Joint deltas from default pose
        joint_deltas = (data.qpos[7:] - self._default_pose).astype(np.float32)

        # Joint velocities
        joint_vel = data.qvel[6:].astype(np.float32)

        obs = np.concatenate([gravity_body, ang_vel, joint_deltas, joint_vel, self._last_action])
        return np.clip(obs, -100.0, 100.0)

    def reset(self, **kwargs):
        import numpy as np
        _obs, info = self.env.reset(**kwargs)
        self._last_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contact = np.ones(4, dtype=bool)

        # Give the robot a small random forward velocity at reset.
        # Without this, random exploration produces zero net forward movement
        # (uncorrelated per-step noise cancels out), so the policy never
        # discovers that walking earns reward. The initial push lets the robot
        # experience forward velocity and learn to maintain it.
        import mujoco as _mj
        data = self.env.unwrapped.data
        data.qvel[0] = np.random.uniform(0.2, 1.0)  # moderate forward push
        _mj.mj_forward(self.env.unwrapped.model, data)

        return self._get_obs(data), info

    def _compute_reward(self, action, data):
        """Compute custom reward from physics state."""
        import numpy as np

        x_vel = data.qvel[0]
        forward_reward = 3.0 * x_vel

        alive_bonus = 0.1  # small — stabilizes training without dominating velocity signal

        quat = data.qpos[3:7]
        w, x, y, z = quat[0], -quat[1], -quat[2], -quat[3]
        gx = 2.0 * (x * z + w * y) * (-1.0)
        gy = 2.0 * (y * z - w * x) * (-1.0)
        orientation_penalty = -2.0 * (gx ** 2 + gy ** 2)

        ctrl_cost = -0.02 * np.sum(action ** 2)

        # Feet air time reward
        foot_pos = data.site_xpos[self._foot_site_ids]
        foot_contact = foot_pos[:, 2] < 0.025
        self._feet_air_time += self._dt
        first_contact = foot_contact & ~self._last_contact
        air_time_reward = np.sum(
            (self._feet_air_time - 0.15) * first_contact
        )
        self._feet_air_time[foot_contact] = 0.0
        self._last_contact = foot_contact

        # Action rate penalty — penalizes jerky movements between consecutive steps
        action_rate_cost = -0.05 * np.sum((action - self._last_action) ** 2)

        return (forward_reward + alive_bonus + orientation_penalty
                + ctrl_cost + 1.0 * air_time_reward + action_rate_cost)

    def step(self, action):
        import numpy as np

        # Clip to [-1, 1], smooth with EMA, then scale to joint targets
        action = np.clip(action, -1.0, 1.0)
        action = 0.8 * action + 0.2 * self._last_action  # low-pass filter reduces jitter
        motor_targets = self._default_pose + action * self.ACTION_SCALE
        motor_targets = np.clip(motor_targets, self._lower, self._upper)

        # Action repeat: hold the same action for multiple env steps.
        # PD position control is smooth — per-step random noise just makes the
        # robot jiggle in place. Repeating actions gives the PD controller time
        # to actually move legs to their targets, producing sustained movements
        # that generate forward velocity during exploration.
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.ACTION_REPEAT):
            _obs, _base_reward, terminated, truncated, info = self.env.step(motor_targets)
            data = self.env.unwrapped.data
            total_reward += self._compute_reward(action, data)
            if terminated or truncated:
                break

        self._last_action = action.astype(np.float32)
        obs = self._get_obs(data)

        return obs, float(total_reward), terminated, truncated, info


def _make_env(render_mode=None):
    """Create the Unitree Go2 environment with action scaling and custom reward."""
    import gymnasium as gym

    xml_path = "/opt/mujoco_menagerie/unitree_go2/scene.xml"
    env = gym.make(
        "Ant-v5",
        xml_file=xml_path,
        render_mode=render_mode,
        **GO2_ENV_KWARGS,
    )

    # Convert torque motors to position-controlled PD servos.
    # CRITICAL: go2.xml uses <motor> actuators (biastype="none"), so setting biasprm
    # alone does NOTHING — MuJoCo ignores bias terms when biastype is "none".
    # go2_mjx.xml already uses <general biastype="affine">, which is why MJX works.
    # We must explicitly set biastype to affine for PD control to work on CPU.
    import mujoco as _mj
    mj_model = env.unwrapped.model
    mj_model.actuator_biastype[:] = _mj.mjtBias.mjBIAS_AFFINE  # Enable PD control
    mj_model.actuator_gainprm[:, 0] = 35.0       # P-gain
    mj_model.actuator_biasprm[:, 0] = 0.0        # no constant bias
    mj_model.actuator_biasprm[:, 1] = -35.0      # position feedback: -Kp * joint_pos
    mj_model.actuator_biasprm[:, 2] = -0.5       # velocity feedback: -Kd * joint_vel
    mj_model.dof_damping[6:] = 0.5239

    # CRITICAL: Override init_qpos with the "home" keyframe (standing pose).
    # Gymnasium's Ant-v5 resets to model.qpos0 which has all joints at 0
    # (legs extended). The Go2 standing pose (thigh=0.9, calf=-1.8) is only
    # defined in the keyframe. Without this, the robot starts splayed out
    # and falls immediately every episode.
    if mj_model.nkey > 0:
        home_qpos = mj_model.key_qpos[0].copy()
    else:
        home_qpos = mj_model.qpos0.copy()
    env.unwrapped.init_qpos = home_qpos

    return Go2ActionScaler(env)


def _get_dims():
    """Get observation and action dimensions from the Go2 environment."""
    env = _make_env()
    state_dim = env.observation_space.shape[0]  # 42 (custom obs)
    action_dim = env.action_space.shape[0]      # 12 (Go2 joints)
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
    """Generate interactive MuJoCo frame replay with slider."""
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
        }} else {{
          document.getElementById('btn-{uid}').textContent = 'Pause';
          timer = setInterval(function() {{
            var sl = document.getElementById('slider-{uid}');
            if (parseInt(sl.value) >= parseInt(sl.max)) {{
              clearInterval(timer);
              timer = null;
              document.getElementById('btn-{uid}').textContent = 'Play';
            }} else {{
              sl.value = parseInt(sl.value) + 1;
              render_{uid}();
            }}
          }}, 80);
        }}
      }};

      changeEp_{uid}();
    }})();
    </script>
    """


def generate_progress_html(history: list[dict], baseline_reward: float, num_iterations: int) -> str:
    """Build a live progress table."""
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

    return (
        f'<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        f'<h3 style="color:#fdcb6e;margin-top:0;">Training Progress — {len(history)}/{num_iterations} iterations</h3>'
        f'{bar}'
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
    """Generate training progress charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [h["iteration"] for h in history]
    rewards = [h["eval_reward"] for h in history]
    losses = [h["loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f0f23")

    ax1.set_facecolor("#1a1a2e")
    ax1.plot(iterations, rewards, color="#00b894", linewidth=2, marker="o", markersize=4, label="Trained policy")
    ax1.axhline(y=baseline_reward, color="#e17055", linestyle="--", linewidth=1.5, label=f"Random baseline ({baseline_reward:.1f})")
    ax1.set_xlabel("Iteration", color="#ccc")
    ax1.set_ylabel("Mean Episode Reward", color="#ccc")
    ax1.set_title("Unitree Go2 — Training Progress", color="#fdcb6e", fontsize=14)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    ax1.tick_params(colors="#888")
    ax1.grid(alpha=0.15)

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
    """Running mean/std normalizer for observations."""

    def __init__(self, dim: int):
        import numpy as np
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        import numpy as np
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, obs):
        import numpy as np
        return (np.asarray(obs) - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, d):
        import numpy as np
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = d["count"]

    def merge(self, other):
        """Merge stats from another normalizer (for combining parallel workers)."""
        import numpy as np
        if other.count == 0:
            return
        if self.count == 0:
            self.mean = other.mean.copy()
            self.var = other.var.copy()
            self.count = other.count
            return
        delta = other.mean - self.mean
        total = self.count + other.count
        new_mean = self.mean + delta * other.count / total
        m_a = self.var * self.count
        m_b = other.var * other.count
        m2 = m_a + m_b + delta**2 * self.count * other.count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


# ── Policy network ───────────────────────────────────────────────────────────

def _build_policy_and_value(state_dim: int, action_dim: int):
    """Build policy (actor) and value (critic) networks."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, HIDDEN_DIM), nn.SiLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            )
            self.mean = nn.Linear(HIDDEN_DIM, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # std=0.6 — enough exploration to stumble into forward motion

        def forward(self, x):
            h = self.net(x)
            return self.mean(h), self.log_std.expand_as(self.mean(h))

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, HIDDEN_DIM), nn.SiLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
                nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Policy().to(device), Value().to(device), device


# ── Rollout collection (Flyte task — fans out in parallel) ───────────────────

NUM_PARALLEL_ENVS = 32


@worker_env.task(retries=5)
async def collect_rollouts(
    checkpoint: File | None,
    num_steps: int = 8000,
) -> File:
    """Collect rollouts using step-based collection (like Brax PPO).

    Collects a fixed number of STEPS, not episodes. When an episode ends,
    the env auto-resets and collection continues. This guarantees consistent
    data volume regardless of episode length.
    """
    import torch
    import numpy as np
    import gymnasium as gym

    state_dim, action_dim = _get_dims()
    policy, _, device = _build_policy_and_value(state_dim, action_dim)
    obs_norm = ObsNormalizer(state_dim)

    if checkpoint is not None:
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        if "obs_norm" in ckpt:
            obs_norm.load_state_dict(ckpt["obs_norm"])
    policy.eval()

    n_envs = NUM_PARALLEL_ENVS
    envs = gym.vector.SyncVectorEnv([lambda: _make_env() for _ in range(n_envs)])

    all_episodes = []
    ep_transitions = [[] for _ in range(n_envs)]

    obs, _ = envs.reset()
    steps_collected = 0

    while steps_collected < num_steps:
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
            steps_collected += 1

            if terminateds[i] or truncateds[i]:
                all_episodes.append(ep_transitions[i])
                ep_transitions[i] = []

        obs = next_obs

    # Flush any in-progress episodes (mark as truncated)
    for i in range(n_envs):
        if ep_transitions[i]:
            ep_transitions[i][-1]["truncated"] = True
            all_episodes.append(ep_transitions[i])

    envs.close()

    path = "/tmp/rollouts.json"
    with open(path, "w") as f:
        json.dump({"episodes": all_episodes, "obs_norm": obs_norm.state_dict()}, f)
    return await File.from_local(path)


@gpu_env.task(retries=3)
async def ppo_update(
    rollout_files: list[File],
    checkpoint: File | None,
    ppo_epochs: int = 5,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    gamma: float = 0.97,
    lam: float = 0.95,
) -> File:
    """Merge rollouts from parallel workers and run PPO update on GPU."""
    import torch
    import numpy as np

    all_episodes = []
    state_dim = None

    for rollout_file in rollout_files:
        rollout_path = await rollout_file.download()
        with open(rollout_path) as f:
            rollout_data = json.load(f)
        all_episodes.extend(rollout_data["episodes"])
        if state_dim is None:
            state_dim = len(rollout_data["episodes"][0][0]["obs"])

    action_dim = len(all_episodes[0][0]["action"])
    log.info(f"PPO update: {len(all_episodes)} episodes on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    policy, value_net, device = _build_policy_and_value(state_dim, action_dim)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    # Merge obs normalizer from all workers
    obs_norm = ObsNormalizer(state_dim)
    for rollout_file in rollout_files:
        rollout_path = await rollout_file.download()
        with open(rollout_path) as f:
            rollout_data = json.load(f)
        worker_norm = ObsNormalizer(state_dim)
        worker_norm.load_state_dict(rollout_data["obs_norm"])
        obs_norm.merge(worker_norm)

    if checkpoint is not None:
        local = await checkpoint.download()
        ckpt = torch.load(local, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        value_net.load_state_dict(ckpt["value"])
        policy_optim.load_state_dict(ckpt["policy_optim"])
        value_optim.load_state_dict(ckpt["value_optim"])
        if "obs_norm" in ckpt:
            base_norm = ObsNormalizer(state_dim)
            base_norm.load_state_dict(ckpt["obs_norm"])
            base_norm.merge(obs_norm)
            obs_norm = base_norm

    # Reward normalization — critical for CPU PPO.
    # Without this, the alive bonus and control penalties dominate the reward signal,
    # drowning out the forward velocity gradient. Brax PPO does this internally.
    all_raw_rewards = [t["reward"] for ep in all_episodes for t in ep]
    reward_std = np.std(all_raw_rewards) + 1e-8

    # Compute GAE
    all_obs, all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], [], []

    for ep in all_episodes:
        raw_obs = np.array([t["obs"] for t in ep], dtype=np.float32)
        obs = torch.tensor(obs_norm.normalize(raw_obs).astype(np.float32))
        actions = torch.tensor([t["action"] for t in ep], dtype=torch.float32)
        rewards = [t["reward"] / reward_std for t in ep]  # normalized rewards for GAE
        old_log_probs = torch.tensor([t["log_prob"] for t in ep], dtype=torch.float32)

        with torch.no_grad():
            values = value_net(obs.to(device)).cpu().numpy()

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

    obs_batch = torch.cat(all_obs).to(device)
    actions_batch = torch.cat(all_actions).to(device)
    old_log_probs_batch = torch.cat(all_old_log_probs).to(device)
    advantages_batch = torch.cat(all_advantages).to(device)
    returns_batch = torch.cat(all_returns).to(device)

    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    total_loss = 0.0
    entropy_coef = 0.01  # moderate exploration
    batch_size = min(256, len(obs_batch))  # smaller batches = more gradient updates per epoch

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
    avg_loss = total_loss / num_updates

    path = "/tmp/ppo_checkpoint.pt"
    torch.save({
        "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
        "value": {k: v.cpu() for k, v in value_net.state_dict().items()},
        "policy_optim": policy_optim.state_dict(),
        "value_optim": value_optim.state_dict(),
        "obs_norm": obs_norm.state_dict(),
        "loss": avg_loss,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }, path)
    return await File.from_local(path)


def run_eval(policy, obs_norm, num_episodes, max_steps):
    """Quick evaluation — returns mean and best reward. No rendering."""
    import torch
    import numpy as np

    if policy is not None:
        policy.eval()
    all_rewards = []

    all_lengths = []
    for ep_idx in range(num_episodes):
        env = _make_env()
        obs, _ = env.reset(seed=ep_idx + 1000)
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            if policy is not None:
                with torch.no_grad():
                    device = next(policy.parameters()).device
                    obs_normed = obs_norm.normalize(obs).astype(np.float32)
                    obs_t = torch.tensor(obs_normed).unsqueeze(0).to(device)
                    mean, _ = policy(obs_t)
                    action = mean.squeeze(0).cpu().numpy()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        env.close()
        all_rewards.append(total_reward)
        all_lengths.append(steps)

    avg_len = sum(all_lengths) / len(all_lengths)
    log.info(f"  Eval: avg_episode_length={avg_len:.0f} steps")
    return sum(all_rewards) / len(all_rewards), max(all_rewards)


@worker_env.task(report=True, retries=5)
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
                    device = next(policy.parameters()).device
                    obs_normed = obs_norm.normalize(obs).astype(np.float32)
                    obs_t = torch.tensor(obs_normed).unsqueeze(0).to(device)
                    mean, _ = policy(obs_t)
                    action = mean.squeeze(0).cpu().numpy()
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

@train_env.task(report=True, retries=3)
async def train_agent(
    num_iterations: int = 60,
    steps_per_iter: int = 80000,
    ppo_epochs: int = 10,
    max_steps: int = 1000,
    lr: float = 3e-4,
    eval_every: int = 10,
    num_workers: int = 10,
    resume_checkpoint: File | None = None,
) -> str:
    """
    PPO training loop for the Unitree Go2 quadruped.

    1. Evaluate random baseline
    2. For each iteration: collect rollouts (step-based) -> PPO update -> evaluate
    3. Build final report with training curves + episode replays

    Step-based collection: each worker collects a fixed number of transitions
    (not episodes). This matches Brax PPO's approach — with short episodes (~12
    steps), episode-based collection produces too little data per iteration.
    Default: 80,000 steps/iter across 10 workers = 8,000 steps/worker,
    matching MJX's 4096 envs × 20 unroll = 81,920 transitions/iter.
    """
    import torch

    steps_per_worker = steps_per_iter // num_workers
    log.info(f"Starting PPO training: {num_iterations} iterations, {steps_per_iter} steps each ({num_workers} workers × {steps_per_worker} steps)")

    await flyte.report.replace.aio(
        "<h2>Unitree Go2 — RL Training</h2>"
        f"<p>Training Unitree Go2 quadruped with PPO for {num_iterations} iterations...</p>"
    )
    await flyte.report.flush.aio()

    # ── Setup ─────────────────────────────────────────────────────────
    state_dim, action_dim = _get_dims()
    obs_norm = ObsNormalizer(state_dim)

    baseline_mean, _ = run_eval(None, obs_norm, num_episodes=5, max_steps=max_steps)
    baseline_reward = baseline_mean
    log.info(f"Random baseline: {baseline_reward:.1f}")

    # ── Training loop ────────────────────────────────────────────────
    best_reward = float("-inf")
    best_checkpoint_file = None
    history = []
    checkpoint_file = None
    start_iter = 1

    import asyncio

    # ── Resume from checkpoint ────────────────────────────────────────
    ctx = flyte.ctx()
    resume_from = None
    if ctx.checkpoints and ctx.checkpoints.prev_checkpoint_path:
        import pathlib
        prev_path = pathlib.Path(ctx.checkpoints.prev_checkpoint_path) / "progress.json"
        if prev_path.exists():
            with open(prev_path) as f:
                progress_state = json.load(f)
            start_iter = progress_state["iteration"] + 1
            history = progress_state["history"]
            best_reward = progress_state["best_reward"]
            resume_from = progress_state.get("checkpoint_uri")
            log.info(f"Resuming from retry checkpoint: iteration {start_iter}, best reward {best_reward:.1f}")

    if resume_from:
        checkpoint_file = File.from_existing_remote(resume_from)
        if best_reward > float("-inf"):
            best_checkpoint_file = checkpoint_file
    elif resume_checkpoint is not None:
        checkpoint_file = resume_checkpoint
        best_checkpoint_file = resume_checkpoint
        local = await checkpoint_file.download()
        ckpt = torch.load(local, map_location="cpu")
        if "obs_norm" in ckpt:
            obs_norm.load_state_dict(ckpt["obs_norm"])
        log.info("Resuming from provided checkpoint")

    for i in range(start_iter, num_iterations + 1):
        log.info(f"── Iteration {i}/{num_iterations} ({num_workers} workers × {steps_per_worker} steps) ──")

        # Linear LR decay
        iter_lr = lr * max(0.1, 1.0 - 0.9 * (i - 1) / max(1, num_iterations - 1))

        # Fan out step-based rollout collection across parallel CPU workers
        rollout_files = await asyncio.gather(*[
            collect_rollouts(
                checkpoint=checkpoint_file,
                num_steps=steps_per_worker,
            )
            for _ in range(num_workers)
        ])
        log.info(f"  Collected ~{steps_per_iter} steps from {num_workers} workers")

        # PPO update on GPU
        checkpoint_file = await ppo_update(
            rollout_files=list(rollout_files),
            checkpoint=checkpoint_file,
            ppo_epochs=ppo_epochs,
            lr=iter_lr,
        )

        # Load checkpoint for eval
        local = await checkpoint_file.download()
        ckpt = torch.load(local, map_location="cpu")
        loss = ckpt.get("loss", 0.0)
        obs_norm = ObsNormalizer(state_dim)
        if "obs_norm" in ckpt:
            obs_norm.load_state_dict(ckpt["obs_norm"])

        policy, _, _ = _build_policy_and_value(state_dim, action_dim)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        eval_mean, eval_best = run_eval(policy, obs_norm, num_episodes=5, max_steps=max_steps)

        history.append({
            "iteration": i,
            "eval_reward": eval_mean,
            "best_reward": eval_best,
            "loss": loss,
        })

        if eval_mean > best_reward:
            best_reward = eval_mean
            best_checkpoint_file = checkpoint_file

        log.info(f"  Reward: {eval_mean:.1f} | Loss: {loss:.4f}")

        # Save progress for retry resumption
        if ctx.checkpoints and ctx.checkpoints.checkpoint_path:
            import pathlib
            ckpt_dir = pathlib.Path(ctx.checkpoints.checkpoint_path)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "progress.json", "w") as f:
                json.dump({
                    "iteration": i,
                    "history": history,
                    "best_reward": best_reward,
                    "checkpoint_uri": checkpoint_file.path if checkpoint_file else None,
                }, f)

        # Periodic replay
        if i % eval_every == 0 or i == num_iterations:
            log.info(f"  Running replay for iteration {i}...")

            replay_file = await evaluate(
                checkpoint=checkpoint_file,
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
            f"<h2>Unitree Go2 — RL Training</h2>{progress}<br/>{charts_html}"
        )
        await flyte.report.flush.aio()

    # ── Replay best model and baseline ────────────────────────────────
    log.info("Generating replay for best model and baseline...")
    replays = []

    best_iter = max(history, key=lambda h: h["eval_reward"]) if history else None
    replay_tasks = [
        evaluate(checkpoint=None, label="Random Policy",
                 num_episodes=5, max_steps=max_steps, capture_frames=True, show_report=False),
    ]
    if best_checkpoint_file is not None:
        replay_tasks.append(
            evaluate(checkpoint=best_checkpoint_file, label=f"Best (Iter {best_iter['iteration']})",
                     num_episodes=5, max_steps=max_steps, capture_frames=True, show_report=False),
        )
    replay_files = await asyncio.gather(*replay_tasks)

    replay_labels = ["Random Policy"]
    if best_checkpoint_file is not None:
        replay_labels.append(f"Best (Iter {best_iter['iteration']})")

    for replay_file, label in zip(replay_files, replay_labels):
        replay_path = await replay_file.download()
        with open(replay_path) as f:
            replay_data = json.load(f)
        if replay_data.get("frames_b64"):
            replays.append({"label": label, "reward": replay_data["best_reward"], "frames_b64": replay_data["frames_b64"]})

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

    # ── Build final report ────────────────────────────────────────────
    charts_html = generate_training_charts(history, baseline_reward)
    replay_html = generate_replay_html(replays) if replays else ""

    best = max(history, key=lambda h: h["eval_reward"])
    improvement = best["eval_reward"] - baseline_reward

    summary_html = (
        '<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        '<h3 style="color:#00b894;margin-top:0;">Training Complete</h3>'
        '<table style="border-collapse:collapse;width:100%;">'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Robot</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">Unitree Go2 Quadruped</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Iterations</td><td style="padding:6px;border-bottom:1px solid #333;">{num_iterations}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Steps/iter</td><td style="padding:6px;border-bottom:1px solid #333;">{steps_per_iter:,}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Random baseline</td><td style="padding:6px;border-bottom:1px solid #333;color:#e17055;">{baseline_reward:.1f}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Best trained reward</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">{best["eval_reward"]:.1f}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Improvement</td><td style="padding:6px;border-bottom:1px solid #333;color:#fdcb6e;">{improvement:+.1f}</td></tr>'
        f'<tr><td style="padding:6px;">Learning rate</td><td style="padding:6px;">{lr}</td></tr>'
        '</table></div>'
    )

    progress = generate_progress_html(history, baseline_reward, num_iterations)

    await flyte.report.replace.aio(
        "<h2>Unitree Go2 — Training Complete</h2>"
        f"{summary_html}<br/>"
        f"{progress}<br/>"
        f"<h3 style='color:#fdcb6e;font-family:monospace;'>Training Progress</h3>{charts_html}<br/>"
        + (f"<h3 style='color:#fdcb6e;font-family:monospace;'>Episode Replays</h3>{replay_html}" if replay_html else "")
    )
    await flyte.report.flush.aio()

    log.info(f"Training complete. Best: {best['eval_reward']:.1f} (baseline: {baseline_reward:.1f})")

    return json.dumps({
        "robot": "Unitree Go2",
        "baseline_reward": baseline_reward,
        "best_reward": best["eval_reward"],
        "improvement": improvement,
        "history": history,
    })
