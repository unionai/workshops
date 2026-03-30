"""
GPU-accelerated RL training with MJX + Brax PPO — Unitree Go2 quadruped.

Trains the Go2 robot dog to walk using MJX (MuJoCo on GPU via JAX) with Brax's
PPO implementation. Everything runs on GPU — physics simulation, policy network,
and gradient updates. 4096 parallel environments on a single GPU.

This is 100-1000x faster than CPU MuJoCo. Full training in minutes, not hours.

Usage:
    # Local (quick demo — needs CUDA GPU)
    flyte run --local workflow.py train_agent --num_timesteps 5_000_000

    # Remote (full training)
    flyte run workflow.py train_agent --num_timesteps 20_000_000
"""

import os
import io
import json
import base64
import logging
import sys

# Use EGL for headless MuJoCo rendering (required in containers without a display)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

import flyte
import flyte.report
from flyte.io import File

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── Environments ─────────────────────────────────────────────────────────────

mjx_image = flyte.Image.from_debian_base().with_apt_packages(
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libgles2", "libglx-mesa0", "git",
).with_pip_packages(
    "mujoco-mjx", "jax[cuda12]", "brax",
    "numpy", "Pillow", "matplotlib", "unionai-reuse",
    "gymnasium[mujoco]",  # for CPU-based replay rendering
).with_commands([
    "git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie",
])

# GPU environment — everything runs on GPU (sim + training)
gpu_env = flyte.TaskEnvironment(
    name="unitree-go2-mjx",
    image=mjx_image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
)

# Worker for CPU-based replay rendering
worker_env = flyte.TaskEnvironment(
    name="unitree-go2-mjx-render",
    image=mjx_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
)

# Orchestrator
train_env = flyte.TaskEnvironment(
    name="unitree-go2-mjx-train",
    image=mjx_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    depends_on=[gpu_env, worker_env],
)

# ── Constants ────────────────────────────────────────────────────────────────

GO2_XML_PATH = "/opt/mujoco_menagerie/unitree_go2/scene_mjx.xml"
TARGET_VELOCITY = 0.6  # m/s


# ── MJX Environment ─────────────────────────────────────────────────────────

def _make_go2_env():
    """Create the Go2 MJX environment using Brax's PipelineEnv.

    Key insight: Go2 uses position-controlled actuators (PD controllers), not
    torque actuators. Actions must be small offsets from the default standing
    pose, not raw motor commands. This follows the Brax Barkour tutorial pattern.
    """
    import jax
    from jax import numpy as jp
    import mujoco
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf
    from brax import math as brax_math

    ACTION_SCALE = 0.3  # Policy output [-1,1] → ±0.3 rad offset from default pose
    # Per-joint limits: [abduction, hip, knee] × 4 legs
    JOINT_LOWER_OFFSET = jp.array([0.2, 0.8, 0.8] * 4)
    JOINT_UPPER_OFFSET = jp.array([0.2, 0.8, 0.8] * 4)

    class Go2Locomotion(PipelineEnv):
        """Unitree Go2 locomotion environment running on GPU via MJX."""

        def __init__(self, **kwargs):
            mj_model = mujoco.MjModel.from_xml_path(GO2_XML_PATH)

            # Tune actuator PD gains (from Barkour tutorial — makes motors stable)
            mj_model.actuator_gainprm[:, 0] = 35.0
            mj_model.actuator_biasprm[:, 1] = -35.0

            # Add joint damping for stability
            mj_model.dof_damping[6:] = 0.5239

            # Get foot site indices for air time reward
            self._foot_site_ids = jp.array([
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ])

            sys = mjcf.load_model(mj_model)
            kwargs["n_frames"] = 5  # 50 Hz control (matching Barkour)
            kwargs["backend"] = "mjx"
            super().__init__(sys, **kwargs)

            # Default standing pose (joint angles only, not base pos/quat)
            home_qpos = mj_model.keyframe("home").qpos if mj_model.nkey > 0 else mj_model.qpos0
            self._default_qpos = jp.array(home_qpos)
            self._default_pose = jp.array(home_qpos[7:])  # 12 joint angles

            # Joint position limits (default ± offset)
            self._lower = self._default_pose - JOINT_LOWER_OFFSET
            self._upper = self._default_pose + JOINT_UPPER_OFFSET

        def reset(self, rng):
            rng, rng_pos, rng_vel = jax.random.split(rng, 3)

            # Start near default standing pose with small noise
            qpos = self._default_qpos + jax.random.uniform(
                rng_pos, shape=(self.sys.nq,), minval=-0.01, maxval=0.01
            )
            qvel = jax.random.uniform(
                rng_vel, shape=(self.sys.nv,), minval=-0.01, maxval=0.01
            )

            data = self.pipeline_init(qpos, qvel)
            obs = self._get_obs(data, jp.zeros(self.sys.nu))
            reward, done = jp.zeros(2)
            metrics = {"reward": reward}
            info = {
                "last_action": jp.zeros(self.sys.nu),
                "feet_air_time": jp.zeros(4),
                "last_contact": jp.ones(4, dtype=jp.bool_),
            }
            return State(pipeline_state=data, obs=obs, reward=reward, done=done,
                         metrics=metrics, info=info)

        def step(self, state, action):
            # Convert policy output to motor position targets:
            # default_pose + action * 0.3, clipped to joint limits
            motor_targets = self._default_pose + action * ACTION_SCALE
            motor_targets = jp.clip(motor_targets, self._lower, self._upper)

            data = self.pipeline_step(state.pipeline_state, motor_targets)

            # ── Reward ──────────────────────────────────────────────────
            # Forward velocity — dominant reward, must outweigh "just standing"
            x_vel = data.qvel[0]
            forward_reward = 3.0 * x_vel

            # Small alive bonus — enough to prefer living but not enough
            # to make standing still the optimal strategy
            alive_bonus = 0.5

            # Orientation penalty — gentle, walking requires some lean
            quat = data.qpos[3:7]
            inv_quat = jp.array([quat[0], -quat[1], -quat[2], -quat[3]])
            gravity_body = brax_math.rotate(jp.array([0.0, 0.0, -1.0]), inv_quat)
            orientation_penalty = -2.0 * jp.sum(jp.square(gravity_body[:2]))

            # Control cost
            ctrl_cost = -0.02 * jp.sum(jp.square(action))

            # ── Feet air time reward ────────────────────────────────────
            # Encourages proper trotting gait — feet should alternate between
            # air and ground. Rewards feet that land after ~0.15s in the air.
            foot_pos = data.site_xpos[self._foot_site_ids]  # (4, 3)
            foot_contact = foot_pos[:, 2] < 0.025  # on ground if z < 2.5cm

            # Track air time: increment while in air, reset on ground contact
            air_time = state.info["feet_air_time"] + self.dt
            air_time = jp.where(foot_contact, 0.0, air_time)

            # Reward feet that just landed after being in air
            first_contact = foot_contact & ~state.info["last_contact"]
            air_time_reward = jp.sum(
                (state.info["feet_air_time"] - 0.15) * first_contact
            )

            reward = (
                forward_reward + alive_bonus + orientation_penalty
                + ctrl_cost + 1.0 * air_time_reward
            )

            # ── Termination ─────────────────────────────────────────────
            z_pos = data.qpos[2]
            # Check if upside down (gravity z-component should be negative in body frame)
            upside_down = gravity_body[2] > 0
            too_low = z_pos < 0.18
            too_high = z_pos > 0.6
            vel_magnitude = jp.sum(data.qvel ** 2)
            unstable = vel_magnitude > 100.0

            done = jp.where(upside_down | too_low | too_high | unstable, 1.0, 0.0)

            # NaN guard
            reward = jp.where(jp.isnan(reward), 0.0, reward)

            obs = self._get_obs(data, action)
            # Preserve all info keys added by Brax's training wrapper (steps,
            # episode_metrics, etc.) — JAX scan requires matching pytree structure
            info = {
                **state.info,
                "last_action": action,
                "feet_air_time": air_time,
                "last_contact": foot_contact,
            }
            return state.replace(
                pipeline_state=data, obs=obs, reward=reward, done=done,
                metrics={"reward": reward},
                info=info,
            )

        def _get_obs(self, data, last_action):
            """Observation: projected gravity + joint deltas + velocities + last action."""
            # Projected gravity (tells policy which way is up)
            quat = data.qpos[3:7]
            inv_quat = jp.array([quat[0], -quat[1], -quat[2], -quat[3]])
            gravity_body = brax_math.rotate(jp.array([0.0, 0.0, -1.0]), inv_quat)

            # Joint positions relative to default (not raw angles)
            joint_deltas = data.qpos[7:] - self._default_pose

            # Angular velocity (body frame)
            ang_vel = data.qvel[3:6]

            obs = jp.concatenate([
                gravity_body,         # 3: which way is up
                ang_vel * 0.25,       # 3: body rotation rate (scaled)
                joint_deltas,         # 12: joint offsets from standing
                data.qvel[6:],        # 12: joint velocities
                last_action,          # 12: what we did last step
            ])
            return jp.clip(obs, -100.0, 100.0)

    return Go2Locomotion()


# ── Report helpers ───────────────────────────────────────────────────────────

def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0f0f23")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'


def generate_training_charts(evals: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e["timestep"] for e in evals]
    rewards = [e["reward"] for e in evals]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    ax.plot(steps, rewards, color="#00b894", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Timesteps", color="#ccc")
    ax.set_ylabel("Mean Episode Reward", color="#ccc")
    ax.set_title("Unitree Go2 (MJX) — Training Progress", color="#fdcb6e", fontsize=14)
    ax.tick_params(colors="#888")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    html = fig_to_html(fig)
    plt.close(fig)
    return html


def generate_replay_html(episodes: list[dict]) -> str:
    uid = f"mj{id(episodes) % 100000}"

    options_html = ""
    for i, ep in enumerate(episodes):
        options_html += f'<option value="{i}">{ep["label"]} (reward: {ep["reward"]:.1f})</option>'

    return f"""
    <div style="font-family: monospace; background: #0f0f23; color: #ccc; padding: 20px; border-radius: 8px;">
      <h3 style="color: #fdcb6e; margin-top: 0;">Episode Replay</h3>
      <div style="margin-bottom: 10px;">
        <label>Episode:
          <select id="ep-{uid}" onchange="changeEp_{uid}()" style="background:#1a1a2e;color:#ccc;padding:4px;border:1px solid #333;">
            {options_html}
          </select>
        </label>
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
      var curEp = 0; var timer = null;
      window.changeEp_{uid} = function() {{
        curEp = parseInt(document.getElementById('ep-{uid}').value);
        var ep = EPS[curEp];
        var sl = document.getElementById('slider-{uid}');
        sl.max = Math.max(ep.frames.length - 1, 0); sl.value = 0;
        render_{uid}();
      }};
      window.render_{uid} = function() {{
        var ep = EPS[curEp];
        var idx = parseInt(document.getElementById('slider-{uid}').value);
        document.getElementById('step-{uid}').textContent = idx;
        if (ep.frames[idx]) document.getElementById('frame-{uid}').src = 'data:image/jpeg;base64,' + ep.frames[idx];
      }};
      window.play_{uid} = function() {{
        if (timer) {{ clearInterval(timer); timer = null; document.getElementById('btn-{uid}').textContent = 'Play'; }}
        else {{
          document.getElementById('btn-{uid}').textContent = 'Pause';
          timer = setInterval(function() {{
            var sl = document.getElementById('slider-{uid}');
            if (parseInt(sl.value) >= parseInt(sl.max)) {{ clearInterval(timer); timer = null; document.getElementById('btn-{uid}').textContent = 'Play'; }}
            else {{ sl.value = parseInt(sl.value) + 1; render_{uid}(); }}
          }}, 80);
        }}
      }};
      changeEp_{uid}();
    }})();
    </script>
    """


# ── GPU Training Task ────────────────────────────────────────────────────────

@gpu_env.task(retries=3)
async def train_on_gpu(
    num_timesteps: int = 20_000_000,
    num_envs: int = 4096,
    episode_length: int = 1000,
    num_evals: int = 10,
) -> File:
    """Train Go2 locomotion entirely on GPU using Brax PPO + MJX.

    4096 parallel environments on a single GPU. Full training in minutes.
    """
    import brax.training.agents.ppo.train as ppo_module
    import jax

    log.info(f"Training on {jax.devices()[0]} with {num_envs} parallel envs")
    log.info(f"Target: {num_timesteps:,} timesteps, {num_evals} evals")

    # Track training progress
    eval_results = []

    def progress_fn(step, metrics):
        reward = float(metrics.get("eval/episode_reward", 0.0))
        eval_results.append({"timestep": int(step), "reward": reward})
        log.info(f"  Step {step:>10,} | Reward: {reward:.1f}")

    env = _make_go2_env()
    log.info(f"Environment created: obs_size={env.observation_size}, action_size={env.action_size}")

    # PPO hyperparameters from Brax Barkour tutorial (working quadruped on MJX)
    make_inference_fn, params, metrics = ppo_module.train(
        environment=env,
        progress_fn=progress_fn,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=num_envs,
        batch_size=256,
        seed=0,
    )

    # Save inference function params + training history
    import pickle
    path = "/tmp/mjx_checkpoint.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "params": jax.device_get(params),
            "eval_results": eval_results,
            "num_timesteps": num_timesteps,
            "num_envs": num_envs,
        }, f)

    return await File.from_local(path)


# ── CPU Replay Rendering ────────────────────────────────────────────────────

@worker_env.task(report=True, retries=5)
async def render_replay(
    checkpoint: File,
    label: str = "Trained",
    num_episodes: int = 3,
    max_steps: int = 1000,
) -> File:
    """Render replay on CPU MuJoCo — reconstructs observations to match Brax env,
    applies action scaling (default_pose + action * 0.3), and renders frames.
    """
    import numpy as np
    import pickle
    import jax
    from jax import numpy as jnp
    from PIL import Image

    local = await checkpoint.download()
    with open(local, "rb") as f:
        ckpt = pickle.load(f)

    from brax.training.agents.ppo import networks as ppo_networks
    import mujoco

    mj_model = mujoco.MjModel.from_xml_path(GO2_XML_PATH)
    action_size = mj_model.nu  # 12

    # Obs: gravity(3) + ang_vel(3) + joint_deltas(12) + joint_vel(12) + last_action(12) = 42
    obs_size = 3 + 3 + 12 + 12 + 12

    # Default standing pose (joint angles only)
    default_pose = mj_model.keyframe("home").qpos[7:] if mj_model.nkey > 0 else mj_model.qpos0[7:]
    default_pose = np.array(default_pose, dtype=np.float32)

    # Joint limits
    lower = default_pose - np.array([0.2, 0.8, 0.8] * 4, dtype=np.float32)
    upper = default_pose + np.array([0.2, 0.8, 0.8] * 4, dtype=np.float32)

    # Apply same actuator tuning as training env
    mj_model.actuator_gainprm[:, 0] = 35.0
    mj_model.actuator_biasprm[:, 1] = -35.0
    mj_model.dof_damping[6:] = 0.5239

    # Rebuild network with normalizer (matches normalize_observations=True in training).
    # params = (normalizer_params, policy_params, value_params)
    from brax.training.acme import running_statistics
    ppo_network = ppo_networks.make_ppo_networks(
        obs_size, action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(ckpt["params"])

    best_reward = -float("inf")
    best_frames = []
    all_rewards = []
    frame_every = 5
    max_frames = 100

    def quat_rotate_inv(q, v):
        """Rotate vector v by inverse of quaternion q (w,x,y,z format)."""
        w, x, y, z = q
        # Inverse quaternion rotation
        t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
        return v + w * t + np.cross(np.array([-x, -y, -z]), t)

    for ep_idx in range(num_episodes):
        mj_model_render = mujoco.MjModel.from_xml_path(GO2_XML_PATH)
        mj_model_render.actuator_gainprm[:, 0] = 35.0
        mj_model_render.actuator_biasprm[:, 1] = -35.0
        mj_model_render.dof_damping[6:] = 0.5239

        mj_data = mujoco.MjData(mj_model_render)
        renderer = mujoco.Renderer(mj_model_render, width=320, height=240)

        if mj_model_render.nkey > 0:
            mj_data.qpos[:] = mj_model_render.key_qpos[0]
        mujoco.mj_forward(mj_model_render, mj_data)
        total_reward = 0.0
        frames = []
        last_action = np.zeros(action_size, dtype=np.float32)

        rng = jax.random.PRNGKey(ep_idx + 1000)

        for step in range(max_steps):
            if step % frame_every == 0 and len(frames) < max_frames:
                cam_names = [mj_model_render.cam(i).name for i in range(mj_model_render.ncam)]
                cam = "track" if "track" in cam_names else -1
                renderer.update_scene(mj_data, camera=cam)
                frame = renderer.render()
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="JPEG", quality=70)
                frames.append(base64.b64encode(buf.getvalue()).decode())

            # Build observation matching Brax env's _get_obs
            quat = mj_data.qpos[3:7]
            gravity_body = quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
            ang_vel = mj_data.qvel[3:6].astype(np.float32)
            joint_deltas = (mj_data.qpos[7:] - default_pose).astype(np.float32)
            joint_vel = mj_data.qvel[6:].astype(np.float32)

            obs = np.concatenate([
                gravity_body.astype(np.float32),
                ang_vel * 0.25,
                joint_deltas,
                joint_vel,
                last_action,
            ])
            obs = np.clip(obs, -100.0, 100.0)
            obs_jax = jnp.array(obs)

            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(obs_jax, act_rng)
            action_np = np.array(action, dtype=np.float32)
            last_action = action_np

            # Apply action scaling: default_pose + action * 0.3, clipped
            motor_targets = default_pose + action_np * 0.3
            motor_targets = np.clip(motor_targets, lower, upper)

            # Step CPU MuJoCo with n_frames=5 (matching training)
            for _ in range(5):
                mj_data.ctrl[:] = motor_targets
                mujoco.mj_step(mj_model_render, mj_data)

            x_vel = mj_data.qvel[0]
            z_pos = mj_data.qpos[2]
            total_reward += 3.0 * x_vel + 0.5
            done = z_pos < 0.18 or z_pos > 0.6

            if done:
                break

        renderer.close()
        all_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    mean_reward = sum(all_rewards) / len(all_rewards)
    log.info(f"[Replay] {label}: mean={mean_reward:.1f}, best={best_reward:.1f}")

    # Build report
    if best_frames:
        replay_html = generate_replay_html([
            {"label": label, "reward": best_reward, "frames_b64": best_frames},
        ])
        await flyte.report.replace.aio(
            f"<h3>{label}</h3>"
            f"<p>Mean reward: {mean_reward:.1f} | Best: {best_reward:.1f}</p>"
            + replay_html
        )
        await flyte.report.flush.aio()

    result_path = "/tmp/replay_result.json"
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
    num_timesteps: int = 20_000_000,
    num_envs: int = 4096,
    episode_length: int = 1000,
    num_evals: int = 10,
) -> str:
    """
    Train Unitree Go2 with GPU-accelerated MJX + Brax PPO.

    1. Train entirely on GPU (4096 parallel envs)
    2. Render replay on CPU
    3. Build final report with training curves + replay
    """
    import asyncio
    import pickle

    log.info(f"Starting MJX training: {num_timesteps:,} timesteps, {num_envs} envs")

    await flyte.report.replace.aio(
        "<h2>Unitree Go2 (MJX) — GPU Training</h2>"
        f"<p>Training with {num_envs:,} parallel environments on GPU...</p>"
    )
    await flyte.report.flush.aio()

    # Train on GPU
    checkpoint_file = await train_on_gpu(
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        episode_length=episode_length,
        num_evals=num_evals,
    )

    # Load training history
    local = await checkpoint_file.download()
    with open(local, "rb") as f:
        ckpt = pickle.load(f)
    eval_results = ckpt["eval_results"]

    log.info("Training complete. Rendering replay...")

    # Render replay on CPU worker
    replay_file = await render_replay(
        checkpoint=checkpoint_file,
        label="Trained Go2",
        num_episodes=5,
        max_steps=episode_length,
    )

    replay_path = await replay_file.download()
    with open(replay_path) as f:
        replay_data = json.load(f)

    # Build final report
    charts_html = generate_training_charts(eval_results) if eval_results else ""
    replay_html = ""
    if replay_data.get("frames_b64"):
        replays = [{"label": "Trained Go2", "reward": replay_data["best_reward"], "frames_b64": replay_data["frames_b64"]}]
        replay_html = generate_replay_html(replays)

        tab = flyte.report.get_tab("Trained Go2")
        frames_json = json.dumps(replay_data["frames_b64"])
        uid = "trained"
        tab.log(
            f"<h3>Trained Go2</h3><p>Reward: {replay_data['best_reward']:.1f}</p>"
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

    best_reward = max((e["reward"] for e in eval_results), default=0.0)
    final_reward = eval_results[-1]["reward"] if eval_results else 0.0

    summary_html = (
        '<div style="font-family:monospace;background:#0f0f23;color:#ccc;padding:20px;border-radius:8px;">'
        '<h3 style="color:#00b894;margin-top:0;">Training Complete</h3>'
        '<table style="border-collapse:collapse;width:100%;">'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Robot</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">Unitree Go2 (MJX/GPU)</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Timesteps</td><td style="padding:6px;border-bottom:1px solid #333;">{num_timesteps:,}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Parallel envs</td><td style="padding:6px;border-bottom:1px solid #333;">{num_envs:,}</td></tr>'
        f'<tr><td style="padding:6px;border-bottom:1px solid #333;">Best reward</td><td style="padding:6px;border-bottom:1px solid #333;color:#00b894;">{best_reward:.1f}</td></tr>'
        f'<tr><td style="padding:6px;">Final reward</td><td style="padding:6px;">{final_reward:.1f}</td></tr>'
        '</table></div>'
    )

    await flyte.report.replace.aio(
        "<h2>Unitree Go2 (MJX) — Training Complete</h2>"
        f"{summary_html}<br/>"
        f"<h3 style='color:#fdcb6e;font-family:monospace;'>Training Progress</h3>{charts_html}<br/>"
        + (f"<h3 style='color:#fdcb6e;font-family:monospace;'>Episode Replay</h3>{replay_html}" if replay_html else "")
    )
    await flyte.report.flush.aio()

    log.info(f"Done. Best reward: {best_reward:.1f}")

    return json.dumps({
        "robot": "Unitree Go2 (MJX)",
        "num_timesteps": num_timesteps,
        "num_envs": num_envs,
        "best_reward": best_reward,
        "eval_results": eval_results,
    })
