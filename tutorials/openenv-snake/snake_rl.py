"""Train an LLM to play Snake via GRPO using a custom OpenEnv environment + Flyte.

Demonstrates:
- Building a custom OpenEnv environment (Snake game)
- Running the env server locally and connecting via EnvClient
- GRPO training loop with distance-based reward shaping
- Visual HTML replay with frame slider embedded in Flyte reports

Run the env server (separate terminal):
  cd tutorials/openenv-snake && python -m snake_env.server.app

Run locally:
  flyte run --local snake_rl.py pipeline --training_steps 3

Run on a cluster:
  flyte run snake_rl.py pipeline --training_steps 10
"""

import base64
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import flyte
import flyte.report
from flyte.io import File
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server import Action, Observation, State
from pydantic import Field as PydanticField

# ---------------------------------------------------------------------------
# Models (shared with snake_env server)
# ---------------------------------------------------------------------------


class SnakeAction(Action):
    """Action: choose a direction to move the snake."""
    direction: str = "RIGHT"


class SnakeObservation(Observation):
    """What the agent sees after each step."""
    grid: List[List[str]] = PydanticField(default_factory=list)
    snake: List[Tuple[int, int]] = PydanticField(default_factory=list)
    apple: Tuple[int, int] = (0, 0)
    score: int = 0
    death_reason: str = ""


class SnakeState(State):
    """Metadata about the current episode."""
    score: int = 0
    grid_size: int = 10


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SnakeEnv(EnvClient[SnakeAction, SnakeObservation, SnakeState]):
    """Client that connects to a Snake OpenEnv server."""

    def _step_payload(self, action: SnakeAction) -> dict:
        return {"direction": action.direction}

    def _parse_result(self, payload: dict) -> StepResult[SnakeObservation]:
        obs_data = payload.get("observation", payload)
        observation = SnakeObservation(
            grid=obs_data.get("grid", []),
            snake=[tuple(p) for p in obs_data.get("snake", [])],
            apple=tuple(obs_data.get("apple", (0, 0))),
            score=obs_data.get("score", 0),
            death_reason=obs_data.get("death_reason", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SnakeState:
        return SnakeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            score=payload.get("score", 0),
            grid_size=payload.get("grid_size", 10),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="snake_rl",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch",
        "transformers",
        "openenv-core",
        "matplotlib",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENV_URL = "http://localhost:8000"

SYSTEM_PROMPT = (
    "You are playing Snake on a 10x10 grid.\n"
    "The grid uses: H=head, S=body, A=apple, .=empty\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Move toward the apple (A) while avoiding walls and your own body (S)."
)

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# ---------------------------------------------------------------------------
# Episode recording for visual replay
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFrame:
    step: int
    grid: list[list[str]]
    score: int
    action: str
    reward: float


@dataclass
class EpisodeRecording:
    frames: list[EpisodeFrame] = field(default_factory=list)
    total_reward: float = 0.0
    final_score: int = 0
    death_reason: str = ""
    length: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory():
    """Free GPU/MPS memory between tasks."""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


def format_observation(grid: list[list[str]], snake, apple, score: int) -> str:
    """Render the grid as text for the LLM."""
    rows = [" ".join(row) for row in grid]
    grid_text = "\n".join(rows)
    head = snake[0] if snake else (0, 0)
    return (
        f"Grid:\n{grid_text}\n\n"
        f"Head position: row={head[0]}, col={head[1]}\n"
        f"Apple position: row={apple[0]}, col={apple[1]}\n"
        f"Score: {score}\n"
        f"Snake length: {len(snake)}\n"
        "Which direction? Reply UP, DOWN, LEFT, or RIGHT."
    )


def parse_direction(text: str) -> str:
    """Extract direction from model output."""
    upper = text.strip().upper()
    for d in DIRECTIONS:
        if d in upper:
            return d
    return random.choice(DIRECTIONS)


def create_snake_client(env_url: str):
    """Create a connected Snake env client."""
    client = SnakeEnv(
        base_url=env_url,
        connect_timeout_s=30.0,
        message_timeout_s=300.0,  # 5 min — long episodes on slow inference
    )
    client.connect()
    return client


def safe_step(client, action, env_url: str):
    """Step with reconnect on WebSocket drop."""
    try:
        return client.step(action)
    except Exception:
        client.close()
        client.connect()
        # After reconnect we've lost the episode, reset and return done
        result = client.reset()
        result.observation.done = True
        result.observation.reward = -1.0
        result.observation.death_reason = "disconnect"
        result.done = True
        return result


def start_env_server():
    """Start the Snake env server as a background process. Returns the process."""
    server_script = os.path.join(os.path.dirname(__file__), "snake_env", "server", "app.py")
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give server time to start
    time.sleep(3)
    return proc


# ---------------------------------------------------------------------------
# HTML Replay Generator
# ---------------------------------------------------------------------------


def generate_replay_html(recordings: list[EpisodeRecording], title: str = "Snake Replay") -> str:
    """Generate HTML with a slider to step through recorded Snake episodes."""

    CELL_COLORS = {
        ".": "#1a1a2e",
        "S": "#00b894",
        "H": "#fdcb6e",
        "A": "#e17055",
    }

    episodes_json = []
    for rec in recordings:
        frames = []
        for f in rec.frames:
            frames.append({
                "step": f.step,
                "grid": f.grid,
                "score": f.score,
                "action": f.action,
                "reward": round(f.reward, 3),
            })
        episodes_json.append({
            "frames": frames,
            "total_reward": round(rec.total_reward, 3),
            "final_score": rec.final_score,
            "death_reason": rec.death_reason,
            "length": rec.length,
        })

    return f"""
    <div style="font-family: monospace; background: #0f0f23; color: #ccc; padding: 20px; border-radius: 8px;">
      <h3 style="color: #fdcb6e; margin-top: 0;">{title}</h3>
      <div style="margin-bottom: 10px;">
        <label>Episode:
          <select id="ep-select" onchange="changeEpisode()" style="background:#1a1a2e;color:#ccc;padding:4px;border:1px solid #333;">
          </select>
        </label>
        <span id="ep-info" style="margin-left: 15px;"></span>
      </div>
      <div style="margin-bottom: 10px;">
        <label>Step: <span id="step-label">0</span></label><br>
        <input type="range" id="step-slider" min="0" max="0" value="0" oninput="renderFrame()"
               style="width: 300px;">
        <button onclick="playReplay()" id="play-btn" style="margin-left:10px;padding:4px 12px;background:#00b894;color:#fff;border:none;border-radius:4px;cursor:pointer;">Play</button>
      </div>
      <div style="display:flex; gap: 20px; align-items: flex-start;">
        <canvas id="snake-canvas" width="300" height="300" style="border: 2px solid #333; border-radius: 4px;"></canvas>
        <div id="frame-info" style="font-size: 14px; line-height: 1.6;"></div>
      </div>
    </div>

    <script>
    const EPISODES = {json.dumps(episodes_json)};
    const COLORS = {json.dumps(CELL_COLORS)};
    let currentEp = 0;
    let playInterval = null;

    function init() {{
      const sel = document.getElementById('ep-select');
      EPISODES.forEach((ep, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.text = 'Episode ' + (i+1) + ' (score: ' + ep.final_score + ')';
        sel.appendChild(opt);
      }});
      changeEpisode();
    }}

    function changeEpisode() {{
      currentEp = parseInt(document.getElementById('ep-select').value);
      const ep = EPISODES[currentEp];
      const slider = document.getElementById('step-slider');
      slider.max = Math.max(ep.frames.length - 1, 0);
      slider.value = 0;
      document.getElementById('ep-info').textContent =
        'Score: ' + ep.final_score + ' | Reward: ' + ep.total_reward +
        (ep.death_reason ? ' | Died: ' + ep.death_reason : ' | Survived');
      renderFrame();
    }}

    function renderFrame() {{
      const ep = EPISODES[currentEp];
      const idx = parseInt(document.getElementById('step-slider').value);
      const frame = ep.frames[idx];
      if (!frame) return;

      document.getElementById('step-label').textContent = frame.step;

      const canvas = document.getElementById('snake-canvas');
      const ctx = canvas.getContext('2d');
      const grid = frame.grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          ctx.fillStyle = COLORS[grid[r][c]] || '#1a1a2e';
          ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
        }}
      }}

      document.getElementById('frame-info').innerHTML =
        'Action: <b>' + frame.action + '</b><br>' +
        'Reward: ' + frame.reward + '<br>' +
        'Score: ' + frame.score;
    }}

    function playReplay() {{
      if (playInterval) {{
        clearInterval(playInterval);
        playInterval = null;
        document.getElementById('play-btn').textContent = 'Play';
        return;
      }}
      document.getElementById('play-btn').textContent = 'Pause';
      const slider = document.getElementById('step-slider');
      playInterval = setInterval(() => {{
        let val = parseInt(slider.value);
        if (val >= parseInt(slider.max)) {{
          clearInterval(playInterval);
          playInterval = null;
          document.getElementById('play-btn').textContent = 'Play';
          return;
        }}
        slider.value = val + 1;
        renderFrame();
      }}, 200);
    }}

    init();
    </script>
    """


# ---------------------------------------------------------------------------
# Game-playing helpers
# ---------------------------------------------------------------------------


def play_episode_record(client, model, tokenizer, device, temperature=0.7):
    """Play one Snake episode. Returns (trajectory, shaped_reward, recording)."""
    import torch


    try:
        result = client.reset()
    except Exception:
        client.connect()
        result = client.reset()

    trajectory = []
    recording = EpisodeRecording()
    total_reward = 0.0

    # Record initial frame
    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, score=obs.score, action="START", reward=0.0,
    ))

    step_num = 0
    while not result.done and step_num < 200:
        obs = result.observation
        user_prompt = format_observation(obs.grid, obs.snake, obs.apple, obs.score)

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

        direction = parse_direction(gen_text)
        result = safe_step(client, SnakeAction(direction=direction), "")

        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        trajectory.append({
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "completion_ids": gen_ids.tolist(),
            "log_probs": log_probs,
            "action": gen_text.strip(),
        })

        # Record frame
        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid, score=obs.score,
            action=direction, reward=reward,
        ))

    recording.total_reward = total_reward
    recording.final_score = result.observation.score
    recording.death_reason = result.observation.death_reason
    recording.length = step_num

    return trajectory, total_reward, recording


def play_episode_baseline(client, policy="random"):
    """Play one Snake episode with a simple policy. Returns recording."""


    try:
        result = client.reset()
    except Exception:
        client.connect()
        result = client.reset()

    recording = EpisodeRecording()
    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, score=obs.score, action="START", reward=0.0,
    ))

    total_reward = 0.0
    step_num = 0

    while not result.done and step_num < 200:
        obs = result.observation

        if policy == "random":
            direction = random.choice(DIRECTIONS)
        elif policy == "greedy":
            # Move toward apple, avoiding immediate death
            head_r, head_c = obs.snake[0]
            apple_r, apple_c = obs.apple
            body_set = set(map(tuple, obs.snake))

            candidates = []
            for d in DIRECTIONS:
                dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[d]
                nr, nc = head_r + dr, head_c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and (nr, nc) not in body_set:
                    dist = abs(nr - apple_r) + abs(nc - apple_c)
                    candidates.append((dist, d))
            if candidates:
                candidates.sort()
                direction = candidates[0][1]
            else:
                direction = random.choice(DIRECTIONS)
        else:
            direction = random.choice(DIRECTIONS)

        result = safe_step(client, SnakeAction(direction=direction), "")
        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid, score=obs.score,
            action=direction, reward=reward,
        ))

    recording.total_reward = total_reward
    recording.final_score = result.observation.score
    recording.death_reason = result.observation.death_reason
    recording.length = step_num

    return recording


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


@env.task(cache="auto")
async def eval_baselines(
    env_url: str = DEFAULT_ENV_URL, num_episodes: int = 50
) -> str:
    """Play Snake with random and greedy policies to set baselines."""


    results = {}
    best_recordings = {}

    for policy in ["random", "greedy"]:
        scores = []
        best_rec = None
        best_score = -1
        client = create_snake_client(env_url)
        try:
            for _ in range(num_episodes):
                rec = play_episode_baseline(client, policy=policy)
                scores.append(rec.final_score)
                if rec.final_score > best_score:
                    best_score = rec.final_score
                    best_rec = rec
        finally:
            client.close()

        avg_score = sum(scores) / len(scores)
        results[policy] = {
            "avg_score": avg_score,
            "max_score": max(scores),
            "avg_length": sum(1 for _ in scores) / len(scores),
        }
        if best_rec:
            best_recordings[policy] = {
                "frames": [{"step": f.step, "grid": f.grid, "score": f.score,
                            "action": f.action, "reward": f.reward}
                           for f in best_rec.frames],
                "total_reward": best_rec.total_reward,
                "final_score": best_rec.final_score,
                "death_reason": best_rec.death_reason,
                "length": best_rec.length,
            }
        print(f"  {policy}: avg_score={avg_score:.2f}, max={max(scores)}")

    return json.dumps({"results": results, "recordings": best_recordings})


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------


@env.task
async def train_step(
    model_path: str,
    env_url: str = DEFAULT_ENV_URL,
    num_rollouts: int = 8,
    group_size: int = 4,
    lr: float = 1e-5,
    step_idx: int = 0,
    checkpoint_file: File | None = None,
    use_bfloat16: bool = True,
    gradient_checkpointing: bool = True,
) -> tuple[File, str]:
    """One GRPO iteration for Snake."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # If a checkpoint File is provided, download and extract it
    if checkpoint_file is not None:
        local_tar = await checkpoint_file.download()
        extract_dir = f"prev_checkpoint_{step_idx}"
        shutil.unpack_archive(local_tar, extract_dir)
        model_path = os.path.join(extract_dir, f"checkpoint_step_{step_idx - 1}")

    device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"  Step {step_idx} | device={device} | dtype={dtype} | grad_ckpt={gradient_checkpointing}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device)
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    client = create_snake_client(env_url)

    all_rewards = []
    all_scores = []
    total_loss = 0.0
    num_groups = max(num_rollouts // group_size, 1)

    try:
        for g in range(num_groups):
            group_trajs = []
            group_rewards = []

            for _ in range(group_size):
                traj, reward, rec = play_episode_record(
                    client, model, tokenizer, device
                )
                group_trajs.append(traj)
                group_rewards.append(reward)
                all_scores.append(rec.final_score)

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
                    if not step_data["completion_ids"]:
                        continue
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

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    # Save checkpoint
    save_dir = f"checkpoint_step_{step_idx}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    tar_path = f"{save_dir}.tar.gz"
    shutil.make_archive(save_dir, "gztar", ".", save_dir)
    checkpoint_file = await File.from_local(tar_path)

    metrics = json.dumps({
        "step": step_idx,
        "avg_reward": avg_reward,
        "avg_score": avg_score,
        "loss": total_loss / num_groups,
    })
    print(f"  Step {step_idx} | avg_score={avg_score:.2f} avg_reward={avg_reward:.2f}")

    # Free GPU/MPS memory before returning
    del model, optimizer
    cleanup_memory()

    return checkpoint_file, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@env.task
async def eval_model(
    model_path: str,
    env_url: str = DEFAULT_ENV_URL,
    num_episodes: int = 20,
    step_idx: int = 0,
    checkpoint_file: File | None = None,
    use_bfloat16: bool = True,
) -> str:
    """Evaluate the model on Snake, returning scores and best replay."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # If a checkpoint File is provided, download and extract it
    if checkpoint_file is not None:
        local_tar = await checkpoint_file.download()
        extract_dir = f"eval_checkpoint_{step_idx}"
        shutil.unpack_archive(local_tar, extract_dir)
        model_path = os.path.join(extract_dir, f"checkpoint_step_{step_idx}")

    device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device)
    model.eval()

    client = create_snake_client(env_url)
    scores = []
    direction_counts = {d: 0 for d in DIRECTIONS}
    best_rec = None
    best_score = -1

    try:
        for _ in range(num_episodes):
            traj, reward, rec = play_episode_record(
                client, model, tokenizer, device, temperature=0.0
            )
            scores.append(rec.final_score)
            if rec.final_score > best_score:
                best_score = rec.final_score
                best_rec = rec
            for s in traj:
                d = parse_direction(s["action"])
                direction_counts[d] += 1
    finally:
        client.close()

    avg_score = sum(scores) / len(scores)

    # Serialize best recording
    best_replay = None
    if best_rec:
        best_replay = {
            "frames": [{"step": f.step, "grid": f.grid, "score": f.score,
                        "action": f.action, "reward": f.reward}
                       for f in best_rec.frames],
            "total_reward": best_rec.total_reward,
            "final_score": best_rec.final_score,
            "death_reason": best_rec.death_reason,
            "length": best_rec.length,
        }

    # Free GPU/MPS memory before returning
    del model
    cleanup_memory()

    return json.dumps({
        "step": step_idx,
        "avg_score": avg_score,
        "max_score": max(scores),
        "scores": scores,
        "direction_distribution": direction_counts,
        "best_replay": best_replay,
    })


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@env.task(report=True)
async def pipeline(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    env_url: str = DEFAULT_ENV_URL,
    training_steps: int = 3,
    rollouts_per_step: int = 8,
    group_size: int = 4,
    eval_episodes: int = 20,
    lr: float = 1e-5,
    use_bfloat16: bool = True,
    gradient_checkpointing: bool = True,
    open_report: bool = False,
) -> tuple[str, File]:
    """Full Snake RL pipeline: baselines -> GRPO training -> eval -> visual report."""



    device = get_device()
    print(f"Device: {device}")
    print(f"Model:  {model_id}")
    print(f"Env:    {env_url}\n")

    # Start env server if using localhost
    server_proc = None
    if "localhost" in env_url or "127.0.0.1" in env_url:
        print("Starting local Snake env server...")
        server_proc = start_env_server()
        print("Server started.\n")

    try:
        # 1. Baselines
        print("=== Evaluating baselines ===")
        baselines_json = json.loads(await eval_baselines(env_url, num_episodes=50))
        baselines = baselines_json["results"]
        baseline_recordings = baselines_json.get("recordings", {})
        print(f"  Random: avg_score={baselines['random']['avg_score']:.2f}")
        print(f"  Greedy: avg_score={baselines['greedy']['avg_score']:.2f}")

        # 2. Evaluate untrained model
        print("\n=== Evaluating untrained model ===")
        eval_results = [json.loads(await eval_model(model_id, env_url, eval_episodes, 0, use_bfloat16=use_bfloat16))]
        print(f"  Untrained: avg_score={eval_results[0]['avg_score']:.2f}")

        # 3. Training loop
        prev_checkpoint = None
        train_metrics = []

        for step in range(1, training_steps + 1):
            print(f"\n=== Training step {step}/{training_steps} ===")

            checkpoint_file, metrics_json = await train_step(
                model_id,
                env_url,
                num_rollouts=rollouts_per_step,
                group_size=group_size,
                lr=lr,
                step_idx=step,
                checkpoint_file=prev_checkpoint,
                use_bfloat16=use_bfloat16,
                gradient_checkpointing=gradient_checkpointing,
            )
            train_metrics.append(json.loads(metrics_json))

            # Eval the checkpoint (pass File so remote task can download it)
            eval_json = await eval_model(
                model_id, env_url, eval_episodes, step,
                checkpoint_file=checkpoint_file,
                use_bfloat16=use_bfloat16,
            )
            eval_results.append(json.loads(eval_json))
            prev_checkpoint = checkpoint_file
            print(f"  Eval avg_score: {eval_results[-1]['avg_score']:.2f}")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    # 4. Generate report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_list = [e["step"] for e in eval_results]
    avg_scores = [e["avg_score"] for e in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Average score over training
    ax = axes[0]
    ax.plot(steps_list, avg_scores, "b-o", markersize=6, linewidth=2, label="GRPO Agent")
    ax.axhline(
        baselines["random"]["avg_score"],
        color="r", linestyle="--",
        label=f"Random ({baselines['random']['avg_score']:.2f})",
    )
    ax.axhline(
        baselines["greedy"]["avg_score"],
        color="g", linestyle="--",
        label=f"Greedy ({baselines['greedy']['avg_score']:.2f})",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Score (Apples)")
    ax.set_title("Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Training reward
    ax = axes[1]
    if train_metrics:
        ax.plot(
            [m["step"] for m in train_metrics],
            [m["avg_reward"] for m in train_metrics],
            "m-o", markersize=6,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Training Reward")
    ax.grid(True, alpha=0.3)

    # Chart 3: Direction distribution
    ax = axes[2]
    for d in DIRECTIONS:
        fracs = []
        for e in eval_results:
            dist = e.get("direction_distribution", {})
            total = sum(dist.values())
            fracs.append(dist.get(d, 0) / max(total, 1))
        ax.plot(steps_list, fracs, "-o", markersize=4, label=d)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Direction Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    # Build replay HTML from best episodes
    replay_recs = []
    # Add baseline best recordings
    for policy_name, rec_data in baseline_recordings.items():
        rec = EpisodeRecording(
            frames=[EpisodeFrame(**f) for f in rec_data["frames"]],
            total_reward=rec_data["total_reward"],
            final_score=rec_data["final_score"],
            death_reason=rec_data["death_reason"],
            length=rec_data["length"],
        )
        replay_recs.append(rec)

    # Add model best replays
    for e in eval_results:
        replay_data = e.get("best_replay")
        if replay_data:
            rec = EpisodeRecording(
                frames=[EpisodeFrame(**f) for f in replay_data["frames"]],
                total_reward=replay_data["total_reward"],
                final_score=replay_data["final_score"],
                death_reason=replay_data["death_reason"],
                length=replay_data["length"],
            )
            replay_recs.append(rec)

    replay_html = generate_replay_html(replay_recs, title="Snake Game Replays")

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Snake RL Training Report</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Avg Score</th><th>Max Score</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['avg_score']:.2f}</td><td>{baselines['random']['max_score']}</td></tr>"
        f"<tr><td>Greedy</td><td>{baselines['greedy']['avg_score']:.2f}</td><td>{baselines['greedy']['max_score']}</td></tr>"
        f"<tr><td><b>GRPO Agent (step 0)</b></td><td>{eval_results[0]['avg_score']:.2f}</td><td>{eval_results[0]['max_score']}</td></tr>"
        f"<tr><td><b>GRPO Agent (final)</b></td><td><b>{final['avg_score']:.2f}</b></td><td><b>{final['max_score']}</b></td></tr>"
        f"</table>"
        f"<h3>Training Progress</h3>{charts_html}"
        f"<h3>Visual Replay</h3>{replay_html}"
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
        f"Final avg_score: {final['avg_score']:.2f} "
        f"(random: {baselines['random']['avg_score']:.2f}, "
        f"greedy: {baselines['greedy']['avg_score']:.2f})"
    )
    return summary, checkpoint_file


# Server:  cd tutorials/openenv-snake && python -m snake_env.server.app
# Local:   flyte run --local snake_rl.1py pipeline --training_steps 3
# Remote:  flyte run snake_rl.py pipeline --training_steps 10