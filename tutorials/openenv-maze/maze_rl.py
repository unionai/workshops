"""Train an LLM to navigate mazes via GRPO using a custom OpenEnv environment + Flyte.

Demonstrates:
- Building a custom OpenEnv environment (maze with DFS generation)
- Running the env server locally and connecting via EnvClient
- GRPO training loop with distance/wall/revisit reward shaping
- Visual HTML replay with path trace embedded in Flyte reports

Run the env server (separate terminal):
  cd tutorials/openenv-maze && python -m maze_env.server.app

Run locally:
  flyte run --local maze_rl.py pipeline --training_steps 3

Run on a cluster:
  flyte run maze_rl.py pipeline --training_steps 10
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
# Models (shared with maze_env server)
# ---------------------------------------------------------------------------


class MazeAction(Action):
    """Action: choose a direction to move in the maze."""
    direction: str = "RIGHT"


class MazeObservation(Observation):
    """What the agent sees after each step."""
    grid: List[List[str]] = PydanticField(default_factory=list)
    agent_pos: Tuple[int, int] = (1, 1)
    exit_pos: Tuple[int, int] = (5, 5)
    steps_taken: int = 0


class MazeState(State):
    """Metadata about the current maze episode."""
    maze_seed: int = 0
    optimal_path_length: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    """Client that connects to a Maze OpenEnv server."""

    def _step_payload(self, action: MazeAction) -> dict:
        return {"direction": action.direction}

    def _parse_result(self, payload: dict) -> StepResult[MazeObservation]:
        obs_data = payload.get("observation", payload)
        observation = MazeObservation(
            grid=obs_data.get("grid", []),
            agent_pos=tuple(obs_data.get("agent_pos", (1, 1))),
            exit_pos=tuple(obs_data.get("exit_pos", (6, 6))),
            steps_taken=obs_data.get("steps_taken", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MazeState:
        return MazeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            maze_seed=payload.get("maze_seed", 0),
            optimal_path_length=payload.get("optimal_path_length", 0),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="maze_rl",
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
    "You are navigating an 8x8 maze.\n"
    "The grid uses: # = wall, . = open path, A = you (agent), E = exit\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Find the shortest path from A to E while avoiding walls (#)."
)

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# ---------------------------------------------------------------------------
# Episode recording for visual replay
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFrame:
    step: int
    grid: list[list[str]]
    agent_pos: tuple[int, int]
    action: str
    reward: float


@dataclass
class EpisodeRecording:
    frames: list[EpisodeFrame] = field(default_factory=list)
    total_reward: float = 0.0
    solved: bool = False
    steps_to_solve: int = 0
    length: int = 0


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


def format_observation(grid: list[list[str]], agent_pos, exit_pos, steps_taken: int) -> str:
    """Render the maze grid as text for the LLM."""
    rows = [" ".join(row) for row in grid]
    grid_text = "\n".join(rows)
    return (
        f"Maze:\n{grid_text}\n\n"
        f"Your position: row={agent_pos[0]}, col={agent_pos[1]}\n"
        f"Exit position: row={exit_pos[0]}, col={exit_pos[1]}\n"
        f"Steps taken: {steps_taken}\n"
        "Which direction? Reply UP, DOWN, LEFT, or RIGHT."
    )


def parse_direction(text: str) -> str:
    """Extract direction from model output."""
    upper = text.strip().upper()
    for d in DIRECTIONS:
        if d in upper:
            return d
    return random.choice(DIRECTIONS)


def create_maze_client(env_url: str):
    """Create a connected Maze env client."""
    client = MazeEnv(
        base_url=env_url,
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
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
        result = client.reset()
        result.observation.done = True
        result.observation.reward = -1.0
        result.done = True
        return result


def start_env_server():
    """Start the Maze env server as a background process."""
    server_script = os.path.join(os.path.dirname(__file__), "maze_env", "server", "app.py")
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    return proc


# ---------------------------------------------------------------------------
# HTML Replay Generator
# ---------------------------------------------------------------------------


def generate_replay_html(recordings: list[EpisodeRecording], title: str = "Maze Replay") -> str:
    """Generate HTML with canvas rendering, path trace overlay, and frame slider."""

    CELL_COLORS = {
        "#": "#2d3436",
        ".": "#dfe6e9",
        "A": "#fdcb6e",
        "E": "#00b894",
    }

    episodes_json = []
    for rec in recordings:
        frames = []
        for f in rec.frames:
            frames.append({
                "step": f.step,
                "grid": f.grid,
                "agent_pos": list(f.agent_pos),
                "action": f.action,
                "reward": round(f.reward, 3),
            })
        episodes_json.append({
            "frames": frames,
            "total_reward": round(rec.total_reward, 3),
            "solved": rec.solved,
            "steps_to_solve": rec.steps_to_solve,
            "length": rec.length,
        })

    return f"""
    <div style="font-family: monospace; background: #0f0f23; color: #ccc; padding: 20px; border-radius: 8px;">
      <h3 style="color: #fdcb6e; margin-top: 0;">{title}</h3>
      <div style="margin-bottom: 10px;">
        <label>Episode:
          <select id="maze-ep-select" onchange="mazeChangeEpisode()" style="background:#1a1a2e;color:#ccc;padding:4px;border:1px solid #333;">
          </select>
        </label>
        <span id="maze-ep-info" style="margin-left: 15px;"></span>
      </div>
      <div style="margin-bottom: 10px;">
        <label>Step: <span id="maze-step-label">0</span></label><br>
        <input type="range" id="maze-step-slider" min="0" max="0" value="0" oninput="mazeRenderFrame()"
               style="width: 300px;">
        <button onclick="mazePlayReplay()" id="maze-play-btn" style="margin-left:10px;padding:4px 12px;background:#00b894;color:#fff;border:none;border-radius:4px;cursor:pointer;">Play</button>
      </div>
      <div style="display:flex; gap: 20px; align-items: flex-start;">
        <canvas id="maze-canvas" width="320" height="320" style="border: 2px solid #333; border-radius: 4px;"></canvas>
        <div id="maze-frame-info" style="font-size: 14px; line-height: 1.6;"></div>
      </div>
    </div>

    <script>
    const MAZE_EPISODES = {json.dumps(episodes_json)};
    const MAZE_COLORS = {json.dumps(CELL_COLORS)};
    let mazeCurrentEp = 0;
    let mazePlayInterval = null;

    function mazeInit() {{
      const sel = document.getElementById('maze-ep-select');
      MAZE_EPISODES.forEach((ep, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.text = 'Episode ' + (i+1) + (ep.solved ? ' (SOLVED in ' + ep.steps_to_solve + ')' : ' (failed)');
        sel.appendChild(opt);
      }});
      mazeChangeEpisode();
    }}

    function mazeChangeEpisode() {{
      mazeCurrentEp = parseInt(document.getElementById('maze-ep-select').value);
      const ep = MAZE_EPISODES[mazeCurrentEp];
      const slider = document.getElementById('maze-step-slider');
      slider.max = Math.max(ep.frames.length - 1, 0);
      slider.value = 0;
      document.getElementById('maze-ep-info').textContent =
        (ep.solved ? 'SOLVED in ' + ep.steps_to_solve + ' steps' : 'Failed (' + ep.length + ' steps)') +
        ' | Reward: ' + ep.total_reward;
      mazeRenderFrame();
    }}

    function mazeRenderFrame() {{
      const ep = MAZE_EPISODES[mazeCurrentEp];
      const idx = parseInt(document.getElementById('maze-step-slider').value);
      const frame = ep.frames[idx];
      if (!frame) return;

      document.getElementById('maze-step-label').textContent = frame.step;

      const canvas = document.getElementById('maze-canvas');
      const ctx = canvas.getContext('2d');
      const grid = frame.grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw base grid
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const cell = grid[r][c];
          ctx.fillStyle = MAZE_COLORS[cell] || '#dfe6e9';
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
          // Grid lines
          ctx.strokeStyle = '#636e72';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
        }}
      }}

      // Draw path trace — show visited cells as faded blue dots
      for (let i = 1; i <= idx; i++) {{
        const prevFrame = ep.frames[i];
        if (prevFrame && prevFrame.agent_pos) {{
          const pr = prevFrame.agent_pos[0];
          const pc = prevFrame.agent_pos[1];
          // Only draw on open cells (not walls)
          if (grid[pr] && grid[pr][pc] !== '#') {{
            ctx.fillStyle = 'rgba(116, 185, 255, 0.4)';
            ctx.beginPath();
            ctx.arc(pc * cellW + cellW/2, pr * cellH + cellH/2, cellW * 0.2, 0, Math.PI * 2);
            ctx.fill();
          }}
        }}
      }}

      // Draw exit marker
      const exitR = ep.frames[0].grid.length - 2;
      const exitC = ep.frames[0].grid[0].length - 2;
      ctx.fillStyle = '#00b894';
      ctx.fillRect(exitC * cellW + 2, exitR * cellH + 2, cellW - 4, cellH - 4);
      ctx.fillStyle = '#fff';
      ctx.font = Math.floor(cellW * 0.5) + 'px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('E', exitC * cellW + cellW/2, exitR * cellH + cellH/2);

      // Draw agent
      if (frame.agent_pos) {{
        const ar = frame.agent_pos[0];
        const ac = frame.agent_pos[1];
        ctx.fillStyle = '#fdcb6e';
        ctx.beginPath();
        ctx.arc(ac * cellW + cellW/2, ar * cellH + cellH/2, cellW * 0.35, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#2d3436';
        ctx.font = Math.floor(cellW * 0.4) + 'px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('A', ac * cellW + cellW/2, ar * cellH + cellH/2);
      }}

      document.getElementById('maze-frame-info').innerHTML =
        'Action: <b>' + frame.action + '</b><br>' +
        'Reward: ' + frame.reward + '<br>' +
        'Step: ' + frame.step;
    }}

    function mazePlayReplay() {{
      if (mazePlayInterval) {{
        clearInterval(mazePlayInterval);
        mazePlayInterval = null;
        document.getElementById('maze-play-btn').textContent = 'Play';
        return;
      }}
      document.getElementById('maze-play-btn').textContent = 'Pause';
      const slider = document.getElementById('maze-step-slider');
      mazePlayInterval = setInterval(() => {{
        let val = parseInt(slider.value);
        if (val >= parseInt(slider.max)) {{
          clearInterval(mazePlayInterval);
          mazePlayInterval = null;
          document.getElementById('maze-play-btn').textContent = 'Play';
          return;
        }}
        slider.value = val + 1;
        mazeRenderFrame();
      }}, 200);
    }}

    mazeInit();
    </script>
    """


# ---------------------------------------------------------------------------
# Game-playing helpers
# ---------------------------------------------------------------------------


def play_episode_record(client, model, tokenizer, device, temperature=0.7):
    """Play one maze episode. Returns (trajectory, shaped_reward, recording)."""
    import torch

    try:
        result = client.reset()
    except Exception:
        client.connect()
        result = client.reset()

    trajectory = []
    recording = EpisodeRecording()
    total_reward = 0.0

    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, agent_pos=tuple(obs.agent_pos),
        action="START", reward=0.0,
    ))

    step_num = 0
    solved = False
    while not result.done and step_num < 100:
        obs = result.observation
        user_prompt = format_observation(obs.grid, obs.agent_pos, obs.exit_pos, obs.steps_taken)

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

        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(gen_ids):
                lp = torch.log_softmax(score[0], dim=-1)
                log_probs.append(lp[gen_ids[i]].item())

        direction = parse_direction(gen_text)
        result = safe_step(client, MazeAction(direction=direction), "")

        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        trajectory.append({
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "completion_ids": gen_ids.tolist(),
            "log_probs": log_probs,
            "action": gen_text.strip(),
        })

        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid,
            agent_pos=tuple(obs.agent_pos),
            action=direction, reward=reward,
        ))

        # Check if solved (reward of 10.0 = reached exit)
        if reward >= 10.0:
            solved = True

    recording.total_reward = total_reward
    recording.solved = solved
    recording.steps_to_solve = step_num if solved else 0
    recording.length = step_num

    return trajectory, total_reward, recording


def play_episode_baseline(client, policy="random"):
    """Play one maze episode with a simple policy. Returns recording."""
    from collections import deque

    try:
        result = client.reset()
    except Exception:
        client.connect()
        result = client.reset()

    recording = EpisodeRecording()
    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, agent_pos=tuple(obs.agent_pos),
        action="START", reward=0.0,
    ))

    total_reward = 0.0
    step_num = 0
    solved = False

    # For wall-follower: track visited for BFS fallback
    visited_for_follower = set()

    while not result.done and step_num < 100:
        obs = result.observation
        agent_r, agent_c = obs.agent_pos
        exit_r, exit_c = obs.exit_pos
        grid = obs.grid

        if policy == "random":
            direction = random.choice(DIRECTIONS)
        elif policy == "wall_follower":
            # BFS-based pathfinding toward exit (acts as smart baseline)
            best_dir = None
            best_dist = float("inf")
            for d in DIRECTIONS:
                dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[d]
                nr, nc = agent_r + dr, agent_c + dc
                if (
                    0 <= nr < len(grid)
                    and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != "#"
                ):
                    dist = abs(nr - exit_r) + abs(nc - exit_c)
                    # Prefer unvisited cells
                    if (nr, nc) in visited_for_follower:
                        dist += 5
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = d
            direction = best_dir or random.choice(DIRECTIONS)
            visited_for_follower.add((agent_r, agent_c))
        else:
            direction = random.choice(DIRECTIONS)

        result = safe_step(client, MazeAction(direction=direction), "")
        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid,
            agent_pos=tuple(obs.agent_pos),
            action=direction, reward=reward,
        ))

        if reward >= 10.0:
            solved = True

    recording.total_reward = total_reward
    recording.solved = solved
    recording.steps_to_solve = step_num if solved else 0
    recording.length = step_num

    return recording


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


@env.task(cache="auto")
async def eval_baselines(
    env_url: str = DEFAULT_ENV_URL, num_episodes: int = 50
) -> str:
    """Play maze with random and wall-follower policies to set baselines."""

    results = {}
    best_recordings = {}

    for policy in ["random", "wall_follower"]:
        solve_count = 0
        steps_to_solve = []
        rewards = []
        best_rec = None
        best_reward = float("-inf")
        client = create_maze_client(env_url)
        try:
            for _ in range(num_episodes):
                rec = play_episode_baseline(client, policy=policy)
                rewards.append(rec.total_reward)
                if rec.solved:
                    solve_count += 1
                    steps_to_solve.append(rec.steps_to_solve)
                if rec.total_reward > best_reward:
                    best_reward = rec.total_reward
                    best_rec = rec
        finally:
            client.close()

        solve_rate = solve_count / num_episodes
        avg_steps = sum(steps_to_solve) / len(steps_to_solve) if steps_to_solve else 0
        avg_reward = sum(rewards) / len(rewards)
        results[policy] = {
            "solve_rate": solve_rate,
            "avg_steps_to_solve": avg_steps,
            "avg_reward": avg_reward,
        }
        if best_rec:
            best_recordings[policy] = {
                "frames": [{"step": f.step, "grid": f.grid,
                            "agent_pos": list(f.agent_pos),
                            "action": f.action, "reward": f.reward}
                           for f in best_rec.frames],
                "total_reward": best_rec.total_reward,
                "solved": best_rec.solved,
                "steps_to_solve": best_rec.steps_to_solve,
                "length": best_rec.length,
            }
        print(f"  {policy}: solve_rate={solve_rate:.2f}, avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.2f}")

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
    """One GRPO iteration for maze navigation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    client = create_maze_client(env_url)

    all_rewards = []
    all_solved = []
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
                all_solved.append(rec.solved)

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

    solve_rate = sum(all_solved) / len(all_solved) if all_solved else 0
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
        "solve_rate": solve_rate,
        "loss": total_loss / num_groups,
    })
    print(f"  Step {step_idx} | solve_rate={solve_rate:.2f} avg_reward={avg_reward:.2f}")

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
    """Evaluate the model on maze navigation, returning metrics and best replay."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    client = create_maze_client(env_url)
    direction_counts = {d: 0 for d in DIRECTIONS}
    solve_count = 0
    steps_to_solve = []
    rewards = []
    best_rec = None
    best_reward = float("-inf")

    try:
        for _ in range(num_episodes):
            traj, reward, rec = play_episode_record(
                client, model, tokenizer, device, temperature=0.0
            )
            rewards.append(reward)
            if rec.solved:
                solve_count += 1
                steps_to_solve.append(rec.steps_to_solve)
            if reward > best_reward:
                best_reward = reward
                best_rec = rec
            for s in traj:
                d = parse_direction(s["action"])
                direction_counts[d] += 1
    finally:
        client.close()

    solve_rate = solve_count / num_episodes
    avg_steps = sum(steps_to_solve) / len(steps_to_solve) if steps_to_solve else 0
    avg_reward = sum(rewards) / len(rewards)

    best_replay = None
    if best_rec:
        best_replay = {
            "frames": [{"step": f.step, "grid": f.grid,
                        "agent_pos": list(f.agent_pos),
                        "action": f.action, "reward": f.reward}
                       for f in best_rec.frames],
            "total_reward": best_rec.total_reward,
            "solved": best_rec.solved,
            "steps_to_solve": best_rec.steps_to_solve,
            "length": best_rec.length,
        }

    del model
    cleanup_memory()

    return json.dumps({
        "step": step_idx,
        "solve_rate": solve_rate,
        "avg_steps_to_solve": avg_steps,
        "avg_reward": avg_reward,
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
    """Full Maze RL pipeline: baselines -> GRPO training -> eval -> visual report."""

    device = get_device()
    print(f"Device: {device}")
    print(f"Model:  {model_id}")
    print(f"Env:    {env_url}\n")

    # Start env server if using localhost
    server_proc = None
    if "localhost" in env_url or "127.0.0.1" in env_url:
        print("Starting local Maze env server...")
        server_proc = start_env_server()
        print("Server started.\n")

    try:
        # 1. Baselines
        print("=== Evaluating baselines ===")
        baselines_json = json.loads(await eval_baselines(env_url, num_episodes=50))
        baselines = baselines_json["results"]
        baseline_recordings = baselines_json.get("recordings", {})
        print(f"  Random:        solve_rate={baselines['random']['solve_rate']:.2f}")
        print(f"  Wall-follower: solve_rate={baselines['wall_follower']['solve_rate']:.2f}")

        # 2. Evaluate untrained model
        print("\n=== Evaluating untrained model ===")
        eval_results = [json.loads(await eval_model(model_id, env_url, eval_episodes, 0, use_bfloat16=use_bfloat16))]
        print(f"  Untrained: solve_rate={eval_results[0]['solve_rate']:.2f}")

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

            eval_json = await eval_model(
                model_id, env_url, eval_episodes, step,
                checkpoint_file=checkpoint_file,
                use_bfloat16=use_bfloat16,
            )
            eval_results.append(json.loads(eval_json))
            prev_checkpoint = checkpoint_file
            print(f"  Eval solve_rate: {eval_results[-1]['solve_rate']:.2f}")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    # 4. Generate report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_list = [e["step"] for e in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Solve rate over training
    ax = axes[0]
    solve_rates = [e["solve_rate"] for e in eval_results]
    ax.plot(steps_list, solve_rates, "b-o", markersize=6, linewidth=2, label="GRPO Agent")
    ax.axhline(
        baselines["random"]["solve_rate"],
        color="r", linestyle="--",
        label=f"Random ({baselines['random']['solve_rate']:.2f})",
    )
    ax.axhline(
        baselines["wall_follower"]["solve_rate"],
        color="g", linestyle="--",
        label=f"Wall-follower ({baselines['wall_follower']['solve_rate']:.2f})",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Chart 2: Avg steps to solve
    ax = axes[1]
    avg_steps_list = [e["avg_steps_to_solve"] for e in eval_results]
    ax.plot(steps_list, avg_steps_list, "m-o", markersize=6, linewidth=2)
    if baselines["wall_follower"]["avg_steps_to_solve"] > 0:
        ax.axhline(
            baselines["wall_follower"]["avg_steps_to_solve"],
            color="g", linestyle="--",
            label=f"Wall-follower ({baselines['wall_follower']['avg_steps_to_solve']:.0f})",
        )
        ax.legend()
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Steps to Solve")
    ax.set_title("Efficiency (lower = better)")
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
    for policy_name, rec_data in baseline_recordings.items():
        rec = EpisodeRecording(
            frames=[EpisodeFrame(
                step=f["step"], grid=f["grid"],
                agent_pos=tuple(f["agent_pos"]),
                action=f["action"], reward=f["reward"],
            ) for f in rec_data["frames"]],
            total_reward=rec_data["total_reward"],
            solved=rec_data["solved"],
            steps_to_solve=rec_data["steps_to_solve"],
            length=rec_data["length"],
        )
        replay_recs.append(rec)

    for e in eval_results:
        replay_data = e.get("best_replay")
        if replay_data:
            rec = EpisodeRecording(
                frames=[EpisodeFrame(
                    step=f["step"], grid=f["grid"],
                    agent_pos=tuple(f["agent_pos"]),
                    action=f["action"], reward=f["reward"],
                ) for f in replay_data["frames"]],
                total_reward=replay_data["total_reward"],
                solved=replay_data["solved"],
                steps_to_solve=replay_data["steps_to_solve"],
                length=replay_data["length"],
            )
            replay_recs.append(rec)

    replay_html = generate_replay_html(replay_recs, title="Maze Navigation Replays")

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Maze RL Training Report</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Solve Rate</th><th>Avg Steps</th><th>Avg Reward</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['solve_rate']:.2f}</td>"
        f"<td>{baselines['random']['avg_steps_to_solve']:.0f}</td>"
        f"<td>{baselines['random']['avg_reward']:.2f}</td></tr>"
        f"<tr><td>Wall-follower</td><td>{baselines['wall_follower']['solve_rate']:.2f}</td>"
        f"<td>{baselines['wall_follower']['avg_steps_to_solve']:.0f}</td>"
        f"<td>{baselines['wall_follower']['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO Agent (step 0)</b></td><td>{eval_results[0]['solve_rate']:.2f}</td>"
        f"<td>{eval_results[0]['avg_steps_to_solve']:.0f}</td>"
        f"<td>{eval_results[0]['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO Agent (final)</b></td><td><b>{final['solve_rate']:.2f}</b></td>"
        f"<td><b>{final['avg_steps_to_solve']:.0f}</b></td>"
        f"<td><b>{final['avg_reward']:.2f}</b></td></tr>"
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
        f"Final solve_rate: {final['solve_rate']:.2f} "
        f"(random: {baselines['random']['solve_rate']:.2f}, "
        f"wall-follower: {baselines['wall_follower']['solve_rate']:.2f})"
    )
    return summary, checkpoint_file


# Server:  cd tutorials/openenv-maze && python -m maze_env.server.app
# Local:   flyte run --local maze_rl.py pipeline --training_steps 3
# Remote:  flyte run maze_rl.py pipeline --training_steps 10