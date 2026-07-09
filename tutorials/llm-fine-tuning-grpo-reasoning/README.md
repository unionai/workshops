# GRPO Fine-Tuning: Teach a Model to *Reason*

Take a small model with **no reasoning ability** and use **GRPO** to make it reason — without ever showing it how. This is the **DeepSeek-R1-Zero** result (reproduced by [TinyZero](https://github.com/Jiayi-Pan/TinyZero)) on a tiny model: pure reinforcement learning, reward only the final answer, and the model *discovers on its own* that thinking step-by-step before answering earns more reward.

> **Why this is the headline GRPO demo:** Qwen2.5 has no "thinking" mode (that came with Qwen3/QwQ). So the base model just blurts a guess. After GRPO, it writes out its reasoning first — and its responses visibly get **longer and more structured** as training proceeds. That "response length grows as it learns to reason" curve is the famous R1 chart, and here you watch it happen live.

---

## Do We Give It a Dataset to Learn From? (Read This First)

This is the question that trips everyone up, and the answer is the whole point of GRPO:

**No — we never give the model a single example of reasoning, or even a single correct answer to copy.**

It helps to compare the two ways you could teach this skill:

| | **Supervised fine-tuning (SFT)** | **GRPO (what we do here)** |
|---|---|---|
| What you must provide | Thousands of `(problem → worked-out reasoning → answer)` **demonstrations** | Just a bag of **problems** + a **reward function** |
| Where the signal comes from | Copying the human-written reasoning | The model's *own* attempts, scored by the reward |
| Who writes the reasoning | You do (expensive, and caps the model at your examples) | The model invents it to earn reward |
| Needs labeled reasoning data? | **Yes** | **No** |

So the "dataset" in this tutorial is **not** a dataset of solutions. It's just a list of **problems** (prompts):

```
Numbers: [3, 7, 9]. Target: 20.
Numbers: [2, 5, 8]. Target: 18.
Numbers: [4, 4, 6]. Target: 10.
...
```

That's it. No solution, no reasoning, no answer key travels with the problem into training. During training, the model generates several attempts at each problem, and a **reward function** checks each one (*"does this expression actually equal the target using the given numbers?"*). Correct attempts get reinforced. The reasoning that leads to correct answers is something the model **grows on its own**, because reasoning happens to raise its odds of being right.

**The reward function is the teacher — not a labeled dataset.** This is the core idea of **RLVR** (Reinforcement Learning with Verifiable Rewards): if you can *check* an answer with a program, you don't need to *demonstrate* the answer. That's why tasks like math, code, and Countdown are perfect for it — correctness is cheap to verify but expensive to demonstrate.

### So what data do you actually need?

Two things, and only two:

1. **Problems (prompts).** Here we *generate* them — see [Where the Problems Come From](#where-the-problems-come-from). They can be synthetic, scraped, or hand-written; they just need to span the skill you want.
2. **A reward function.** For Countdown it's ~20 lines: parse the answer, check it uses the given numbers once and evaluates to the target. No model, no labels.

The one thing your problems *do* need: they have to sit in the model's **learnable zone** — solvable often enough that some attempts succeed and give the reward something to reinforce. If every attempt fails, there's no signal (see the [code tutorial](../llm-fine-tuning-grpo-code) for what happens when problems are too hard). Countdown is generated to always be solvable, so this is handled for you.

### What about evaluation — surely *that* needs a dataset?

Yes, and this is the subtle part. Evaluation needs a **held-out set of problems** — a "dataset" in the sense of a collection of test cases (this pipeline generates one and de-duplicates it from the training problems, so held-out problems are genuinely unseen). But it still needs **no labeled solutions**: the *same* verifier that computes the training reward also grades the eval answers.

The only ground truth evaluation uses is the **target** — and for Countdown the target is *part of the problem itself* (`Numbers: [3,7,9]. Target: 20`). The grader evaluates the model's expression and asks "does it equal 20 using 3, 7, 9?" No answer key, no human labels.

The general rule across all of these tutorials:

> RLVR needs a **verifiable target** per problem — for both the reward *and* the eval. Sometimes it's baked into the problem (Countdown's target); sometimes it ships with the dataset (the [math tutorial](../llm-fine-tuning-grpo-math) uses GSM8K's gold answers, which grade both reward and eval). Either way you need something to *check against* — but never the *reasoning* that reaches it.

---

## The Task: Countdown

Given a few numbers and a target, write an arithmetic expression using each number exactly once (with `+ - * /`) that equals the target:

```
Numbers: [3, 7, 9]. Target: 20.
  → <think>3 * 9 = 27, that's too big. 27 - 7 = 20. Yes.</think><answer>3 * 9 - 7</answer>   ✓
```

It's the ideal reasoning task for a small model:
- **Verifiable** — just evaluate the expression (a safe AST evaluator, **no sandbox** needed).
- **Homogeneous** — one skill, endless instances → the skill *transfers* to held-out problems.
- **Reasoning helps** — hard enough that searching/thinking beats guessing, so GRPO rewards reasoning.
- **Always solvable** — each problem is generated by folding its numbers into the target, so a solution is guaranteed to exist.

## How GRPO Learns to Reason

For each problem, the model generates several completions. The reward scores each, and GRPO reinforces the ones that did better *within that group*:

```
Problem: "Numbers: [3, 7, 9]. Target: 20."

  ├── Attempt 1: "<think>3*9=27, 27-7=20</think><answer>3 * 9 - 7</answer>"  → correct → reward 1.0
  ├── Attempt 2: "<answer>3 + 7 + 9</answer>"                                 → 19 ✗    → reward 0.0
  ├── Attempt 3: "<think>7*3=21, 21-9=12</think><answer>7 * 3 - 9</answer>"   → 12 ✗    → reward 0.0
  └── Attempt 4: "<think>9-7=2, 2*3... no. 9*3=27, -7=20</think><answer>9 * 3 - 7</answer>" → correct → reward 1.0

Attempts 1 & 4 get positive advantage, 2 & 3 negative. Notice the winners *reasoned first*.
Over many problems, the policy shifts toward "think, then answer" — because that pattern
correlates with reward. Nobody told it to reason. It discovered reasoning pays.
```

The visible signature of this is the **response length climbing** during training: the model learns that spending tokens on a `<think>` block before committing to an answer raises its hit rate, so completions get longer and more deliberate on their own.

## The Reward

Two reward functions, combined with weights `[1.0, 0.2]`:

| Reward | Weight | Measures |
|--------|--------|----------|
| **Answer correctness** | 1.0 | the expression uses each given number once **and** equals the target |
| **Format** | 0.2 | output has `<think>…</think><answer>…</answer>` |

Correctness is the real objective; the small format reward just keeps the reasoning parseable and nudges the model toward the think-then-answer structure early. Both are computed by *checking*, never by comparing to a stored answer.

## What's in the Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│ Prepare Data │────▶│   GRPO Training   │────▶│  Evaluate  │
│  (CPU task)  │     │   (GPU task)      │     │ (GPU task) │
└──────────────┘     └──────────────────┘     └────────────┘
 Generate solvable    Reward = correct        Base vs GRPO:
 Countdown problems   answer + <think> format solve rate, response
 (no solutions!)      (LoRA or full)          length, reasoning traces
```

1. **Prepare data** — Generate solvable Countdown problems (numbers + target only, no solutions).
2. **Train with GRPO** — Generate attempts, reward correct ones, reinforce. Reasoning emerges.
3. **Evaluate** — Run held-out problems through base vs trained; compare solve rate, response length, and read the actual reasoning traces.

## Where the Problems Come From

We generate them so we never need a label and can guarantee solvability. Each problem is built by picking a few random numbers and *folding them into a target* with random `+ - * ` operations:

```python
nums = [3, 7, 9]
# fold: 3, then (3 op 7), then (... op 9), e.g. 3 * 7 = 21, 21 - 9 = 12  → target 12
```

Because the target is produced *from* the numbers, a valid expression always exists (at minimum, the one we used to generate it) — though the model is free to find a different one. Train and eval problems are de-duplicated so held-out problems are genuinely unseen. No human ever labels or solves anything.

## Run

```bash
cd tutorials/llm-fine-tuning-grpo-reasoning
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# 1.5B — reasoning emerges most clearly here
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-1.5B-Instruct"

# 0.5B — faster; shows the direction (longer, more structured answers)
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --num_generations 8 --batch_size 8
```

Uses **LoRA** by default (`--method full` for full fine-tuning). Reasoning emerges clearly at ~1.5B; a 0.5B shows the direction but shorter chains — run both and compare.

## What to Watch

- **Training report** — the **Response Length** chart climbing alongside the solve rate is the model teaching itself to reason. This is the moment to point at during a workshop.
- **Eval report** — base vs GRPO solve rate, mean response length (base blurts, GRPO reasons), and side-by-side **reasoning traces** you can read.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | instruct model to fine-tune |
| `--method` | `lora` | `lora` or `full` |
| `--epochs` | `3` | training epochs |
| `--lr` | `1e-5` | learning rate |
| `--batch_size` | `8` | completions per step (÷ by `--num_generations`) |
| `--num_generations` | `8` | completions per prompt (the "group") |
| `--max_completion_length` | `320` | room for reasoning + answer |
| `--beta` | `0.005` | KL penalty vs. the base model |
| `--n_numbers` | `3` | numbers per puzzle (4 = harder) |
| `--max_num` | `9` | largest number used |
| `--max_train_samples` | `300` | training problems |
| `--num_eval_examples` | `60` | held-out problems compared before/after |

## Why This Works Where Code Didn't

Its sibling [code tutorial](../llm-fine-tuning-grpo-code) showed that a 0.5B can't learn *general* coding skill from 100 diverse problems — each problem is a novel generation, so nothing transfers. Countdown is the opposite: **one skill applied to endless instances**, so what the model learns generalizes — the same reason the [letter-counting](../llm-fine-tuning-grpo) and [math](../llm-fine-tuning-grpo-math) demos work. The lesson: **match the task shape to what RL can actually do — and remember the reward, not a labeled dataset, is doing the teaching.**

## Further Reading

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RL induces reasoning from a base model with no reasoning demonstrations (R1-Zero).
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — introduces GRPO.
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — reproduces the R1-Zero "aha moment" on a small model using exactly this Countdown task.
