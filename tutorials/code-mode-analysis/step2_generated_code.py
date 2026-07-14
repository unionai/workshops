"""Step 2 — the model writes the code, the sandbox runs it.

Step 1's orchestrator was a decorated function: we wrote it, Flyte registered it.
`orchestrate_local` is the same sandbox, but it takes the program as a *string* —
which means the program doesn't have to exist until runtime, and doesn't have to
be written by a human.

That is the entire trick behind code mode. The rest of this file is a
generate → execute → retry loop in about forty lines:

  1. ask the model for a Python program, given the tool signatures
  2. run it in Monty with those tools bound
  3. if it raises, hand the error back to the model and let it try again

Step 3 replaces all of this with `Agent(code_mode=True)`, which does the same
thing with more polish. It's worth seeing the moving parts once.

Run it:

    uv run flyte run step2_generated_code.py analyze \\
        --question "Did tipping change between January and December 2024?"
"""

from __future__ import annotations

import inspect

import flyte.sandbox
from flyte.sandbox import ORCHESTRATOR_SYNTAX_PROMPT

import tools
from config import env
from dataset import MONTHS
from llm import MAX_TOKENS, MODEL, SANDBOX_RULES


def _tool_catalog() -> str:
    """Describe the tools from their signatures and docstrings.

    The only tool documentation there is: add a function and the prompt updates
    itself.
    """
    entries = []
    for fn in tools.ALL_TOOLS:
        signature = str(inspect.signature(fn)).replace("'", "")
        doc = inspect.getdoc(fn) or ""
        entries.append(f"def {fn.__name__}{signature}:\n    \"\"\"{doc}\"\"\"")
    return "\n\n".join(entries)


def _system_prompt() -> str:
    return f"""You are a data analyst who answers questions by writing a single Python program.

{tools.DATA_DESCRIPTION}

Available functions (already in scope — do NOT import them, do NOT define them):

{_tool_catalog()}

{ORCHESTRATOR_SYNTAX_PROMPT}

{SANDBOX_RULES}

The variable `months` is in scope: a list of every available month string.

Write ONE Python program that answers the question. Call `query` to get numbers,
use plain Python to compare and rank them, and call the create_* functions to put
your findings in the report. End with a short plain-string summary as the last
expression — that string is the return value.

Respond with a single ```python code block and nothing else."""


async def generate_code(question: str, history: list[dict]) -> str:
    """Ask the model for a program. Returns the code inside the fenced block."""
    import anthropic

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(),
        messages=[*history, {"role": "user", "content": question}],
        thinking={"type": "adaptive"},
    )
    text = "\n".join(b.text for b in response.content if b.type == "text")

    # Pull the code out of the ```python fence.
    if "```" in text:
        block = text.split("```", 2)[1]
        if block.startswith("python"):
            block = block[len("python"):]
        return block.strip()
    return text.strip()


@env.task(report=True)
async def analyze(question: str, max_retries: int = 2) -> str:
    """Generate a program, run it in the sandbox, retry on failure, render a report."""
    import flyte.report

    tools.new_report()
    history: list[dict] = []
    code = await generate_code(question, history)

    for attempt in range(1 + max_retries):
        try:
            summary = await flyte.sandbox.orchestrate_local(
                code,
                inputs={"months": MONTHS},
                tasks=tools.ALL_TOOLS,
            )
            break
        except Exception as exc:
            if attempt == max_retries:
                raise
            # Hand the error back and let the model fix its own code.
            print(f"attempt {attempt + 1} failed: {exc}")
            history = [
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {"role": "user", "content": f"That failed with:\n\n{exc}\n\nFix the program."},
            ]
            code = await generate_code(question, history)

    from report import render

    await flyte.report.replace.aio(
        render(question, code, tools.collect_report(), str(summary))
    )
    await flyte.report.flush.aio()
    return str(summary)


if __name__ == "__main__":
    import flyte

    flyte.init_from_config(image_builder="remote")
    run = flyte.run(analyze, question="Did tipping change between January and December 2024?")
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs())
