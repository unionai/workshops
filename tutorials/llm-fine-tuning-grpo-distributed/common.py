"""Shared pieces: prompt construction, code assembly, sandboxed test execution,
and the dataset / base-model tasks.

Carried over from ../llm-fine-tuning-grpo-code so this tutorial stands alone
(tutorials in this repo are self-contained). The one substantive change is that
`run_tests_sandboxed` no longer assumes a bubblewrap session — it takes whatever
session it is handed, so the same function serves the bwrap path (single-GPU,
learner-local) and the userns path (reusable verifier pool).
"""

import logging
import os
import re
import tempfile

import flyte
import flyte.io

from config import HF_TOKEN, cpu_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


MBPP_DATASET = "google-research-datasets/mbpp"

# For instruct models, a bare "def foo():" completion prompt makes them ramble
# (explanations, example usage, prose) which breaks the sandbox. A chat prompt
# with an explicit "code only" instruction gives a fair baseline — the base
# model produces clean code, so the GRPO delta reflects skill, not just format.
CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a problem description and a "
    "function signature, respond with ONLY the complete Python function that "
    "solves it — the signature and its body. Do not include any explanation, "
    "prose, example usage, test code, or text outside the function."
)


def build_code_prompt(tokenizer, raw_prompt: str, use_chat_template: bool = True) -> str:
    """Wrap the raw (problem + signature) prompt in the chat template with a
    code-only instruction, when the tokenizer supports it. Falls back to the raw
    completion-style prompt (correct for base models with no chat template)."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return raw_prompt


def assemble_code(func_def: str, completion: str, setup: str = "") -> str:
    """Turn a model completion into a runnable script.

    Handles both styles: a completion that is just the function body (prepend the
    signature) and one that already includes `def name(...)` (use as-is). Strips
    markdown code fences that instruct models often add.
    """
    m = re.search(r"def\s+(\w+)", func_def)
    fname = m.group(1) if m else None
    completion = re.sub(r"^\s*```(?:python)?\n?", "", completion)
    completion = re.sub(r"\n?```\s*$", "", completion)
    if fname and re.search(rf"\bdef\s+{re.escape(fname)}\b", completion):
        code = completion
    else:
        code = func_def + "\n" + completion
    if setup:
        code = setup + "\n" + code
    return code


def extract_test_list(tests: str) -> list[str]:
    """Split the newline-joined MBPP test string into individual assert statements."""
    return [l.strip() for l in tests.strip().split("\n") if l.strip().startswith("assert")]


def func_def_from_prompt(prompt: str) -> str:
    """The last line of an MBPP prompt is the function signature."""
    return prompt.strip().split("\n")[-1]


async def run_tests_sandboxed(
    sbx, code: str, test_list: list[str], timeout_s: float = 5.0,
) -> tuple[bool, int, int]:
    """Run test cases against generated code inside a sandbox.

    Executes untrusted LLM-generated code in an isolated environment with no
    network access. Each test assertion is checked individually so we can report
    partial credit (passed/total) even though the reward itself is binary.

    Args:
        sbx: An open sandbox session (bubblewrap or userns — this function
            doesn't care which, which is what lets the same reward logic run
            both on the learner and on the reusable verifier pool).
        code: The complete generated code (full function definition).
        test_list: List of assert strings to run against the code.
    """
    total = len(test_list)
    if total == 0:
        return False, 0, 0

    # Build a script that runs each test and prints PASS/FAIL per line
    test_script = code + "\n\n"
    for i, test in enumerate(test_list):
        test_script += (
            f"try:\n"
            f"    {test}\n"
            f"    print('PASS:{i}')\n"
            f"except Exception:\n"
            f"    print('FAIL:{i}')\n"
        )

    proc = await sbx.run(
        test_script,
        script_type="python",
        stdout=True,
        stderr=True,
        network_mode="blocked",
        timeout_s=timeout_s,
    )
    out, err = await proc.communicate_text()

    if not out or proc.returncode != 0:
        log.debug(f"[Sandbox] returncode={proc.returncode} stderr={err[:200] if err else 'None'}")

    passed = out.count("PASS:") if out else 0
    return passed == total, passed, total


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    max_candidate_samples: int = 250,
    max_eval_samples: int = 50,
) -> flyte.io.Dir:
    """Load MBPP coding problems and prepare a candidate pool + eval split.

    MBPP columns: text (problem description), code (solution), test_list (assert strings).
    We build a prompt that includes the problem description and the function signature
    extracted from the reference solution, then let the model complete the body.
    """
    import random

    from datasets import Dataset, DatasetDict, load_dataset

    log.info("Loading MBPP dataset...")

    # MBPP has train (374), test (500), validation (90), prompt (10)
    mbpp = load_dataset(MBPP_DATASET, "full")

    all_rows = []
    for split in ["train", "validation", "test"]:
        for row in mbpp[split]:
            # Extract function name from first test assertion:
            # "assert min_cost(...)" -> "min_cost"
            first_test = row["test_list"][0]
            match = re.search(r"assert\s+(\w+)\s*\(", first_test)
            if not match:
                continue
            func_name = match.group(1)

            # Extract the function signature from the reference solution
            func_sig = None
            for line in row["code"].split("\n"):
                if line.strip().startswith(f"def {func_name}"):
                    func_sig = line.rstrip()
                    break
            if not func_sig:
                continue

            prompt_text = f"{row['text']}\n\n{func_sig}"

            all_rows.append({
                "prompt": prompt_text,
                "func_prompt": prompt_text,  # duplicate — "prompt" is reserved by GRPOTrainer
                "tests": "\n".join(row["test_list"]),
                "setup_code": row.get("test_setup_code", "").strip(),
                "name": func_name,
            })

    log.info(f"Loaded {len(all_rows)} valid MBPP problems")

    rng = random.Random(42)
    rng.shuffle(all_rows)

    n_train = min(max_candidate_samples, len(all_rows) - max_eval_samples)
    n_eval = min(max_eval_samples, len(all_rows) - n_train)

    processed = DatasetDict({
        "train": Dataset.from_list(all_rows[:n_train]),
        "eval": Dataset.from_list(all_rows[n_train:n_train + n_eval]),
    })

    output_dir = os.path.join(tempfile.mkdtemp(), "dataset")
    processed.save_to_disk(output_dir)
    log.info(f"Dataset ready: {n_train} candidates, {n_eval} eval")

    return await flyte.io.Dir.from_local(output_dir)


@cpu_env.task(cache="auto")
async def download_model(model_name: str) -> flyte.io.Dir:
    """Download the base model weights once and cache them, so the learner, the
    rollout workers, and the evaluator don't each re-fetch from HuggingFace.

    Marginal for a 0.5B; a real win (and a dodge of HF rate-limits) for a 14B —
    especially at Level 2, where several rollout replicas would otherwise pull the
    same ~28GB concurrently.
    """
    from huggingface_hub import snapshot_download

    local = os.path.join(tempfile.mkdtemp(), "model")
    snapshot_download(
        repo_id=model_name,
        local_dir=local,
        token=HF_TOKEN,
        ignore_patterns=["*.pth", "*.onnx", "*.gguf", "*.msgpack", "*.h5"],
    )
    log.info(f"Downloaded {model_name} -> {local}")
    return await flyte.io.Dir.from_local(local)
