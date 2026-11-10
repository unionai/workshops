"""Load the model the GPU pool's prefetch stored, and answer a few questions with it. CPU only, no node grant.

    QUEUE=<your GPU queue> FLYTE_CONFIG=<your cluster config> python checks/predict_check.py <prefetched dir uri>
"""

import html
import os
import subprocess
import sys
import time

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(name="nc_predict_check", resources=flyte.Resources(cpu="4", memory="16Gi", disk="30Gi"))

QUESTIONS = [
    "What is the capital of France? Answer in one word.",
    "What is 17 + 25? Answer with just the number.",
    "Name one symptom of the flu, in a few words.",
]


def _install(packages: list[str], extra: list[str] = ()) -> None:
    """The public Flyte image's Python is a uv venv with no pip; install into /tmp and put it on the path."""
    import shutil

    target = "/tmp/predict-packages"
    proc = subprocess.run([shutil.which("uv"), "pip", "install", "--python", sys.executable, "--target", target, *extra, *packages],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"install failed: {proc.stderr[-1500:]}")
    sys.path.insert(0, target)


@env.task(report=True)
async def predict(uri: str) -> list[dict[str, str]]:
    t0 = time.time()
    _install(["torch"], ["--index-url", "https://download.pytorch.org/whl/cpu"])
    _install(["transformers>=4.51", "accelerate"])
    t_install = time.time()
    local = await flyte.io.Dir.from_existing_remote(uri).download()
    t_download = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForCausalLM.from_pretrained(local, torch_dtype="auto")
    t_load = time.time()

    answers = []
    for q in QUESTIONS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
        a = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Q: {q}\nA: {a}\n", flush=True)
        answers.append({"question": q, "answer": a})

    timing = (f"install {t_install - t0:.0f} s · download {t_download - t_install:.1f} s · load {t_load - t_download:.1f} s · "
              f"answer {time.time() - t_load:.1f} s")
    print(timing, flush=True)
    rows = "".join(f"<tr><td>{html.escape(x['question'])}</td><td><b>{html.escape(x['answer'])}</b></td></tr>" for x in answers)
    await flyte.report.replace.aio(
        f"<h2>Qwen3-0.6B from the GPU pool's store</h2><p><code>{html.escape(uri)}</code></p><p>{timing}</p>"
        f"<table><tr><th>question</th><th>answer</th></tr>{rows}</table>", do_flush=True)
    return answers


if __name__ == "__main__":
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))  # defaults to the usual config lookup
    run = flyte.with_runcontext(queue=os.environ["QUEUE"]).run(predict, uri=sys.argv[1])
    print(run.name, run.url)
