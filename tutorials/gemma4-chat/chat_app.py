"""Gradio chat UI for Gemma 4, fronting the vLLM server.

This is a Flyte 2 pickled-mode app: `@env.server` is shipped to the Flyte
container along with the Gradio UI definition. The vLLM server's URL is
injected at runtime via the `vllm_url` parameter (resolved via AppEndpoint
+ depends_on).

Deploy (after `python vllm_server.py` has finished):
    python chat_app.py
"""

from __future__ import annotations

import flyte
import flyte.app

from config import CHAT_APP_NAME, MODEL


# Frontend image. We don't need vllm here — just an OpenAI client + Gradio.
chat_image = (
    flyte.Image.from_debian_base(
        name="gemma4-chat-image",
        # Same as vllm_server.py — push to devbox-local registry and build only
        # for the host architecture (aarch64) so QEMU-emulated amd64 doesn't
        # segfault during `uv venv`.
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    # Gradio 5.x: needed for `gr.Chatbot(type="messages")` (the metadata-titled
    # 🧠 Thinking panel relies on the messages format). Gradio 6.x dropped that
    # kwarg and 4.x doesn't have messages format at all. Pinning to a known-
    # good 5.x release; the explicit `while True: sleep` fallback in
    # `_run` keeps the pod alive even if launch()'s blocking behavior changes.
    .with_pip_packages("gradio==5.42.0", "openai>=1.50.0")
)

env = flyte.app.AppEnvironment(
    name=CHAT_APP_NAME,
    image=chat_image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    port=7860,
    requires_auth=False,
    parameters=[
        # Pass the vLLM URL directly as a string — bypass `AppEndpoint`.
        #
        # Background: `AppEndpoint(app_name=...)` has two modes:
        #   - public=False: needs INTERNAL_APP_ENDPOINT_PATTERN env var, set
        #     by `depends_on` deploy-time wiring. But depends_on conflicts
        #     with pkl-bundle interactive deploy because VLLMAppEnvironment
        #     has no @server function.
        #   - public=True: resolves the URL via Flyte's App API, but it
        #     returns the *external* `<svc>.localhost` URL which only
        #     resolves on the host — not from inside cluster pods.
        #
        # The cluster-internal Knative DNS form is reliable from a sibling
        # pod and matches Knative's `address.url` for the service. Hardcoding
        # the project/domain is acceptable for this tutorial.
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{MODEL.app_name}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=MODEL.model_id),
    ],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,   # 30 min — match vLLM so the UI doesn't disappear before the model
    ),
)


def _split_thinking(text: str) -> tuple[str, str]:
    """Split a Gemma-4 chat response into (thinking, answer).

    The IT model emits its chain-of-thought wrapped between two special
    tokens, which vLLM renders as text in the streamed completion:
      <|channel>thought
      ...reasoning...
      <channel|>
      ...final answer...

    These are present whenever the model is capable of thinking, even when
    thinking is disabled (in which case the thought block is empty). We
    treat content between the markers as thinking, and content after the
    closing marker as the answer.

    Robust to partial markers so it can be called incrementally on a
    growing streaming buffer.
    """
    OPEN, OPEN_TAIL = "<|channel>", "thought\n"
    CLOSE = "<channel|>"
    j = text.find(OPEN)
    if j == -1:
        return "", text.strip()
    pre = text[:j]
    rest = text[j + len(OPEN):]
    if rest.startswith(OPEN_TAIL):
        rest = rest[len(OPEN_TAIL):]
    k = rest.find(CLOSE)
    if k == -1:
        thinking, answer = rest, pre
    else:
        thinking = rest[:k]
        answer = (pre + rest[k + len(CLOSE):])
    return thinking.strip(), answer.strip()


@env.server
def chat_server(vllm_url: str, model_id: str):
    """Run the Gradio chat UI. Blocking."""
    import sys
    import traceback
    try:
        _run(vllm_url, model_id)
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(vllm_url: str, model_id: str):
    import gradio as gr
    from openai import OpenAI

    print("[chat_server] gradio version:", gr.__version__, flush=True)
    base_url = vllm_url.rstrip("/") + "/v1"
    print(f"[chat_server] Connecting to vLLM at {base_url} (model={model_id})", flush=True)
    client = OpenAI(base_url=base_url, api_key="not-used")

    DEFAULT_SYSTEM = "You are a helpful assistant."

    # Rough chars-per-token heuristic for converting the user-facing thinking-
    # budget slider (in tokens) to a character cap on the streamed buffer.
    CHARS_PER_TOKEN = 3.5

    # Internal hard cap on total tokens (thinking + answer). The user-facing
    # control is the "Thinking budget" slider; this is just a safety ceiling
    # so the model can't ramble past it. Well below the model's max_model_len.
    MAX_TOTAL_TOKENS = 4096

    def chat(message, history, system_prompt, enable_thinking, think_budget,
             temperature, top_p):
        if not message or not message.strip():
            yield "", history
            return

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history

        # Build the messages list for /v1/chat/completions. The -it model's
        # chat_template.jinja handles `<|turn>...<turn|>` formatting and the
        # `<|think|>` insertion when we set `chat_template_kwargs.enable_thinking`.
        sys_text = system_prompt.strip() or "You are a helpful assistant."
        msgs = [{"role": "system", "content": sys_text}]
        for t in history[:-2]:
            if "metadata" in t:
                continue   # skip the prior thinking-panel placeholders
            msgs.append({"role": t["role"], "content": t["content"]})

        budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

        stream = client.chat.completions.create(
            model=model_id,
            messages=msgs,
            stream=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=MAX_TOTAL_TOKENS,
            extra_body={
                # Forwarded to the chat template as `enable_thinking=...`.
                # vLLM passes this through to the jinja template kwargs.
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                # Keep special tokens like <|channel> / <channel|> in the
                # streamed output so _split_thinking can find the boundary
                # between the thought block and the answer. Default is True,
                # which strips them and leaves us with just `thought\n...`.
                "skip_special_tokens": False,
            },
        )

        buf = ""
        capped = False
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                thinking, answer = _split_thinking(buf)
                history[-2]["content"] = thinking
                history[-1]["content"] = answer
                yield "", history

                # Cap thinking length: if we've exceeded the budget AND the
                # model hasn't started the answer yet (no `<channel|>` seen
                # → answer is still empty), abort and do a second pass.
                if (budget_chars and not answer and len(thinking) >= budget_chars):
                    capped = True
                    break
        finally:
            stream.close()

        if capped:
            history[-2]["content"] += f"\n\n_[capped at ~{think_budget} tokens]_"
            yield "", history

            # Second pass: thinking disabled, force a direct answer using the
            # truncated thought as priming context. The original Ollama
            # version did exactly this; OpenAI-compat API maps cleanly.
            followup = msgs + [
                {"role": "assistant", "content": history[-2]["content"]},
                {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
            ]
            answer_stream = client.chat.completions.create(
                model=model_id,
                messages=followup,
                stream=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=MAX_TOTAL_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "skip_special_tokens": False,
                },
            )
            buf2 = ""
            try:
                for chunk in answer_stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    buf2 += delta
                    # With thinking disabled the model still emits an empty
                    # `<|channel>thought\n<channel|>` envelope before the
                    # answer — strip it the same way.
                    _, ans = _split_thinking(buf2)
                    history[-1]["content"] = ans
                    yield "", history
            finally:
                answer_stream.close()

        # If the model never wrote a `<think>...</think>` block, drop the
        # empty thinking placeholder so the UI doesn't show a blank panel.
        if not history[-2]["content"]:
            history.pop(-2)
            yield "", history

    with gr.Blocks(title=f"Gemma 4 Chat ({model_id})") as demo:
        gr.Markdown(
            f"# Gemma 4 Chat\n"
            f"Served by vLLM on Flyte. Model: `{model_id}` · Endpoint: `{base_url}`"
        )
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Caps the chain-of-thought. When hit, we stop reasoning and re-prompt for a direct answer.",
            )
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            enable_thinking = gr.Checkbox(
                value=True, label="Enable thinking",
                info="Adds <|think|> to the system prompt — model reasons step-by-step before answering.",
                scale=1,
            )
        chatbot = gr.Chatbot(type="messages", label="Conversation", height=500)
        msg = gr.Textbox(label="Your message", placeholder="Type and press Enter")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        inputs = [msg, chatbot, system_prompt, enable_thinking, think_budget, temperature, top_p]
        outputs = [msg, chatbot]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: [], outputs=chatbot)

    print("[chat_server] About to demo.launch()", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    print("[chat_server] demo.launch() returned — sleeping forever to keep pod alive", flush=True)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Chat UI deployed: {app.url}")
