"""Gemma 4 live-camera describer, served as a Flyte 2 app.

Webcam → Gemma 4 IT vision → streaming caption every few seconds. Reuses the
same `gemma4-26b-a4b-it-vllm` server as `chat_app.py`; this is a separate
Gradio frontend app pointed at the same OpenAI-compatible endpoint.

Two modes:
- independent: each caption is a fresh description of the current frame
- narrative:   the prompt includes recent captions so the model focuses on
               what changed — feels like live commentary

Webcam access (`getUserMedia`) requires a secure context — works on
`http://localhost:30081/...` directly on the Spark, but **not** from a
Tailscaled remote browser. Set `GRADIO_SHARE=1` to launch a public HTTPS
tunnel via Gradio servers in that case.

Deploy:
    python live_camera_app.py
"""

from __future__ import annotations

import os

import flyte
import flyte.app

from config import MODEL  # MODEL.app_name, MODEL.model_id


LIVE_CAM_APP_NAME = "gemma4-live-camera"

# Pull GRADIO_SHARE / CAMERA_CADENCE etc. from THIS shell's env at deploy time
# and propagate them into the pod via env_vars. Without this the @env.server
# function would only see the pod's environment (set by Knative), not yours.
_propagated_envs = {
    k: os.environ[k]
    for k in ("GRADIO_SHARE", "CAMERA_CADENCE", "MAX_SIDE", "MAX_OUT_TOKENS")
    if k in os.environ
}

chat_image = (
    flyte.Image.from_debian_base(
        name="gemma4-live-camera-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages("gradio==5.42.0", "openai>=1.50.0", "pillow>=10.0.0")
)

env = flyte.app.AppEnvironment(
    name=LIVE_CAM_APP_NAME,
    image=chat_image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    port=7867,
    requires_auth=False,
    env_vars=_propagated_envs,
    parameters=[
        # Same internal-URL trick as `chat_app.py`. Avoids depends_on +
        # AppEndpoint complexity for a tutorial deploy on the local devbox.
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{MODEL.app_name}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=MODEL.model_id),
    ],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=900,   # 15 min idle → kill pod (and its gradio.live tunnel)
    ),
)


@env.server
def live_camera_server(vllm_url: str, model_id: str):
    """Run the Gradio live-camera UI. Blocking."""
    import base64
    import io
    import os
    import threading
    import time

    import gradio as gr
    from openai import OpenAI
    from PIL import Image

    base_url = vllm_url.rstrip("/") + "/v1"
    print(f"[live_camera] gradio={gr.__version__}  vllm={base_url}  model={model_id}", flush=True)
    client = OpenAI(base_url=base_url, api_key="not-used")

    # ---------------- Knative idle-timer keep-alive ---------------------------
    # When users access this app via the gradio.live HTTPS tunnel, the request
    # path is:
    #   browser → gradio.live → frpc inside pod → gradio on :7867
    # That bypasses Knative's queue-proxy sidecar (which listens on :8012),
    # so Knative sees zero ingress activity and scales the pod (and the
    # tunnel) down at `scaledown_after`. Workaround: while there's *real*
    # caption activity, periodically poke our own queue-proxy from inside
    # the pod. That registers as legitimate traffic and resets the idle
    # timer. Stops pinging when the user stops captioning.
    import urllib.request

    last_activity_ts = [0.0]   # mutable container so closure can mutate
    KEEPALIVE_PERIOD_S = 60
    ACTIVITY_WINDOW_S = 300

    def _keepalive_loop():
        while True:
            time.sleep(KEEPALIVE_PERIOD_S)
            if time.time() - last_activity_ts[0] > ACTIVITY_WINDOW_S:
                continue   # no recent activity → let Knative scale down
            try:
                urllib.request.urlopen("http://localhost:8012/", timeout=3)
            except Exception:
                pass   # best-effort — don't crash the server thread

    threading.Thread(target=_keepalive_loop, daemon=True).start()

    # --- tunables ---
    CADENCE_S = float(os.environ.get("CAMERA_CADENCE", "3"))
    HISTORY_LEN = 4                                                # narrative context window
    MAX_SIDE = int(os.environ.get("MAX_SIDE", "384"))              # downsample long side
    MAX_OUT_TOKENS = int(os.environ.get("MAX_OUT_TOKENS", "60"))   # cap reply length

    # Explicit backpressure: drop frames that arrive while a caption is in flight.
    inflight = threading.Lock()

    INDEPENDENT_PROMPT = (
        "Describe this image in one short sentence. Direct, specific, no preamble."
    )
    NARRATIVE_PROMPT = (
        "You are narrating a live camera feed. Recent captions:\n{history}\n\n"
        "Describe the new frame in one short sentence. Focus on what changed. "
        "Present tense, no preamble."
    )

    def encode_frame(frame) -> str:
        """numpy RGB → base64 JPEG. Downsampled so the vision encoder is quick."""
        img = Image.fromarray(frame)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    def format_history(history: list[str]) -> str:
        return "\n".join(f"- {h}" for h in history[-HISTORY_LEN:])

    def stash_frame(frame):
        """Fast no-op for gr.Image.stream — just records the latest frame."""
        return frame

    def caption_tick(frame, mode: str, history: list[str], running: bool):
        """gr.Timer handler: caption the latest stashed frame on cadence."""
        if frame is None or not running:
            return

        if not inflight.acquire(blocking=False):
            return

        # Mark activity so the keep-alive loop knows to ping Knative.
        last_activity_ts[0] = time.time()

        try:
            img_b64 = encode_frame(frame)
            if mode == "narrative" and history:
                prompt = NARRATIVE_PROMPT.format(history=format_history(history))
            else:
                prompt = INDEPENDENT_PROMPT

            yield f"_querying **{model_id}**…_", history, format_history(history)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ]

            try:
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    stream=True,
                    temperature=0.4,
                    max_tokens=MAX_OUT_TOKENS,
                    extra_body={
                        # Caption use-case: skip thinking entirely.
                        "chat_template_kwargs": {"enable_thinking": False},
                        "skip_special_tokens": True,  # we don't parse <|channel> here
                    },
                )
            except Exception as e:
                yield f"**Error**: {e}", history, format_history(history)
                return

            reply = ""
            try:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    reply += delta
                    yield reply, history, format_history(history)
            except Exception as e:
                yield (f"**Error during streaming**: {e}\n\nPartial: {reply}",
                       history, format_history(history))
                return
            finally:
                stream.close()

            reply = reply.strip()
            if not reply:
                yield (
                    "_(empty response — model may have stopped on a stop token or "
                    "refused the frame)_",
                    history,
                    format_history(history),
                )
                return
            new_history = (history + [reply])[-HISTORY_LEN:]
            yield reply, new_history, format_history(new_history)
        finally:
            inflight.release()

    def start():
        return gr.Timer(value=CADENCE_S, active=True), True, [], "_Starting…_", ""

    def stop():
        return gr.Timer(active=False), False

    with gr.Blocks(title="Gemma 4 Live Camera") as demo:
        gr.Markdown(
            "# Gemma 4 Live Camera\n"
            "Webcam → Gemma 4 IT vision → streaming caption. "
            f"Captioning every **{CADENCE_S:g}s** "
            "(set `CAMERA_CADENCE` env var on the deploy to change). "
            f"Endpoint: `{base_url}` · model: `{model_id}`"
        )

        with gr.Row():
            mode = gr.Radio(
                ["narrative", "independent"], value="narrative", label="Mode",
                info="narrative = use recent captions as context (commentary feel)",
            )

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"], streaming=True, type="numpy",
                    label="Camera", height=360,
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop")
            with gr.Column(scale=1):
                caption = gr.Markdown(
                    "_Press **Start** to begin captioning…_",
                    label="Latest caption",
                )
                with gr.Accordion("Recent captions (narrative context)", open=False):
                    history_md = gr.Markdown()

        running = gr.State(False)
        history = gr.State([])
        latest_frame = gr.State(None)

        cam.stream(
            stash_frame,
            inputs=cam,
            outputs=latest_frame,
            stream_every=0.2,
            concurrency_limit=None,
        )

        caption_timer = gr.Timer(value=CADENCE_S, active=False)
        caption_timer.tick(
            caption_tick,
            inputs=[latest_frame, mode, history, running],
            outputs=[caption, history, history_md],
        )

        start_btn.click(
            start,
            outputs=[caption_timer, running, history, caption, history_md],
        )
        stop_btn.click(stop, outputs=[caption_timer, running])

    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    # Capture launch() return value so we can flush the public share URL to
    # stdout — Gradio prints it itself but Python's stdout buffering inside
    # the Knative pod swallows it before our infinite-sleep fallback runs.
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0", server_port=7867, share=share, prevent_thread_lock=True,
    )
    print(f"[live_camera] local URL: {local_url}", flush=True)
    if share_url:
        print(f"[live_camera] PUBLIC HTTPS URL: {share_url}", flush=True)
    else:
        print("[live_camera] no share URL (set GRADIO_SHARE=1 on deploy)", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Live-camera app deployed: {app.url}")
