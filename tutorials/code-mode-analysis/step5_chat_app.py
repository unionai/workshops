"""Step 5 (stretch) — serve it as a chat app.

The web layer is one declaration: `AgentChatAppEnvironment` brings the chat UI, the
tools sidebar, progress streaming, and the chat endpoint. We supply the agent.

There are two ways to run it, and the difference is exactly the point of step 3.

  Local — serve the app on your laptop, agent in-process:

      uv run python step5_chat_app.py

  No cluster, no image build, no deploy. The whole loop runs in the web process,
  which makes it the right way to iterate on the prompt or the UI. What you give
  up is the thing step 3 was about: the sandbox's `query` calls run in-process
  too, so there are no durable child tasks and no real fan-out.

  Deployed — the app runs on the cluster and every question is a durable run:

      uv run python step5_chat_app.py deploy

  `task_entrypoint=answer` is what buys that. An app's request handler has no task
  context, so calling the agent straight from it would run the sandboxed queries
  inside the app pod. With a task entrypoint, each question becomes a real
  `flyte.run`: the queries fan out as durable child tasks and the UI streams the
  run's progress. `passthrough_auth=True` forwards the signed-in user's credentials
  to those runs, so the analysis executes as them and not as a shared identity.

Either way you need the model key. For the local server, put it in a `.env` file —
`config.py` loads it. For the deployed one, register a Flyte secret in the same
project and domain you deploy to:

    flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import timedelta

import flyte
from flyte.ai.agents import AgentResult
from flyte.app import Scaling
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

import tools
from analysis import build_agent
from config import image
from config import env as task_env


class _Runner:
    """A callable with an `.aio`, which is the shape AgentProtocol expects."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, message: str, memory=None) -> AgentResult:
        return asyncio.run(self._fn(message, memory))

    async def aio(self, message: str, memory=None) -> AgentResult:
        return await self._fn(message, memory)


# {{docs-fragment adapter}}
class TaxiAnalyst:
    """The code-mode agent, dressed for the chat UI.

    The UI renders whatever it finds on `AgentResult.charts`. Our render tools
    don't *return* HTML — they append it to a per-run collector and hand the model
    back a one-line confirmation, which is what keeps the sandbox observations
    small. This class is the seam between those two facts.

    It also gives every question a fresh collector, so two people chatting at the
    same time don't end up in each other's reports.

    Nothing is built in __init__: `flyte.deploy` pickles this object, and an Agent
    holds a live HTTP client, which is not picklable. Build agents on use, not on
    import.
    """

    _introspection = None

    def tool_descriptions(self) -> list[dict]:
        if self._introspection is None:
            # Only so the UI can list the tools in its sidebar.
            self._introspection, _ = build_agent(code_mode=True)
        return self._introspection.tool_descriptions()

    async def _answer(self, message: str, memory=None) -> AgentResult:
        tools.new_report()
        agent, usage = build_agent(code_mode=True)
        result = await agent.run.aio(message, memory=memory)

        return AgentResult(
            code="\n\n# --- next program ---\n\n".join(usage.programs) or result.code,
            charts=tools.collect_report(),
            summary=result.summary,
            error=result.error,
            attempts=result.attempts,
        )

    @property
    def run(self) -> _Runner:
        """The chat UI calls `agent.run(...)` or `agent.run.aio(...)`.

        Built on access rather than stored, and by hand rather than with @syncify:
        `flyte.deploy` pickles this object, and a syncified method drags a
        `contextvars.Context` along with it, which cannot be pickled.
        """
        return _Runner(self._answer)
# {{/docs-fragment adapter}}


agent = TaxiAnalyst()


def _run_link() -> str:
    """A link back to this task's own run, so the answer is click-through evidence.

    Returns "" when there is no run to link to (i.e. the local server), which is
    exactly when there is nothing to see anyway.
    """
    try:
        from flyte._initialize import get_client

        action = flyte.ctx().action
        url = get_client().console.run_url(
            project=action.project, domain=action.domain, run_name=action.run_name
        )
        return (
            f'<p style="margin:12px 0 0;font-size:.85rem">'
            f'<a href="{url}" target="_blank" rel="noopener">'
            f"View this run in Flyte →</a> "
            f'<span style="opacity:.6">every query the model wrote is a child task'
            f"</span></p>"
        )
    except Exception:
        return ""


@task_env.task(report=True)
async def answer(message: str, history: list[dict[str, str]]) -> dict:
    """One question = one durable run. Only the deployed app uses this.

    Because it runs inside a task context, the `query` calls the sandboxed program
    makes dispatch as durable child tasks — the thing the local server cannot
    give you.
    """
    result = await agent.run.aio(message, memory=history)

    # The UI renders `charts` as raw HTML, so the link rides along with them.
    blocks = list(result.charts)
    link = _run_link()
    if link:
        blocks.append(link)

    return {
        "summary": result.summary or "",
        "charts": blocks,
        "code": result.code,
        "error": result.error,
    }


# {{docs-fragment chat_app}}
DEPLOY = "deploy" in sys.argv


def chat_env(local: bool = False) -> AgentChatAppEnvironment:
    """Build the chat app.

    `local=False` is the deployed shape and must be what lives at module level: the
    app pod re-imports this module to start the server, and its argv has no "deploy"
    in it. Gate the task entrypoint on a flag like that and the deployed app quietly
    loses it — the agent then runs *inside the web process*, DuckDB materializes a
    month of trips in the app pod, and the container gets OOM-killed.
    """
    durable = (
        {}
        if local
        # A durable run per question, executed as the caller. Neither applies
        # locally: no cluster to dispatch to, nobody to authenticate as.
        else {"task_entrypoint": answer, "passthrough_auth": True, "requires_auth": True}
    )
    return AgentChatAppEnvironment(
        name="taxi-analyst-chat",
        agent=agent,
        **durable,
        title="NYC Taxi analyst",
        subtitle="Ask a question. Claude writes a program, the sandbox runs it, "
        "and every query it writes becomes a durable task.",
        theme=CustomTheme(
            accent_color="#f2b01e", accent_hover_color="#f5c754",
            button_text_color="#0a0a0f",
        ),
        prompt_nudges=[
            {
                "label": "Tipping by borough",
                "prompt": "Do riders in Brooklyn and the Bronx really tip less than "
                "Manhattan, or is something else going on?",
            },
            {
                "label": "A year of tipping",
                "prompt": "How did the tip rate move month by month through 2024? "
                "Chart it.",
            },
            {
                "label": "Airport runs",
                "prompt": "How do JFK and LaGuardia airport pickups compare on trip "
                "distance, fare, and tip rate across 2024?",
            },
        ],
        depends_on=[task_env],
        image=image,  # fastapi + uvicorn are in the shared image (config.py)
        secrets=task_env.secrets,
        # The app pod only orchestrates — the analysis runs in `answer` — but the
        # agent loop and the chat UI still need headroom.
        resources=flyte.Resources(cpu=1, memory="2Gi"),
        # Apps default to scale-to-zero (replicas=(0, 1)), so the pod is torn down
        # when idle and the next visitor waits through a cold start. Keep one replica
        # warm and let it burst to three. Set this back to (0, 1) after the workshop —
        # an always-on replica costs money whether anyone is using it or not.
        scaling=Scaling(replicas=(1, 3), scaledown_after=timedelta(minutes=30)),
    )


# Module level, and deliberately the *deployed* shape: the app pod imports this
# module to boot the server, so this is the object it serves.
env = chat_env()
# {{/docs-fragment chat_app}}


if __name__ == "__main__":
    if DEPLOY:
        flyte.init_from_config(image_builder="remote")
        print(flyte.deploy(env))
    else:
        import uvicorn

        flyte.init()  # local: tasks run in-process, not on the cluster
        port = int(os.environ.get("PORT", "8080"))
        print(f"\n  NYC Taxi analyst → http://localhost:{port}\n")
        uvicorn.run(chat_env(local=True).build_fastapi_app(), host="0.0.0.0", port=port)
