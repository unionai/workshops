"""Step 5: the agent crashes after kicking off the fine-tunes, resumes, and nothing is redone.

A durable run is not one process. This task dies after its third live model call on the
first attempt, which for this agent is typically right after the fine-tune results come
back. Flyte retries it in a fresh container, where:

- the model turns it already paid for are *replayed* from their durable records
  (`ai_node` wraps each turn in `flyte.trace`), so the model is not called again;
- the evals and fine-tunes it already ran are cache hits on their child actions, so no
  T4 spins up a second time and no model is trained twice;
- the work picks up where it left off, and the decision is produced once.

That is the Flyte half. The Grafana half is what the plugin adds on top of a stock
OpenTelemetry setup: both attempts record into the *same* trace, because the trace id is
derived from the run rather than minted per process, and the replayed steps appear in it
marked `flyte.replayed`, so the trace has no holes where durability did its job.

The crash is gated to the first attempt on a backend (attempt numbers start at 0). Under
`--local` Flyte does not persist trace records across attempts, so the crash is skipped
and the agent simply runs.

Replay depends on the transcript hashing the same way on every attempt; see the note on
stable message ids in graph.py.

    flyte run step5_crash_resume.py resilient_engineer
    flyte run --local step5_crash_resume.py resilient_engineer     # runs once, no crash
"""

from __future__ import annotations

import json
import os

import flyte
import flyte.report
from langchain_core.language_models.chat_models import BaseChatModel

from config import GRAFANA_LINKS, observed_env
from graph import Request
from graph import engineer as _engineer
from llm import chat_model
from report import decision_html


class CrashingChatModel(BaseChatModel):
    """A chat model that dies after N *live* calls.

    Replayed turns never reach `_agenerate`, so on the retry the counter only sees the
    calls that actually happen. The absence of the first N live calls on attempt two is
    the replay, made visible.
    """

    inner: BaseChatModel
    crash_after: int
    armed: bool
    live_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return f"crashing-{self.inner._llm_type}"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.live_calls += 1
        print(f"  live model call #{self.live_calls}", flush=True)
        result = await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if self.armed and self.live_calls >= self.crash_after:
            # After the call returned and was recorded, so the retry replays it.
            raise RuntimeError(f"simulated worker crash after {self.live_calls} model calls (first attempt only)")
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        # Let the inner model format the tools, then bind them onto this wrapper so the
        # calls still route through _agenerate.
        bound = self.inner.bind_tools(tools, **kwargs)
        return self.bind(**dict(getattr(bound, "kwargs", {}) or {}))

    def with_structured_output(self, schema, **kwargs):
        return self.inner.with_structured_output(schema, **kwargs)


@observed_env.task(report=True, retries=3, links=GRAFANA_LINKS)
async def resilient_engineer(
    crash_after: int = 3,
    min_accuracy: float = 0.95,
    max_latency_ms: float = 150.0,
    candidates_to_screen: int = 5,
    max_fine_tunes: int = 3,
    budget: int = 12,
    dataset: str = "v1",
    model: str | None = None,
) -> dict:
    """Run the factory, crash once partway through, resume, finish. One trace."""
    on_backend = os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None
    attempt = flyte.ctx().attempt_number if flyte.ctx() else 0
    armed = on_backend and attempt == 0  # FLYTE_ATTEMPT_NUMBER is 0-based on a backend: the first attempt is 0
    print(f"attempt {attempt} on_backend={on_backend} crash_armed={armed}", flush=True)

    request = Request(min_accuracy, max_latency_ms, candidates_to_screen, max_fine_tunes, budget, dataset=dataset)
    result = await _engineer(
        request,
        model_spec=model,
        model=CrashingChatModel(inner=chat_model(model), crash_after=crash_after, armed=armed),
    )
    await flyte.report.replace.aio(decision_html(result))
    await flyte.report.flush.aio()
    print(json.dumps(result, indent=2))
    if on_backend:
        print(
            f"finished on attempt {attempt}: turns and tool runs before the crash were replayed, not redone", flush=True
        )
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(resilient_engineer)
    print(run.url)
