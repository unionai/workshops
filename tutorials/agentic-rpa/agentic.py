"""The agentic version: an agent picks mechanisms from the ladder at run time. Each tool call it
makes is a Flyte action, so the run graph is the lineage of the agent's decisions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import flyte
from flyte.ai.agents import Agent

import config
import llm
import tools
from config import env

INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text().format(
    approval_threshold=config.APPROVAL_THRESHOLD,
    portal_user=config.PORTAL_USER,
    portal_password=config.PORTAL_PASSWORD,
    portal_url=config.PORTAL_URL_ON_VM,
    share_price_changes=f"{config.SHARE_DIR}/price_changes_<date>.xlsx",
    downloads_dir=config.DOWNLOADS_DIR,
)

agent = Agent(
    name="price-update",
    instructions=INSTRUCTIONS,
    model=config.ORCHESTRATOR_MODEL,
    call_llm=llm.make_call_llm(effort="medium"),
    tools=[
        tools.erp_list_skus, tools.erp_set_price_via_api, tools.erp_audit,
        tools.vm_system_state, tools.vm_restart_service,
        tools.vm_shell, tools.vm_stage_import,
        tools.vm_list_files, tools.vm_read_file, tools.vm_write_file,
        tools.build_price_changes, tools.read_price_changes,
        tools.portal_download_prices,
        tools.computer_use,
        tools.request_approval,
    ],
    max_turns=40,
    # Steps have side effects on one desktop; run them in the order the agent chose.
    parallel_tool_calls=False,
)

# Active once the task is deployed (`flyte deploy agentic.py env`); `flyte run` ignores it.
daily = flyte.Trigger(
    "daily-price-update",
    flyte.Cron("0 6 * * *", timezone="America/Chicago"),
    inputs={"trigger_time": flyte.TriggerTime, "chaos": False},
)


@env.task(triggers=daily, report=True)
async def price_update(trigger_time: datetime | None = None, chaos: bool = False) -> dict:
    run_date = (trigger_time.date() if trigger_time else date.today()).isoformat()
    await tools.set_portal_chaos(chaos)
    result = await agent.run.aio(f"Run the daily supplier price update for {run_date}.")
    if result.error:
        # Fail the run, so a broken bot shows up as a failed run rather than a green one.
        raise RuntimeError(f"price update for {run_date} failed after {result.attempts} turns: {result.error}")
    return {"run_date": run_date, "summary": result.summary, "turns": result.attempts}
