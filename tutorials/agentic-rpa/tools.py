"""The bot's tools, one Flyte task per capability, grouped by rung of the mechanism ladder.

Every call is its own action in the run, with typed inputs and outputs, so the run graph is a
record of exactly which mechanisms the bot used and what each one returned.
"""

from __future__ import annotations

import csv
import html
import io
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date

import flyte
import httpx
from flyte.io import File

import computer_use as cu
import config
import vm
from config import browser_env, cu_env, env

# The toy apps on the VM, each a supervisord program (see target_vm/supervisord.conf).
SERVICES = ("erp_desktop", "erp_api", "supplier_portal", "computer_server", "drop_watcher")


def _today(run_date: str) -> str:
    return run_date or date.today().isoformat()


def _erp() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=config.ERP_URL, headers={"X-API-Key": config.ERP_API_KEY}, timeout=30)


# ---------------------------------------------------------------- rung 1: API


@env.task
async def erp_list_skus() -> list[dict]:
    """List all active SKUs in the Acme ERP with their current prices (REST API)."""
    async with _erp() as c:
        r = await c.get("/skus")
        r.raise_for_status()
        return r.json()


@env.task
async def erp_set_price_via_api(sku: str, new_price: float) -> dict:
    """Try to update a SKU's price through the ERP REST API. Returns the API's answer, including refusals."""
    async with _erp() as c:
        r = await c.post(f"/skus/{sku}/price", json={"price": new_price})
        return {"ok": r.is_success, "status": r.status_code, "detail": r.json().get("detail", r.text)}


@env.task
async def erp_audit(since: str = "") -> list[dict]:
    """Price-change audit log from the ERP (REST API). `since` is an ISO timestamp; empty for all."""
    async with _erp() as c:
        r = await c.get("/audit", params={"since": since} if since else None)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------- rung 2: system hooks / state


@env.task
async def vm_system_state() -> dict:
    """Health of the RPA VM: its app services, disk, uptime, and recent hook/import log lines (SSH)."""
    res = await vm.run(
        "echo '## services'; supervisorctl status; echo '## disk'; df -h /srv | tail -1; "
        "echo '## load'; uptime; echo '## hooks'; tail -n 5 /var/log/acme/hooks.log 2>/dev/null; "
        "echo '## imports'; tail -n 5 /var/log/acme/import.log 2>/dev/null; true"
    )
    return {"report": res["stdout"]}


@env.task
async def vm_restart_service(service: str) -> dict:
    """Restart one supervised app on the VM, e.g. `erp_desktop` if the ERP window is hung or missing."""
    if service not in SERVICES:
        raise ValueError(f"service must be one of {sorted(SERVICES)}")
    return await vm.run(f"supervisorctl restart {service} && sleep 3 && supervisorctl status {service}")


# ---------------------------------------------------------------- rung 3: scripts and shell


@env.task
async def vm_shell(command: str, timeout_s: int = 120) -> dict:
    """Run a bash command on the RPA VM as user `rpa` over SSH. Returns exit_code, stdout, stderr."""
    return await vm.run(command, timeout=timeout_s)


@env.task
async def vm_stage_import(path: str) -> dict:
    """Run the VM's stage_import script on a price-change workbook: validates columns, copies it to the import folder, logs."""
    import json

    res = await vm.run(f"/opt/acme/scripts/stage_import.sh {vm.q(path)}")
    try:
        return {**json.loads(res["stdout"].strip().splitlines()[-1]), "exit_code": res["exit_code"]}
    except Exception:
        return res


# ---------------------------------------------------------------- rung 4: files


@env.task
async def vm_list_files(directory: str) -> list[str]:
    """List files in a directory on the VM (newest first), e.g. the rpa user's Downloads or the share folder."""
    res = await vm.run(f"ls -1t {vm.q(directory)}", check=True)
    return [line for line in res["stdout"].splitlines() if line]


@env.task
async def vm_read_file(path: str) -> File:
    """Copy a file from the VM into Flyte storage (SFTP) so other tools can use it."""
    local = os.path.join(tempfile.mkdtemp(), path.rsplit("/", 1)[-1])
    await vm.get(path, local)
    return await File.from_local(local)


@env.task
async def vm_write_file(file: File, path: str) -> str:
    """Copy a file from Flyte storage onto the VM at `path` (SFTP). Returns the path."""
    local = await file.download()
    await vm.run(f"mkdir -p {vm.q(path.rsplit('/', 1)[0])}", check=True)
    await vm.put(local, path)
    return path


@env.task
async def build_price_changes(supplier_csv: File, run_date: str = "") -> File:
    """Diff the supplier price file against current ERP prices; write price_changes_<date>.xlsx (only changed SKUs)."""
    from openpyxl import Workbook

    async with supplier_csv.open("rb") as f:
        rows = list(csv.DictReader(io.StringIO(bytes(await f.read()).decode())))
    async with _erp() as c:
        r = await c.get("/skus")
        r.raise_for_status()
        erp = {s["sku"]: s for s in r.json()}

    wb = Workbook()
    ws = wb.active
    ws.title = "price_changes"
    ws.append(["sku", "description", "current_price", "new_price", "pct_change"])
    for row in rows:
        cur = erp.get(row["sku"])
        new = round(float(row["supplier_price"]), 2)
        if cur and abs(new - cur["price"]) >= 0.01:
            ws.append([row["sku"], row["description"], cur["price"], new, round(100 * (new - cur["price"]) / cur["price"], 1)])
    out = os.path.join(tempfile.mkdtemp(), f"price_changes_{_today(run_date)}.xlsx")
    wb.save(out)
    return await File.from_local(out)


@env.task
async def read_price_changes(workbook: File) -> list[dict]:
    """Read a price_changes workbook into rows of {sku, description, current_price, new_price, pct_change}."""
    from openpyxl import load_workbook

    local = await workbook.download()
    ws = load_workbook(local, read_only=True).active
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    return [dict(zip(header, r)) for r in it if r and r[0]]


# ---------------------------------------------------------------- rung 5: browser (DOM-driven)


@browser_env.task(report=True)
async def portal_download_prices(run_date: str = "") -> File:
    """Log into the supplier portal with Playwright and export today's price list CSV.
    Fails with the selector that broke if the portal's layout has changed."""
    from playwright.async_api import TimeoutError as PwTimeout
    from playwright.async_api import async_playwright

    d = _today(run_date)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(accept_downloads=True, viewport={"width": 1280, "height": 800})
        step = "open login page"
        try:
            await page.goto(config.PORTAL_URL, timeout=20_000)
            step = "sign in"
            await page.fill("#username", config.PORTAL_USER)
            await page.fill("#password", config.PORTAL_PASSWORD)
            await page.click("#login-btn")
            step = "find #export-btn"
            async with page.expect_download(timeout=10_000) as dl:
                await page.click("#export-btn", timeout=8_000)
            download = await dl.value
            local = os.path.join(tempfile.mkdtemp(), f"prices_{d}.csv")
            await download.save_as(local)
            return await File.from_local(local)
        except PwTimeout as e:
            import base64

            shot = base64.b64encode(await page.screenshot()).decode()
            tab = flyte.report.get_tab("Browser")
            tab.log(f"<h3>Failed at: {html.escape(step)}</h3><p>{html.escape(str(e).splitlines()[0])}</p>"
                    f"<img style='width:640px;border:1px solid #ccc' src='data:image/png;base64,{shot}'/>")
            await flyte.report.flush.aio()
            raise RuntimeError(
                f"Portal automation failed at step '{step}': the page no longer matches the script "
                f"(likely a portal redesign). Playwright: {str(e).splitlines()[0]}"
            ) from None
        finally:
            await browser.close()


# ---------------------------------------------------------------- rung 6: computer use


@dataclass
class ComputerUseResult:
    status: str
    summary: str
    steps: int
    actions: list[str] = field(default_factory=list)
    recording: File | None = None


@cu_env.task(report=True)
async def computer_use(goal: str, max_steps: int = 40) -> ComputerUseResult:
    """Operate the VM's desktop like a person (screenshots, mouse, keyboard) to accomplish `goal`.
    Slow and expensive: use only when no API, script, file or DOM-based tool can do the step.
    The session is screen-recorded; the mp4 is attached to the result."""
    rec = cu.Recorder(cu.new_session_name())
    await rec.start()
    try:
        res = await cu.run_loop(goal, max_steps=max_steps)
    finally:
        local = await rec.stop()
    recording = await File.from_local(local) if local else None
    if local and os.path.getsize(local) < 25_000_000:
        import base64

        b64 = base64.b64encode(open(local, "rb").read()).decode()
        flyte.report.get_tab(cu.REPORT_TAB).log(
            f"<h3>Recording</h3><video controls style='width:960px' src='data:video/mp4;base64,{b64}'></video>"
        )
        await flyte.report.flush.aio()
    return ComputerUseResult(res.status, res.summary, res.steps, res.actions, recording)


# ---------------------------------------------------------------- human in the loop


@env.task
async def request_approval(summary: str) -> bool:
    """Pause the run until a person approves or rejects. Use before committing a large batch of changes."""
    cond = await flyte.new_condition.aio(
        "approve-price-changes", prompt=summary, prompt_type="markdown", data_type=bool,
        description="Agentic RPA: approve price changes before they are entered in the ERP",
    )
    return bool(await cond.wait.aio())


# ---------------------------------------------------------------- demo setup (not given to the agent)


@env.task
async def set_portal_chaos(on: bool) -> str:
    """Toggle the supplier portal's 'overnight redesign' that breaks the Playwright script."""
    await vm.run("touch /srv/acme/chaos" if on else "rm -f /srv/acme/chaos", check=True)
    return "chaos on" if on else "chaos off"


@env.task
async def reset_demo() -> str:
    """Reset ERP prices and folders on the VM to the seed state."""
    await vm.run(
        "rm -f /srv/acme/chaos /home/rpa/Downloads/* /srv/share/* /srv/import/* /var/log/acme/import.log; "
        "supervisorctl stop erp_desktop erp_api; ACME_DB=/srv/acme/acme.db /opt/acme/venv/bin/python /opt/acme/acme_db.py --reset; "
        "supervisorctl start erp_api erp_desktop",
        check=True,
    )
    return "reset"
