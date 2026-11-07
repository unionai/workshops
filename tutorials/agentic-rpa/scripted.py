"""The deterministic twin: the same tools in a fixed order, no LLM except the desktop step and
the browser fallback. Most migrated bots should look like this."""

from __future__ import annotations

from datetime import date

import flyte

import config
import tools
from config import env


@env.task(report=True)
async def price_update_scripted(run_date: str = "", chaos: bool = False) -> dict:
    d = run_date or date.today().isoformat()
    await tools.set_portal_chaos(chaos)

    with flyte.group("check-vm"):
        state = await tools.vm_system_state()

    with flyte.group("get-supplier-prices"):
        try:
            csv_file = await tools.portal_download_prices(d)
            source = "playwright"
        except Exception as e:  # the portal changed: hand this one step to computer use
            fallback = await tools.computer_use(
                f"In the desktop web browser, open the supplier portal ({config.PORTAL_URL_ON_VM}), sign in as "
                f"'{config.PORTAL_USER}' with password '{config.PORTAL_PASSWORD}', and download the supplier "
                f"price list CSV for {d} (after signing in, {config.PORTAL_URL_ON_VM}/dashboard?d={d} shows that day's list). "
                f"Confirm prices_{d}.csv is in {config.DOWNLOADS_DIR}. "
                f"(The scripted download failed: {e})",
                max_steps=30,
            )
            if fallback.status != "done":
                raise RuntimeError(f"supplier price download failed: {fallback.summary}")
            csv_file = await tools.vm_read_file(f"{config.DOWNLOADS_DIR}/prices_{d}.csv")
            source = "computer_use"

    with flyte.group("build-and-stage"):
        workbook = await tools.build_price_changes(csv_file, d)
        remote = await tools.vm_write_file(workbook, f"{config.SHARE_DIR}/price_changes_{d}.xlsx")
        staged = await tools.vm_stage_import(remote)
        changes = await tools.read_price_changes(workbook)

    if len(changes) > config.APPROVAL_THRESHOLD:
        approved = await tools.request_approval(
            f"**{len(changes)} price changes** for {d} exceed the auto-approve limit of {config.APPROVAL_THRESHOLD}. Enter them in the ERP?"
        )
        if not approved:
            return {"date": d, "status": "rejected", "changes": len(changes)}

    with flyte.group("enter-in-erp-desktop"):
        lines = "\n".join(f"- {c['sku']}: {c['new_price']:.2f}" for c in changes)
        entered = await tools.computer_use(
            "In the 'Acme ERP 7.2 - Price Maintenance' window, enter these new prices with reason "
            f"'Supplier cost update', one SKU at a time (Find, type the new price, Save, confirm if asked):\n{lines}",
            max_steps=20 + 6 * len(changes),
        )

    with flyte.group("verify"):
        current = {s["sku"]: s["price"] for s in await tools.erp_list_skus()}
        mismatches = [c["sku"] for c in changes if tools.cents(current.get(c["sku"], -1)) != tools.cents(c["new_price"])]

    return {
        "date": d,
        "status": "ok" if not mismatches else "mismatch",
        "price_source": source,
        "changes": len(changes),
        "staged": staged,
        "desktop": entered.status,
        "mismatches": mismatches,
        "vm_state": state["report"][:500],
    }
