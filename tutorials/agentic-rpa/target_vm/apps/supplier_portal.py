"""Toy supplier portal (rung 5): login, then export today's price list as CSV.

Chaos mode (the file $ACME_HOME/chaos exists) is "the vendor shipped a redesign overnight":
a cookie banner covers the page, and the export moves into a Reports menu with a new label and id.
Playwright selectors written against the normal layout fail; a person (or computer use) copes.

    uvicorn supplier_portal:app --port 8002
"""

from __future__ import annotations

import csv
import html
import io
import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

import acme_db

HOME = Path(os.environ.get("ACME_HOME", "/srv/acme"))
USER = os.environ.get("PORTAL_USER", "acme-buyer")
PASSWORD = os.environ.get("PORTAL_PASSWORD", "demo-password")
SESSION = "portal-session-ok"

app = FastAPI(title="BuildSupply Partner Portal")

STYLE = """
<style>
 body{font-family:DejaVu Sans,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1d2733}
 header{background:#0b6e4f;color:#fff;padding:14px 24px;font-size:20px}
 main{padding:24px;max-width:900px}
 .card{background:#fff;border:1px solid #d5dbe1;border-radius:6px;padding:18px;margin-bottom:16px}
 button,.btn{background:#0b6e4f;color:#fff;border:0;padding:9px 16px;border-radius:4px;font-size:15px;cursor:pointer;text-decoration:none;display:inline-block}
 input{padding:7px;font-size:15px;margin:4px 0 10px;width:240px}
 table{border-collapse:collapse} td{padding:4px 12px;border-bottom:1px solid #eee}
 details{display:inline-block;position:relative} summary{list-style:none;cursor:pointer;padding:8px 14px;border:1px solid #0b6e4f;border-radius:4px;color:#0b6e4f}
 details[open] .menu{position:absolute;background:#fff;border:1px solid #ccc;padding:8px;min-width:260px;z-index:5}
 .menu a{display:block;padding:6px;color:#1d2733}
 #consent{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:10}
 #consent .box{background:#fff;padding:24px;border-radius:8px;max-width:420px}
</style>"""


def _chaos() -> bool:
    return (HOME / "chaos").exists()


def _authed(request: Request) -> bool:
    return request.cookies.get("session") == SESSION


@app.get("/", response_class=HTMLResponse)
def login_page(error: str = "") -> str:
    msg = f"<p style='color:#b00020'>{html.escape(error)}</p>" if error else ""
    return f"""<html><head><title>BuildSupply Partner Portal - Sign in</title>{STYLE}</head><body>
<header>BuildSupply Partner Portal</header><main><div class="card"><h2>Partner sign in</h2>{msg}
<form method="post" action="/login">
 <label>Username</label><br><input id="username" name="username" autocomplete="off"><br>
 <label>Password</label><br><input id="password" name="password" type="password"><br>
 <button id="login-btn" type="submit">Sign in</button>
</form></div></main></body></html>"""


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)) -> Response:
    if username != USER or password != PASSWORD:
        return RedirectResponse("/?error=Invalid+credentials", status_code=303)
    resp = RedirectResponse("/dashboard", status_code=303)
    resp.set_cookie("session", SESSION, httponly=True)
    return resp


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, d: str = "") -> Response:
    if not _authed(request):
        return RedirectResponse("/", status_code=303)
    # The price list for `d` (YYYY-MM-DD), or for today. The variable keeps its old name.
    today = date.fromisoformat(d).isoformat() if d else date.today().isoformat()
    news = """<div class="card"><h3>Announcements</h3><table>
 <tr><td>Oct 01</td><td>Q4 freight surcharge update for Gulf region</td></tr>
 <tr><td>Sep 22</td><td>New SKUs: synthetic underlayment 10 sq</td></tr></table></div>"""
    if not _chaos():
        export = f"""<div class="card"><h3>Price lists</h3>
 <p>Daily price file for <b>{today}</b> is available.</p>
 <a id="export-btn" class="btn" href="/export?d={today}">Export price list (CSV)</a></div>"""
        consent = ""
    else:
        export = f"""<div class="card"><h3>Workspace</h3>
 <details><summary>Reports &#9662;</summary><div class="menu">
  <a href="/orders">Open orders</a>
  <a data-qa="dl-pricing-v2" href="/export?d={today}">Download supplier pricing ({today})</a>
  <a href="/rebates">Rebate statements</a></div></details></div>"""
        consent = """<div id="consent"><div class="box"><h3>We value your privacy</h3>
 <p>We use cookies to improve the portal experience.</p>
 <button onclick="document.getElementById('consent').remove()">Accept all</button></div></div>"""
    return HTMLResponse(f"""<html><head><title>BuildSupply Partner Portal - Dashboard</title>{STYLE}</head><body>
<header>BuildSupply Partner Portal &nbsp;&middot;&nbsp; Acme</header><main>{consent}{news}{export}</main></body></html>""")


@app.get("/export")
def export(request: Request, d: str) -> Response:
    if not _authed(request):
        raise HTTPException(401, "sign in first")
    rows = acme_db.supplier_prices(date.fromisoformat(d))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prices_{d}.csv"'},
    )


@app.get("/{page}", response_class=HTMLResponse)
def other(page: str) -> str:
    return f"<html><head>{STYLE}</head><body><header>BuildSupply Partner Portal</header><main><p>{html.escape(page)}: nothing here yet.</p><a href='/dashboard'>Back</a></main></body></html>"
