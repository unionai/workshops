"""Toy Acme ERP REST API (rung 1). Reads are open to the bot; price overrides are not.

    uvicorn erp_api:app --port 8001
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException

import acme_db

API_KEY = os.environ.get("ACME_API_KEY", "demo-key")

app = FastAPI(title="Acme ERP API", version="1.0")


def _auth(x_api_key: str = Header(default="")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(401, "missing or bad X-API-Key")


@app.on_event("startup")
def _startup() -> None:
    acme_db.init()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/skus", dependencies=[Depends(_auth)])
def skus() -> list[dict]:
    return acme_db.list_skus()


@app.get("/skus/{sku}", dependencies=[Depends(_auth)])
def sku(sku: str) -> dict:
    row = acme_db.get_sku(sku)
    if row is None:
        raise HTTPException(404, f"unknown sku {sku}")
    return row


@app.post("/skus/{sku}/price", dependencies=[Depends(_auth)])
def set_price(sku: str) -> dict:
    # The reason the demo needs a desktop step: like many ERPs, this one only accepts price
    # overrides through its thick client.
    raise HTTPException(
        403,
        "Price overrides are only accepted through the Acme ERP desktop client "
        "(Price Maintenance screen), policy ERP-114.",
    )


@app.get("/audit", dependencies=[Depends(_auth)])
def audit(since: str | None = None) -> list[dict]:
    return acme_db.audit(since)
