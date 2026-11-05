"""System hook (rung 2): watch the drop folder and fire a signed webhook for each new file.

Polling keeps it dependency-free; on Windows the same role is a FileSystemWatcher or a
scheduled task. With no HOOK_URL set it only logs, so the VM works before the hook app exists.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import platform
import time
import urllib.request
from pathlib import Path

DROP = Path(os.environ.get("ACME_DROP", "/srv/drop"))
HOOK_URL = os.environ.get("HOOK_URL", "")
HOOK_SECRET = os.environ.get("HOOK_SECRET", "")
POLL_S = float(os.environ.get("HOOK_POLL_S", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s drop_watcher %(message)s")
log = logging.getLogger()


def fire(path: Path) -> None:
    data = path.read_bytes()
    body = json.dumps({
        "event": "file.dropped",
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "host": platform.node(),
    }).encode()
    if not HOOK_URL:
        log.info("no HOOK_URL; would send %s", body.decode())
        return
    sig = hmac.new(HOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        HOOK_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Acme-Signature": f"sha256={sig}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        log.info("hook %s -> %s %s", path.name, resp.status, resp.read(200).decode(errors="replace"))


def main() -> None:
    DROP.mkdir(parents=True, exist_ok=True)
    seen = {p.name for p in DROP.iterdir()}
    log.info("watching %s (%d existing files ignored)", DROP, len(seen))
    while True:
        for p in sorted(DROP.iterdir()):
            if p.name in seen or p.name.startswith("."):
                continue
            seen.add(p.name)
            try:
                fire(p)
            except Exception as e:  # keep watching; the next file still gets its hook
                log.error("hook for %s failed: %s", p.name, e)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
