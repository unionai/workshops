import json
import logging
import os
from datetime import datetime

TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def setup_logging(name: str) -> logging.Logger:
    """Set up logging with noisy library suppression. Returns a logger for the calling module."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


class Logger:
    def __init__(self, path="trace_log.jsonl", verbose=False):
        os.makedirs(TRACES_DIR, exist_ok=True)
        self.path = os.path.join(TRACES_DIR, path)
        self.verbose = verbose

    async def log(self, **kwargs):
        kwargs["timestamp"] = datetime.utcnow().isoformat()
        if self.verbose:
            print("[LOG]", kwargs)
        with open(self.path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")