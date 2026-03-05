"""Temporal worker — run with: python worker.py (requires temporal server start-dev)."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from activities import copy_artifacts, materialize_features, prepare_data, train_model
from workflows import FraudDetectionWorkflow

TASK_QUEUE = "fraud-detection-task-queue"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client, task_queue=TASK_QUEUE,
        workflows=[FraudDetectionWorkflow],
        activities=[prepare_data, train_model, materialize_features, copy_artifacts],
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    logger.info("Worker started, polling %s", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
