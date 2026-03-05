"""Start fraud detection workflow — requires: temporal server start-dev, python worker.py."""

import asyncio
import logging
import os
import shutil
import uuid

from temporalio.client import Client

from workflows import FraudDetectionWorkflow, FraudDetectionResult

TASK_QUEUE = "fraud-detection-task-queue"
WORKFLOW_ID_PREFIX = "fraud-detection"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    client = await Client.connect("localhost:7233")

    workflow_id = f"{WORKFLOW_ID_PREFIX}-{uuid.uuid4().hex[:8]}"
    work_dir = os.path.abspath("./artifacts")

    logger.info("Starting workflow %s (work_dir=%s)", workflow_id, work_dir)

    result = await client.execute_workflow(
        FraudDetectionWorkflow.run,
        args=[work_dir, True],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    assert isinstance(result, FraudDetectionResult)
    logger.info("Workflow completed successfully")
    logger.info("  Model:  %s", result.model_path)
    logger.info("  Feast:  %s", result.feast_path)
    logger.info("  Data:   %s", result.data_dir)

    # Copy to cwd for local app testing (matches Flyte behavior)
    if os.path.exists(result.model_path):
        shutil.copy2(result.model_path, "model.joblib")
        logger.info("Copied model to model.joblib")
    if os.path.exists(result.feast_path):
        if os.path.exists("feast_artifacts"):
            shutil.rmtree("feast_artifacts")
        shutil.copytree(result.feast_path, "feast_artifacts")
        logger.info("Copied Feast artifacts to feast_artifacts/")

    print("\n" + "=" * 60)
    print("Pipeline complete. Run the scoring app:")
    print("  python app.py")
    print("\nTest with:")
    print('  curl "http://localhost:8080/score?user_id=42&amt=25.00&category=grocery_pos&merch_lat=33.9&merch_long=-80.3"')
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
