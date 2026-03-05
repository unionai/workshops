"""
Fraud detection workflow: prepare → (train || materialize) — mirrors Flyte pipeline.
Workflows are deterministic; all I/O goes through activities.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import copy_artifacts, materialize_features, prepare_data, train_model


@dataclass
class FraudDetectionResult:
    """Result of the fraud detection pipeline."""

    model_path: str
    feast_path: str
    data_dir: str


@workflow.defn(name="fraud-detection-workflow")
class FraudDetectionWorkflow:
    """Full fraud detection pipeline: prepare → (train + materialize) in parallel."""

    @workflow.run
    async def run(self, work_dir: str = "./artifacts", copy_to_output: bool = True) -> FraudDetectionResult:
        """Execute pipeline: work_dir=artifact root, copy_to_output=copy to work_dir/latest for serving."""
        workflow.logger.info("Starting fraud detection pipeline")

        # Unique run directory — workflow_run_id is deterministic per execution
        run_id = workflow.info().workflow_run_id.replace(":", "-").replace("/", "-")
        data_dir = str(Path(work_dir) / run_id)

        # Step 1: Prepare data
        workflow.logger.info("Preparing data...")
        await workflow.execute_activity(
            prepare_data,
            data_dir,
            start_to_close_timeout=timedelta(seconds=600),
        )

        # Step 2: Train model and materialize features in parallel (fan-out)
        workflow.logger.info("Training model and materializing features (parallel)...")
        model_path, feast_path = await asyncio.gather(
            workflow.execute_activity(
                train_model,
                data_dir,
                start_to_close_timeout=timedelta(seconds=600),
            ),
            workflow.execute_activity(
                materialize_features,
                data_dir,
                start_to_close_timeout=timedelta(seconds=300),
            ),
        )

        # Step 3: Optionally copy to standard output location for app serving
        if copy_to_output:
            output_dir = str(Path(work_dir) / "latest")
            await workflow.execute_activity(
                copy_artifacts,
                model_path,
                feast_path,
                output_dir,
                start_to_close_timeout=timedelta(seconds=60),
            )
            model_path = str(Path(output_dir) / "model.joblib")
            feast_path = str(Path(output_dir) / "feast_artifacts")

        workflow.logger.info("Pipeline complete")
        return FraudDetectionResult(
            model_path=model_path,
            feast_path=feast_path,
            data_dir=data_dir,
        )
