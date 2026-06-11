"""MLflow experiment tracking for RAG evaluation runs."""

from __future__ import annotations

import logging
import os

import mlflow

logger = logging.getLogger(__name__)


class MLflowLogger:
    def __init__(self, experiment_name: str = "enterprise-rag"):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_eval(self, metrics: dict, params: dict, tags: dict | None = None) -> str:
        """Log an evaluation run and return the run ID."""
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            run_id = run.info.run_id
            logger.info("MLflow run logged: %s", run_id)
            return run_id

    def log_inference(self, question: str, latency_ms: float, tokens: int, cost: float) -> None:
        """Log a single inference call — useful for production dashboards."""
        with mlflow.start_run(run_name="inference"):
            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "tokens_used": tokens,
                "estimated_cost_usd": cost,
            })
            mlflow.set_tag("question_preview", question[:100])
