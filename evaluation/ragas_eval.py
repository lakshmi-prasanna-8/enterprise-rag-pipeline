"""
RAGAS evaluation: measures faithfulness, answer relevancy, and context recall.
Logs results to MLflow for experiment tracking.

This is the key differentiator on your resume — most RAG candidates
don't measure quality, they just deploy.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from evaluation.mlflow_logger import MLflowLogger

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).parent / "benchmark_dataset.json"


def run_evaluation(
    chain=None,
    benchmark_path: str | Path = BENCHMARK_PATH,
    experiment_name: str | None = None,
) -> dict:
    """
    Run RAGAS evaluation on the benchmark dataset and log to MLflow.

    Returns a dict of metric names → scores.
    """
    from dotenv import load_dotenv
    load_dotenv()

    if chain is None:
        from pipeline.rag_chain import RAGChain
        chain = RAGChain.from_env()

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    logger.info("Running RAGAS eval on %d questions…", len(benchmark))

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in benchmark:
        response = chain.query(item["question"])
        questions.append(item["question"])
        answers.append(response.answer)
        contexts.append([doc["preview"] for doc in response.sources])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    logger.info("RAGAS scores: %s", scores)

    mlflow_logger = MLflowLogger(
        experiment_name=experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "enterprise-rag")
    )
    mlflow_logger.log_eval(
        metrics=scores,
        params={
            "top_k": os.getenv("TOP_K", "5"),
            "chunk_size": os.getenv("CHUNK_SIZE", "512"),
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "n_questions": len(benchmark),
        },
    )

    return scores


if __name__ == "__main__":
    import pprint
    scores = run_evaluation()
    pprint.pprint(scores)
