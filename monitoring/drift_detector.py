"""
Embedding drift detection using cosine similarity distribution comparison.
Alerts when query embedding distribution shifts significantly from
the reference distribution captured at index time.

Mirrors drift detection built for production MLOps pipelines at
Aetna (5+ active models) and American Airlines (6 production models).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

from monitoring.prometheus_metrics import EMBEDDING_DRIFT_SCORE

logger = logging.getLogger(__name__)

REFERENCE_PATH = "./data/drift_reference.pkl"
DRIFT_THRESHOLD = 0.15  # KS statistic threshold for alerting


class EmbeddingDriftDetector:
    """
    Compares live query embeddings against a reference distribution
    using the Kolmogorov-Smirnov test on pairwise cosine similarities.

    Usage:
        detector = EmbeddingDriftDetector()
        detector.set_reference(reference_embeddings)
        drift_score = detector.check(new_query_embeddings)
    """

    def __init__(self, reference_path: str = REFERENCE_PATH, window_size: int = 200):
        self.reference_path = Path(reference_path)
        self.window_size = window_size
        self._reference_sims: np.ndarray | None = None
        self._load_reference()

    def set_reference(self, embeddings: list[list[float]]) -> None:
        """Capture baseline pairwise similarity distribution."""
        arr = np.array(embeddings)
        self._reference_sims = self._pairwise_similarities(arr)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reference_path, "wb") as f:
            pickle.dump(self._reference_sims, f)
        logger.info("Drift reference set from %d embeddings", len(embeddings))

    def check(self, embeddings: list[list[float]]) -> float:
        """
        Compute drift score vs reference.
        Returns KS statistic in [0, 1]; higher = more drift.
        """
        if self._reference_sims is None:
            logger.warning("No drift reference set — skipping drift check")
            return 0.0

        arr = np.array(embeddings[-self.window_size:])
        live_sims = self._pairwise_similarities(arr)
        ks_stat, p_value = ks_2samp(self._reference_sims, live_sims)

        EMBEDDING_DRIFT_SCORE.set(ks_stat)

        if ks_stat > DRIFT_THRESHOLD:
            logger.warning(
                "Embedding drift detected! KS=%.3f (threshold=%.3f, p=%.4f)",
                ks_stat, DRIFT_THRESHOLD, p_value,
            )
        else:
            logger.debug("Drift check passed: KS=%.3f", ks_stat)

        return float(ks_stat)

    def _load_reference(self) -> None:
        if self.reference_path.exists():
            with open(self.reference_path, "rb") as f:
                self._reference_sims = pickle.load(f)
            logger.info("Loaded drift reference from %s", self.reference_path)

    @staticmethod
    def _pairwise_similarities(arr: np.ndarray) -> np.ndarray:
        """Compute upper-triangle pairwise cosine similarities."""
        n = len(arr)
        sims = []
        for i in range(min(n, 50)):  # Cap at 50 for performance
            for j in range(i + 1, min(n, 50)):
                sim = 1.0 - cosine(arr[i], arr[j])
                sims.append(sim)
        return np.array(sims) if sims else np.array([0.0])
