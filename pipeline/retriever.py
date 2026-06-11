"""
Retrieval layer: similarity search with MMR re-ranking and score filtering.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: list[float]
    query: str
    top_k: int


class RAGRetriever:
    """
    Wraps a vector store and exposes similarity search with:
    - Configurable top-k
    - MMR (Maximal Marginal Relevance) to reduce redundancy
    - Score threshold filtering to drop low-quality chunks
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int | None = None,
        score_threshold: float = 0.3,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ):
        self.vector_store = vector_store
        self.top_k = top_k or int(os.getenv("TOP_K", 5))
        self.score_threshold = score_threshold
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query."""
        if self.use_mmr:
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=self.top_k,
                fetch_k=self.top_k * 3,
                lambda_mult=self.mmr_lambda,
            )
            scores = [0.0] * len(docs)  # MMR doesn't return scores natively
        else:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.top_k
            )
            docs_with_scores = [
                (doc, score)
                for doc, score in docs_with_scores
                if score >= self.score_threshold
            ]
            docs = [d for d, _ in docs_with_scores]
            scores = [s for _, s in docs_with_scores]

        logger.info(
            "Retrieved %d chunks for query: '%s…'",
            len(docs),
            query[:60],
        )
        return RetrievalResult(
            documents=docs,
            scores=scores,
            query=query,
            top_k=self.top_k,
        )
