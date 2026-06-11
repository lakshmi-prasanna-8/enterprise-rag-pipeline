"""
Embedding service supporting OpenAI and HuggingFace backends.
Handles batching and retry logic for production reliability.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["openai", "huggingface"]


class EmbeddingService:
    """
    Wraps OpenAI and HuggingFace embedding models behind a common interface.
    Defaults to OpenAI text-embedding-3-small (~$0.02/1M tokens).
    Falls back to HuggingFace all-MiniLM-L6-v2 for zero-cost local dev.
    """

    def __init__(
        self,
        backend: EmbeddingBackend = "openai",
        model_name: str | None = None,
        batch_size: int = 100,
    ):
        self.backend = backend
        self.batch_size = batch_size
        self._model = self._build_model(model_name)

    def _build_model(self, model_name: str | None) -> Embeddings:
        if self.backend == "openai":
            name = model_name or os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
            logger.info("Using OpenAI embeddings: %s", name)
            return OpenAIEmbeddings(
                model=name,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        # HuggingFace fallback
        name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Using HuggingFace embeddings: %s", name)
        return HuggingFaceEmbeddings(model_name=name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches with automatic retry."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._model.embed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug("Embedded batch %d/%d", i // self.batch_size + 1, -(-len(texts) // self.batch_size))
        return all_embeddings

    @property
    def langchain_embeddings(self) -> Embeddings:
        """Return the raw LangChain embeddings object for direct use in vector stores."""
        return self._model
