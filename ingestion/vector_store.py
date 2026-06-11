"""
Vector store abstraction: FAISS for local dev, Pinecone for production.
Single factory interface so the rest of the codebase is backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

VectorStoreBackend = Literal["faiss", "pinecone"]


class VectorStoreFactory:
    """
    Creates and manages vector store instances.

    Usage:
        store = VectorStoreFactory.create(chunks, embeddings, backend="faiss")
        results = store.similarity_search("query", k=5)
    """

    @staticmethod
    def create(
        documents: list[Document],
        embeddings: Embeddings,
        backend: VectorStoreBackend | None = None,
    ) -> VectorStore:
        backend = backend or os.getenv("VECTOR_STORE_BACKEND", "faiss")
        if backend == "pinecone":
            return VectorStoreFactory._create_pinecone(documents, embeddings)
        return VectorStoreFactory._create_faiss(documents, embeddings)

    @staticmethod
    def load(
        embeddings: Embeddings,
        backend: VectorStoreBackend | None = None,
    ) -> VectorStore:
        """Load an existing index without re-ingesting documents."""
        backend = backend or os.getenv("VECTOR_STORE_BACKEND", "faiss")
        if backend == "pinecone":
            return VectorStoreFactory._load_pinecone(embeddings)
        return VectorStoreFactory._load_faiss(embeddings)

    # ── FAISS ──────────────────────────────────────────────────────────────

    @staticmethod
    def _create_faiss(documents: list[Document], embeddings: Embeddings) -> FAISS:
        logger.info("Building FAISS index from %d chunks…", len(documents))
        store = FAISS.from_documents(documents, embeddings)
        index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        store.save_local(index_path)
        logger.info("FAISS index saved to %s", index_path)
        return store

    @staticmethod
    def _load_faiss(embeddings: Embeddings) -> FAISS:
        index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        logger.info("Loading FAISS index from %s", index_path)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # ── Pinecone ────────────────────────────────────────────────────────────

    @staticmethod
    def _create_pinecone(documents: list[Document], embeddings: Embeddings) -> PineconeVectorStore:
        index_name = os.getenv("PINECONE_INDEX_NAME", "enterprise-rag")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        if index_name not in [i.name for i in pc.list_indexes()]:
            logger.info("Creating Pinecone index: %s", index_name)
            pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                ),
            )

        logger.info("Upserting %d chunks to Pinecone index %s…", len(documents), index_name)
        store = PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=index_name,
        )
        return store

    @staticmethod
    def _load_pinecone(embeddings: Embeddings) -> PineconeVectorStore:
        index_name = os.getenv("PINECONE_INDEX_NAME", "enterprise-rag")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
