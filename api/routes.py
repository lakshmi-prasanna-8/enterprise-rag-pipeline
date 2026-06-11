"""API route handlers."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, UploadFile, File
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    HealthResponse, IngestRequest, IngestResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-loaded chain (initialised on first request)
_chain = None


def get_chain():
    global _chain
    if _chain is None:
        from pipeline.rag_chain import RAGChain
        _chain = RAGChain.from_env()
    return _chain


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — confirms the API and vector store are up."""
    return HealthResponse(
        status="ok",
        vector_store=os.getenv("VECTOR_STORE_BACKEND", "faiss"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "openai"),
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG query endpoint.
    Retrieves relevant document chunks and generates a grounded answer.
    """
    try:
        chain = get_chain()
        response = chain.query(request.question)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=[SourceDocument(**s) for s in response.sources],
        latency_ms=response.generation.latency_ms,
        tokens_used=response.generation.total_tokens,
        estimated_cost_usd=response.generation.estimated_cost_usd,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Trigger a manual ingestion run from a server-side source path.
    In production this is driven by the Airflow DAG on a schedule.
    """
    try:
        from ingestion.loaders import DocumentLoader
        from ingestion.chunker import DocumentChunker
        from ingestion.embedder import EmbeddingService
        from ingestion.vector_store import VectorStoreFactory

        loader = DocumentLoader(request.source_path)
        docs = loader.load()

        chunker = DocumentChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
        )
        chunks = chunker.split(docs)

        embedder = EmbeddingService(backend=os.getenv("EMBEDDING_BACKEND", "openai"))
        VectorStoreFactory.create(chunks, embedder.langchain_embeddings)

        # Reset cached chain so it reloads fresh index
        global _chain
        _chain = None

    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(chunks),
    )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
