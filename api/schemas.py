"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The user's question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    use_mmr: bool = Field(True, description="Use MMR re-ranking to reduce redundancy")


class SourceDocument(BaseModel):
    file: str
    page: int | None = None
    chunk_index: int | None = None
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    latency_ms: float
    tokens_used: int
    estimated_cost_usd: float


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_backend: str


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to documents folder on the server")


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
