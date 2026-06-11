"""
Prometheus metrics for the RAG pipeline.
Tracks request count, latency, token usage, and retrieval hit rate.
"""

from prometheus_client import Counter, Histogram, Gauge

RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total number of RAG query requests",
)

RAG_LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 0.8, 1.0, 2.0, 5.0],
)

RAG_TOKEN_USAGE = Histogram(
    "rag_token_usage_total",
    "Total tokens used per RAG request",
    buckets=[100, 250, 500, 1000, 2000, 4000, 8000],
)

INGESTION_DOCS_PROCESSED = Counter(
    "rag_ingestion_docs_total",
    "Total documents processed through ingestion pipeline",
)

INGESTION_CHUNKS_CREATED = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks created during ingestion",
)

VECTOR_STORE_SIZE = Gauge(
    "rag_vector_store_size",
    "Number of vectors currently in the store",
)

EMBEDDING_DRIFT_SCORE = Gauge(
    "rag_embedding_drift_score",
    "Current embedding distribution drift score (0=stable, 1=high drift)",
)
