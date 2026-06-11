"""
Airflow DAG: enterprise document ingestion pipeline.
Mirrors production MLOps pipeline built at JPMorgan Chase and American Airlines.
Runs nightly to pick up new documents and re-index.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

SOURCE_PATH = "/opt/airflow/data/source_docs"


def load_documents(**context) -> None:
    from ingestion.loaders import DocumentLoader

    loader = DocumentLoader(SOURCE_PATH)
    docs = loader.load()
    context["ti"].xcom_push(key="doc_count", value=len(docs))
    print(f"Loaded {len(docs)} documents")


def chunk_documents(**context) -> None:
    from ingestion.loaders import DocumentLoader
    from ingestion.chunker import DocumentChunker
    import json, os

    loader = DocumentLoader(SOURCE_PATH)
    docs = loader.load()

    chunker = DocumentChunker(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
    )
    chunks = chunker.split(docs)
    context["ti"].xcom_push(key="chunk_count", value=len(chunks))
    print(f"Created {len(chunks)} chunks")


def embed_and_index(**context) -> None:
    from ingestion.loaders import DocumentLoader
    from ingestion.chunker import DocumentChunker
    from ingestion.embedder import EmbeddingService
    from ingestion.vector_store import VectorStoreFactory
    import os

    loader = DocumentLoader(SOURCE_PATH)
    docs = loader.load()
    chunker = DocumentChunker(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
    )
    chunks = chunker.split(docs)

    backend = os.getenv("EMBEDDING_BACKEND", "openai")
    embedder = EmbeddingService(backend=backend)
    VectorStoreFactory.create(chunks, embedder.langchain_embeddings)
    print(f"Indexed {len(chunks)} chunks into vector store")


def validate_index(**context) -> None:
    """Smoke test: run a test query and assert we get results."""
    from ingestion.embedder import EmbeddingService
    from ingestion.vector_store import VectorStoreFactory
    import os

    embedder = EmbeddingService(backend=os.getenv("EMBEDDING_BACKEND", "openai"))
    store = VectorStoreFactory.load(embedder.langchain_embeddings)
    results = store.similarity_search("test query", k=3)
    assert len(results) > 0, "Index validation failed: no results returned"
    print(f"Index validation passed — got {len(results)} results")


with DAG(
    dag_id="enterprise_rag_ingestion",
    default_args=DEFAULT_ARGS,
    description="Nightly ingestion pipeline: load → chunk → embed → index",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "nlp", "ingestion"],
) as dag:

    t1 = PythonOperator(task_id="load_documents", python_callable=load_documents)
    t2 = PythonOperator(task_id="chunk_documents", python_callable=chunk_documents)
    t3 = PythonOperator(task_id="embed_and_index", python_callable=embed_and_index)
    t4 = PythonOperator(task_id="validate_index", python_callable=validate_index)

    t1 >> t2 >> t3 >> t4
