"""
Top-level RAG chain: wires retriever → prompt builder → generator.
This is the single entry point for query execution.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import logging
import os
from dataclasses import dataclass

from langchain_core.vectorstores import VectorStore

from ingestion.embedder import EmbeddingService
from ingestion.vector_store import VectorStoreFactory
from pipeline.retriever import RAGRetriever, RetrievalResult
from pipeline.prompt_builder import PromptBuilder
from pipeline.generator import LLMGenerator, GenerationResult
from monitoring.prometheus_metrics import RAG_LATENCY, RAG_TOKEN_USAGE, RAG_REQUESTS

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    retrieval: RetrievalResult
    generation: GenerationResult


class RAGChain:
    """
    End-to-end RAG pipeline.

    Usage:
        chain = RAGChain.from_env()
        response = chain.query("What is the maintenance schedule for aircraft X?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int | None = None,
        use_mmr: bool = True,
        max_context_tokens: int = 3000,
    ):
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            top_k=top_k or int(os.getenv("TOP_K", 5)),
            use_mmr=use_mmr,
        )
        self.prompt_builder = PromptBuilder(max_context_tokens=max_context_tokens)
        self.generator = LLMGenerator()

    @classmethod
    def from_env(cls) -> "RAGChain":
        """Construct the chain by loading the vector store from env config."""
        backend = os.getenv("EMBEDDING_BACKEND", "openai")
        embedder = EmbeddingService(backend=backend)
        store = VectorStoreFactory.load(embedder.langchain_embeddings)
        return cls(vector_store=store)

    def query(self, question: str) -> RAGResponse:
        """Run a question through the full RAG pipeline."""
        RAG_REQUESTS.inc()

        retrieval = self.retriever.retrieve(question)
        messages = self.prompt_builder.build(question, retrieval.documents)
        generation = self.generator.generate(messages)

        RAG_LATENCY.observe(generation.latency_ms / 1000)
        RAG_TOKEN_USAGE.observe(generation.total_tokens)

        sources = [
            {
                "file": doc.metadata.get("file_name", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "preview": doc.page_content[:200],
            }
            for doc in retrieval.documents
        ]

        return RAGResponse(
            question=question,
            answer=generation.answer,
            sources=sources,
            retrieval=retrieval,
            generation=generation,
        )
