"""Unit tests for the retriever using a mock vector store."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from pipeline.retriever import RAGRetriever, RetrievalResult


def make_docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"Chunk {i} content", metadata={"file_name": f"doc_{i}.pdf"})
        for i in range(n)
    ]


class TestRAGRetriever:
    def setup_method(self):
        self.mock_store = MagicMock()
        self.mock_store.similarity_search_with_score.return_value = [
            (doc, 0.85) for doc in make_docs(5)
        ]
        self.mock_store.max_marginal_relevance_search.return_value = make_docs(5)

    def test_retrieve_returns_result(self):
        retriever = RAGRetriever(self.mock_store, top_k=5, use_mmr=False)
        result = retriever.retrieve("test question")
        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 5

    def test_mmr_retrieve(self):
        retriever = RAGRetriever(self.mock_store, top_k=5, use_mmr=True)
        result = retriever.retrieve("test question")
        self.mock_store.max_marginal_relevance_search.assert_called_once()
        assert len(result.documents) == 5

    def test_score_threshold_filters(self):
        self.mock_store.similarity_search_with_score.return_value = [
            (make_docs(1)[0], 0.1),  # below threshold
            (make_docs(1)[0], 0.9),  # above threshold
        ]
        retriever = RAGRetriever(self.mock_store, top_k=5, use_mmr=False, score_threshold=0.5)
        result = retriever.retrieve("query")
        assert len(result.documents) == 1

    def test_query_stored_in_result(self):
        retriever = RAGRetriever(self.mock_store, use_mmr=False)
        result = retriever.retrieve("my question")
        assert result.query == "my question"
