"""Unit tests for the document chunker."""

import pytest
from langchain_core.documents import Document
from ingestion.chunker import DocumentChunker


def make_doc(text: str, source: str = "test.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "file_name": source})


class TestDocumentChunker:
    def test_basic_split(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        doc = make_doc("Hello world. " * 50)
        chunks = chunker.split([doc])
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        doc = make_doc("A" * 1000)
        chunks = chunker.split([doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= 220  # allow slight overlap overshoot

    def test_metadata_preserved(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        doc = make_doc("Hello world. " * 50, source="my_doc.pdf")
        chunks = chunker.split([doc])
        for chunk in chunks:
            assert chunk.metadata["file_name"] == "my_doc.pdf"

    def test_chunk_index_assigned(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        doc = make_doc("Hello world. " * 50)
        chunks = chunker.split([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_document(self):
        chunker = DocumentChunker()
        chunks = chunker.split([make_doc("")])
        assert len(chunks) == 0

    def test_markdown_strategy(self):
        chunker = DocumentChunker(chunk_size=200, strategy="markdown")
        md = "# Section 1\n\nSome content.\n\n# Section 2\n\nMore content."
        doc = Document(page_content=md, metadata={"source": "doc.md", "file_name": "doc.md"})
        chunks = chunker.split([doc])
        assert len(chunks) >= 1
