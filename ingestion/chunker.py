"""
Chunking strategies for enterprise documents.
RecursiveCharacterTextSplitter with configurable chunk size and overlap.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["recursive", "markdown", "token"]


class DocumentChunker:
    """
    Splits documents into chunks suitable for embedding and retrieval.

    Strategy selection:
    - recursive: best for mixed/prose content (default)
    - markdown: preserves heading hierarchy for .md files
    - token: exact token budget for cost-sensitive embedding
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: ChunkStrategy = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        if self.strategy == "markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        if self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",
            )
        # Default: recursive
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into chunks, preserving metadata."""
        chunks = self._splitter.split_documents(documents)

        # Enrich metadata with chunk position info
        chunk_counter: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            idx = chunk_counter.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk_counter[source] = idx + 1

        logger.info(
            "Chunked %d documents → %d chunks (size=%d, overlap=%d, strategy=%s)",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
            self.strategy,
        )
        return chunks
