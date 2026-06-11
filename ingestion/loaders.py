"""
Document loaders supporting PDF, HTML, JSON, and plain text.
Mirrors enterprise ingestion pipeline built at American Airlines
handling 500K+ operational documents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".json", ".txt", ".md"}


class DocumentLoader:
    """
    Unified document loader that dispatches to the right parser
    based on file extension and yields LangChain Document objects.
    """

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)

    def load(self) -> list[Document]:
        """Load all documents from a file or directory."""
        docs: list[Document] = []
        if self.source_path.is_file():
            docs = list(self._load_file(self.source_path))
        elif self.source_path.is_dir():
            for path in self.source_path.rglob("*"):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    docs.extend(self._load_file(path))
        else:
            raise FileNotFoundError(f"Path not found: {self.source_path}")

        logger.info("Loaded %d raw documents from %s", len(docs), self.source_path)
        return docs

    def _load_file(self, path: Path) -> Iterator[Document]:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                yield from self._load_pdf(path)
            elif ext in {".html", ".htm"}:
                yield from self._load_html(path)
            elif ext == ".json":
                yield from self._load_json(path)
            else:
                yield from self._load_text(path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    def _load_pdf(self, path: Path) -> Iterator[Document]:
        loader = PyPDFLoader(str(path))
        for doc in loader.load():
            doc.metadata["source_type"] = "pdf"
            doc.metadata["file_name"] = path.name
            yield doc

    def _load_html(self, path: Path) -> Iterator[Document]:
        loader = BSHTMLLoader(str(path))
        for doc in loader.load():
            doc.metadata["source_type"] = "html"
            doc.metadata["file_name"] = path.name
            yield doc

    def _load_json(self, path: Path) -> Iterator[Document]:
        with path.open() as f:
            data = json.load(f)
        # Support both a single dict and a list of records
        records = data if isinstance(data, list) else [data]
        for i, record in enumerate(records):
            text = record.get("text") or record.get("content") or json.dumps(record)
            yield Document(
                page_content=text,
                metadata={
                    "source_type": "json",
                    "file_name": path.name,
                    "record_index": i,
                    **{k: v for k, v in record.items() if k not in {"text", "content"}},
                },
            )

    def _load_text(self, path: Path) -> Iterator[Document]:
        loader = TextLoader(str(path), encoding="utf-8")
        for doc in loader.load():
            doc.metadata["source_type"] = "text"
            doc.metadata["file_name"] = path.name
            yield doc
