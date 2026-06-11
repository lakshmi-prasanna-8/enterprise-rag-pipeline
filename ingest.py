from dotenv import load_dotenv
load_dotenv()

from ingestion.loaders import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingService
from ingestion.vector_store import VectorStoreFactory

docs = DocumentLoader('./sample_docs').load()
print(f"Loaded {len(docs)} documents")

chunks = DocumentChunker().split(docs)
print(f"Created {len(chunks)} chunks")

embedder = EmbeddingService(backend='huggingface')

VectorStoreFactory.create(chunks, embedder.langchain_embeddings)
print("Index saved! You are ready to query.")