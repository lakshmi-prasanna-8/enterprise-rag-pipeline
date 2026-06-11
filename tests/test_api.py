"""Integration tests for the FastAPI endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.fixture
def mock_chain():
    chain = MagicMock()
    response = MagicMock()
    response.question = "What is RAG?"
    response.answer = "RAG stands for Retrieval-Augmented Generation."
    response.sources = [
        {
            "file": "intro.pdf",
            "page": 1,
            "chunk_index": 0,
            "preview": "RAG stands for Retrieval-Augmented Generation...",
        }
    ]
    response.generation.latency_ms = 450.0
    response.generation.total_tokens = 512
    response.generation.estimated_cost_usd = 0.003
    chain.query.return_value = response
    return chain


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "Enterprise RAG Pipeline" in resp.json()["service"]


@pytest.mark.asyncio
async def test_query_endpoint(mock_chain):
    with patch("api.routes.get_chain", return_value=mock_chain):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/query",
                json={"question": "What is RAG?"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert data["latency_ms"] > 0


@pytest.mark.asyncio
async def test_query_requires_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/query", json={})
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert b"rag_requests_total" in resp.content
