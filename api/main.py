"""
FastAPI application entry point.
Production-grade RAG API with structured logging, CORS, and Prometheus metrics.
"""

from __future__ import annotations

import logging
import os

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RequestLoggingMiddleware
from api.routes import router

load_dotenv()

# Structured logging setup
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))

app = FastAPI(
    title="Enterprise RAG Pipeline",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Supports PDF, HTML, JSON, and text document ingestion "
        "with FAISS/Pinecone vector storage and GPT-4 generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Enterprise RAG Pipeline",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
