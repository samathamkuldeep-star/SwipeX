"""FastAPI Application factory and lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.endpoints import health
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import RequestIDMiddleware, setup_logging

setup_logging()
logger = logging.getLogger("swipex.ai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown."""
    import asyncio

    logger.info("Starting SwipeX AI/NLP Service...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"LLM Enabled: {settings.LLM_ENABLED}")

    # Load spaCy model (blocking — run in thread)
    from app.services.ai.skill_extraction import load_spacy_model

    await asyncio.to_thread(load_spacy_model)

    # Load sentence-transformers embedding service (blocking — run in thread)
    from app.services.ai.embeddings import load_embedding_service

    await asyncio.to_thread(load_embedding_service)

    # Initialise ChromaDB vector store (potentially blocking HTTP connection)
    from app.services.ai.vector_store import get_vector_store

    await get_vector_store().init()

    logger.info("All services initialised — ready to serve.")
    yield

    logger.info("Shutting down SwipeX AI/NLP Service...")



def create_application() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="SwipeX AI/NLP Service",
        description="AI/ML Career Intelligence, Resume Extraction, ATS Scoring & Semantic Recommendations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)

    # Custom Exception Handlers
    register_exception_handlers(app)

    # Top-level health/ready routes
    app.include_router(health.router, tags=["Health"])

    # Versioned API routes under /api/v1
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_application()
