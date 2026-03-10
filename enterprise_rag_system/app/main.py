"""
Enterprise Knowledge Intelligence System — Application Entry Point
Initialises the FastAPI application, registers middleware, mounts the API
router, and provides lifecycle hooks for startup / shutdown logging.

Run directly:
    python -m app.main

Or via uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from app.config import API_HOST, API_PORT
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Knowledge Intelligence System",
    description=(
        "A production-grade Retrieval Augmented Generation (RAG) platform. "
        "Upload documents and ask natural-language questions answered by a "
        "local Mistral-7B-Instruct model grounded on your document corpus."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Allow cross-origin requests from any frontend during development.
# Tighten allow_origins in production to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    logger.info("=" * 60)
    logger.info("Enterprise Knowledge Intelligence System — Starting up")
    logger.info(f"API listening on http://{API_HOST}:{API_PORT}")
    logger.info("Interactive docs: http://127.0.0.1:8000/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Enterprise Knowledge Intelligence System — Shutting down")

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Health / root endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Confirm the service is running."""
    return {
        "service": "Enterprise Knowledge Intelligence System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Kubernetes / load-balancer health probe."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
