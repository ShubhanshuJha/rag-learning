"""
FastAPI application entrypoint.

Registers the /ingest and /ask routers, and exposes /health, which
verifies both Weaviate and Ollama are actually reachable — check this
first any time /ask or /ingest behaves oddly.
"""

import httpx
from fastapi import FastAPI, Response

from app.config import settings
from app.models.schemas import HealthResponse
from app.routers import ask_router, ingest_router
from app.services import vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AWS DMS Documentation Assistant",
    description="Ingest technical PDFs and ask grounded questions against them.",
    version="0.1.0",
)

app.include_router(ingest_router.router, tags=["ingest"])
app.include_router(ask_router.router, tags=["ask"])


@app.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    weaviate_status = "ok"
    ollama_status = "ok"

    try:
        with vector_store.get_client() as client:
            if not client.is_ready():
                weaviate_status = "unreachable"
    except Exception as exc:
        logger.warning("Weaviate health check failed: %s", exc)
        weaviate_status = "unreachable"

    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            resp = await http_client.get(settings.ollama_api_endpoint)
            ollama_status = "ok" if resp.status_code == 200 else "unreachable"
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        ollama_status = "unreachable"

    if weaviate_status != "ok" or ollama_status != "ok":
        response.status_code = 503

    return HealthResponse(weaviate=weaviate_status, ollama=ollama_status)
