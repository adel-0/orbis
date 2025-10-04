from fastapi import Request, HTTPException
import os

from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.summary_service import SummaryService
from services.data_ingestion_service import DataIngestionService
from services.service_container import ServiceContainer
from orbis_core.search import RerankService, BM25Service, HybridSearchService
from orbis_core.scheduling import SchedulerService


def get_container(request: Request) -> ServiceContainer:
    # Be resilient if startup didn't attach the container (e.g., in certain test contexts)
    container = getattr(request.app.state, "service_container", None)
    if container is None:
        from app.core.container import create_container  # local import to avoid cycles in startup
        container = create_container()
        request.app.state.service_container = container
    return container


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.service_container.get_service("embedding")


def get_vector_service(request: Request) -> VectorService:
    return request.app.state.service_container.get_service("vector")


def get_summary_service(request: Request) -> SummaryService:
    return request.app.state.service_container.get_service("summary")


def get_data_ingestion_service(request: Request) -> DataIngestionService:
    return request.app.state.service_container.get_service("data_ingestion")


def get_scheduler_service(request: Request) -> SchedulerService:
    return request.app.state.service_container.get_service("scheduler")


def get_rerank_service(request: Request) -> RerankService:
    return request.app.state.service_container.get_service("rerank")


def get_bm25_service(request: Request) -> BM25Service:
    return request.app.state.service_container.get_service("bm25")


def get_hybrid_search_service(request: Request) -> HybridSearchService:
    return request.app.state.service_container.get_service("hybrid_search")


async def require_api_key(request: Request) -> None:
    required = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    if not required:
        return None
    provided = request.headers.get("X-API-Key")
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API key auth enabled but API_KEY not configured")
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


