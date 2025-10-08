import os
from functools import lru_cache

from fastapi import HTTPException, Request

from config.settings import settings
from core.agents.llm_routing_agent import QueryRoutingAgent
from core.agents.orchestrator import AgenticRAGOrchestrator
from core.agents.summary_agent import SearchResultsSummarizer
from core.services.generic_data_ingestion import GenericDataIngestionService
from infrastructure.data_processing.scheduler_service import SchedulerService
from orbis_core.llm.openai_client import OpenAIClientService
from orbis_core.search import RerankService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService


# Simple cached service creation - replaces complex service container
@lru_cache(maxsize=1)
def _create_openai_client_service() -> OpenAIClientService:
    return OpenAIClientService(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
    )

@lru_cache(maxsize=1)
def _create_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@lru_cache(maxsize=1)
def _create_vector_service() -> GenericVectorService:
    return GenericVectorService()

@lru_cache(maxsize=1)
def _create_rerank_service() -> RerankService:
    return RerankService()

@lru_cache(maxsize=1)
def _create_search_summarizer() -> SearchResultsSummarizer:
    return SearchResultsSummarizer(_create_openai_client_service())

@lru_cache(maxsize=1)
def _create_data_ingestion() -> GenericDataIngestionService:
    return GenericDataIngestionService(_create_embedding_service(), _create_vector_service())

@lru_cache(maxsize=1)
def _create_scheduler() -> SchedulerService:
    return SchedulerService(_create_data_ingestion())

@lru_cache(maxsize=1)
def _create_query_routing() -> QueryRoutingAgent:
    return QueryRoutingAgent()

@lru_cache(maxsize=1)
def _create_agentic_rag() -> AgenticRAGOrchestrator:
    return AgenticRAGOrchestrator(
        vector_service=_create_vector_service(),
        embedding_service=_create_embedding_service(),
        rerank_service=_create_rerank_service(),
        openai_client_service=_create_openai_client_service()
    )


# Simple service getters - no complex container needed
def get_embedding_service(request: Request) -> EmbeddingService:
    return _create_embedding_service()

def get_vector_service(request: Request) -> GenericVectorService:
    return _create_vector_service()

def get_search_results_summarizer(request: Request) -> SearchResultsSummarizer:
    return _create_search_summarizer()

def get_data_ingestion_service(request: Request) -> GenericDataIngestionService:
    return _create_data_ingestion()

def get_scheduler_service(request: Request) -> SchedulerService:
    return _create_scheduler()

def get_rerank_service(request: Request) -> RerankService:
    return _create_rerank_service()

def get_agentic_rag_orchestrator(request: Request) -> AgenticRAGOrchestrator:
    return _create_agentic_rag()

def get_query_routing_agent(request: Request) -> QueryRoutingAgent:
    return _create_query_routing()

def get_openai_client_service(request: Request) -> OpenAIClientService:
    return _create_openai_client_service()



# Simplified API key validation
def _check_api_key_required() -> tuple[bool, str]:
    """Check if API key is required and get expected key"""
    if os.getenv("API_KEY_ENABLED", "false").lower() != "true":
        return False, ""

    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(500, "API key auth enabled but API_KEY not configured")

    return True, expected
async def require_api_key(request: Request) -> None:
    """Simple API key validation"""
    required, expected = _check_api_key_required()
    if not required:
        return

    if request.headers.get("X-API-Key") != expected:
        raise HTTPException(401, "Invalid or missing API key")
