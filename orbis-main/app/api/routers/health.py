import logging
from datetime import datetime

from fastapi import APIRouter, Depends

from app.api.dependencies import get_embedding_service, get_vector_service
from core.schemas import HealthResponse
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: GenericVectorService = Depends(get_vector_service),
):
    """Simple health check for essential services"""

    # Check basic service availability
    model_loaded = True
    chroma_connected = True
    azure_openai_configured = True
    collection_info = None
    total_documents = 0

    try:
        embedding_service.get_model_info()
    except Exception as e:
        logger.warning(f"Embedding service unavailable: {e}")
        model_loaded = False
        azure_openai_configured = False

    try:
        collection_info = vector_service.get_collection_info()
        total_documents = collection_info.get("total_tickets", 0)
    except Exception as e:
        logger.warning(f"Vector service unavailable: {e}")
        chroma_connected = False

    status = "healthy" if (model_loaded and chroma_connected and azure_openai_configured) else "unhealthy"

    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        chroma_connected=chroma_connected,
        azure_openai_configured=azure_openai_configured,
        total_tickets=total_documents,
        database_info=collection_info,
    )


