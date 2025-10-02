import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_embedding_service, require_api_key
from models.schemas import EmbeddingProviderInfo
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["embedding-provider"])


@router.get("/embedding/provider", response_model=EmbeddingProviderInfo, dependencies=[Depends(require_api_key)])
async def get_embedding_provider_info(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        info = embedding_service.get_model_info()
        return EmbeddingProviderInfo(**info)
    except Exception as exc:
        logger.error(f"Failed to get embedding provider info: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding provider info: {str(exc)}")



