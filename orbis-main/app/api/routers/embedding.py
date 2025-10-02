import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_embedding_service,
    get_vector_service,
    require_api_key,
)
from app.db.models import DataSource
from app.db.session import get_db_session
from core.schemas import EmbedRequest, EmbedResponse
from core.services.generic_content_service import GenericContentService
from infrastructure.connectors.azure_devops.work_item_service import WorkItemService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["embedding"])


def _load_content():
    """Load all content for embedding"""
    with GenericContentService() as content_service:
        all_content = content_service.get_all_content_for_embedding()
        return all_content


@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
async def embed_content(
    request: EmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: GenericVectorService = Depends(get_vector_service),
):
    """Generate embeddings for all content from all configured data sources using incremental approach"""
    try:
        # Process all content from database using incremental approach
        result = await embedding_service.generate_embeddings_from_db_content(
            vector_service=vector_service,
            content_ids=None,  # Process all content
            data_source_id=None,  # From all sources
            force_rebuild=request.force_rebuild,
            batch_size=None
        )

        return EmbedResponse(
            message=result.get("message", "Embeddings processed"),
            total_tickets=result.get("total_items", 0),
            processed_tickets=result.get("processed_items", 0),
            success=result.get("success", True),
        )
    except Exception as exc:
        logger.error(f"Failed to embed content: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to embed content: {str(exc)}") from exc


@router.post("/embed/{source_name}", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
async def embed_source(
    source_name: str,
    request: EmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: GenericVectorService = Depends(get_vector_service),
):
    """Generate embeddings for all content types from a specific source using incremental approach"""
    try:
        # Get data source ID
        with get_db_session() as db:
            data_source = db.query(DataSource).filter(DataSource.name == source_name).first()
            if not data_source:
                return EmbedResponse(
                    message=f"Data source '{source_name}' not found",
                    total_tickets=0,
                    processed_tickets=0,
                    success=False,
                )

        # Process content for this specific data source using incremental approach
        result = await embedding_service.generate_embeddings_from_db_content(
            vector_service=vector_service,
            content_ids=None,  # Process all content for this source
            data_source_id=data_source.id,
            force_rebuild=request.force_rebuild,
            batch_size=None
        )

        return EmbedResponse(
            message=f"Embeddings processed for source '{source_name}': {result.get('message', '')}",
            total_tickets=result.get("total_items", 0),
            processed_tickets=result.get("processed_items", 0),
            success=result.get("success", True),
        )
    except Exception as exc:
        logger.error(f"Failed to embed content for source '{source_name}': {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to embed content for source '{source_name}': {str(exc)}") from exc


@router.get("/embed/state", dependencies=[Depends(require_api_key)])
async def get_embedding_state(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: GenericVectorService = Depends(get_vector_service),
):
    """Get current embedding state including what needs updating"""
    try:
        state = await embedding_service.get_embedding_state()
        return state
    except Exception as exc:
        logger.error(f"Failed to get embedding state: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding state: {str(exc)}") from exc




