import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_data_ingestion_service, require_api_key
from app.db.session import get_db_session
from engine.schemas import (
    DataIngestionRequest,
    DataIngestionResponse,
)
from engine.services.generic_data_ingestion import GenericDataIngestionService
from infrastructure.data_processing.data_source_service import DataSourceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["ingestion"])


@router.post("/ingest", response_model=DataIngestionResponse, dependencies=[Depends(require_api_key)])
async def trigger_data_ingestion(
    request: DataIngestionRequest,
    ingestion_service: GenericDataIngestionService = Depends(get_data_ingestion_service),
):
    """Trigger data ingestion for all enabled data sources"""
    try:
        result = await ingestion_service.ingest_all_sources(
            force_full_sync=request.force_full_sync,
            skip_embedding=request.skip_embedding,
            source_names=None,
        )
        return DataIngestionResponse(**result)
    except Exception as exc:
        logger.error(f"Failed to trigger ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger ingestion: {str(exc)}") from exc


@router.post("/ingest/{source_name}", dependencies=[Depends(require_api_key)])
async def ingest_data_source(source_name: str, request: dict[str, Any] = None):
    """Ingest data from a specific source using generic ingestion service"""
    try:
        request = request or {}

        # Get data source from database
        with get_db_session() as db:
            ds_service = DataSourceService(db)
            data_source = ds_service.get_data_source(source_name)
            if not data_source:
                raise HTTPException(status_code=404, detail=f"Data source '{source_name}' not found")

        # Use generic ingestion service
        generic_ingestion_service = GenericDataIngestionService()
        result = await generic_ingestion_service.ingest_source(
            source_name=data_source.name,
            source_type=data_source.source_type,
            source_config=data_source.config,
            incremental=request.get('incremental', True)
        )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to ingest data source {source_name}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest data source: {str(exc)}") from exc
