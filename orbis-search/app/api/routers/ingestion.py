import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_data_ingestion_service, require_api_key
from models.schemas import (
    DataIngestionRequest,
    DataIngestionResponse,
    IngestionStatusResponse,
)
from services.data_ingestion_service import DataIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["ingestion"])


@router.post("/ingest", response_model=DataIngestionResponse, dependencies=[Depends(require_api_key)])
async def trigger_data_ingestion(
    request: DataIngestionRequest,
    ingestion_service: DataIngestionService = Depends(get_data_ingestion_service),
):
    try:
        result = await ingestion_service.ingest_workitems(
            force_full_sync=request.force_full_sync,
            skip_embedding=request.skip_embedding,
            source_names=request.source_names,
        )
        return DataIngestionResponse(**result)
    except Exception as exc:
        logger.error(f"Failed to trigger ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger ingestion: {str(exc)}")


@router.get("/ingest/status", response_model=IngestionStatusResponse, dependencies=[Depends(require_api_key)])
async def get_ingestion_status(
    ingestion_service: DataIngestionService = Depends(get_data_ingestion_service),
):
    try:
        status = ingestion_service.get_ingestion_status()
        return IngestionStatusResponse(**status)
    except Exception as exc:
        logger.error(f"Failed to get ingestion status: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to get ingestion status: {str(exc)}")


