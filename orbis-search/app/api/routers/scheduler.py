import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_scheduler_service, require_api_key
from models.schemas import SchedulerStatusResponse, DataIngestionResponse
from orbis_core.scheduling import SchedulerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["scheduler"])


@router.get("/scheduler/status", response_model=SchedulerStatusResponse, dependencies=[Depends(require_api_key)])
async def get_scheduler_status(scheduler: SchedulerService = Depends(get_scheduler_service)):
    try:
        status = scheduler.get_status()
        return SchedulerStatusResponse(**status)
    except Exception as exc:
        logger.error(f"Failed to get scheduler status: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(exc)}")


@router.post("/scheduler/trigger", response_model=DataIngestionResponse, dependencies=[Depends(require_api_key)])
async def trigger_scheduled_ingestion(
    force_full_sync: bool = False,
    scheduler: SchedulerService = Depends(get_scheduler_service),
):
    try:
        result = await scheduler.trigger_manual_ingestion(force_full_sync=force_full_sync)
        return DataIngestionResponse(**result)
    except Exception as exc:
        logger.error(f"Failed to trigger scheduled ingestion: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger scheduled ingestion: {str(exc)}")


@router.post("/scheduler/start", dependencies=[Depends(require_api_key)])
async def start_scheduler(scheduler: SchedulerService = Depends(get_scheduler_service)):
    try:
        scheduler.start()
        scheduler.ensure_task_running()
        return {"message": "Scheduler started successfully"}
    except Exception as exc:
        logger.error(f"Failed to start scheduler: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(exc)}")


@router.post("/scheduler/stop", dependencies=[Depends(require_api_key)])
async def stop_scheduler(scheduler: SchedulerService = Depends(get_scheduler_service)):
    try:
        scheduler.stop()
        return {"message": "Scheduler stopped successfully"}
    except Exception as exc:
        logger.error(f"Failed to stop scheduler: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(exc)}")


