import logging
from datetime import datetime
from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.db.session import DatabaseManager
from models.schemas import HealthResponse
from services.work_item_service import WorkItemService
from services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(container: ServiceContainer = Depends(get_container)):
    if not container or not container.is_initialized():
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            model_loaded=False,
            chroma_connected=False,
            azure_openai_configured=False,
            total_tickets=0,
            database_info=DatabaseManager.get_database_info(),
        )

    health_status = container.get_health_status()
    health_status.database_info = DatabaseManager.get_database_info()

    try:
        with WorkItemService() as wi_service:
            health_status.total_tickets = wi_service.get_total_work_item_count()
    except Exception as exc:
        logger.warning(f"Failed to get ticket count from database: {exc}")

    return health_status


