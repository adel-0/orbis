"""
Orbis API - FastAPI bootstrap with simple dependency management.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routers import (
    embedding,
    health,
    ingestion,
    process,
    scheduler,
)
from app.db.session import DatabaseManager, get_db_session
from config.settings import settings
from engine.services.config_loader import ConfigLoader
from utils.app_setup import register_routers, setup_cors
from orbis_core.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting Orbis API...")
        DatabaseManager.init_database()
        logger.info(f"Database: {DatabaseManager.get_database_info()}")

        # Load data source configurations
        config_loader = ConfigLoader()
        instances_loaded = config_loader.load_all_instances()
        logger.info(f"Loaded {instances_loaded} data source instances")

        logger.info("API startup completed successfully - agentic RAG system ready")
        yield
    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        raise
    finally:
        logger.info("Shutting down API - no complex service cleanup needed")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Orbis API for semantic search of Azure DevOps tickets with SQLite database",
    lifespan=lifespan,
)

# Configure CORS and routers
setup_cors(app, settings.CORS_ALLOW_ORIGINS, settings.CORS_ALLOW_CREDENTIALS)

routers = [
    # System endpoints
    health,

    # Core functionality
    process,

    # Data management (logical order: ingest -> embed -> schedule)
    ingestion, embedding, scheduler,
]
register_routers(app, routers)

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False, log_level="info")
