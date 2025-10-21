"""
OnCall Copilot API - FastAPI bootstrap with modular routers and app.state services.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from app.db.session import DatabaseManager
from app.core.container import create_container
from app.api.routers import embedding, search, ingestion, scheduler, datasources, health, provider, field_discovery
from orbis_core.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting OnCall Copilot API...")
        DatabaseManager.init_database(default_db_name="orbis_search.db")
        container = create_container()
        app.state.service_container = container
        logger.info(f"Database: {DatabaseManager.get_database_info()}")
        logger.info("API startup completed successfully")
        yield
    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        raise
    finally:
        try:
            container = getattr(app.state, "service_container", None)
            if container:
                container.shutdown_services()
        except Exception as exc:
            logger.error(f"Error during shutdown: {exc}")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="OnCall Copilot API for semantic search of Azure DevOps tickets with SQLite database",
    lifespan=lifespan,
)

# Configure CORS
cors_allow_origins = settings.CORS_ALLOW_ORIGINS
cors_allow_credentials = settings.CORS_ALLOW_CREDENTIALS
# Credentials with wildcard origins is not allowed; force-disable credentials in that case
if cors_allow_credentials and (cors_allow_origins == ["*"] or "*" in cors_allow_origins):
    logger.warning("CORS credentials cannot be used with wildcard origins. Disabling credentials.")
    cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embedding.router)
app.include_router(search.router)
app.include_router(ingestion.router)
app.include_router(scheduler.router)
app.include_router(datasources.router)
app.include_router(field_discovery.router)
app.include_router(provider.router)
app.include_router(health.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False, log_level="info")