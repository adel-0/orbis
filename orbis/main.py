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
from app.db.session import DatabaseManager
from config.settings import settings
from core.services.config_loader import ConfigLoader
from utils.app_setup import register_routers, setup_cors
from orbis_core.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Starting Orbis API...")
        DatabaseManager.init_database()
        logger.info(f"✅ Database: {DatabaseManager.get_database_info()}")

        # Load data source configurations
        config_loader = ConfigLoader()
        instances_loaded = config_loader.load_all_instances()
        logger.info(f"✅ Loaded {instances_loaded} data source instances")

        # Auto-ingest wiki data and pre-compute summaries for fast contextual analysis
        try:
            from core.agents.wiki_summarization import WikiSummarizationService
            from core.services.generic_data_ingestion import GenericDataIngestionService

            # First, ensure wiki data is ingested (without embedding - that happens via /embed endpoint)
            logger.info("🔄 Auto-ingesting wiki data...")
            ingestion_service = GenericDataIngestionService()

            # Get only wiki data sources
            from infrastructure.data_processing.data_source_service import (
                DataSourceService,
            )
            with DataSourceService() as ds_service:
                wiki_sources = [ds.name for ds in ds_service.get_enabled_sources() if ds.source_type == "azdo_wiki"]

            ingestion_result = await ingestion_service.ingest_all_sources(
                source_names=wiki_sources,  # Only ingest wiki sources
                force_full_sync=False,
                skip_embedding=True  # Embedding happens via /embed endpoint, not during startup
            )

            if ingestion_result.get("total_processed", 0) > 0:
                logger.info(f"✅ Wiki ingestion completed: {ingestion_result.get('total_processed', 0)} items processed")
            else:
                logger.info("ℹ️ No new wiki data to ingest")

            # Then pre-compute summaries (no longer needs vector/embedding services)
            wiki_service = WikiSummarizationService()
            precomputation_results = await wiki_service.precompute_all_project_summaries()

            total_wikis = precomputation_results["processed"] + precomputation_results["cached"] + precomputation_results["failed"] + precomputation_results["skipped_no_content"]
            if total_wikis > 0:
                logger.info(f"✅ Wiki summaries ready: {precomputation_results['processed']} processed, {precomputation_results['cached']} cached, {precomputation_results['failed']} failed")
            else:
                logger.info("✅ Wiki summary pre-computation completed (no wikis configured)")
        except Exception as e:
            logger.warning(f"⚠️ Wiki summary pre-computation failed: {e}")
            logger.debug("Wiki summarization will work on-demand (slower first requests)")

        logger.info("🎯 API startup completed successfully - agentic RAG system ready")
        yield
    except Exception as exc:
        logger.error(f"❌ Startup failed: {exc}")
        raise
    finally:
        logger.info("🛑 Shutting down API - no complex service cleanup needed")


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
