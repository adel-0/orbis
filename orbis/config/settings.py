"""
Configuration management for Orbis using SQLite database.
"""

import logging

from dotenv import load_dotenv

from app.db.models import DataSource as DataSourceModel
from app.db.session import DatabaseManager
from infrastructure.data_processing.data_source_service import DataSourceService
from utils.env import get_env, get_env_list

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with database-backed data source configuration"""

    # API Configuration
    API_TITLE: str = "Orbis API"
    API_VERSION: str = "2.0.0"
    API_HOST: str = get_env("API_HOST", "127.0.0.1")
    API_PORT: int = get_env("API_PORT", 7887)


    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: str = get_env("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = get_env("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_MODEL: str = get_env("AZURE_OPENAI_MODEL", "gpt-5-mini")
    AZURE_OPENAI_API_VERSION: str = get_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = get_env("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")

    # Embedding settings
    EMBEDDING_DEVICE: str = get_env("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BULK_BATCH_SIZE: int = get_env("EMBEDDING_BULK_BATCH_SIZE", 32)

    # Local embedding settings
    LOCAL_EMBEDDING_MODEL: str = get_env("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    LOCAL_RERANK_MODEL: str = get_env("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # Cache directories for models
    SENTENCE_TRANSFORMERS_HOME: str = get_env("SENTENCE_TRANSFORMERS_HOME", "data/models/sentence-transformers")
    HF_HOME: str = get_env("HF_HOME", "data/models/huggingface")

    # Vector database settings
    CHROMA_COLLECTION_NAME: str = get_env("CHROMA_COLLECTION_NAME", "documents")
    CHROMA_DB_PATH: str = get_env("CHROMA_DB_PATH", "data/chroma_db")

    # Data ingestion settings
    USE_INCREMENTAL_SYNC: bool = get_env("USE_INCREMENTAL_SYNC", True)

    # Scheduler settings
    SCHEDULER_ENABLED: bool = get_env("SCHEDULER_ENABLED", False)
    # Daily start time in HH:MM (local time)
    SCHEDULED_INGESTION_TIME: str = get_env("SCHEDULED_INGESTION_TIME", "02:00")
    # Ingestion interval in hours
    SCHEDULER_INTERVAL_HOURS: int = get_env("SCHEDULER_INTERVAL_HOURS", 24)

    # CORS settings
    CORS_ALLOW_ORIGINS: list[str] = get_env_list("CORS_ALLOW_ORIGINS", ["*"])
    CORS_ALLOW_CREDENTIALS: bool = get_env("CORS_ALLOW_CREDENTIALS", False)

    # Wiki summarization settings
    WIKI_MAX_INPUT_TOKENS_PER_CHUNK: int = get_env("WIKI_MAX_INPUT_TOKENS_PER_CHUNK", 15000)
    WIKI_OVERLAP_TOKENS: int = get_env("WIKI_OVERLAP_TOKENS", 500)
    WIKI_CACHE_DURATION_HOURS: int = get_env("WIKI_CACHE_DURATION_HOURS", 2400)
    WIKI_TARGET_OUTPUT_TOKENS_PER_PAGE: int = get_env("WIKI_TARGET_OUTPUT_TOKENS_PER_PAGE", 3000)
    WIKI_MAX_CONTENT_SIZE_MB: int = get_env("WIKI_MAX_CONTENT_SIZE_MB", 10)
    WIKI_CONTENT_SIZE_CHECK_ENABLED: bool = get_env("WIKI_CONTENT_SIZE_CHECK_ENABLED", True)
    WIKI_MAX_OUTPUT_TOKENS: int = get_env("WIKI_MAX_OUTPUT_TOKENS", 20000)
    WIKI_DEBUG_LOGGING_ENABLED: bool = get_env("WIKI_DEBUG_LOGGING_ENABLED", True)
    WIKI_DEBUG_LOG_DIR: str = get_env("WIKI_DEBUG_LOG_DIR", "logs/wiki_summaries")
    WIKI_MODEL_CONTEXT_LIMIT: int = get_env("WIKI_MODEL_CONTEXT_LIMIT", 128000)
    WIKI_SAFE_OUTPUT_LIMIT: int = get_env("WIKI_SAFE_OUTPUT_LIMIT", 16000)



    # Optional encryption key for sensitive data at rest (Fernet key)
    ENCRYPTION_KEY: str = get_env("ENCRYPTION_KEY", "")

    def __init__(self):
        # Initialize database on first use
        self._ensure_database_initialized()

    def _ensure_database_initialized(self):
        """Ensure database is initialized"""
        try:
            DatabaseManager.init_database()
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize database: {e}"
            )




    def _build_auth_kwargs(self, source: DataSourceModel) -> dict:
        """Build authentication kwargs based on auth type"""
        auth_kwargs = {}
        auth_type = source.config.get('auth_type', 'pat')
        if auth_type == "pat":
            auth_kwargs['pat'] = source.config.get('pat')
        elif auth_type == "oauth2":
            auth_kwargs.update({
                'client_id': source.config.get('client_id'),
                'client_secret': source.config.get('client_secret'),
                'tenant_id': source.config.get('tenant_id')
            })
        return auth_kwargs

    def save_data_sources(self, data_sources: list[DataSourceModel]):
        """Save data sources to database"""
        try:
            with DataSourceService() as service:
                # Get existing sources
                existing_sources = {ds.name: ds for ds in service.get_all_data_sources()}

                # Update or create sources
                for source in data_sources:
                    base_kwargs = {
                        'organization': source.config.get('organization'),
                        'project': source.config.get('project'),
                        'auth_type': source.config.get('auth_type', 'pat'),
                        'query_ids': source.config.get('query_ids'),
                        'enabled': source.enabled
                    }

                    if source.name in existing_sources:
                        # Update existing
                        update_kwargs = {**base_kwargs, **self._build_auth_kwargs(source)}
                        service.update_data_source(source.name, **update_kwargs)
                    else:
                        # Create new
                        create_kwargs = {'name': source.name, **base_kwargs, **self._build_auth_kwargs(source)}
                        service.create_data_source(**create_kwargs)

                # Remove sources that are no longer in the list
                source_names = {source.name for source in data_sources}
                for existing_name in existing_sources:
                    if existing_name not in source_names:
                        service.delete_data_source(existing_name)

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to save data sources to database: {e}"
            )
            raise

    def add_data_source(self, data_source: DataSourceModel):
        """Add a new data source"""
        try:
            with DataSourceService() as service:
                create_kwargs = {
                    'name': data_source.name,
                    'organization': data_source.config.get('organization'),
                    'project': data_source.config.get('project'),
                    'auth_type': data_source.config.get('auth_type', 'pat'),
                    'query_ids': data_source.config.get('query_ids'),
                    'enabled': data_source.enabled,
                    **self._build_auth_kwargs(data_source)
                }
                service.create_data_source(**create_kwargs)
        except ValueError as e:
            raise e
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to add data source: {e}"
            )
            raise


# Global settings instance
settings = Settings()
