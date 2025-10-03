"""
Configuration management for OnCall Copilot using SQLite database.
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.db.session import DatabaseManager
from services.data_source_service import DataSourceService, DataSource


class Settings:
    """Application settings with database-backed data source configuration"""
    
    # API Configuration
    API_TITLE: str = "OnCall Copilot API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "7887"))
    
    # Azure OpenAI settings (for summary service only)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_MODEL: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    
    # Embedding settings
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BULK_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BULK_BATCH_SIZE", "32"))  # For bulk /embed operations
    EMBEDDING_MAX_CHUNK_SIZE: int = int(os.getenv("EMBEDDING_MAX_CHUNK_SIZE", "512"))
    
    # Local embedding settings
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L12-v2")

    # Hugging Face settings
    HUGGING_FACE_TOKEN: str = os.getenv("HUGGING_FACE_TOKEN", "")

    # Reranking settings
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "mixedbread-ai/mxbai-rerank-base-v2")
    RERANK_DEVICE: str = os.getenv("RERANK_DEVICE", "auto")
    RERANK_MAX_LENGTH: Optional[int] = int(os.getenv("RERANK_MAX_LENGTH")) if os.getenv("RERANK_MAX_LENGTH") else None
    RERANK_MAX_CHUNK_SIZE: int = int(os.getenv("RERANK_MAX_CHUNK_SIZE", str(int(os.getenv("EMBEDDING_MAX_CHUNK_SIZE", "512")))))
    
    # Vector database settings
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "items")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "data/chroma_db/orbis-search")
    
    # Data ingestion settings
    USE_INCREMENTAL_SYNC: bool = os.getenv("USE_INCREMENTAL_SYNC", "true").lower() == "true"
    
    # Scheduler settings
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
    # Daily start time in HH:MM (local time)
    SCHEDULED_INGESTION_TIME: str = os.getenv("SCHEDULED_INGESTION_TIME", "02:00")
    # Ingestion interval in hours
    SCHEDULED_INGESTION_INTERVAL_HOURS: int = int(
        os.getenv("SCHEDULED_INGESTION_INTERVAL_HOURS", "24")
    )

    # CORS settings
    CORS_ALLOW_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()
    ]
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

    # Azure DevOps client tuning
    ADO_MAX_CONCURRENT_REQUESTS: int = int(os.getenv("ADO_MAX_CONCURRENT_REQUESTS", "50"))
    ADO_BATCH_SIZE: int = int(os.getenv("ADO_BATCH_SIZE", "200"))
    ADO_TIMEOUT_SECONDS: int = int(os.getenv("ADO_TIMEOUT_SECONDS", "300"))

    # Optional encryption key for sensitive data at rest (Fernet key)
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")
    
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
    
    def get_data_sources(self) -> List[DataSource]:
        """Get all enabled data sources from database"""
        try:
            with DataSourceService() as service:
                db_sources = service.get_all_data_sources(enabled_only=True)
                return [DataSource.from_db_model(db_source) for db_source in db_sources]
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to load data sources from database: {e}"
            )
            return []
    
    def load_data_sources(self) -> List[DataSource]:
        """Get all data sources from database (including disabled ones)"""
        try:
            with DataSourceService() as service:
                db_sources = service.get_all_data_sources()
                return [DataSource.from_db_model(db_source) for db_source in db_sources]
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to load data sources from database: {e}"
            )
            return []

    def add_data_source(self, data_source: DataSource):
        """Add a new data source"""
        try:
            with DataSourceService() as service:
                create_kwargs = {
                    'name': data_source.name,
                    'organization': data_source.organization,
                    'project': data_source.project,
                    'auth_type': data_source.auth_type,
                    'query_ids': data_source.query_ids,
                    'fields': data_source.fields,
                    'enabled': data_source.enabled
                }
                if data_source.auth_type == "pat":
                    create_kwargs['pat'] = data_source.pat
                elif data_source.auth_type == "oauth2":
                    create_kwargs['client_id'] = data_source.client_id
                    create_kwargs['client_secret'] = data_source.client_secret
                    create_kwargs['tenant_id'] = data_source.tenant_id
                
                service.create_data_source(**create_kwargs)
        except ValueError as e:
            raise e
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to add data source: {e}"
            )
            raise
    
    def remove_data_source(self, name: str):
        """Remove a data source by name"""
        try:
            with DataSourceService() as service:
                if not service.delete_data_source(name):
                    raise ValueError(f"Data source '{name}' not found")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to remove data source: {e}"
            )
            raise
    
    def update_data_source(self, data_source: DataSource):
        """Update an existing data source"""
        try:
            with DataSourceService() as service:
                update_kwargs = {
                    'organization': data_source.organization,
                    'project': data_source.project,
                    'auth_type': data_source.auth_type,
                    'query_ids': data_source.query_ids,
                    'fields': data_source.fields,
                    'enabled': data_source.enabled
                }
                if data_source.auth_type == "pat":
                    update_kwargs['pat'] = data_source.pat
                elif data_source.auth_type == "oauth2":
                    update_kwargs['client_id'] = data_source.client_id
                    update_kwargs['client_secret'] = data_source.client_secret
                    update_kwargs['tenant_id'] = data_source.tenant_id
                
                service.update_data_source(data_source.name, **update_kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to update data source: {e}"
            )
            raise
    
# Global settings instance
settings = Settings()