"""
Service for managing data sources in the database.
Replaces JSON-based data source configuration.
"""

import os
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import DataSource as DataSourceModel
from app.db.session import get_db_session

# Optional encryption using Fernet
try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_FERNET = True
except Exception:
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    HAS_FERNET = False


class DataSourceService:
    """Service for managing data sources with encryption support"""

    def __init__(self, db: Session | None = None):
        self.db = db
        self._should_close_db = False
        self._fernet: Any | None = None

        if self.db is None:
            self.db = get_db_session()
            self._should_close_db = True

        # Initialize encryption if key is provided and cryptography is available
        key = os.getenv('ENCRYPTION_KEY', '')
        if key and HAS_FERNET:
            try:
                # Accept only a valid Fernet key (base64 urlsafe 32 bytes). Do not treat ciphertext as key.
                self._fernet = Fernet(key.encode())
            except Exception as e:
                # Fail fast: invalid key provided
                raise ValueError("Invalid ENCRYPTION_KEY. Provide a valid Fernet key generated via cryptography.Fernet.generate_key().") from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_db and self.db:
            self.db.close()

    def create_data_source(self, name: str, source_type: str, config: dict[str, Any],
                         enabled: bool = True, context_tags: list[str] = None,
                         priority: int = 1) -> DataSourceModel:
        """Create a generic data source with any source type and configuration"""
        try:
            # Encrypt sensitive fields in config
            encrypted_config = config.copy()
            sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

            for field in sensitive_fields:
                if field in encrypted_config and encrypted_config[field]:
                    encrypted_config[field] = self._encrypt(str(encrypted_config[field]))

            # Create data source model
            data_source = DataSourceModel(
                name=name,
                source_type=source_type,
                config=encrypted_config,
                enabled=enabled,
                context_tags=context_tags,
                priority=priority
            )

            self.db.add(data_source)
            self.db.commit()
            self.db.refresh(data_source)

            # Decrypt for return
            decrypted_config = encrypted_config.copy()
            for field in sensitive_fields:
                if field in decrypted_config and decrypted_config[field]:
                    decrypted_config[field] = self._decrypt(str(decrypted_config[field]))
            data_source.config = decrypted_config

            return data_source

        except IntegrityError as e:
            self.db.rollback()
            raise ValueError(f"Data source with name '{name}' already exists") from e

    def get_data_source(self, name: str) -> DataSourceModel | None:
        """Get a data source by name with decrypted configuration"""
        data_source = self.db.query(DataSourceModel).filter(DataSourceModel.name == name).first()
        if data_source and data_source.config:
            # Decrypt sensitive fields
            decrypted_config = data_source.config.copy()
            sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

            for field in sensitive_fields:
                if field in decrypted_config and decrypted_config[field]:
                    decrypted_config[field] = self._decrypt(str(decrypted_config[field]))

            data_source.config = decrypted_config

        return data_source

    def get_all_data_sources(self, enabled_only: bool = False) -> list[DataSourceModel]:
        """Get all data sources with decrypted configurations"""
        query = self.db.query(DataSourceModel)
        if enabled_only:
            query = query.filter(DataSourceModel.enabled)

        data_sources = query.order_by(DataSourceModel.source_type, DataSourceModel.name).all()

        # Decrypt sensitive fields in configs
        for data_source in data_sources:
            if data_source.config:
                decrypted_config = data_source.config.copy()
                sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

                for field in sensitive_fields:
                    if field in decrypted_config and decrypted_config[field]:
                        decrypted_config[field] = self._decrypt(str(decrypted_config[field]))

                data_source.config = decrypted_config

        return data_sources

    def update_data_source(self, name: str, **kwargs) -> DataSourceModel:
        """Update a generic data source"""
        data_source = self.get_data_source(name)
        if not data_source:
            raise ValueError(f"Data source with name '{name}' not found")

        for key, value in kwargs.items():
            if key == 'config' and value:
                # Handle config updates with encryption
                config = value.copy()
                sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

                for field in sensitive_fields:
                    if field in config and config[field]:
                        config[field] = self._encrypt(str(config[field]))

                setattr(data_source, key, config)
            elif hasattr(data_source, key):
                setattr(data_source, key, value)

        self.db.commit()
        self.db.refresh(data_source)

        # Decrypt config for return
        if data_source.config:
            decrypted_config = data_source.config.copy()
            sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

            for field in sensitive_fields:
                if field in decrypted_config and decrypted_config[field]:
                    decrypted_config[field] = self._decrypt(str(decrypted_config[field]))

            data_source.config = decrypted_config

        return data_source

    def delete_data_source(self, name: str) -> bool:
        """Delete a data source"""
        data_source = self.get_data_source(name)
        if not data_source:
            return False

        self.db.delete(data_source)
        self.db.commit()
        return True

    def enable_data_source(self, name: str) -> DataSourceModel:
        """Enable a data source"""
        return self.update_data_source(name, enabled=True)

    def disable_data_source(self, name: str) -> DataSourceModel:
        """Disable a data source"""
        return self.update_data_source(name, enabled=False)

    def get_enabled_sources(self) -> list[DataSourceModel]:
        """Get all enabled data sources"""
        return self.get_all_data_sources(enabled_only=True)

    # Encryption helpers
    def _encrypt(self, value: str) -> str:
        if not value:
            return value
        if self._fernet is None:
            return value
        try:
            return self._fernet.encrypt(value.encode()).decode()
        except Exception:
            return value

    def _decrypt(self, value: str) -> str:
        if not value:
            return value
        if self._fernet is None:
            return value
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except InvalidToken:
            return value
        except Exception:
            return value

    def get_data_source_by_id(self, source_id: int) -> DataSourceModel | None:
        """Get data source by ID with decrypted configuration"""
        data_source = self.db.query(DataSourceModel).filter(DataSourceModel.id == source_id).first()
        if data_source and data_source.config:
            # Decrypt sensitive fields
            decrypted_config = data_source.config.copy()
            sensitive_fields = ['pat', 'client_secret', 'api_token', 'password', 'secret']

            for field in sensitive_fields:
                if field in decrypted_config and decrypted_config[field]:
                    decrypted_config[field] = self._decrypt(str(decrypted_config[field]))

            data_source.config = decrypted_config

        return data_source

    def get_last_sync_time(self, source_name: str) -> str | None:
        """Get the last successful sync time for a data source"""
        from app.db.models import IngestionLog

        # Get the data source
        data_source = self.get_data_source(source_name)
        if not data_source:
            return None

        # Get the most recent successful ingestion
        last_ingestion = self.db.query(IngestionLog).filter(
            IngestionLog.data_source_id == data_source.id,
            IngestionLog.status == "completed"
        ).order_by(IngestionLog.completed_at.desc()).first()

        if last_ingestion and last_ingestion.completed_at:
            return last_ingestion.completed_at.isoformat()

        return None

