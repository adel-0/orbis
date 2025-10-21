"""
Service for managing data sources in the database.
Replaces JSON-based data source configuration.
"""

from typing import List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.models import DataSource as DataSourceModel
from app.db.session import get_db_session
import os

# Optional encryption using Fernet
try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_FERNET = True
except Exception:
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    HAS_FERNET = False


class DataSourceService:
    """Service for managing Azure DevOps data sources"""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self._should_close_db = False
        self._fernet: Optional[Any] = None
        
        if self.db is None:
            self.db = get_db_session()
            self._should_close_db = True

        # Initialize encryption if key is provided and cryptography is available
        key = os.getenv('ENCRYPTION_KEY', '')
        if key and HAS_FERNET:
            try:
                # Accept only a valid Fernet key (base64 urlsafe 32 bytes). Do not treat ciphertext as key.
                self._fernet = Fernet(key.encode())
            except Exception:
                # Fail fast: invalid key provided
                raise ValueError("Invalid ENCRYPTION_KEY. Provide a valid Fernet key generated via cryptography.Fernet.generate_key().")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_db and self.db:
            self.db.close()
    
    def create_data_source(self, name: str, organization: str, project: str,
                          auth_type: str, query_ids: List[str], enabled: bool = True,
                          pat: str = None, client_id: str = None,
                          client_secret: str = None, tenant_id: str = None,
                          fields: List[str] = None) -> DataSourceModel:
        """Create a new data source with either PAT or OAuth2 authentication"""
        try:
            # Validate authentication parameters
            if auth_type == "pat":
                if not pat:
                    raise ValueError("PAT is required for PAT authentication")
                pat_to_store = self._encrypt(pat)
                data_source = DataSourceModel(
                    name=name,
                    organization=organization,
                    project=project,
                    auth_type=auth_type,
                    pat=pat_to_store,
                    query_ids=query_ids,
                    fields=fields,
                    enabled=enabled
                )
            elif auth_type == "oauth2":
                if not all([client_id, client_secret, tenant_id]):
                    raise ValueError("client_id, client_secret, and tenant_id are required for OAuth2 authentication")
                client_secret_to_store = self._encrypt(client_secret)
                data_source = DataSourceModel(
                    name=name,
                    organization=organization,
                    project=project,
                    auth_type=auth_type,
                    client_id=client_id,
                    client_secret=client_secret_to_store,
                    tenant_id=tenant_id,
                    query_ids=query_ids,
                    fields=fields,
                    enabled=enabled
                )
            else:
                raise ValueError(f"Invalid auth_type: {auth_type}. Must be 'pat' or 'oauth2'")
            
            self.db.add(data_source)
            self.db.commit()
            self.db.refresh(data_source)
            
            return data_source
        
        except IntegrityError:
            self.db.rollback()
            raise ValueError(f"Data source with name '{name}' already exists")
    
    def get_data_source(self, name: str) -> Optional[DataSourceModel]:
        """Get a data source by name"""
        ds = self.db.query(DataSourceModel).filter(DataSourceModel.name == name).first()
        if ds:
            self._decrypt_credentials(ds)
        return ds
    
    def get_all_data_sources(self, enabled_only: bool = False) -> List[DataSourceModel]:
        """Get all data sources"""
        query = self.db.query(DataSourceModel)
        if enabled_only:
            query = query.filter(DataSourceModel.enabled)
        result = query.order_by(DataSourceModel.name).all()
        for ds in result:
            self._decrypt_credentials(ds)
        return result
    
    def update_data_source(self, name: str, **kwargs) -> DataSourceModel:
        """Update a data source"""
        data_source = self.get_data_source(name)
        if not data_source:
            raise ValueError(f"Data source with name '{name}' not found")
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(data_source, key):
                if key == 'pat' and value is not None:
                    setattr(data_source, key, self._encrypt(value))
                elif key == 'client_secret' and value is not None:
                    setattr(data_source, key, self._encrypt(value))
                else:
                    setattr(data_source, key, value)
        
        try:
            self.db.commit()
            self.db.refresh(data_source)
            return data_source
        except IntegrityError:
            self.db.rollback()
            raise ValueError(f"Failed to update data source '{name}': integrity constraint violation")
    
    def delete_data_source(self, name: str) -> bool:
        """Delete a data source and all its work items"""
        data_source = self.get_data_source(name)
        if not data_source:
            return False
        
        self.db.delete(data_source)
        self.db.commit()
        return True

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
    
    def _decrypt_credentials(self, data_source: DataSourceModel) -> None:
        """Decrypt credentials in place"""
        if data_source.pat:
            data_source.pat = self._decrypt(data_source.pat)
        if data_source.client_secret:
            data_source.client_secret = self._decrypt(data_source.client_secret)


class DataSource:
    """Data source configuration class"""
    
    def __init__(self, name: str, organization: str, project: str, auth_type: str,
                 query_ids: List[str], enabled: bool = True, pat: str = None,
                 client_id: str = None, client_secret: str = None, tenant_id: str = None,
                 fields: List[str] = None):
        self.name = name
        self.organization = organization
        self.project = project
        self.auth_type = auth_type
        self.pat = pat
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.query_ids = query_ids if isinstance(query_ids, list) else [query_ids]
        self.fields = fields
        self.enabled = enabled
    @classmethod
    def from_db_model(cls, db_model: DataSourceModel) -> 'DataSource':
        """Convert database model to DataSource object"""
        return cls(
            name=db_model.name,
            organization=db_model.organization,
            project=db_model.project,
            auth_type=db_model.auth_type,
            pat=db_model.pat,
            client_id=db_model.client_id,
            client_secret=db_model.client_secret,
            tenant_id=db_model.tenant_id,
            query_ids=db_model.query_ids,
            fields=db_model.fields,
            enabled=db_model.enabled
        )