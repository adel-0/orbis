"""
SQLAlchemy models for OnCall Copilot database.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class DataSource(Base):
    """Data source configuration model"""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    organization = Column(String(200), nullable=False)
    project = Column(String(200), nullable=False)
    
    # Authentication - either PAT or OAuth2
    auth_type = Column(String(20), nullable=False, default="pat")  # "pat" or "oauth2"
    pat = Column(Text, nullable=True)  # Only for PAT auth
    client_id = Column(String(200), nullable=True)  # Only for OAuth2 auth
    client_secret = Column(Text, nullable=True)  # Only for OAuth2 auth
    tenant_id = Column(String(200), nullable=True)  # Only for OAuth2 auth
    
    query_ids = Column(JSON, nullable=False)
    fields = Column(JSON, nullable=True)  # Azure DevOps fields to fetch
    enabled = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    work_items = relationship("WorkItem", back_populates="data_source", cascade="all, delete-orphan")
    ingestion_logs = relationship("IngestionLog", back_populates="data_source", cascade="all, delete-orphan")


class WorkItem(Base):
    """Work item model"""
    __tablename__ = "work_items"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(50), nullable=False, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)

    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    comments = Column(JSON, nullable=True)
    work_item_type = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    priority = Column(String(50), nullable=True)
    severity = Column(String(50), nullable=True)
    assigned_to = Column(String(200), nullable=True)
    created_by = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)
    area_path = Column(String(500), nullable=True)
    iteration_path = Column(String(500), nullable=True)

    azure_created_date = Column(DateTime(timezone=True), nullable=True)
    azure_changed_date = Column(DateTime(timezone=True), nullable=True)
    azure_resolved_date = Column(DateTime(timezone=True), nullable=True)
    azure_url = Column(Text, nullable=True)

    # NEW: Store all additional/dynamic fields that don't fit the standard schema
    additional_fields = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    data_source = relationship("DataSource", back_populates="work_items")

    def __repr__(self):
        return f"<WorkItem(id={self.id}, external_id={self.external_id}, title='{self.title[:50]}...')>"


class IngestionLog(Base):
    """Ingestion log model for tracking data ingestion runs"""
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)

    sync_type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time_seconds = Column(Integer, nullable=True)

    fetched_workitems = Column(Integer, default=0)
    total_workitems = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    data_source = relationship("DataSource", back_populates="ingestion_logs")

    def __repr__(self):
        return f"<IngestionLog(id={self.id}, data_source_id={self.data_source_id}, status='{self.status}')>"


