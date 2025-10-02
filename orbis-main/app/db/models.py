"""
SQLAlchemy models for OnCall Copilot database.
"""


from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# Generic models - no hardcoded enums

Base = declarative_base()


class DataSource(Base):
    """Generic data source configuration model - supports ANY data source type"""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    source_type = Column(String(50), nullable=False)  # From registry (e.g., 'azdo_workitems', 'project_wiki', 'jira_issues')
    config = Column(JSON, nullable=False)  # All source-specific configuration as JSON
    enabled = Column(Boolean, default=True, nullable=False)

    # Optional metadata for search optimization
    context_tags = Column(JSON, nullable=True)  # ["regional", "customer_specific", "core_product", etc.]
    priority = Column(Integer, default=1)  # Search priority (1=highest, 5=lowest)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Generic content relationship
    content_items = relationship("Content", back_populates="data_source", cascade="all, delete-orphan")
    ingestion_logs = relationship("IngestionLog", back_populates="data_source", cascade="all, delete-orphan")




class Content(Base):
    """Generic content model - supports ANY content type from ANY data source"""
    __tablename__ = "content"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(200), nullable=False, index=True)  # ID from external source
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)

    # Generic content fields
    content_type = Column(String(50), nullable=False, index=True)  # "work_item", "wiki_page", "document", "issue", etc.
    title = Column(String(1000), nullable=False)
    content = Column(Text, nullable=True)  # Main content field (description, body, etc.)

    # All type-specific metadata stored as JSON
    content_metadata = Column(JSON, nullable=False, default=dict)  # All type-specific fields

    # Optional source system timestamps and reference (only if from external system)
    source_created_date = Column(DateTime(timezone=True), nullable=True)  # When created in source system
    source_updated_date = Column(DateTime(timezone=True), nullable=True)  # When last updated in source system
    source_reference = Column(Text, nullable=True)  # URL, path, or other reference to original

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    data_source = relationship("DataSource", back_populates="content_items")


    def __repr__(self):
        return f"<Content(id={self.id}, type={self.content_type}, external_id={self.external_id}, title='{self.title[:50]}...')>"


class IngestionLog(Base):
    """Generic ingestion log model for tracking data ingestion runs from ANY source type"""
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)

    sync_type = Column(String(20), nullable=False)  # "full" or "incremental"
    status = Column(String(20), nullable=False)  # "running", "completed", "failed"
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time_seconds = Column(Integer, nullable=True)

    # Generic counters for any content type
    total_fetched = Column(Integer, default=0)  # Items fetched from source
    total_saved = Column(Integer, default=0)   # Items successfully saved
    error_message = Column(Text, nullable=True)

    data_source = relationship("DataSource", back_populates="ingestion_logs")

    def __repr__(self):
        return f"<IngestionLog(id={self.id}, data_source_id={self.data_source_id}, status='{self.status}')>"


class ContentEmbedding(Base):
    """Tracks which content has been embedded and detects changes"""
    __tablename__ = "content_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(Integer, ForeignKey("content.id"), nullable=False, unique=True, index=True)
    content_hash = Column(String(16), nullable=False, index=True)  # xxHash of content
    vector_id = Column(String(200), nullable=False)  # ID in vector database
    embedding_model = Column(String(100), nullable=False)  # Track which model was used
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    content = relationship("Content")

    def __repr__(self):
        return f"<ContentEmbedding(id={self.id}, content_id={self.content_id}, model='{self.embedding_model}')>"


class WikiSummaryCache(Base):
    """Persistent cache for wiki summaries with automatic refresh based on age"""
    __tablename__ = "wiki_summary_cache"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(200), unique=True, nullable=False, index=True)  # "wiki_name_project_code"
    wiki_name = Column(String(100), nullable=False, index=True)
    project_code = Column(String(10), nullable=True, index=True)  # NULL for general wikis

    # Cached summary data (JSON serialized WikiSummary)
    summary_data = Column(JSON, nullable=False)

    # Cache management
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_refreshed_at = Column(DateTime(timezone=True), server_default=func.now())
    refresh_count = Column(Integer, default=1, nullable=False)

    # Summary metadata for quick access
    summary_confidence = Column(String(10), nullable=True)  # Store as string for flexibility
    tokens_used = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<WikiSummaryCache(cache_key='{self.cache_key}', last_refreshed='{self.last_refreshed_at}')>"




