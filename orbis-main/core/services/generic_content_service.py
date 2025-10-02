"""
Generic Content Service - Handles all content types through unified interface.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, joinedload

from app.db.models import Content as ContentModel
from app.db.models import DataSource
from app.db.session import get_db_session
from core.config.data_sources import DataSourceConfig, DataSourceConfigRegistry
from core.schemas import BaseContent, Ticket, WikiPageContent

logger = logging.getLogger(__name__)


class GenericContentService:
    """Service for managing all content types through unified interface"""

    def __init__(self, db: Session | None = None):
        self.db = db
        self._should_close_db = False
        self.config_registry = DataSourceConfigRegistry()

        if self.db is None:
            self.db = get_db_session()
            self._should_close_db = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_db and self.db:
            self.db.close()

    def _get_data_source_config(self, data_source_id: int) -> DataSourceConfig | None:
        """Get configuration for a data source by ID"""
        data_source = self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            return None

        try:
            return self.config_registry.get_source_config(data_source.source_type)
        except ValueError:
            # Unknown source type - will use generic handling
            return None

    def create_or_update_content(self, external_id: str, data_source_id: int,
                               content_type: str, content_data: dict[str, Any],
                               *, auto_commit: bool = True) -> ContentModel:
        """Create or update content of any type.

        Args:
            external_id: External ID from source system
            data_source_id: ID of the data source this content belongs to
            content_type: Type of content ("work_item", "wiki_page", "document", etc.)
            content_data: Raw dictionary of content fields
            auto_commit: When False, defer committing to the caller (useful for bulk ops)
        """
        try:
            # Check if content already exists
            existing_content = self.db.query(ContentModel).filter(
                and_(
                    ContentModel.external_id == str(external_id),
                    ContentModel.data_source_id == data_source_id,
                    ContentModel.content_type == content_type
                )
            ).first()

            if existing_content:
                # Update existing content
                content = self._update_content_from_data(existing_content, content_data, content_type)
                logger.debug(f"Updated {content_type} {external_id}")
            else:
                # Create new content
                content = self._create_content_from_data(external_id, data_source_id, content_type, content_data)
                self.db.add(content)
                logger.debug(f"Created {content_type} {external_id}")

            if auto_commit:
                self.db.commit()
                self.db.refresh(content)

            return content

        except Exception as e:
            if auto_commit:
                self.db.rollback()
            logger.error(f"Error creating/updating {content_type} {external_id}: {e}")
            raise

    def _create_content_from_data(self, external_id: str, data_source_id: int,
                                content_type: str, data: dict[str, Any]) -> ContentModel:
        """Create ContentModel from raw data using configuration-driven mapping"""

        config = self._get_data_source_config(data_source_id)
        if config and config.field_mappings:
            return self._create_configured_content(external_id, data_source_id, content_type, data, config)
        else:
            # Generic content creation for unknown types or unconfigured sources
            return self._create_generic_content(external_id, data_source_id, content_type, data)

    def _create_configured_content(self, external_id: str, data_source_id: int, content_type: str,
                                 data: dict[str, Any], config: DataSourceConfig) -> ContentModel:
        """Create ContentModel using configuration-driven field mappings"""

        # Map standard fields using configuration
        field_mappings = config.field_mappings
        title = data.get(field_mappings.get("title", "title"), "")
        content = data.get(field_mappings.get("content", "content"), "")

        # Build metadata from configured fields
        metadata = {}
        for field in config.metadata_fields or []:
            if field in data:
                metadata[field] = data[field]

        return ContentModel(
            external_id=str(external_id),
            data_source_id=data_source_id,
            content_type=content_type,
            title=title,
            content=content,
            content_metadata=metadata,
            source_created_date=data.get(field_mappings.get("created_date")),
            source_updated_date=data.get(field_mappings.get("updated_date")),
            source_reference=data.get(field_mappings.get("reference"))
        )



    def _create_generic_content(self, external_id: str, data_source_id: int,
                              content_type: str, data: dict[str, Any]) -> ContentModel:
        """Create ContentModel for unknown content types"""
        # Extract common fields, put everything else in metadata
        title = data.get("title") or data.get("name") or data.get("subject") or str(external_id)
        content = data.get("content") or data.get("body") or data.get("description") or ""

        # Remove common fields from metadata
        metadata = {k: v for k, v in data.items()
                   if k not in ["title", "content", "created_date", "updated_date", "url"]}

        return ContentModel(
            external_id=str(external_id),
            data_source_id=data_source_id,
            content_type=content_type,
            title=title,
            content=content,
            content_metadata=metadata,
            source_created_date=data.get("created_date"),
            source_updated_date=data.get("updated_date"),
            source_reference=data.get("url") or data.get("path") or data.get("reference")
        )

    def _update_content_from_data(self, content: ContentModel, data: dict[str, Any], content_type: str) -> ContentModel:
        """Update existing ContentModel with new data using configuration-driven mapping"""

        config = self._get_data_source_config(content.data_source_id)
        if config and config.field_mappings:
            return self._update_configured_content(content, data, config)
        else:
            return self._update_generic_content(content, data)

    def _update_configured_content(self, content: ContentModel, data: dict[str, Any], config: DataSourceConfig) -> ContentModel:
        """Update ContentModel using configuration-driven field mappings"""

        field_mappings = config.field_mappings

        # Update standard fields
        if field_mappings.get("title") and field_mappings["title"] in data:
            content.title = data[field_mappings["title"]]

        if field_mappings.get("content") and field_mappings["content"] in data:
            content.content = data[field_mappings["content"]]

        if field_mappings.get("updated_date") and field_mappings["updated_date"] in data:
            content.source_updated_date = data[field_mappings["updated_date"]]

        if field_mappings.get("reference") and field_mappings["reference"] in data:
            content.source_reference = data[field_mappings["reference"]]

        # Update metadata
        metadata = content.content_metadata or {}
        for field in config.metadata_fields or []:
            if field in data:
                metadata[field] = data[field]

        content.content_metadata = metadata
        content.updated_at = datetime.utcnow()

        return content



    def _update_generic_content(self, content: ContentModel, data: dict[str, Any]) -> ContentModel:
        """Update generic content"""
        content.title = data.get("title") or data.get("name") or data.get("subject") or content.title
        content.content = data.get("content") or data.get("body") or data.get("description") or content.content
        content.source_updated_date = data.get("updated_date", content.source_updated_date)
        content.source_reference = data.get("url") or data.get("path") or data.get("reference") or content.source_reference

        # Update metadata with all fields
        metadata = content.content_metadata or {}
        new_metadata = {k: v for k, v in data.items()
                       if k not in ["title", "content", "created_date", "updated_date", "url"]}
        metadata.update(new_metadata)
        content.content_metadata = metadata
        content.updated_at = datetime.utcnow()

        return content

    def get_content_by_external_id(self, external_id: str, data_source_id: int,
                                 content_type: str) -> ContentModel | None:
        """Get content by external ID and type"""
        return self.db.query(ContentModel).filter(
            and_(
                ContentModel.external_id == str(external_id),
                ContentModel.data_source_id == data_source_id,
                ContentModel.content_type == content_type
            )
        ).first()

    def get_content_by_data_source(self, data_source_id: int,
                                 content_type: str | None = None,
                                 limit: int = 1000) -> list[ContentModel]:
        """Get all content for a data source, optionally filtered by type"""
        query = self.db.query(ContentModel).filter(ContentModel.data_source_id == data_source_id)

        if content_type:
            query = query.filter(ContentModel.content_type == content_type)

        return query.order_by(desc(ContentModel.updated_at)).limit(limit).all()

    def delete_content_by_data_source(self, data_source_id: int, content_type: str | None = None):
        """Delete all content for a data source, optionally filtered by type"""
        query = self.db.query(ContentModel).filter(ContentModel.data_source_id == data_source_id)

        if content_type:
            query = query.filter(ContentModel.content_type == content_type)

        deleted_count = query.count()
        query.delete()
        self.db.commit()

        logger.info(f"Deleted {deleted_count} {content_type or 'content'} records for data source {data_source_id}")
        return deleted_count

    def get_all_content_for_embedding(self, data_source_ids: list[int] | None = None) -> list[BaseContent]:
        """Get all content ready for embedding, converted to BaseContent format"""
        query = self.db.query(ContentModel).options(joinedload(ContentModel.data_source))

        if data_source_ids:
            query = query.filter(ContentModel.data_source_id.in_(data_source_ids))

        content_models = query.all()

        # Convert to BaseContent format for embedding service
        base_contents = []
        for content in content_models:
            config = self._get_data_source_config(content.data_source_id)
            if config and config.schema_class:
                base_content = self._convert_using_config(content, config)
            else:
                base_content = self._convert_to_base_content(content)

            base_contents.append(base_content)

        return base_contents

    def _convert_using_config(self, content: ContentModel, config: DataSourceConfig) -> BaseContent:
        """Convert ContentModel to appropriate schema class using configuration"""

        if config.schema_class == "Ticket":
            return self._convert_to_ticket(content)
        elif config.schema_class == "WikiPageContent":
            return self._convert_to_wiki_page_content(content)
        else:
            return self._convert_to_base_content(content)

    def _convert_to_ticket(self, content: ContentModel) -> Ticket:
        """Convert ContentModel to Ticket schema"""
        metadata = content.content_metadata or {}
        ticket = Ticket(
            id=content.external_id,
            source_name=content.data_source.name if content.data_source else "",
            title=content.title,
            description=content.content or "",
            comments=metadata.get("additional_fields", {}).get("comments", []),
            area_path=metadata.get("area_path", ""),
            additional_fields=metadata.get("additional_fields", {}),
            source_type=content.data_source.source_type if content.data_source else ""
        )

        return ticket

    def _convert_to_wiki_page_content(self, content: ContentModel) -> WikiPageContent:
        """Convert ContentModel to WikiPageContent schema"""
        metadata = content.content_metadata or {}
        wiki_page = WikiPageContent(
            id=content.external_id,
            source_name=content.data_source.name if content.data_source else "",
            title=content.title,
            content=content.content or "",
            path=metadata.get("path", content.external_id),
            author=metadata.get("author", ""),
            source_type=content.data_source.source_type if content.data_source else ""
        )

        return wiki_page

    def _convert_to_base_content(self, content: ContentModel) -> BaseContent:
        """Convert ContentModel to BaseContent schema for unknown types"""
        base_content = BaseContent(
            id=content.external_id,
            source_name=content.data_source.name if content.data_source else "",
            title=content.title,
            content_type=content.content_type,
            source_type=content.data_source.source_type if content.data_source else ""
        )

        return base_content
