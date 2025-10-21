"""
Generic Content Service - Handles all content types through unified interface.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, joinedload

from app.db.models import Content as ContentModel
from app.db.session import get_db_session
from engine.schemas import BaseContent, Ticket, WikiPageContent

logger = logging.getLogger(__name__)


class GenericContentService:
    """Service for managing all content types through unified interface"""

    def __init__(self, db: Session | None = None):
        self.db = db
        self._should_close_db = False

        if self.db is None:
            self.db = get_db_session()
            self._should_close_db = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_db and self.db:
            self.db.close()

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
        """
        Create ContentModel from data.
        Expects data to already have proper field names (title, content, etc.)
        as connectors handle their own metadata extraction.
        """
        # Extract standard fields
        title = data.get("title") or data.get("name") or data.get("subject") or str(external_id)
        content = data.get("content") or data.get("body") or data.get("description") or ""

        # All other fields go into metadata
        metadata = {k: v for k, v in data.items()
                   if k not in ["title", "content", "created_date", "updated_date", "url", "reference", "source_reference"]}

        return ContentModel(
            external_id=str(external_id),
            data_source_id=data_source_id,
            content_type=content_type,
            title=title,
            content=content,
            content_metadata=metadata,
            source_created_date=data.get("created_date"),
            source_updated_date=data.get("updated_date"),
            source_reference=data.get("source_reference") or data.get("url") or data.get("path") or data.get("reference")
        )

    def _update_content_from_data(self, content: ContentModel, data: dict[str, Any], content_type: str) -> ContentModel:
        """
        Update existing ContentModel with new data.
        Expects data to already have proper field names as connectors handle extraction.
        """
        # Update standard fields
        content.title = data.get("title") or data.get("name") or data.get("subject") or content.title
        content.content = data.get("content") or data.get("body") or data.get("description") or content.content
        content.source_updated_date = data.get("updated_date", content.source_updated_date)
        content.source_reference = (data.get("source_reference") or data.get("url") or
                                   data.get("path") or data.get("reference") or content.source_reference)

        # Update metadata with all fields (excluding standard fields)
        metadata = content.content_metadata or {}
        new_metadata = {k: v for k, v in data.items()
                       if k not in ["title", "content", "created_date", "updated_date", "url", "reference", "source_reference"]}
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

        # Convert to BaseContent format based on content_type
        base_contents = []
        for content in content_models:
            # Determine conversion based on content_type
            if content.content_type == "work_item":
                base_content = self._convert_to_ticket(content)
            elif content.content_type in ["wiki_page", "wiki"]:
                base_content = self._convert_to_wiki_page_content(content)
            else:
                base_content = self._convert_to_base_content(content)

            base_contents.append(base_content)

        return base_contents

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
