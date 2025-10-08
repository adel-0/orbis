"""
Service for managing work items using the generic content model.
Replaces the old WorkItem-specific service with generic content architecture.
"""

import logging
from typing import Any

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

from app.db.models import Content as ContentModel
from app.db.models import DataSource as DataSourceModel
from app.db.session import get_db_session
from engine.schemas import Ticket
from engine.services.generic_content_service import GenericContentService

from orbis_core.connectors.azure_devops.constants import DEFAULT_DB_BATCH_SIZE

logger = logging.getLogger(__name__)


class WorkItemService:
    """Service for managing work items/tickets using generic content model"""

    def __init__(self, db: Session | None = None):
        self.db = db
        self._should_close_db = False

        if self.db is None:
            self.db = get_db_session()
            self._should_close_db = True

        self.content_service = GenericContentService(self.db)

    @classmethod
    def get_collection_name(cls) -> str:
        """Return the ChromaDB collection name for work items"""
        return "workitems_collection"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_db and self.db:
            self.db.close()

    def create_or_update_work_item(self, external_id: str, data_source_id: int,
                                  work_item_data: dict[str, Any], *, auto_commit: bool = True) -> ContentModel:
        """Create or update a work item using generic content model.

        Args:
            external_id: External ID from Azure DevOps
            data_source_id: ID of the data source this work item belongs to
            work_item_data: Raw dictionary of work item fields
            auto_commit: When False, defer committing to the caller (useful for bulk ops)
        """
        return self.content_service.create_or_update_content(
            external_id=external_id,
            data_source_id=data_source_id,
            content_type="work_item",
            content_data=work_item_data,
            auto_commit=auto_commit
        )

    def bulk_create_or_update_work_items(self, work_items_data: list[dict[str, Any]],
                                        data_source_id: int) -> int:
        """Bulk create or update work items for better performance."""
        updated_count = 0
        batch_size = DEFAULT_DB_BATCH_SIZE

        try:
            for index, item_data in enumerate(work_items_data, start=1):
                external_id = str(item_data.get('Id', ''))
                if not external_id:
                    continue

                self.create_or_update_work_item(
                    external_id,
                    data_source_id,
                    item_data,
                    auto_commit=False,
                )
                updated_count += 1

                # Commit progress periodically to avoid huge transactions
                if index % batch_size == 0:
                    self.db.commit()
                    logger.info(f"Bulk save progress: {index}/{len(work_items_data)} committed")

            # Final commit for remaining records
            self.db.commit()
            logger.info(f"Bulk save completed: {updated_count} work items committed")
            return updated_count
        except Exception as exc:
            # Roll back on any failure during bulk operation
            self.db.rollback()
            logger.error(f"Bulk save failed after {updated_count} items: {exc}")
            raise

    def get_work_item(self, external_id: str, data_source_id: int) -> ContentModel | None:
        """Get a work item by external ID and data source"""
        return self.content_service.get_content_by_external_id(
            external_id=external_id,
            data_source_id=data_source_id,
            content_type="work_item"
        )

    def get_work_items_by_data_source(self, data_source_id: int,
                                     limit: int | None = None) -> list[ContentModel]:
        """Get all work items for a data source"""
        return self.content_service.get_content_by_data_source(
            data_source_id=data_source_id,
            content_type="work_item",
            limit=limit or 50000
        )

    def get_total_work_item_count(self, enabled_sources_only: bool = True) -> int:
        """Get total count of work items, optionally restricting to enabled sources."""
        query = self.db.query(ContentModel).filter(ContentModel.content_type == "work_item")
        if enabled_sources_only:
            query = query.join(DataSourceModel).filter(DataSourceModel.enabled)
        return query.count()

    def get_work_items_by_source_name(self, source_name: str,
                                     limit: int | None = None) -> list[ContentModel]:
        """Get all work items for a data source by name"""
        query = self.db.query(ContentModel).join(DataSourceModel).filter(
            and_(
                DataSourceModel.name == source_name,
                ContentModel.content_type == "work_item"
            )
        ).order_by(desc(ContentModel.updated_at))

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_all_work_items(self, limit: int | None = None,
                          enabled_sources_only: bool = True) -> list[ContentModel]:
        """Get all work items from all data sources"""
        query = self.db.query(ContentModel).join(DataSourceModel).filter(
            ContentModel.content_type == "work_item"
        )

        if enabled_sources_only:
            query = query.filter(DataSourceModel.enabled)

        query = query.order_by(desc(ContentModel.updated_at))

        if limit:
            query = query.limit(limit)

        return query.all()

    def search_work_items(self, search_term: str, limit: int = 50) -> list[ContentModel]:
        """Search work items by title and content"""
        search_pattern = f"%{search_term}%"

        return self.db.query(ContentModel).join(DataSourceModel).filter(
            and_(
                DataSourceModel.enabled,
                ContentModel.content_type == "work_item",
                or_(
                    ContentModel.title.ilike(search_pattern),
                    ContentModel.content.ilike(search_pattern)
                )
            )
        ).limit(limit).all()

    def delete_work_items_by_data_source(self, data_source_id: int) -> int:
        """Delete all work items for a data source"""
        return self.content_service.delete_content_by_data_source(
            data_source_id=data_source_id,
            content_type="work_item"
        )

    def get_work_item_count_by_source(self) -> dict[str, int]:
        """Get count of work items per data source"""
        results = self.db.query(
            DataSourceModel.name,
            self.db.query(ContentModel).filter(
                and_(
                    ContentModel.data_source_id == DataSourceModel.id,
                    ContentModel.content_type == "work_item"
                )
            ).count().label('count')
        ).all()

        return dict(results)

    def convert_to_tickets(self, work_items: list[ContentModel]) -> list[Ticket]:
        """Convert work item content to Ticket objects"""
        tickets = []

        for work_item in work_items:
            try:
                metadata = work_item.content_metadata or {}
                ticket = Ticket(
                    external_id=work_item.external_id,
                    source_name=work_item.data_source.name if work_item.data_source else None,
                    title=work_item.title,
                    description=work_item.content or "",
                    comments=metadata.get("additional_fields", {}).get("comments", []),
                    work_item_type=metadata.get("work_item_type", ""),
                    state=metadata.get("state", ""),
                    tags=metadata.get("tags", []),
                    area_path=metadata.get("area_path", ""),
                    assigned_to=metadata.get("assigned_to", ""),
                    priority=metadata.get("priority", ""),
                    severity=metadata.get("severity", ""),
                    url=work_item.source_reference or ""
                )
                tickets.append(ticket)
            except Exception as e:
                # Log warning but continue processing
                logger.warning(
                    f"Failed to convert work item {work_item.external_id} to ticket: {e}"
                )
                continue

        return tickets

    def get_tickets_from_all_sources(self) -> list[Ticket]:
        """Get all work items as Ticket objects"""
        work_items = self.get_all_work_items()
        return self.convert_to_tickets(work_items)

    def get_tickets_from_source(self, source_name: str) -> list[Ticket]:
        """Get work items from specific source as Ticket objects"""
        work_items = self.get_work_items_by_source_name(source_name)
        return self.convert_to_tickets(work_items)

    # Generic interface methods for configuration-driven ingestion
    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
        """
        Fetch work items using configuration parameters.
        This implements the generic connector interface using the Azure DevOps client.
        """
        try:
            import aiohttp

            from orbis_core.connectors.azure_devops import (
                AzureDevOpsClient,
            )

            # Create Azure DevOps client from config
            if config.get('auth_type') == 'oauth2':
                client = AzureDevOpsClient(
                    organization=config['organization'],
                    project=config['project'],
                    client_id=config['client_id'],
                    client_secret=config['client_secret'],
                    tenant_id=config['tenant_id'],
                    use_oauth2=True
                )
            else:
                client = AzureDevOpsClient(
                    organization=config['organization'],
                    project=config['project'],
                    auth_token=config['pat']
                )

            # Get query IDs from config
            query_ids = config.get('query_ids', [])
            if not query_ids:
                logger.warning("No query_ids provided in config")
                return []

            # Use incremental sync if supported
            last_sync_time = config.get('last_sync_time') if incremental else None

            # Create session with proper settings
            limit = max(10, int(client.max_concurrent_requests) * 2)
            connector = aiohttp.TCPConnector(limit=limit, limit_per_host=client.max_concurrent_requests)
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Try incremental sync first if last sync time available
                all_workitems = []

                if incremental and last_sync_time:
                    try:
                        # Use reporting API for incremental sync
                        workitem_ids = await client.get_workitems_from_reporting_api(session, last_sync_time)
                        if workitem_ids:
                            # Get full details for the work items
                            workitem_details = await client.get_workitem_details_batch(session, workitem_ids)
                            # Process with comments
                            all_workitems = await client.process_workitems_with_comments_parallel(session, workitem_details)
                    except Exception as e:
                        logger.warning(f"Incremental sync failed: {str(e)}")
                        all_workitems = []

                # Fall back to full sync if:
                # 1. Full sync explicitly requested (incremental=False), OR
                # 2. No items from incremental sync AND no last_sync_time (first run)
                if not all_workitems and (not incremental or not last_sync_time):
                    if not last_sync_time:
                        logger.info("No last_sync_time found - performing initial full sync")
                    for query_id in query_ids:
                        try:
                            # Get work item IDs from query
                            workitem_ids = await client.get_workitems_from_query(session, query_id)
                            if workitem_ids:
                                # Get full details for the work items
                                workitem_details = await client.get_workitem_details_batch(session, workitem_ids)
                                # Process with comments
                                batch_workitems = await client.process_workitems_with_comments_parallel(session, workitem_details)
                                all_workitems.extend(batch_workitems)
                        except Exception as e:
                            logger.error(f"Failed to fetch work items for query {query_id}: {str(e)}")
                            continue

            logger.info(f"Fetched {len(all_workitems)} work items from Azure DevOps")
            return all_workitems

        except Exception as e:
            logger.error(f"Failed to fetch work items: {str(e)}")
            raise

    def get_searchable_text(self, work_item: dict) -> str:
        """Convert work item to searchable text"""
        parts = []

        # Add title
        title = work_item.get('Title') or work_item.get('title', '')
        if title:
            parts.append(str(title))

        # Add description
        description = work_item.get('Description') or work_item.get('description', '')
        if description:
            parts.append(str(description))

        # Add comments
        comments = work_item.get('Comments') or work_item.get('comments', [])
        if comments and isinstance(comments, list):
            parts.extend([str(comment) for comment in comments])
        elif comments and isinstance(comments, str):
            parts.append(str(comments))

        return " ".join(parts)

    def get_metadata(self, work_item: dict) -> dict[str, Any]:
        """
        Return metadata that should be stored for filtering/display.

        MANDATORY FIELDS:
        - title: Work item title
        - content: Work item description
        - source_reference: Azure DevOps URL

        OPTIONAL FIELDS:
        - Connector-specific fields for filtering/display
        """

        def extract_user_display_name(user_data):
            if isinstance(user_data, dict):
                return user_data.get('displayName', '')
            elif isinstance(user_data, str):
                return user_data
            else:
                return ''

        # MANDATORY FIELDS - all connectors must provide these
        metadata = {
            'title': work_item.get('Title', '') or work_item.get('title', ''),
            'content': work_item.get('Description', '') or work_item.get('description', ''),
            'source_reference': work_item.get('url', ''),
        }

        # CONNECTOR-SPECIFIC FIELDS
        metadata.update({
            'area_path': work_item.get('AreaPath', ''),
            'state': work_item.get('State', ''),
            'assigned_to': extract_user_display_name(work_item.get('AssignedTo', '')),
            'work_item_type': work_item.get('WorkItemType', ''),
            'priority': work_item.get('Priority', ''),
            'severity': work_item.get('Severity', ''),
            'created_by': extract_user_display_name(work_item.get('CreatedBy', '')),
            'tags': work_item.get('Tags', '').split(';') if work_item.get('Tags') else [],
            'azure_created_date': work_item.get('CreatedDate', ''),
            'azure_changed_date': work_item.get('ChangedDate', ''),
            'azure_resolved_date': work_item.get('ResolvedDate', ''),
        })

        # Include any additional fields that don't fit the standard schema
        standard_fields = {
            'Id', 'Title', 'Description', 'Comments', 'State', 'WorkItemType',
            'CreatedDate', 'ChangedDate', 'ResolvedDate', 'AreaPath',
            'Tags', 'AssignedTo', 'CreatedBy', 'Priority', 'Severity', 'url'
        }

        additional_fields = {k: v for k, v in work_item.items()
                           if k not in standard_fields and v is not None}

        if additional_fields:
            metadata['additional_fields'] = additional_fields

        return metadata

    def get_content_id(self, work_item: dict) -> str:
        """Return unique ID for this content item"""
        work_item_id = work_item.get('Id') or work_item.get('id', '')
        return f"workitem_{work_item_id}"
