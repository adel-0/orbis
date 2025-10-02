"""
Service for managing work items in the database.
Replaces JSON-based work item storage.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, and_, or_
import logging

from app.db.models import WorkItem as WorkItemModel, DataSource as DataSourceModel
from app.db.session import get_db_session
from models.schemas import Ticket


logger = logging.getLogger(__name__)


class WorkItemService:
    """Service for managing work items/tickets"""
    
    def __init__(self, db: Optional[Session] = None):
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
    
    def create_or_update_work_item(self, external_id: str, data_source_id: int, 
                                  work_item_data: Dict[str, Any], *, auto_commit: bool = True) -> WorkItemModel:
        """Create or update a work item.

        Args:
            external_id: External ID from Azure DevOps
            data_source_id: ID of the data source this work item belongs to
            work_item_data: Raw dictionary of work item fields
            auto_commit: When False, defer committing to the caller (useful for bulk ops)
        """
        
        # Check if work item already exists
        existing = self.db.query(WorkItemModel).filter(
            and_(
                WorkItemModel.external_id == external_id,
                WorkItemModel.data_source_id == data_source_id
            )
        ).first()
        
        # Parse dates from Azure DevOps format
        def parse_azure_date(date_str):
            if not date_str:
                return None
            try:
                # Azure DevOps typically returns ISO format dates
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return None
        
        # Standardized internal field names expected from AzureDevOpsClient:
        # Title, Description, Comments, State, WorkItemType, CreatedDate, ChangedDate,
        # ResolvedDate (optional), AreaPath, IterationPath (optional), Tags (semicolon string)
        
        # Define standard fields that have dedicated columns
        standard_fields = {
            'Id', 'Title', 'Description', 'Comments', 'State', 'WorkItemType', 
            'CreatedDate', 'ChangedDate', 'ResolvedDate', 'AreaPath', 'IterationPath',
            'Tags', 'AssignedTo', 'CreatedBy', 'Priority', 'Severity', 'url'
        }
        
        # Helper function to extract display name from user objects
        def extract_user_display_name(user_data):
            if isinstance(user_data, dict):
                return user_data.get('displayName', '')
            elif isinstance(user_data, str):
                return user_data
            else:
                return ''

        work_item_fields = {
            'external_id': external_id,
            'data_source_id': data_source_id,
            'title': work_item_data.get('Title', ''),
            'description': work_item_data.get('Description', ''),
            'comments': work_item_data.get('Comments', []) if work_item_data.get('Comments') else [],
            'work_item_type': work_item_data.get('WorkItemType', ''),
            'state': work_item_data.get('State', ''),
            'priority': work_item_data.get('Priority', ''),
            'severity': work_item_data.get('Severity', ''),
            'assigned_to': extract_user_display_name(work_item_data.get('AssignedTo', '')),
            'created_by': extract_user_display_name(work_item_data.get('CreatedBy', '')),
            'tags': work_item_data.get('Tags', '').split(';') if work_item_data.get('Tags') else [],
            'area_path': work_item_data.get('AreaPath', ''),
            'iteration_path': work_item_data.get('IterationPath', ''),
            'azure_created_date': parse_azure_date(work_item_data.get('CreatedDate')),
            'azure_changed_date': parse_azure_date(work_item_data.get('ChangedDate')),
            'azure_resolved_date': parse_azure_date(work_item_data.get('ResolvedDate')),
            'azure_url': work_item_data.get('url', ''),
            
            # NEW: Store all additional fields that don't fit the standard schema
            'additional_fields': {k: v for k, v in work_item_data.items() 
                                if k not in standard_fields and v is not None}
        }
        
        if existing:
            # Update existing work item
            for key, value in work_item_fields.items():
                if key != 'external_id' and key != 'data_source_id':  # Don't update these
                    setattr(existing, key, value)
            
            if auto_commit:
                self.db.commit()
                self.db.refresh(existing)
            return existing
        else:
            # Create new work item
            work_item = WorkItemModel(**work_item_fields)
            self.db.add(work_item)
            if auto_commit:
                self.db.commit()
                self.db.refresh(work_item)
            return work_item
    
    def bulk_create_or_update_work_items(self, work_items_data: List[Dict[str, Any]], 
                                        data_source_id: int) -> int:
        """Bulk create or update work items for better performance.

        Uses deferred commits and periodic batch commits to improve throughput
        and provide progress logging for large datasets.
        """
        updated_count = 0
        batch_size = 500

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
    def get_work_items_by_data_source(self, data_source_id: int, 
                                     limit: Optional[int] = None) -> List[WorkItemModel]:
        """Get all work items for a data source"""
        query = self.db.query(WorkItemModel).filter(
            WorkItemModel.data_source_id == data_source_id
        ).options(joinedload(WorkItemModel.data_source)).order_by(desc(WorkItemModel.updated_at))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()

    def get_total_work_item_count(self, enabled_sources_only: bool = True) -> int:
        """Get total count of work items, optionally restricting to enabled sources."""
        query = self.db.query(WorkItemModel)
        if enabled_sources_only:
            query = query.join(DataSourceModel).filter(DataSourceModel.enabled == True)
        return query.count()
    
    def get_work_items_by_source_name(self, source_name: str, 
                                     limit: Optional[int] = None) -> List[WorkItemModel]:
        """Get all work items for a data source by name"""
        query = self.db.query(WorkItemModel).join(DataSourceModel).filter(
            DataSourceModel.name == source_name
        ).options(joinedload(WorkItemModel.data_source)).order_by(desc(WorkItemModel.updated_at))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_all_work_items(self, limit: Optional[int] = None, 
                          enabled_sources_only: bool = True) -> List[WorkItemModel]:
        """Get all work items from all data sources"""
        query = self.db.query(WorkItemModel).join(DataSourceModel)
        
        if enabled_sources_only:
            query = query.filter(DataSourceModel.enabled == True)
        
        query = query.options(joinedload(WorkItemModel.data_source)).order_by(desc(WorkItemModel.updated_at))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    def delete_work_items_by_data_source(self, data_source_id: int) -> int:
        """Delete all work items for a data source"""
        deleted_count = self.db.query(WorkItemModel).filter(
            WorkItemModel.data_source_id == data_source_id
        ).delete()
        self.db.commit()
        return deleted_count
    def convert_to_tickets(self, work_items: List[WorkItemModel]) -> List[Ticket]:
        """Convert work items to Ticket objects"""
        tickets = []
        
        for work_item in work_items:
            try:
                ticket = Ticket(
                    id=work_item.external_id,
                    title=work_item.title,
                    description=work_item.description or "",
                    comments=work_item.comments or [],
                    source_name=work_item.data_source.name if work_item.data_source else None,
                    organization=work_item.data_source.organization if work_item.data_source else None,
                    project=work_item.data_source.project if work_item.data_source else None,
                    area_path=work_item.area_path,
                    iteration_path=work_item.iteration_path,
                    created_date=work_item.azure_created_date,
                    additional_fields=work_item.additional_fields or {}
                )
                tickets.append(ticket)
            except Exception as e:
                # Log warning but continue processing
                logger.warning(
                    f"Failed to convert work item {work_item.external_id} to ticket: {e}"
                )
                continue
        
        return tickets
    
    def get_tickets_from_all_sources(self) -> List[Ticket]:
        """Get all work items as Ticket objects"""
        work_items = self.get_all_work_items()
        return self.convert_to_tickets(work_items)
