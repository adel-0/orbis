"""
Data ingestion service for Azure DevOps work items using SQLite database.
Handles the complete pipeline: fetch -> save to database -> embed.
"""

import logging
import asyncio
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from app.db.models import IngestionLog
from app.db.session import get_db_session
from services.data_source_service import DataSourceService
from services.work_item_service import WorkItemService
from orbis_core.connectors.azure_devops import Client
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from orbis_core.search import BM25Service
from models.schemas import Ticket

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Service for ingesting work item data from Azure DevOps into SQLite database"""
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None,
                 vector_service: Optional[VectorService] = None,
                 bm25_service: Optional[BM25Service] = None):
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.bm25_service = bm25_service
        self.use_incremental_sync = True
    
    async def ingest_single_source(self, source_name: str, 
                                  force_full_sync: bool = False) -> Dict[str, Any]:
        """Ingest work items from a single data source into database"""
        
        start_time = datetime.now(UTC)
        
        with get_db_session() as db:
            # Get data source from database
            ds_service = DataSourceService(db)
            data_source = ds_service.get_data_source(source_name)
            
            if not data_source:
                raise ValueError(f"Data source '{source_name}' not found")
            
            if not data_source.enabled:
                raise ValueError(f"Data source '{source_name}' is disabled")
            
            # Create ingestion log entry
            ingestion_log = IngestionLog(
                data_source_id=data_source.id,
                sync_type="full" if force_full_sync else "incremental",
                status="running",
                started_at=start_time
            )
            db.add(ingestion_log)
            db.commit()
            db.refresh(ingestion_log)
            
            try:
                # Initialize Azure DevOps client using the from_data_source class method
                # This automatically handles both PAT and OAuth2 authentication
                client = Client.from_data_source(data_source)
                
                # Determine last run time for incremental sync
                last_run_time = None
                if self.use_incremental_sync and not force_full_sync:
                    # Get last successful ingestion time
                    last_successful_log = db.query(IngestionLog).filter(
                        IngestionLog.data_source_id == data_source.id,
                        IngestionLog.status == "success"
                    ).order_by(IngestionLog.completed_at.desc()).first()
                    
                    if last_successful_log and last_successful_log.completed_at:
                        last_run_time = last_successful_log.completed_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                
                # Fetch work items from all queries
                all_workitems = []
                for query_id in data_source.query_ids:
                    logger.info(f"Fetching work items from query {query_id} for {source_name}")
                    
                    workitems, _ = await client.get_workitems(
                        query_id=query_id,
                        use_incremental_sync=not force_full_sync,
                        last_run_datetime=last_run_time,
                        skip_comments=False
                    )
                    
                    if workitems:
                        all_workitems.extend(workitems)
                        logger.info(f"Fetched {len(workitems)} work items from query {query_id}")
                
                # Save work items to database
                wi_service = WorkItemService(db)
                
                if all_workitems:
                    if force_full_sync:
                        # Full sync - remove existing work items first
                        deleted_count = wi_service.delete_work_items_by_data_source(data_source.id)
                        logger.info(f"Deleted {deleted_count} existing work items for full sync")
                    
                    # Bulk insert/update work items
                    updated_count = wi_service.bulk_create_or_update_work_items(
                        all_workitems, data_source.id
                    )
                    
                    logger.info(f"Saved {updated_count} work items to database for {source_name}")
                    total_workitems = updated_count
                else:
                    logger.info(f"No work items to save for source {source_name}")
                    total_workitems = 0
                
                # Update ingestion log
                end_time = datetime.now(UTC)
                execution_time = (end_time - start_time).total_seconds()
                
                ingestion_log.status = "success"
                ingestion_log.fetched_workitems = len(all_workitems)
                ingestion_log.total_workitems = total_workitems
                ingestion_log.execution_time_seconds = int(execution_time)
                ingestion_log.completed_at = end_time
                db.commit()
                
                return {
                    "success": True,
                    "source_name": source_name,
                    "fetched_workitems": len(all_workitems),
                    "total_workitems": total_workitems,
                    "execution_time_seconds": execution_time,
                    "sync_type": "full" if force_full_sync else "incremental",
                    "timestamp": end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                }
                
            except Exception as e:
                # Update ingestion log with error
                end_time = datetime.now(UTC)
                execution_time = (end_time - start_time).total_seconds()
                
                ingestion_log.status = "failed"
                ingestion_log.execution_time_seconds = int(execution_time)
                ingestion_log.error_message = str(e)
                ingestion_log.completed_at = end_time
                db.commit()
                
                logger.error(f"Failed to ingest data from {source_name}: {e}")
                raise
    
    async def ingest_workitems(self, force_full_sync: bool = False, 
                              skip_embedding: bool = False,
                              source_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Unified method for ingesting work items - handles both single and multi-source scenarios
        
        Args:
            force_full_sync: Force full sync instead of incremental
            skip_embedding: Skip automatic embedding generation
            source_names: Specific sources to sync (None = all enabled sources)
        
        Returns:
            Unified response format compatible with DataIngestionResponse
        """
        start_time = datetime.now(UTC)
        
        # Get data sources based on request
        with get_db_session() as db:
            ds_service = DataSourceService(db)
            
            if source_names:
                # Specific sources requested
                data_sources = []
                for source_name in source_names:
                    source = ds_service.get_data_source(source_name)
                    if source:
                        if not source.enabled:
                            logger.warning(f"Data source '{source_name}' is disabled, skipping")
                            continue
                        data_sources.append(source)
                    else:
                        logger.warning(f"Data source '{source_name}' not found, skipping")
            else:
                # All enabled sources
                data_sources = ds_service.get_all_data_sources(enabled_only=True)
        
        if not data_sources:
            return {
                "success": False,
                "message": "No enabled data sources found",
                "sources": [],
                "total_fetched_workitems": 0,
                "total_saved_workitems": 0,
                "total_workitems_across_sources": 0,
                "execution_time_seconds": 0.0,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "embedding_result": None
            }
        
        # Handle single source scenario (for backward compatibility)
        if len(data_sources) == 1 and not source_names:
            try:
                result = await self.ingest_single_source(data_sources[0].name, force_full_sync)
                
                # Generate embeddings if requested and services are available
                embedding_result = None
                if not skip_embedding and self.embedding_service and self.vector_service:
                    try:
                        tickets = self.load_all_tickets()
                        if tickets:
                            embedding_result = await self.embedding_service.generate_embeddings(tickets)
                            logger.info(f"Generated embeddings for {len(tickets)} tickets")

                            # Also build BM25 index when doing embeddings
                            if self.bm25_service:
                                bm25_docs = []
                                for ticket in tickets:
                                    concatenated_text = self.vector_service._concatenate_ticket_text(ticket)
                                    bm25_docs.append({
                                        'ticket': ticket,
                                        'concatenated_text': concatenated_text
                                    })
                                self.bm25_service.index_documents(bm25_docs, save_to_disk=True)
                                logger.info(f"Indexed {len(tickets)} documents for BM25 search")
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings: {e}")
                        embedding_result = {"error": str(e)}
                
                # Return single-source compatible format
                end_time = datetime.now(UTC)
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    "success": result["success"],
                    "message": f"Successfully ingested from {data_sources[0].name}",
                    "sources": [result],
                    "total_fetched_workitems": result["fetched_workitems"],
                    "total_saved_workitems": result["total_workitems"],
                    "total_workitems_across_sources": result["total_workitems"],
                    "execution_time_seconds": execution_time,
                    "timestamp": result["timestamp"],
                    "embedding_result": embedding_result
                }
                
            except Exception as e:
                end_time = datetime.now(UTC)
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    "success": False,
                    "message": f"Failed to ingest from {data_sources[0].name}: {str(e)}",
                    "sources": [],
                    "total_fetched_workitems": 0,
                    "total_saved_workitems": 0,
                    "total_workitems_across_sources": 0,
                    "execution_time_seconds": execution_time,
                    "timestamp": end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "error": str(e)
                }
        
        # Multi-source scenario (parallelized)
        results: List[Dict[str, Any]] = []
        total_fetched = 0
        total_saved = 0

        async def _ingest_source(name: str) -> Dict[str, Any]:
            try:
                return await self.ingest_single_source(name, force_full_sync)
            except Exception as e:
                logger.error(f"Failed to ingest from source {name}: {e}")
                return {
                    "success": False,
                    "source_name": name,
                    "error": str(e),
                    "fetched_workitems": 0,
                    "total_workitems": 0,
                    "execution_time_seconds": 0,
                    "sync_type": "full" if force_full_sync else "incremental",
                }

        tasks = [
            _ingest_source(source.name)
            for source in data_sources
        ]

        parallel_results = await asyncio.gather(*tasks, return_exceptions=False)
        results.extend(parallel_results)

        for res in parallel_results:
            total_fetched += res.get("fetched_workitems", 0)
            total_saved += res.get("total_workitems", 0)
        
        # Generate embeddings if requested and services are available
        embedding_result = None
        if not skip_embedding and self.embedding_service and self.vector_service:
            try:
                tickets = self.load_all_tickets()
                if tickets:
                    embedding_result = await self.embedding_service.generate_embeddings(tickets)
                    logger.info(f"Generated embeddings for {len(tickets)} tickets")

                    # Also build BM25 index when doing embeddings
                    if self.bm25_service:
                        bm25_docs = []
                        for ticket in tickets:
                            concatenated_text = self.vector_service._concatenate_ticket_text(ticket)
                            bm25_docs.append({
                                'ticket': ticket,
                                'concatenated_text': concatenated_text
                            })
                        self.bm25_service.index_documents(bm25_docs, save_to_disk=True)
                        logger.info(f"Indexed {len(tickets)} documents for BM25 search")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                embedding_result = {"error": str(e)}
        
        success_count = sum(1 for r in results if r.get("success", False))
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()
        
        return {
            "success": success_count > 0,
            "message": f"Processed {success_count}/{len(results)} sources successfully",
            "sources": results,
            "total_saved_workitems": total_saved,
            "total_fetched_workitems": total_fetched,
            "total_workitems_across_sources": total_saved,
            "execution_time_seconds": execution_time,
            "timestamp": end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "embedding_result": embedding_result
        }

    async def ingest_all_sources(self, force_full_sync: bool = False,
                                skip_embedding: bool = False,
                                **kwargs: Any) -> Dict[str, Any]:
        """
        Alias for ingest_workitems() to maintain compatibility with orbis-core SchedulerService.
        Ingests from all enabled sources.
        """
        return await self.ingest_workitems(
            force_full_sync=force_full_sync,
            skip_embedding=skip_embedding,
            source_names=None  # None = all enabled sources
        )

    def get_ingestion_status(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Get unified ingestion status (single or multi-source format)"""
        
        with get_db_session() as db:
            ds_service = DataSourceService(db)
            
            if source_name:
                # Get status for specific source
                data_source = ds_service.get_data_source(source_name)
                
                if not data_source:
                    return {"error": f"Data source '{source_name}' not found"}
                
                # Get latest ingestion log
                latest_log = db.query(IngestionLog).filter(
                    IngestionLog.data_source_id == data_source.id
                ).order_by(IngestionLog.started_at.desc()).first()
                
                # Get work item count
                wi_service = WorkItemService(db)
                work_items = wi_service.get_work_items_by_data_source(data_source.id)
                
                # Return single-source compatible format
                source_status = {
                    "source_name": source_name,
                    "last_run_timestamp": latest_log.completed_at.isoformat() if latest_log and latest_log.completed_at else None,
                    "total_workitems": len(work_items),
                    "workitems_date_range": self._get_date_range(work_items) if work_items else None,
                    "organization": data_source.organization,
                    "project": data_source.project,
                    "enabled": data_source.enabled
                }
                return {
                    "sources": [source_status],
                    "total_sources": 1,
                    "enabled_sources": 1 if data_source.enabled else 0,
                    "global_last_run": source_status["last_run_timestamp"],
                    "total_workitems_across_sources": len(work_items)
                }
            else:
                # Get status for all sources
                all_sources = ds_service.get_all_data_sources()
                
                if len(all_sources) == 1:
                    # Single source system - return single-source format
                    source = all_sources[0]
                    return self.get_ingestion_status(source.name)
                else:
                    # Multi-source system - return multi-source format
                    statuses = []
                    total_workitems = 0
                    global_last_run = None
                    
                    for source in all_sources:
                        # Get latest ingestion log
                        latest_log = db.query(IngestionLog).filter(
                            IngestionLog.data_source_id == source.id
                        ).order_by(IngestionLog.started_at.desc()).first()
                        
                        # Get work item count
                        wi_service = WorkItemService(db)
                        work_items = wi_service.get_work_items_by_data_source(source.id)
                        
                        source_status = {
                            "source_name": source.name,
                            "last_run_timestamp": latest_log.completed_at.isoformat() if latest_log and latest_log.completed_at else None,
                            "total_workitems": len(work_items),
                            "workitems_date_range": self._get_date_range(work_items) if work_items else None,
                            "organization": source.organization,
                            "project": source.project,
                            "enabled": source.enabled
                        }
                        
                        statuses.append(source_status)
                        total_workitems += len(work_items)
                        
                        # Track global last run
                        if latest_log and latest_log.completed_at:
                            if not global_last_run or latest_log.completed_at > global_last_run:
                                global_last_run = latest_log.completed_at
                    
                    return {
                        "sources": statuses,
                        "total_sources": len(all_sources),
                        "enabled_sources": sum(1 for s in all_sources if s.enabled),
                        "global_last_run": global_last_run.isoformat() if global_last_run else None,
                        "total_workitems_across_sources": total_workitems
                    }
    
    def _get_date_range(self, work_items: List) -> Optional[Dict[str, str]]:
        """Get date range for work items"""
        if not work_items:
            return None
        
        try:
            dates = [item.created_date for item in work_items if hasattr(item, 'created_date') and item.created_date]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                return {
                    "earliest": min_date.isoformat() if hasattr(min_date, 'isoformat') else str(min_date),
                    "latest": max_date.isoformat() if hasattr(max_date, 'isoformat') else str(max_date)
                }
        except Exception as e:
            logger.warning(f"Failed to get date range: {e}")
        
        return None
    
    def load_all_tickets(self) -> List[Ticket]:
        """Load all tickets from database for embedding"""
        with WorkItemService() as wi_service:
            return wi_service.get_tickets_from_all_sources()
    
    def get_work_items_for_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Get work items for a specific source from database"""
        with get_db_session() as db:
            ds_service = DataSourceService(db)
            data_source = ds_service.get_data_source(source_name)
            
            if not data_source:
                return []
            
            wi_service = WorkItemService(db)
            work_items = wi_service.get_work_items_by_data_source(data_source.id)
            
            # Convert to dictionary format for backward compatibility
            result = []
            for item in work_items:
                result.append({
                    'Id': item.external_id,
                    'Title': item.title,
                    'Description': item.description,
                    'Comments': item.comments,
                    'source_name': source_name,
                    'organization': data_source.organization,
                    'project': data_source.project
                })
            
            return result
