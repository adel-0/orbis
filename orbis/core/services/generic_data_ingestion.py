"""
Generic Data Ingestion Service - Configuration-driven approach.
Handles ingestion for ANY data source type via configuration registry.
Replaces hardcoded Azure DevOps specific logic.
"""

import importlib
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func

from app.db.models import IngestionLog
from app.db.session import get_db_session
from core.config.data_sources import get_data_source_config
from infrastructure.data_processing.data_source_service import DataSourceService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService

logger = logging.getLogger(__name__)


class GenericDataIngestionService:
    """Handles ingestion for ANY data source type via configuration"""

    def __init__(self,
                 embedding_service: EmbeddingService | None = None,
                 vector_service: GenericVectorService | None = None):
        self.embedding_service = embedding_service
        self.vector_service = vector_service

    async def ingest_source(self, source_name: str, source_type: str,
                          source_config: dict[str, Any],
                          incremental: bool = True) -> dict[str, Any]:
        """
        Ingest data from ANY data source type using configuration-driven approach.

        Args:
            source_name: Name of the data source instance
            source_type: Type of data source (from config registry)
            source_config: Source-specific configuration dictionary
            incremental: Whether to perform incremental sync

        Returns:
            Dictionary containing ingestion results
        """
        start_time = datetime.now(UTC)

        try:
            # Get connector configuration for dynamic import
            ds_config = get_data_source_config(source_type)

            # Dynamic import of connector
            module = importlib.import_module(ds_config["connector_module"])
            connector_class = getattr(module, ds_config["connector_class"])

            # Create connector instance
            connector = connector_class()

            # Get collection name from connector (self-describing)
            collection_name = connector_class.get_collection_name()

            # Fetch raw data using connector
            logger.info(f"Fetching data from {source_name} (type: {source_type})")
            raw_items = await self._fetch_data_from_connector(connector, source_config, incremental)

            # Let connector convert to searchable format
            content_items = []
            for item in raw_items:
                # Connector decides how to make content searchable
                searchable_text = self._get_searchable_text(connector, item)
                metadata = self._get_metadata(connector, item)
                content_id = self._get_content_id(connector, item)

                # Enrich metadata with DataSource instance information
                metadata['source_name'] = source_name
                metadata['source_type'] = source_type
                if 'organization' in source_config:
                    metadata['organization'] = source_config['organization']
                if 'project' in source_config:
                    metadata['project'] = source_config['project']

                content_items.append({
                    "id": content_id,
                    "text": searchable_text,
                    "metadata": metadata,
                    "source_type": source_type,
                    "source_name": source_name
                })

            # Store raw content in regular database first
            await self._store_content_in_db(content_items, source_name, source_type)

            # Optional embedding generation - controlled by service initialization
            embedding_result = None
            if self.embedding_service and self.vector_service:
                embedding_result = await self._generate_embeddings(
                    content_items, collection_name
                )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return {
                "success": True,
                "processed": len(content_items),
                "source_type": source_type,
                "source_name": source_name,
                "collection": collection_name,
                "execution_time_seconds": execution_time,
                "embedding_result": embedding_result,
                "timestamp": datetime.now(UTC).isoformat()
            }

        except Exception as e:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            logger.error(f"Ingestion failed for {source_name} ({source_type}): {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "source_type": source_type,
                "source_name": source_name,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now(UTC).isoformat()
            }

    async def _fetch_data_from_connector(self, connector, source_config: dict[str, Any],
                                       incremental: bool) -> list[dict[str, Any]]:
        """
        Fetch data from connector using the generic interface.
        All connectors must implement the fetch_data() method.
        """
        if not hasattr(connector, 'fetch_data'):
            raise ValueError(f"Connector {type(connector).__name__} must implement fetch_data() method")

        return await connector.fetch_data(source_config, incremental)

    def _get_searchable_text(self, connector, item: dict[str, Any]) -> str:
        """Get searchable text using connector's method"""
        if hasattr(connector, 'get_searchable_text'):
            return connector.get_searchable_text(item)

        raise ValueError(f"Connector {type(connector).__name__} must implement get_searchable_text() method")

    def _get_metadata(self, connector, item: dict[str, Any]) -> dict[str, Any]:
        """Get metadata using connector's method"""
        if hasattr(connector, 'get_metadata'):
            return connector.get_metadata(item)

        raise ValueError(f"Connector {type(connector).__name__} must implement get_metadata() method")

    def _get_content_id(self, connector, item: dict[str, Any]) -> str:
        """Get content ID using connector's method"""
        if hasattr(connector, 'get_content_id'):
            return connector.get_content_id(item)

        raise ValueError(f"Connector {type(connector).__name__} must implement get_content_id() method")

    async def _generate_embeddings(self, content_items: list[dict[str, Any]],
                                 collection_name: str) -> dict[str, Any]:
        """Generate embeddings for content items"""
        try:
            if not content_items:
                return {'success': True, 'processed_items': 0}

            # Convert dictionary items to BaseContent objects for the embedding service
            from core.schemas import BaseContent, Ticket, WikiPageContent

            base_content_items: list[BaseContent] = []
            for item in content_items:
                metadata = item.get("metadata", {})
                source_type = item.get("source_type", "")

                # Determine content type and create appropriate BaseContent subclass
                if "workitem" in source_type.lower():
                    # Create Ticket object
                    content_obj = Ticket(
                        id=item["id"],
                        title=metadata.get("title", ""),
                        description=metadata.get("content", ""),
                        source_name=item.get("source_name"),
                        source_type=source_type,
                        organization=metadata.get("organization"),
                        project=metadata.get("project"),
                        comments=[],  # Comments are already concatenated in text
                        area_path=metadata.get("area_path"),
                        additional_fields=metadata
                    )
                elif "wiki" in source_type.lower():
                    # Create WikiPageContent object
                    content_obj = WikiPageContent(
                        id=item["id"],
                        title=metadata.get("title", ""),
                        content=metadata.get("content", ""),
                        source_name=item.get("source_name"),
                        source_type=source_type,
                        organization=metadata.get("organization"),
                        project=metadata.get("project"),
                        path=metadata.get("path"),
                        author=metadata.get("author"),
                        last_modified=metadata.get("last_modified")
                    )
                else:
                    # Generic BaseContent for other types
                    content_obj = BaseContent(
                        id=item["id"],
                        title=metadata.get("title", ""),
                        content_type=source_type,
                        source_name=item.get("source_name"),
                        source_type=source_type,
                        organization=metadata.get("organization"),
                        project=metadata.get("project")
                    )

                base_content_items.append(content_obj)

            # Generate embeddings using embedding service (it will handle text extraction)
            result = await self.embedding_service.generate_embeddings(
                base_content_items,
                vector_service=self.vector_service,
                clear_existing=False
            )

            return {
                'success': result.get('success', False),
                'processed_items': result.get('processed_items', 0),
                'collection': collection_name
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_items': 0
            }

    async def _store_in_vector_db(self, content_items: list[dict[str, Any]],
                                embeddings: list[list[float]], collection_name: str):
        """Store content and embeddings in vector database"""
        try:
            # Use vector service to store
            # This will need to be updated when we create the generic vector service
            await self.vector_service.add_content_batch(
                content_items, collection_name, embeddings
            )

        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            raise

    async def _store_content_in_db(self, content_items: list[dict[str, Any]],
                                 source_name: str, source_type: str) -> int:
        """Store raw content in the regular database"""
        try:

            from app.db.models import Content, DataSource

            with get_db_session() as db:
                # Get the data source
                data_source = db.query(DataSource).filter(
                    DataSource.name == source_name
                ).first()

                if not data_source:
                    logger.error(f"Data source '{source_name}' not found in database")
                    return 0

                stored_count = 0

                for item in content_items:
                    # Extract core content fields
                    external_id = item.get("id", "")
                    title = item.get("metadata", {}).get("title", "")
                    content_text = item.get("text", "")
                    metadata = item.get("metadata", {})

                    # Determine content type from metadata
                    content_type = metadata.get("content_type", "unknown")

                    # Check if content already exists
                    existing = db.query(Content).filter(
                        Content.external_id == external_id,
                        Content.data_source_id == data_source.id
                    ).first()

                    if existing:
                        # Update existing record
                        existing.title = title
                        existing.content = content_text
                        existing.content_metadata = metadata
                        existing.source_updated_date = self._parse_datetime(metadata.get("source_updated_date"))
                        existing.updated_at = func.now()
                    else:
                        # Create new record
                        content_record = Content(
                            external_id=external_id,
                            data_source_id=data_source.id,
                            content_type=content_type,
                            title=title,
                            content=content_text,
                            content_metadata=metadata,
                            source_created_date=self._parse_datetime(metadata.get("source_created_date")),
                            source_updated_date=self._parse_datetime(metadata.get("source_updated_date")),
                            source_reference=metadata.get("source_reference")
                        )
                        db.add(content_record)

                    stored_count += 1

                db.commit()
                logger.info(f"Stored {stored_count} content items in database for {source_name}")
                return stored_count

        except Exception as e:
            logger.error(f"Failed to store content in database: {e}")
            return 0

    def _parse_datetime(self, dt_str):
        """Parse datetime string, return None if invalid"""
        if not dt_str or not isinstance(dt_str, str):
            return None
        try:
            from dateutil.parser import parse
            return parse(dt_str)
        except Exception:
            return None

    async def ingest_data_source_by_name(self, source_name: str,
                                       force_full_sync: bool = False,
                                       skip_embedding: bool = False) -> dict[str, Any]:
        """
        Ingest a data source by name from the database.
        This provides compatibility with existing API endpoints.
        """
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
                # Determine source type
                source_type = self._determine_source_type(data_source)

                # Build source config from data source
                source_config = self._build_source_config(data_source, force_full_sync)

                # Skip embeddings if requested
                if skip_embedding:
                    self.embedding_service = None
                    self.vector_service = None

                # Perform ingestion
                result = await self.ingest_source(
                    source_name, source_type, source_config,
                    incremental=not force_full_sync
                )

                # Update ingestion log
                execution_time = (datetime.now(UTC) - start_time).total_seconds()
                ingestion_log.status = "completed" if result.get('success') else "failed"
                ingestion_log.completed_at = datetime.now(UTC)
                ingestion_log.execution_time_seconds = execution_time
                ingestion_log.total_fetched = result.get('processed', 0)
                ingestion_log.total_saved = result.get('processed', 0)

                if not result.get('success'):
                    ingestion_log.error_message = result.get('error', 'Unknown error')

                db.commit()

                return result

            except Exception as e:
                logger.error(f"Ingestion failed for {source_name}: {str(e)}")

                # Update log with error
                ingestion_log.status = "failed"
                ingestion_log.completed_at = datetime.now(UTC)
                ingestion_log.error_message = str(e)
                ingestion_log.execution_time_seconds = (datetime.now(UTC) - start_time).total_seconds()
                db.commit()

                return {
                    'success': False,
                    'error': str(e),
                    'execution_time_seconds': (datetime.now(UTC) - start_time).total_seconds(),
                    'timestamp': datetime.now(UTC).isoformat()
                }

    def _determine_source_type(self, data_source) -> str:
        """Determine source type from data source model"""
        # Use source_type field from database model
        if hasattr(data_source, 'source_type') and data_source.source_type:
            return data_source.source_type

        # Fallback: Use explicit data_source_type if available (backward compatibility)
        if hasattr(data_source, 'data_source_type') and data_source.data_source_type:
            # Handle both string and enum-like types
            source_type = str(data_source.data_source_type)
            if hasattr(data_source.data_source_type, 'value'):
                source_type = data_source.data_source_type.value
            elif hasattr(data_source.data_source_type, 'name'):
                source_type = data_source.data_source_type.name.lower()

            return source_type

        # If no explicit type is provided, we need the source_type to be explicitly set
        raise ValueError(f"Cannot determine source type for data source '{data_source.name}'. Please set source_type explicitly.")

    def _build_source_config(self, data_source, force_full_sync: bool) -> dict[str, Any]:
        """Build source configuration from data source model - pass through all config"""
        # Start with a copy of the original config to preserve all fields
        config = dict(data_source.config)

        # Add sync time for incremental syncs
        if not force_full_sync:
            with get_db_session() as db:
                ds_service = DataSourceService(db)
                last_sync = ds_service.get_last_sync_time(data_source.name)
                if last_sync:
                    config['last_sync_time'] = last_sync

        return config

    async def ingest_all_sources(self,
                                force_full_sync: bool = False,
                                skip_embedding: bool = False,
                                source_names: list[str] | None = None) -> dict[str, Any]:
        """Ingest from all enabled sources or specified sources"""

        start_time = datetime.now(UTC)

        with get_db_session() as db:
            ds_service = DataSourceService(db)

            if source_names:
                sources = [ds_service.get_data_source(name) for name in source_names]
                sources = [s for s in sources if s is not None]  # Filter out None values
            else:
                sources = ds_service.get_enabled_sources()

            if not sources:
                execution_time = (datetime.now(UTC) - start_time).total_seconds()
                return {
                    'success': True,
                    'message': 'No sources to ingest',
                    'sources': [],
                    'execution_time_seconds': execution_time,
                    'timestamp': datetime.now(UTC).isoformat()
                }

            results = []
            overall_success = True

            for source in sources:
                try:
                    result = await self.ingest_data_source_by_name(
                        source.name,
                        force_full_sync=force_full_sync,
                        skip_embedding=skip_embedding
                    )
                    results.append(result)

                    if not result.get('success', False):
                        overall_success = False

                except Exception as e:
                    logger.error(f"Failed to ingest source {source.name}: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'source_name': source.name,
                        'timestamp': datetime.now(UTC).isoformat()
                    })
                    overall_success = False

            # Calculate total processed items across all sources
            total_processed = sum(
                result.get('processed', 0) for result in results
                if result.get('success', False)
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return {
                'success': overall_success,
                'sources': results,
                'total_fetched_items': total_processed,
                'total_saved_items': total_processed,
                'total_items_across_sources': total_processed,
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now(UTC).isoformat()
            }

