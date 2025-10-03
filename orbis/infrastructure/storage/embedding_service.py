import asyncio
import logging
from typing import Any

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from config.settings import settings
from core.schemas import BaseContent
from infrastructure.storage.generic_vector_service import GenericVectorService
from orbis_core.embedding import EmbeddingService as CoreEmbeddingService
from orbis_core.utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Main embedding service with database tracking and incremental embedding support"""

    def __init__(self):
        self.provider: CoreEmbeddingService | None = None
        # Optional vector service attachment for convenience methods
        self.vector_service: GenericVectorService | None = None
        self._provider_initialized = False

    def _ensure_provider_initialized(self):
        """Lazily initialize the core embedding service"""
        if self._provider_initialized:
            return

        logger.info("Initializing core embedding service")
        self.provider = CoreEmbeddingService(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BULK_BATCH_SIZE,
            max_chunk_size=getattr(settings, 'EMBEDDING_MAX_CHUNK_SIZE', 384),
            hugging_face_token=getattr(settings, 'HUGGING_FACE_TOKEN', None)
        )

        self._provider_initialized = True

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode a list of texts to embeddings"""
        self._ensure_provider_initialized()
        if not self.provider:
            raise RuntimeError("Embedding provider not initialized")

        return self.provider.encode_texts(texts, batch_size)

    def encode_single_text(self, text: str) -> list[float]:
        """Encode a single text to embedding"""
        self._ensure_provider_initialized()
        if not self.provider:
            raise RuntimeError("Embedding provider not initialized")

        return self.provider.encode_single_text(text)

    def get_model_info(self) -> dict:
        """Get information about the current embedding provider"""
        self._ensure_provider_initialized()
        if not self.provider:
            return {"loaded": False, "provider": "none"}

        return self.provider.get_model_info()

    def is_loaded(self) -> bool:
        """Check if the provider is loaded"""
        self._ensure_provider_initialized()
        return self.provider is not None and self.provider.is_loaded()

    def clear_cache(self):
        """Clear cache if supported by the provider"""
        self._ensure_provider_initialized()
        if hasattr(self.provider, 'clear_cache'):
            self.provider.clear_cache()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query (async wrapper for encode_single_text)"""
        self._ensure_provider_initialized()
        if not self.provider:
            raise RuntimeError("Embedding provider not initialized")

        # Use the core service's async method
        return await self.provider.embed_query(query)



    async def generate_embeddings(
        self,
        content_items: list[BaseContent],
        vector_service: GenericVectorService | None = None,
        batch_size: int | None = None,
        clear_existing: bool = False
    ) -> dict[str, Any]:
        """Generate embeddings for content items and optionally store them via a vector service.

        Args:
            content_items: List of content items to embed
            vector_service: Optional vector store to persist embeddings. If not provided,
                will use self.vector_service if available. If neither is available, embeddings
                will be generated and returned but not stored.
            batch_size: Optional override for batching
            clear_existing: If True, clears the vector collection before storing the first batch

        Returns:
            Dict with summary of the operation. Includes 'embeddings' only when not stored.
        """
        if content_items is None:
            content_items = []
        total = len(content_items)
        if total == 0:
            return {
                "message": "No content items provided",
                "total_items": 0,
                "processed_items": 0,
                "stored": False,
                "success": True,
            }

        bs = batch_size or settings.EMBEDDING_BULK_BATCH_SIZE
        vs = vector_service or self.vector_service

        all_embeddings: list[list[float]] = []
        num_batches = (total + bs - 1) // bs
        logger.info(
            f"Embedding {total} content items in {num_batches} batches (batch_size={bs})"
        )

        progress = ProgressTracker(total, operation_name="Embedding content")

        try:
            for i in range(0, total, bs):
                batch_content_items = content_items[i:i + bs]
                # Concatenate text using embedding configurations (same as VectorService)
                batch_texts: list[str] = []
                for content_item in batch_content_items:
                    # Get embedding config for this content item's data source
                    embedding_config = self._get_embedding_config_for_content(content_item)

                    parts: list[str] = [content_item.title]

                    # Handle different content types dynamically
                    if hasattr(content_item, 'description') and content_item.description:
                        parts.append(content_item.description)
                    if hasattr(content_item, 'content') and content_item.content:
                        parts.append(content_item.content)
                    if hasattr(content_item, 'comments') and content_item.comments:
                        parts.extend(content_item.comments)
                    if hasattr(content_item, 'extracted_text') and content_item.extracted_text:
                        parts.append(content_item.extracted_text)

                    # Add configured additional fields for embedding if enabled
                    if (embedding_config and
                        embedding_config.get('enabled', False) and
                        hasattr(content_item, 'additional_fields') and
                        content_item.additional_fields):

                        embedding_fields = embedding_config.get('embedding_fields', [])
                        for field_name in embedding_fields:
                            field_value = content_item.additional_fields.get(field_name)
                            if field_value and isinstance(field_value, str) and field_value.strip():
                                parts.append(field_value.strip())

                    batch_texts.append(" ".join(parts))

                # Ensure provider is initialized before embedding (lazy loading)
                self._ensure_provider_initialized()
                # Generate embeddings for the batch off the event loop
                batch_embeddings = await asyncio.to_thread(
                    self.encode_texts, batch_texts, batch_size=bs
                )

                if vs is not None:
                    # Store immediately per batch off the event loop
                    await asyncio.to_thread(
                        vs.store_embeddings,
                        batch_content_items,
                        batch_embeddings,
                        clear_existing=(i == 0 and clear_existing)
                    )
                else:
                    # Accumulate if we are not storing
                    all_embeddings.extend(batch_embeddings)

                # Clear CUDA cache to free memory if applicable
                self.clear_cache()

                # Update progress with ETA logging
                progress.update(len(batch_embeddings))

            elapsed_total = progress.get_elapsed_time()
            if vs is not None:
                return {
                    "message": f"Successfully embedded and stored {progress.processed} content items in {elapsed_total:.1f}s",
                    "total_items": total,
                    "processed_items": progress.processed,
                    "stored": True,
                    "success": True,
                }
            else:
                return {
                    "message": f"Successfully embedded {progress.processed} content items (not stored) in {elapsed_total:.1f}s",
                    "total_items": total,
                    "processed_items": progress.processed,
                    "stored": False,
                    "embeddings": all_embeddings,
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _get_embedding_config_for_content(self, content_item: BaseContent) -> dict[str, Any] | None:
        """Get embedding configuration for a single content item's data source"""
        if not content_item.source_name:
            return None

        try:
            # Import here to avoid circular imports
            from infrastructure.data_processing.data_source_service import (
                DataSourceService,
            )

            with DataSourceService() as ds_service:
                data_source = ds_service.get_data_source(content_item.source_name)
                if data_source and data_source.config:
                    # embedding_field_config should be stored in the config JSON field
                    return data_source.config.get('embedding_field_config')
                return None
        except Exception as e:
            logger.warning(f"Failed to get embedding configuration for {content_item.source_name}: {e}")
            return None

    def _get_current_embedding_model(self) -> str:
        """Get the current embedding model identifier"""
        return f"local:{settings.LOCAL_EMBEDDING_MODEL}"

    async def generate_embeddings_incremental(
        self,
        content_items: list[BaseContent],
        vector_service: GenericVectorService | None = None,
        force_rebuild: bool = False,
        batch_size: int | None = None
    ) -> dict[str, Any]:
        """
        Generate embeddings only for new/changed content items.
        
        Args:
            content_items: List of content items to process
            vector_service: Optional vector store
            force_rebuild: If True, re-embed all content regardless of changes
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with results including skipped/processed counts
        """
        if not content_items:
            return {
                "message": "No content items provided",
                "total_items": 0,
                "processed_items": 0,
                "skipped_items": 0,
                "success": True,
            }

        logger.info(f"Starting incremental embedding for {len(content_items)} content items")
        
        if force_rebuild:
            return await self._embed_all_items(content_items, vector_service, batch_size, clear_existing=True)
        
        # Identify items that need embedding
        items_to_embed = []
        skipped_count = 0
        
        from app.db.session import get_db_session
        with get_db_session() as db:
            for content_item in content_items:
                if await self._needs_embedding(db, content_item):
                    items_to_embed.append(content_item)
                else:
                    skipped_count += 1

        logger.info(f"Found {len(items_to_embed)} items to embed, skipping {skipped_count} unchanged items")
        
        if not items_to_embed:
            return {
                "message": "No content changes detected, all embeddings are up to date",
                "total_items": len(content_items),
                "processed_items": 0,
                "skipped_items": skipped_count,
                "success": True,
            }

        # Generate embeddings for changed items only
        result = await self._embed_items_and_track(items_to_embed, vector_service, batch_size)
        result["total_items"] = len(content_items)
        result["skipped_items"] = skipped_count
        
        return result

    async def generate_embeddings_from_db_content(
        self,
        vector_service: GenericVectorService | None = None,
        content_ids: list[int] | None = None,
        data_source_id: int | None = None,
        force_rebuild: bool = False,
        batch_size: int | None = None
    ) -> dict[str, Any]:
        """
        Generate embeddings for content stored in database.
        
        Args:
            vector_service: Optional vector store
            content_ids: Specific content IDs to process, or None for all
            data_source_id: Filter by data source, or None for all sources
            force_rebuild: If True, re-embed all content regardless of changes
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with results
        """
        from app.db.models import Content
        from app.db.session import get_db_session
        from sqlalchemy.orm import joinedload
        
        with get_db_session() as db:
            # Build query with eager loading of data_source relationship
            query = db.query(Content).options(joinedload(Content.data_source))
            
            if content_ids is not None:
                query = query.filter(Content.id.in_(content_ids))
            
            if data_source_id is not None:
                query = query.filter(Content.data_source_id == data_source_id)
            
            content_records = query.all()
            
            if not content_records:
                return {
                    "message": "No content found matching criteria",
                    "total_items": 0,
                    "processed_items": 0,
                    "skipped_items": 0,
                    "success": True,
                }

        logger.info(f"Processing {len(content_records)} content records from database")
        
        if force_rebuild:
            return await self._embed_db_content_all(content_records, vector_service, batch_size, clear_existing=True)
        
        # Identify records that need embedding
        records_to_embed = []
        skipped_count = 0
        
        with get_db_session() as db:
            for record in content_records:
                if await self._db_content_needs_embedding(db, record):
                    records_to_embed.append(record)
                else:
                    skipped_count += 1

        logger.info(f"Found {len(records_to_embed)} records to embed, skipping {skipped_count} unchanged records")
        
        if not records_to_embed:
            return {
                "message": "No content changes detected, all embeddings are up to date",
                "total_items": len(content_records),
                "processed_items": 0,
                "skipped_items": skipped_count,
                "success": True,
            }

        # Generate embeddings for changed records only
        result = await self._embed_db_content_and_track(records_to_embed, vector_service, batch_size)
        result["total_items"] = len(content_records)
        result["skipped_items"] = skipped_count
        
        return result

    async def get_embedding_state(self) -> dict[str, Any]:
        """Get state of current embeddings"""
        from app.db.models import Content, ContentEmbedding
        from app.db.session import get_db_session
        
        current_model = self._get_current_embedding_model()
        
        with get_db_session() as db:
            # Count total content items
            total_content = db.query(func.count(Content.id)).scalar() or 0
            
            # Count embedded items for current model
            embedded_count = db.query(func.count(ContentEmbedding.id)).filter(
                ContentEmbedding.embedding_model == current_model
            ).scalar() or 0
            
            # Count items that need updates (content changed since embedding)
            needs_update_count = 0
            for content in db.query(Content).all():
                if await self._db_content_needs_embedding(db, content):
                    needs_update_count += 1
            
            return {
                "total_content_items": total_content,
                "embedded_items": embedded_count,
                "items_needing_update": needs_update_count,
                "current_embedding_model": current_model,
                "up_to_date": needs_update_count == 0
            }

    async def _needs_embedding(self, db: Session, content_item: BaseContent) -> bool:
        """Check if a BaseContent item needs embedding"""
        from orbis_core.utils.content_hash import hash_object
        from app.db.models import ContentEmbedding

        # Calculate current hash
        current_hash = hash_object(content_item, 'title', 'content')
        current_model = self._get_current_embedding_model()
        
        # Get content ID (assuming it has an id attribute)
        content_id = getattr(content_item, 'id', None)
        if content_id is None:
            # If no ID, always embed (it's new content)
            return True
        
        # Check if embedding exists and is current
        existing = db.query(ContentEmbedding).filter(
            and_(
                ContentEmbedding.content_id == content_id,
                ContentEmbedding.embedding_model == current_model
            )
        ).first()
        
        if existing is None:
            # No embedding exists
            return True
        
        if existing.content_hash != current_hash:
            # Content has changed
            return True
        
        # Embedding is up to date
        return False

    async def _db_content_needs_embedding(self, db: Session, content_record) -> bool:
        """Check if a database Content record needs embedding"""
        from orbis_core.utils.content_hash import hash_text_content
        from app.db.models import ContentEmbedding
        
        # Calculate current hash
        current_hash = hash_text_content(content_record.title or '', content_record.content or '')
        current_model = self._get_current_embedding_model()
        
        # Check if embedding exists and is current
        existing = db.query(ContentEmbedding).filter(
            and_(
                ContentEmbedding.content_id == content_record.id,
                ContentEmbedding.embedding_model == current_model
            )
        ).first()
        
        if existing is None:
            # No embedding exists
            return True
        
        if existing.content_hash != current_hash:
            # Content has changed
            return True
        
        # Embedding is up to date
        return False

    async def _embed_all_items(
        self,
        content_items: list[BaseContent],
        vector_service: GenericVectorService | None,
        batch_size: int | None,
        clear_existing: bool = False
    ) -> dict[str, Any]:
        """Embed all items regardless of existing embeddings"""
        if clear_existing:
            await self._clear_embedding_tracking()
        
        result = await self.generate_embeddings(
            content_items=content_items,
            vector_service=vector_service,
            batch_size=batch_size,
            clear_existing=clear_existing
        )
        
        # Track all items as embedded
        await self._track_embeddings_for_items(content_items)
        
        return result

    async def _embed_db_content_all(
        self,
        content_records: list,
        vector_service: GenericVectorService | None,
        batch_size: int | None,
        clear_existing: bool = False
    ) -> dict[str, Any]:
        """Embed all database content records regardless of existing embeddings"""
        if clear_existing:
            await self._clear_embedding_tracking()
        
        # Convert database records to BaseContent-like objects
        content_items = []
        for record in content_records:
            # Create a simple object with the necessary attributes
            # The data_source relationship should be eagerly loaded by the query
            content_item = type('ContentItem', (), {
                'id': record.id,
                'title': record.title or '',
                'content': record.content or '',
                'metadata': record.content_metadata or {},
                'source_name': record.data_source.name if record.data_source else None,
                'source_type': record.data_source.source_type if record.data_source else None,
            })()
            content_items.append(content_item)
        
        result = await self.generate_embeddings(
            content_items=content_items,
            vector_service=vector_service,
            batch_size=batch_size,
            clear_existing=clear_existing
        )
        
        # Track all records as embedded
        await self._track_embeddings_for_db_content(content_records)
        
        return result

    async def _embed_items_and_track(
        self,
        content_items: list[BaseContent],
        vector_service: GenericVectorService | None,
        batch_size: int | None
    ) -> dict[str, Any]:
        """Embed specific items and track them"""
        result = await self.generate_embeddings(
            content_items=content_items,
            vector_service=vector_service,
            batch_size=batch_size,
            clear_existing=False
        )
        
        # Track the embedded items
        await self._track_embeddings_for_items(content_items)
        
        return result

    async def _embed_db_content_and_track(
        self,
        content_records: list,
        vector_service: GenericVectorService | None,
        batch_size: int | None
    ) -> dict[str, Any]:
        """Embed specific database content records and track them"""
        # Convert database records to BaseContent-like objects
        content_items = []
        for record in content_records:
            # The data_source relationship should be eagerly loaded by the query
            content_item = type('ContentItem', (), {
                'id': record.id,
                'title': record.title or '',
                'content': record.content or '',
                'metadata': record.content_metadata or {},
                'source_name': record.data_source.name if record.data_source else None,
                'source_type': record.data_source.source_type if record.data_source else None,
            })()
            content_items.append(content_item)
        
        result = await self.generate_embeddings(
            content_items=content_items,
            vector_service=vector_service,
            batch_size=batch_size,
            clear_existing=False
        )
        
        # Track the embedded records
        await self._track_embeddings_for_db_content(content_records)
        
        return result

    async def _track_embeddings_for_items(self, content_items: list[BaseContent]):
        """Track that items have been embedded"""
        from app.db.models import ContentEmbedding
        from app.db.session import get_db_session
        from orbis_core.utils.content_hash import hash_object

        current_model = self._get_current_embedding_model()

        with get_db_session() as db:
            for content_item in content_items:
                content_id = getattr(content_item, 'id', None)
                if content_id is None:
                    continue

                current_hash = hash_object(content_item, 'title', 'content')
                vector_id = f"content_{content_id}"  # Vector database ID pattern
                
                # Upsert embedding tracking record
                existing = db.query(ContentEmbedding).filter(
                    ContentEmbedding.content_id == content_id
                ).first()
                
                if existing:
                    existing.content_hash = current_hash
                    existing.embedding_model = current_model
                    existing.updated_at = func.now()
                else:
                    embedding_record = ContentEmbedding(
                        content_id=content_id,
                        content_hash=current_hash,
                        vector_id=vector_id,
                        embedding_model=current_model
                    )
                    db.add(embedding_record)
            
            db.commit()

    async def _track_embeddings_for_db_content(self, content_records: list):
        """Track that database content records have been embedded"""
        from app.db.models import ContentEmbedding
        from app.db.session import get_db_session
        from orbis_core.utils.content_hash import hash_text_content
        
        current_model = self._get_current_embedding_model()
        
        with get_db_session() as db:
            for record in content_records:
                current_hash = hash_text_content(record.title or '', record.content or '')
                vector_id = f"content_{record.id}"  # Vector database ID pattern
                
                # Upsert embedding tracking record
                existing = db.query(ContentEmbedding).filter(
                    ContentEmbedding.content_id == record.id
                ).first()
                
                if existing:
                    existing.content_hash = current_hash
                    existing.embedding_model = current_model
                    existing.updated_at = func.now()
                else:
                    embedding_record = ContentEmbedding(
                        content_id=record.id,
                        content_hash=current_hash,
                        vector_id=vector_id,
                        embedding_model=current_model
                    )
                    db.add(embedding_record)
            
            db.commit()

    async def _clear_embedding_tracking(self):
        """Clear all embedding tracking records (for force rebuild)"""
        from app.db.models import ContentEmbedding
        from app.db.session import get_db_session
        
        with get_db_session() as db:
            db.query(ContentEmbedding).delete()
            db.commit()
        logger.info("Cleared all embedding tracking records")
