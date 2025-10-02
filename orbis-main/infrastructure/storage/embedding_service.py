import asyncio
import importlib.util
import logging
import time
from typing import Any, Protocol

from openai import AzureOpenAI
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from config.settings import settings
from core.schemas import BaseContent
from infrastructure.storage.generic_vector_service import GenericVectorService

# Defer heavy imports to runtime; only check availability here
LOCAL_DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("sentence_transformers") is not None
    and importlib.util.find_spec("torch") is not None
)

logger = logging.getLogger(__name__)

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""
    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        ...

    def encode_single_text(self, text: str) -> list[float]:
        ...

    def get_model_info(self) -> dict:
        ...

    def is_loaded(self) -> bool:
        ...

class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers"""

    def __init__(self):
        if not LOCAL_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Local embedding dependencies (torch, sentence-transformers) are not installed. "
                "Please install them with: pip install torch sentence-transformers transformers"
            )

        # Import here to avoid import-time side effects when running tests on environments
        # where torch may not load cleanly.
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._SentenceTransformer = SentenceTransformer
        self._torch = torch

        self.model = None  # will hold an instance of SentenceTransformer
        self.device = settings.EMBEDDING_DEVICE
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._load_model()

    def _load_model(self):
        """Load the all-mpnet-base-v2 embedding model"""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Check if CUDA is available
            if self.device == "cuda" and not self._torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

            # Load model - all-mpnet-base-v2 is already optimized and doesn't need half precision
            self.model = self._SentenceTransformer(self.model_name, device=self.device)

            # Test the model
            test_embedding = self.model.encode("test", convert_to_tensor=True)
            logger.info(f"Local model loaded successfully. Embedding dimension: {test_embedding.shape[0]}")

        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            raise

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode a list of texts to embeddings"""
        if not self.model:
            raise RuntimeError("Local embedding model not loaded")

        if not texts:
            return []

        batch_size = batch_size or self.batch_size

        try:
            # Encode texts in batches - all-mpnet-base-v2 is optimized for speed
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            # Convert to list of lists
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts with local model: {e}")
            raise

    def encode_single_text(self, text: str) -> list[float]:
        """Encode a single text to embedding"""
        embeddings = self.encode_texts([text])
        return embeddings[0] if embeddings else []

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.model:
            return {"loaded": False, "provider": "local"}

        return {
            "loaded": True,
            "provider": "local",
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None

    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        if self.device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

class AzureOpenAIEmbeddingProvider:
    """Azure OpenAI embedding provider using text-embedding-v3-large"""

    def __init__(self):
        self.client: AzureOpenAI | None = None
        self.deployment_name = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        self.api_version = settings.AZURE_OPENAI_EMBEDDING_API_VERSION
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client for embeddings"""
        try:
            if not settings.AZURE_OPENAI_EMBEDDING_ENDPOINT or not settings.AZURE_OPENAI_EMBEDDING_API_KEY:
                raise ValueError("Azure OpenAI embedding endpoint and API key must be configured")

            logger.info(f"Initializing Azure OpenAI embedding client with deployment: {self.deployment_name}")

            self.client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                api_key=settings.AZURE_OPENAI_EMBEDDING_API_KEY,
                api_version=self.api_version
            )
            logger.info("Azure OpenAI embedding client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI embedding client: {e}")
            raise

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode a list of texts to embeddings using Azure OpenAI"""
        if not self.client:
            raise RuntimeError("Azure OpenAI embedding client not initialized")

        if not texts:
            return []

        try:
            embeddings = []
            # Process texts in batches for better performance
            batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                response = self.client.embeddings.create(
                    model=self.deployment_name,
                    input=batch_texts,
                    encoding_format="float"
                )

                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts with Azure OpenAI: {e}")
            raise

    def encode_single_text(self, text: str) -> list[float]:
        """Encode a single text to embedding"""
        embeddings = self.encode_texts([text])
        return embeddings[0] if embeddings else []

    def get_model_info(self) -> dict:
        """Get information about the Azure OpenAI embedding model"""
        if not self.client:
            return {"loaded": False, "provider": "azure"}

        return {
            "loaded": True,
            "provider": "azure",
            "deployment_name": self.deployment_name,
            "endpoint": settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
            "api_version": self.api_version,
            "embedding_dimension": None  # Dynamic based on deployment
        }

    def is_loaded(self) -> bool:
        """Check if the client is initialized"""
        return self.client is not None

class EmbeddingService:
    """Main embedding service that uses strategy pattern to support multiple providers"""

    def __init__(self):
        self.provider: EmbeddingProvider | None = None
        # Optional vector service attachment for convenience methods
        self.vector_service: GenericVectorService | None = None
        self._provider_initialized = False

    def _ensure_provider_initialized(self):
        """Lazily initialize the appropriate embedding provider based on configuration"""
        if self._provider_initialized:
            return

        if settings.EMBEDDING_PROVIDER == "azure":
            logger.info("Initializing Azure OpenAI embedding provider")
            self.provider = AzureOpenAIEmbeddingProvider()
        else:
            if not LOCAL_DEPENDENCIES_AVAILABLE:
                raise ImportError(
                    "Local embedding provider requested but dependencies are not installed. "
                    "Either install local dependencies (torch, sentence-transformers, transformers) "
                    "or set EMBEDDING_PROVIDER=azure in your environment."
                )
            logger.info("Initializing local embedding provider")
            self.provider = LocalEmbeddingProvider()

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

        info = self.provider.get_model_info()
        info["current_provider"] = settings.EMBEDDING_PROVIDER
        return info

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

        # Use asyncio.to_thread to make the sync call async
        return await asyncio.to_thread(self.provider.encode_single_text, query)



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

        bs = batch_size or settings.EMBEDDING_BATCH_SIZE
        vs = vector_service or self.vector_service

        all_embeddings: list[list[float]] = []
        processed = 0
        start_time = time.time()
        num_batches = (total + bs - 1) // bs
        logger.info(
            f"Embedding {total} content items in {num_batches} batches (batch_size={bs})"
        )
        last_percent_logged = -1

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
                processed += len(batch_embeddings)

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

                # Aggregated progress logging every 10% or on completion
                percent = int((processed * 100) / total)
                if percent >= last_percent_logged + 10 or processed == total:
                    elapsed = time.time() - start_time
                    rate = (processed / elapsed) if elapsed > 0 else 0.0
                    remaining = total - processed
                    eta_seconds = (remaining / rate) if rate > 0 else 0.0
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    logger.info(
                        f"Progress: {processed}/{total} ({percent}%) - elapsed {elapsed:.1f}s, ETA {eta_min:02d}:{eta_sec:02d}"
                    )
                    last_percent_logged = percent

            elapsed_total = time.time() - start_time
            if vs is not None:
                return {
                    "message": f"Successfully embedded and stored {processed} content items in {elapsed_total:.1f}s",
                    "total_items": total,
                    "processed_items": processed,
                    "stored": True,
                    "success": True,
                }
            else:
                return {
                    "message": f"Successfully embedded {processed} content items (not stored) in {elapsed_total:.1f}s",
                    "total_items": total,
                    "processed_items": processed,
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
        if hasattr(settings, 'AZURE_OPENAI_EMBEDDING_MODEL'):
            return f"azure:{settings.AZURE_OPENAI_EMBEDDING_MODEL}"
        elif hasattr(settings, 'LOCAL_EMBEDDING_MODEL'):
            return f"local:{settings.LOCAL_EMBEDDING_MODEL}"
        else:
            return "unknown"

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
        from utils.content_hash import hash_content
        from app.db.models import ContentEmbedding
        
        # Calculate current hash
        current_hash = hash_content(content_item)
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
        from utils.content_hash import hash_text_content
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
        from utils.content_hash import hash_content
        
        current_model = self._get_current_embedding_model()
        
        with get_db_session() as db:
            for content_item in content_items:
                content_id = getattr(content_item, 'id', None)
                if content_id is None:
                    continue
                
                current_hash = hash_content(content_item)
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
        from utils.content_hash import hash_text_content
        
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
