from typing import List, Optional, Dict, Any
import logging
from config import settings
import asyncio
from models.schemas import Ticket
from services.vector_service import VectorService

from orbis_core.embedding import EmbeddingService as CoreEmbeddingService
from orbis_core.utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service wrapper for orbis-search with Ticket support"""

    def __init__(self):
        # Initialize core embedding service with lazy loading
        self.core_service = CoreEmbeddingService(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BULK_BATCH_SIZE,
            max_chunk_size=settings.EMBEDDING_MAX_CHUNK_SIZE,
            hugging_face_token=settings.HUGGING_FACE_TOKEN
        )
        # Optional vector service attachment for convenience methods
        self.vector_service: Optional[VectorService] = None

    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encode a list of texts to embeddings"""
        return self.core_service.encode_texts(texts, batch_size)

    def encode_single_text(self, text: str, max_length: Optional[int] = None) -> List[float]:
        """Encode a single text to embedding with chunking for long texts"""
        return self.core_service.encode_single_text(text, max_length)

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return self.core_service.get_model_info()

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.core_service.is_loaded()

    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        self.core_service.clear_cache()
    

    async def generate_embeddings(
        self,
        tickets: List[Ticket],
        vector_service: Optional[VectorService] = None,
        batch_size: Optional[int] = None,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Generate embeddings for tickets and optionally store them via a vector service.

        Args:
            tickets: List of tickets to embed
            vector_service: Optional vector store to persist embeddings. If not provided,
                will use self.vector_service if available. If neither is available, embeddings
                will be generated and returned but not stored.
            batch_size: Optional override for batching
            clear_existing: If True, clears the vector collection before storing the first batch

        Returns:
            Dict with summary of the operation. Includes 'embeddings' only when not stored.
        """
        if tickets is None:
            tickets = []
        total = len(tickets)
        if total == 0:
            return {
                "message": "No tickets provided",
                "total_tickets": 0,
                "processed_tickets": 0,
                "stored": False,
                "success": True,
            }

        bs = batch_size or settings.EMBEDDING_BULK_BATCH_SIZE
        vs = vector_service or self.vector_service

        all_embeddings: List[List[float]] = []
        num_batches = (total + bs - 1) // bs
        logger.info(
            f"Embedding {total} tickets in {num_batches} batches (batch_size={bs})"
        )

        progress = ProgressTracker(total, operation_name="Embedding tickets")

        try:
            for i in range(0, total, bs):
                batch_tickets = tickets[i:i + bs]
                # Concatenate text using embedding configurations (same as VectorService)
                batch_texts: List[str] = []
                for t in batch_tickets:
                    parts: List[str] = [t.title]
                    if t.description:
                        parts.append(t.description)
                    if t.comments:
                        parts.extend(t.comments)

                    batch_texts.append(" ".join(parts))

                # Generate embeddings for the batch off the event loop
                batch_embeddings = await asyncio.to_thread(
                    self.encode_texts, batch_texts, batch_size=bs
                )

                if vs is not None:
                    # Store immediately per batch off the event loop
                    await asyncio.to_thread(
                        vs.store_embeddings,
                        batch_tickets,
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
                    "message": f"Successfully embedded and stored {progress.processed} tickets in {elapsed_total:.1f}s",
                    "total_tickets": total,
                    "processed_tickets": progress.processed,
                    "stored": True,
                    "success": True,
                }
            else:
                return {
                    "message": f"Successfully embedded {progress.processed} tickets (not stored) in {elapsed_total:.1f}s",
                    "total_tickets": total,
                    "processed_tickets": progress.processed,
                    "stored": False,
                    "embeddings": all_embeddings,
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise