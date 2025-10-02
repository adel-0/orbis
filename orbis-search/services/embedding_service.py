from typing import List, Optional, Dict, Any
import logging
from config import settings
import asyncio
from models.schemas import Ticket
from services.vector_service import VectorService
from services.text_chunking import chunk_text, should_chunk_text
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service using local sentence-transformers"""
    
    def __init__(self):
        # Import here to avoid import-time side effects when running tests on environments
        # where torch may not load cleanly.
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore

        self._SentenceTransformer = SentenceTransformer
        self._torch = torch

        self.model = None  # will hold an instance of SentenceTransformer
        self.device = settings.EMBEDDING_DEVICE
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self.bulk_batch_size = settings.EMBEDDING_BULK_BATCH_SIZE
        # Optional vector service attachment for convenience methods
        self.vector_service: Optional[VectorService] = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model with Hugging Face authentication if needed"""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Set up Hugging Face token if available
            if settings.HUGGING_FACE_TOKEN:
                import os
                os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HUGGING_FACE_TOKEN
                logger.info("Using Hugging Face authentication token")

            # Check if CUDA is available
            if self.device == "cuda" and not self._torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

            # Load model with authentication
            load_kwargs = {"device": self.device}
            if settings.HUGGING_FACE_TOKEN:
                load_kwargs["token"] = settings.HUGGING_FACE_TOKEN

            self.model = self._SentenceTransformer(self.model_name, **load_kwargs)
            
            # Expose the tokenizer for token-aware chunking
            self.tokenizer = self.model.tokenizer
            
            # Test the model
            test_embedding = self.model.encode("test", convert_to_tensor=True)
            logger.info(f"Local model loaded successfully. Embedding dimension: {test_embedding.shape[0]}")
            
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encode a list of texts to embeddings"""
        if not self.model:
            raise RuntimeError("Local embedding model not loaded")
        
        if not texts:
            return []
        
        batch_size = batch_size or self.bulk_batch_size
        
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
    
    def encode_single_text(self, text: str, max_length: Optional[int] = None) -> List[float]:
        """Encode a single text to embedding with chunking for long texts"""
        # Get max sequence length from model if not provided
        if max_length is None:
            model_max_length = getattr(self.model, 'max_seq_length', settings.EMBEDDING_MAX_CHUNK_SIZE)
            # Use configured chunk size or model limit, whichever is smaller
            max_length = min(model_max_length, settings.EMBEDDING_MAX_CHUNK_SIZE)

        # If text is short enough, process normally with batch_size=1 for memory efficiency
        if not should_chunk_text(text, max_length, tokenizer=self.tokenizer):
            embeddings = self.encode_texts([text], batch_size=1)
            return embeddings[0] if embeddings else []

        # Split long text into chunks and process separately
        return self._encode_long_text_chunked(text, max_length)

    def _encode_long_text_chunked(self, text: str, max_length: int) -> List[float]:
        """Encode long text by splitting into chunks and streaming-average embeddings.

        This implementation keeps constant memory by not storing per-chunk embeddings.
        """
        chunks = chunk_text(text, max_length, tokenizer=self.tokenizer)

        if not chunks:
            return []

        try:
            import numpy as np
            running_sum: Optional[Any] = None  # np.ndarray once initialized
            count = 0

            for chunk in chunks:
                chunk_embedding = self.encode_texts([chunk], batch_size=1)
                if chunk_embedding:
                    # Convert to float32 numpy array to minimize memory footprint
                    vec = np.asarray(chunk_embedding[0], dtype=np.float32)
                    if running_sum is None:
                        running_sum = vec
                    else:
                        running_sum = running_sum + vec
                    count += 1

                # Clear cache after each chunk to free VRAM
                self.clear_cache()

            if running_sum is None or count == 0:
                return []

            avg_embedding = running_sum / float(count)
            # Normalize the averaged embedding
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            return avg_embedding.astype(np.float32).tolist()

        except Exception as e:
            logger.error(f"Failed to encode long text with chunking: {e}")
            # Fallback: try to encode just the first chunk
            try:
                embeddings = self.encode_texts([chunks[0]], batch_size=1)
                return embeddings[0] if embeddings else []
            except:
                return []
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.model:
            return {"loaded": False}
        
        return {
            "loaded": True,
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
        processed = 0
        start_time = time.time()
        num_batches = (total + bs - 1) // bs
        logger.info(
            f"Embedding {total} tickets in {num_batches} batches (batch_size={bs})"
        )
        last_percent_logged = -1

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
                processed += len(batch_embeddings)

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
                    "message": f"Successfully embedded and stored {processed} tickets in {elapsed_total:.1f}s",
                    "total_tickets": total,
                    "processed_tickets": processed,
                    "stored": True,
                    "success": True,
                }
            else:
                return {
                    "message": f"Successfully embedded {processed} tickets (not stored) in {elapsed_total:.1f}s",
                    "total_tickets": total,
                    "processed_tickets": processed,
                    "stored": False,
                    "embeddings": all_embeddings,
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise