"""
Embedding service using local sentence-transformers.
Provides text embedding generation with token-aware chunking for long documents.
"""
from typing import Any, Optional
import logging

from .text_chunking import chunk_text, should_chunk_text

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service using local sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str = "cpu", batch_size: int = 32,
                 max_chunk_size: int = 384, hugging_face_token: str | None = None):
        """
        Initialize embedding service.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ("cpu" or "cuda")
            batch_size: Batch size for embedding generation
            max_chunk_size: Maximum tokens per chunk
            hugging_face_token: Optional Hugging Face API token
        """
        # Import here to avoid import-time side effects
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore

        self._SentenceTransformer = SentenceTransformer
        self._torch = torch

        self.model = None
        self.device = device
        self.model_name = model_name
        self.bulk_batch_size = batch_size
        self.max_chunk_size = max_chunk_size
        self.hugging_face_token = hugging_face_token
        self._load_model()

    def _load_model(self):
        """Load the embedding model with Hugging Face authentication if needed"""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Set up Hugging Face token if available
            if self.hugging_face_token:
                import os
                os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hugging_face_token
                logger.info("Using Hugging Face authentication token")

            # Check if CUDA is available
            if self.device == "cuda" and not self._torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

            # Load model with authentication
            load_kwargs = {"device": self.device}
            if self.hugging_face_token:
                load_kwargs["token"] = self.hugging_face_token

            self.model = self._SentenceTransformer(self.model_name, **load_kwargs)

            # Expose the tokenizer for token-aware chunking
            self.tokenizer = self.model.tokenizer

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

        batch_size = batch_size or self.bulk_batch_size

        try:
            # Encode texts in batches
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

    def encode_single_text(self, text: str, max_length: int | None = None) -> list[float]:
        """Encode a single text to embedding with chunking for long texts"""
        # Get max sequence length from model if not provided
        if max_length is None:
            model_max_length = getattr(self.model, 'max_seq_length', self.max_chunk_size)
            # Use configured chunk size or model limit, whichever is smaller
            max_length = min(model_max_length, self.max_chunk_size)

        # If text is short enough, process normally with batch_size=1 for memory efficiency
        if not should_chunk_text(text, max_length, tokenizer=self.tokenizer):
            embeddings = self.encode_texts([text], batch_size=1)
            return embeddings[0] if embeddings else []

        # Split long text into chunks and process separately
        return self._encode_long_text_chunked(text, max_length)

    def _encode_long_text_chunked(self, text: str, max_length: int) -> list[float]:
        """Encode long text by splitting into chunks and streaming-average embeddings.

        This implementation keeps constant memory by not storing per-chunk embeddings.
        """
        chunks = list(chunk_text(text, max_length, tokenizer=self.tokenizer))

        if not chunks:
            return []

        try:
            import numpy as np
            running_sum: Any | None = None  # np.ndarray once initialized
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
