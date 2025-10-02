import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from config import settings
from services.text_chunking import chunk_text, should_chunk_text

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - tests inject a fake reranker; avoid hard failure on import
    CrossEncoder = None  # type: ignore


logger = logging.getLogger(__name__)


class RerankService:
    """Cross-encoder based reranker for query-document pairs.

    Always-on service intended to refine vector search candidates.
    Uses a local BGE reranker model by default.
    """

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
        device: str = "auto",
        max_length: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self._device = self._resolve_device(device)
        self.max_length = max_length
        # Use embedding chunk size as default for consistency
        self.max_chunk_size = max_chunk_size or settings.EMBEDDING_MAX_CHUNK_SIZE
        self.model: Optional[CrossEncoder] = None  # type: ignore

        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():  # noqa: SIM108
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> None:
        if CrossEncoder is None:
            logger.warning(
                "sentence-transformers not available; RerankService will be disabled (tests may provide a fake)."
            )
            return

        try:
            init_kwargs: Dict[str, Any] = {"device": self._device}
            if self.max_length is not None:
                init_kwargs["max_length"] = self.max_length
            self.model = CrossEncoder(self.model_name, **init_kwargs)
            
            # Expose tokenizer for token-aware chunking
            self.tokenizer = self.model.tokenizer
            
            logger.info(
                f"RerankService loaded model '{self.model_name}' on device '{self._device}'"
            )
        except Exception as exc:
            logger.error(f"Failed to load reranker model '{self.model_name}': {exc}")
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_pairs(self, query: str, items: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for item in items:
            # Prefer precomputed concatenated text; fall back to available fields
            candidate_text = (
                item.get("concatenated_text")
                or getattr(item.get("ticket"), "title", "")
                or ""
            )
            if not candidate_text and item.get("ticket") is not None:
                ticket = item.get("ticket")
                # Best-effort string build
                parts: List[str] = []
                title = getattr(ticket, "title", None)
                desc = getattr(ticket, "description", None)
                comments = getattr(ticket, "comments", None)
                if title:
                    parts.append(str(title))
                if desc:
                    parts.append(str(desc))
                if comments:
                    try:
                        parts.extend([str(c) for c in comments])
                    except Exception:
                        pass
                candidate_text = " ".join(parts)
            pairs.append((query, str(candidate_text)))
        return pairs


    def _rerank_with_chunked_query(self, query: str, items: List[Dict[str, Any]]) -> List[float]:
        """Rerank items using chunked query AND chunked candidates with constant memory usage"""
        if not items:
            return []

        query_chunks = list(chunk_text(query, self.max_chunk_size, tokenizer=self.tokenizer))
        logger.info(f"Chunking query into {len(query_chunks)} chunks for reranking")

        # Initialize final scores for all items
        final_scores = [0.0] * len(items)

        # Calculate query chunk weights for weighted averaging
        query_chunk_weights = [len(self.tokenizer.encode(chunk)) for chunk in query_chunks]
        total_query_weight = sum(query_chunk_weights)

        if total_query_weight == 0:
            return final_scores

        # Process each query chunk independently to maintain constant memory usage
        for q_chunk_idx, q_chunk in enumerate(query_chunks):
            try:
                q_chunk_weight = query_chunk_weights[q_chunk_idx] / total_query_weight

                # Process each item individually with this query chunk
                for item_idx in range(len(items)):
                    # Extract candidate text on demand and chunk per item to avoid storing all chunks for all items
                    item = items[item_idx]
                    candidate_text = (
                        item.get("concatenated_text")
                        or getattr(item.get("ticket"), "title", "")
                        or ""
                    )
                    if not candidate_text and item.get("ticket") is not None:
                        ticket = item.get("ticket")
                        parts = []
                        for attr in ["title", "description", "comments"]:
                            val = getattr(ticket, attr, None)
                            if val:
                                if attr == "comments" and isinstance(val, list):
                                    parts.extend([str(c) for c in val])
                                else:
                                    parts.append(str(val))
                        candidate_text = " ".join(parts)

                    candidate_chunks = list(chunk_text(str(candidate_text), self.max_chunk_size, tokenizer=self.tokenizer))

                    # Calculate candidate chunk weights
                    cand_chunk_weights = [len(self.tokenizer.encode(c)) for c in candidate_chunks]
                    total_cand_weight = sum(cand_chunk_weights)

                    if total_cand_weight == 0:
                        continue

                    # Score all (query_chunk, candidate_chunk) pairs and average
                    item_score_for_q_chunk = 0.0
                    for c_chunk_idx, c_chunk in enumerate(candidate_chunks):
                        c_chunk_weight = cand_chunk_weights[c_chunk_idx] / total_cand_weight

                        # Score single pair (always batch_size=1 for constant memory)
                        single_pair = [(q_chunk, c_chunk)]
                        pair_score = self.model.predict(
                            single_pair,
                            batch_size=1,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )[0]

                        # Weighted contribution from this candidate chunk
                        item_score_for_q_chunk += pair_score * c_chunk_weight

                    # Add weighted score from this query chunk to final score
                    final_scores[item_idx] += item_score_for_q_chunk * q_chunk_weight

                # Clear CUDA cache after each query chunk to maintain constant memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

                # Log progress for long operations
                if len(query_chunks) > 3:
                    logger.info(f"Processed query chunk {q_chunk_idx+1}/{len(query_chunks)}")

            except Exception as exc:
                logger.error(f"Failed to process query chunk {q_chunk_idx+1}: {exc}")
                # Skip this chunk (no contribution to scores)

        return final_scores

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Score and return the top_k items with added 'rerank_score'.

        If the model is not loaded, returns the first top_k items unchanged.
        Automatically chunks long queries to prevent VRAM issues.
        """
        if not items:
            return []
        if not self.is_loaded():
            # Graceful no-op fallback
            return items[:top_k]

        assert self.model is not None  # for type checkers
        try:
            # Check if query needs chunking
            if should_chunk_text(query, self.max_chunk_size, tokenizer=self.tokenizer):
                logger.info(f"Query length ({len(self.tokenizer.encode(query))} tokens) exceeds chunk size ({self.max_chunk_size}), using chunked reranking")
                scores = self._rerank_with_chunked_query(query, items)
            else:
                # Use original single-pass approach for short queries
                pairs = self._build_pairs(query, items)
                scores = self.model.predict(
                    pairs,
                    batch_size=1,  # Always 1 for memory efficiency
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                scores = list(scores)

            # Attach scores and sort descending
            for item, score in zip(items, scores):
                item["rerank_score"] = float(score)
            items_sorted = sorted(items, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return items_sorted[:top_k]
        except Exception as exc:
            logger.error(f"RerankService.rerank failed; returning unreranked results: {exc}")
            return items[:top_k]


