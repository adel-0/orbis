"""
Cross-encoder based reranking service for query-document pairs.
Uses local BGE reranker model for refining search candidates.
"""
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class RerankService:
    """Cross-encoder based reranker for query-document pairs."""

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
        device: str = "auto",
        max_length: int | None = None,
        max_chunk_size: int = 384,
    ) -> None:
        self.model_name = model_name
        self._device = self._resolve_device(device)
        self.max_length = max_length
        self.max_chunk_size = max_chunk_size
        self.model: Any | None = None
        self.tokenizer: Any | None = None

        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            logger.warning(
                "sentence-transformers not available; RerankService will be disabled."
            )
            return

        try:
            init_kwargs: dict[str, Any] = {"device": self._device}
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

    def get_tokenizer(self):
        """Expose tokenizer for text chunking operations"""
        return self.tokenizer

    def _build_pairs(self, query: str, items: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
        """Build query-document pairs for reranking"""
        pairs: list[tuple[str, str]] = []
        for item in items:
            # Get concatenated text from item
            candidate_text = item.get("concatenated_text", "")
            if not candidate_text:
                # Fallback: try to build from ticket/document fields
                doc = item.get("ticket") or item.get("document", {})
                if isinstance(doc, dict):
                    parts = []
                    for field in ["title", "description", "content"]:
                        if doc.get(field):
                            parts.append(str(doc[field]))
                    candidate_text = " ".join(parts)
                else:
                    # Try object attributes
                    parts = []
                    for attr in ["title", "description", "content"]:
                        val = getattr(doc, attr, None)
                        if val:
                            parts.append(str(val))
                    candidate_text = " ".join(parts)

            pairs.append((query, str(candidate_text)))
        return pairs

    def _rerank_with_chunked_query(self, query: str, items: list[dict[str, Any]]) -> list[float]:
        """Rerank items using chunked query for long queries"""
        from ..embedding.text_chunking import chunk_text

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

        # Process each query chunk independently
        for q_chunk_idx, q_chunk in enumerate(query_chunks):
            try:
                q_chunk_weight = query_chunk_weights[q_chunk_idx] / total_query_weight

                # Process each item individually
                predict_call_count = 0  # Track calls to clear cache frequently
                for item_idx in range(len(items)):
                    item = items[item_idx]
                    candidate_text = item.get("concatenated_text", "")

                    if not candidate_text:
                        # Build from available fields
                        doc = item.get("ticket") or item.get("document", {})
                        parts = []
                        if isinstance(doc, dict):
                            for field in ["title", "description", "content"]:
                                if doc.get(field):
                                    parts.append(str(doc[field]))
                        else:
                            for attr in ["title", "description", "content"]:
                                val = getattr(doc, attr, None)
                                if val:
                                    parts.append(str(val))
                        candidate_text = " ".join(parts)

                    candidate_chunks = list(chunk_text(str(candidate_text), self.max_chunk_size, tokenizer=self.tokenizer))

                    # Calculate candidate chunk weights
                    cand_chunk_weights = [len(self.tokenizer.encode(c)) for c in candidate_chunks]
                    total_cand_weight = sum(cand_chunk_weights)

                    if total_cand_weight == 0:
                        continue

                    # Score all (query_chunk, candidate_chunk) pairs
                    item_score_for_q_chunk = 0.0
                    for c_chunk_idx, c_chunk in enumerate(candidate_chunks):
                        c_chunk_weight = cand_chunk_weights[c_chunk_idx] / total_cand_weight

                        # Score single pair
                        single_pair = [(q_chunk, c_chunk)]
                        pair_score = self.model.predict(
                            single_pair,
                            batch_size=1,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )[0]

                        # Weighted contribution
                        item_score_for_q_chunk += pair_score * c_chunk_weight

                        predict_call_count += 1

                        # Clear CUDA cache every 8 predict calls to prevent VRAM accumulation
                        if predict_call_count % 8 == 0:
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except:
                                pass

                    # Add weighted score from this query chunk
                    final_scores[item_idx] += item_score_for_q_chunk * q_chunk_weight

                # Final CUDA cache clear for this query chunk
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

                # Log progress
                if len(query_chunks) > 3:
                    logger.info(f"Processed query chunk {q_chunk_idx+1}/{len(query_chunks)}")

            except Exception as exc:
                logger.error(f"Failed to process query chunk {q_chunk_idx+1}: {exc}")

        return final_scores

    def rerank(self, query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Score and return the top_k items with added 'rerank_score'."""
        from ..embedding.text_chunking import should_chunk_text

        if not items:
            return []
        if not self.is_loaded():
            # Graceful fallback
            return items[:top_k]

        assert self.model is not None

        try:
            # Log VRAM usage before reranking
            try:
                import torch
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"VRAM before reranking: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            except:
                pass

            # Check if query needs chunking
            if should_chunk_text(query, self.max_chunk_size, tokenizer=self.tokenizer):
                logger.info(f"Query length ({len(self.tokenizer.encode(query))} tokens) exceeds chunk size, using chunked reranking")
                scores = self._rerank_with_chunked_query(query, items)
            else:
                # Use batch_size=1 to minimize VRAM usage
                logger.info(f"Reranking {len(items)} items with batch_size=1")
                pairs = self._build_pairs(query, items)

                # Process in smaller chunks to avoid VRAM accumulation
                chunk_size = 16
                all_scores = []
                for i in range(0, len(pairs), chunk_size):
                    chunk_pairs = pairs[i:i+chunk_size]
                    chunk_scores = self.model.predict(
                        chunk_pairs,
                        batch_size=1,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    all_scores.extend(list(chunk_scores))

                    # Clear CUDA cache after each chunk
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass

                scores = all_scores

            # Log VRAM usage after reranking
            try:
                import torch
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"VRAM after reranking: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                    torch.cuda.empty_cache()
            except:
                pass

            # Attach scores and sort
            for item, score in zip(items, scores):
                item["rerank_score"] = float(score)
            items_sorted = sorted(items, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return items_sorted[:top_k]
        except Exception as exc:
            logger.error(f"RerankService.rerank failed; returning unreranked results: {exc}")
            return items[:top_k]
