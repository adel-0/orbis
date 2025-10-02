import logging
from collections.abc import Sequence
from typing import Any

from utils.constants import DEFAULT_RERANK_BATCH_SIZE, DEFAULT_RERANK_MODEL

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
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "auto",
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model: CrossEncoder | None = None  # type: ignore

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
            init_kwargs: dict[str, Any] = {"device": self._device}
            if self.max_length is not None:
                init_kwargs["max_length"] = self.max_length
            self.model = CrossEncoder(self.model_name, **init_kwargs)
            logger.info(
                f"RerankService loaded model '{self.model_name}' on device '{self._device}'"
            )
        except Exception as exc:
            logger.error(f"Failed to load reranker model '{self.model_name}': {exc}")
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_pairs(self, query: str, items: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for item in items:
            # Prefer precomputed concatenated text; fall back to available fields
            candidate_text = item.get("concatenated_text", "")

            # If no concatenated text, try to build from content object
            if not candidate_text:
                content_obj = item.get("content")
                if content_obj is not None:
                    # Best-effort string build from any content type
                    parts: list[str] = []

                    # Common content fields
                    for field_name in ['title', 'description', 'content', 'extracted_text']:
                        field_value = getattr(content_obj, field_name, None)
                        if field_value:
                            parts.append(str(field_value))

                    # Handle comments (if it's a list)
                    comments = getattr(content_obj, "comments", None)
                    if comments:
                        try:
                            if isinstance(comments, list):
                                parts.extend([str(c) for c in comments])
                            else:
                                parts.append(str(comments))
                        except Exception:
                            pass

                    candidate_text = " ".join(parts)

            # Ensure we have some text, even if empty
            if not candidate_text:
                candidate_text = ""

            pairs.append((query, str(candidate_text)))
        return pairs

    def rerank(self, query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Score and return the top_k items with added 'rerank_score'.

        If the model is not loaded, returns the first top_k items unchanged.
        """
        if not items:
            return []
        if not self.is_loaded():
            # Graceful no-op fallback
            return items[:top_k]

        assert self.model is not None  # for type checkers
        try:
            pairs = self._build_pairs(query, items)
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # Attach scores and sort descending
            for item, score in zip(items, list(scores), strict=False):
                item["rerank_score"] = float(score)
            items_sorted = sorted(items, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return items_sorted[:top_k]
        except Exception as exc:
            logger.error(f"RerankService.rerank failed; returning unreranked results: {exc}")
            return items[:top_k]


