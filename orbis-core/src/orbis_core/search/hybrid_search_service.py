"""
Hybrid search service combining semantic (vector) and keyword (BM25) search
using Reciprocal Rank Fusion (RRF) for ranking and confidence scoring.
"""
import logging
from typing import Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Service for hybrid search combining semantic and keyword search"""

    def __init__(self):
        pass

    def _calculate_recency_boost(self, created_date: datetime | None,
                                  max_boost: float = 0.05) -> float:
        """
        Calculate recency boost based on creation date.

        Tickets within 30 days get max boost, then linear gradient down to 2 years.
        No boost beyond 2 years.

        Args:
            created_date: Creation date (assumed UTC)
            max_boost: Maximum boost to apply (default 0.05 = 5%)

        Returns:
            Boost factor between 0.0 and max_boost
        """
        if not created_date:
            return 0.0

        # Assume UTC
        now = datetime.now(timezone.utc)
        if created_date.tzinfo is None:
            created_date = created_date.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_days = (now - created_date).days

        if age_days < 0:
            return 0.0

        # Gradient boost calculation
        if age_days <= 30:
            return max_boost
        elif age_days <= 730:
            # Linear decay from max_boost to 0 over 700 days
            boost = max_boost * (1.0 - (age_days - 30) / 700.0)
            return max(0.0, boost)
        else:
            return 0.0

    def combine_results(self,
                       semantic_results: list[dict[str, Any]],
                       keyword_results: list[dict[str, Any]],
                       query: str,
                       top_k: int = 50,
                       rrf_k: int = 60,
                       confidence_threshold: float = 0.3,
                       id_field: str = "id") -> list[dict[str, Any]]:
        """
        Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF).

        Args:
            semantic_results: Results from vector search with 'similarity_score'
            keyword_results: Results from BM25 search with 'bm25_score'
            query: Original query string (kept for API compatibility)
            top_k: Number of top results to return
            rrf_k: The 'k' parameter for RRF (default 60)
            confidence_threshold: Minimum confidence score (default 0.3)
            id_field: Field name to use for document ID (default "id")

        Returns:
            Combined, reranked, and filtered results with 'rrf_score' and 'confidence_score'
        """
        # Helper to get document ID
        def get_doc_id(result: dict[str, Any]) -> Any:
            # Try to get from ticket/document object
            doc = result.get("ticket") or result.get("document")
            if doc:
                if isinstance(doc, dict):
                    return doc.get(id_field)
                else:
                    return getattr(doc, id_field, None)
            # Fallback to result dict
            return result.get(id_field)

        # Helper to get created date
        def get_created_date(result: dict[str, Any]) -> datetime | None:
            doc = result.get("ticket") or result.get("document")
            if doc:
                if isinstance(doc, dict):
                    date_val = doc.get("created_date") or doc.get("created_at")
                else:
                    date_val = getattr(doc, "created_date", None) or getattr(doc, "created_at", None)
                if isinstance(date_val, datetime):
                    return date_val
            return None

        # Create rank lookups
        semantic_ranks = {get_doc_id(r): idx + 1 for idx, r in enumerate(semantic_results) if get_doc_id(r) is not None}
        keyword_ranks = {get_doc_id(r): idx + 1 for idx, r in enumerate(keyword_results) if get_doc_id(r) is not None}

        # Create lookups for results
        semantic_lookup = {get_doc_id(r): r for r in semantic_results if get_doc_id(r) is not None}
        keyword_lookup = {get_doc_id(r): r for r in keyword_results if get_doc_id(r) is not None}

        # Get all unique document IDs
        all_doc_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Combine using RRF
        combined_results = []
        for doc_id in all_doc_ids:
            sem_result = semantic_lookup.get(doc_id)
            kw_result = keyword_lookup.get(doc_id)

            # Get document object (prefer semantic)
            doc = (sem_result.get("ticket") or sem_result.get("document")) if sem_result else (kw_result.get("ticket") or kw_result.get("document"))

            # Calculate RRF score
            rrf_score = 0.0
            if doc_id in semantic_ranks:
                rrf_score += 1.0 / (rrf_k + semantic_ranks[doc_id])
            if doc_id in keyword_ranks:
                rrf_score += 1.0 / (rrf_k + keyword_ranks[doc_id])

            # Get scores
            cosine_similarity = sem_result.get('similarity_score', 0.0) if sem_result else 0.0
            bm25_score = kw_result.get('bm25_score', 0.0) if kw_result else 0.0

            # Calculate recency boost
            created_date = get_created_date(sem_result if sem_result else kw_result)
            recency_boost = self._calculate_recency_boost(created_date)

            # Calculate confidence score
            confidence_score = cosine_similarity + recency_boost
            confidence_score = max(0.0, min(1.0, confidence_score))

            # Get concatenated text
            concatenated_text = (
                sem_result.get('concatenated_text', '') if sem_result
                else kw_result.get('concatenated_text', '') if kw_result
                else ''
            )

            # Build result
            result = {
                'rrf_score': rrf_score,
                'confidence_score': confidence_score,
                'similarity_score': cosine_similarity,
                'bm25_score': bm25_score,
                'recency_boost': recency_boost,
                'concatenated_text': concatenated_text
            }

            # Add document/ticket
            if doc:
                if "ticket" in (sem_result or kw_result):
                    result['ticket'] = doc
                else:
                    result['document'] = doc

            combined_results.append(result)

        # Sort by RRF score
        combined_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        # Filter by confidence threshold
        filtered_results = [r for r in combined_results if r['confidence_score'] >= confidence_threshold]

        # Return top_k
        return filtered_results[:top_k]
