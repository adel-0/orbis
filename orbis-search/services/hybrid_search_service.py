"""
Hybrid search service combining semantic (vector) and keyword (BM25) search
using Reciprocal Rank Fusion (RRF) for ranking and confidence scoring.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Service for hybrid search combining semantic and keyword search"""

    def __init__(self):
        pass

    def _calculate_recency_boost(self, created_date: Optional[datetime],
                                  max_boost: float = 0.05) -> float:
        """
        Calculate recency boost for a ticket based on creation date.

        Tickets within 30 days get max boost, then linear gradient down to 2 years.
        No boost beyond 2 years.

        Args:
            created_date: Creation date of the ticket (assumed UTC)
            max_boost: Maximum boost to apply (default 0.05 = 5%)

        Returns:
            Boost factor between 0.0 and max_boost
        """
        if not created_date:
            return 0.0

        # Assume UTC for simplicity
        now = datetime.now(timezone.utc)
        if created_date.tzinfo is None:
            created_date = created_date.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_days = (now - created_date).days

        if age_days < 0:
            # Future date, no boost
            return 0.0

        # Gradient boost calculation:
        # 0-30 days: max_boost (full boost)
        # 30-730 days (2 years): linear decay from max_boost to 0
        # 730+ days: no boost

        if age_days <= 30:
            # Full boost for recent tickets
            return max_boost
        elif age_days <= 730:
            # Linear decay from max_boost to 0 over 700 days (30 to 730)
            # Formula: max_boost * (1 - (age_days - 30) / 700)
            boost = max_boost * (1.0 - (age_days - 30) / 700.0)
            return max(0.0, boost)
        else:
            # No boost for old tickets
            return 0.0

    def combine_results(self,
                       semantic_results: List[Dict[str, Any]],
                       keyword_results: List[Dict[str, Any]],
                       query: str,
                       top_k: int = 50,
                       rrf_k: int = 60,
                       confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF)
        for ranking and cosine similarity + recency boost for confidence scoring.

        Args:
            semantic_results: Results from vector search with 'similarity_score'
            keyword_results: Results from BM25 search with 'bm25_score'
            query: Original query string (unused in RRF, kept for API compatibility)
            top_k: Number of top results to return after ranking and filtering
            rrf_k: The 'k' parameter for RRF (default 60, recommended by research)
            confidence_threshold: Minimum confidence score to include in results (default 0.3)

        Returns:
            Combined, reranked, and filtered results with 'rrf_score' and 'confidence_score'
        """
        # Create rank lookups: rank = position in list (1-indexed)
        semantic_ranks = {r['ticket'].id: idx + 1 for idx, r in enumerate(semantic_results)}
        keyword_ranks = {r['ticket'].id: idx + 1 for idx, r in enumerate(keyword_results)}

        # Create separate lookups for semantic and keyword results to preserve scores
        semantic_lookup = {r['ticket'].id: r for r in semantic_results}
        keyword_lookup = {r['ticket'].id: r for r in keyword_results}

        # Get all unique ticket IDs
        all_ticket_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Combine using RRF for ranking
        combined_results = []
        for ticket_id in all_ticket_ids:
            # Get results from both lookups (may be in one or both)
            sem_result = semantic_lookup.get(ticket_id)
            kw_result = keyword_lookup.get(ticket_id)

            # Get ticket object (prefer semantic, fallback to keyword)
            ticket = sem_result['ticket'] if sem_result else kw_result['ticket']

            # Calculate RRF score: sum of 1/(k + rank) for each ranker
            rrf_score = 0.0
            if ticket_id in semantic_ranks:
                rrf_score += 1.0 / (rrf_k + semantic_ranks[ticket_id])
            if ticket_id in keyword_ranks:
                rrf_score += 1.0 / (rrf_k + keyword_ranks[ticket_id])

            # Get raw cosine similarity (already normalized to [0,1])
            cosine_similarity = sem_result.get('similarity_score', 0.0) if sem_result else 0.0

            # Get BM25 score (from keyword results)
            bm25_score = kw_result.get('bm25_score', 0.0) if kw_result else 0.0

            # Calculate recency boost
            recency_boost = self._calculate_recency_boost(ticket.created_date)

            # Calculate confidence score: cosine + recency (no BM25)
            confidence_score = cosine_similarity + recency_boost
            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]

            # Get concatenated_text (prefer semantic, fallback to keyword)
            concatenated_text = (
                sem_result.get('concatenated_text', '') if sem_result
                else kw_result.get('concatenated_text', '') if kw_result
                else ''
            )

            combined_results.append({
                'ticket': ticket,
                'rrf_score': rrf_score,
                'confidence_score': confidence_score,
                'similarity_score': cosine_similarity,
                'bm25_score': bm25_score,
                'recency_boost': recency_boost,
                'concatenated_text': concatenated_text
            })

        # 1. Sort by RRF score (determines ranking)
        combined_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        # 2. Filter by confidence score threshold
        filtered_results = [r for r in combined_results if r['confidence_score'] >= confidence_threshold]

        # 3. Return top_k of filtered results
        return filtered_results[:top_k]