"""
Simplified Cross-Collection Reranking Service

Single-stage reranking with optional diversity filtering.
"""

import logging
from typing import Any

from core.schemas import ScopeAnalysisResult, SearchResult
from orbis_core.search import RerankService

logger = logging.getLogger(__name__)


def apply_diversity_filter(results: list[SearchResult], max_per_type: int = 3) -> list[SearchResult]:
    """
    Simple post-processing diversity filter.
    Limits results per content type to prevent over-representation.
    """
    if not results or max_per_type <= 0:
        return results
    
    type_counts = {}
    filtered_results = []
    
    for result in results:
        content_type = result.content_type
        current_count = type_counts.get(content_type, 0)
        
        if current_count < max_per_type:
            filtered_results.append(result)
            type_counts[content_type] = current_count + 1
    
    logger.debug(f"Diversity filter: {dict(type_counts)} across {len(filtered_results)} results")
    return filtered_results


class CrossCollectionReranker:
    """
    Simplified cross-collection reranker with single-stage processing.
    """
    
    def __init__(self, rerank_service: RerankService | None = None):
        self.rerank_service = rerank_service or RerankService()
    
    async def rerank_cross_collection_results(self,
                                            all_results: list[SearchResult],
                                            query: str,
                                            scope_analysis: ScopeAnalysisResult | None = None,
                                            target_count: int = 5,
                                            enable_diversity: bool = True) -> list[SearchResult]:
        """
        Single-stage reranking with optional scope boost and diversity filter.
        """
        if not all_results:
            return []

        logger.info(f"Reranking {len(all_results)} results")

        try:
            # Single rerank step
            if self.rerank_service.is_loaded():
                reranked_results = await self._rerank_with_service(all_results, query)
            else:
                reranked_results = all_results

            # Apply scope boost if available
            if scope_analysis:
                self._apply_scope_boost(reranked_results, scope_analysis)

            # Sort by final score
            sorted_results = sorted(reranked_results,
                                  key=lambda x: getattr(x, 'final_score', x.similarity_score),
                                  reverse=True)

            # Apply diversity filter if enabled
            if enable_diversity:
                final_results = apply_diversity_filter(sorted_results, max_per_type=target_count // 2 + 1)
            else:
                final_results = sorted_results

            logger.info(f"Reranking completed: {len(final_results)} final results")
            return final_results[:target_count]

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return sorted(all_results, key=lambda x: x.similarity_score, reverse=True)[:target_count]
    
    async def _rerank_with_service(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Apply reranking service using content.get_rerank_text()"""
        candidates = []
        for result in results:
            candidate = {
                'concatenated_text': result.content.get_rerank_text(),
                'content': result.content
            }
            candidates.append(candidate)
        
        reranked_candidates = self.rerank_service.rerank(
            query=query,
            items=candidates,
            top_k=len(candidates)
        )
        
        # Update results with rerank scores
        for i, candidate in enumerate(reranked_candidates):
            if i < len(results):
                results[i].rerank_score = candidate.get('rerank_score', 0.0)
                results[i].final_score = results[i].rerank_score
        
        return results
    
    def _apply_scope_boost(self, results: list[SearchResult], scope_analysis: ScopeAnalysisResult) -> None:
        """Apply simple scope boost to rerank scores"""
        if not scope_analysis.recommended_source_types:
            return
        
        for result in results:
            boost = 1.0
            if hasattr(result.content, 'source_type') and result.content.source_type:
                if result.content.source_type in scope_analysis.recommended_source_types:
                    boost = 1.2  # Simple 20% boost for recommended sources
            
            current_score = getattr(result, 'final_score', result.similarity_score)
            result.final_score = current_score * boost
    
    def is_configured(self) -> bool:
        """Check if reranking is available"""
        return True