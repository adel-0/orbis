import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_embedding_service,
    get_vector_service,
    get_summary_service,
    get_rerank_service,
    get_bm25_service,
    get_hybrid_search_service,
    require_api_key,
)
from models.schemas import SearchRequest, SearchResponse
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.summary_service import SummaryService
from orbis_core.search import RerankService, BM25Service, HybridSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["search"])


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search_tickets(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: VectorService = Depends(get_vector_service),
    summary_service: SummaryService = Depends(get_summary_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    bm25_service: BM25Service = Depends(get_bm25_service),
    hybrid_search_service: HybridSearchService = Depends(get_hybrid_search_service),
):
    try:
        collection_info = vector_service.get_collection_info()
        if collection_info.get("total_tickets", 0) == 0:
            raise HTTPException(status_code=400, detail="No embeddings found. Please run /embed first.")

        # Build where clause from filters
        where_clause = vector_service.build_where_clause(request)

        # Fetch candidates for hybrid search: min(10 * top_k, 100)
        fetch_k = min(10 * request.top_k, 100)

        # 1. Semantic search
        query_embedding = embedding_service.encode_single_text(request.query)
        semantic_candidates = vector_service.search_candidates(
            query_embedding, n_results=fetch_k, where=where_clause
        )

        # 2. BM25 keyword search (on full corpus)
        if not bm25_service.is_initialized():
            raise HTTPException(status_code=400, detail="BM25 index not initialized. Please run /embed first.")

        keyword_candidates = bm25_service.search(request.query, top_k=fetch_k)

        # 3. Apply same filters to BM25 results
        if where_clause:
            keyword_candidates = vector_service.filter_results(keyword_candidates, request)

        # 4. Hybrid combination with dynamic weighting and recency boosting
        hybrid_results = hybrid_search_service.combine_results(
            semantic_results=semantic_candidates,
            keyword_results=keyword_candidates,
            query=request.query,
            top_k=fetch_k
        )

        # 5. Optionally rerank and blend with RRF scores
        if request.enable_reranking and rerank_service and rerank_service.is_loaded():
            # Get rerank scores for all hybrid results
            reranked_results = rerank_service.rerank(request.query, hybrid_results, top_k=len(hybrid_results))

            # Create lookup for rerank scores
            rerank_lookup = {r['ticket'].id: r.get('rerank_score', 0.0) for r in reranked_results}

            # Normalize RRF scores to [0,1] using min-max
            rrf_scores = [r.get('rrf_score', 0.0) for r in hybrid_results]
            min_rrf = min(rrf_scores) if rrf_scores else 0.0
            max_rrf = max(rrf_scores) if rrf_scores else 1.0
            rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

            # Blend scores: 25% rerank, 75% RRF
            alpha = 0.25  # Weight for rerank score
            for result in hybrid_results:
                ticket_id = result['ticket'].id

                # Normalize RRF score to [0,1]
                rrf_norm = (result.get('rrf_score', 0.0) - min_rrf) / rrf_range

                # Get rerank score (already in [0,1])
                rerank_score = rerank_lookup.get(ticket_id, 0.0)

                # Blend scores
                blended_score = alpha * rerank_score + (1 - alpha) * rrf_norm

                # Store both scores
                result['rerank_score'] = rerank_score
                result['blended_score'] = blended_score

            # Sort by blended score
            hybrid_results.sort(key=lambda x: x.get('blended_score', 0.0), reverse=True)
            results = hybrid_results[:request.top_k]
        else:
            # No reranking - use RRF results directly
            results = hybrid_results[:request.top_k]

        summary = None
        if request.include_summary and summary_service and summary_service.is_configured():
            tickets = [r["ticket"] for r in results]
            # Use blended score if available (when reranking enabled), otherwise confidence score
            relevancy_scores = [
                r.get("blended_score") or r.get("confidence_score") or r.get("rrf_score", 0.0)
                for r in results
            ]
            summary = summary_service.generate_summary(
                request.query, tickets, similarity_scores=relevancy_scores
            )

        # Build filter summary for response
        filters_applied = {}
        if request.source_names:
            filters_applied["source_names"] = request.source_names
        if request.organizations:
            filters_applied["organizations"] = request.organizations
        if request.projects:
            filters_applied["projects"] = request.projects
        if request.area_path_prefix:
            filters_applied["area_path_prefix"] = request.area_path_prefix
        if request.area_path_contains:
            filters_applied["area_path_contains"] = request.area_path_contains
        if request.iteration_path_prefix:
            filters_applied["iteration_path_prefix"] = request.iteration_path_prefix
        if request.iteration_path_contains:
            filters_applied["iteration_path_contains"] = request.iteration_path_contains

        return SearchResponse(
            results=results,
            summary=summary,
            total_results=len(results),
            query=request.query,
            filters_applied=filters_applied if filters_applied else None
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to search tickets: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to search tickets: {str(exc)}")


