import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_embedding_service, get_vector_service, get_bm25_service, require_api_key
from models.schemas import EmbedRequest, EmbedResponse
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from orbis_core.search import BM25Service
from services.work_item_service import WorkItemService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["embedding"])


def _load_tickets():
    with WorkItemService() as wi_service:
        return wi_service.get_tickets_from_all_sources()


@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
async def embed_tickets(
    request: EmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: VectorService = Depends(get_vector_service),
    bm25_service: BM25Service = Depends(get_bm25_service),
):
    try:
        # Validate at least one operation is requested
        if request.skip_embedding and request.skip_bm25_indexing:
            return EmbedResponse(
                message="No operation requested. At least one of skip_embedding or skip_bm25_indexing must be false",
                total_tickets=0,
                processed_tickets=0,
                success=False,
            )

        tickets = _load_tickets()
        if not tickets:
            return EmbedResponse(
                message="No tickets found to embed",
                total_tickets=0,
                processed_tickets=0,
                success=False,
            )

        messages = []
        embedding_result = None

        # Handle vector embeddings
        if not request.skip_embedding:
            collection_info = vector_service.get_collection_info()
            if collection_info.get("total_tickets", 0) > 0 and not request.force_rebuild:
                messages.append("Embeddings already exist (use force_rebuild=true to rebuild)")
                embedding_result = {
                    "total_tickets": collection_info["total_tickets"],
                    "processed_tickets": collection_info["total_tickets"],
                    "success": True,
                }
            else:
                embedding_result = await embedding_service.generate_embeddings(
                    tickets=tickets,
                    vector_service=vector_service,
                    batch_size=None,
                    clear_existing=request.force_rebuild,
                )
                messages.append(embedding_result.get("message", "Embeddings generated"))

        # Handle BM25 indexing
        if not request.skip_bm25_indexing:
            # Check if index exists and force_rebuild is False
            if bm25_service.is_initialized() and not request.force_rebuild:
                messages.append("BM25 index already exists (use force_rebuild=true to rebuild)")
            else:
                # Clear existing index if force_rebuild
                if request.force_rebuild and bm25_service.is_initialized():
                    bm25_service.clear_index()
                    logger.info("Cleared existing BM25 index")

                # Prepare documents with concatenated text for BM25
                bm25_docs = []
                for ticket in tickets:
                    concatenated_text = vector_service._concatenate_ticket_text(ticket)
                    bm25_docs.append({
                        'ticket': ticket,
                        'concatenated_text': concatenated_text
                    })
                bm25_service.index_documents(bm25_docs, save_to_disk=True)
                logger.info(f"Indexed {len(tickets)} documents for BM25 search")
                messages.append(f"BM25 index generated ({len(tickets)} documents)")

        # Combine results
        total_tickets = len(tickets)
        processed_tickets = embedding_result.get("processed_tickets", total_tickets) if embedding_result else total_tickets
        success = embedding_result.get("success", True) if embedding_result else True

        return EmbedResponse(
            message=" | ".join(messages),
            total_tickets=total_tickets,
            processed_tickets=processed_tickets,
            success=success,
        )
    except Exception as exc:
        logger.error(f"Failed to embed tickets: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to embed tickets: {str(exc)}")


