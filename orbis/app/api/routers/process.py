"""
Agentic RAG API Router for Orbis

Provides endpoints for the agentic RAG system that includes:
- Project detection
- Ticket scope and intent analysis
- Multi-modal search
- Documentation aggregation
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_agentic_rag_orchestrator, require_api_key
from engine.agents.orchestrator import AgenticRAGOrchestrator
from engine.schemas import AgenticRAGRequest, AgenticRAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["analysis"])


@router.post("/process", response_model=AgenticRAGResponse, dependencies=[Depends(require_api_key)])
async def process_content_agentic(
    request: AgenticRAGRequest,
    orchestrator: AgenticRAGOrchestrator = Depends(get_agentic_rag_orchestrator),
):
    """
    Process content using the complete agentic RAG workflow.

    This endpoint processes content through:
    1. Project detection (non-LLM pattern matching)
    2. Content Scope Analysis: Scope and intent analysis with documentation context
    3. Multi-modal search across different source types
    4. Documentation Aggregation: Documentation aggregation and final summary

    The response includes the complete analysis pipeline results with
    actionable recommendations and source references.
    """
    try:
        logger.debug("DEBUG: Checking orchestrator configuration...")
        if not orchestrator.is_configured():
            logger.error("❌ Agentic RAG system not configured")
            raise HTTPException(
                status_code=503,
                detail="Agentic RAG system is not properly configured. Please check service health."
            )

        logger.info(f"Processing: {request.content[:60]}...")

        # Process the content through the complete agentic RAG workflow
        response = await orchestrator.process_content(request)

        logger.info(f"Completed: {response.processing_time_ms}ms, confidence {response.overall_confidence:.2f}, {len(response.referenced_sources)} sources")

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to process agentic RAG request: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process agentic RAG request: {str(exc)}"
        ) from exc


