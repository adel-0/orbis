"""
Simplified Agentic RAG Orchestrator for Orbis

Clean 3-step pipeline following 2024/2025 RAG best practices:
1. Query Analysis - Understand intent and determine sources
2. Parallel Retrieval - Search recommended sources
3. Response Synthesis - Generate final answer

Replaces the complex 5-step pipeline with a simpler, more maintainable approach.
"""

import logging
import time

from config.settings import settings
from engine.agents.documentation_aggregator import DocumentationAggregator
from engine.agents.query_analyzer import QueryAnalyzer
from engine.schemas import (
    AgenticRAGRequest,
    AgenticRAGResponse,
    ProjectContext,
    ScopeAnalysisResult,
    SourceReference,
)
from engine.services.generic_multi_modal_search import GenericMultiModalSearch
from orbis_core.llm.openai_client import OpenAIClientService
from orbis_core.search import RerankService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    """
    Simplified RAG orchestrator - 3 steps: Analyze → Retrieve → Synthesize
    """

    def __init__(self,
                 vector_service: GenericVectorService | None = None,
                 embedding_service: EmbeddingService | None = None,
                 rerank_service: RerankService | None = None,
                 openai_client_service: OpenAIClientService | None = None):

        # Shared OpenAI client
        openai_client = openai_client_service or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Initialize simple 3-step pipeline
        self.query_analyzer = QueryAnalyzer(openai_client=openai_client)
        self.search_service = GenericMultiModalSearch(
            vector_service=vector_service,
            embedding_service=embedding_service,
            rerank_service=rerank_service
        )
        self.synthesizer = DocumentationAggregator(openai_client_service=openai_client)

        logger.info("Simplified Agentic RAG Orchestrator initialized (3-step pipeline)")

    def is_configured(self) -> bool:
        """Check if orchestrator is properly configured"""
        try:
            return (
                self.query_analyzer.is_configured() and
                self.search_service.is_configured() and
                self.synthesizer.is_configured()
            )
        except Exception as e:
            logger.error(f"Configuration check failed: {e}")
            return False

    async def process_content(self, request: AgenticRAGRequest) -> AgenticRAGResponse:
        """
        Process request through simplified 3-step pipeline.

        Args:
            request: Request with content and optional metadata

        Returns:
            Complete RAG response with analysis and sources
        """
        start_time = time.time()
        logger.info(f"Processing: {request.content[:80]}...")

        try:
            # STEP 1: Query Analysis (replaces project detection + scope + routing)
            logger.info("Step 1: Analyzing query...")
            analysis = await self.query_analyzer.analyze(request)

            logger.info(f"Analysis: project={analysis.project_code}, "
                       f"intent='{analysis.intent[:50]}...', "
                       f"sources={analysis.recommended_sources}, "
                       f"confidence={analysis.confidence:.2f}")

            # STEP 2: Parallel Multi-Source Retrieval
            logger.info("Step 2: Retrieving from sources...")
            search_results = await self.search_service.search(
                query=request.content,
                source_types=analysis.recommended_sources,
                filters=self._build_filters(analysis),
                top_k=20
            )

            logger.info(f"Retrieved: {search_results.total_results} results "
                       f"from {len(search_results.collections_searched)} collections")

            # STEP 3: Response Synthesis
            logger.info("Step 3: Synthesizing response...")

            # Create a scope analysis for backward compatibility with synthesizer
            scope_analysis = ScopeAnalysisResult(
                scope_description=f"Project: {analysis.project_code or 'general'}",
                intent_description=analysis.intent,
                confidence=analysis.confidence,
                recommended_source_types=analysis.recommended_sources
            )

            final_summary, source_references, overall_confidence = await self.synthesizer.aggregate_and_summarize(
                original_content=request.content,
                scope_analysis=scope_analysis,
                search_results=search_results
            )

            # Build response
            processing_time = int((time.time() - start_time) * 1000)

            response = AgenticRAGResponse(
                project_context=ProjectContext(project_code=analysis.project_code),
                scope_analysis=scope_analysis,
                final_summary=final_summary,
                referenced_sources=source_references,
                overall_confidence=overall_confidence,
                processing_time_ms=processing_time
            )

            logger.info(f"Completed in {processing_time}ms: "
                       f"{len(source_references)} sources, "
                       f"confidence {overall_confidence:.2f}")
            return response

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return self._create_error_response(
                f"Processing failed: {str(e)}",
                start_time
            )

    def _build_filters(self, analysis) -> dict:
        """Build search filters from analysis"""
        filters = {}

        # Add project filter if detected
        if analysis.project_code:
            # Get wiki repos for this project
            wiki_repos = self.query_analyzer.get_project_wiki_repos(analysis.project_code)
            if wiki_repos:
                filters["wiki_repos"] = wiki_repos

        return filters if filters else None

    def _create_error_response(self, error_message: str, start_time: float) -> AgenticRAGResponse:
        """Create error response"""
        processing_time = int((time.time() - start_time) * 1000)

        return AgenticRAGResponse(
            project_context=ProjectContext(project_code=None),
            scope_analysis=ScopeAnalysisResult(
                scope_description="Error occurred",
                intent_description="Unable to process",
                confidence=0.0,
                recommended_source_types=[]
            ),
            final_summary=f"**Error**\n\n{error_message}\n\nPlease try again or contact support.",
            referenced_sources=[],
            overall_confidence=0.0,
            processing_time_ms=processing_time
        )
