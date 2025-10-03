"""
Agentic RAG Orchestrator for Orbis

Main workflow orchestrator that coordinates the agentic RAG system:
1. Project detection (non-LLM)
2. Content Scope Analysis: Scope and intent analysis with documentation context
3. Intelligent Routing: LLM-powered data source selection and routing
4. Multi-modal search based on routing recommendations and scope analysis
5. Documentation Aggregation: Documentation aggregation and final summary
"""

import logging
import time

from config.settings import settings
from core.agents.documentation_aggregator import DocumentationAggregator
from core.agents.llm_routing_agent import QueryRoutingAgent
from core.agents.scope_analyzer import ScopeAnalyzer
from core.schemas import (
    AgenticRAGRequest,
    AgenticRAGResponse,
    ProjectContext,
    ScopeAnalysisResult,
    SourceReference,
)
from core.services.generic_multi_modal_search import GenericMultiModalSearch
from core.services.project_detection import ProjectDetectionService
from orbis_core.llm.openai_client import OpenAIClientService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService
from infrastructure.storage.rerank_service import RerankService
from utils.constants import (
    ORCHESTRATOR_MIN_CONFIDENCE_THRESHOLD,
    ORCHESTRATOR_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

class AgenticRAGOrchestrator:
    """
    Main orchestrator for the agentic RAG system.

    Coordinates the complete workflow from content input to final summary,
    managing the interaction between all services and agents.
    """

    def __init__(self,
                 vector_service: GenericVectorService | None = None,
                 embedding_service: EmbeddingService | None = None,
                 rerank_service: RerankService | None = None,
                 openai_client_service: OpenAIClientService | None = None):
        # Initialize shared OpenAI client
        openai_client = openai_client_service or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Initialize all services with dependency injection support (now using generic services)
        self.project_detection = ProjectDetectionService()
        self.scope_analyzer = ScopeAnalyzer(
            vector_service=vector_service,
            embedding_service=embedding_service,
            openai_client_service=openai_client
        )
        
        # Initialize LLM routing agent for intelligent data source selection
        from infrastructure.data_processing.data_source_service import DataSourceService
        data_source_service = DataSourceService()
        self.routing_agent = QueryRoutingAgent(
            openai_client_service=openai_client,
            data_source_service=data_source_service
        )

        # Pass shared services to GenericMultiModalSearch to prevent duplicate initialization
        self.multi_search = GenericMultiModalSearch(
            vector_service=vector_service,
            embedding_service=embedding_service,
            rerank_service=rerank_service
        )

        self.doc_aggregator = DocumentationAggregator(openai_client_service=openai_client)

        # Orchestrator configuration
        self.config = {
            "timeout_seconds": ORCHESTRATOR_TIMEOUT_SECONDS,  # Total workflow timeout
            "min_confidence_threshold": ORCHESTRATOR_MIN_CONFIDENCE_THRESHOLD,  # Minimum confidence to proceed
            "enable_fallback": True,  # Enable fallback responses
            "log_intermediate_results": True,  # Log each step for debugging
        }

        logger.info("Agentic RAG Orchestrator initialized")

    def is_configured(self) -> bool:
        """
        Check if the Agentic RAG Orchestrator is properly configured.

        Returns:
            bool: True if all core services are properly configured
        """
        try:
            return (
                self.multi_search.is_configured() and
                self.scope_analyzer.is_configured() and
                self.doc_aggregator.is_configured()
            )
        except Exception as e:
            logger.error(f"Error checking orchestrator configuration: {e}")
            return False

    async def process_content(self, request: AgenticRAGRequest) -> AgenticRAGResponse:
        """
        Main method: Process content through the complete agentic RAG workflow.

        Args:
            request: AgenticRAGRequest containing content and metadata

        Returns:
            AgenticRAGResponse with complete analysis and recommendations
        """
        start_time = time.time()

        logger.info(f"Starting agentic RAG processing for content: {request.content[:100]}...")

        try:
            # Step 1: Project Detection
            logger.info("🔍 Step 1: Project detection")
            logger.info(f"🎫 Starting project detection for area_path: '{request.area_path}'")
            project_context = self._detect_project_context(request)

            if self.config["log_intermediate_results"]:
                if project_context:
                    logger.info(f"✅ Project detected: {project_context.project_code}")
                else:
                    logger.info("🔍 No specific project detected, using general context")

            # Step 2: Content Scope Analysis
            logger.info("🧠 Step 2: Content scope and intent analysis")
            logger.info(f"🎫 AGENTIC DECISION: Analyzing content scope with {'project-specific' if project_context else 'general'} documentation context")
            scope_analysis = await self._analyze_scope_and_intent(request, project_context)

            if not scope_analysis:
                logger.error("Content scope analysis failed")
                return self._create_error_response(
                    "Failed to analyze content scope and intent",
                    project_context,
                    start_time
                )

            if self.config["log_intermediate_results"]:
                logger.info("✅ Content scope analysis completed:")
                logger.info(f"  📋 Scope: {scope_analysis.scope_description[:100]}...")
                logger.info(f"  🎯 Intent: {scope_analysis.intent_description[:100]}...")
                logger.info(f"  📊 Confidence: {scope_analysis.confidence:.2f}")
                logger.info(f"  📚 Recommended sources: {scope_analysis.recommended_source_types}")
                logger.info(f"🧠 AGENTIC DECISION: Scope analyzer recommends searching {len(scope_analysis.recommended_source_types)} source types: {', '.join(scope_analysis.recommended_source_types)}")

            # Check if confidence is sufficient to proceed
            if scope_analysis.confidence < self.config["min_confidence_threshold"]:
                logger.warning(f"⚠️ Content scope analysis confidence ({scope_analysis.confidence:.2f}) below threshold ({self.config['min_confidence_threshold']})")
                logger.warning(f"🚨 AGENTIC DECISION: Low confidence detected - proceeding with {'fallback enabled' if self.config['enable_fallback'] else 'error response'}")
                if not self.config["enable_fallback"]:
                    logger.info("🚨 AGENTIC DECISION: Returning low confidence response due to disabled fallback")
                    return self._create_low_confidence_response(
                        request, project_context, scope_analysis, start_time
                    )
                else:
                    logger.info("🚨 AGENTIC DECISION: Continuing with fallback enabled despite low confidence")

            # Step 3: Intelligent Routing - Data Source Selection
            logger.info("🤖 Step 3: Intelligent routing - Data source selection")
            logger.info(f"🧠 AGENTIC DECISION: Analyzing query for intelligent data source routing")
            routing_recommendations = await self._get_routing_recommendations(request, scope_analysis, project_context)
            
            if self.config["log_intermediate_results"] and routing_recommendations:
                logger.info("✅ Routing analysis completed:")
                logger.info(f"  📊 Recommended sources: {len(routing_recommendations)}")
                for rec in routing_recommendations[:3]:  # Log top 3
                    logger.info(f"  • {rec.source_name} ({rec.source_type}): {rec.relevance_score:.2f} - {rec.reasoning[:80]}...")
                logger.info(f"🤖 AGENTIC DECISION: Routing agent recommends {len(routing_recommendations)} specific data source instances")

            # Step 4: Multi-Modal Search
            logger.info("🔎 Step 4: Multi-modal search")
            logger.info(f"🔍 AGENTIC DECISION: Initiating multi-modal search with intelligent routing recommendations")
            logger.debug("DEBUG: Starting multi-modal search with routing recommendations...")
            search_results = await self._perform_multi_modal_search(
                request.content,
                scope_analysis,
                project_context,
                routing_recommendations
            )

            if self.config["log_intermediate_results"]:
                logger.info("✅ Search completed:")
                logger.info(f"  🎫 Work items: {len(getattr(search_results, 'workitem_results', []))}")
                logger.info(f"  📖 Wiki pages: {len(getattr(search_results, 'wiki_results', []))}")
                logger.info(f"  💻 Code files: {len(getattr(search_results, 'code_results', []))}")
                logger.info(f"  📄 PDF docs: {len(getattr(search_results, 'pdf_results', []))}")
                logger.info(f"  📊 Total results: {search_results.total_results}")
                logger.info(f"  🔧 Collections searched: {search_results.collections_searched}")
                logger.info(f"🔍 AGENTIC DECISION: Search returned {search_results.total_results} total results across {len(search_results.collections_searched)} collections")
                if search_results.total_results == 0:
                    logger.warning("🔍 AGENTIC DECISION: No search results found - documentation aggregator will use fallback approach")
                else:
                    top_sources = []
                    if hasattr(search_results, 'workitem_results') and search_results.workitem_results:
                        top_sources.append(f"{len(search_results.workitem_results)} work items")
                    if hasattr(search_results, 'wiki_results') and search_results.wiki_results:
                        top_sources.append(f"{len(search_results.wiki_results)} wiki pages")
                    if hasattr(search_results, 'code_results') and search_results.code_results:
                        top_sources.append(f"{len(search_results.code_results)} code files")
                    if hasattr(search_results, 'pdf_results') and search_results.pdf_results:
                        top_sources.append(f"{len(search_results.pdf_results)} PDF docs")
                    logger.info(f"🔍 AGENTIC DECISION: Primary sources available: {', '.join(top_sources)}")

            # Step 5: Documentation Aggregation
            logger.info("📖 Step 4: Documentation aggregation")
            logger.info(f"📚 AGENTIC DECISION: Aggregating {search_results.total_results} search results with scope guidance for final synthesis")
            logger.debug("DEBUG: Starting documentation aggregator with search results...")
            final_summary, source_references, overall_confidence = await self._aggregate_documentation(
                request.content,
                scope_analysis,
                search_results
            )

            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

            if self.config["log_intermediate_results"]:
                logger.info("✅ Documentation aggregation completed:")
                logger.info(f"  📝 Summary length: {len(final_summary)} characters")
                logger.info(f"  📚 Source references: {len(source_references)}")
                logger.info(f"  📊 Overall confidence: {overall_confidence:.2f}")
                logger.info(f"  ⏱️ Processing time: {processing_time}ms")
                logger.info(f"📚 AGENTIC DECISION: Final synthesis completed with {len(source_references)} authoritative sources")
                logger.info(f"📚 AGENTIC DECISION: Overall pipeline confidence: {overall_confidence:.2f} (scope: {scope_analysis.confidence:.2f} → final: {overall_confidence:.2f})")
                if overall_confidence != scope_analysis.confidence:
                    confidence_change = overall_confidence - scope_analysis.confidence
                    direction = "increased" if confidence_change > 0 else "decreased"
                    logger.info(f"📚 AGENTIC DECISION: Confidence {direction} by {abs(confidence_change):.2f} during documentation synthesis")

            # Create final response
            logger.debug("DEBUG: Creating final AgenticRAGResponse...")
            response = AgenticRAGResponse(
                project_context=project_context or self._create_default_project_context(),
                scope_analysis=scope_analysis,
                final_summary=final_summary,
                referenced_sources=source_references,
                overall_confidence=overall_confidence,
                processing_time_ms=processing_time
            )

            logger.info(f"🎉 Agentic RAG processing completed successfully in {processing_time}ms")
            return response

        except Exception as e:
            logger.error(f"Error in agentic RAG processing: {e}")
            return self._create_error_response(
                f"Processing failed: {str(e)}",
                project_context if 'project_context' in locals() else None,
                start_time
            )

    def _detect_project_context(self, request: AgenticRAGRequest) -> ProjectContext | None:
        """Step 1: Detect project context using non-LLM pattern matching"""
        try:
            return self.project_detection.detect_project(
                area_path=request.area_path
            )
        except Exception as e:
            logger.error(f"Error in project detection: {e}")
            return None

    async def _analyze_scope_and_intent(self,
                                      request: AgenticRAGRequest,
                                      project_context: ProjectContext | None) -> ScopeAnalysisResult | None:
        """Step 2: Analyze scope and intent using scope analyzer"""
        try:
            return await self.scope_analyzer.analyze_scope_and_intent(
                content=request.content,
                project_context=project_context
            )
        except Exception as e:
            logger.error(f"Error in content scope analysis: {e}")
            return None

    async def _get_routing_recommendations(self,
                                         request: AgenticRAGRequest,
                                         scope_analysis: ScopeAnalysisResult,
                                         project_context: ProjectContext | None):
        """Step 3: Get intelligent routing recommendations for data source selection"""
        try:
            # Combine content and scope analysis for routing context
            routing_query = f"{request.content}\n\nScope: {scope_analysis.scope_description}\nIntent: {scope_analysis.intent_description}"
            
            # Get detailed routing recommendations
            recommendations = await self.routing_agent.get_detailed_source_recommendations(routing_query)
            
            # Filter recommendations above confidence threshold
            filtered_recommendations = [
                rec for rec in recommendations 
                if rec.relevance_score >= self.routing_agent.routing_config.get("confidence_threshold", 0.5)
            ]
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error getting routing recommendations: {e}")
            # Return empty list so search falls back to scope analysis recommendations
            return []

    async def _perform_multi_modal_search(self,
                                        content: str,
                                        scope_analysis: ScopeAnalysisResult,
                                        project_context: ProjectContext | None,
                                        routing_recommendations=None):
        """Step 4: Perform multi-modal search based on routing recommendations and scope analysis"""
        try:
            project_code = project_context.project_code if project_context else None

            # If we have routing recommendations, use them for enhanced search
            if routing_recommendations:
                # Extract source names and weights from routing recommendations
                recommended_sources = [(rec.source_name, rec.search_weight) for rec in routing_recommendations]
                logger.info(f"🤖 Using {len(recommended_sources)} routing recommendations for search")
                
                return await self.multi_search.search_with_routing_recommendations(
                    query=content,
                    scope_analysis=scope_analysis,
                    routing_recommendations=routing_recommendations,
                    project_code=project_code
                )
            else:
                # Fallback to original scope-based search
                logger.info("🔍 Falling back to scope-based search (no routing recommendations)")
                return await self.multi_search.search_by_scope_analysis(
                    query=content,
                    scope_analysis=scope_analysis,
                    project_code=project_code
                )
                
        except Exception as e:
            logger.error(f"Error in multi-modal search: {e}")
            # Return empty search results
            from core.services.generic_multi_modal_search import (
                GenericAggregatedSearchResult,
            )
            return GenericAggregatedSearchResult()

    async def _aggregate_documentation(self,
                                     content: str,
                                     scope_analysis: ScopeAnalysisResult,
                                     search_results) -> tuple[str, list[SourceReference], float]:
        """Step 4: Aggregate documentation using documentation aggregator"""
        try:
            return await self.doc_aggregator.aggregate_and_summarize(
                original_content=content,
                scope_analysis=scope_analysis,
                search_results=search_results
            )
        except Exception as e:
            logger.error(f"Error in documentation aggregation: {e}")
            # Return fallback response
            fallback_summary = f"""**Error in Documentation Aggregation**

An error occurred while aggregating the documentation: {str(e)}

**Basic Analysis:**
- Scope: {scope_analysis.scope_description}
- Intent: {scope_analysis.intent_description}
- Recommended Sources: {', '.join(scope_analysis.recommended_source_types)}

Please contact technical support for assistance with this issue."""

            return fallback_summary, [], 0.2

    def _create_default_project_context(self) -> ProjectContext:
        """Create a default project context for general issues"""
        return ProjectContext(
            project_code=None
        )

    def _create_error_response(self,
                             error_message: str,
                             project_context: ProjectContext | None,
                             start_time: float) -> AgenticRAGResponse:
        """Create an error response"""
        from core.schemas import ScopeAnalysisResult

        processing_time = int((time.time() - start_time) * 1000)

        # Create minimal scope analysis
        fallback_scope = ScopeAnalysisResult(
            scope_description="Unable to analyze due to system error",
            intent_description="System assistance unavailable",
            confidence=0.0,
            recommended_source_types=[]
        )

        return AgenticRAGResponse(
            project_context=project_context or self._create_default_project_context(),
            scope_analysis=fallback_scope,
            final_summary=f"**System Error**\n\n{error_message}\n\nPlease try again or contact technical support.",
            referenced_sources=[],
            overall_confidence=0.0,
            processing_time_ms=processing_time
        )

    def _create_low_confidence_response(self,
                                      request: AgenticRAGRequest,
                                      project_context: ProjectContext | None,
                                      scope_analysis: ScopeAnalysisResult,
                                      start_time: float) -> AgenticRAGResponse:
        """Create a response when content scope analysis confidence is too low"""
        processing_time = int((time.time() - start_time) * 1000)

        fallback_summary = f"""**Limited Analysis Available**

The automated analysis has low confidence in understanding this content. Here's what was detected:

**Scope Analysis:** {scope_analysis.scope_description}

**Intent Analysis:** {scope_analysis.intent_description}

**Confidence Level:** {scope_analysis.confidence:.0%} (below threshold)

**Recommended Actions:**
1. Review the content for clarity and add more specific details
2. Manually search the recommended source types: {', '.join([t.replace('_', ' ') for t in scope_analysis.recommended_source_types])}
3. Contact team members familiar with the general system
4. Consider providing additional context about the specific environment or configuration

For better automated assistance, please provide more detailed information about the issue, including specific error messages, steps to reproduce, and environmental context."""

        return AgenticRAGResponse(
            project_context=project_context or self._create_default_project_context(),
            scope_analysis=scope_analysis,
            final_summary=fallback_summary,
            referenced_sources=[],
            overall_confidence=scope_analysis.confidence,
            processing_time_ms=processing_time
        )

