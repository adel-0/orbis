"""
QueryAnalyzer - Unified query analysis agent for Orbis

Combines project detection, intent understanding, and source routing in a single LLM call.
Includes project-specific wiki context for accurate scope detection.
Replaces: ProjectDetectionService, ScopeAnalyzer, QueryRoutingAgent
"""

import logging
from datetime import datetime, timedelta, timezone

from app.db.models import WikiSummaryCache
from app.db.session import get_db_session
from config.settings import settings
from engine.schemas import AgenticRAGRequest, QueryAnalysis, WikiSummary
from orbis_core.llm.openai_client import OpenAIClientService
from utils.constants import AREA_PATH_MAPPINGS, PROJECT_WIKI_REPOS, WIKI_CACHE_REFRESH_DAYS

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Simple, unified query analyzer.

    Single LLM call to understand:
    - Project context (from area path + content)
    - User intent (what they want to achieve)
    - Which data sources to search
    """

    def __init__(self, openai_client: OpenAIClientService | None = None):
        """Initialize with optional OpenAI client"""
        self.openai_client = openai_client or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Get available source types from config
        from engine.config.data_sources import list_data_source_types
        self.available_sources = list_data_source_types()

    async def analyze(self, request: AgenticRAGRequest) -> QueryAnalysis:
        """
        Analyze query in a single LLM call with project-specific wiki context.

        Args:
            request: Request with content and optional area_path

        Returns:
            QueryAnalysis with project, intent, and recommended sources
        """
        try:
            # Detect project from area path (fast, non-LLM)
            project_hint = None
            if request.area_path:
                project_hint = self._detect_project_hint(request.area_path)

            # Get cached wiki context if project detected (fast DB read)
            wiki_summaries = []
            if project_hint:
                logger.debug(f"Project hint detected: {project_hint}, fetching wiki context...")
                wiki_summaries = await self._get_cached_wiki_summaries(project_hint)
                if wiki_summaries:
                    logger.info(f"Loaded {len(wiki_summaries)} wiki summaries for {project_hint}")

            # Build context for the LLM (includes wiki context if available)
            context = self._build_context(request, wiki_summaries)

            # Single LLM call with structured output
            logger.info("Analyzing query...")
            response = self.openai_client.client.beta.chat.completions.parse(
                model=self.openai_client.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                response_format=QueryAnalysis,
                max_completion_tokens=1000  # Keep it short - we just need analysis
            )

            result = response.choices[0].message.parsed
            if not result:
                logger.error("No analysis result from LLM")
                return self._get_fallback_analysis(request)

            logger.info(f"Analysis complete: project={result.project_code}, "
                       f"sources={len(result.recommended_sources)}, "
                       f"confidence={result.confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._get_fallback_analysis(request)

    def _build_context(self, request: AgenticRAGRequest, wiki_summaries: list[WikiSummary] = None) -> str:
        """
        Build context string for LLM including wiki context if available.

        Args:
            request: User request
            wiki_summaries: Optional list of wiki summaries for project context

        Returns:
            Formatted context string
        """
        parts = [f"Query: {request.content}"]

        # Add area path if available
        if request.area_path:
            parts.append(f"Area Path: {request.area_path}")

            # Hint at detected project
            detected = self._detect_project_hint(request.area_path)
            if detected:
                parts.append(f"Note: Area path suggests project '{detected}'")

        # Add wiki context if available (project-specific components and documentation)
        if wiki_summaries:
            wiki_context = self._format_wiki_context(wiki_summaries)
            parts.append(f"\n{wiki_context}")

        # List available sources
        parts.append(f"\nAvailable data sources: {', '.join(self.available_sources)}")

        return "\n".join(parts)

    def _detect_project_hint(self, area_path: str) -> str | None:
        """Quick project detection hint (non-LLM, for context only)"""
        for area_prefix, project_code in AREA_PATH_MAPPINGS.items():
            if area_path.startswith(area_prefix):
                return project_code
        return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for query analysis"""
        return """You are a query analysis agent for a RAG system.

Your job is to analyze user queries and determine:
1. **project_code**: Which project this relates to (SG for St. Gallen, VS for Valais, or None for general)
2. **intent**: What the user wants to achieve (be concise and specific)
3. **recommended_sources**: Which data source types would be most relevant (from the available list)
4. **confidence**: Your confidence in this analysis (0.0-1.0)

Guidelines:
- For project_code: Use area path hints when available, or infer from content mentions of locations/projects
- For intent: Focus on the user's goal (e.g., "troubleshoot error", "understand configuration", "find related tickets")
- For recommended_sources: Choose 2-4 most relevant sources; include both wiki and workitems when applicable
- Set confidence based on query clarity and available context

**Using Project Documentation Context:**
When PROJECT DOCUMENTATION CONTEXT is provided, it contains project-specific wiki summaries with:
- High-level project documentation summaries
- Key Components: Specific component/module/interface names used in that project

Use this context to:
- Cross-reference query terms with Key Components to detect relevant components
- Understand project-specific terminology and modules
- Increase confidence when query mentions components listed in Key Components
- Better understand project scope and architecture

If the query mentions terms that match Key Components, this strongly indicates project-specific scope.

Be concise and direct. Respond with the structured output format."""

    def _get_fallback_analysis(self, request: AgenticRAGRequest) -> QueryAnalysis:
        """Return fallback analysis when LLM fails"""
        # Try simple project detection
        project = None
        if request.area_path:
            project = self._detect_project_hint(request.area_path)

        return QueryAnalysis(
            project_code=project,
            intent="General assistance request",
            recommended_sources=self.available_sources,  # Search all sources
            confidence=0.3
        )

    def get_project_wiki_repos(self, project_code: str | None) -> list[str]:
        """Get wiki repositories for a project"""
        if not project_code:
            return []
        return PROJECT_WIKI_REPOS.get(project_code, [])

    async def _get_cached_wiki_summaries(self, project_code: str) -> list[WikiSummary]:
        """
        Get cached wiki summaries from DB for a project.
        Fast read - no LLM calls.

        Returns:
            List of WikiSummary objects from cache, or empty list if not cached
        """
        try:
            wiki_repos = self.get_project_wiki_repos(project_code)
            if not wiki_repos:
                logger.debug(f"No wiki repos configured for project {project_code}")
                return []

            summaries = []

            with get_db_session() as db:
                for wiki_name in wiki_repos:
                    cache_key = f"{wiki_name}_{project_code}"

                    cache_entry = db.query(WikiSummaryCache).filter(
                        WikiSummaryCache.cache_key == cache_key
                    ).first()

                    if not cache_entry:
                        logger.debug(f"No cache entry for {wiki_name}")
                        continue

                    # Check if cache is still fresh
                    refresh_threshold = datetime.now(timezone.utc) - timedelta(days=WIKI_CACHE_REFRESH_DAYS)
                    last_refreshed = cache_entry.last_refreshed_at

                    # Handle timezone-naive datetimes (SQLite compatibility)
                    if last_refreshed.tzinfo is None:
                        last_refreshed = last_refreshed.replace(tzinfo=timezone.utc)

                    if last_refreshed < refresh_threshold:
                        logger.debug(f"Cache for {wiki_name} is stale")
                        continue

                    # Deserialize WikiSummary
                    summary_data = cache_entry.summary_data
                    wiki_summary = WikiSummary(
                        wiki_name=summary_data["wiki_name"],
                        summary=summary_data["summary"],
                        key_components=summary_data.get("key_components", []),
                        summary_confidence=summary_data.get("summary_confidence", 0.85),
                        tokens_used=summary_data.get("tokens_used", 0)
                    )
                    summaries.append(wiki_summary)

            logger.info(f"Retrieved {len(summaries)} cached wiki summaries for {project_code}")
            return summaries

        except Exception as e:
            logger.error(f"Error retrieving cached wiki summaries: {e}")
            return []

    def _format_wiki_context(self, wiki_summaries: list[WikiSummary]) -> str:
        """
        Format wiki summaries into context string for LLM.

        Args:
            wiki_summaries: List of WikiSummary objects

        Returns:
            Formatted string with project documentation context
        """
        if not wiki_summaries:
            return ""

        context_parts = ["PROJECT DOCUMENTATION CONTEXT:"]

        for summary in wiki_summaries:
            context_parts.append(f"\n--- {summary.wiki_name} ---")
            context_parts.append(summary.summary)

            if summary.key_components:
                components_str = ", ".join(summary.key_components)
                context_parts.append(f"\nKey Components: {components_str}")

        return "\n".join(context_parts)

    def is_configured(self) -> bool:
        """Check if analyzer is properly configured"""
        return (
            self.openai_client is not None and
            bool(self.available_sources)
        )
