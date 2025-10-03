"""
Scope and Intent Analyzer for Orbis

This is the first agent in the agentic RAG system that analyzes content with documentation context
to determine scope (what components/interfaces are involved) and intent (what user wants to achieve).
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from config.settings import settings
from core.agents.wiki_summarization import WikiSummarizationService
from core.schemas import ProjectContext, ScopeAnalysisResult, WikiSummary
from core.services.project_detection import ProjectDetectionService
from infrastructure.data_processing.data_source_service import DataSourceService
from orbis_core.llm.openai_client import OpenAIClientService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService
from utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class ScopeAnalyzer:
    """
    Scope Analyzer: Analyzes content scope and intent with documentation context.

    This analyzer receives content and project context, retrieves relevant documentation
    context (without embedding search), and uses LLM to understand what the content
    concerns and what the user intends to achieve.
    """

    def __init__(self, vector_service: GenericVectorService | None = None, embedding_service: EmbeddingService | None = None, openai_client_service: OpenAIClientService | None = None):
        # Use shared OpenAI client
        self.openai_client_service = openai_client_service or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Initialize supporting services with dependency injection
        self.wiki_service = WikiSummarizationService(
            vector_service=vector_service,
            embedding_service=embedding_service,
            openai_client_service=self.openai_client_service
        )
        self.project_service = ProjectDetectionService()

        # Analysis configuration
        self.config = {
            "max_output_tokens": 5000,  # Maximum tokens for LLM response generation
            "confidence_threshold": 0.5,  # Minimum confidence to proceed
            "debug_logging_enabled": True,  # Enable file logging of responses for debugging
            "debug_log_dir": "logs/scope_analysis"  # Directory for response debug logs
        }


        # Get available data sources dynamically from configuration
        from core.config.data_sources import list_data_source_types
        self.available_source_types = list_data_source_types()

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

    @property
    def client(self):
        """Get the shared OpenAI client"""
        return self.openai_client_service.client

    @property
    def deployment_name(self):
        """Get the deployment name"""
        return self.openai_client_service.deployment_name

    def _check_client_available(self):
        """Check if OpenAI client is available"""
        try:
            if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
                logger.error("Azure OpenAI credentials not configured. Scope Analyzer cannot function.")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to check Azure OpenAI client availability: {e}")
            return False

    def _log_response_to_file(self, content: str, filename: str, response_type: str = "response") -> None:
        """Log LLM response content to file for debugging purposes"""
        try:
            if not self.config.get("debug_logging_enabled", False):
                return

            # Create debug log directory if it doesn't exist
            log_dir = Path(self.config["debug_log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)  # Sanitize filename
            full_filename = f"{timestamp}_{safe_filename}_{response_type}.md"

            file_path = log_dir / full_filename

            # Write content with metadata header
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Debug Log: {response_type.title()}\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Context:** {filename}\n")
                f.write(f"**Type:** {response_type}\n")
                f.write(f"**Content Length:** {len(content)} characters\n\n")
                f.write("---\n\n")
                f.write(content)

        except Exception as e:
            logger.warning(f"Failed to log response to file: {e}")

    def _is_auto_summarization_enabled(self) -> bool:
        """Check if auto-summarization is enabled for any azdo_wiki connector"""
        try:
            with DataSourceService() as ds_service:
                # Get all enabled data sources and filter by azdo_wiki type
                all_sources = ds_service.get_all_data_sources(enabled_only=True)
                wiki_sources = [source for source in all_sources if source.source_type == "azdo_wiki"]

                for wiki_source in wiki_sources:
                    # Check if enable_auto_summarization is set to False for any wiki connector
                    if not wiki_source.config.get('enable_auto_summarization', True):
                        logger.info(f"Auto-summarization disabled for connector: {wiki_source.name}")
                        return False
                return True
        except Exception as e:
            logger.error(f"Error checking auto-summarization setting: {e}")
            # Default to enabled if we can't check
            return True

    async def analyze_scope_and_intent(self,
                                     content: str,
                                     project_context: ProjectContext | None) -> ScopeAnalysisResult | None:
        """
        Main method for Scope Analyzer: Analyze content scope and intent with documentation context.

        Args:
            content: Full content to analyze (title + description, query, etc.)
            project_context: Detected project context from project detection service

        Returns:
            ScopeAnalysisResult with scope, intent, confidence, and recommended sources
        """
        if not self._check_client_available():
            logger.error("Scope Analyzer not properly configured - Azure OpenAI client not available")
            return None

        try:
            logger.debug(f"Analyzing {len(content)} chars with {'project' if project_context else 'general'} context")

            # Step 1: Get documentation context
            documentation_context = await self._get_documentation_context(project_context)
            logger.debug(f"Loaded {len(documentation_context)} chars of context")

            # Step 2: Analyze with LLM
            analysis_result = await self._perform_llm_analysis(
                content,
                project_context,
                documentation_context
            )

            if analysis_result:
                logger.debug(f"Analysis: confidence {analysis_result.confidence:.2f}, {len(analysis_result.recommended_source_types)} sources")
            else:
                logger.error("LLM analysis returned no result")

            return analysis_result

        except Exception as e:
            logger.error(f"Error in Scope Analyzer scope analysis: {e}")
            return None

    async def _get_documentation_context(self, project_context: ProjectContext | None) -> str:
        """
        Get documentation context for the LLM analysis.

        This provides high-level context about OnCall Dispatch and project-specific
        information without using embedding search.
        """
        context_parts = []


        # Add project-specific context if available
        if project_context and project_context.project_code:
            try:
                # Get project wiki repositories
                wiki_repos = self.project_service.get_project_wiki_repos(project_context.project_code)

                if wiki_repos:
                    logger.debug(f"Found {len(wiki_repos)} wiki repos for {project_context.project_code}")
                    if self._is_auto_summarization_enabled():
                        wiki_summaries = await self.wiki_service.get_project_wiki_summaries(
                            project_context.project_code,
                            wiki_repos
                        )

                        if wiki_summaries:
                            logger.debug(f"Loaded {len(wiki_summaries)} wiki summaries")
                            project_context_text = self._format_project_context(
                                project_context.project_code,
                                wiki_summaries
                            )
                            context_parts.append(f"=== PROJECT {project_context.project_code} CONTEXT ===\n{project_context_text}")
                        else:
                            logger.warning(f"No wiki summaries for {project_context.project_code}")
                    else:
                        logger.debug("Auto-summarization disabled")
                else:
                    logger.debug(f"No wiki repos for {project_context.project_code}")

            except Exception as e:
                logger.error(f"Error getting project context for {project_context.project_code}: {e}")

        combined_context = "\n\n".join(context_parts)
        logger.debug(f"Documentation context prepared: {len(combined_context)} characters")

        return combined_context



    def _format_project_context(self, project_code: str, wiki_summaries: list[WikiSummary]) -> str:
        """Format project-specific context from wiki summaries"""
        project_info = []

        project_info.append(f"PROJECT: {project_code}")

        for summary in wiki_summaries:
            project_info.append(f"\n--- {summary.wiki_name} ---")
            project_info.append(summary.summary)

            if summary.key_components:
                project_info.append(f"\nKey Components: {', '.join(summary.key_components)}")

        return "\n".join(project_info)

    async def _perform_llm_analysis(self,
                                  content: str,
                                  project_context: ProjectContext | None,
                                  documentation_context: str) -> ScopeAnalysisResult | None:
        """Perform the main LLM analysis of scope and intent"""
        try:
            # Build the analysis prompt
            prompt = self._build_analysis_prompt(content, project_context, documentation_context)

            # Call Azure OpenAI
            logger.debug(f"🧠 AGENTIC SCOPE: Sending request to Azure OpenAI - Model: {self.deployment_name}, Max tokens: {self.config['max_output_tokens']}")

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "developer",
                        "content": self._get_developer_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                reasoning_effort="low",  # Fast analysis for scope determination
                verbosity="low",         # Structured JSON output
                max_completion_tokens=self.config["max_output_tokens"]
            )

            logger.debug(f"🧠 AGENTIC SCOPE: Received response from Azure OpenAI - Usage: {response.usage}, Model: {response.model if hasattr(response, 'model') else 'unknown'}")

            # Parse the response with better error handling
            if not response.choices or len(response.choices) == 0:
                logger.error("🧠 AGENTIC SCOPE: No choices returned in OpenAI response")
                return None

            message_content = response.choices[0].message.content
            if message_content is None:
                logger.error("🧠 AGENTIC SCOPE: OpenAI returned None content - possible content filtering")
                logger.error("🧠 AGENTIC SCOPE: Check if prompt triggers content policies or safety filters")
                return None

            analysis_text = message_content.strip()
            if not analysis_text:
                logger.error("🧠 AGENTIC SCOPE: OpenAI returned empty content after stripping")
                return None

            logger.debug(f"Scope Analyzer LLM response: {analysis_text[:200]}...")
            logger.info(f"🧠 AGENTIC SCOPE: LLM returned {len(analysis_text)} character analysis - parsing structured information")

            # Log the full response for debugging
            context_name = f"scope_analysis_{project_context.project_code if project_context else 'unknown'}"
            self._log_response_to_file(analysis_text, context_name, "scope_analysis")

            # Parse JSON response from LLM
            parsed_result = self._parse_analysis_response(analysis_text, content)
            if not parsed_result:
                logger.error("Failed to parse LLM response")
            return parsed_result

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return None

    def _get_developer_prompt(self) -> str:
        """Get the developer prompt for Scope Analyzer from YAML"""
        available_sources_str = ', '.join(self.available_source_types)
        variables = {
            'available_source_types': available_sources_str
        }
        return self.prompt_loader.get_developer_prompt("scope_analysis", variables)

    def _build_analysis_prompt(self,
                             content: str,
                             project_context: ProjectContext | None,
                             documentation_context: str) -> str:
        """Build the analysis prompt for the LLM using YAML template"""

        project_info = ""
        if project_context and project_context.project_code:
            project_info = f"""
PROJECT CONTEXT:
- Detected Project: {project_context.project_code}
"""

        available_sources_str = ', '.join(self.available_source_types)
        variables = {
            'content': content,
            'project_info': project_info,
            'documentation_context': documentation_context,
            'available_source_types': available_sources_str
        }

        return self.prompt_loader.get_user_prompt("scope_analysis", variables)

    def _parse_analysis_response(self, analysis_text: str, content: str) -> ScopeAnalysisResult | None:
        """
        Parse JSON response from LLM.
        """
        if not analysis_text.strip():
            return None

        try:
            import json

            # Try to parse as JSON
            analysis_data = json.loads(analysis_text.strip())

            return ScopeAnalysisResult(
                scope_description=analysis_data["scope_description"],
                intent_description=analysis_data["intent_description"],
                confidence=float(analysis_data["confidence"]),
                recommended_source_types=analysis_data["recommended_source_types"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing JSON analysis response: {e}")
            logger.debug(f"Raw LLM response: {analysis_text[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing analysis response: {e}")
            return None


    def is_configured(self) -> bool:
        """Check if Scope Analyzer is properly configured"""
        return self.client is not None and self.wiki_service.is_configured()
