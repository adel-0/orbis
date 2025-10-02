"""
Summary Agent for OnCall Copilot

Generates intelligent summaries of search results and ticket collections using LLM analysis.
This agent autonomously synthesizes multiple tickets into coherent, actionable summaries.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from config.settings import settings
from core.schemas import BaseContent
from infrastructure.llm.openai_client import OpenAIClientService
from utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class SearchResultsSummarizer:
    """Summarizer for generating intelligent summaries of search results and ticket collections"""

    def __init__(self, openai_client_service: OpenAIClientService | None = None):
        # Use shared OpenAI client
        self.openai_client_service = openai_client_service or OpenAIClientService()
        # Initialize prompt loader
        self.prompt_loader = PromptLoader()
        # Configuration
        self.config = {
            "debug_logging_enabled": True,  # Enable file logging of responses for debugging
            "debug_log_dir": "logs/summary_generation"  # Directory for response debug logs
        }

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
                logger.warning("Azure OpenAI credentials not configured. Summary agent will be disabled.")
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
            raise

    def generate_summary(self, query: str, content_items: list[BaseContent], similarity_scores: list[float] | None = None) -> str | None:
        """Generate an intelligent summary of search results"""
        if not self._check_client_available():
            logger.warning("Azure OpenAI not configured - cannot generate summary")
            return None

        if not content_items:
            logger.warning("No content items provided for summary generation")
            return None

        try:
            # Prepare context for the summary
            context_parts = [f"Query: {query}\n\nRelevant content:"]

            for i, content_item in enumerate(content_items, 1):
                item_text = f"{i}. {content_item.content_type.title()} {content_item.id}: {content_item.title}"
                if hasattr(content_item, 'description') and content_item.description:
                    item_text += f"\n   Description: {content_item.description}"
                if hasattr(content_item, 'comments') and content_item.comments:
                    item_text += f"\n   Comments: {'; '.join(content_item.comments[:3])}"  # Limit to first 3 comments
                # Add per-ticket relevancy when available
                if similarity_scores and i-1 < len(similarity_scores) and similarity_scores[i-1] is not None:
                    score = float(similarity_scores[i-1])
                    item_text += f"\n   Relevancy: {score:.2f}"
                context_parts.append(item_text)

            context = "\n\n".join(context_parts)

            # Determine overall relevancy calibration
            confidence_note = ""
            if similarity_scores:
                filtered = [float(s) for s in similarity_scores if s is not None]
                if filtered:
                    max_score = max(filtered)
                    avg_score = sum(filtered) / len(filtered)
                    # Calibrate tone/length based on max relevancy
                    if max_score >= 0.8:
                        confidence = "high"
                        confidence_note = (
                            "Be confident and decisive. Provide a clear recommendation and the most likely fix. "
                            "Assume the top content items closely match the query."
                        )
                    elif max_score >= 0.65:
                        confidence = "medium"
                        confidence_note = (
                            "Be balanced. Indicate reasonable confidence but note where validation is needed. "
                            "Prefer probable fixes and mention alternatives briefly."
                        )
                    else:
                        confidence = "low"
                        confidence_note = (
                            "Be cautious and brief. State uncertainty and suggest concrete next steps to gather more context."
                        )

                    relevancy_summary = f"Overall relevancy: {confidence} (max={max_score:.2f}, avg={avg_score:.2f})."
                else:
                    relevancy_summary = "Overall relevancy: unknown."
            else:
                relevancy_summary = "Overall relevancy: unknown."



            # Get prompts from external template
            prompt_variables = {
                "query": query,
                "context": context,
                "relevancy_summary": relevancy_summary,
                "confidence_note": confidence_note
            }

            developer_prompt = self.prompt_loader.get_developer_prompt("summary_generation", prompt_variables)
            user_prompt = self.prompt_loader.get_user_prompt("summary_generation", prompt_variables)

            # Generate summary
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "developer", "content": developer_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning_effort="medium",  # Balanced reasoning for actionable insights
                verbosity="low"             # Concise actionable output
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated successfully")

            # Log the full response for debugging
            context_name = "search_results_summary"
            self._log_response_to_file(summary, context_name, "search_summary")

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured"""
        has_client = self.client is not None
        has_endpoint = bool(settings.AZURE_OPENAI_ENDPOINT)
        has_api_key = bool(settings.AZURE_OPENAI_API_KEY)
        return has_client and has_endpoint and has_api_key

    def get_config_info(self) -> dict:
        """Get configuration information"""
        return {
            "configured": self.is_configured(),
            "endpoint": settings.AZURE_OPENAI_ENDPOINT,
            "deployment": self.deployment_name,
            "has_api_key": bool(settings.AZURE_OPENAI_API_KEY)
        }
