"""
Wiki Summarization Service for Orbis

Provides high-level summaries of project wikis for contextual analysis,
handling very large wiki content (hundreds of thousands of tokens).
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.db.models import WikiSummaryCache
from app.db.session import get_db_session
from config.settings import settings
from core.schemas import WikiSummary
from orbis_core.llm.openai_client import OpenAIClientService
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService
from utils.constants import (
    PROJECT_WIKI_REPOS,
    WIKI_CACHE_REFRESH_DAYS,
    WIKI_CACHE_STARTUP_TIMEOUT_MINUTES,
)
from utils.prompt_loader import PromptLoader
from orbis_core.utils.token_utils import count_tokens

logger = logging.getLogger(__name__)

class WikiSummarizationService:
    """Service for summarizing large wiki content for contextual analysis"""

    def __init__(self, vector_service: GenericVectorService | None = None, embedding_service: EmbeddingService | None = None, openai_client_service: OpenAIClientService | None = None):
        # Use dependency injection for services
        self.openai_client_service = openai_client_service or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=settings.AZURE_OPENAI_MODEL
        )
        self.vector_service = vector_service
        self.embedding_service = embedding_service

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

        # Use configuration from settings
        self.config = {
            "max_input_tokens_per_chunk": settings.WIKI_MAX_INPUT_TOKENS_PER_CHUNK,
            "overlap_tokens": settings.WIKI_OVERLAP_TOKENS,
            "cache_duration_hours": settings.WIKI_CACHE_DURATION_HOURS,
            "target_output_tokens_per_page": settings.WIKI_TARGET_OUTPUT_TOKENS_PER_PAGE,
            "max_content_size_mb": settings.WIKI_MAX_CONTENT_SIZE_MB,
            "content_size_check_enabled": settings.WIKI_CONTENT_SIZE_CHECK_ENABLED,
            "max_output_tokens": settings.WIKI_MAX_OUTPUT_TOKENS,
            "debug_logging_enabled": settings.WIKI_DEBUG_LOGGING_ENABLED,
            "debug_log_dir": settings.WIKI_DEBUG_LOG_DIR
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
                logger.warning("Azure OpenAI credentials not configured. Wiki summarization will be disabled.")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to check Azure OpenAI client availability: {e}")
            return False

    def _log_summary_to_file(self, content: str, filename: str, summary_type: str = "summary") -> None:
        """Log summary content to file for debugging purposes"""
        try:
            if not self.config.get("debug_logging_enabled", False):
                return

            # Create debug log directory if it doesn't exist
            log_dir = Path(self.config["debug_log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)  # Sanitize filename
            full_filename = f"{timestamp}_{safe_filename}_{summary_type}.md"

            file_path = log_dir / full_filename

            # Write content with metadata header
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Debug Log: {summary_type.title()}\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Wiki/Context:** {filename}\n")
                f.write(f"**Type:** {summary_type}\n")
                f.write(f"**Length:** {len(content)} characters\n\n")
                f.write("---\n\n")
                f.write(content)

            logger.debug(f"Debug logged {summary_type} to: {file_path}")

        except Exception as e:
            logger.warning(f"Failed to log {summary_type} to file: {e}")
            # Don't fail the summarization process if logging fails

    async def get_project_wiki_summaries(self,
                                       project_code: str | None,
                                       wiki_repos: list[str]) -> list[WikiSummary]:
        """
        Get high-level summaries for project wikis.

        Args:
            project_code: Project code (SG, VS) or None for general
            wiki_repos: List of wiki repository names to summarize

        Returns:
            List of WikiSummary objects with high-level overviews
        """
        if not self.client:
            logger.warning("Wiki summarization not available - Azure OpenAI not configured")
            return []

        summaries = []

        for wiki_name in wiki_repos:
            try:
                # Check persistent cache first
                cached_summary = await self._get_cached_summary(wiki_name, project_code)
                if cached_summary:
                    logger.debug(f"Using cached summary for {wiki_name}")
                    summaries.append(cached_summary)
                    continue

                # Generate new summary if not cached
                summary = await self._generate_and_cache_summary(wiki_name, project_code)
                if summary:
                    summaries.append(summary)

            except Exception as e:
                logger.error(f"Error getting wiki summary for {wiki_name}: {e}")
                continue

        return summaries

    def _validate_content_size(self, wiki_pages: list[dict[str, str]], wiki_name: str) -> bool:
        """
        Validate that wiki content size is within acceptable limits.

        Args:
            wiki_pages: List of wiki pages with content
            wiki_name: Name of the wiki for logging

        Returns:
            True if content size is acceptable, False if too large
        """
        if not self.config["content_size_check_enabled"]:
            return True

        try:
            # Calculate total content size in bytes (text only)
            total_size_bytes = sum(
                len(page['content'].encode('utf-8'))
                for page in wiki_pages
                if page.get('content')
            )

            total_size_mb = total_size_bytes / (1024 * 1024)
            max_size_mb = self.config["max_content_size_mb"]

            if total_size_mb > max_size_mb:
                logger.warning(
                    f"Wiki '{wiki_name}' content size ({total_size_mb:.1f}MB) exceeds limit "
                    f"({max_size_mb}MB). Skipping summarization to avoid excessive processing time."
                )
                return False

            logger.info(f"Wiki '{wiki_name}' content size: {total_size_mb:.1f}MB (within {max_size_mb}MB limit)")
            return True

        except Exception as e:
            logger.error(f"Error validating content size for '{wiki_name}': {e}")
            # On error, allow processing but log the issue
            return True

    async def _get_wiki_content(self, wiki_name: str, project_code: str | None) -> list[dict[str, str]] | None:
        """
        Retrieve wiki content from the regular database, preserving page structure.

        Returns:
            List of dicts with 'title' and 'content' for each wiki page
        """
        try:
            from sqlalchemy import text
            from sqlalchemy.exc import SQLAlchemyError

            from app.db.models import Content, DataSource

            with get_db_session() as db:
                # Find wiki content by querying the regular database
                # Join Content with DataSource to filter by source type
                # Use SQLite's json_extract function properly
                query = db.query(Content).join(DataSource).filter(
                    DataSource.source_type == "azdo_wiki",
                    text("json_extract(content.content_metadata, '$.wiki_name') = :wiki_name")
                ).params(wiki_name=wiki_name)

                content_records = query.all()

                if not content_records:
                    logger.warning(f"No wiki content found for {wiki_name}")
                    return None

                # Convert database records to the expected format
                wiki_pages = []
                total_content_chars = 0
                for record in content_records:
                    if record.content:  # Only include records with actual content
                        wiki_pages.append({
                            'title': record.title or 'Untitled',
                            'content': record.content
                        })
                        total_content_chars += len(record.content)

                logger.info(f"Retrieved {len(wiki_pages)} pages from {wiki_name} from regular database")
                return wiki_pages if wiki_pages else None

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving wiki content for {wiki_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving wiki content for {wiki_name}: {e}")
            return None

    async def _summarize_wiki(self, wiki_name: str, wiki_pages: list[dict[str, str]]) -> WikiSummary | None:
        """
        Summarize wiki content using improved page-first chunking strategy.
        """
        try:
            if not wiki_pages:
                return None

            # Validate content size before processing
            if not self._validate_content_size(wiki_pages, wiki_name):
                logger.info(f"Skipping wiki '{wiki_name}' due to size limits")
                return None

            # Aggregate pages into batches and summarize efficiently
            final_summary = await self._aggregate_and_summarize_pages(wiki_pages, wiki_name)

            if not final_summary:
                return None

            # Extract key components from LLM response
            key_components = self._extract_key_components_from_llm_response(final_summary)

            # Estimate total tokens processed
            total_tokens = sum(count_tokens(page['content'], self.openai_client_service.model_name) for page in wiki_pages)

            return WikiSummary(
                wiki_name=wiki_name,
                summary=final_summary,
                key_components=key_components,
                summary_confidence=0.85,  # Good confidence for LLM summaries
                tokens_used=min(total_tokens, self.config["max_output_tokens"])
            )

        except Exception as e:
            logger.error(f"Error summarizing wiki {wiki_name}: {e}")
            return None

    async def _aggregate_and_summarize_pages(self, wiki_pages: list[dict[str, str]], wiki_name: str) -> str | None:
        """
        Efficiently aggregate wiki pages into batches and summarize with minimal API calls.

        Strategy:
        1. Group pages into batches that fit within token limits
        2. Summarize each batch in one API call
        3. If multiple batches, combine batch summaries in final call
        """
        try:
            # Group pages into batches that fit within chunk size limits
            page_batches = self._group_pages_into_batches(wiki_pages)
            logger.info(f"Grouped {len(wiki_pages)} pages into {len(page_batches)} batches for {wiki_name}")

            if len(page_batches) == 1:
                # Single batch - one API call for everything
                return await self._summarize_page_batch(page_batches[0], wiki_name, is_final=True)

            # Multiple batches - summarize each batch, then combine
            batch_summaries = []
            for i, batch in enumerate(page_batches):
                batch_summary = await self._summarize_page_batch(
                    batch,
                    f"{wiki_name} (batch {i+1}/{len(page_batches)})",
                    is_final=False
                )
                if batch_summary:
                    batch_summaries.append(batch_summary)

            if not batch_summaries:
                return None

            # Combine batch summaries - use batched approach if too large
            return await self._combine_batch_summaries(batch_summaries, wiki_name)

        except Exception as e:
            logger.error(f"Error in efficient page aggregation for {wiki_name}: {e}")
            return None

    def _group_pages_into_batches(self, wiki_pages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
        """
        Group wiki pages into batches that fit within token limits for input + output.

        Strategy: Reserve space for both input content and output summary generation.

        Returns:
            List of page batches, where each batch fits within safe token limits
        """
        batches = []
        current_batch = []
        current_batch_tokens = 0

        safe_input_limit = self._calculate_safe_input_limit()
        logger.info(f"Using safe input limit: {safe_input_limit} tokens per batch")

        for page in wiki_pages:
            page_content = page.get('content', '')
            page_title = page.get('title', 'Untitled')

            # Accurate token estimation using tiktoken
            formatted_content = f"## {page_title}\n{page_content}\n\n"
            page_tokens = count_tokens(formatted_content, self.openai_client_service.model_name)

            # Check if page alone exceeds safe limit
            if page_tokens > safe_input_limit:
                # Large page - split it or create separate batch
                logger.warning(f"Page '{page_title}' ({page_tokens} tokens) exceeds safe limit ({safe_input_limit})")
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                # For very large pages, we'll still try to process them individually
                batches.append([page])
                continue

            # Check if adding this page would exceed batch limit
            if current_batch_tokens + page_tokens > safe_input_limit:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [page]
                current_batch_tokens = page_tokens
            else:
                # Add to current batch
                current_batch.append(page)
                current_batch_tokens += page_tokens

        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)

        self._log_batch_statistics(batches)
        return batches

    def _calculate_safe_input_limit(self) -> int:
        """Calculate safe input token limit reserving space for output and prompts"""
        max_total_tokens = settings.WIKI_MODEL_CONTEXT_LIMIT
        output_tokens = 8000  # Reasonable output size
        prompt_overhead = 1000  # Space for prompts and formatting

        safe_input_limit = max_total_tokens - output_tokens - prompt_overhead
        return min(safe_input_limit, self.config["max_input_tokens_per_chunk"])

    def _log_batch_statistics(self, batches: list[list[dict[str, str]]]) -> None:
        """Log statistics for each batch"""
        for i, batch in enumerate(batches):
            total_content = "\n\n".join(f"## {p.get('title', 'Untitled')}\n{p.get('content', '')}" for p in batch)
            estimated_tokens = count_tokens(total_content, self.openai_client_service.model_name)
            logger.info(f"Batch {i+1}: {len(batch)} pages, {estimated_tokens} tokens")

    async def _call_llm_for_summary(self, content: str, context: str, summary_type: str = "batch", is_final: bool = False) -> str | None:
        """
        Unified method for making LLM calls for summarization.

        Args:
            content: The content to summarize (already formatted)
            context: Context for logging/prompts
            summary_type: Type of summary (batch, comprehensive, group)
            is_final: Whether this is the final summary or intermediate
        """
        try:
            if not content.strip():
                logger.warning(f"Content is empty for {context}")
                return None

            # Choose appropriate prompts based on summary type
            if summary_type == "comprehensive":
                developer_prompt = self.prompt_loader.get_developer_prompt("wiki_comprehensive_summarization")
                user_prompt = self.prompt_loader.get_user_prompt("wiki_comprehensive_summarization", {
                    "wiki_name": context,
                    "combined_summaries": content
                })
            else:
                developer_prompt = self.prompt_loader.get_developer_prompt("wiki_batch_summarization")
                if is_final:
                    user_prompt = self.prompt_loader.get_user_prompt("wiki_batch_summarization", {
                        "summary_type": "Wiki",
                        "context": context,
                        "combined_content": content,
                        "detail_level": "detailed"
                    })
                else:
                    user_prompt = self.prompt_loader.get_user_prompt("wiki_batch_summarization", {
                        "summary_type": "Section",
                        "context": context,
                        "combined_content": content,
                        "detail_level": "comprehensive"
                    })

            max_tokens = min(self.config["max_output_tokens"], settings.WIKI_SAFE_OUTPUT_LIMIT)

            # Set reasoning_effort and verbosity based on summary type
            if summary_type == "comprehensive" or is_final:
                reasoning_effort = "high"  # More thorough analysis for final summaries
                verbosity = "medium"         # More detailed output for final summaries
            else:
                reasoning_effort = "low"     # Fast extraction for batch summaries
                verbosity = "low"            # Concise output for batch summaries

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "developer", "content": developer_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity
            )

            summary = response.choices[0].message.content
            if summary:
                summary = summary.strip()
                summary_tokens = count_tokens(summary, self.openai_client_service.model_name)
                logger.info(f"Generated {summary_type} summary for {context}: {summary_tokens} tokens")
                self._log_summary_to_file(summary, context, f"{summary_type}_summary")
                return summary
            else:
                logger.warning(f"LLM returned empty/null response for {context}")
                return None

        except ValueError as e:
            logger.error(f"Invalid input for LLM call for {context}: {e}")
            return None
        except KeyError as e:
            logger.error(f"Missing configuration for LLM call for {context}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in LLM call for {context}: {e}")
            return None

    async def _summarize_page_batch(self, pages: list[dict[str, str]], context: str, is_final: bool = False) -> str | None:
        """
        Summarize a batch of wiki pages in a single API call.
        """
        try:
            # Combine all pages in the batch
            combined_content = "\n\n".join([
                f"## {page.get('title', 'Untitled')}\n{page.get('content', '')}"
                for page in pages
            ])

            return await self._call_llm_for_summary(combined_content, context, "batch", is_final)

        except Exception as e:
            logger.error(f"Error summarizing page batch {context}: {e}")
            return None

    async def _combine_batch_summaries(self, batch_summaries: list[str], wiki_name: str) -> str | None:
        """Combine multiple batch summaries into final comprehensive summary using simple chunking"""
        try:
            logger.info(f"Combining {len(batch_summaries)} batch summaries for {wiki_name}")

            # Simple approach: if we have too many summaries, chunk them and summarize in stages
            safe_input_limit = self._calculate_safe_input_limit()

            # Group summaries into manageable chunks
            summary_chunks = self._chunk_summaries_by_token_limit(batch_summaries, safe_input_limit)

            if len(summary_chunks) == 1:
                # Single chunk - summarize directly
                combined_summaries = "\n\n".join([
                    f"### Section {i+1}\n{summary}"
                    for i, summary in enumerate(summary_chunks[0])
                ])
                final_summary = await self._call_llm_for_summary(combined_summaries, wiki_name, "comprehensive")
            else:
                # Multiple chunks - summarize each chunk then combine results
                chunk_summaries = []
                for i, chunk in enumerate(summary_chunks):
                    combined_chunk = "\n\n".join([
                        f"### Section {j+1}\n{summary}"
                        for j, summary in enumerate(chunk)
                    ])
                    chunk_summary = await self._call_llm_for_summary(
                        combined_chunk,
                        f"{wiki_name} (chunk {i+1}/{len(summary_chunks)})",
                        "comprehensive"
                    )
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)

                # Final combination of chunk summaries
                if chunk_summaries:
                    final_combined = "\n\n".join([
                        f"### Part {i+1}\n{summary}"
                        for i, summary in enumerate(chunk_summaries)
                    ])
                    final_summary = await self._call_llm_for_summary(final_combined, wiki_name, "comprehensive")
                else:
                    final_summary = None

            if final_summary:
                self._log_summary_to_file(final_summary, wiki_name, "final_summary")

            return final_summary

        except Exception as e:
            logger.error(f"Error combining batch summaries for {wiki_name}: {e}")
            return None


    def _chunk_summaries_by_token_limit(self, summaries: list[str], token_limit: int) -> list[list[str]]:
        """Group summaries into chunks that fit within token limit"""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for summary in summaries:
            summary_tokens = count_tokens(summary, self.openai_client_service.model_name)

            # If adding this summary would exceed the limit, start a new chunk
            if current_tokens + summary_tokens > token_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [summary]
                current_tokens = summary_tokens
            else:
                current_chunk.append(summary)
                current_tokens += summary_tokens

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks







    def _extract_key_components_from_llm_response(self, summary: str) -> list[str]:
        """
        Extract key components from LLM response that should include a KEY_COMPONENTS section.

        Falls back to empty list if no KEY_COMPONENTS section found.
        """
        try:
            # Look for KEY_COMPONENTS section in the summary
            if "KEY_COMPONENTS" in summary:
                # Extract content after KEY_COMPONENTS
                components_section = summary.split("KEY_COMPONENTS", 1)[1]

                # Extract list items (lines starting with -, *, or numbers)
                import re
                component_lines = re.findall(r'^\s*[-*\d\.)]\s*(.+?)$', components_section, re.MULTILINE)

                # Clean up component names
                components = []
                for line in component_lines:
                    # Remove extra formatting and get the component name
                    component = re.sub(r'[\-\*\d\.\)\s]+', '', line.strip(), count=1).strip()
                    if component and len(component) > 2:  # Filter out very short entries
                        components.append(component)

                logger.info(f"Extracted {len(components)} key components from LLM response")
                return components[:10]  # Limit to top 10

            logger.warning("No KEY_COMPONENTS section found in LLM summary")
            return []

        except Exception as e:
            logger.error(f"Error extracting key components from LLM response: {e}")
            return []


    def is_configured(self) -> bool:
        """Check if wiki summarization is properly configured"""
        return self.client is not None

    # ====== PERSISTENT CACHE METHODS ======

    async def precompute_all_project_summaries(self, timeout_minutes: int = WIKI_CACHE_STARTUP_TIMEOUT_MINUTES) -> dict[str, Any]:
        """
        Pre-compute all project wiki summaries during startup.

        Args:
            timeout_minutes: Maximum time to spend on pre-computation

        Returns:
            Dict with computation results and statistics
        """
        start_time = datetime.now()
        timeout_time = start_time + timedelta(minutes=timeout_minutes)

        results = {
            "processed": 0,
            "cached": 0,
            "failed": 0,
            "skipped_no_content": 0,
            "total_processing_time_ms": 0,
            "projects": {}
        }

        if not self._check_client_available():
            logger.warning("Azure OpenAI not configured - skipping wiki summary pre-computation")
            return results

        logger.info(f"🧠 Starting wiki summary pre-computation (timeout: {timeout_minutes} minutes)")

        try:
            # Process all configured projects
            for project_code, wiki_repos in PROJECT_WIKI_REPOS.items():
                if datetime.now() > timeout_time:
                    logger.warning(f"Pre-computation timeout reached, stopping at project {project_code}")
                    break

                project_results = {"wikis": [], "processing_time_ms": 0}
                project_start = datetime.now()

                logger.info(f"Processing project {project_code} wikis: {wiki_repos}")

                for wiki_name in wiki_repos:
                    try:
                        # Check if cache needs refresh
                        needs_refresh = await self._cache_needs_refresh(wiki_name, project_code)

                        if not needs_refresh:
                            logger.debug(f"Wiki {wiki_name} cache is fresh, skipping")
                            results["cached"] += 1
                            project_results["wikis"].append({"wiki": wiki_name, "status": "cached"})
                            continue

                        # Generate and cache summary
                        summary = await self._generate_and_cache_summary(wiki_name, project_code)

                        if summary:
                            results["processed"] += 1
                            project_results["wikis"].append({"wiki": wiki_name, "status": "processed"})
                            logger.info(f"✅ Pre-computed summary for {project_code}/{wiki_name}")
                        else:
                            results["skipped_no_content"] += 1
                            project_results["wikis"].append({"wiki": wiki_name, "status": "no_content"})
                            logger.warning(f"⚠️ No content found for {project_code}/{wiki_name}")

                    except Exception as e:
                        results["failed"] += 1
                        project_results["wikis"].append({"wiki": wiki_name, "status": "failed", "error": str(e)})
                        logger.error(f"❌ Failed to pre-compute {project_code}/{wiki_name}: {e}")

                    # Check timeout between wikis
                    if datetime.now() > timeout_time:
                        logger.warning("Pre-computation timeout reached during wiki processing")
                        break

                project_results["processing_time_ms"] = int((datetime.now() - project_start).total_seconds() * 1000)
                results["projects"][project_code] = project_results

            results["total_processing_time_ms"] = int((datetime.now() - start_time).total_seconds() * 1000)

            total_wikis = results["processed"] + results["cached"] + results["failed"] + results["skipped_no_content"]
            logger.info(f"🎯 Wiki pre-computation completed: {total_wikis} wikis processed in {results['total_processing_time_ms']}ms")
            logger.info(f"   Processed: {results['processed']}, Cached: {results['cached']}, Failed: {results['failed']}, No Content: {results['skipped_no_content']}")

            return results

        except Exception as e:
            logger.error(f"❌ Error during wiki summary pre-computation: {e}")
            results["total_processing_time_ms"] = int((datetime.now() - start_time).total_seconds() * 1000)
            return results

    async def _get_cached_summary(self, wiki_name: str, project_code: str | None) -> WikiSummary | None:
        """Get summary from persistent cache if available and fresh"""
        try:
            cache_key = self._build_cache_key(wiki_name, project_code)

            with get_db_session() as db:
                cache_entry = db.query(WikiSummaryCache).filter(
                    WikiSummaryCache.cache_key == cache_key
                ).first()

                if not cache_entry:
                    return None

                # Check if cache is still fresh
                refresh_threshold = datetime.now() - timedelta(days=WIKI_CACHE_REFRESH_DAYS)
                if cache_entry.last_refreshed_at < refresh_threshold:
                    logger.debug(f"Cache entry for {cache_key} is stale (last refreshed: {cache_entry.last_refreshed_at})")
                    return None

                # Deserialize and return WikiSummary
                return self._deserialize_wiki_summary(cache_entry.summary_data)

        except Exception as e:
            logger.error(f"Error retrieving cached summary for {wiki_name}: {e}")
            return None

    async def _cache_needs_refresh(self, wiki_name: str, project_code: str | None) -> bool:
        """Check if a wiki summary cache entry needs refresh"""
        try:
            cache_key = self._build_cache_key(wiki_name, project_code)

            with get_db_session() as db:
                cache_entry = db.query(WikiSummaryCache).filter(
                    WikiSummaryCache.cache_key == cache_key
                ).first()

                if not cache_entry:
                    return True  # No cache entry, needs refresh

                # Check if cache is stale
                refresh_threshold = datetime.now() - timedelta(days=WIKI_CACHE_REFRESH_DAYS)
                return cache_entry.last_refreshed_at < refresh_threshold

        except Exception as e:
            logger.error(f"Error checking cache refresh status for {wiki_name}: {e}")
            return True  # On error, assume refresh needed

    async def _generate_and_cache_summary(self, wiki_name: str, project_code: str | None) -> WikiSummary | None:
        """Generate a new wiki summary and cache it persistently"""
        try:
            start_time = datetime.now()

            # Get wiki content
            wiki_pages = await self._get_wiki_content(wiki_name, project_code)
            if not wiki_pages:
                logger.warning(f"No content found for wiki: {wiki_name}")
                return None

            # Summarize the wiki
            summary = await self._summarize_wiki(wiki_name, wiki_pages)
            if not summary:
                logger.warning(f"Failed to generate summary for wiki: {wiki_name}")
                return None

            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Cache the summary persistently
            await self._cache_summary_persistently(wiki_name, project_code, summary, processing_time_ms)

            return summary

        except Exception as e:
            logger.error(f"Error generating and caching summary for {wiki_name}: {e}")
            return None

    async def _cache_summary_persistently(self, wiki_name: str, project_code: str | None,
                                        summary: WikiSummary, processing_time_ms: int) -> None:
        """Save wiki summary to persistent cache"""
        try:
            cache_key = self._build_cache_key(wiki_name, project_code)
            now = datetime.now()

            with get_db_session() as db:
                # Check if cache entry already exists
                existing_entry = db.query(WikiSummaryCache).filter(
                    WikiSummaryCache.cache_key == cache_key
                ).first()

                if existing_entry:
                    # Update existing entry
                    existing_entry.summary_data = self._serialize_wiki_summary(summary)
                    existing_entry.last_refreshed_at = now
                    existing_entry.refresh_count += 1
                    existing_entry.summary_confidence = str(summary.summary_confidence) if hasattr(summary, 'summary_confidence') else None
                    existing_entry.tokens_used = summary.tokens_used if hasattr(summary, 'tokens_used') else None
                    existing_entry.processing_time_ms = processing_time_ms

                    logger.debug(f"Updated persistent cache entry for {cache_key}")
                else:
                    # Create new cache entry
                    cache_entry = WikiSummaryCache(
                        cache_key=cache_key,
                        wiki_name=wiki_name,
                        project_code=project_code,
                        summary_data=self._serialize_wiki_summary(summary),
                        summary_confidence=str(summary.summary_confidence) if hasattr(summary, 'summary_confidence') else None,
                        tokens_used=summary.tokens_used if hasattr(summary, 'tokens_used') else None,
                        processing_time_ms=processing_time_ms,
                        last_refreshed_at=now,
                        refresh_count=1
                    )
                    db.add(cache_entry)

                    logger.debug(f"Created persistent cache entry for {cache_key}")

                db.commit()

        except Exception as e:
            logger.error(f"Error caching summary persistently for {wiki_name}: {e}")

    def _build_cache_key(self, wiki_name: str, project_code: str | None) -> str:
        """Build consistent cache key"""
        return f"{wiki_name}_{project_code or 'general'}"

    def _serialize_wiki_summary(self, summary: WikiSummary) -> dict[str, Any]:
        """Serialize WikiSummary to JSON-compatible dict"""
        return {
            "wiki_name": summary.wiki_name,
            "summary": summary.summary,
            "key_components": summary.key_components,
            "summary_confidence": getattr(summary, 'summary_confidence', None),
            "tokens_used": getattr(summary, 'tokens_used', None)
        }

    def _deserialize_wiki_summary(self, summary_data: dict[str, Any]) -> WikiSummary:
        """Deserialize WikiSummary from cached data"""
        return WikiSummary(
            wiki_name=summary_data["wiki_name"],
            summary=summary_data["summary"],
            key_components=summary_data.get("key_components", []),
            summary_confidence=summary_data.get("summary_confidence"),
            tokens_used=summary_data.get("tokens_used")
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the persistent cache"""
        try:
            with get_db_session() as db:
                total_entries = db.query(WikiSummaryCache).count()

                # Count by project
                project_counts = {}
                for project_code in PROJECT_WIKI_REPOS.keys():
                    count = db.query(WikiSummaryCache).filter(
                        WikiSummaryCache.project_code == project_code
                    ).count()
                    project_counts[project_code] = count

                # Count stale entries
                refresh_threshold = datetime.now() - timedelta(days=WIKI_CACHE_REFRESH_DAYS)
                stale_count = db.query(WikiSummaryCache).filter(
                    WikiSummaryCache.last_refreshed_at < refresh_threshold
                ).count()

                return {
                    "total_cached_summaries": total_entries,
                    "project_counts": project_counts,
                    "stale_summaries": stale_count,
                    "fresh_summaries": total_entries - stale_count,
                    "refresh_threshold_days": WIKI_CACHE_REFRESH_DAYS,
                    "llm_configured": self.client is not None
                }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def clear_cache(self, project_code: str | None = None) -> dict[str, Any]:
        """Clear persistent cache entries"""
        try:
            with get_db_session() as db:
                query = db.query(WikiSummaryCache)

                if project_code:
                    query = query.filter(WikiSummaryCache.project_code == project_code)

                deleted_count = query.count()
                query.delete()
                db.commit()

                logger.info(f"Cleared {deleted_count} wiki summary cache entries" +
                           (f" for project {project_code}" if project_code else ""))

                return {
                    "deleted_entries": deleted_count,
                    "project_filter": project_code,
                    "status": "success"
                }

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {"error": str(e), "status": "failed"}
