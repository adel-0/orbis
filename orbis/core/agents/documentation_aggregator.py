"""
Documentation Aggregator for Orbis

This is the second agent in the agentic RAG system that takes the search results from
the multi-modal search service and aggregates them into a comprehensive, actionable summary.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from core.schemas import ScopeAnalysisResult, SourceReference
from core.services.generic_multi_modal_search import GenericAggregatedSearchResult
from orbis_core.llm.openai_client import OpenAIClientService

logger = logging.getLogger(__name__)

class DocumentationAggregator:
    """
    Documentation Aggregator: Aggregates documentation from multiple sources and creates final summary.

    This aggregator receives the original content, scope analysis, and search results
    from all source types, then creates a comprehensive summary with actionable recommendations.
    """

    def __init__(self, openai_client_service: OpenAIClientService | None = None):
        # Use shared OpenAI client
        self.openai_client_service = openai_client_service or OpenAIClientService(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Aggregation configuration
        self.config = {
            "max_output_tokens": 50000,  # Maximum tokens for LLM response generation
            "max_sources_per_type": 5,  # Limit sources per type to manage context
            "max_input_tokens": 100000,  # Maximum input context capacity
            "debug_logging_enabled": True,  # Enable file logging of responses for debugging
            "debug_log_dir": "logs/documentation_aggregation"  # Directory for response debug logs
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
                logger.error("Azure OpenAI credentials not configured. Documentation Aggregator cannot function.")
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

    async def aggregate_and_summarize(self,
                                    original_content: str,
                                    scope_analysis: ScopeAnalysisResult,
                                    search_results: GenericAggregatedSearchResult) -> tuple[str, list[SourceReference], float]:
        """
        Main method for Documentation Aggregator: Aggregate documentation and create final summary.

        Args:
            original_content: Original content to process
            scope_analysis: Result from Scope Analyzer's analysis
            search_results: Aggregated search results from multi-modal search

        Returns:
            Tuple of (final_summary, source_references, overall_confidence)
        """
        if not self._check_client_available():
            logger.error("Documentation Aggregator not properly configured - Azure OpenAI client not available")
            return "Documentation Aggregator is not available due to configuration issues.", [], 0.0

        try:
            logger.info("📖 Documentation Aggregator: Starting documentation aggregation and summarization")
            # Step 1: Process and rank all search results
            ranked_sources = self._rank_and_filter_sources(search_results)

            if not ranked_sources:
                logger.warning("⚠️ No relevant sources found for aggregation")
                return self._create_fallback_summary(original_content, scope_analysis), [], 0.3

            # Step 2: Create source references for output
            source_references = self._create_source_references(ranked_sources)

            # Step 3: Build aggregation context
            aggregation_context = self._build_aggregation_context(
                original_content,
                scope_analysis,
                ranked_sources
            )
            # Step 4: Generate comprehensive summary
            logger.debug(f"Generating summary (max {self.config['max_output_tokens']} tokens)")
            final_summary = await self._generate_comprehensive_summary(aggregation_context)

            # Step 5: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                scope_analysis,
                search_results,
                len(ranked_sources)
            )
            logger.debug(f"Confidence: {overall_confidence:.2f}, sources: {len(source_references)}, summary: {len(final_summary)} chars")

            return final_summary, source_references, overall_confidence

        except Exception as e:
            logger.error(f"Aggregation error: {e}, using fallback")
            fallback_summary = self._create_fallback_summary(original_content, scope_analysis)
            return fallback_summary, [], 0.2

    def _rank_and_filter_sources(self, search_results: GenericAggregatedSearchResult) -> list[dict[str, Any]]:
        """
        Rank and filter sources from all search results to get the most relevant ones.

        Returns a list of source dictionaries with metadata for aggregation.
        """
        all_sources = []

        # Process results by source type
        for source_type, results in search_results.results_by_source_type.items():
            for result in results[:self.config["max_sources_per_type"]]:
                # Extract title from content or metadata
                title = self._extract_title(result)

                source_dict = {
                    "type": source_type,
                    "title": title,
                    "content": result.concatenated_text,
                    "score": getattr(result, 'rerank_score', None) or result.similarity_score,
                    "source_name": result.metadata.get('source_name', 'Unknown'),
                    "id": self._extract_content_id(result),
                    "url": self._generate_generic_url(result, source_type),
                    "metadata": result.metadata
                }
                all_sources.append(source_dict)

        # Also include reranked results if available (these are cross-collection ranked)
        if search_results.reranked_results:
            for result in search_results.reranked_results:
                title = self._extract_title(result)
                source_type = getattr(result, 'content_type', 'unknown')

                source_dict = {
                    "type": source_type,
                    "title": title,
                    "content": result.concatenated_text,
                    "score": getattr(result, 'rerank_score', None) or result.similarity_score,
                    "source_name": result.metadata.get('source_name', 'Unknown'),
                    "id": self._extract_content_id(result),
                    "url": self._generate_generic_url(result, source_type),
                    "metadata": result.metadata,
                    "is_reranked": True
                }
                all_sources.append(source_dict)

        # Remove duplicates (can occur when reranked results overlap with type-specific results)
        unique_sources = {}
        for source in all_sources:
            key = f"{source['type']}-{source['id']}"
            if key not in unique_sources or source.get('is_reranked', False):
                unique_sources[key] = source

        all_sources = list(unique_sources.values())

        # Sort by relevance score and return top sources
        all_sources.sort(key=lambda x: x["score"], reverse=True)

        # Limit total sources to manage context window
        max_total_sources = 15
        return all_sources[:max_total_sources]

    def _create_source_references(self, ranked_sources: list[dict[str, Any]]) -> list[SourceReference]:
        """Create SourceReference objects from ranked sources"""
        references = []

        for source in ranked_sources:
            # Create snippet from content (first 200 characters)
            snippet = source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"]

            reference = SourceReference(
                source_type=source["type"],
                source_name=source["source_name"],
                title=source["title"],
                relevance_score=source["score"],
                snippet=snippet,
                url=source.get("url")
            )
            references.append(reference)

        return references

    def _build_aggregation_context(self,
                                 original_content: str,
                                 scope_analysis: ScopeAnalysisResult,
                                 ranked_sources: list[dict[str, Any]]) -> str:
        """Build the context for LLM aggregation"""
        context_parts = []

        # Original ticket
        context_parts.append(f"=== ORIGINAL CONTENT ===\n{original_content}")

        # Ticket Scope Analysis
        scope_analysis_section = f"""=== SCOPE ANALYSIS ===
Scope: {scope_analysis.scope_description}
Intent: {scope_analysis.intent_description}
Confidence: {scope_analysis.confidence:.2f}
Recommended Sources: {', '.join(scope_analysis.recommended_source_types)}
"""
        context_parts.append(scope_analysis_section)

        # Documentation sources
        context_parts.append("=== RELEVANT DOCUMENTATION SOURCES ===")

        for i, source in enumerate(ranked_sources, 1):
            source_section = f"""
--- Source {i}: {source['type'].upper()} ---
Title: {source['title']}
Source: {source['source_name']}
Relevance Score: {source['score']:.2f}
Content:
{source['content'][:1500]}{'...' if len(source['content']) > 1500 else ''}
"""
            context_parts.append(source_section)

        full_context = "\n\n".join(context_parts)

        # Check context length and truncate if necessary
        if len(full_context) > self.config["max_input_tokens"]:
            logger.warning("Context exceeds limit, truncating...")
            # Keep the most important parts and truncate source content
            truncated_sources = []
            for source in ranked_sources[:8]:  # Keep fewer sources
                truncated_content = source['content'][:800] + "..." if len(source['content']) > 800 else source['content']
                truncated_sources.append({**source, 'content': truncated_content})

            # Rebuild with truncated content
            context_parts = context_parts[:2]  # Keep ticket and analysis
            context_parts.append("=== RELEVANT DOCUMENTATION SOURCES ===")
            for i, source in enumerate(truncated_sources, 1):
                source_section = f"""
--- Source {i}: {source['type'].upper()} ---
Title: {source['title']}
Source: {source['source_name']}
Relevance Score: {source['score']:.2f}
Content: {source['content']}
"""
                context_parts.append(source_section)

            full_context = "\n\n".join(context_parts)

        return full_context

    async def _generate_comprehensive_summary(self, aggregation_context: str) -> str:
        """Generate the comprehensive summary using LLM"""
        try:
            prompt = f"""
{aggregation_context}

TASK:
As the Documentation Aggregator in an agentic RAG system, analyze all the documentation sources above and create a comprehensive summary that helps resolve the original ticket.

Your summary should include:

1. **Problem Analysis**: What the ticket is about based on the scope analysis and the documentation found

2. **Relevant Solutions**: Specific solutions, patterns, or guidance found in the documentation sources
   - Reference specific tickets, wiki pages, code files, or documentation
   - Include any error codes, configuration steps, or code examples mentioned

3. **Actionable Recommendations**: Clear next steps the user should take
   - Prioritize recommendations based on relevance and the intent analysis
   - Include specific file paths, configuration parameters, or implementation steps where available

4. **Additional Context**: Any important background information from the sources that provides valuable context

5. **Source References**: Briefly mention which sources were most helpful (by title/type)

Format your response as a clear, structured summary that a technical support engineer can use immediately to help resolve the issue.

Focus on being practical and actionable rather than theoretical. If the documentation doesn't provide a clear solution, be honest about limitations and suggest concrete next steps for further investigation.
"""

            logger.debug(f"📖 LLM Request - Model: {self.deployment_name}, Max tokens: {self.config['max_output_tokens']}")
            logger.debug(f"📖 LLM Request - Context length: {len(aggregation_context)} characters")

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "developer",
                        "content": """You are the Documentation Aggregator in an Orbis support system. Your role is to synthesize information from multiple sources (tickets, wikis, code, PDFs) into actionable summaries that help resolve technical issues.

You excel at:
- Identifying the most relevant information across diverse sources
- Creating practical, step-by-step recommendations
- Connecting related issues and solutions from different documentation types
- Providing clear guidance for technical implementation
- Being honest about limitations when information is incomplete

Always structure your responses clearly and reference specific sources when making recommendations."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=self.config["max_output_tokens"]
            )

            summary = response.choices[0].message.content
            if not summary:
                logger.warning("📖 LLM returned empty response - using fallback")
                return self._create_fallback_summary_from_sources(aggregation_context)

            summary = summary.strip()
            logger.debug(f"📖 LLM Response - Generated summary: {len(summary)} characters")
            logger.debug(f"📖 LLM Response preview: {summary[:200]}...")

            # Log the full response for debugging
            context_name = "documentation_aggregation"
            self._log_response_to_file(summary, context_name, "aggregated_summary")

            return summary

        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            return self._create_fallback_summary_from_sources(aggregation_context)

    def _create_fallback_summary(self, original_content: str, scope_analysis: ScopeAnalysisResult) -> str:
        """Create a fallback summary when no sources are found"""
        return f"""**Problem Analysis:**
Based on the ticket analysis, this appears to concern {scope_analysis.scope_description.lower()}. The user's intent is to {scope_analysis.intent_description.lower()}.

**Analysis Confidence:** {scope_analysis.confidence:.0%}

**Recommended Next Steps:**
1. Search for similar issues in the recommended source types: {', '.join([t.replace('_', ' ') for t in scope_analysis.recommended_source_types])}
2. If this is a code-related issue, examine the relevant source repositories
3. If this is a code-related issue, examine the relevant source repositories
4. Consult with team members who have experience with the identified components

**Note:** Limited documentation was found for this specific issue. Manual investigation may be required to provide a complete solution.
"""

    def _create_fallback_summary_from_sources(self, context: str) -> str:
        """Create a basic summary when LLM generation fails"""
        # Extract key information from context
        lines = context.split('\n')

        # Find the scope and intent
        scope = "the reported issue"
        intent = "resolve a technical problem"

        for line in lines:
            if line.startswith("Scope:"):
                scope = line.split(":", 1)[1].strip()
            elif line.startswith("Intent:"):
                intent = line.split(":", 1)[1].strip()

        return f"""**Problem Analysis:**
This ticket concerns {scope}. The user wants to {intent}.

**Documentation Found:**
Multiple sources were found and analyzed, but an automated summary could not be generated due to a technical issue.

**Recommended Action:**
Please review the source references provided with this response for relevant documentation and solutions.

**Note:** This is a fallback summary. For detailed analysis, please contact technical support.
"""

    def _extract_title(self, result) -> str:
        """Extract title from search result content or metadata"""
        # Try various title fields
        if hasattr(result, 'content'):
            content = result.content
            if hasattr(content, 'title') and content.title:
                return content.title

        # Fall back to metadata
        if hasattr(result, 'metadata') and result.metadata:
            for field in ['title', 'Title', 'name', 'filename', 'path', 'url']:
                if field in result.metadata and result.metadata[field]:
                    title = str(result.metadata[field])
                    # Clean up the title if it's a path or URL
                    if title.startswith('http') and '/edit/' in title:
                        # Extract work item ID from Azure DevOps URLs
                        if 'workitem' in title:
                            import re
                            match = re.search(r'workitem_(\d+)', title)
                            if match:
                                return f"Work Item #{match.group(1)}"
                        if '_wiki' in title:
                            return "Wiki Documentation"
                    elif '/' in title and title.endswith('.md'):
                        # Extract meaningful name from file path
                        return title.split('/')[-1].replace('.md', '').replace('%2D', '-').replace('_', ' ')
                    return title

        # Try to extract from content snippet if available
        if hasattr(result, 'content') and hasattr(result.content, 'snippet'):
            snippet = result.content.snippet[:100]
            if snippet:
                # Look for headers or meaningful first lines
                lines = snippet.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('#'):
                        return line.lstrip('#').strip()
                    elif len(line) > 10 and len(line) < 80:
                        return line

        # Smart fallback based on content type
        if hasattr(result, 'metadata') and result.metadata:
            if 'workitems' in str(result.metadata).lower():
                return "Support Work Item"
            elif 'wiki' in str(result.metadata).lower():
                return "Wiki Documentation"

        # Ultimate fallback
        return "Documentation Reference"

    def _extract_content_id(self, result) -> str:
        """Extract content ID from search result"""
        # Try content ID first
        if hasattr(result, 'content'):
            content = result.content
            if hasattr(content, 'id') and content.id:
                return str(content.id)

        # Fall back to metadata
        if hasattr(result, 'metadata') and result.metadata:
            for field in ['content_id', 'id', 'Id', 'ID']:
                if field in result.metadata and result.metadata[field]:
                    return str(result.metadata[field])

        # Generate from hash if no ID found
        import hashlib
        content_str = result.concatenated_text[:100] if hasattr(result, 'concatenated_text') else 'unknown'
        return hashlib.md5(content_str.encode()).hexdigest()[:12]

    def _generate_generic_url(self, result, source_type: str) -> str | None:
        """Generate URL based on source type and metadata"""
        try:
            metadata = getattr(result, 'metadata', {}) or {}
            content = getattr(result, 'content', None)

            # Try to get URL from content object first
            if hasattr(result, 'content'):
                content = result.content

                # Try content-based URL generation first
                if source_type in ['azdo_workitems', 'workitem']:
                    url = self._generate_workitem_url(content, metadata)
                    if url:
                        return url
                elif source_type in ['azdo_wiki', 'project_wiki', 'wiki']:
                    url = self._generate_wiki_url(content, metadata)
                    if url:
                        return url
                elif source_type in ['azdo_code', 'code']:
                    url = self._generate_code_url(content, metadata)
                    if url:
                        return url

            # Fallback to metadata-based URL generation
            if source_type in ['azdo_workitems', 'workitem']:
                url = self._generate_workitem_url_from_metadata(metadata)
                if url:
                    return url
            elif source_type in ['azdo_wiki', 'project_wiki', 'wiki']:
                url = self._generate_wiki_url_from_metadata(metadata)
                if url:
                    return url
            else:
                # Generic URL construction
                if 'url' in metadata:
                    return metadata['url']

        except Exception as e:
            logger.debug(f"Error generating URL for {source_type}: {e}")

        return None

    def _generate_workitem_url_from_metadata(self, metadata: dict[str, Any]) -> str | None:
        """Generate work item URL from metadata"""
        try:
            # Extract organization and project from metadata (these come from the connector config)
            org = metadata.get('organization', '')
            project = metadata.get('project', '')

            # Extract the work item ID - try multiple possible field names
            content_id = metadata.get('content_id') or metadata.get('id') or metadata.get('original_id', '')

            # Clean the content_id to remove any prefixes like "workitem_"
            if content_id and str(content_id).startswith('workitem_'):
                content_id = str(content_id).replace('workitem_', '')

            # Clean up values
            org = str(org).strip() if org else ''
            project = str(project).strip() if project else ''
            content_id = str(content_id).strip() if content_id else ''

            if org and project and content_id:
                # URL encode the project name for Azure DevOps URLs
                import urllib.parse
                encoded_project = urllib.parse.quote(project, safe='')
                return f"https://dev.azure.com/{org}/{encoded_project}/_workitems/edit/{content_id}"
            elif org and content_id:
                # Fallback for cases without project
                return f"https://dev.azure.com/{org}/_workitems/edit/{content_id}"
        except Exception:
            pass
        return None

    def _generate_wiki_url_from_metadata(self, metadata: dict[str, Any]) -> str | None:
        """Generate wiki URL from metadata"""
        try:
            # Extract organization and project from metadata (these come from the connector config)
            org = metadata.get('organization', '')
            project = metadata.get('project', '')
            path = metadata.get('path') or metadata.get('original_id', '')

            # Clean up values
            org = str(org).strip() if org else ''
            project = str(project).strip() if project else ''
            path = str(path).strip() if path else ''

            if org and project and path:
                import urllib.parse
                encoded_project = urllib.parse.quote(project, safe='')

                # For Azure DevOps wiki URLs, try to construct the proper format
                # Extract wiki name from path (e.g., "SG" from path that starts with "/SG/...")
                if path.startswith('/'):
                    path_parts = path.split('/')
                    if len(path_parts) > 1:
                        wiki_name = path_parts[1]  # e.g., "SG" from "/SG/ÜL-NEZ/Umsysteme/DIASweb.md"
                        # Get the page name from the last part of the path
                        path_parts[-1].replace('.md', '') if path_parts[-1].endswith('.md') else path_parts[-1]

                        # For wiki URLs, we need the numeric page ID. Since we don't have it in metadata,
                        # we'll try to construct a git-based URL instead
                        encoded_path = urllib.parse.quote(path, safe='/%')
                        return f"https://dev.azure.com/{org}/{encoded_project}/_git/Wiki.{wiki_name}?path={encoded_path}"

                # Fallback: try generic wiki URL construction
                encoded_path = urllib.parse.quote(path, safe='/%')
                return f"https://dev.azure.com/{org}/{encoded_project}/_git/Wiki?path={encoded_path}"

        except Exception:
            pass
        return None


    def _calculate_overall_confidence(self,
                                    scope_analysis: ScopeAnalysisResult,
                                    search_results: GenericAggregatedSearchResult,
                                    num_sources: int) -> float:
        """Calculate overall confidence in the aggregated response"""
        # Base confidence from scope analysis
        base_confidence = scope_analysis.confidence

        # Boost confidence based on number and quality of sources found
        source_boost = 0.0
        if num_sources > 0:
            # More sources generally mean better coverage
            source_boost = min(0.2, num_sources * 0.02)

            # Quality boost based on search result types
            results_by_type = search_results.results_by_source_type
            if any('workitem' in source_type or 'azdo_workitems' in source_type for source_type in results_by_type.keys()):
                source_boost += 0.1  # Work items are very relevant
            if any('wiki' in source_type or 'project_wiki' in source_type for source_type in results_by_type.keys()):
                source_boost += 0.05  # Wiki provides good context
            if any('code' in source_type for source_type in results_by_type.keys()):
                source_boost += 0.1  # Code results when needed are valuable
            if any('pdf' in source_type or 'document' in source_type for source_type in results_by_type.keys()):
                source_boost += 0.05  # PDF docs provide official guidance

        # Calculate final confidence (capped at 0.95)
        overall_confidence = min(0.95, base_confidence + source_boost)

        return overall_confidence

    def _generate_workitem_url(self, content, metadata: dict[str, Any] = None) -> str | None:
        """Generate URL for work item (Azure DevOps ticket)"""
        try:
            # Prefer metadata values as they come from the connector configuration
            org = (metadata.get('organization') if metadata else '') or getattr(content, 'organization', '')
            project = (metadata.get('project') if metadata else '') or getattr(content, 'project', '')
            ticket_id = getattr(content, 'id', '') or (metadata.get('content_id') if metadata else '') or (metadata.get('original_id') if metadata else '')

            # Clean the ticket_id to remove any prefixes like "workitem_"
            if ticket_id and str(ticket_id).startswith('workitem_'):
                ticket_id = str(ticket_id).replace('workitem_', '')

            # Clean up values
            org = str(org).strip() if org else ''
            project = str(project).strip() if project else ''
            ticket_id = str(ticket_id).strip() if ticket_id else ''

            if org and project and ticket_id:
                # Azure DevOps URL with project
                import urllib.parse
                encoded_project = urllib.parse.quote(project, safe='')
                return f"https://dev.azure.com/{org}/{encoded_project}/_workitems/edit/{ticket_id}"
            elif org and ticket_id:
                # Fallback without project
                return f"https://dev.azure.com/{org}/_workitems/edit/{ticket_id}"
        except Exception:
            pass
        return None

    def _generate_wiki_url(self, content, metadata: dict[str, Any] = None) -> str | None:
        """Generate URL for wiki page"""
        try:
            # Prefer metadata values as they come from the connector configuration
            org = (metadata.get('organization') if metadata else '') or getattr(content, 'organization', '')
            project = (metadata.get('project') if metadata else '') or getattr(content, 'project', '')
            path = getattr(content, 'path', '') or (metadata.get('path') if metadata else '') or (metadata.get('original_id') if metadata else '')

            # Clean up values
            org = str(org).strip() if org else ''
            project = str(project).strip() if project else ''
            path = str(path).strip() if path else ''

            if org and project and path:
                import urllib.parse
                encoded_project = urllib.parse.quote(project, safe='')

                # For Azure DevOps wiki URLs, construct git-based URLs
                # Extract wiki name from path (e.g., "SG" from path that starts with "/SG/...")
                if path.startswith('/'):
                    path_parts = path.split('/')
                    if len(path_parts) > 1:
                        wiki_name = path_parts[1]  # e.g., "SG" from "/SG/ÜL-NEZ/Umsysteme/DIASweb.md"

                        # Construct git-based wiki URL
                        encoded_path = urllib.parse.quote(path, safe='/%')
                        return f"https://dev.azure.com/{org}/{encoded_project}/_git/Wiki.{wiki_name}?path={encoded_path}"

                # Fallback: try generic wiki URL construction
                encoded_path = urllib.parse.quote(path, safe='/%')
                return f"https://dev.azure.com/{org}/{encoded_project}/_git/Wiki?path={encoded_path}"

        except Exception:
            pass
        return None

    def _generate_code_url(self, content, metadata: dict[str, Any] = None) -> str | None:
        """Generate URL for source code file"""
        try:
            # Try content first, then fallback to metadata
            org = getattr(content, 'organization', '') or (metadata.get('organization') if metadata else '')
            project = getattr(content, 'project', '') or (metadata.get('project') if metadata else '')
            file_path = getattr(content, 'file_path', '') or (metadata.get('path') if metadata else '') or (metadata.get('original_id') if metadata else '')
            repo_name = getattr(content, 'repository_name', project)  # Default to project name if no repo specified

            # Clean up values
            org = str(org).strip() if org else ''
            project = str(project).strip() if project else ''
            file_path = str(file_path).strip() if file_path else ''

            if org and project and file_path:
                # Azure DevOps repository URL structure with project and repo
                import urllib.parse
                encoded_project = urllib.parse.quote(project, safe='')
                encoded_path = urllib.parse.quote(file_path, safe='/')
                return f"https://dev.azure.com/{org}/{encoded_project}/_git/{repo_name}?path={encoded_path}"
            elif org and file_path:
                # Fallback for basic repository structure
                import urllib.parse
                encoded_path = urllib.parse.quote(file_path, safe='/')
                return f"https://dev.azure.com/{org}/_git/repos?path={encoded_path}"
        except Exception:
            pass
        return None

    def _generate_pdf_url(self, content) -> str | None:
        """Generate URL for PDF document"""
        # For now, return None as PDF URLs depend on storage location
        return None

    def is_configured(self) -> bool:
        """Check if Documentation Aggregator is properly configured"""
        return self.client is not None
