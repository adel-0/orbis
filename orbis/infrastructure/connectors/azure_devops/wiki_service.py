"""
Azure DevOps Wiki Service - Generic interface wrapper.
Provides the simple connector interface for wiki integration.
Wraps the existing AzureDevOpsWikiIntegrator to provide compatibility.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class WikiService:
    """Azure DevOps Wiki connector with generic interface"""

    def __init__(self):
        self.integrator = None

    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
        """
        Fetch wiki pages using configuration parameters.
        This implements the generic connector interface using the Azure DevOps wiki client.

        Note: Currently wiki sync is always full sync as Azure DevOps wiki API
        doesn't provide reliable incremental sync capabilities for wiki pages.
        """
        try:
            # Log sync mode
            if incremental:
                logger.info("Wiki incremental sync requested, but performing full sync (wiki API limitation)")
            else:
                logger.info("Performing full wiki sync")
            from infrastructure.connectors.azure_devops.azure_devops_wiki_client import (
                AzureDevOpsWikiClient,
            )

            # Use discovery mode unless specific included_wikis are provided
            included_wikis = config.get('included_wikis', [])
            excluded_wikis = config.get('excluded_wikis', [])

            if included_wikis:
                # Specific wikis mode - first discover wikis to get their IDs
                wiki_pages = []

                # Discover all wikis to get their IDs
                if config.get('auth_type') == 'oauth2':
                    discovered_wikis = await AzureDevOpsWikiClient.discover_wikis(
                        organization=config['organization'],
                        project=config['project'],
                        client_id=config['client_id'],
                        client_secret=config['client_secret'],
                        tenant_id=config['tenant_id'],
                        use_oauth2=True
                    )
                else:
                    discovered_wikis = await AzureDevOpsWikiClient.discover_wikis(
                        organization=config['organization'],
                        project=config['project'],
                        auth_token=config['pat'],
                        use_oauth2=False
                    )

                # Create a mapping of wiki names to wiki info
                wiki_lookup = {wiki['name']: wiki for wiki in discovered_wikis}

                # Fetch pages from each included wiki
                for wiki_name in included_wikis:
                    try:
                        if wiki_name not in wiki_lookup:
                            logger.warning(f"Wiki {wiki_name} not found in discovered wikis")
                            continue

                        wiki_info = wiki_lookup[wiki_name]

                        if config.get('auth_type') == 'oauth2':
                            client = AzureDevOpsWikiClient.create_for_discovered_wiki(
                                organization=config['organization'],
                                project=config['project'],
                                wiki_info=wiki_info,
                                client_id=config['client_id'],
                                client_secret=config['client_secret'],
                                tenant_id=config['tenant_id'],
                                use_oauth2=True
                            )
                        else:
                            client = AzureDevOpsWikiClient.create_for_discovered_wiki(
                                organization=config['organization'],
                                project=config['project'],
                                wiki_info=wiki_info,
                                auth_token=config['pat'],
                                use_oauth2=False
                            )

                        pages = await client.get_wiki_pages()
                        wiki_pages.extend(pages)
                    except Exception as e:
                        logger.warning(f"Failed to fetch pages from wiki {wiki_name}: {e}")
                        continue
            else:
                # Discovery mode - get all wikis
                if config.get('auth_type') == 'oauth2':
                    discovered_wikis = await AzureDevOpsWikiClient.discover_wikis(
                        organization=config['organization'],
                        project=config['project'],
                        client_id=config['client_id'],
                        client_secret=config['client_secret'],
                        tenant_id=config['tenant_id'],
                        use_oauth2=True
                    )
                else:
                    discovered_wikis = await AzureDevOpsWikiClient.discover_wikis(
                        organization=config['organization'],
                        project=config['project'],
                        auth_token=config['pat'],
                        use_oauth2=False
                    )

                # Filter wikis: exclude excluded_wikis, include only included_wikis if specified
                filtered_wikis = [wiki for wiki in discovered_wikis if wiki['name'] not in excluded_wikis]
                if included_wikis:
                    filtered_wikis = [wiki for wiki in filtered_wikis if wiki['name'] in included_wikis]

                # Get pages from all discovered wikis
                wiki_pages = []
                for wiki_info in filtered_wikis:
                    try:
                        if config.get('auth_type') == 'oauth2':
                            client = AzureDevOpsWikiClient.create_for_discovered_wiki(
                                organization=config['organization'],
                                project=config['project'],
                                wiki_info=wiki_info,
                                client_id=config['client_id'],
                                client_secret=config['client_secret'],
                                tenant_id=config['tenant_id'],
                                use_oauth2=True
                            )
                        else:
                            client = AzureDevOpsWikiClient.create_for_discovered_wiki(
                                organization=config['organization'],
                                project=config['project'],
                                wiki_info=wiki_info,
                                auth_token=config['pat'],
                                use_oauth2=False
                            )

                        pages = await client.get_wiki_pages()
                        wiki_pages.extend(pages)
                    except Exception as e:
                        logger.warning(f"Failed to fetch pages from wiki {wiki_info['name']}: {e}")
                        continue

            # Convert wiki pages to the expected format for generic interface
            wiki_data = []
            for page in wiki_pages:
                # Convert page data to generic format
                wiki_item = {
                    'id': page.get('id', ''),
                    'title': page.get('title', ''),
                    'content': page.get('content', ''),
                    'content_type': 'wiki_page',
                    'source_metadata': {
                        'path': page.get('path', ''),
                        'wiki_name': page.get('wiki_name', ''),
                        'page_id': page.get('id', ''),
                        'html_content': page.get('html_content', ''),
                        'image_references': page.get('image_references', [])
                    },
                    'extracted_metadata': {
                        'content_length': len(page.get('content', '')),
                        'image_count': len(page.get('image_references', [])),
                        'version': page.get('version', '1')
                    },
                    'last_modified': page.get('last_modified'),
                    'author': page.get('author')
                }
                wiki_data.append(wiki_item)

            # Summary logging to reduce verbosity
            wiki_count = len({page.get('source_metadata', {}).get('wiki_name', '') for page in wiki_data})
            total_chars = sum(len(page.get('content', '')) for page in wiki_data)
            logger.info(f"📊 Wiki ingestion complete: {len(wiki_data)} pages from {wiki_count} wikis, {total_chars:,} total characters")
            return wiki_data

        except Exception as e:
            logger.error(f"Failed to fetch wiki data: {str(e)}")
            raise

    def get_searchable_text(self, wiki_item: dict) -> str:
        """Convert wiki item to searchable text"""
        parts = []

        # Add title
        title = wiki_item.get('title', '')
        if title:
            parts.append(str(title))

        # Add content (markdown or plain text)
        content = wiki_item.get('content', '')
        if content:
            # Clean up markdown formatting for better searchability
            cleaned_content = self._clean_markdown_for_search(content)
            parts.append(cleaned_content)

        # Add any extracted text from source metadata
        source_metadata = wiki_item.get('source_metadata', {})
        if source_metadata.get('html_content'):
            # If we have HTML content, extract text from it
            html_text = self._extract_text_from_html(source_metadata['html_content'])
            if html_text and html_text != content:
                parts.append(html_text)

        return " ".join(parts)

    def get_metadata(self, wiki_item: dict) -> dict[str, Any]:
        """Return metadata that should be stored for filtering/display"""

        source_metadata = wiki_item.get('source_metadata', {})
        extracted_metadata = wiki_item.get('extracted_metadata', {})

        metadata = {
            'content_type': wiki_item.get('content_type', 'wiki_page'),
            'path': source_metadata.get('path', ''),
            'wiki_name': source_metadata.get('wiki_name', ''),
            'page_id': source_metadata.get('page_id', ''),
            'author': wiki_item.get('author', ''),
            'last_modified': self._format_datetime(wiki_item.get('last_modified')),
            'content_length': extracted_metadata.get('content_length', 0),
            'image_count': extracted_metadata.get('image_count', 0)
        }

        # Add image references if available
        image_references = source_metadata.get('image_references', [])
        if image_references:
            metadata['image_references'] = image_references

        # Include any additional metadata
        for key, value in extracted_metadata.items():
            if key not in ['content_length', 'image_count'] and value is not None:
                metadata[f'extracted_{key}'] = value

        return metadata

    def get_content_id(self, wiki_item: dict) -> str:
        """Return unique ID for this content item"""
        item_id = wiki_item.get('id', '')
        if item_id:
            return str(item_id)

        # Fallback: generate ID from path and title
        source_metadata = wiki_item.get('source_metadata', {})
        path = source_metadata.get('path', '')
        page_id = source_metadata.get('page_id', '')

        if page_id:
            return f"wiki_{page_id}"
        elif path:
            # Create ID from path
            import hashlib
            path_hash = hashlib.md5(path.encode()).hexdigest()[:12]
            return f"wiki_path_{path_hash}"
        else:
            # Last resort: hash of title and content
            title = wiki_item.get('title', '')
            content = wiki_item.get('content', '')
            content_str = f"{title}_{content}"[:200]  # Limit length
            content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]
            return f"wiki_content_{content_hash}"

    def _clean_markdown_for_search(self, markdown_content: str) -> str:
        """Clean markdown content for better searchability"""
        if not markdown_content:
            return ""

        import re

        # Remove markdown image syntax
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', markdown_content)

        # Remove Azure DevOps attachment syntax
        content = re.sub(r'\[\[attachment:([^\]]+)\]\]', r'\1', content)

        # Remove markdown links, keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

        # Remove markdown headers
        content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)

        # Remove markdown emphasis (bold, italic)
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
        content = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', content)

        # Remove code blocks and inline code
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)

        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        return content

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML content"""
        from orbis_core.utils.constants import clean_html_content
        return clean_html_content(html_content)

    def _format_datetime(self, dt) -> str:
        """Format datetime for metadata storage"""
        if dt is None:
            return ""

        if isinstance(dt, datetime):
            return dt.isoformat()
        elif isinstance(dt, str):
            return dt
        else:
            return str(dt)
