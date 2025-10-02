"""
Azure DevOps Wiki REST API client for wiki content ingestion.
Provides functionality to fetch wiki pages, process markdown content, and extract images.
"""

import logging
import re
from typing import Any
from urllib.parse import quote

import aiohttp
import markdown
from bs4 import BeautifulSoup

from infrastructure.connectors.azure_devops.auth import AzureDevOpsAuthMixin
from utils.constants import DEFAULT_RATE_LIMIT_DELAY

from .constants import API_VERSIONS

logger = logging.getLogger(__name__)

class AzureDevOpsWikiClient(AzureDevOpsAuthMixin):
    """Client for Azure DevOps Wiki REST API operations"""

    def __init__(self, organization: str, project: str, **kwargs):
        super().__init__()

        if not organization or not project:
            raise ValueError("Organization and project are required")

        self.organization = organization
        self.project = project
        self.wiki_name = kwargs.get('wiki_name')
        self.wiki_id = kwargs.get('wiki_id')
        self.repository_id = kwargs.get('repository_id')
        self.use_oauth2 = kwargs.get('use_oauth2', False)

        # API versions from constants
        self.wiki_api_version = API_VERSIONS['wiki']
        self.git_api_version = API_VERSIONS['git']

        # Authentication setup
        if self.use_oauth2:
            self.client_id = kwargs.get('client_id')
            self.client_secret = kwargs.get('client_secret')
            self.tenant_id = kwargs.get('tenant_id')
        else:
            self.pat = kwargs.get('auth_token')

        # Base URLs
        self.base_url = self._build_base_url()
        self.wiki_base_url = self._build_wiki_url()

        # Wiki exclusion list from data source config
        self.excluded_wikis = kwargs.get('excluded_wikis', [])

        # Rate limiting (can be overridden via config)
        self.rate_limit_delay = kwargs.get('rate_limit_delay', DEFAULT_RATE_LIMIT_DELAY)

    @property
    def is_code_wiki(self) -> bool:
        """Check if this is a code wiki based on naming pattern"""
        return self.wiki_name and self.wiki_name.endswith('.wiki') and not self.wiki_name.startswith('HxGN-')

    def _build_base_url(self) -> str:
        """Build base URL with proper encoding"""
        encoded_project = quote(self.project) if ' ' in self.project else self.project
        return f"https://dev.azure.com/{self.organization}/{encoded_project}"

    def _build_wiki_url(self) -> str:
        """Build wiki base URL"""
        if self.wiki_id:
            return f"{self.base_url}/_apis/wiki/wikis/{self.wiki_id}"
        elif self.wiki_name:
            return f"{self.base_url}/_apis/wiki/wikis/{self.wiki_name}"
        return None

    async def _discover_via_wiki_api(self, session, auth_headers, organization: str, encoded_project: str) -> list[dict[str, Any]]:
        """Discover wikis using the Wiki API"""
        discovered_wikis = []
        wiki_url = f"https://dev.azure.com/{organization}/{encoded_project}/_apis/wiki/wikis"
        wiki_params = {"api-version": API_VERSIONS['wiki']}

        async with session.get(wiki_url, headers=auth_headers, params=wiki_params) as response:
            if response.status != 200:
                return discovered_wikis

            data = await response.json()
            wikis = data.get("value", [])

            # Get excluded wiki names from data source config (passed via constructor or defaults to empty)
            excluded_wiki_names = getattr(self, 'excluded_wikis', [])

            for wiki in wikis:
                wiki_name = wiki.get("name", "")
                if wiki_name in excluded_wiki_names:
                    continue

                # Determine wiki type based on API response structure
                repository_data = wiki.get("repository", {})
                has_repository = bool(repository_data)
                repository_id = repository_data.get("id") if has_repository else None
                remote_url = repository_data.get("remoteUrl", "") if has_repository else ""

                wiki_info = {
                    "name": wiki_name,
                    "url": remote_url,
                    "wiki_id": wiki.get("id"),
                    "repository_id": repository_id
                }

                if has_repository:
                    # This is a code wiki (has repository backing)
                    project_code = wiki_name[5:] if wiki_name.startswith("Wiki.") else wiki_name.replace('.wiki', '')
                    wiki_info.update({
                        "type": "code_wiki",
                        "data_source_type": "code_wiki",
                        "project_code": project_code,
                        "description": f"Code wiki repository: {wiki_name}"
                    })
                else:
                    # This is a project wiki (no repository, built-in to project)
                    wiki_info.update({
                        "type": "project_wiki",
                        "data_source_type": "project_wiki",
                        "description": f"Project wiki: {wiki_name}"
                    })

                discovered_wikis.append(wiki_info)

        return discovered_wikis

    async def _find_repository(self, session, possible_names: list[str]) -> dict[str, Any] | None:
        """Find repository by trying multiple possible names"""
        git_url = f"https://dev.azure.com/{self.organization}/{self.project}/_apis/git/repositories"
        auth_headers = await self._get_auth_headers()
        params = {"api-version": API_VERSIONS['wiki']}

        async with session.get(git_url, headers=auth_headers, params=params) as response:
            if response.status != 200:
                return None

            repos_data = await response.json()
            repos = repos_data.get("value", [])

            for repo in repos:
                if repo.get("name") in possible_names:
                    return repo
            return None

    @classmethod
    def from_data_source(cls, data_source):
        """Create client from DataSource model"""
        from infrastructure.data_processing.data_source_service import DataSourceService

        ds_service = DataSourceService()

        # Use the wiki_name from the data source if provided
        # This allows us to create clients for specific discovered wikis
        wiki_name = data_source.config.get('wiki_name')
        auth_type = data_source.config.get('auth_type', 'pat')

        if auth_type == "pat":
            decrypted_pat = ds_service._decrypt(data_source.config.get('pat'))
            return cls(
                organization=data_source.config.get('organization'),
                project=data_source.config.get('project'),
                auth_token=decrypted_pat,
                wiki_name=wiki_name,
                use_oauth2=False
            )
        elif auth_type == "oauth2":
            decrypted_client_secret = ds_service._decrypt(data_source.config.get('client_secret'))
            return cls(
                organization=data_source.config.get('organization'),
                project=data_source.config.get('project'),
                client_id=data_source.config.get('client_id'),
                client_secret=decrypted_client_secret,
                tenant_id=data_source.config.get('tenant_id'),
                wiki_name=wiki_name,
                use_oauth2=True
            )
        else:
            raise ValueError(f"Unsupported auth_type: {auth_type}")

    @classmethod
    async def discover_wikis(cls, organization: str, project: str, auth_token: str = None,
                           client_id: str = None, client_secret: str = None,
                           tenant_id: str = None, use_oauth2: bool = False) -> list[dict[str, Any]]:
        """Discover all Wiki repositories in a project that start with 'Wiki.'"""
        discovered_wikis = []

        try:

            # Encode project name if needed
            encoded_project = quote(project) if ' ' in project else project

            # Create a temporary client to access the APIs
            temp_client = cls(
                organization=organization,
                project=encoded_project,
                auth_token=auth_token,
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                use_oauth2=use_oauth2,
                wiki_name="temp"
            )

            async with aiohttp.ClientSession() as session:
                auth_headers = await temp_client._get_auth_headers()

                if auth_headers:
                    # Try Wiki API first (finds published wikis)
                    discovered_wikis = await temp_client._discover_via_wiki_api(session, auth_headers, organization, encoded_project)

                    # Always also check Git API for unpublished wiki repositories
                    # This ensures we find repositories like Wiki.SG, Wiki.VS that exist but aren't published as wikis yet
                    url = f"{temp_client.base_url}/_apis/git/repositories"
                    params = {"api-version": temp_client.git_api_version}

                    async with session.get(url, headers=auth_headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            repositories = data.get("value", [])

                            # Get excluded wiki names from config
                            excluded_wiki_names = getattr(temp_client, 'excluded_wikis', [])
                            if excluded_wiki_names:
                                logger.info(f"Excluding wikis: {excluded_wiki_names}")

                            len(discovered_wikis)

                            # Look for Wiki.* repositories
                            for repo in repositories:
                                repo_name = repo.get("name", "")
                                repo_url = repo.get("remoteUrl", "")

                                # Check if this wiki is excluded by configuration
                                if repo_name in excluded_wiki_names:
                                    continue

                                # Skip if already found in Wiki API
                                already_found = any(wiki['name'] == repo_name for wiki in discovered_wikis)
                                if already_found:
                                    continue

                                # Detect Wiki repositories - multiple patterns
                                is_wiki = False
                                wiki_type = "git_wiki"
                                project_code = None

                                # Pattern 1: Ends with .wiki (most common)
                                if repo_name.endswith(".wiki"):
                                    is_wiki = True
                                    project_code = repo_name.replace('.wiki', '')

                                # Pattern 2: Contains 'wiki' in name (case insensitive)
                                elif "wiki" in repo_name.lower():
                                    # Additional checks to avoid false positives
                                    if any(keyword in repo_name.lower() for keyword in ['wiki', 'documentation', 'docs']):
                                        # Skip if it's clearly not a wiki (e.g., 'wiki-tools', 'documentation-generator')
                                        if not any(skip in repo_name.lower() for skip in ['tool', 'generator', 'helper', 'utility', 'service']):
                                            is_wiki = True
                                            # Extract project code by removing common wiki suffixes
                                            if repo_name.lower().endswith('.wiki'):
                                                project_code = repo_name[:-5]  # Remove '.wiki' suffix
                                            elif repo_name.lower().startswith('wiki.'):
                                                project_code = repo_name[5:]  # Remove 'wiki.' prefix
                                            else:
                                                project_code = repo_name.replace('wiki', '').replace('Wiki', '').strip('._-')
                                            wiki_type = "potential_wiki"

                                # Pattern 3: Follows Wiki.ProjectName convention
                                elif repo_name.startswith("Wiki."):
                                    is_wiki = True
                                    project_code = repo_name[5:] if len(repo_name) > 5 else repo_name
                                    wiki_type = "wiki_convention"

                                if is_wiki:
                                    # CRITICAL FIX: For Git repositories that are wikis, handle both published and unpublished wikis
                                    try:
                                        repo_id = repo.get("id")
                                        if repo_id:
                                            # Try to access the wiki via the Wiki API first (for published wikis)
                                            wiki_test_url = f"{temp_client.base_url}/_apis/wiki/wikis/{repo_id}"
                                            test_params = {"api-version": API_VERSIONS['wiki']}

                                            wiki_id = None
                                            api_wiki_name = None
                                            is_published_wiki = False

                                            try:
                                                async with session.get(wiki_test_url, headers=auth_headers, params=test_params) as test_response:
                                                    if test_response.status == 200:
                                                        wiki_data = await test_response.json()
                                                        wiki_id = wiki_data.get("id")
                                                        api_wiki_name = wiki_data.get("name")
                                                        is_published_wiki = True
                                            except Exception:
                                                # Wiki API test failed, this might be an unpublished wiki repository
                                                pass

                                            # Use the actual wiki name from the API response if available, otherwise use repo name
                                            wiki_name = api_wiki_name if api_wiki_name else repo_name

                                            # For published wikis, use the wiki_id from the API
                                            # For unpublished wikis, we'll use the repository for Git-based access
                                            if is_published_wiki and wiki_id:
                                                effective_wiki_id = wiki_id
                                                data_source_type = "published_wiki"
                                            else:
                                                # This is an unpublished Git repository that serves as a wiki
                                                effective_wiki_id = None  # Don't set wiki_id for unpublished wikis
                                                data_source_type = "git_wiki"
                                                logger.debug(f"Found unpublished wiki repository: {repo_name}")

                                            wiki_info = {
                                                "name": wiki_name,
                                                "url": repo_url,
                                                "type": wiki_type,
                                                "data_source_type": data_source_type,
                                                "project_code": project_code,
                                                "description": f"{wiki_type.replace('_', ' ').title()}: {repo_name}",
                                                "repository_id": repo_id,
                                                "wiki_id": effective_wiki_id  # Only set for published wikis
                                            }
                                            discovered_wikis.append(wiki_info)
                                    except Exception as e:
                                        # Skip this repository if we can't process it
                                        logger.warning(f"Failed to process potential wiki repository {repo_name}: {e}")
                                        continue

                            len(discovered_wikis)

                            # Log summary of all discovered wikis - condensed format
                            if discovered_wikis:
                                wiki_names = [f"{wiki['name']} ({wiki['type']})" for wiki in discovered_wikis]
                                logger.info(f"📋 Discovered {len(discovered_wikis)} wikis: {', '.join(wiki_names[:5])}{'...' if len(discovered_wikis) > 5 else ''}")
                        else:
                            error_text = await response.text()
                            logger.error(f"Git API failed: {response.status} - {error_text}")
                            logger.error(f"Response headers: {dict(response.headers)}")
                            logger.error(f"Request URL: {url}")
                            logger.error(f"Request params: {params}")
                else:
                    logger.error(f"Failed to get authentication headers for {project}")
        except Exception as e:
            logger.error(f"Error discovering wikis: {e}", exc_info=True)
            raise

        # Log breakdown of discovered wikis by type
        wiki_types = {}
        for wiki in discovered_wikis:
            wiki_type = wiki.get('type', 'unknown')
            wiki_types[wiki_type] = wiki_types.get(wiki_type, 0) + 1

        for wiki_type, count in wiki_types.items():
            logger.info(f"Found {count} {wiki_type} wiki(s)")

        return discovered_wikis

    @classmethod
    def create_for_discovered_wiki(cls, organization: str, project: str, wiki_info: dict[str, Any],
                                 auth_token: str = None, client_id: str = None, client_secret: str = None,
                                 tenant_id: str = None, use_oauth2: bool = False) -> 'AzureDevOpsWikiClient':
        """Create a client instance for a specific discovered wiki"""
        wiki_name = wiki_info.get('name', '')
        wiki_id = wiki_info.get('wiki_id')
        repository_id = wiki_info.get('repository_id')

        if not wiki_name:
            raise ValueError("Wiki name is required")

        # CRITICAL FIX: Always use the actual wiki name for the client
        # This ensures proper wiki type detection (e.g., SG.wiki vs UUID)
        # The wiki_id will be used internally for API calls when needed


        # Handle project names with spaces
        encoded_project = quote(project) if ' ' in project else project

        if use_oauth2:
            return cls(
                organization=organization,
                project=encoded_project,
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                wiki_name=wiki_name,  # Always use the actual wiki name
                wiki_id=wiki_id,      # Pass wiki_id for API calls
                repository_id=repository_id,  # Pass repository_id for code wikis
                use_oauth2=True
            )
        else:
            return cls(
                organization=organization,
                project=encoded_project,
                auth_token=auth_token,
                wiki_name=wiki_name,  # Always use the actual wiki name
                wiki_id=wiki_id,      # Pass wiki_id for API calls
                repository_id=repository_id,  # Pass repository_id for code wikis
                use_oauth2=False
            )

    # Authentication methods inherited from AzureDevOpsAuthMixin
    # _get_auth_headers() and _refresh_oauth_token() are now provided by the mixin

    async def get_wiki_pages(self, path: str = "", recursive: bool = True) -> list[dict[str, Any]]:
        """Get all wiki pages from the configured wiki using the appropriate API"""
        # Validate that we have a specific wiki to work with
        if not self.wiki_name:
            raise ValueError("Wiki name is required to fetch wiki pages. Use discover_wikis() for auto-discovery.")

        # If we have a wiki_id, try the Wiki API first (for published wikis)
        if self.wiki_id:
            try:
                return await self._get_wiki_pages_via_api(path, recursive)
            except Exception as e:
                logger.warning(f"Wiki API failed for {self.wiki_name}: {e}")

                # If Wiki API fails, fall back to Git API
                logger.info(f"Attempting to access {self.wiki_name} as Git-based wiki repository")
                return await self._get_code_wiki_pages(path, recursive)
        else:
            # No wiki_id means this is an unpublished Git-based wiki, use Git API directly
            logger.info(f"Accessing {self.wiki_name} as unpublished Git-based wiki repository")
            return await self._get_code_wiki_pages(path, recursive)

    async def _get_wiki_pages_via_api(self, path: str = "", recursive: bool = True) -> list[dict[str, Any]]:
        """Get wiki pages using the Azure DevOps Wiki API (works for both code and project wikis)"""

        async with aiohttp.ClientSession() as session:
            auth_headers = await self._get_auth_headers()

            # Use the Wiki API endpoint - works for both code and project wikis
            wiki_pages_url = f"{self.base_url}/_apis/wiki/wikis/{self.wiki_id}/pages"

            # Set up parameters according to Microsoft documentation
            params = {
                "api-version": API_VERSIONS['wiki'],
                "path": path if path else "/",  # Root path if none specified
                "recursionLevel": "Full" if recursive else "OneLevel",
                "includeContent": "true"  # Get page content
            }

            logger.info(f"Fetching wiki pages from {wiki_pages_url} with params: {params}")

            async with session.get(wiki_pages_url, headers=auth_headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to get wiki pages: {response.status} - {error_text}")
                    logger.error(f"URL: {wiki_pages_url}")
                    logger.error(f"Params: {params}")

                    # For 404 errors (wiki not found), raise exception to trigger Git API fallback
                    if response.status == 404:
                        raise ValueError(f"Wiki not found via Wiki API: {error_text}")

                    return []

                data = await response.json()

                # Handle both single page and multiple pages response
                if isinstance(data, dict) and 'subPages' in data:
                    # Single page with subpages
                    pages = [data] + data.get('subPages', [])
                elif isinstance(data, dict) and 'value' in data:
                    # Multiple pages response
                    pages = data['value']
                elif isinstance(data, list):
                    # Direct list of pages
                    pages = data
                else:
                    # Single page
                    pages = [data] if data else []

                # Process each page
                processed_pages = []
                for page_data in pages:
                    processed_page = await self._process_wiki_page(page_data)
                    if processed_page:
                        processed_pages.append(processed_page)

                logger.info(f"Successfully fetched {len(processed_pages)} wiki pages")
                return processed_pages

    async def _process_wiki_page(self, page_data: dict) -> dict:
        """Process a single wiki page from the API response"""
        try:
            page_path = page_data.get("path", "")
            page_title = page_data.get("title") or page_path.split("/")[-1] if page_path else "Unknown"
            page_id = page_data.get("id", f"wiki_{hash(page_path)}")
            content = page_data.get("content", "")

            if not content:
                logger.warning(f"Page {page_path} has no content - skipping")
                return None

            # Process markdown and extract images
            html_content, image_references = self._process_markdown_content(content)

            result = {
                "id": str(page_id),
                "path": page_path,
                "title": page_title,
                "content": content,
                "html_content": html_content,
                "image_references": image_references,
                "author": page_data.get("lastModifiedBy", {}).get("displayName"),
                "last_modified": page_data.get("lastModifiedDate"),
                "version": page_data.get("version", "1")
            }

            logger.debug(f"Successfully processed page {page_path} with {len(content)} characters of content")
            return result

        except Exception as e:
            logger.error(f"Error processing wiki page {page_data.get('path', 'unknown')}: {e}", exc_info=True)
            return None

    async def _get_code_wiki_pages(self, path: str = "", recursive: bool = True) -> list[dict[str, Any]]:
        """Get pages from a code wiki using the Git API"""

        async with aiohttp.ClientSession() as session:
            auth_headers = await self._get_auth_headers()

            # Use repository_id if available, otherwise find repository by name
            repo_id = self.repository_id
            if not repo_id:
                repo_name = self.wiki_name.replace('.wiki', '')
                possible_repo_names = [repo_name, f"Wiki.{repo_name}", self.wiki_name]

                target_repo = await self._find_repository(session, possible_repo_names)
                if not target_repo:
                    logger.error(f"Repository for wiki {self.wiki_name} not found")
                    return []
                repo_id = target_repo.get("id")

            if not repo_id:
                logger.error(f"Repository ID not available for wiki {self.wiki_name}")
                return []

            # Get the repository items (files and folders)
            items_url = f"https://dev.azure.com/{self.organization}/{self.project}/_apis/git/repositories/{repo_id}/items"
            items_params = {
                "api-version": API_VERSIONS['git'],
                "path": path,
                "recursionLevel": "full" if recursive else "oneLevel"
            }

            async with session.get(items_url, headers=auth_headers, params=items_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to get repository items: {response.status} - {error_text}")
                    return []

                items_data = await response.json()
                items = items_data.get("value", [])

                # Filter for markdown files
                markdown_files = [item for item in items if item.get("path", "").endswith(('.md', '.markdown'))]



                if not markdown_files:
                    logger.warning(f"No markdown files found in repository {repo_id} for wiki {self.wiki_name}")
                    return []

                # Process each markdown file
                processed_pages = []
                for file_item in markdown_files:
                    try:
                        file_path = file_item.get("path", "")

                        # Get the file content
                        content_url = f"https://dev.azure.com/{self.organization}/{self.project}/_apis/git/repositories/{repo_id}/items"
                        content_params = {
                            "api-version": API_VERSIONS['git'],
                            "path": file_path,
                            "includeContent": "true"
                        }

                        async with session.get(content_url, headers=auth_headers, params=content_params) as content_response:
                            if content_response.status != 200:
                                logger.warning(f"Failed to get content for {file_path}: {content_response.status}")
                                continue

                            content = await content_response.text()
                            if not content:
                                logger.warning(f"Empty content for file {file_path}")
                                continue

                            # Process the markdown content
                            html_content, image_references = self._process_markdown_content(content)
                            title = self._extract_title_from_path(file_path)

                            processed_page = {
                                "id": str(file_item.get("objectId", "")),
                                "path": file_path,
                                "title": title,
                                "content": content,
                                "html_content": html_content,
                                "image_references": image_references,
                                "author": file_item.get("latestChange", {}).get("author", {}).get("name", "Unknown"),
                                "last_modified": file_item.get("latestChange", {}).get("timestamp"),
                                "version": str(file_item.get("latestChange", {}).get("changeId", "")),
                                "wiki_name": self.wiki_name  # Add wiki name for proper identification
                            }

                            processed_pages.append(processed_page)

                    except Exception as e:
                        logger.error(f"Error processing file {file_item.get('path', 'unknown')}: {e}")
                        continue

                logger.info(f"📄 Processed {len(processed_pages)} wiki pages from repository {self.wiki_name}")
                return processed_pages

    async def _get_project_wiki_pages(self, path: str = "", recursive: bool = True) -> list[dict[str, Any]]:
        """Get pages from a project wiki using the Wiki API"""

        async with aiohttp.ClientSession() as session:
            auth_headers = await self._get_auth_headers()

            # Get page list with content included
            url = f"{self.wiki_base_url}/pages"
            params = {
                "api-version": self.wiki_api_version,
                "recursionLevel": "full" if recursive else "oneLevel",
                "includeContent": "true"
            }

            if path:
                params["path"] = path


            async with session.get(url, headers=auth_headers, params=params) as response:
                logger.info(f"Wiki pages API response status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    pages = data.get("value", [])
                    logger.info(f"Found {len(pages)} total pages in wiki {self.wiki_name}")

                    # Filter out parent pages (they don't have content)
                    content_pages = [page for page in pages if page.get("content")]
                    logger.info(f"Found {len(content_pages)} content pages (excluding parent pages)")

                    if not content_pages:
                        logger.warning(f"No content pages found in wiki {self.wiki_name}")
                        return []

                    # Process each page
                    processed_pages = []
                    for page_info in content_pages:
                        processed_page = self._process_wiki_page_direct(page_info)
                        if processed_page:
                            processed_pages.append(processed_page)

                    logger.info(f"Successfully processed {len(processed_pages)} pages from project wiki {self.wiki_name}")
                    return processed_pages
                else:
                    error_text = await response.text()
                    logger.error(f"Wiki pages API failed: {response.status} - {error_text}")
                    return []

    def _extract_title_from_path(self, path: str) -> str:
        """Extract a title from a file path"""
        if not path:
            return "Unknown"

        # Remove file extension
        name = path.replace('.md', '').replace('.markdown', '')

        # Remove path separators and convert to title case
        name = name.replace('/', ' ').replace('\\', ' ').strip()

        # Convert to title case
        return ' '.join(word.capitalize() for word in name.split())

    def _process_wiki_page_direct(self, page_info: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single wiki page that already has content included"""
        try:
            page_path = page_info.get("path", "")
            page_id = page_info.get("id")
            page_title = page_info.get("title", "Unknown")

            logger.debug(f"Processing wiki page: {page_path} (ID: {page_id}, Title: {page_title})")

            if not page_id:
                logger.warning(f"Page {page_path} has no ID - skipping")
                return None

            # Extract content directly from page_info since we're getting it with includeContent=true
            content = page_info.get("content", "")
            if not content:
                logger.warning(f"Page {page_path} has no content - skipping")
                return None

            # Process markdown and extract images
            html_content, image_references = self._process_markdown_content(content)

            result = {
                "id": str(page_id),
                "path": page_path,
                "title": page_title,
                "content": content,
                "html_content": html_content,
                "image_references": image_references,
                "author": page_info.get("lastModifiedBy", {}).get("displayName"),
                "last_modified": page_info.get("lastModifiedDate"),
                "version": page_info.get("version", "1")
            }

            logger.debug(f"Successfully processed page {page_path} with {len(content)} characters of content")
            return result

        except Exception as e:
            logger.error(f"Error processing wiki page {page_info.get('path', 'unknown')}: {e}", exc_info=True)
            return None


    def _process_markdown_content(self, markdown_content: str) -> tuple[str, list[str]]:
        """Process markdown content to HTML and extract image references"""
        if not markdown_content:
            return "", []

        try:
            # Convert markdown to HTML
            html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])

            # Extract image references using regex
            image_references = []

            # Markdown image syntax: ![alt](src)
            markdown_images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', markdown_content)
            for alt, src in markdown_images:
                image_references.append({
                    "type": "markdown",
                    "src": src,
                    "alt": alt
                })

            # HTML img tags
            soup = BeautifulSoup(html_content, 'html.parser')
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                alt = img.get('alt', '')
                if src:
                    image_references.append({
                        "type": "html",
                        "src": src,
                        "alt": alt
                    })

            # Azure DevOps attachment links: [[attachment:filename.png]]
            attachment_images = re.findall(r'\[\[attachment:([^\]]+)\]\]', markdown_content)
            for attachment in attachment_images:
                image_references.append({
                    "type": "attachment",
                    "src": attachment,
                    "alt": attachment
                })

            return html_content, image_references

        except Exception as e:
            logger.error(f"Error processing markdown content: {e}")
            return markdown_content, []
