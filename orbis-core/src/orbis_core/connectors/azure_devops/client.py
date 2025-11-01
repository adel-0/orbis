"""
Azure DevOps REST API client for work item data ingestion.
Provides functionality equivalent to the PowerShell get_workitems.ps1 script.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import aiohttp

from .auth import AzureDevOpsAuthMixin
from .constants import API_VERSIONS, CORE_WORKITEM_FIELDS, DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class Client(AzureDevOpsAuthMixin):
    """Client for Azure DevOps REST API operations with support for PAT and OAuth2 authentication"""

    def __init__(self, organization: str, project: str, auth_token: str = None,
                 client_id: str = None, client_secret: str = None,
                 tenant_id: str = None, use_oauth2: bool = False,
                 max_concurrent_requests: int = 20, timeout_seconds: int = 300):
        super().__init__()

        if not organization:
            raise ValueError("Organization is required")
        if not project:
            raise ValueError("Project is required")

        self.organization = organization
        self.project = project
        self.use_oauth2 = use_oauth2

        # API versions from constants
        self.query_api_version = API_VERSIONS['query']
        self.workitem_api_version = API_VERSIONS['workitem']
        self.comments_api_version = API_VERSIONS['comments']
        self.reporting_api_version = API_VERSIONS['reporting']

        # Core fields from constants
        self.core_fields = CORE_WORKITEM_FIELDS

        # Parallel processing settings
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_size = DEFAULT_BATCH_SIZE  # API maximum
        self.timeout_seconds = timeout_seconds

        # Initialize authentication
        if use_oauth2:
            if not all([client_id, client_secret, tenant_id]):
                raise ValueError("OAuth2 requires client_id, client_secret, and tenant_id")
            self.client_id = client_id
            self.client_secret = client_secret
            self.tenant_id = tenant_id
        else:
            if not auth_token:
                raise ValueError("Personal Access Token is required when not using OAuth2")
            self.pat = auth_token

    # Authentication methods inherited from AzureDevOpsAuthMixin
    # _get_auth_headers() and _refresh_oauth_token() are now provided by the mixin

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Client':
        """Create a Client from a configuration dictionary"""
        auth_type = config.get('auth_type', 'pat')
        if auth_type == "pat":
            return cls(
                organization=config.get('organization'),
                project=config.get('project'),
                auth_token=config.get('pat'),
                use_oauth2=False,
                max_concurrent_requests=config.get('max_concurrent_requests', 20),
                timeout_seconds=config.get('timeout_seconds', 300)
            )
        elif auth_type == "oauth2":
            return cls(
                organization=config.get('organization'),
                project=config.get('project'),
                client_id=config.get('client_id'),
                client_secret=config.get('client_secret'),
                tenant_id=config.get('tenant_id'),
                use_oauth2=True,
                max_concurrent_requests=config.get('max_concurrent_requests', 20),
                timeout_seconds=config.get('timeout_seconds', 300)
            )
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")

    @classmethod
    def from_data_source(cls, data_source: Any) -> 'Client':
        """Create a Client from a DataSource model object"""
        auth_type = getattr(data_source, 'auth_type', 'pat')
        if auth_type == "pat":
            return cls(
                organization=data_source.organization,
                project=data_source.project,
                auth_token=data_source.pat,
                use_oauth2=False
            )
        elif auth_type == "oauth2":
            return cls(
                organization=data_source.organization,
                project=data_source.project,
                client_id=data_source.client_id,
                client_secret=data_source.client_secret,
                tenant_id=data_source.tenant_id,
                use_oauth2=True
            )
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")

    def get_encoded_project_path(self) -> str:
        """Get URL-encoded project path"""
        return quote(self.project)

    async def invoke_rest_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str,
        max_retries: int = 3
    ) -> dict[str, Any]:
        """Invoke REST API with basic retry logic"""
        for attempt in range(max_retries):
            try:
                headers = await self._get_auth_headers()
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response.raise_for_status()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2)

    def clean_html_string(self, html_content: str) -> str:
        """Clean HTML content from strings"""
        if not html_content or not isinstance(html_content, str):
            return html_content or ""

        import html
        import re

        # Decode common HTML entities
        content = html.unescape(html_content)

        # Remove all HTML tags
        content = re.sub(r'<[^>]*>', '', content)

        # Remove empty or redundant tags (if any remain)
        content = re.sub(r'<([a-zA-Z0-9]+)>\s*</\1>', '', content)

        # Collapse multiple spaces and newlines into single spaces
        content = re.sub(r'\s+', ' ', content)

        # Trim whitespace
        return content.strip()

    def clean_object(self, obj: Any) -> Any:
        """Recursively clean HTML content from objects"""
        if isinstance(obj, str):
            return self.clean_html_string(obj)
        elif isinstance(obj, list):
            return [self.clean_object(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.clean_object(value) for key, value in obj.items()}
        else:
            return obj

    async def get_workitems_from_reporting_api(
        self,
        session: aiohttp.ClientSession,
        start_datetime: str | None = None
    ) -> list[int]:
        """Get work item IDs using Reporting API for incremental sync"""

        logger.info("Fetching work items using Reporting API...")
        project_escaped = self.get_encoded_project_path()

        # Build reporting API URL
        reporting_url = (
            f"https://dev.azure.com/{self.organization}/{project_escaped}/"
            f"_apis/wit/reporting/workitemrevisions"
            f"?includeLatestOnly=true&maxPageSize=200&api-version={self.reporting_api_version}"
        )

        if start_datetime:
            reporting_url += f"&startDateTime={start_datetime}"

        all_ids = []
        batch_count = 0

        while reporting_url:
            batch_count += 1
            logger.info(f"Fetching reporting batch {batch_count}...")

            response = await self.invoke_rest_with_retry(session, reporting_url)

            if response.get('values'):
                batch_ids = [item['id'] for item in response['values']]
                all_ids.extend(batch_ids)
                logger.info(f"Found {len(batch_ids)} work items in batch {batch_count}")

            # Check for next page
            reporting_url = response.get('nextLink') if not response.get('isLastBatch', True) else None

        # Remove duplicates and return unique IDs
        unique_ids = list(set(all_ids))
        logger.info(f"Total unique work items from Reporting API: {len(unique_ids)}")
        return unique_ids

    async def get_workitems_from_query(
        self,
        session: aiohttp.ClientSession,
        query_id: str
    ) -> list[int]:
        """Get work item IDs using WIQL query"""

        logger.info("Performing full sync using saved query...")
        project_escaped = self.get_encoded_project_path()

        wiql_url = (
            f"https://dev.azure.com/{self.organization}/{project_escaped}/"
            f"_apis/wit/wiql/{query_id}?api-version={self.query_api_version}"
        )

        logger.info(f"WIQL URL: {wiql_url}")
        logger.info(f"Query ID: {query_id}")

        response = await self.invoke_rest_with_retry(session, wiql_url)

        # Extract IDs (support both forms of WIQL response)
        if response.get('workItems'):
            return [item['id'] for item in response['workItems']]
        elif response.get('workItemRelations'):
            return [rel['target']['id'] for rel in response['workItemRelations'] if rel.get('target')]
        else:
            raise ValueError("Unable to locate work item IDs in WIQL response")

    async def get_workitem_details_batch(
        self,
        session: aiohttp.ClientSession,
        ids: list[int]
    ) -> list[dict[str, Any]]:
        """Get work item details in batches with all available fields"""

        logger.info("Fetching work item details...")
        all_workitems = []

        # Process in batches
        for i in range(0, len(ids), self.batch_size):
            batch_ids = ids[i:i + self.batch_size]
            batch_ids_str = ",".join(map(str, batch_ids))

            # Don't specify fields parameter to get all available fields for each work item
            items_url = (
                f"https://dev.azure.com/{self.organization}/_apis/wit/workitems"
                f"?ids={batch_ids_str}&$expand=all&api-version={self.workitem_api_version}"
            )

            batch_num = (i // self.batch_size) + 1
            total_batches = (len(ids) + self.batch_size - 1) // self.batch_size

            logger.info(f"Fetching batch {batch_num}/{total_batches}: items {i + 1} to {min(i + self.batch_size, len(ids))}")

            response = await self.invoke_rest_with_retry(session, items_url)
            all_workitems.extend(response.get('value', []))

            processed = min(i + self.batch_size, len(ids))
            logger.info(f"Fetched {processed}/{len(ids)} work items")

        return all_workitems

    async def get_workitem_comments(
        self,
        session: aiohttp.ClientSession,
        workitem_id: int
    ) -> list[str]:
        """Get comments for a single work item"""

        project_escaped = self.get_encoded_project_path()
        comments_url = (
            f"https://dev.azure.com/{self.organization}/{project_escaped}/"
            f"_apis/wit/workItems/{workitem_id}/comments?api-version={self.comments_api_version}"
        )

        for attempt in range(3):
            try:
                headers = await self._get_auth_headers()
                async with session.get(comments_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [comment.get('text', '') for comment in data.get('comments', [])]
            except Exception:
                if attempt == 2:
                    return []
                await asyncio.sleep(2)

        return []

    async def process_workitems_with_comments_parallel(
        self,
        session: aiohttp.ClientSession,
        workitems: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process work items with comments in parallel"""

        logger.info(f"Processing {len(workitems)} work items with comments using parallel processing...")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_single_workitem(workitem: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                workitem_id = workitem['id']
                comments = await self.get_workitem_comments(session, workitem_id)

                # Create work item object
                # Standardize field names
                # Convert Azure DevOps API URL to user-facing web URL
                # API URL: https://dev.azure.com/{org}/{project}/_apis/wit/workItems/{id}
                # Web URL: https://dev.azure.com/{org}/{project}/_workitems/edit/{id}
                import urllib.parse
                project_encoded = urllib.parse.quote(self.project, safe='')
                web_url = f"https://dev.azure.com/{self.organization}/{project_encoded}/_workitems/edit/{workitem_id}"

                result = {
                    'Id': workitem_id,
                    'Comments': comments,
                    'url': web_url
                }

                # Add all fields from the work item
                fields = workitem.get('fields', {})
                for field_name, field_value in fields.items():
                    # Map field names to simpler names
                    if field_name == 'System.Title':
                        result['Title'] = field_value
                    elif field_name == 'System.Description':
                        result['Description'] = field_value
                    elif field_name == 'System.State':
                        result['State'] = field_value
                    elif field_name == 'System.WorkItemType':
                        result['WorkItemType'] = field_value
                    elif field_name == 'System.ChangedDate':
                        result['ChangedDate'] = field_value
                    elif field_name == 'System.CreatedDate':
                        result['CreatedDate'] = field_value
                    elif field_name == 'System.AreaPath':
                        result['AreaPath'] = field_value
                    elif field_name == 'System.IterationPath':
                        result['IterationPath'] = field_value
                    elif field_name == 'System.AssignedTo':
                        result['AssignedTo'] = field_value
                    elif field_name == 'System.CreatedBy':
                        result['CreatedBy'] = field_value
                    elif field_name == 'System.Tags':
                        result['Tags'] = field_value
                    else:
                        # Keep original field name for other fields
                        result[field_name] = field_value

                return result

        # Process all work items in parallel
        start_time = time.time()
        tasks = [process_single_workitem(workitem) for workitem in workitems]

        # Process with progress updates
        processed_items = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            processed_items.append(result)
            completed += 1

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / (elapsed / 60) if elapsed > 0 else 0
                logger.info(f"Processed: {completed}/{len(workitems)} ({rate:.1f} items/min)")

        elapsed = time.time() - start_time
        rate = len(processed_items) / (elapsed / 60) if elapsed > 0 else 0
        logger.info(f"Comment processing completed: {len(processed_items)} work items in {elapsed/60:.1f} minutes ({rate:.1f} items/min)")

        return processed_items

    async def get_workitems(
        self,
        query_id: str = None,
        use_incremental_sync: bool = True,
        last_run_datetime: str | None = None,
        skip_comments: bool = False
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Main method to get work items from Azure DevOps

        Returns:
            Tuple of (work_items_list, current_timestamp)
        """

        logger.info("Starting work item data ingestion pipeline...")

        limit = max(10, int(self.max_concurrent_requests) * 2)
        connector = aiohttp.TCPConnector(limit=limit, limit_per_host=self.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

            # Step 1: Get work item IDs
            all_ids = []

            if use_incremental_sync and last_run_datetime:
                logger.info(f"Performing incremental sync from: {last_run_datetime}")
                try:
                    all_ids = await self.get_workitems_from_reporting_api(session, last_run_datetime)
                except Exception as e:
                    logger.warning(f"Incremental sync failed, falling back to full sync: {str(e)}")
                    all_ids = []

            if not all_ids:
                if not query_id:
                    raise ValueError("Query ID is required for full sync")
                all_ids = await self.get_workitems_from_query(session, query_id)

            logger.info(f"Total work items to process: {len(all_ids)}")

            if not all_ids:
                logger.warning("No work items found")
                current_timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                return [], current_timestamp

            # Step 2: Get work item details
            workitems = await self.get_workitem_details_batch(session, all_ids)

            # Step 3: Process comments (if not skipped)
            if skip_comments:
                logger.info("Skipping comment processing")
                workitems_data = []
                for item in workitems:
                    workitem_data = {
                        'Id': item['id'],
                        'Comments': []
                    }

                    # Add all fields
                    fields = item.get('fields', {})
                    for field_name, field_value in fields.items():
                        if field_name == 'System.Title':
                            workitem_data['Title'] = field_value
                        elif field_name == 'System.Description':
                            workitem_data['Description'] = field_value
                        else:
                            workitem_data[field_name] = field_value

                    workitems_data.append(workitem_data)
            else:
                workitems_data = await self.process_workitems_with_comments_parallel(session, workitems)

            # Step 4: Clean HTML content
            logger.info("Cleaning HTML content from work items...")
            cleaned_data = self.clean_object(workitems_data)

            # Step 5: Generate timestamp for next run
            current_timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            logger.info(f"Data ingestion pipeline completed successfully! Processed {len(cleaned_data)} work items")

            return cleaned_data, current_timestamp
