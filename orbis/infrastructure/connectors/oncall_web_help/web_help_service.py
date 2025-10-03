"""
OnCall Web Help Service
Implements standard connector interface for local HTML file ingestion.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import os
import hashlib

logger = logging.getLogger(__name__)


class OnCallWebHelpService:
    """
    OnCall Web Help connector implementing standard interface.
    
    Fetches HTML help documentation from local filesystem using configuration-driven approach.
    Extracts title and textual content from HTML files.
    """

    def __init__(self):
        """Lightweight initialization - no external connections."""
        pass

    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
        """
        Fetch HTML help files from local filesystem using provided configuration.
        
        Expected config structure:
        {
            "folder_path": "/path/to/html/files",
            "options": {
                "file_extensions": [".html", ".htm"],
                "recursive": True,
                "exclude_patterns": ["temp*", "*.bak"]
            }
        }
        
        Args:
            config: All configuration parameters as dictionary
            incremental: Whether to fetch only new/updated items (file modification time based)
            
        Returns:
            List of HTML file dictionaries with extracted content
        """
        try:
            # 1. Validate required config
            self._validate_required_config(config)
            
            # 2. Get configuration parameters with default path support
            from utils.data_paths import resolve_data_path
            
            configured_path = config.get('folder_path')
            folder_path = resolve_data_path(configured_path, 'oncall_web_help', create_if_missing=False)
            options = config.get('options', {})
            file_extensions = options.get('file_extensions', ['.html', '.htm'])
            recursive = options.get('recursive', True)
            
            # 3. Find HTML files
            logger.info(f"Scanning for HTML files in {folder_path}")
            html_files = self._find_html_files(folder_path, file_extensions, recursive)
            
            # 4. Process each HTML file
            help_data = []
            for html_file in html_files:
                try:
                    file_data = await self._process_html_file(html_file)
                    if file_data:
                        help_data.append(file_data)
                except Exception as e:
                    logger.warning(f"Failed to process {html_file}: {e}")
                    continue
            
            logger.info(f"Processed {len(help_data)} HTML help files from {folder_path}")
            return help_data
            
        except Exception as e:
            logger.error(f"Failed to fetch OnCall Web Help data: {e}")
            raise

    def get_content_id(self, item: dict) -> str:
        """
        Extract unique identifier from HTML help item.
        
        Args:
            item: Raw HTML file dictionary
            
        Returns:
            Unique string identifier for the item
        """
        # Use relative file path as the primary identifier
        file_path = item.get('file_path', '')
        if file_path:
            # Create a stable ID from the relative path
            relative_path = item.get('relative_path', file_path)
            return f"oncall_web_help_{hashlib.md5(relative_path.encode()).hexdigest()[:12]}"
        
        # Fallback to title-based ID
        title = item.get('title', '')
        if title:
            return f"oncall_web_help_title_{hashlib.md5(title.encode()).hexdigest()[:12]}"
        
        # Last resort: content hash
        content = item.get('content', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"oncall_web_help_content_{content_hash}"

    def get_last_modified(self, item: dict) -> datetime | None:
        """
        Extract last modified timestamp from HTML help item.
        
        Args:
            item: Raw HTML file dictionary
            
        Returns:
            Last modified datetime or None if not available
        """
        # Get modification time from file metadata
        source_metadata = item.get('source_metadata', {})
        modified_timestamp = source_metadata.get('modified_timestamp')
        
        if modified_timestamp:
            try:
                return datetime.fromtimestamp(modified_timestamp)
            except (ValueError, OSError) as e:
                logger.warning(f"Could not parse timestamp {modified_timestamp}: {e}")
        
        return None

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._validate_required_config(config)
            return True, ""
        except ValueError as e:
            return False, str(e)

    def _validate_required_config(self, config: dict[str, Any]):
        """Validate that all required configuration is present."""
        from utils.data_paths import resolve_data_path
        
        # Resolve path (using default if not specified)
        configured_path = config.get('folder_path')
        folder_path = resolve_data_path(configured_path, 'oncall_web_help', create_if_missing=False)
        
        if not folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Folder path is not a directory: {folder_path}")

    def _find_html_files(self, folder_path: Path, extensions: list[str], recursive: bool) -> list[Path]:
        """Find all HTML files in the specified folder."""
        html_files = []
        
        # Normalize extensions to lowercase
        extensions = [ext.lower() for ext in extensions]
        
        if recursive:
            search_pattern = "**/*"
        else:
            search_pattern = "*"
        
        for file_path in folder_path.glob(search_pattern):
            if not file_path.is_file():
                continue
                
            # Check file extension
            if file_path.suffix.lower() not in extensions:
                continue
            
            html_files.append(file_path)
        
        return sorted(html_files)


    async def _process_html_file(self, file_path: Path) -> dict[str, Any] | None:
        """Process a single HTML file and extract content."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Extract title, content, and metadata using BeautifulSoup
            title, text_content, html_modified_date = self._extract_html_content(html_content)
            
            # Get file stats
            stat = file_path.stat()
            
            # Use HTML modified date if available, otherwise use file modification time
            if html_modified_date:
                try:
                    # Parse the HTML date (format: 2023-12-11)
                    modified_dt = datetime.strptime(html_modified_date, '%Y-%m-%d')
                    modified_timestamp = modified_dt.timestamp()
                    last_modified_str = modified_dt.isoformat()
                except (ValueError, TypeError):
                    # Fallback to file modification time if HTML date parsing fails
                    modified_timestamp = stat.st_mtime
                    last_modified_str = datetime.fromtimestamp(stat.st_mtime).isoformat()
            else:
                modified_timestamp = stat.st_mtime
                last_modified_str = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Create the data structure
            file_data = {
                'id': str(file_path),
                'title': title or file_path.stem,
                'content': text_content,
                'content_type': 'oncall_web_help',
                'source_metadata': {
                    'file_path': str(file_path),
                    'relative_path': str(file_path.name),
                    'file_size': stat.st_size,
                    'modified_timestamp': modified_timestamp,
                    'html_modified_date': html_modified_date,
                    'original_html': html_content
                },
                'extracted_metadata': {
                    'content_length': len(text_content),
                    'title_length': len(title) if title else 0,
                    'html_size': len(html_content)
                },
                'last_modified': last_modified_str,
                'author': 'system'
            }
            
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to process HTML file {file_path}: {e}")
            return None

    def _extract_html_content(self, html_content: str) -> tuple[str, str, str | None]:
        """
        Extract title, text content, and metadata from HTML.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Tuple of (title, text_content, html_modified_date)
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title - check h3 first (OnCall help format), then title tag, then h1
            title = ""
            h3_tag = soup.find('h3')
            if h3_tag:
                title = h3_tag.get_text().strip()
            
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
            
            # If no title tag, try to get h1
            if not title:
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text().strip()
            
            # Extract data-modifieddate if available (OnCall help format)
            html_modified_date = None
            div_with_date = soup.find('div', {'data-modifieddate': True})
            if div_with_date:
                html_modified_date = div_with_date.get('data-modifieddate')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            return title, text_content, html_modified_date
            
        except ImportError:
            # Fallback to basic HTML cleaning if BeautifulSoup is not available
            logger.warning("BeautifulSoup not available, using basic HTML extraction")
            return self._extract_html_content_basic(html_content)
        except Exception as e:
            logger.error(f"Error extracting HTML content: {e}")
            return "", "", None

    def _extract_html_content_basic(self, html_content: str) -> tuple[str, str, str | None]:
        """
        Basic HTML content extraction without BeautifulSoup.
        
        Args:
            html_content: Raw HTML content

        Returns:
            Tuple of (title, text_content, html_modified_date)
        """
        from orbis_core.utils.constants import clean_html_content
        import re
        
        # Extract title using regex - check h3 first (OnCall format), then title, then h1
        title = ""
        h3_match = re.search(r'<h3[^>]*>(.*?)</h3>', html_content, re.IGNORECASE | re.DOTALL)
        if h3_match:
            title = clean_html_content(h3_match.group(1)).strip()
        
        if not title:
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = clean_html_content(title_match.group(1)).strip()
        
        # If no title, try h1
        if not title:
            h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
            if h1_match:
                title = clean_html_content(h1_match.group(1)).strip()
        
        # Extract data-modifieddate if available
        html_modified_date = None
        date_match = re.search(r'data-modifieddate=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if date_match:
            html_modified_date = date_match.group(1)
        
        # Clean HTML content to get text
        text_content = clean_html_content(html_content)
        
        # Clean up whitespace
        text_content = ' '.join(text_content.split())
        
        return title, text_content, html_modified_date