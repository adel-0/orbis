"""
Shared constants for Orbis.
Centralizes commonly used configuration values.
"""

import logging
from pathlib import Path

import yaml

# Default processing settings
DEFAULT_MAX_CONCURRENT_REQUESTS: int = 20
DEFAULT_RATE_LIMIT_DELAY: float = 0.05  # seconds


def _load_project_config():
    """Load project configuration from YAML file"""
    config_path = Path("config/project_config.yaml")

    if not config_path.exists():
        logging.getLogger(__name__).warning(f"Project config file not found at {config_path}")
        return {}, {}

    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        area_path_mappings = config.get('area_path_mappings', {})
        project_configs = config.get('project_configs', {})

        # Extract wiki repos for each project
        project_wiki_repos = {}
        for project_code, project_config in project_configs.items():
            project_wiki_repos[project_code] = project_config.get('wiki_repos', [])

        return area_path_mappings, project_wiki_repos

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load project config: {e}")
        return {}, {}


# Project detection mappings loaded from config
AREA_PATH_MAPPINGS, PROJECT_WIKI_REPOS = _load_project_config()


# Agent timeout and thresholds
ORCHESTRATOR_TIMEOUT_SECONDS: int = 120
ORCHESTRATOR_MIN_CONFIDENCE_THRESHOLD: float = 0.3

# Wiki summarization configuration
WIKI_MAX_TOKENS_PER_CHUNK: int = 15000  # Input token limit for processing chunks
WIKI_OVERLAP_TOKENS: int = 500
WIKI_CACHE_DURATION_HOURS: int = 2400  # Legacy - replaced by persistent cache refresh
WIKI_PAGE_SUMMARY_TOKENS: int = 3000  # Target output tokens for individual page summaries
WIKI_COMPREHENSIVE_SUMMARY_TOKENS: int = 8000  # Target output tokens for comprehensive summaries

# Persistent wiki cache configuration
WIKI_CACHE_REFRESH_DAYS: int = 30  # Refresh wiki summaries every 30 days
WIKI_CACHE_STARTUP_TIMEOUT_MINUTES: int = 10  # Maximum time to spend on startup pre-computation

# LLM token limits
LLM_MAX_OUTPUT_TOKENS: int = 2000  # Maximum tokens for LLM response generation

# Rerank service
DEFAULT_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_BATCH_SIZE: int = 16

# HTML cleaning utility function
def clean_html_content(html_content: str) -> str:
    """
    Clean HTML content from strings - consolidated from multiple implementations.

    Args:
        html_content: String that may contain HTML tags and entities

    Returns:
        Cleaned plain text string
    """
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
