"""
Default data paths for local connectors.
Provides centralized path management for local data sources.
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from current file and go up to find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # Go up one level from utils/
    return project_root


def get_default_data_sources_path() -> Path:
    """Get the default data sources directory."""
    return get_project_root() / "data" / "sources"


def get_default_path_for_source_type(source_type: str) -> Path:
    """
    Get the default directory path for a specific source type.
    
    Args:
        source_type: The data source type (e.g., 'oncall_web_help', 'documents')
        
    Returns:
        Path to the default directory for this source type
    """
    source_type_mapping = {
        'oncall_web_help': 'oncall_web_help'
    }
    
    # Map source type to directory name, or use source type as-is if not mapped
    dir_name = source_type_mapping.get(source_type, source_type)
    
    return get_default_data_sources_path() / dir_name




def is_using_default_path(config_path: str, source_type: str) -> bool:
    """
    Check if the configured path is the default path for this source type.
    
    Args:
        config_path: The configured path from configuration
        source_type: The data source type
        
    Returns:
        True if using default path, False otherwise
    """
    try:
        configured = Path(config_path).resolve()
        default = get_default_path_for_source_type(source_type).resolve()
        return configured == default
    except (ValueError, OSError):
        return False


def resolve_data_path(config_path: str | None, source_type: str, create_if_missing: bool = False) -> Path:
    """
    Resolve the data path for a connector, using default if not specified.
    
    Args:
        config_path: Path from configuration (can be None or empty)
        source_type: The data source type 
        create_if_missing: Whether to create the directory if it doesn't exist (ignored)
        
    Returns:
        Resolved Path object
    """
    if config_path and config_path.strip():
        # Use configured path
        path = Path(config_path)
    else:
        # Use default path
        path = get_default_path_for_source_type(source_type)
    
    return path