"""
Data source configuration registry - Single source of truth for all supported data sources.
This replaces hardcoded assumptions throughout the system with data-driven configuration.
"""

from dataclasses import dataclass
from typing import Any

DATA_SOURCE_CONFIGS = {
    "azdo_workitems": {
        "collection_name": "workitems_collection",
        "connector_module": "infrastructure.connectors.azure_devops.work_item_service",
        "connector_class": "WorkItemService",
        "searchable_content_hint": "concatenate title, description, comments, attachments",
        "typical_filters": ["area_path", "state", "assigned_to", "work_item_type", "project"],
        "search_weight": 1.1,  # Work items with contextual information
        "content_type": "work_item",
        "field_mappings": {
            "title": "title",
            "content": "description",
            "created_date": "azure_created_date",
            "updated_date": "azure_changed_date",
            "reference": "azure_url"
        },
        "metadata_fields": [
            "work_item_type", "state", "priority", "severity", "assigned_to",
            "created_by", "tags", "area_path",
            "azure_resolved_date", "additional_fields"
        ],
        "schema_class": "Ticket"
    },
    "azdo_wiki": {
        "collection_name": "wiki_collection",
        "connector_module": "infrastructure.connectors.azure_devops.wiki_service",
        "connector_class": "WikiService",
        "searchable_content_hint": "concatenate title, content, images, attachments",
        "typical_filters": ["project", "path", "author"],
        "search_weight": 1.3,  # Rich documentation content
        "content_type": "wiki_page",
        "field_mappings": {
            "title": "title",
            "content": "content",
            "created_date": "last_modified",
            "updated_date": "last_modified",
            "reference": "url"
        },
        "metadata_fields": [
            "path", "html_content", "image_references", "author", "version"
        ],
        "schema_class": "WikiPageContent"
    },
    "oncall_web_help": {
        "collection_name": "oncall_web_help_collection",
        "connector_module": "infrastructure.connectors.oncall_web_help.web_help_service",
        "connector_class": "OnCallWebHelpService",
        "searchable_content_hint": "concatenate title, content from HTML files",
        "typical_filters": ["file_path", "relative_path"],
        "search_weight": 0.8,  # Help documentation content
        "content_type": "oncall_web_help",
        "field_mappings": {
            "title": "title",
            "content": "content",
            "created_date": "last_modified",
            "updated_date": "last_modified",
            "reference": "file_path"
        },
        "metadata_fields": [
            "file_path", "relative_path", "file_size", "modified_timestamp", "html_size"
        ],
        "schema_class": "OnCallWebHelpContent"
    }
}


def get_data_source_config(source_type: str) -> dict[str, Any]:
    """
    Get configuration for a data source type.

    Args:
        source_type: The data source type (e.g. 'azdo_workitems', 'azdo_wiki')

    Returns:
        Configuration dictionary for the data source

    Raises:
        ValueError: If source_type is not supported
    """
    if source_type not in DATA_SOURCE_CONFIGS:
        raise ValueError(f"Unknown data source type: {source_type}")
    return DATA_SOURCE_CONFIGS[source_type]


def list_data_source_types() -> list[str]:
    """
    List all supported data source types.

    Returns:
        List of supported data source type strings
    """
    return list(DATA_SOURCE_CONFIGS.keys())


def is_valid_source_type(source_type: str) -> bool:
    """
    Check if a source type is valid/supported.

    Args:
        source_type: The data source type to validate

    Returns:
        True if source type is supported, False otherwise
    """
    return source_type in DATA_SOURCE_CONFIGS


def get_collection_name(source_type: str) -> str:
    """
    Get the collection name for a data source type.

    Args:
        source_type: The data source type

    Returns:
        Collection name for the data source

    Raises:
        ValueError: If source_type is not supported
    """
    config = get_data_source_config(source_type)
    return config["collection_name"]


def get_all_collection_names() -> list[str]:
    """
    Get all collection names used by registered data sources.

    Returns:
        List of unique collection names
    """
    collection_names = []
    for config in DATA_SOURCE_CONFIGS.values():
        collection_name = config["collection_name"]
        if collection_name not in collection_names:
            collection_names.append(collection_name)
    return collection_names


def get_priority_boost_for_source_type(source_type: str) -> float:
    """
    Get the priority boost (search weight) for a data source type.

    Args:
        source_type: The data source type

    Returns:
        Priority boost value from configuration, defaults to 1.0 if not found
    """
    try:
        config = get_data_source_config(source_type)
        return config.get("search_weight", 1.0)
    except ValueError:
        # Unknown source type, return default
        return 1.0


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    collection_name: str
    connector_module: str
    connector_class: str
    searchable_content_hint: str
    typical_filters: list[str]
    search_weight: float = 1.0  # Default search weight
    content_type: str = ""  # The content type this source produces
    field_mappings: dict[str, str] = None  # Maps standard fields to source-specific fields
    metadata_fields: list[str] = None  # Fields to include in metadata
    schema_class: str = "BaseContent"  # Schema class for conversion


class DataSourceConfigRegistry:
    """Registry for data source configurations"""

    def __init__(self):
        self._configs = {}
        self._load_configs()

    def _load_configs(self):
        """Load configurations from DATA_SOURCE_CONFIGS"""
        for source_type, config_dict in DATA_SOURCE_CONFIGS.items():
            self._configs[source_type] = DataSourceConfig(
                name=source_type,
                collection_name=config_dict["collection_name"],
                connector_module=config_dict["connector_module"],
                connector_class=config_dict["connector_class"],
                searchable_content_hint=config_dict["searchable_content_hint"],
                typical_filters=config_dict["typical_filters"],
                search_weight=config_dict.get("search_weight", 1.0),
                content_type=config_dict.get("content_type", ""),
                field_mappings=config_dict.get("field_mappings", {}),
                metadata_fields=config_dict.get("metadata_fields", []),
                schema_class=config_dict.get("schema_class", "BaseContent")
            )

    def get_all_source_configs(self) -> list[DataSourceConfig]:
        """Get all registered data source configurations"""
        return list(self._configs.values())

    def get_source_config(self, source_type: str) -> DataSourceConfig:
        """Get configuration for a specific source type"""
        if source_type not in self._configs:
            raise ValueError(f"Unknown data source type: {source_type}")
        return self._configs[source_type]
