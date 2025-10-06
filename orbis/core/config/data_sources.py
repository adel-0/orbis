"""
Data source connector registry - Minimal mapping from source type to connector class.
Connectors are self-describing (they define their own collection_name, metadata schema, etc.)
This file only provides the module/class mapping needed for dynamic import.
"""

from typing import Any

# Minimal connector registry - only module/class paths for dynamic import
DATA_SOURCE_CONNECTORS = {
    "azdo_workitems": {
        "connector_module": "infrastructure.connectors.azure_devops.work_item_service",
        "connector_class": "WorkItemService",
    },
    "azdo_wiki": {
        "connector_module": "infrastructure.connectors.azure_devops.wiki_service",
        "connector_class": "WikiService",
    },
    "oncall_web_help": {
        "connector_module": "infrastructure.connectors.oncall_web_help.web_help_service",
        "connector_class": "OnCallWebHelpService",
    }
}


def get_data_source_config(source_type: str) -> dict[str, Any]:
    """
    Get connector configuration for a data source type.
    Returns minimal config with connector module/class for dynamic import.

    Args:
        source_type: The data source type (e.g. 'azdo_workitems', 'azdo_wiki')

    Returns:
        Configuration dictionary with connector_module and connector_class

    Raises:
        ValueError: If source_type is not supported
    """
    if source_type not in DATA_SOURCE_CONNECTORS:
        raise ValueError(f"Unknown data source type: {source_type}")
    return DATA_SOURCE_CONNECTORS[source_type]


def list_data_source_types() -> list[str]:
    """
    List all supported data source types.

    Returns:
        List of supported data source type strings
    """
    return list(DATA_SOURCE_CONNECTORS.keys())


def is_valid_source_type(source_type: str) -> bool:
    """
    Check if a source type is valid/supported.

    Args:
        source_type: The data source type to validate

    Returns:
        True if source type is supported, False otherwise
    """
    return source_type in DATA_SOURCE_CONNECTORS


def get_collection_name(source_type: str) -> str:
    """
    Get the collection name for a data source type from the connector itself.
    Connectors are self-describing and define their own collection names.

    Args:
        source_type: The data source type

    Returns:
        Collection name from the connector's get_collection_name() method

    Raises:
        ValueError: If source_type is not supported
    """
    import importlib

    config = get_data_source_config(source_type)
    module = importlib.import_module(config["connector_module"])
    connector_class = getattr(module, config["connector_class"])
    return connector_class.get_collection_name()


def get_all_collection_names() -> list[str]:
    """
    Get all collection names by querying each connector.
    Connectors are self-describing and define their own collection names.

    Returns:
        List of unique collection names from all registered connectors
    """
    collection_names = []
    for source_type in list_data_source_types():
        collection_name = get_collection_name(source_type)
        if collection_name not in collection_names:
            collection_names.append(collection_name)
    return collection_names


def get_priority_boost_for_source_type(source_type: str) -> float:
    """
    Get the priority boost (search weight) for a data source type.
    Returns default of 1.0 since priority is now set per instance in the database.

    Args:
        source_type: The data source type

    Returns:
        Default priority boost of 1.0
    """
    # Priority is now configured per DataSource instance in the database
    # This function returns a default for backwards compatibility
    return 1.0
