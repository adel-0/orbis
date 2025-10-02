"""
Environment variable parsing utilities.
Provides type-safe environment variable parsing with defaults.
"""

import os
from typing import TypeVar

T = TypeVar('T', bound=str | int | float | bool)


def get_env(key: str, default: T, cast_type: type[T] = None) -> T:
    """
    Get environment variable with type casting and default value.

    Args:
        key: Environment variable key
        default: Default value if key not found
        cast_type: Type to cast the value to (inferred from default if not provided)

    Returns:
        Environment variable value cast to the appropriate type
    """
    value = os.getenv(key)
    if value is None:
        return default

    # Infer type from default if not explicitly provided
    if cast_type is None:
        cast_type = type(default)

    # Handle boolean conversion
    if cast_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    # Handle list conversion (comma-separated)
    if cast_type is list:
        return [item.strip() for item in value.split(",") if item.strip()]

    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default


def get_env_list(key: str, default: list[str] = None) -> list[str]:
    """
    Get environment variable as a list (comma-separated values).

    Args:
        key: Environment variable key
        default: Default list if key not found

    Returns:
        List of string values
    """
    if default is None:
        default = []

    value = os.getenv(key)
    if not value:
        return default

    return [item.strip() for item in value.split(",") if item.strip()]
