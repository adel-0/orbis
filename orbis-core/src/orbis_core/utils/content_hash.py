"""
Content hashing utilities for tracking content changes.
Uses xxHash for fast, non-cryptographic content fingerprinting.
"""

import xxhash
from typing import Any


def hash_content_dict(content: dict[str, Any]) -> str:
    """
    Generate a xxHash of content dictionary for change detection.

    Args:
        content: Content dictionary with 'title' and 'content' keys

    Returns:
        16-character hexadecimal hash string
    """
    # Extract main content fields for hashing
    title = str(content.get('title', ''))
    text = str(content.get('content', ''))

    # Create consistent string representation
    content_str = f"title:{title}\ncontent:{text}"
    return xxhash.xxh64(content_str.encode('utf-8')).hexdigest()


def hash_text_content(title: str, content: str) -> str:
    """
    Generate a xxHash of title and content strings.

    Args:
        title: Content title
        content: Content text

    Returns:
        16-character hexadecimal hash string
    """
    content_str = f"title:{title}\ncontent:{content}"
    return xxhash.xxh64(content_str.encode('utf-8')).hexdigest()


def hash_object(obj: Any, *fields: str) -> str:
    """
    Generate a xxHash of an object using specified fields.

    Args:
        obj: Object to hash (can be dict-like or have attributes)
        *fields: Field names to include in the hash

    Returns:
        16-character hexadecimal hash string
    """
    parts = []
    for field in fields:
        # Try dict access first, then attribute access
        if isinstance(obj, dict):
            value = str(obj.get(field, ''))
        else:
            value = str(getattr(obj, field, ''))
        parts.append(f"{field}:{value}")

    content_str = "\n".join(parts)
    return xxhash.xxh64(content_str.encode('utf-8')).hexdigest()
