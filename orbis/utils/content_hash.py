"""
Content hashing utilities for tracking embedding changes.
Uses xxHash for fast, non-cryptographic content fingerprinting.
"""

import xxhash

from core.schemas import BaseContent


def hash_content(content: BaseContent) -> str:
    """
    Generate a xxHash of content for change detection.

    Args:
        content: Content object to hash

    Returns:
        16-character hexadecimal hash string
    """
    # Create a consistent string representation for hashing
    content_str = _content_to_hashable_string(content)
    return xxhash.xxh64(content_str.encode('utf-8')).hexdigest()


def _content_to_hashable_string(content: BaseContent) -> str:
    """
    Convert BaseContent object to a consistent string for hashing.

    Only includes fields that affect embedding generation:
    - title (used in embedding text)
    - content (main text content)

    Excludes metadata, timestamps, and IDs that don't affect embeddings.
    """
    title = getattr(content, 'title', '') or ''
    text = getattr(content, 'content', '') or ''

    return f"title:{title}\ncontent:{text}"
