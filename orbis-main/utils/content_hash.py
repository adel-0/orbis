"""
Content hashing utilities for tracking embedding changes.
Uses xxHash for fast, non-cryptographic content fingerprinting.
"""

import xxhash
from typing import Any

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