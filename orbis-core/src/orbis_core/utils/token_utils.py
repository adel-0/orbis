"""
Token utility functions for accurate token counting using tiktoken
"""

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tokenizer(model: str = "gpt-4.1") -> tiktoken.Encoding:
    """Get the tokenizer for a specific model with caching"""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # GPT-5 models likely use o200k_base encoding, use it as fallback
        logger.debug(f"Model {model} not found in tiktoken, using o200k_base encoding")
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-5") -> int:
    """Count the exact number of tokens in text for a specific model"""
    if not text:
        return 0

    try:
        tokenizer = get_tokenizer(model)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback to rough estimation
        return len(text) // 4


def estimate_tokens_for_messages(messages: list[dict], model: str = "gpt-5") -> int:
    """
    Estimate tokens for OpenAI chat completion messages
    Accounts for message formatting overhead
    """
    try:
        tokenizer = get_tokenizer(model)

        # Base tokens for message formatting
        total_tokens = 3  # Base overhead

        for message in messages:
            total_tokens += 3  # per message overhead
            for key, value in message.items():
                if value:
                    total_tokens += len(tokenizer.encode(str(value)))
                if key == "name":  # name field has extra token
                    total_tokens += 1

        total_tokens += 3  # reply priming
        return total_tokens

    except Exception as e:
        logger.error(f"Error estimating message tokens: {e}")
        # Fallback estimation
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4


def chunk_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0, model: str = "gpt-4") -> list[str]:
    """
    Chunk text into smaller pieces based on token count

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        model: Model to use for tokenization

    Returns:
        List of text chunks
    """
    if not text:
        return []

    try:
        tokenizer = get_tokenizer(model)
        tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            # Move start position considering overlap
            start = end - overlap_tokens if overlap_tokens > 0 else end

        return chunks

    except Exception as e:
        logger.error(f"Error chunking text by tokens: {e}")
        # Fallback to character-based chunking
        char_limit = max_tokens * 4  # Rough estimation
        overlap_chars = overlap_tokens * 4

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + char_limit, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars if overlap_chars > 0 else end

        return chunks


def validate_token_limit(text: str, max_tokens: int, model: str = "gpt-5") -> bool:
    """Check if text is within token limit"""
    return count_tokens(text, model) <= max_tokens
