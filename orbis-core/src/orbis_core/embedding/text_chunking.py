"""
Shared text chunking utilities for embedding and reranking services.
"""
from typing import Iterator, Any


def chunk_text(text: str, max_length: int, tokenizer: Any) -> Iterator[str]:
    """Yield text chunks with ~10% overlap using token-based chunking."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_length:
        yield text
        return

    overlap_size = max(1, int(max_length * 0.1))
    stride = max_length - overlap_size

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + max_length]

        if not chunk_tokens:
            continue

        # Don't yield a final chunk that is only overlap, unless it's the only chunk
        if len(chunk_tokens) <= overlap_size and i > 0:
            break

        yield tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if i + max_length >= len(tokens):
            break


def should_chunk_text(text: str, max_chunk_size: int, tokenizer: Any) -> bool:
    """Determine if text needs chunking based on token count."""
    # This is not streaming, but encode() is fast C++
    return len(tokenizer.encode(text)) > max_chunk_size
