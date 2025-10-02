# EmbeddingGemma-300m Implementation Plan

## Current Status: NOT COMPATIBLE

Your current implementation uses standard `sentence-transformers.encode()` without the required prompt templates that EmbeddingGemma needs.

## Required Changes

### 1. Modify LocalEmbeddingProvider (`services/embedding_service.py`)
- Detect EmbeddingGemma models
- Replace `.encode()` with `.encode_document()` for ticket embeddings
- Replace `.encode()` with `.encode_query()` for search queries

### 2. Implement Prompt Templates
- **Documents**: `title: {title | "none"} | text: {content}`
- **Queries**: `task: search result | query: {content}`

### 3. Update Text Concatenation
Current (line 250): `" ".join(parts)`
Required: Apply proper prompt template with ticket title

## Key Incompatibilities
1. Missing prompt templates
2. No task-specific encoding methods
3. No distinction between document/query encoding
4. Generic `.encode()` instead of specialized methods

## Benefits After Implementation
- Better embedding quality for search tasks
- 768-dimensional embeddings (vs current model's dimensions)
- Support for 100+ languages
- Optimized for retrieval use cases