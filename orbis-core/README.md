# Orbis Core

Shared infrastructure library for Orbis applications.

## Components

- **connectors**: Data source connectors (Azure DevOps)
- **embedding**: Embedding generation with text chunking
- **search**: Search infrastructure (BM25, hybrid search, reranking)
- **llm**: LLM client services (OpenAI)
- **database**: Database session management
- **utils**: Shared utilities (logging, token counting, hashing)

## Installation

This package is part of the Orbis monorepo workspace and is installed automatically with `uv sync`.

## Usage

```python
from orbis_core.connectors.azure_devops import AzureDevOpsClient
from orbis_core.embedding import EmbeddingService
from orbis_core.search import RerankService
```
