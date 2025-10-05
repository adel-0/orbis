# Orbis Core

Shared infrastructure library for Orbis applications, providing common functionality for data ingestion, embedding generation, search, and utilities.

## Overview

Orbis Core consolidates data ingestion, embedding generation, search capabilities, and utility functions previously duplicated across Orbis (agentic RAG) and Orbis Search applications. It follows a modular architecture with seven main components.

For detailed documentation, see [ORBIS_CORE.md](../ORBIS_CORE.md) in the repository root.

## Quick Start

### Installation

This package is part of the Orbis monorepo workspace and is installed automatically:

```bash
cd orbis/
uv sync
```

### Basic Usage

```python
# Azure DevOps connector
from orbis_core.connectors.azure_devops import AzureDevOpsClient

client = AzureDevOpsClient(organization="myorg", pat="token")
work_items = await client.get_work_items([123, 456])

# Embedding service
from orbis_core.embedding import EmbeddingService

service = EmbeddingService(model_name="intfloat/multilingual-e5-small")
embedding = service.encode_single_text("text to embed")

# Search infrastructure
from orbis_core.search import RerankService

reranker = RerankService(model_name="mixedbread-ai/mxbai-rerank-base-v2")
scores = reranker.rerank(query="user question", texts=candidate_docs)
```

## Components

- **connectors**: Azure DevOps REST API client with PAT and OAuth2 authentication
- **embedding**: Local embedding generation using sentence-transformers with text chunking
- **search**: BM25 keyword search, semantic reranking, and hybrid search with RRF
- **llm**: Azure OpenAI client singleton with connection pooling
- **database**: SQLAlchemy session factory and database initialization
- **scheduling**: Background task scheduling for periodic operations
- **utils**: Logging, token counting, content hashing, similarity metrics, progress tracking

## Documentation

See [ORBIS_CORE.md](../ORBIS_CORE.md) for:
- Detailed component documentation
- API reference and examples
- Design principles
- Dependencies and installation

## Version

Current version: **0.1.0**
