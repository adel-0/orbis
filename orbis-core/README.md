# Orbis Core

Shared infrastructure library for Orbis applications, providing common functionality for data ingestion, embedding generation, search, and utilities. Eliminates approximately 2,600 lines of code duplication.

See the [main README](../README.md) for workspace setup.

## Components

### Connectors
**Azure DevOps Connector** - REST API client with PAT/OAuth2 authentication for work items, queries, and projects.

```python
from orbis_core.connectors.azure_devops import Client
client = Client(organization="myorg", pat="token")
work_items = await client.get_work_items([123, 456])
```

### Embedding
**Embedding Service** - Local embedding generation with sentence-transformers, automatic chunking for long documents, CUDA/CPU auto-detection.

```python
from orbis_core.embedding.embedding_service import EmbeddingService
service = EmbeddingService(model_name="intfloat/multilingual-e5-small")
embedding = service.encode_single_text("text")
```

### Search
**BM25 Service** - Fast keyword search.
**Rerank Service** - Cross-encoder semantic reranking.
**Hybrid Search Service** - Combines BM25 and semantic search with RRF and recency boosting.

```python
from orbis_core.search import BM25Service, RerankService, HybridSearchService
```

### LLM
**OpenAI Client** - Singleton Azure OpenAI client with connection pooling.

### Database
**Database Manager** - SQLAlchemy session factory and initialization.
**ContentEmbedding Model** - Tracks embedded content with change detection using content hashing.

### Scheduling
**Scheduler Service** - Background task scheduling for periodic operations.

### Utils
**Logging** - Structured logging configuration.
**Token Utils** - tiktoken-based token counting and chunking.
**Content Hash** - SHA256 hashing for change detection.
**Similarity** - Cosine similarity and score normalization.
**Progress Tracker** - Progress tracking with ETA for long operations.

## Design Principles

- Configuration-driven components with parameters instead of hardcoded values
- Single source of truth eliminating duplication
- Device-agnostic ML components with automatic CUDA/CPU detection
- Token-aware text processing with automatic chunking
- Production-tested components from live systems
