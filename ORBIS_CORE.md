# Orbis Core

Orbis Core is a shared infrastructure library that provides common functionality for both Orbis (agentic RAG system) and Orbis Search (hybrid search engine). It consolidates data ingestion, embedding generation, search capabilities, and utility functions that were previously duplicated across both applications.

## Architecture

Orbis Core follows a modular architecture with seven main components:

```
orbis-core/
└── src/orbis_core/
    ├── connectors/         # Data source connectors
    ├── embedding/          # Embedding generation
    ├── search/             # Search infrastructure
    ├── llm/                # LLM client services
    ├── database/           # Database management
    ├── scheduling/         # Task scheduling
    └── utils/              # Shared utilities
```

## Components

### Connectors

**Azure DevOps Connector** (`orbis_core.connectors.azure_devops`)

Provides REST API client for Azure DevOps with authentication and work item operations:
- `AzureDevOpsAuthMixin`: PAT and OAuth2 authentication
- `AzureDevOpsClient`: Core REST API operations for work items, queries, and projects
- `API_VERSIONS`: API version constants and field mappings

```python
from orbis_core.connectors.azure_devops import AzureDevOpsClient

client = AzureDevOpsClient(
    organization="myorg",
    pat="personal_access_token"
)
work_items = await client.get_work_items([123, 456])
```

### Embedding

**Embedding Service** (`orbis_core.embedding`)

Generates embeddings using sentence-transformers with support for long documents:
- `EmbeddingService`: Local embedding generation with CUDA/CPU support
- `chunk_text()`: Token-based text chunking with configurable overlap
- Streaming average for memory-efficient long text handling

```python
from orbis_core.embedding.embedding_service import EmbeddingService

service = EmbeddingService(model_name="intfloat/multilingual-e5-small")
embedding = service.encode_single_text("Long document text...")
```

Key features:
- Automatic chunking for documents exceeding model token limits
- Device auto-detection (CUDA/CPU)
- Batch processing support

### Search

**Search Infrastructure** (`orbis_core.search`)

Provides keyword search, semantic reranking, and hybrid search capabilities:

**BM25 Service** - Fast keyword search using bm25s library:
```python
from orbis_core.search.bm25_service import BM25Service

service = BM25Service()
service.build_index(texts=documents, ids=doc_ids)
results = service.search(query="search term", top_k=10)
```

**Rerank Service** - Cross-encoder reranking for semantic relevance:
```python
from orbis_core.search.rerank_service import RerankService

service = RerankService(model_name="mixedbread-ai/mxbai-rerank-base-v2")
scores = service.rerank(query="user question", texts=candidate_docs)
```

**Hybrid Search Service** - Combines BM25 and semantic search using Reciprocal Rank Fusion (RRF):
```python
from orbis_core.search.hybrid_search_service import HybridSearchService

service = HybridSearchService()
results = service.combine_results(
    semantic_results=semantic_matches,
    keyword_results=bm25_matches,
    tickets=ticket_data
)
```

The hybrid search service includes recency boosting (30-day to 2-year gradient) to prioritize recent content.

### LLM

**OpenAI Client** (`orbis_core.llm`)

Provides singleton Azure OpenAI client with connection pooling:
```python
from orbis_core.llm.openai_client import OpenAIClientService

service = OpenAIClientService()
response = service.client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Database

**Database Session Management** (`orbis_core.database`)

Provides SQLAlchemy session factory and database initialization:
```python
from orbis_core.database.session import DatabaseManager

manager = DatabaseManager(database_url="sqlite:///app.db")
manager.init_database(base_metadata)

with manager.get_session() as session:
    # Use session for queries
    pass
```

### Scheduling

**Scheduler Service** (`orbis_core.scheduling`)

Provides background task scheduling for periodic operations:
```python
from orbis_core.scheduling import SchedulerService

scheduler = SchedulerService()
scheduler.add_job(
    func=sync_data,
    trigger="interval",
    hours=24,
    id="daily_sync"
)
scheduler.start()
```

### Utils

**Shared Utilities** (`orbis_core.utils`)

Common utility functions used across applications:

**Logging** - Structured logging configuration:
```python
from orbis_core.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

**Token Utils** - tiktoken-based token counting and validation:
```python
from orbis_core.utils.token_utils import count_tokens, chunk_text_by_tokens

token_count = count_tokens(text, model="gpt-5-nano")
chunks = chunk_text_by_tokens(text, max_tokens=1000)
```

**Content Hash** - SHA256 hashing for change detection:
```python
from orbis_core.utils.content_hash import compute_content_hash

hash_value = compute_content_hash(content)
```

**Similarity** - Cosine similarity and score normalization:
```python
from orbis_core.utils.similarity import cosine_similarity, normalize_scores

similarity = cosine_similarity(vector1, vector2)
normalized = normalize_scores(scores)
```

**Constants** - Shared constants including HTML cleaning:
```python
from orbis_core.utils.constants import clean_html_content, DEFAULT_RERANK_MODEL

clean_text = clean_html_content(html_string)
```

## Design Principles

**Configuration-Driven**: Components accept configuration parameters rather than hardcoded values, enabling flexibility across different use cases.

**Single Source of Truth**: Each component exists once in orbis-core and is shared by both applications, eliminating duplication and inconsistencies.

**Device Agnostic**: ML components (embedding, reranking) automatically detect and use CUDA when available, falling back to CPU.

**Token-Aware**: Text processing utilities respect model token limits, with automatic chunking and validation.

**Production-Tested**: All components are extracted from proven implementations used in production systems.

## Dependencies

Core dependencies include:
- `aiohttp` - Async HTTP client
- `sqlalchemy` - Database ORM
- `openai` - Azure OpenAI integration
- `sentence-transformers` - Embedding models
- `torch` - ML framework (CUDA 12.9)
- `bm25s` - BM25 keyword search
- `tiktoken` - Token counting
- `msal` - Microsoft authentication

## Installation

Orbis Core is part of the Orbis monorepo workspace and is installed automatically:

```bash
cd orbis/
uv sync
```

Applications declare the dependency in their `pyproject.toml`:
```toml
dependencies = [
    "orbis-core",
    # other dependencies
]
```

## Version

Current version: **0.1.0**
