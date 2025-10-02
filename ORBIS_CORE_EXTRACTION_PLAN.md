# Orbis Core: Shared Library Extraction Plan

## Summary

This document outlines the strategy to extract shared infrastructure from `oncall-copilot` and `oncall-copilot-classic` into a unified `orbis-core` library. The extraction will reduce code duplication significantly while maintaining the distinct identities and purposes of both applications.

**Project Renaming:**
- `oncall-copilot` → **Orbis** (Advanced agentic RAG system)
- `oncall-copilot-classic` → **Orbis Search** (Lightweight hybrid search engine)
- Shared infrastructure → **Orbis Core** (Common library)

---

## Table of Contents

1. [Component Priority Ranking](#component-priority-ranking)
2. [Detailed Component Analysis](#detailed-component-analysis)
3. [Library Structure](#library-structure)
4. [Development Setup](#development-setup)
5. [Migration Plan](#migration-plan)
6. [Risk Mitigation](#risk-mitigation)
7. [Success Metrics](#success-metrics)
8. [Recommendations](#recommendations)

---

## Component Priority Ranking

Components are ranked based on:
- **Impact**: Business value and code quality improvement (1-10)
- **Reuse Potential**: How useful across both applications (1-10)
- **Extraction Difficulty**: Complexity and coupling level (1-10, lower is easier)
- **Source**: Which codebase has the better implementation

### Priority 1: High Impact, High Reuse, Medium-Low Difficulty

| Component | Impact | Reuse | Difficulty | LOC | Source |
|-----------|--------|-------|------------|-----|--------|
| Azure DevOps Connector | 10/10 | 10/10 | 5/10 | 760 | oncall-copilot |
| Embedding Service | 10/10 | 10/10 | 6/10 | 296 | oncall-copilot-classic |
| Reranking Service | 8/10 | 9/10 | 3/10 | 180 | oncall-copilot-classic |

**Total Priority 1**: ~1,236 LOC

### Priority 2: High Impact, Single-Source, Medium Difficulty

| Component | Impact | Reuse | Difficulty | LOC | Source |
|-----------|--------|-------|------------|-----|--------|
| BM25 Service | 9/10 | 8/10 | 4/10 | 196 | oncall-copilot-classic |
| Hybrid Search Service | 9/10 | 8/10 | 4/10 | 151 | oncall-copilot-classic |
| OpenAI Client Service | 7/10 | 10/10 | 2/10 | 50 | oncall-copilot |

**Total Priority 2**: ~397 LOC

### Priority 3: Medium Impact, Foundational, Low Difficulty

| Component | Impact | Reuse | Difficulty | LOC | Source |
|-----------|--------|-------|------------|-----|--------|
| Database Session Management | 6/10 | 10/10 | 2/10 | 133 | Either (identical) |
| Logging Utilities | 7/10 | 9/10 | 2/10 | 120 | oncall-copilot |
| Constants | 7/10 | 9/10 | 2/10 | 180 | oncall-copilot |
| Token Utilities | 7/10 | 9/10 | 2/10 | 127 | oncall-copilot |
| Content Hash | 5/10 | 8/10 | 2/10 | 40 | oncall-copilot |
| Similarity Utils | 5/10 | 8/10 | 2/10 | 30 | oncall-copilot |

**Total Priority 3**: ~630 LOC

### Priority 4: Do Not Extract

| Component | Reason |
|-----------|--------|
| Vector Services | Fundamentally different architectures (multi-collection vs single-collection) |
| Data Models | Different domain models (generic Content vs specific WorkItem) |
| API Routers | App-specific business logic |
| Agent/Orchestration | Unique to oncall-copilot (agentic RAG) |
| Field Discovery | Specific to oncall-copilot-classic optimization |

---

## Detailed Component Analysis

### 1. Azure DevOps Connector Infrastructure ⭐⭐⭐⭐⭐

**Impact**: 10/10 | **Reuse**: 10/10 | **Difficulty**: 5/10 | **Source**: `oncall-copilot`

#### Why oncall-copilot version is superior:

1. **Better Architecture**:
   - Separation of concerns with `AzureDevOpsAuthMixin`
   - Cleaner OAuth2 and PAT support
   - Configuration-driven design

2. **More Maintainable**:
   - Modular structure (auth, client, services)
   - Constants externalized
   - Better error handling

3. **Feature Complete**:
   - Work items support
   - Parallel processing with rate limiting
   - Incremental sync support

**Note**: Wiki support is NOT extracted as it's only used by Orbis, not Orbis Search.

#### Components to Extract:

```
infrastructure/connectors/azure_devops/
├── auth.py                          # 80 LOC
│   └── AzureDevOpsAuthMixin: PAT + OAuth2 authentication
├── azure_devops_client.py           # 350 LOC
│   └── AzureDevOpsClient: Core REST API operations
├── constants.py                     # 50 LOC
│   ├── API_VERSIONS
│   ├── CORE_WORKITEM_FIELDS
│   └── DEFAULT_BATCH_SIZE
└── work_item_service.py             # 280 LOC
    └── WorkItemService: Work item ingestion

Total: ~760 LOC
```

#### Migration Strategy:

**For Orbis (oncall-copilot):**
```python
# Before
from infrastructure.connectors.azure_devops import AzureDevOpsClient

# After
from orbis_core.connectors.azure_devops import AzureDevOpsClient
```

**For Orbis Search (oncall-copilot-classic):**
```python
# Before
from services.azure_devops_client import AzureDevOpsClient  # 524 LOC

# After
from orbis_core.connectors.azure_devops import AzureDevOpsClient
# Remove services/azure_devops_client.py entirely
```

#### Benefits:

- Eliminates **~500 LOC** duplication
- Single source of truth for ADO integration
- Bug fixes benefit both projects
- OAuth2 support available in both apps

---

### 2. Embedding Service Infrastructure ⭐⭐⭐⭐⭐

**Impact**: 10/10 | **Reuse**: 10/10 | **Difficulty**: 3/10 | **Source**: `oncall-copilot-classic`

#### Why oncall-copilot-classic version:

1. **Critical Chunking Support**:
   - Token-aware text chunking for long documents
   - Streaming average for memory efficiency
   - Direct tokenizer integration
   - Handles model token limits gracefully

2. **Simpler Architecture**:
   - Local embedding provider only (always used)
   - No provider abstraction needed (dual provider support not required)
   - Direct sentence-transformers integration
   - Less complexity, easier maintenance

3. **Production-Tested**:
   - Proven implementation in Orbis Search
   - Handles large document ingestion
   - Efficient batch processing

#### Components to Extract:

```
infrastructure/embedding/
├── embedding_service.py             # 260 LOC
│   ├── EmbeddingService: Main service
│   ├── Local provider (sentence-transformers)
│   ├── CUDA/CPU device management
│   └── Batch processing
└── text_chunking.py                 # 36 LOC
    ├── chunk_text(): Token-based chunking with overlap
    └── should_chunk_text(): Token count validation

Total: ~296 LOC
```

#### Implementation Details:

**Embedding Service with Chunking:**
```python
class EmbeddingService:
    def encode_single_text(self, text: str) -> list[float]:
        # Token-based chunking for long documents
        if should_chunk_text(text, self.max_length, self.tokenizer):
            return self._encode_long_text_chunked(text, self.max_length)
        return self.encode_texts([text])[0]

    def _encode_long_text_chunked(self, text: str, max_length: int) -> list[float]:
        # Streaming average for memory efficiency
        chunks = chunk_text(text, max_length, self.tokenizer)
        embeddings = [self.model.encode(chunk) for chunk in chunks]
        return np.mean(embeddings, axis=0).tolist()
```

#### Migration Strategy:

**For Orbis:**
```python
# Before
from infrastructure.storage.embedding_service import EmbeddingService

# After
from orbis_core.embedding import EmbeddingService
# Simplifies from provider architecture to direct implementation
```

**For Orbis Search:**
```python
# Before
from services.embedding_service import EmbeddingService  # 296 LOC

# After
from orbis_core.embedding import EmbeddingService
# Remove services/embedding_service.py entirely
```

#### Benefits:

- Eliminates **~296 LOC** duplication
- Critical chunking support for both applications
- Simpler architecture (local-only, as always used)
- Memory-efficient long text handling
- Production-tested implementation

---

### 3. Reranking Service ⭐⭐⭐⭐

**Impact**: 8/10 | **Reuse**: 9/10 | **Difficulty**: 3/10 | **Source**: `oncall-copilot-classic`

#### Why oncall-copilot-classic version:

- **Tokenizer exposure**: Critical for chunking long documents
- **Production-tested**: Used extensively in Orbis Search
- Same CrossEncoder functionality as oncall-copilot
- Device auto-detection (CUDA/CPU)
- Batch processing support

#### Component to Extract:

```
infrastructure/search/
└── rerank_service.py                # 180 LOC
    ├── RerankService: Cross-encoder reranking
    ├── Device auto-detection (CUDA/CPU)
    ├── Tokenizer exposure for chunking
    └── Batch processing support

Total: ~180 LOC
```

#### Key Feature - Tokenizer Access:

```python
class RerankService:
    def _load_model(self):
        self.model = CrossEncoder(self.model_name, **init_kwargs)
        self.tokenizer = self.model.tokenizer  # Exposed for chunking

    def get_tokenizer(self):
        """Expose tokenizer for text chunking operations"""
        return self.tokenizer
```

#### Migration Strategy:

Both apps:
```python
# Before (oncall-copilot)
from infrastructure.storage.rerank_service import RerankService

# Before (oncall-copilot-classic)
from services.rerank_service import RerankService

# After
from orbis_core.search import RerankService
```

#### Benefits:

- Eliminates **~180 LOC** duplication
- Single reranker implementation
- Consistent cross-encoder behavior
- Shared model caching

---

### 4. BM25 & Hybrid Search Services ⭐⭐⭐⭐

**Impact**: 9/10 | **Reuse**: 8/10 | **Difficulty**: 4/10 | **Source**: `oncall-copilot-classic` (exclusive)

#### Why Extract (Currently Only in Orbis Search):

- **BM25**: Fast keyword search using `bm25s` library
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) + recency boosting
- Production-tested algorithm with excellent search quality
- Would enable hybrid search in Orbis (currently semantic-only)

#### Components to Extract:

```
infrastructure/search/
├── bm25_service.py                  # 196 LOC
│   ├── BM25Service: Keyword search with bm25s
│   ├── Index persistence (save/load)
│   └── Tokenization and scoring
└── hybrid_search_service.py         # 151 LOC
    ├── HybridSearchService: RRF fusion
    ├── Reciprocal Rank Fusion algorithm
    └── Recency boost calculation (30d-2yr gradient)

Total: ~347 LOC
```

#### Key Algorithms:

**Reciprocal Rank Fusion (RRF):**
```python
def combine_results(semantic_results, keyword_results, rrf_k=60):
    for ticket_id in all_ticket_ids:
        semantic_rank = semantic_ranks.get(ticket_id, None)
        keyword_rank = keyword_ranks.get(ticket_id, None)

        rrf_score = 0
        if semantic_rank:
            rrf_score += 1 / (rrf_k + semantic_rank)
        if keyword_rank:
            rrf_score += 1 / (rrf_k + keyword_rank)
```

**Recency Boosting:**
```python
def calculate_recency_boost(created_date, max_boost=0.05):
    age_days = (now - created_date).days
    if age_days <= 30:
        return max_boost  # Full boost
    elif age_days <= 730:
        # Linear decay over 2 years
        return max_boost * (1.0 - (age_days - 30) / 700.0)
    else:
        return 0.0  # No boost for old tickets
```

#### Migration Strategy:

**For Orbis Search:**
```python
# Before
from services.bm25_service import BM25Service
from services.hybrid_search_service import HybridSearchService

# After
from orbis_core.search import BM25Service, HybridSearchService
```

**For Orbis (New Feature):**
```python
# Optional enhancement to multi-modal search
from orbis_core.search import BM25Service, HybridSearchService

# Can now add keyword search to agentic RAG workflow
```

#### Benefits:

- Eliminates **347 LOC** in Orbis Search
- Enables hybrid search in Orbis
- Proven RRF algorithm shared
- Recency boosting available to both

---

### 5. OpenAI Client Service ⭐⭐⭐⭐

**Impact**: 7/10 | **Reuse**: 10/10 | **Difficulty**: 2/10 | **Source**: `oncall-copilot`

#### Why oncall-copilot version:

- Singleton pattern with `@lru_cache` decorator
- Clean service wrapper pattern
- Shared client instance (resource efficient)
- oncall-copilot-classic has scattered inline OpenAI usage

#### Component to Extract:

```
infrastructure/llm/
└── openai_client.py                 # 50 LOC
    ├── get_azure_openai_client(): Cached singleton
    └── OpenAIClientService: Service wrapper

Total: ~50 LOC
```

#### Implementation:

```python
@lru_cache(maxsize=1)
def get_azure_openai_client() -> AzureOpenAI:
    """Get a shared Azure OpenAI client instance."""
    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    )

class OpenAIClientService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = get_azure_openai_client()
        return self._client
```

#### Migration Strategy:

**For Orbis:**
```python
# Before
from infrastructure.llm.openai_client import OpenAIClientService

# After
from orbis_core.llm import OpenAIClientService
```

**For Orbis Search:**
```python
# Before (scattered usage in summary_service.py)
from openai import AzureOpenAI
client = AzureOpenAI(...)

# After
from orbis_core.llm import OpenAIClientService
openai_service = OpenAIClientService()
client = openai_service.client
```

#### Benefits:

- Single OpenAI client initialization
- Better resource management
- Standardized LLM access pattern
- Easier to add retry logic, rate limiting centrally

---

### 6. Database Session Management ⭐⭐⭐

**Impact**: 6/10 | **Reuse**: 10/10 | **Difficulty**: 2/10 | **Source**: Either (nearly identical)

#### Why Extract:

Both implementations are virtually identical (~133 LOC each). Provides:
- Database initialization
- Session factory management
- Context manager support
- Migration support (Alembic)

#### Component to Extract:

```
database/
└── session.py                       # 133 LOC
    ├── DatabaseManager: Singleton database manager
    ├── init_database(): Database creation
    ├── get_session(): Session factory
    └── get_database_info(): Diagnostics

Total: ~133 LOC
```

#### Migration Strategy:

Both apps:
```python
# Before
from app.db.session import DatabaseManager

# After
from orbis_core.database import DatabaseManager
# Remove app/db/session.py entirely
```

#### Benefits:

- Eliminates **~133 LOC** duplication
- Unified database initialization pattern
- Consistent session management
- Single place for database diagnostics

---

### 7. Shared Utilities ⭐⭐⭐⭐

**Impact**: 7/10 | **Reuse**: 9/10 | **Difficulty**: 2/10 | **Source**: `oncall-copilot`

#### Why oncall-copilot version:

- More comprehensive utility collection
- Better organized module structure
- Higher quality implementations

#### Components to Extract:

```
utils/
├── logging.py                       # 120 LOC
│   ├── setup_logging(): Structured logging config
│   ├── get_logger(): Logger factory
│   └── Custom formatters
├── constants.py                     # 180 LOC
│   ├── API constants
│   ├── Search thresholds
│   ├── Model defaults
│   └── clean_html_content(): HTML cleaning
├── token_utils.py                   # 127 LOC
│   ├── count_tokens(): tiktoken-based counting
│   ├── estimate_tokens_for_messages(): OpenAI message estimation
│   ├── chunk_text_by_tokens(): Token-aware chunking
│   └── validate_token_limit(): Token validation
├── content_hash.py                  # 40 LOC
│   ├── compute_content_hash(): SHA256 hashing
│   └── Content change detection
└── similarity.py                    # 30 LOC
    ├── cosine_similarity(): Vector similarity
    └── normalize_scores(): Score normalization

Total: ~497 LOC
```

#### Migration Strategy:

**For Orbis:**
```python
# Before
from utils.logging import setup_logging, get_logger
from utils.constants import DEFAULT_RERANK_MODEL
from utils.token_utils import count_tokens

# After
from orbis_core.utils import setup_logging, get_logger
from orbis_core.utils import DEFAULT_RERANK_MODEL
from orbis_core.utils import count_tokens
```

**For Orbis Search:**
```python
# After
from orbis_core.utils import setup_logging, get_logger
from orbis_core.utils import count_tokens  # New capability
```

#### Enhancements:

Merge `text_chunking.py` from Orbis Search into `token_utils.py`:
```python
# Add to token_utils.py
def chunk_text_with_tokenizer(text: str, max_length: int, tokenizer: Any) -> Iterator[str]:
    """Token-based chunking using model tokenizer (from Orbis Search)"""
    # Implementation from services/text_chunking.py
```

#### Benefits:

- Standardized logging across both apps
- Shared constant definitions (reduce config drift)
- Token counting available in both (tiktoken-based)
- Content hashing for change detection
- Similarity calculations shared

---

## Library Structure

### Modern Src Layout

Following Python packaging best practices (per packaging.python.org), we use **src layout** instead of flat layout. This ensures tests run against the installed version and prevents accidental imports of development code.

### Complete Directory Tree

```
orbis-core/
├── pyproject.toml                   # Package metadata and dependencies
├── README.md                        # Library documentation
├── LICENSE                          # MIT License
├── .gitignore
└── src/
    └── orbis_core/
        ├── __init__.py              # Version and main exports
        │
        ├── connectors/              # Data source connectors
        │   ├── __init__.py
        │   └── azure_devops/
        │       ├── __init__.py
        │       ├── auth.py          # Authentication mixin (PAT + OAuth2)
        │       ├── azure_devops_client.py  # Core REST API client
        │       ├── constants.py     # API versions and field mappings
        │       └── work_item_service.py    # Work item operations
        │
        ├── embedding/               # Embedding generation
        │   ├── __init__.py
        │   ├── embedding_service.py # Main embedding service (local)
        │   └── text_chunking.py     # Text chunking utilities
        │
        ├── search/                  # Search infrastructure
        │   ├── __init__.py
        │   ├── bm25_service.py      # BM25 keyword search
        │   ├── hybrid_search_service.py # RRF fusion + recency
        │   └── rerank_service.py    # Cross-encoder reranking
        │
        ├── llm/                     # LLM client services
        │   ├── __init__.py
        │   └── openai_client.py     # Shared OpenAI client
        │
        ├── database/                # Database management
        │   ├── __init__.py
        │   └── session.py           # Session factory and manager
        │
        └── utils/                   # Shared utilities
            ├── __init__.py
            ├── logging.py           # Structured logging
            ├── constants.py         # Shared constants
            ├── token_utils.py       # Token counting and chunking
            ├── content_hash.py      # Content hashing
            └── similarity.py        # Similarity calculations

Total: ~2,600 LOC extracted
```

**Key Benefits of Src Layout:**
- Ensures tests run against installed version
- Prevents accidental import of development code
- Forces proper package installation (`pip install -e .`)
- Modern best practice recommended by packaging.python.org

### Package Configuration (pyproject.toml)

```toml
[project]
name = "orbis-core"
version = "0.1.0"
description = "Shared infrastructure for Orbis applications"
readme = "README.md"
requires-python = ">=3.13"
license = {text = "MIT"}

dependencies = [
    "aiohttp>=3.9.0",
    "sqlalchemy>=2.0.0",
    "cryptography>=43.0.0",
    "msal>=1.20.0",
    "openai>=1.98.0",
    "tiktoken>=0.8.0",
    "sentence-transformers>=5.1.0",
    "torch",
    "bm25s>=0.2.14",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.12.0",
    "mypy>=1.17.0",
]

[tool.uv]
index-url = "https://pypi.org/simple/"

[tool.uv.sources]
torch = { index = "pytorch-cu129" }

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true
```

---

## Development Setup

#### Monorepo

**Structure:**
```
orbis/                               # Single git repository
├── .git/
├── .gitignore
├── pyproject.toml                   # Workspace definition
├── README.md
│
├── orbis-core/                      # Shared library
│   ├── pyproject.toml
│   ├── README.md
│   └── src/orbis_core/
│       ├── connectors/
│       ├── embedding/
│       ├── search/
│       ├── llm/
│       ├── database/
│       └── utils/
│
├── orbis/                 # Formerly oncall-copilot
│   ├── pyproject.toml               # depends on: orbis-core
│   ├── README.md
│   └── src/orbis/
│       └── [app code]
│
└── orbis-search/                    # Formerly oncall-copilot-classic
    ├── pyproject.toml               # depends on: orbis-core
    ├── README.md
    └── src/orbis_search/
        └── [app code]
```

**Advantages:**
- ✅ **Atomic Changes**: Single PR can update core + both apps simultaneously
- ✅ **Simple Workflow**: One git clone, one uv sync, everything works
- ✅ **No Version Coordination**: Workspace dependencies always in sync
- ✅ **Fast Iteration**: Edit core, immediately available in apps (editable install)
- ✅ **Single CI/CD**: One pipeline for all components
- ✅ **Visibility**: See all related changes in one place

**Disadvantages:**
- ⚠️ **Access Control**: All-or-nothing (can't restrict access to specific apps)
- ⚠️ **Large Clone**: Must clone all code even if working on one app
- ⚠️ **Single Failure Point**: Issues with one component can block entire repo

---

### Monorepo Implementation Details

**Root Workspace Configuration (`orbis/pyproject.toml`):**

```toml
[tool.uv.workspace]
members = [
    "orbis-core",
    "orbis",
    "orbis-search"
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.12.0",
    "mypy>=1.17.0",
]
```

**Application Dependencies (`orbis/pyproject.toml`):**

```toml
[project]
name = "orbis"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    "orbis-core",  # Workspace dependency - always latest
    "fastapi==0.116.1",
    "uvicorn[standard]==0.35.0",
    "chromadb==1.0.20",
    "pydantic==2.11.7",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
    "pillow>=11.3.0",
]
```

**Development Workflow:**

```bash
# Initial setup
git clone <your-repo-url> orbis && cd orbis
uv sync  # Installs all workspace packages as editable

# Run applications
cd orbis && uv run main.py
cd orbis-search && uv run main.py

# Edit core library - changes immediately available
cd orbis-core/src/orbis_core
vim connectors/azure_devops/auth.py

# Add dependencies
cd orbis-core && uv add numpy
cd orbis && uv add pyyaml
cd .. && uv sync  # Sync workspace
```

**Versioning Strategy:**

- **Development**: Workspace references (no versions)
- **Production** (future): Pin versions if needed: `orbis-core==0.2.1`

---

## Migration Plan

### Overview

The migration will be executed in **5 phases** in sequence. Each phase builds on the previous one.

### Phase 1: Foundation

**Goal**: Extract low-risk, foundational components

**Components**:
- Database Session Management
- Shared Utilities

**Steps**:

1. **Create Monorepo Structure**
   ```bash
   # Rename projects
   orbis/
   ├── oncall-copilot/              → orbis/
   └── oncall-copilot-classic/      → orbis-search/

   # Add workspace structure
   orbis/
   ├── pyproject.toml              # Workspace config
   ├── orbis-core/                 # New package
   ├── orbis/            # Renamed
   └── orbis-search/               # Renamed
   ```

2. **Create orbis-core Package with Src Layout**
   - Create `orbis-core/src/orbis_core/` directory structure
   - Add `pyproject.toml` with dependencies
   - Add workspace configuration to root `pyproject.toml`
   - Run `uv sync` to verify workspace setup

3. **Extract Database Session**
   - Copy `app/db/session.py` to `src/orbis_core/database/session.py`
   - Update imports in both applications: `from orbis_core.database import DatabaseManager`
   - Verify database initialization in both apps

4. **Extract Utilities**
   - Copy from oncall-copilot to `src/orbis_core/utils/`:
     - `utils/logging.py`
     - `utils/constants.py`
     - `utils/token_utils.py`
     - `utils/content_hash.py`
     - `utils/similarity.py`
   - Update imports in both applications: `from orbis_core.utils import ...`
   - Verify functionality

5. **Cleanup**
   - Remove duplicated files from applications
   - Update documentation
   - Commit changes

**Validation**:
- Both applications start successfully
- Logging output is correct
- Database connections work
- No breaking changes

---

### Phase 2: Core Integration

**Goal**: Extract Azure DevOps connector and embedding service

**Components**:
- Azure DevOps Connector (work items only, no wiki)
- Embedding Service (oncall-copilot-classic version)

**Steps**:

1. **Extract Azure DevOps Connector**
   - Copy from `oncall-copilot/infrastructure/connectors/azure_devops/`:
     - `auth.py`
     - `azure_devops_client.py`
     - `constants.py`
     - `work_item_service.py` (NOT wiki_service.py)
   - Create `src/orbis_core/connectors/azure_devops/`
   - Update imports in orbis
   - Replace `services/azure_devops_client.py` in orbis-search
   - Verify work item ingestion in both apps

2. **Extract Embedding Service**
   - Copy from `oncall-copilot-classic/services/`:
     - `embedding_service.py` → `src/orbis_core/embedding/`
     - `text_chunking.py` → `src/orbis_core/embedding/`
   - Update imports in both applications
   - Verify embedding generation with chunking

3. **Cleanup**
   - Remove `services/azure_devops_client.py` from orbis-search
   - Remove `services/embedding_service.py` from orbis-search
   - Remove embedding provider files from orbis
   - Commit changes

**Validation**:
- Work item ingestion works in both apps
- OAuth2 authentication works
- Embedding generation works with chunking
- Long document handling via chunking

---

### Phase 3: Search Infrastructure

**Goal**: Consolidate search components

**Components**:
- Reranking Service (oncall-copilot-classic version with tokenizer)
- BM25 Service
- Hybrid Search Service
- OpenAI Client

**Steps**:

1. **Extract Reranking Service**
   - Copy from `oncall-copilot-classic/services/rerank_service.py`
   - Create `src/orbis_core/search/rerank_service.py`
   - Update imports in both applications
   - Verify reranking works with tokenizer exposure

2. **Extract BM25 and Hybrid Search**
   - Copy from `oncall-copilot-classic/services/`:
     - `bm25_service.py` → `src/orbis_core/search/`
     - `hybrid_search_service.py` → `src/orbis_core/search/`
   - Update imports in orbis-search
   - Verify hybrid search and RRF algorithm

3. **Extract OpenAI Client**
   - Copy from `oncall-copilot/infrastructure/llm/openai_client.py`
   - Create `src/orbis_core/llm/openai_client.py`
   - Refactor orbis-search summary service to use shared client
   - Verify LLM operations

4. **Cleanup**
   - Remove search service files from both applications
   - Commit changes

**Validation**:
- Reranking works with tokenizer
- BM25 and hybrid search work correctly
- RRF scores are correct
- OpenAI client singleton functions properly

---

### Phase 4: Cleanup and Documentation

**Goal**: Final cleanup and documentation

**Steps**:

1. **Remove All Duplicate Code**
   - Delete replaced files from both applications:
     - orbis: provider files, old connector files
     - orbis-search: `services/azure_devops_client.py`, `services/embedding_service.py`, `services/rerank_service.py`, etc.
   - Clean up unused imports
   - Verify both applications run correctly

2. **Write Library Documentation**
   - README.md with usage examples
   - API reference for each module
   - Architecture overview
   - Installation and development guide

3. **Update Application Documentation**
   - Update READMEs for both applications
   - Document dependencies on orbis-core
   - Update architecture diagrams

4. **Version and Tag**
   - Set orbis-core version to `0.1.0`
   - Tag in git: `orbis-core-v0.1.0`
   - Create release notes

**Validation**:
- All duplicate code removed
- No unused imports
- Documentation complete
- Both applications run successfully

---

### Phase 5: Final Verification

**Goal**: Comprehensive verification of extraction

**Steps**:

1. **End-to-End Verification**
   - Run complete data ingestion pipeline in both apps
   - Generate embeddings for test datasets
   - Perform searches in both apps
   - Verify all functionality works as expected

2. **Code Review**
   - Review all changes for consistency
   - Ensure proper error handling
   - Verify logging is appropriate
   - Check for any remaining duplication

3. **Performance Verification**
   - Compare performance metrics before/after
   - Verify no significant regressions
   - Document any performance changes

4. **Final Documentation Review**
   - Ensure all documentation is up to date
   - Verify code examples work
   - Check for completeness

**Validation**:
- All functionality works correctly
- Performance is acceptable
- Documentation is complete
- Ready for ongoing development

---

## Risk Mitigation

### Technical Risks

#### Risk 1: Breaking Changes During Extraction

**Likelihood**: Medium
**Impact**: High
**Mitigation**:

1. **Feature Flags**
   ```python
   # Allow toggling between old and new implementations during transition
   USE_ORBIS_CORE = os.getenv("USE_ORBIS_CORE", "true") == "true"

   if USE_ORBIS_CORE:
       from orbis_core.connectors import AzureDevOpsClient
   else:
       from services.azure_devops_client import AzureDevOpsClient
   ```

2. **Parallel Run Validation**
   - Run both implementations side-by-side
   - Compare outputs
   - Validate identical behavior

3. **Gradual Migration**
   - One component at a time
   - Validation gate after each component
   - Don't proceed if issues found

**Detection**:
- Application startup failures
- Runtime errors
- Output discrepancies

**Response**:
- Revert specific component
- Fix and re-validate
- Don't proceed to next phase

---

#### Risk 2: Dependency Conflicts

**Likelihood**: Low (using uv workspace)
**Impact**: Medium
**Mitigation**:

1. **Use uv Workspace**
   - Single lock file for all packages
   - Consistent dependency resolution
   - Automatic conflict detection

2. **Pin Critical Dependencies**
   ```toml
   [project]
   dependencies = [
       "openai==1.98.0",  # Exact version
       "sqlalchemy>=2.0.0,<3.0.0",  # Version range
   ]
   ```

3. **Verify Dependency Installation**
   ```bash
   # Verify fresh install works
   rm -rf .venv
   uv sync
   # Verify applications start successfully
   ```

4. **Document Dependency Constraints**
   - Why specific versions are pinned
   - Known compatibility issues
   - Upgrade path

**Detection**:
- uv sync failures
- Import errors
- Runtime errors

**Response**:
- Adjust version constraints
- Update dependencies
- Test compatibility

---

#### Risk 3: Import Circular Dependencies

**Likelihood**: Low
**Impact**: Medium
**Mitigation**:

1. **Clear Dependency Hierarchy**
   ```
   Applications → orbis-core (one-way)
   orbis-core modules → No application dependencies
   ```

2. **Avoid Cross-Module Dependencies in Core**
   - Each module should be independent
   - Shared utilities at top level only

3. **Lazy Imports**
   ```python
   # Instead of top-level import
   def get_client():
       from orbis_core.connectors import AzureDevOpsClient
       return AzureDevOpsClient()
   ```

4. **Static Analysis**
   - Use `mypy` to detect circular imports
   - Use `import-linter` to enforce rules

**Detection**:
- Import errors at startup
- Mypy errors
- Runtime errors

**Response**:
- Refactor to remove circular dependency
- Use lazy imports
- Restructure modules

---

### Process Risks

#### Risk 4: Scope Creep

**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:

1. **Stick to Priority Ranking**
   - Only extract prioritized components
   - Resist temptation to extract everything

2. **Phase-Based Approach**
   - Clear scope for each phase
   - Validation gate before next phase
   - Don't add features during extraction

3. **Document "Do Not Extract"**
   - Clear rationale for exclusions
   - Prevent re-discussion

4. **Time Boxes**
   - Set deadlines for each phase
   - Move to next phase even if not perfect

**Detection**:
- Timeline slippage
- Adding non-prioritized components
- Feature additions during extraction

**Response**:
- Review scope and cut non-essential work
- Defer enhancements to post-migration
- Focus on completing planned extraction

---

## Conclusion

This extraction plan provides a focused, actionable roadmap to reduce code duplication while maintaining the distinct identities of Orbis (agentic RAG) and Orbis Search (hybrid search).

**Key Takeaways**:

1. **~2,600 LOC** extracted into shared library (src layout)
2. **5-phase sequential approach** with validation after each phase
3. **Monorepo with uv workspace** recommended (objectively evaluated vs alternatives)
4. **Prioritized extraction**: Azure DevOps connector → Embedding → Search components
5. **Maintain app separation**: Core provides infrastructure, not business logic

**Components Extracted**:
- Azure DevOps Connector (work items only, no wiki): 760 LOC
- Embedding Service (oncall-copilot-classic with chunking): 296 LOC
- Reranking Service (oncall-copilot-classic with tokenizer): 180 LOC
- BM25 & Hybrid Search: 347 LOC
- OpenAI Client: 50 LOC
- Database Session & Utilities: ~630 LOC
- **Total**: ~2,263 LOC

**Components NOT Extracted**:
- Wiki support (only used by Orbis)
- Vector services (architectures too different)
- Data models (app-specific)
- Orchestrator/agents (only in Orbis)
- Generic data ingestion (only in Orbis)

**Next Steps**:

1. Review and confirm repository strategy (monorepo recommended)
2. Begin Phase 1: Foundation (database + utilities)
3. Continue through phases sequentially
4. Validate after each phase before proceeding
