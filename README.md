# Orbis

<img width="781" height="728" alt="orbis" src="https://github.com/user-attachments/assets/6db8f4c3-5c26-4c75-a583-d8ead46751dc" />

A modular search platform for Azure DevOps work items and wiki content, built as a monorepo workspace containing shared infrastructure and specialized applications.

## Architecture

### Applications

**Orbis** - Advanced agentic RAG system providing multi-agent workflows for contextual analysis of Azure DevOps tickets. Implements project-aware intelligence, multi-source search, and wiki summarization through sophisticated orchestration.

**Orbis Search** - Lightweight hybrid search engine combining BM25 keyword search with semantic embeddings. Uses reciprocal rank fusion (RRF) and recency boosting for result optimization.

**Orbis Core** - Shared library providing common infrastructure: Azure DevOps connectors, embedding services, search components, database management, and utilities. Eliminates approximately 2,600 lines of code duplication across applications.

### Technology Stack

- Python 3.13+
- FastAPI for REST APIs
- SQLite with SQLAlchemy ORM
- Sentence Transformers for local embeddings
- Azure OpenAI for LLM operations
- ChromaDB (Orbis) and BM25S (Orbis Search) for vector operations
- UV for dependency management

## Development Setup

### Prerequisites

- Python 3.13+
- UV package manager
- Azure DevOps access (PAT or OAuth2)
- Azure OpenAI API access

### Initial Setup

```bash
# Clone repository
git clone <repository-url> orbis
cd orbis

# Install all workspace dependencies
uv sync

# Configure environment variables
# Copy .env.example to .env in each application directory and configure
```

### Running Applications

**Orbis (Advanced RAG)**
```bash
cd orbis
uv run main.py
# API available at http://localhost:7887
```

**Orbis Search (Hybrid Search)**
```bash
cd orbis-search
uv run main.py
# API available at http://localhost:7887
```

### Development Workflow

The monorepo uses UV workspace management. All packages are installed as editable by default, enabling immediate code changes across the workspace.

```bash
# Install dependencies for all workspace members
uv sync

# Add dependency to specific package
cd orbis-core && uv add numpy
cd orbis && uv add pyyaml
cd orbis-search && uv add requests

# Sync workspace after dependency changes
cd .. && uv sync

# Code changes in orbis-core are immediately available to applications
```

### Workspace Structure

```
orbis/
├── pyproject.toml          # Workspace configuration
├── orbis-core/             # Shared library
│   └── src/orbis_core/
│       ├── connectors/     # Azure DevOps integration
│       ├── embedding/      # Embedding generation
│       ├── search/         # Search components
│       ├── llm/            # LLM clients
│       ├── database/       # Database management
│       └── utils/          # Common utilities
├── orbis/                  # Agentic RAG application
│   ├── main.py
│   ├── infrastructure/     # Storage, LLM, connectors
│   ├── core/               # Agents, services, orchestration
│   ├── app/                # API routers, database models
│   └── config/             # Settings and configuration
└── orbis-search/           # Hybrid search application
    ├── main.py
    ├── app/                # API routers, database models
    ├── models/             # Data models
    └── services/           # Application services
```

## Configuration

Each application requires environment-specific configuration through `.env` files:

**Required Variables**
- `AZURE_DEVOPS_PAT` or OAuth2 credentials
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `DATABASE_URL` (defaults to SQLite)

**Optional Configuration**
- Data source definitions (see application-specific documentation)
- Model configurations for embeddings and reranking
- Search parameters and thresholds

Refer to individual application directories for detailed configuration guidance.

## Common Development Commands

```bash
# Code quality
ruff check .
ruff format .

# Type checking
mypy .

# Database operations (in application directory)
# Database is auto-initialized on application startup

# Clean build artifacts
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type d -name ".venv" -exec rm -r {} +
```

## Architecture Decisions

**Monorepo Approach**: Single repository enables atomic changes across core library and applications. Workspace dependencies ensure consistency without version coordination overhead.

**Src Layout**: Orbis Core uses modern src layout to ensure tests run against installed packages and prevent accidental imports of development code.

**Configuration-Driven Design**: Applications use configuration files for data sources and behavior, reducing code changes required for new integrations.

**Service Decoupling**: Generic services eliminate hardcoded assumptions while maintaining sophisticated functionality.
