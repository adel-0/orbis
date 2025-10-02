# Orbis Search

A lightweight semantic search system for Azure DevOps tickets. This is a streamlined version of Orbis focused on providing fast, accurate semantic search of relevant Azure DevOps work items using local embedding models and vector database technology.

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.13+
- **uv**: Package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Hardware**: 4GB+ RAM, GPU recommended (NVIDIA GTX 1060 or better)
- **Azure DevOps**: Personal Access Token with Work Items (Read) permission

### Installation

```bash
# Clone repository
git clone https://github.com/adel-0/orbis-search
cd orbis-search

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start the server
uv run main.py
```

Access the API at `http://localhost:7887` and interactive docs at `http://localhost:7887/docs`

## 🐳 Docker Deployment

For production deployments or quick setup without local dependencies:

```bash
# Quick setup with docker-compose
docker-compose up -d

# Verify deployment
curl http://localhost:7887/health
```

### Docker Configuration

1. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Configure your Azure DevOps data sources via the API
   # Optionally configure Azure OpenAI for summarization
   ```

2. **Build and Run**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **GPU Support** (Optional):
   - Install nvidia-docker2
   - Container automatically detects and uses GPU if available
   - Falls back to CPU if GPU unavailable

### Docker Volumes
- `./data` → `/app/data`: Persistent storage for database, vector store, and model cache
- Models are downloaded once and cached across restarts

### Container Management
```bash
# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Update and restart
git pull
docker-compose build --no-cache
docker-compose up -d
```

## 🌟 Key Features

### Multi-Source Data Ingestion
- Connect to multiple Azure DevOps organizations simultaneously
- Access different projects within and across organizations
- Execute multiple WIQL queries per data source
- Parallel processing with intelligent rate limiting

### Semantic Search Engine
- **Local Embedding Models**: Uses sentence-transformers for text embeddings
- **Vector Database**: ChromaDB with persistent storage
- **Real-time Search**: Sub-second response times
- **Cross-Source Search**: Unified search across all configured sources

### AI-Powered Analysis
- Optional GPT-4o-mini integration for result summarization
- Context enhancement with product documentation
- Specialized OnCall Dispatch Project Engineer persona

## 📋 Configuration

### Essential Environment Variables

```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=7887

# Local Embedding Configuration
EMBEDDING_DEVICE=gpu  # or cpu
EMBEDDING_BATCH_SIZE=32
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Azure OpenAI (optional - for summarization only)
AZURE_OPENAI_ENDPOINT=https://resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_MODEL=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Vector Database Configuration
CHROMA_COLLECTION_NAME=workitems
CHROMA_DB_PATH=data/chroma_db

# Database (defaults to SQLite)
DATABASE_URL=sqlite:///data/database/orbis_search.db

# Data Ingestion Configuration
USE_INCREMENTAL_SYNC=true

# Scheduler Configuration
SCHEDULER_ENABLED=false
SCHEDULER_INTERVAL_HOURS=6
```

See `.env.example` for complete configuration options.

## 🔌 API Usage

### Data Source Management

```bash
# Create a data source
curl -X POST "http://localhost:7887/datasources" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-team",
    "organization": "my-org",
    "project": "My Project",
    "pat": "your-pat-token",
    "query_ids": ["query-id"],
    "enabled": true
  }'

# List all sources
curl -X GET "http://localhost:7887/datasources"
```

### Data Ingestion

```bash
# Ingest from all enabled sources
curl -X POST "http://localhost:7887/ingest" \
  -H "Content-Type: application/json" \
  -d '{"force_full_sync": false}'
```

### Generate Embeddings

```bash
# Generate embeddings for ingested tickets
curl -X POST "http://localhost:7887/embed" \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": false}'
```

### Semantic Search

```bash
# Search across all tickets
curl -X POST "http://localhost:7887/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication failure",
    "top_k": 5
  }'
```

## 📚 Documentation

- **[Search and Filtering Architecture](docs/SEARCH_AND_FILTERING_ARCHITECTURE.md)** - How filtering and search works
- **[Embedding Configuration Guide](docs/EMBEDDING_CONFIGURATION_GUIDE.md)** - Optimizing embedded content
- **[Database Schema and Workflows](docs/DATABASE_SCHEMA_AND_WORKFLOWS.md)** - Database structure and workflows
- **[Architecture Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)** - Visual system architecture

## 🏗️ System Architecture

```
├── main.py                       # FastAPI application
├── config.py                     # Configuration management
├── app/
│   ├── api/                      # API endpoints
│   │   └── routers/              # Modular route handlers
│   ├── core/                     # Core functionality
│   └── db/                       # Database models
├── services/                     # Business logic
│   ├── embedding_service.py      # Embedding orchestration
│   ├── vector_service.py         # Vector database operations
│   ├── data_ingestion_service.py # Data ingestion pipeline
│   └── azure_devops_client.py    # Azure DevOps integration
└── data/                         # Persistent storage
    ├── database/                 # SQLite database
    └── chroma_db/                # Vector embeddings
```

## ⚡ Performance

- **Embedding Generation**: 2-3 seconds per 100 tickets (GPU), 10-15 seconds (CPU)
- **Search Latency**: <500ms for vector search, +0.6-1.8s with reranking
- **Scalability**: Handles 50,000+ tickets efficiently
- **Memory Usage**: ~2GB during embedding generation

## 🧪 Testing

```bash
# Run test suite
uv run pytest -q
```

## 🔧 Troubleshooting

### Common Issues

**Azure DevOps Authentication**
- Verify PAT has "Work Items (Read)" permission
- Check PAT expiration in Azure DevOps settings

**CUDA Out of Memory**
```bash
export EMBEDDING_DEVICE=cpu
```

**ChromaDB Issues**
```bash
# Rebuild vector database
rm -rf data/chroma_db
curl -X POST "http://localhost:7887/embed" -d '{"force_rebuild": true}'
```
