# Orbis Search

A lightweight hybrid search system for Azure DevOps work items, combining BM25 keyword search with semantic embeddings. Uses reciprocal rank fusion (RRF) and recency boosting for fast, accurate search without the overhead of agentic processing.

See the [main README](../README.md) for monorepo architecture and workspace setup instructions.

## Quick Start

### Prerequisites
- **Python**: 3.13+
- **uv**: Package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Hardware**: 4GB+ RAM, GPU recommended (NVIDIA GTX 1060 or better)
- **Azure DevOps**: Personal Access Token with Work Items (Read) permission

### Installation

```bash
# From monorepo root, install all dependencies
cd orbis
uv sync

# Configure environment in orbis-search/ directory
# Create .env file with your credentials

# Start the server
cd orbis-search
uv run main.py
```

Access the API at `http://localhost:7887` and interactive docs at `http://localhost:7887/docs`

## API Usage

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

For the complete hybrid search flow explanation, see the [main README](../README.md#how-it-works).
