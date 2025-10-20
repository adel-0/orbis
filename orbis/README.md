# Orbis

An agentic RAG system that provides intelligent analysis of Azure DevOps tickets using multi-agent workflows, semantic search, and AI-powered synthesis. Built with FastAPI, ChromaDB, and Azure OpenAI for enterprise-grade performance.

See the [main README](../README.md) for monorepo architecture and workspace setup instructions.

## Quick Start

### Standard Installation

```bash
# From monorepo root, install all dependencies
cd orbis
uv sync

# Configure environment in orbis/ directory
# Create .env file with your credentials

# Start the API server
cd orbis
uv run main.py
```

The server runs on http://localhost:7887 with interactive documentation at `/docs`.

### Docker Deployment

```bash
# From orbis/ directory
cp .env.example .env  # Configure your credentials
docker-compose up -d

# Verify deployment
curl http://localhost:7887/health
```

See [Docker Deployment section](#docker-deployment) for detailed configuration.

## Documentation

### System Architecture
- **[Agentic RAG System Diagram](docs/AGENTIC_RAG_SYSTEM_DIAGRAM.md)** - Complete system architecture and data flow
- **[Agentic RAG Implementation](docs/AGENTIC_RAG_IMPLEMENTATION.md)** - Detailed implementation guide
- **[Database Schema and Workflows](docs/DATABASE_SCHEMA_AND_WORKFLOWS.md)** - Database structure and workflows

### Configuration and Integration
- **[Data Source Configuration Guide](docs/DATASOURCE_CONFIGURATION_GUIDE.md)** - Setting up data sources and connectors
- **[Connector Interface Standard](docs/CONNECTOR_INTERFACE_STANDARD.md)** - Guide for implementing new connectors
- **[Features Overview](docs/FEATURES.md)** - Complete feature list and capabilities

## Configuration

### Environment Variables

Configure your environment variables for the system to function properly:

```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=7887

# Azure OpenAI (chat/summarization)
AZURE_OPENAI_ENDPOINT=https://resource.openai.azure.com/
AZURE_OPENAI_API_KEY=api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini

# Azure OpenAI Embeddings
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_API_KEY=embedding-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large

# Embedding Provider Configuration
EMBEDDING_PROVIDER=azure   # or "local"
EMBEDDING_DEVICE=cpu       # or "cuda"
EMBEDDING_BATCH_SIZE=32
LOCAL_EMBEDDING_MODEL=google/embeddinggemma-300m

# Database Configuration
DATABASE_URL=sqlite:///data/database/orbis.db

# Security
API_KEY_ENABLED=false
API_KEY=your-api-key-here

# Performance Tuning
ADO_MAX_CONCURRENT_REQUESTS=50
ADO_BATCH_SIZE=200
ADO_TIMEOUT_SECONDS=300
```

### Data Source Configuration

Data sources are configured using YAML configuration files in the `config/instances/` directory. Each file defines connection details, queries, and processing options for a data source.

Example configuration structure:
```yaml
# config/instances/example-azdo-workitems.yaml
name: "project-tickets"
source_type: "azdo_workitems"
enabled: true
organization: "your-org"
project: "Your Project"
pat: "${AZURE_DEVOPS_PAT}"
wiql_queries:
  - "SELECT [System.Id] FROM WorkItems WHERE [System.State] = 'Active'"
```

See `docs/DATASOURCE_CONFIGURATION_GUIDE.md` for complete configuration details.

## API Reference

### Core Endpoints

#### Agentic RAG Analysis
```bash
POST /analyze
{
  "ticket_content": "database connection timeout in production environment",
  "include_context": true
}
```

#### Data Ingestion
```bash
POST /ingest
{
  "force_full_sync": false,
  "source_names": ["source1", "source2"]  # Optional
}
```

#### Embedding Generation
```bash
POST /embed
{
  "force_rebuild": false
}
```

#### System Health
```bash
GET /health
```

#### Scheduling Management
```bash
GET /scheduler/status
POST /scheduler/start
POST /scheduler/stop
```

Interactive API documentation is available at `http://localhost:7887/docs`.

## Docker Deployment

### Prerequisites

- **Operating System**: Linux, Windows with WSL2, or macOS
- **Memory**: 8GB+ RAM recommended
- **Storage**: 20GB+ free disk space
- **Docker**: Docker Desktop or Docker Engine with Docker Compose
- **GPU** (Optional): NVIDIA GPU with CUDA support
  - 12GB+ VRAM recommended for optimal performance
  - 8GB VRAM minimum for basic operation
  - nvidia-docker2 for GPU acceleration

### Quick Docker Setup

**Automated Setup (Linux/macOS):**
```bash
chmod +x docker-setup.sh
./docker-setup.sh
```

**Manual Setup:**
```bash
# Clone and prepare repository
git clone <your-repository-url>
cd orbis
mkdir -p data/database data/chroma_db

# Environment configuration
cp .env .env  # Copy from existing .env file
# Edit .env with your specific values

# Build and run
docker-compose build
docker-compose up -d

# Verify deployment
curl http://localhost:7887/health
```

### Docker Configuration

Key environment variables for Docker deployment:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_PORT` | Host port for API | 7887 |
| `EMBEDDING_PROVIDER` | local/azure | local |
| `EMBEDDING_DEVICE` | cuda/cpu | cpu |
| `LOCAL_EMBEDDING_MODEL` | Embedding model | google/embeddinggemma-300m (or intfloat/multilingual-e5-large for low VRAM) |
| `EMBEDDING_BATCH_SIZE` | Processing batch size | 32 |

### Docker Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Monitor resources
docker stats orbis-api-1

# GPU monitoring (if available)
docker exec -it orbis-api-1 nvidia-smi

# Backup data
tar -czf backup-$(date +%Y%m%d).tar.gz data/
```

### Docker Troubleshooting

**GPU Not Available:**
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**Port Already in Use:**
```bash
# Check what's using port 7887
netstat -tulpn | grep :7887
# Change API_PORT in .env if needed
```

**Memory Issues:**
```bash
# Check container memory usage
docker stats orbis-api-1
# Reduce EMBEDDING_BATCH_SIZE in .env
```

## Usage Examples

### 1. System Setup

```bash
# Start the API server
uv run main.py

# Server starts on http://localhost:7887
# Interactive docs: http://localhost:7887/docs
```

### 2. Data Source Configuration

Configure data sources by creating YAML files in `config/instances/`:

```bash
# Create a work items configuration
# config/instances/my-project-workitems.yaml
name: "my-project-tickets"
source_type: "azdo_workitems"
enabled: true
organization: "my-org"
project: "My Project"
pat: "${AZURE_DEVOPS_PAT}"
```

### 3. Data Ingestion and Processing

```bash
# Trigger data ingestion for all enabled sources
curl -X POST "http://localhost:7887/ingest"

# Generate embeddings for semantic search
curl -X POST "http://localhost:7887/embed"
```

### 4. Agentic RAG Analysis

```bash
# Analyze a ticket with intelligent context
curl -X POST "http://localhost:7887/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_content": "User authentication failing in production environment",
    "include_context": true
  }'
```

### 5. System Monitoring

```bash
# Check system health
curl -X GET "http://localhost:7887/health"

# Monitor scheduler status
curl -X GET "http://localhost:7887/scheduler/status"
```

## Hardware Requirements

- **Python**: 3.13+, uv package manager
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **GPU**: NVIDIA GTX 1060+ (8GB VRAM) for local embeddings (optional)
- **Storage**: 20GB+ free space