# Orbis

An agentic RAG system that provides intelligent analysis of Azure DevOps tickets using advanced semantic search and AI-powered summarization. Built with FastAPI, ChromaDB, and Azure OpenAI for enterprise-grade performance.

## 🚀 Quick Start

### Standard Installation

```bash
# Clone repository
git clone https://github.com/adel-0/orbis
cd orbis

# Install dependencies using uv
uv sync --extra local  # Local development with embedding models
# uv sync                # Azure-only minimal dependencies

# Configure environment
cp .env .env  # Copy from existing .env file
# Edit .env with your Azure DevOps and OpenAI credentials

# Start the API server
uv run main.py
```

The server runs on http://localhost:7887 with interactive documentation at `/docs`.

### Docker Deployment

```bash
# Quick setup (Linux/macOS)
chmod +x docker-setup.sh
./docker-setup.sh

# Or manual setup
cp .env .env  # Copy from existing .env file
docker-compose up -d

# Verify deployment
curl http://localhost:7887/health
```

See [Docker section](#-docker-deployment) for detailed configuration.

## 🏗️ System Architecture

The system is built around a modular agentic RAG architecture with clear separation of concerns:

```
├── main.py                          # FastAPI application entry point
├── config/                          # Configuration management
├── app/
│   ├── api/
│   │   ├── dependencies.py          # Dependency injection
│   │   └── routers/                 # REST API endpoints
│   │       ├── analyze.py           # /analyze (main agentic RAG)
│   │       ├── embedding.py         # /embed
│   │       ├── ingestion.py         # /ingest
│   │       ├── health.py            # /health
│   │       └── scheduler.py         # /scheduler/*
│   └── db/
│       ├── models.py                # Database models
│       └── session.py               # Database session management
├── core/
│   ├── services/                    # Core business logic
│   │   ├── agentic_rag_orchestrator.py      # Main orchestration engine
│   │   ├── generic_data_ingestion.py        # Multi-source data ingestion
│   │   ├── wiki_summarization.py            # Wiki content processing
│   │   └── project_detection.py             # AI-powered project detection
│   └── agents/                      # Specialized AI agents
│       ├── ticket_scope_analyzer.py         # Intent and scope analysis
│       └── documentation_aggregator.py      # Response synthesis
├── infrastructure/                  # External integrations
│   ├── connectors/                  # Data source connectors
│   ├── storage/                     # Vector and embedding services
│   └── ai_providers/                # OpenAI integrations
└── utils/                           # Shared utilities
```

See `docs/AGENTIC_RAG_SYSTEM_DIAGRAM.md` for detailed architecture diagrams.

## ✨ Core Features

### Agentic RAG System
- **Project Detection**: Intelligent project identification from ticket metadata
- **Scope Analysis**: AI-powered intent analysis with contextual understanding
- **Multi-Agent Orchestration**: Specialized agents for different analysis tasks
- **Documentation Synthesis**: Automated generation of comprehensive responses

### Data Ingestion & Processing
- **Generic Connector System**: Modular architecture supporting multiple data source types
- **Azure DevOps Integration**: Work items, wikis, and project data ingestion
- **Parallel Processing**: Concurrent data ingestion with intelligent rate limiting
- **Configuration-Driven**: Add new data sources via simple configuration files
- **Incremental Synchronization**: Efficient updates with source-specific sync states

### Advanced Search Capabilities
- **Multi-Collection Vector Storage**: Type-specific collections for optimal search
- **Hybrid Embedding Support**: Local sentence-transformers or Azure OpenAI embeddings  
- **Cross-Collection Reranking**: Intelligent result ranking across content types
- **Semantic Similarity Search**: Sub-second response times with ChromaDB
- **Context-Aware Filtering**: Project and metadata-based result filtering

### Enterprise Features
- **Multi-Source Configuration**: Support for multiple organizations and projects
- **Automated Scheduling**: Background synchronization with configurable intervals
- **Health Monitoring**: Comprehensive system diagnostics and status reporting
- **WebSocket Progress**: Real-time updates for long-running operations
- **Security**: API key authentication and secure credential management

## 📚 Documentation

### System Architecture
- **[Agentic RAG System Diagram](docs/AGENTIC_RAG_SYSTEM_DIAGRAM.md)** - Complete system architecture and data flow
- **[Agentic RAG Implementation](docs/AGENTIC_RAG_IMPLEMENTATION.md)** - Detailed implementation guide
- **[Database Schema and Workflows](docs/DATABASE_SCHEMA_AND_WORKFLOWS.md)** - Database structure and workflows

### Configuration and Integration
- **[Data Source Configuration Guide](docs/DATASOURCE_CONFIGURATION_GUIDE.md)** - Setting up data sources and connectors
- **[Connector Interface Standard](docs/CONNECTOR_INTERFACE_STANDARD.md)** - Guide for implementing new connectors
- **[Features Overview](docs/FEATURES.md)** - Complete feature list and capabilities

## ⚙️ Configuration

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

## 🔧 API Reference

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

## 🐳 Docker Deployment

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

## 🚀 Usage Examples

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

## 📊 Performance Characteristics

### Agentic RAG Response Times
- **Project Detection**: ~200-500ms per analysis
- **Scope Analysis**: ~1-2 seconds with wiki context
- **Multi-Agent Processing**: ~2-4 seconds total analysis time
- **Response Generation**: Sub-5 second end-to-end processing

### Data Ingestion Performance
- **Parallel Processing**: Concurrent ingestion across all configured sources
- **API Rate Limiting**: Intelligent throttling with exponential backoff
- **Incremental Updates**: Source-specific sync states for efficiency
- **Batch Processing**: Configurable batch sizes (default: 200 items)

### Search and Embedding Performance
- **Vector Search**: <500ms query response time
- **Embedding Generation**: 2-3 seconds per 100 items (local), 1-2 seconds (Azure)
- **Cross-Collection Search**: Multi-collection queries with result reranking
- **Memory Usage**: ~2GB RAM during embedding operations

### Scalability
- **Data Volume**: Handles 50,000+ documents across multiple sources
- **Concurrent Users**: Supports 10+ simultaneous API requests
- **Storage Requirements**: ~50MB per 1,000 documents including vectors

## 🔧 Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 4GB+ system memory
- **Storage**: 5GB+ free space for models and vector database
- **Network**: Internet access for Azure OpenAI integration (optional)

### Recommended Requirements
- **CPU**: 8 cores, 3.0GHz+
- **GPU**: NVIDIA GTX 1060 or better (8GB+ VRAM recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection for cloud services

### Software Dependencies
- **Python**: 3.13+ (required by pyproject.toml)
- **uv**: Fast Python package installer and dependency manager
- **CUDA**: 12.9+ for GPU acceleration (optional)
- **Docker**: For containerized deployment

## 🔒 Security & Compliance

### Authentication
- **API Key Protection**: Optional API key authentication for all endpoints
- **PAT Security**: Secure storage with optional encryption at rest
- **Token Isolation**: Separate PATs for different organizations

### Data Privacy
- **Local Processing**: Models and data stored locally by default
- **No External Dependencies**: Works without cloud services (Azure OpenAI optional)
- **Source Attribution**: Clear identification of data origins
- **Audit Trail**: Detailed logging of all data operations

### Network Security
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Rate Limiting**: Built-in throttling for external API calls
- **Reverse Proxy Ready**: Compatible with nginx and other proxies

## 🛠️ Troubleshooting

### Common Issues

**Azure DevOps Authentication Errors**
```bash
# Verify PAT permissions and expiration
curl -u ":your-pat-token" \
  "https://dev.azure.com/your-org/_apis/wit/wiql/your-query-id?api-version=6.1-preview.2"
```

**Data Ingestion Failures**
```bash

# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

**CUDA Out of Memory**
```bash
# Force CPU-only mode
export EMBEDDING_DEVICE=cpu
```

**ChromaDB Corruption**
```bash
# Remove and regenerate database
rm -rf data/chroma_db
curl -X POST "http://localhost:7887/embed" \
  -d '{"force_rebuild": true}'
```

### Performance Tuning

**For High-Performance Systems:**
- `EMBEDDING_BATCH_SIZE=32`
- `EMBEDDING_DEVICE=cuda`
- `ADO_MAX_CONCURRENT_REQUESTS=100`

**For Resource-Constrained Systems:**
- `EMBEDDING_BATCH_SIZE=8`
- `EMBEDDING_DEVICE=cpu`
- `ADO_MAX_CONCURRENT_REQUESTS=25`

## 🧪 Testing

Run tests using the configured test environment:

```bash
# Install dependencies
uv sync --extra dev

# Run tests with pytest
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run specific test files
uv run pytest test_agentic_rag.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
- Check the [troubleshooting section](#-troubleshooting)
- Review logs: `docker-compose logs api` (Docker) or check console output (local)
- Verify configuration and environment variables
- Check health endpoint: `curl http://localhost:7887/health`

---

**Note**: Protect endpoints by setting `API_KEY_ENABLED=true` and providing `X-API-Key` in your requests for production deployments.