# Orbis

<div align="center">
  <img src="https://github.com/user-attachments/assets/cb2da20e-761d-47b5-88f9-3ece0c8db7e0" alt="orbis_black" width="500" height="500" />
</div>

A modular search platform for Azure DevOps work items and wiki content, built as a monorepo workspace with shared infrastructure and specialized applications.

## Architecture

### Applications

**Orbis** - Agentic RAG system with multi-agent workflows for contextual analysis. Uses agents for project detection, scope analysis, and documentation synthesis with semantic search and cross-collection reranking.

**Orbis Search** - Hybrid search engine combining BM25 keyword search with semantic embeddings. Uses RRF and recency boosting for fast search without agentic overhead.

**Orbis Core** - Shared library providing [Azure DevOps](https://azure.microsoft.com/en-us/products/devops) connectors, embedding services, search components, LLM clients, database management, and utilities.

### Feature Comparison

| Feature | Orbis | Orbis Search |
|---------|-------|--------------|
| **Search Type** | Agentic RAG | Hybrid Search |
| **Agents** | Multi-agent workflows | None |
| **Vector Store** | ChromaDB | ChromaDB |
| **Keyword Search** | No | BM25S |
| **Semantic Search** | Yes | Yes |
| **Reranking** | Cross-collection | Optional |
| **Result Fusion** | Cross-collection merge | RRF |
| **Recency Boosting** | No | Yes |
| **Response Type** | Synthesized answer | Ranked results |
| **Use Case** | Complex analysis | Fast lookup |

### Technology Stack

- [Python 3.13+](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/) for REST APIs
- [SQLite](https://www.sqlite.org/) with [SQLAlchemy](https://www.sqlalchemy.org/) ORM
- [Sentence Transformers](https://www.sbert.net/) for local embeddings
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for LLM operations
- [ChromaDB](https://www.trychroma.com/) (Orbis) and [BM25S](https://github.com/xhluca/bm25s) (Orbis Search) for vector operations
- [UV](https://docs.astral.sh/uv/) for dependency management

## Running Applications

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

## How It Works

**Agentic RAG Flow (Orbis)**:
1. Project Detection agent identifies relevant projects from ticket metadata
2. Scope Analyzer agent determines intent and extracts key context
3. Multi-source semantic search retrieves work items and wiki content
4. Cross-collection reranking prioritizes most relevant results
5. Documentation Aggregator agent synthesizes response with citations

**Hybrid Search Flow (Orbis Search)**:
1. BM25 keyword search identifies lexically relevant work items
2. Semantic embedding search finds conceptually similar items
3. Reciprocal Rank Fusion (RRF) combines both result sets
4. Recency boosting applies time-based score adjustments
5. Optional cross-encoder reranking for final relevance scoring
