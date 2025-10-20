# Orbis

<img width="781" height="728" alt="orbis" src="https://github.com/user-attachments/assets/6db8f4c3-5c26-4c75-a583-d8ead46751dc" />

A modular search platform for Azure DevOps work items and wiki content, built as a monorepo workspace with shared infrastructure and specialized applications.

## Architecture

### Applications

**Orbis** - Agentic RAG system with multi-agent workflows for contextual analysis. Uses agents for project detection, scope analysis, and documentation synthesis with semantic search and cross-collection reranking.

**Orbis Search** - Hybrid search engine combining BM25 keyword search with semantic embeddings. Uses RRF and recency boosting for fast search without agentic overhead.

**Orbis Core** - Shared library providing Azure DevOps connectors, embedding services, search components, LLM clients, database management, and utilities.

### Technology Stack

- Python 3.13+
- FastAPI for REST APIs
- SQLite with SQLAlchemy ORM
- Sentence Transformers for local embeddings
- Azure OpenAI for LLM operations
- ChromaDB (Orbis) and BM25S (Orbis Search) for vector operations
- UV for dependency management

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
