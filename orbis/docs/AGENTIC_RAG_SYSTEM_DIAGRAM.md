# Agentic RAG System Architecture

Presents the actual implementation as of the current codebase state: multi-collection vector storage with ChromaDB (23,633 work items + 125 wiki pages), configuration-driven architecture with generic services, cross-collection search with reranking, and hybrid SQLite + ChromaDB storage with wiki summary caching.

## 1) Data Ingestion & Embedding Flow (Continuous)

```mermaid
graph TB
  %% Data Sources (ACTUALLY IMPLEMENTED)
  subgraph Sources[Data Sources - Active]
    ADO[Azure DevOps Work Items<br/>✅ 23,633 documents indexed]
    WIKI[Azure DevOps Wiki<br/>✅ 125 pages indexed]
  end

  %% Future Sources (NOT YET IMPLEMENTED)
  subgraph FutureSources[Data Sources - Future]
    DOCS[Document Sources PDFs etc<br/>⚠️ Not implemented]
    GIT[Git/Code Sources<br/>⚠️ Not implemented]
  end

  %% Integrators with actual implementations
  subgraph Integrators[Configuration-Driven Connectors]
    ADO_INT[WorkItemService<br/>✅ azure_devops.work_item_service<br/>Collection: workitems_collection]
    WIKI_INT[WikiService<br/>✅ azure_devops.wiki_service<br/>Collection: wiki_collection]
  end



  %% Generic Ingestion Service (ACTUAL IMPLEMENTATION)
  subgraph IngestGeneric[GenericDataIngestionService - Universal Pipeline]
    direction TB
    
    subgraph PROCESS[Universal Processing Pipeline]
      CONFIG[Load Data Source Config<br/>✅ data_sources.py]
      CONNECTOR[Dynamic Connector Loading<br/>✅ Reflection-based]
      CONTENT[Content Extraction & Normalization<br/>✅ Generic content model]
      EMBED[Dual Embedding Generation<br/>✅ Azure OpenAI + Local options]
      STORE[Multi-Collection Storage<br/>✅ ChromaDB + SQLite]
    end
  end

  %% Hybrid Storage Architecture (ACTUAL IMPLEMENTATION)
  subgraph Storage[Hybrid Storage - SQLite + ChromaDB]
    subgraph VectorDB[ChromaDB Vector Collections]
      WI_COLL[(workitems_collection<br/>✅ 23,633 documents<br/>Generic content schema)]
      WIKI_COLL[(wiki_collection<br/>✅ 125 documents<br/>Generic content schema)]
    end
    
    subgraph RelationalDB[SQLite Relational Storage]
      CONTENT_TBL[(content table<br/>✅ Universal content model)]
      DATASRC_TBL[(data_source table<br/>✅ JSON configuration)]
      CACHE_TBL[(wiki_summary_cache<br/>✅ Persistent cache)]
      INGESTION_TBL[(ingestion_log<br/>✅ Process tracking)]
    end
  end

  %% Caching Implementation (ACTUAL)
  subgraph Cache[Persistent Caching]
    WIKI_CACHE[(WikiSummaryCache<br/>✅ SQLite-based persistent)]
    EMBED_CACHE[EmbeddingService.clear_cache()<br/>✅ Provider-level caching]
  end

  %% Data Flow Connections (ACTUAL IMPLEMENTATION)
  ADO --> ADO_INT
  WIKI --> WIKI_INT

  ADO_INT -->|JSON Config| CONFIG
  WIKI_INT -->|JSON Config| CONFIG
  
  CONFIG --> CONNECTOR
  CONNECTOR --> CONTENT
  CONTENT --> EMBED
  EMBED --> STORE

  STORE --> WI_COLL
  STORE --> WIKI_COLL
  STORE --> CONTENT_TBL
  STORE --> DATASRC_TBL

  WIKI_INT -->|Summaries| WIKI_CACHE
  EMBED -->|Cached embeddings| EMBED_CACHE

  %% Styling
  style WI_COLL fill:#e8f5e9,stroke:#4caf50
  style WIKI_COLL fill:#fce4ec,stroke:#e91e63
  style CONTENT_TBL fill:#f3e5f5,stroke:#9c27b0

  style WIKI_CACHE fill:#e0f2f1,stroke:#009688
  style EMBED_CACHE fill:#e0f2f1,stroke:#009688
```

**Key Implementation Achievements:**

Configuration-driven architecture reduced new data source integration to 1 connector + 1 config entry. Generic services with universal content model eliminate hardcoded type assumptions. Multi-collection storage successfully indexed 23,633 work items + 125 wiki pages. Hybrid storage combines ChromaDB vectors + SQLite relational data for optimal performance. Persistent caching stores wiki summaries to reduce LLM API calls.

## 2) Runtime Orchestration (ACTUAL CURRENT IMPLEMENTATION)

```mermaid
sequenceDiagram
  autonumber
  participant API as FastAPI Application
  participant Orc as AgenticRAGOrchestrator
  participant PD as ProjectDetectionService
  participant Cache as WikiSummaryCache
  participant SA as ScopeAnalyzer
  participant Search as GenericMultiModalSearch
  participant Vec as GenericVectorService
  participant Rank as CrossCollectionReranker
  participant DA as DocumentationAggregator

  API->>Orc: POST /process with content data
  Note right of API: Input: content, area_path, project_info

  %% Project detection (ACTUAL - pattern-based, non-LLM)
  Orc->>PD: detect_project_from_area_path()
  PD-->>Orc: ProjectInfo with project_code, confidence

  %% Wiki context retrieval (ACTUAL - cached summaries)
  Orc->>Cache: get_cached_wiki_summaries(project_code)
  Cache-->>Orc: Cached project wiki summaries

  %% Scope analysis (ACTUAL - with wiki context)
  Orc->>SA: analyze_scope(content, wiki_context)
  Note right of SA: Uses LLM service internally<br/>with cached wiki summaries
  SA-->>Orc: ScopeAnalysis with search intent

  %% Multi-modal search (ACTUAL - configuration-driven)
  Orc->>Search: search_across_collections(query, collections, filters)

  par Parallel search across active collections
    Search->>Vec: search(workitems_collection, query, filters)
    Note right of Vec: Uses embedding service<br/>for vector similarity
    Search->>Vec: search(wiki_collection, query, filters)
  end

  Vec-->>Search: Results with confidence scores and metadata

  %% Cross-collection reranking (ACTUAL)
  Search->>Rank: rerank_cross_collection_results(results)
  Note right of Rank: Implemented reranking<br/>with score normalization

  Rank-->>Search: Reranked, normalized results
  Search-->>Orc: SearchResults with confidence

  %% Documentation aggregation (ACTUAL)
  Orc->>DA: aggregate_documentation(search_results, original_content)
  Note right of DA: Uses LLM service internally<br/>for response generation
  DA-->>Orc: Final response with citations

  %% Return to API
  Orc-->>API: AgenticRAGResponse with summary
  API-->>API: JSON response with structured data
```

**Key Implementation Features:**
- ✅ **WikiSummaryCache**: Persistent SQLite cache avoids redundant LLM API calls
- ✅ **Pattern-Based Project Detection**: Fast, non-LLM area path pattern matching  
- ✅ **Generic Content Model**: Universal schema works across all content types
- ✅ **Cross-Collection Reranker**: Score normalization and source diversity implemented
- ✅ **Configuration-Driven Search**: Dynamic collection routing based on data source configs
- ✅ **Performance**: ~2-5 second response times for complete agentic RAG pipeline

## 3) Implemented Architecture Components (CURRENT STATE)

```mermaid
graph LR

  subgraph "✅ Generic Vector Service"
    GVS[GenericVectorService<br/>✅ infrastructure/storage/]
    CCM[ChromaDB Collection Manager<br/>✅ Dynamic collection creation]
    
    GVS -->|manages| CCM
    
    WIC[(workitems_collection<br/>✅ 23,633 documents)]
    WIKIC[(wiki_collection<br/>✅ 125 documents)]
    
    CCM --> WIC
    CCM --> WIKIC
  end

  subgraph "✅ Cross-Collection Search & Reranking"
    GMMS[GenericMultiModalSearch<br/>✅ services/search/]
    CCR[CrossCollectionReranker<br/>✅ Score normalization implemented]
    
    GMMS --> CCR
  end

  subgraph "✅ Configuration-Driven Architecture"
    CONFIG[data_sources.py<br/>✅ JSON configuration]
    LOADER[Dynamic Connector Loading<br/>✅ Reflection-based]
    GINGST[GenericDataIngestionService<br/>✅ Universal pipeline]
    
    CONFIG --> LOADER
    LOADER --> GINGST
  end

  subgraph "✅ Hybrid Storage Implementation"
    SQLITE[(SQLite RelationalDB<br/>✅ content, data_source,<br/>ingestion_log, wiki_cache)]
    CHROMADB[(ChromaDB VectorDB<br/>✅ Multi-collection)]
    CACHE[WikiSummaryCache<br/>✅ Persistent SQLite cache]
    
    SQLITE -->|metadata| CHROMADB
    SQLITE --> CACHE
  end

  subgraph "✅ Dual Embedding Strategy"
    ES[EmbeddingService<br/>✅ infrastructure/storage/]
    AZURE[Azure OpenAI Provider<br/>✅ text-embedding-ada-002]
    LOCAL[Local Provider<br/>✅ sentence-transformers]
    
    ES --> AZURE
    ES --> LOCAL
  end

  subgraph "⚠️ Future Components"
    DOCS[Document Connector<br/>⚠️ PDF processing not implemented]
    GIT[Git Connector<br/>⚠️ Source code not implemented]
    DIST_CACHE[Distributed Cache<br/>⚠️ TTL-based cache not implemented]
  end

  GVS --> GMMS
  CONFIG --> GINGST
  GINGST --> SQLITE
  GINGST --> CHROMADB
  ES --> CHROMADB

  style GVS fill:#e8f5e9,stroke:#4caf50
  style GMMS fill:#e3f2fd,stroke:#2196f3
  style CONFIG fill:#fff3e0,stroke:#ff9800
  style SQLITE fill:#f3e5f5,stroke:#9c27b0
  style ES fill:#e0f2f1,stroke:#009688
  style DOCS fill:#ffebee,stroke:#f44336
  style GIT fill:#ffebee,stroke:#f44336
  style DIST_CACHE fill:#ffebee,stroke:#f44336
```

**Implementation Status Summary:**
- ✅ **Core Agentic RAG Pipeline**: Fully functional with 23,633 work items + 125 wiki pages
- ✅ **Configuration-Driven Architecture**: New data sources require only 1 connector + 1 config entry
- ✅ **Generic Services**: Universal content model eliminates hardcoded assumptions
- ✅ **Multi-Collection Search**: Cross-collection search with reranking implemented
- ✅ **Hybrid Storage**: SQLite relational + ChromaDB vector for optimal performance
- ✅ **Persistent Caching**: Wiki summaries cached to optimize LLM API usage
- ⚠️ **Future Extensions**: PDF documents and Git source code integration planned
