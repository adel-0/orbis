# Architecture Diagrams

This document contains visual diagrams to help understand the Orbis Search system architecture, data flows, and component relationships.

## Table of Contents

1. [Complete System Architecture](#complete-system-architecture)
2. [Search and Filtering Flow](#search-and-filtering-flow)
3. [Data Ingestion Flow](#data-ingestion-flow)
4. [Embedding Generation Flow](#embedding-generation-flow)
5. [Component Relationships](#component-relationships)

## Complete System Architecture

The following diagram shows the complete search and filtering architecture from data ingestion to search results:

```mermaid
graph TD
    A[User Query + Filters] --> B[Search API Endpoint]
    B --> C[Build ChromaDB WHERE Clause]
    B --> D[Generate Query Embedding]
    
    C --> E[ChromaDB Vector Search]
    D --> E
    E --> F[Fetch 10x Candidates<br/>with Filters Applied]
    
    F --> G[Reranking Service]
    G --> H[Return Top-K Results]
    
    I[Azure DevOps] --> J[Data Ingestion Service]
    J --> K[Work Items Database<br/>SQLite]
    
    K --> L[Field Discovery Service]
    L --> M[Analyze Available Fields]
    M --> N[Suggest Embedding Config]
    
    K --> O[Embedding Service]
    O --> P[Load Embedding Configs]
    P --> Q[Concatenate Text Fields]
    Q --> R[Generate Embeddings]
    R --> S[ChromaDB Vector Storage]
    
    S --> E
    
    T[Data Source Config] --> K
    T --> P
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style I fill:#fff3e0
    style S fill:#f3e5f5
    style K fill:#e8f5e8
```

## Search and Filtering Flow

### Two-Tier Search Process

```mermaid
graph LR
    A[Search Request] --> B{Filters Applied?}
    B -->|Yes| C[Build WHERE Clause]
    B -->|No| D[No Filter Constraint]
    
    C --> E[ChromaDB Query with Filters]
    D --> E
    
    A --> F[Generate Query Embedding]
    F --> E
    
    E --> G[Vector Similarity Search]
    G --> H[Fetch 10x Candidates]
    H --> I[Rerank Results]
    I --> J[Return Top-K]
    
    style A fill:#e3f2fd
    style J fill:#e8f5e8
    style E fill:#fff3e0
```

### Filter Types and Processing

```mermaid
graph TD
    A[Search Filters] --> B[Source Names]
    A --> C[Organizations]  
    A --> D[Projects]
    A --> E[Area Path]
    A --> F[Iteration Path]
    
    B --> G[Exact Match or IN]
    C --> G
    D --> G
    
    E --> H[Prefix or Contains Regex]
    F --> H
    
    G --> I[Combine with AND Logic]
    H --> I
    
    I --> J[ChromaDB WHERE Clause]
    
    style A fill:#e1f5fe
    style J fill:#c8e6c9
```

## Data Ingestion Flow

### Azure DevOps to Database

```mermaid
sequenceDiagram
    participant AzDO as Azure DevOps
    participant Sched as Scheduler
    participant Ingest as Data Ingestion Service
    participant DB as SQLite Database
    
    Sched->>Ingest: Trigger Sync
    Ingest->>DB: Check Last Sync Time
    DB-->>Ingest: Timestamp
    
    Ingest->>AzDO: Query Changes Since Last Sync
    AzDO-->>Ingest: Work Items + Comments
    
    loop For Each Work Item
        Ingest->>Ingest: Parse Standard Fields
        Ingest->>Ingest: Extract Additional Fields
        Ingest->>DB: Upsert Work Item
    end
    
    Ingest->>DB: Update Last Sync Time
```

### Work Item Processing

```mermaid
graph TD
    A[Raw Azure DevOps Work Item] --> B[Parse Standard Fields]
    A --> C[Extract Additional Fields]
    A --> D[Collect Comments]
    
    B --> E[Store in Dedicated Columns]
    C --> F[Store in additional_fields JSON]
    D --> G[Store in comments JSON Array]
    
    E --> H[Work Item Record]
    F --> H
    G --> H
    
    H --> I[Database Storage]
    
    style A fill:#fff3e0
    style H fill:#e8f5e8
    style I fill:#e1f5fe
```

## Embedding Generation Flow

### Configuration-Driven Text Concatenation

```mermaid
graph TD
    A[Work Item] --> B[Always Include Title]
    A --> C[Always Include Description]
    A --> D[Always Include Comments]
    
    A --> E[Check Embedding Config]
    E --> F{Config Enabled?}
    F -->|Yes| G[Load Configured Fields]
    F -->|No| H[Skip Additional Fields]
    
    G --> I[Extract Text from Additional Fields]
    I --> J[Add to Concatenation]
    
    B --> K[Concatenate All Text]
    C --> K
    D --> K
    J --> K
    H --> K
    
    K --> L[Generate Embedding]
    L --> M[Store in ChromaDB]
    
    style A fill:#e3f2fd
    style M fill:#f3e5f5
```

### Embedding Storage Structure

```mermaid
graph LR
    A[Work Item Data] --> B[Text Concatenation]
    B --> C[Embedding Vector]
    
    A --> D[Filterable Metadata]
    
    C --> E[ChromaDB Document]
    D --> E
    B --> E
    
    E --> F[Vector: [0.1, 0.2, ...]]
    E --> G[Metadata: source, org, project]
    E --> H[Document: concatenated text]
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#fff3e0
    style G fill:#e8f5e8
    style H fill:#fce4ec
```

## Component Relationships

### Service Dependencies

```mermaid
graph TD
    A[Web API] --> B[Vector Service]
    A --> C[Embedding Service]
    A --> D[Work Item Service]
    A --> E[Data Source Service]
    A --> F[Field Discovery Service]
    A --> G[Rerank Service]
    
    B --> H[ChromaDB]
    C --> I[Embedding Providers]
    D --> J[SQLite Database]
    E --> J
    F --> D
    
    I --> K[Local Sentence Transformers]
    I --> L[Azure OpenAI]
    
    style A fill:#e3f2fd
    style H fill:#f3e5f5
    style J fill:#e8f5e8
```

### Data Flow Between Components

```mermaid
graph LR
    A[(SQLite Database)] --> B[Work Item Service]
    B --> C[Field Discovery Service]
    B --> D[Embedding Service]
    
    C --> E[Configuration Suggestions]
    E --> F[Data Source Service]
    F --> A
    
    D --> G[Vector Service]
    G --> H[(ChromaDB)]
    
    H --> I[Search Service]
    I --> J[Rerank Service]
    J --> K[Search Results]
    
    style A fill:#e8f5e8
    style H fill:#f3e5f5
    style K fill:#c8e6c9
```

## Filter Processing Details

### WHERE Clause Building Logic

```mermaid
graph TD
    A[Search Request] --> B{Has source_names?}
    B -->|Yes| C[Add source_name filter]
    B -->|No| D{Has organizations?}
    
    C --> D
    D -->|Yes| E[Add organization filter]
    D -->|No| F{Has projects?}
    
    E --> F
    F -->|Yes| G[Add project filter]
    F -->|No| H{Has area_path filters?}
    
    G --> H
    H -->|Yes| I[Add area_path regex]
    H -->|No| J{Has iteration_path filters?}
    
    I --> J
    J -->|Yes| K[Add iteration_path regex]
    J -->|No| L{Multiple filters?}
    
    K --> L
    L -->|Yes| M[Combine with $and]
    L -->|No| N[Single filter]
    
    M --> O[Final WHERE Clause]
    N --> O
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
```

### Search Performance Flow

```mermaid
graph LR
    A[User Query] --> B[Apply Metadata Filters]
    B --> C[Reduced Search Space]
    C --> D[Vector Similarity Search]
    D --> E[Top Candidates]
    E --> F[Reranking]
    F --> G[Optimized Results]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style G fill:#c8e6c9
```

---

These diagrams provide visual representations of the key architectural concepts and workflows in the Orbis Search system. For detailed implementation information, refer to:

- [Search and Filtering Architecture](./SEARCH_AND_FILTERING_ARCHITECTURE.md)
- [Embedding Configuration Guide](./EMBEDDING_CONFIGURATION_GUIDE.md)
- [Database Schema and Workflows](./DATABASE_SCHEMA_AND_WORKFLOWS.md)
