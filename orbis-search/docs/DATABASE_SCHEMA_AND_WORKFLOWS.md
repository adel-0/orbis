# Database Schema and Workflows

This document provides a comprehensive overview of the database schema, data relationships, and key workflows in the Orbis Search system.

## Table of Contents

1. [Database Schema Overview](#database-schema-overview)
2. [Core Entities](#core-entities)
3. [Data Relationships](#data-relationships)
4. [Key Workflows](#key-workflows)
5. [Data Flow Diagrams](#data-flow-diagrams)

## Database Schema Overview

Orbis Search uses SQLite for relational data storage and ChromaDB for vector embeddings. The system maintains data across two primary storage systems:

### SQLite Database (Relational Data)
- **Location**: `data/database/orbis_search.db`
- **Purpose**: Stores structured data, configurations, and metadata
- **Tables**: DataSources, WorkItems, and related entities

### ChromaDB (Vector Storage)
- **Location**: `data/chroma_db/`
- **Purpose**: Stores embeddings and metadata for semantic search
- **Collections**: Work item embeddings with searchable metadata

## Core Entities

### 1. DataSource

**Purpose**: Represents external data sources (Azure DevOps organizations/projects)

```python
class DataSourceModel(Base):
    __tablename__ = "data_sources"
    
    id: int                           # Primary key
    name: str                         # Unique data source name
    organization: str                 # Azure DevOps organization
    project: str                      # Azure DevOps project
    personal_access_token: str        # Encrypted PAT
    enabled: bool = True              # Whether source is active
    embedding_field_config: dict      # JSON: embedding configuration
    last_sync: datetime              # Last successful sync timestamp
    created_at: datetime             # Creation timestamp
    updated_at: datetime             # Last update timestamp
```

**Key Features**:
- **Encrypted Credentials**: PAT stored using Fernet encryption
- **Embedding Configuration**: Per-source field selection for embeddings
- **Sync Tracking**: Monitors last successful synchronization

### 2. WorkItem

**Purpose**: Represents individual work items from Azure DevOps

```python
class WorkItemModel(Base):
    __tablename__ = "work_items"
    
    id: str                          # Work item ID from Azure DevOps
    data_source_id: int              # Foreign key to DataSource
    title: str                       # Work item title
    description: str                 # Work item description
    work_item_type: str              # Bug, Task, Story, etc.
    state: str                       # Active, Closed, etc.
    priority: str                    # Priority level
    severity: str                    # Severity level
    area_path: str                   # Team/area hierarchy
    iteration_path: str              # Sprint/iteration hierarchy
    assigned_to: str                 # Assigned person
    created_by: str                  # Creator
    tags: List[str]                  # Work item tags
    comments: List[str]              # All comments
    additional_fields: dict          # JSON: custom/additional fields
    created_at: datetime             # Creation in Azure DevOps
    updated_at: datetime             # Last update in Azure DevOps
    synced_at: datetime              # Last sync to our system
    
    # Relationships
    data_source: DataSourceModel     # Related data source
```

**Key Features**:
- **Standard Fields**: Core Azure DevOps fields for filtering and display
- **Dynamic Fields**: Additional custom fields stored as JSON
- **Comments Integration**: All comments stored as searchable list
- **Sync Tracking**: Tracks when work item was last synchronized

## Data Relationships

### Entity Relationship Diagram

```mermaid
erDiagram
    DataSource ||--o{ WorkItem : "has many"
    WorkItem ||--|| VectorEmbedding : "has one"
    VectorEmbedding }o--|| ChromaDB : "stored in"

    DataSource {
        int id PK
        string name
        string organization
        string project
        string personal_access_token
        boolean enabled
        json embedding_field_config
        datetime last_sync
    }

    WorkItem {
        string id PK
        int data_source_id FK
        string title
        string description
        string work_item_type
        string state
        json additional_fields
    }

    VectorEmbedding {
        string id
        vector embedding_vector
        json metadata
        string concatenated_text
    }

    ChromaDB {
        string collection_name
        string storage_type
    }
```

### Relationship Details

**DataSource → WorkItem (1:Many)**
- One data source can have many work items
- Work items belong to exactly one data source
- Cascade delete: removing data source removes all its work items

**WorkItem → ChromaDB (1:1)**
- Each work item has one corresponding embedding
- Embedding metadata references work item attributes
- Embeddings are recreated when work items change

## Key Workflows

### 1. Data Source Registration Workflow

```mermaid
graph LR
    A[User Creates<br/>Data Source] --> B[Encrypt & Store<br/>Credentials]
    B --> C[Test Connection<br/>& Validate]
    C --> D[Store in DB<br/>as DataSource]
    D --> E[Set enabled=true<br/>if successful]

    style A fill:#e3f2fd
    style E fill:#c8e6c9
```

**Steps**:
1. User provides Azure DevOps organization, project, and PAT
2. System encrypts PAT using Fernet encryption
3. System tests connection to Azure DevOps API
4. If successful, stores DataSource with `enabled=true`
5. If failed, stores but sets `enabled=false`

### 2. Work Item Ingestion Workflow

```mermaid
graph TD
    A[Scheduler<br/>Triggers Sync] --> B[Check Last<br/>Sync Time]
    B --> C[Azure DevOps<br/>API Call]
    C --> D[Process Delta<br/>Changes Only]
    D --> E[Parse & Store<br/>Work Items]
    E --> F[Update/Insert<br/>Database]

    style A fill:#fff3e0
    style F fill:#c8e6c9
```

**Steps**:
1. Scheduler runs periodic sync (configurable interval)
2. System checks `last_sync` timestamp for each enabled data source
3. Calls Azure DevOps Reporting API for changes since last sync
4. Processes each changed work item:
   - Standard fields stored in dedicated columns
   - Additional fields stored in `additional_fields` JSON column
   - Comments collected and stored as JSON array
5. Updates `last_sync` timestamp on success

### 3. Embedding Generation Workflow

```mermaid
graph LR
    A[User Triggers<br/>/embed] --> B[Load Work Items<br/>from Database]
    B --> C[Get Embedding<br/>Configurations]
    C --> D[Concatenate<br/>Text Fields]
    D --> E[Generate<br/>Embeddings]
    E --> F[Store in<br/>ChromaDB]

    style A fill:#e3f2fd
    style F fill:#f3e5f5
```

**Steps**:
1. User calls `/embed` endpoint (optionally with `force_rebuild=true`)
2. System loads all work items from enabled data sources
3. For each data source, loads embedding field configuration
4. For each work item:
   - Concatenates title, description, comments (always)
   - Adds configured additional fields (if enabled)
   - Generates embedding using configured provider (local/Azure)
5. Stores embeddings with metadata in ChromaDB
6. Metadata includes filterable fields for search

### 4. Search Workflow

```mermaid
graph LR
    A[User Query +<br/>Filters] --> B[Build Filter<br/>Criteria]
    B --> C[Generate Query<br/>Embedding]
    C --> D[Search ChromaDB<br/>with Filters]
    D --> E[Rerank Results<br/>Top-K]
    E --> F[Return Final<br/>Results]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

**Steps**:
1. User submits search request with query text and optional filters
2. System builds ChromaDB WHERE clause from filters
3. System generates embedding for query text
4. ChromaDB searches with cosine similarity and applies filters
5. System fetches 10x more candidates than needed
6. Reranking service improves result relevance
7. Returns top-k results with similarity scores

### 5. Field Discovery Workflow

```mermaid
graph LR
    A[User Requests<br/>Field Analysis] --> B[Sample Work<br/>Items N=100]
    B --> C[Analyze Field<br/>Types & Values]
    C --> D[Suggest Fields<br/>for Embedding]
    D --> E[Apply Selection<br/>Criteria]
    E --> F[Return Config<br/>Template]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

**Steps**:
1. User calls field discovery API for a data source
2. System samples up to 100 work items from that source
3. Analyzes both standard fields and `additional_fields` JSON
4. For each field, calculates:
   - Coverage percentage (how many items have this field)
   - Data types (string, number, boolean, etc.)
   - Sample values (for preview)
5. Applies filtering criteria:
   - Only suggests text (string) fields
   - Excludes system fields (IDs, dates, metadata)
   - Prioritizes high-coverage fields
6. Returns suggested configuration ready for application

## Data Flow Diagrams

### Complete System Data Flow

```mermaid
graph TD
    ADO[Azure DevOps] --> DIS[Data Ingestion<br/>Service]

    DIS --> DB[(SQLite<br/>Database)]
    API[Web API<br/>Endpoints] --> DB

    DIS --> FDS[Field Discovery<br/>Service]
    DB --> WI[Work Items<br/>+ Metadata]

    FDS --> ES[Embedding<br/>Service]
    WI --> ES

    ES --> CHROMA[(ChromaDB<br/>Vectors + Metadata)]

    CHROMA --> VS[Vector Search<br/>+ Reranking]
    VS --> API

    style ADO fill:#fff3e0
    style DB fill:#e8f5e8
    style CHROMA fill:#f3e5f5
    style API fill:#e3f2fd
```

### Embedding Data Flow

```mermaid
graph TD
    WI[Work Item Data] --> ALWAYS[Always Included:<br/>Title, Description, Comments]
    WI --> CONFIG[If Configured:<br/>Tags, AcceptCriteria,<br/>Custom Fields]

    ALWAYS --> CONCAT[Text Concatenation:<br/>Title Description Comment1 Comment2 Tags...]
    CONFIG --> CONCAT

    CONCAT --> EMBED[Embedding Generation:<br/>Vector: 0.1, 0.2, 0.3, ..., 0.n]

    EMBED --> STORE[ChromaDB Storage]

    STORE --> VEC[Vector:<br/>Embedding vector for<br/>similarity search]
    STORE --> META[Metadata:<br/>Filterable fields<br/>source, org, project, etc]
    STORE --> DOC[Document:<br/>Original concatenated text]

    style WI fill:#e3f2fd
    style CONCAT fill:#fff3e0
    style EMBED fill:#fce4ec
    style STORE fill:#f3e5f5
```

### Search Data Flow

```mermaid
graph TD
    QUERY[User Query:<br/>authentication issues] --> EMBED[Query Embedding:<br/>Vector: 0.2, 0.1, 0.4, ..., 0.m]

    QUERY --> FILTERS[Filters:<br/>source_names: ProjectA<br/>area_path: Backend]
    FILTERS --> WHERE[WHERE Clause:<br/>source_name IN ProjectA<br/>AND area_path REGEX ^Backend]

    EMBED --> CHROMA[ChromaDB Vector Search]
    WHERE --> CHROMA

    CHROMA --> CANDIDATES[Search Results:<br/>• Cosine Similarity<br/>• Apply Filters<br/>• Return 10x Candidates]

    CANDIDATES --> RERANK[Reranking Service:<br/>Improve relevance using<br/>query context]

    RERANK --> RESULTS[Final Results + Scores:<br/>Top-K most relevant<br/>work items]

    style QUERY fill:#e3f2fd
    style CHROMA fill:#f3e5f5
    style RESULTS fill:#c8e6c9
```

---

This schema documentation provides the foundation for understanding how data flows through the Orbis Search system and how the various components interact to provide intelligent search capabilities.
