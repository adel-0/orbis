# Search and Filtering Architecture

This document explains how the OnCall Copilot search and filtering system works, including the complete data flow from work item ingestion to search results.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Filtering Mechanisms](#filtering-mechanisms)
4. [Embedding Process](#embedding-process)
5. [Search Workflow](#search-workflow)
6. [Configuration Management](#configuration-management)
7. [Performance Considerations](#performance-considerations)

## System Overview

The OnCall Copilot implements a **two-tier search architecture**:

1. **Metadata Filtering**: Fast pre-filtering using structured data
2. **Semantic Search**: AI-powered similarity search on embedded content

This approach combines the speed of traditional filtering with the intelligence of semantic search.

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Azure DevOps  │───▶│  Data Ingestion  │───▶│   Work Items    │
│   Work Items    │    │    Service       │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Search Results  │◀───│   Rerank Service │◀───│ Embedding       │
│                 │    │                  │    │ Service         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                                               │
         │                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vector Service  │───▶│   ChromaDB       │◀───│ Vector Storage  │
│ (Search)        │    │  Collection      │    │   + Metadata    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Filtering Mechanisms

### 1. Metadata-Based Filtering (Pre-Search)

Filtering happens **before** semantic search using structured metadata stored alongside embeddings in ChromaDB.

#### Available Filters

| Filter Type | Description | Pattern Matching |
|-------------|-------------|------------------|
| `source_names` | Data source names | Exact match or IN clause |
| `organizations` | Azure DevOps organizations | Exact match or IN clause |
| `projects` | Project names | Exact match or IN clause |
| `area_path_prefix` | Area path starts with | Regex: `^{prefix}` |
| `area_path_contains` | Area path contains | Regex: `{text}` |
| `iteration_path_prefix` | Iteration path starts with | Regex: `^{prefix}` |
| `iteration_path_contains` | Iteration path contains | Regex: `{text}` |

#### Filter Logic

```python
# Single value filters use equality
{"source_name": {"$eq": "MySource"}}

# Multiple values use IN clause
{"source_name": {"$in": ["Source1", "Source2"]}}

# Path filters use regex
{"area_path": {"$regex": "^MyTeam"}}

# Multiple filters combine with AND
{"$and": [
    {"source_name": {"$eq": "MySource"}},
    {"area_path": {"$regex": "^MyTeam"}}
]}
```

### 2. Metadata Storage

Each work item stores the following metadata alongside its embedding:

```json
{
  "id": "work_item_id",
  "title": "Work item title",
  "description": "Work item description",
  "comments": "[\"comment1\", \"comment2\"]",
  "concatenated_text": "Full text used for embedding",
  "source_name": "data_source_name",
  "organization": "azure_devops_org",
  "project": "project_name",
  "area_path": "Team\\SubTeam\\Component",
  "iteration_path": "Release\\Sprint1"
}
```

## Embedding Process

### 1. Content Selection

The system determines what content to embed based on **configurable field settings** per data source:

#### Always Embedded (Core Fields)
- **Title**: Always included in embeddings
- **Description**: Included if present
- **Comments**: All comments included if present

#### Conditionally Embedded (Configurable Fields)
- **Additional Fields**: Only included if specifically configured for the data source
- **Field Selection**: Based on embedding field configuration stored in the database

### 2. Text Concatenation Logic

```python
def _concatenate_ticket_text(ticket, embedding_config):
    parts = [ticket.title]  # Always include title
    
    # Core fields
    if ticket.description:
        parts.append(ticket.description)
    if ticket.comments:
        parts.extend(ticket.comments)
    
    # Configured additional fields
    if embedding_config and embedding_config.get('enabled'):
        embedding_fields = embedding_config.get('embedding_fields', [])
        for field_name in embedding_fields:
            field_value = ticket.additional_fields.get(field_name)
            if field_value and isinstance(field_value, str):
                parts.append(field_value.strip())
    
    return " ".join(parts)
```

### 3. Field Configuration Structure

Each data source can have an `embedding_field_config`:

```json
{
  "embedding_fields": [
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "System.Tags",
    "Custom.BusinessValue"
  ],
  "enabled": true,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

## Search Workflow

### 1. Complete Search Process

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Query Processing                                          │
│   • Parse search request                                    │
│   • Extract filters and query text                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Filter Building                                          │
│   • Build ChromaDB WHERE clause from filters               │
│   • Combine multiple filters with AND logic                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Query Embedding                                          │
│   • Convert query text to embedding vector                 │
│   • Use same provider as used for indexing                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Candidate Retrieval                                     │
│   • Search ChromaDB with filters applied                   │
│   • Fetch 10x more candidates than needed (min 10*top_k)   │
│   • Use cosine similarity for ranking                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Reranking                                               │
│   • Apply reranking service to improve relevance           │
│   • Return top-k final results                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Summary Generation (Optional)                           │
│   • Generate AI summary of results                         │
│   • Include similarity scores                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Search API Flow

```python
# Example search request
POST /search
{
  "query": "authentication issues",
  "top_k": 5,
  "source_names": ["MyProject"],
  "area_path_prefix": "Backend",
  "projects": ["ProjectA", "ProjectB"]
}

# Processing steps:
# 1. Build filter: {"$and": [
#      {"source_name": {"$in": ["MyProject"]}},
#      {"area_path": {"$regex": "^Backend"}},
#      {"project": {"$in": ["ProjectA", "ProjectB"]}}
#    ]}
# 2. Embed query: [0.1, 0.2, 0.3, ...]
# 3. Search ChromaDB with filter
# 4. Rerank candidates
# 5. Return top 5 results
```

## Configuration Management

### 1. Field Discovery Process

The system can automatically analyze your work items to suggest embedding configurations:

```python
# Discover available fields
POST /field-discovery/discover/{source_name}

# Get suggested configuration
POST /field-discovery/suggest-config/{source_name}

# Apply configuration to data source
PUT /datasources/{source_name}/embedding-config
```

### 2. Configuration Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Field Discovery │───▶│   Analysis &     │───▶│   Apply Config  │
│   Service       │    │   Suggestions    │    │   to DataSource │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                              ┌─────────────────┐
│ Sample Work     │                              │ Re-embed with   │
│ Items Analysis  │                              │ New Config      │
└─────────────────┘                              └─────────────────┘
```

### 3. Field Selection Criteria

The field discovery service suggests fields based on:

- **Data Type**: Only text (string) fields are suggested
- **Coverage**: Fields present in a meaningful percentage of work items
- **Content Quality**: Excludes system fields (IDs, dates, GUIDs)
- **Relevance**: Excludes metadata fields unlikely to improve search

## Performance Considerations

### 1. Filtering Performance

- **Metadata filters are fast**: Applied at the database level in ChromaDB
- **Regex filters are slower**: Use sparingly, prefer exact matches when possible
- **Multiple filters**: Combined efficiently with ChromaDB's query optimization

### 2. Embedding Performance

- **Embedding size matters**: More fields = larger embeddings = more storage/compute
- **Quality vs. Quantity**: Include relevant fields, exclude noise
- **Batch processing**: Embeddings generated in configurable batches

### 3. Search Performance

- **Two-stage search**: Balances accuracy and speed
- **Candidate pool**: Fetches 10x candidates for reranking optimization
- **Caching**: Embedding models cached in memory

## Common Patterns

### 1. Team-Specific Search

```python
# Search within a specific team's work
{
  "query": "performance issues",
  "area_path_prefix": "MyTeam\\Backend",
  "source_names": ["CompanyProject"]
}
```

### 2. Cross-Project Analysis

```python
# Search across multiple projects
{
  "query": "security vulnerabilities",
  "projects": ["ProjectA", "ProjectB", "ProjectC"],
  "organizations": ["MyOrg"]
}
```

### 3. Sprint-Specific Items

```python
# Find items in current sprint
{
  "query": "blocked tasks",
  "iteration_path_contains": "Sprint 23",
  "projects": ["CurrentProject"]
}
```

## Troubleshooting

### Common Issues

1. **No search results**: Check if embeddings exist (`GET /embed/status`)
2. **Poor search quality**: Review embedding field configuration
3. **Slow searches**: Reduce candidate pool size or simplify filters
4. **Missing work items**: Verify data source is enabled and synchronized

### Debugging Tools

- **Field Discovery**: Analyze what fields are available
- **Embedding Status**: Check embedding generation status
- **Collection Info**: Verify vector storage statistics
- **Search Debugging**: Use verbose logging to trace search process

---

For more specific implementation details, see:
- [Dynamic Fields Implementation](../DYNAMIC_FIELDS_IMPLEMENTATION.md)
- [Service Documentation](./SERVICES.md)
- [API Reference](./API_REFERENCE.md)
