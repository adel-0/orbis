# Embedding Configuration Guide

This guide explains how to configure what content gets embedded for semantic search, and how to optimize embedding quality for your specific Azure DevOps setup.

## Table of Contents

1. [Understanding Embeddings](#understanding-embeddings)
2. [Configuration Overview](#configuration-overview)
3. [Field Discovery Process](#field-discovery-process)
4. [Configuration Management](#configuration-management)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Understanding Embeddings

### What are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning. When you search for "authentication issues", the system finds work items with similar meanings, even if they use different words like "login problems" or "credential failures".

### What Gets Embedded?

The system combines multiple text fields from your work items into a single text string that gets converted to an embedding vector:

```
Title: "User cannot log into application"
Description: "Users report login failures after recent deployment..."
Comments: ["Reproduced on staging", "Related to SSL certificate"]
Additional Fields: "Impact: High priority customer escalation"

Combined Text: "User cannot log into application Users report login failures after recent deployment... Reproduced on staging Related to SSL certificate Impact: High priority customer escalation"
```

### Default Embedding Content

**Always Included:**
- Work item **Title**
- Work item **Description** (if present)
- All **Comments** (if present)

**Conditionally Included:**
- **Additional Fields** (only if configured per data source)

## Configuration Overview

### Per-Data-Source Configuration

Each data source (Azure DevOps organization/project combination) can have its own embedding configuration:

```json
{
  "embedding_field_config": {
    "embedding_fields": [
      "Microsoft.VSTS.Common.AcceptanceCriteria",
      "System.Tags",
      "Custom.BusinessValue"
    ],
    "enabled": true,
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

### Field Selection Impact

**Including More Fields:**
- ✅ Better semantic understanding
- ✅ More context for search
- ❌ Larger embeddings (more storage/compute)
- ❌ Potential noise if irrelevant fields included

**Including Fewer Fields:**
- ✅ Faster processing
- ✅ Less storage required
- ❌ May miss important context
- ❌ Reduced search accuracy

## Field Discovery Process

### 1. Automatic Field Analysis

The system can analyze your existing work items to discover available fields:

```bash
POST /field-discovery/discover/MyDataSource
```

**Response:**
```json
{
  "source_name": "MyDataSource",
  "total_work_items_analyzed": 150,
  "available_fields": {
    "Microsoft.VSTS.Common.AcceptanceCriteria": {
      "count": 98,
      "data_types": ["str"],
      "sample_values": ["Given user clicks login...", "Verify system responds..."],
      "coverage_percentage": 65.3
    },
    "System.Tags": {
      "count": 142,
      "data_types": ["str"],
      "sample_values": ["security; login", "ui; performance"],
      "coverage_percentage": 94.7
    }
  }
}
```

### 2. Intelligent Suggestions

The system suggests fields based on analysis:

```bash
POST /field-discovery/suggest-config/MyDataSource
```

**Response:**
```json
{
  "embedding_fields": [
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "System.Tags",
    "Custom.BusinessRequirements"
  ],
  "reasoning": [
    "Microsoft.VSTS.Common.AcceptanceCriteria: 65.3% coverage, text field",
    "System.Tags: 94.7% coverage, text field",
    "Custom.BusinessRequirements: 45.2% coverage, text field"
  ],
  "total_fields_analyzed": 28,
  "suggested_count": 3
}
```

### 3. Field Filtering Logic

The system automatically excludes fields that are:

**System/Metadata Fields:**
- IDs, GUIDs, URLs
- Dates and timestamps
- Version numbers
- Revision counts
- Person IDs

**Non-Text Fields:**
- Numeric values
- Boolean flags
- Binary data

**Common Azure DevOps System Fields (Excluded):**
- `System.Id`
- `System.CreatedDate`
- `System.ChangedDate`
- `System.AuthorizedDate`
- `System.Watermark`
- `System.Rev`

## Configuration Management

### 1. Get Ready-to-Use Configuration

Get a complete configuration template based on field analysis:

```bash
GET /field-discovery/config-template/MyDataSource
```

**Response:**
```json
{
  "embedding_field_config": {
    "embedding_fields": ["field1", "field2"],
    "enabled": true,
    "last_updated": null
  },
  "analysis_summary": {
    "total_fields_found": 25,
    "suggested_fields": 3,
    "work_items_analyzed": 100
  },
  "available_fields": { /* detailed field analysis */ },
  "suggestion_reasoning": [ /* why each field was suggested */ ]
}
```

### 2. Apply Configuration

Apply the configuration to your data source:

```bash
PUT /datasources/MyDataSource/embedding-config
Content-Type: application/json

{
  "embedding_fields": [
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "System.Tags"
  ],
  "enabled": true
}
```

### 3. Re-embed with New Configuration

After changing configuration, regenerate embeddings:

```bash
POST /embed
{
  "force_rebuild": true
}
```

## Best Practices

### 1. Field Selection Guidelines

**Recommended Fields to Include:**

| Field Type | Examples | Reasoning |
|------------|----------|-----------|
| **Requirements** | AcceptanceCriteria, BusinessRequirements | Rich context for search |
| **Descriptions** | DetailedDescription, UserStory | Natural language content |
| **Categories** | Tags, Components, Features | Semantic grouping |
| **Impact** | Priority explanation, BusinessValue | Business context |

**Fields to Avoid:**

| Field Type | Examples | Reasoning |
|------------|----------|-----------|
| **System Data** | CreatedBy, LastModifiedDate | No semantic value |
| **IDs/References** | WorkItemId, ParentId | Not searchable content |
| **Structured Data** | EstimatedHours, StoryPoints | Numeric, not textual |
| **Workflow State** | StateChangeDate, Reason | Process metadata |

### 2. Coverage Considerations

**High Coverage Fields (>70%):**
- Prioritize for inclusion
- Provide consistent context across work items
- Examples: Title, Tags, WorkItemType

**Medium Coverage Fields (30-70%):**
- Include if content-rich
- Examples: AcceptanceCriteria, BusinessValue

**Low Coverage Fields (<30%):**
- Usually exclude unless highly valuable
- May add noise to embeddings

### 3. Testing Configuration Changes

1. **Start Conservative**: Begin with only high-value, high-coverage fields
2. **Test Search Quality**: Run searches and evaluate results
3. **Iterate**: Add fields gradually and measure impact
4. **Monitor Performance**: Track embedding generation time and storage

### 4. Team-Specific Configurations

Different teams may benefit from different configurations:

**Development Teams:**
```json
{
  "embedding_fields": [
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "Microsoft.VSTS.TCM.ReproSteps",
    "System.Tags"
  ]
}
```

**Business Teams:**
```json
{
  "embedding_fields": [
    "Custom.BusinessRequirements",
    "Custom.BusinessValue",
    "Custom.UserImpact",
    "System.Tags"
  ]
}
```

## Configuration Examples

### Example 1: Software Development Team

```json
{
  "embedding_field_config": {
    "embedding_fields": [
      "Microsoft.VSTS.Common.AcceptanceCriteria",
      "Microsoft.VSTS.TCM.ReproSteps",
      "System.Tags",
      "Custom.TechnicalNotes"
    ],
    "enabled": true
  }
}
```

**Result**: Embeddings include user stories, test steps, categorization, and technical context.

### Example 2: Business Analysis Team

```json
{
  "embedding_field_config": {
    "embedding_fields": [
      "Custom.BusinessRequirements",
      "Custom.BusinessJustification",
      "Custom.UserImpact",
      "Microsoft.VSTS.Common.AcceptanceCriteria"
    ],
    "enabled": true
  }
}
```

**Result**: Embeddings focus on business context and user impact.

### Example 3: Minimal Configuration

```json
{
  "embedding_field_config": {
    "embedding_fields": [
      "System.Tags"
    ],
    "enabled": true
  }
}
```

**Result**: Only title, description, comments, and tags are embedded. Fast but may miss domain-specific context.

## Troubleshooting

### Common Issues

#### Poor Search Results

**Symptoms**: Searches return irrelevant results or miss obvious matches.

**Solutions**:
1. Check if important context fields are excluded
2. Review field coverage - low coverage fields may add noise
3. Ensure business domain terminology is captured in embedded fields

#### Slow Embedding Generation

**Symptoms**: Embedding process takes very long or times out.

**Solutions**:
1. Reduce number of embedded fields
2. Check if large text fields are included (some fields may contain huge amounts of text)
3. Increase batch size in configuration
4. Consider using fewer high-value fields

#### No Additional Fields Available

**Symptoms**: Field discovery shows only standard fields.

**Solutions**:
1. Verify Azure DevOps work items actually contain additional fields
2. Check if custom fields are properly synchronized during ingestion
3. Ensure work item types have custom fields configured

### Debugging Commands

```bash
# Check current embedding configuration
GET /datasources/MyDataSource

# Verify field availability
POST /field-discovery/discover/MyDataSource

# Check embedding status
GET /embed/status

# Test search with specific filters
POST /search
{
  "query": "test query",
  "source_names": ["MyDataSource"]
}
```

### Performance Monitoring

Monitor these metrics after configuration changes:

1. **Embedding Generation Time**: Should remain reasonable (<1 hour for typical datasets)
2. **Search Response Time**: Should stay under 2-3 seconds
3. **Storage Usage**: Monitor vector database size growth
4. **Search Quality**: Manually test search results relevance

---

For implementation details, see:
- [Search and Filtering Architecture](./SEARCH_AND_FILTERING_ARCHITECTURE.md)
- [Database Schema and Workflows](./DATABASE_SCHEMA_AND_WORKFLOWS.md)
- [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md)
