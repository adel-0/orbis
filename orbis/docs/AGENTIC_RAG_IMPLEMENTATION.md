# Agentic RAG System Implementation

A multi-agent RAG system providing intelligent ticket analysis through configuration-driven workflows with autonomous query routing and cross-collection search capabilities.

## Architecture

### Core Workflow

1. **Project Detection** (Non-LLM)
   - Area path-based pattern matching for Swiss projects
   - Currently detects SG (St. Gallen) and VS (Valais) projects
   - Simple, efficient detection based on Azure DevOps area paths

2. **LLM Routing Agent** (New - AI-Powered)
   - Autonomous query analysis using Azure OpenAI
   - Intelligent data source selection and routing
   - Context analysis and repository recommendations
   - Enhanced query understanding with confidence scoring

3. **Ticket Scope Analysis**
   - Advanced LLM analysis with documentation context
   - Analyzes scope, intent, and technical requirements
   - Code search detection (`requires_code_search` field)
   - Project-specific wiki summaries for enhanced context

4. **Configuration-Driven Multi-Modal Search**
   - Generic search across ANY registered data source types
   - Dynamic collection routing based on configuration
   - Cross-collection reranking for improved relevance
   - Parallel execution with up to 4 concurrent operations

5. **Documentation Aggregation**
   - Multi-source result aggregation with advanced filtering
   - Context-aware summarization with source ranking
   - Comprehensive reference creation with relevance scores
   - Overall confidence calculation and response validation

## Implementation Details

### Core Services

#### LLM Routing Agent (`engine/agents/llm_routing_agent.py`) - **590 lines**
- **Purpose**: Autonomous AI-powered query analysis and routing
- **Features**:
  - Advanced query understanding using Azure OpenAI
  - Intelligent data source selection and prioritization
  - Context tag analysis and categorization
  - Repository profiling and recommendations
  - Confidence-based routing decisions

#### Project Detection Service (`engine/services/project_detection.py`)
- **Purpose**: Simple area path-based project identification
- **Features**:
  - Area path pattern matching for Swiss projects (SG, VS)
  - Streamlined detection logic
  - Wiki repository mapping (SG → "Wiki.SG", VS → "Wiki.VS")

#### Wiki Summarization Service (`engine/services/wiki_summarization.py`) - **485 lines**
- **Purpose**: Large-scale wiki content processing and summarization
- **Features**:
  - Handles hundreds of thousands of tokens
  - Intelligent chunked summarization
  - Basic caching for performance
  - Component extraction and analysis

#### Ticket Scope Analyzer (`engine/agents/scope_analyzer.py`) - **431 lines**
- **Purpose**: Advanced ticket scope and intent analysis
- **Features**:
  - Enhanced LLM analysis with documentation context
  - Code search requirement detection
  - Structured analysis with confidence scoring
  - Dynamic source type recommendations

#### Generic Multi-Modal Search Service (`engine/services/generic_multi_modal_search.py`) - **381 lines**
- **Purpose**: Configuration-driven search across any data source types
- **Features**:
  - Works with ANY registered data source types
  - Dynamic collection routing based on configuration
  - Cross-collection reranking service integration
  - Parallel search with configurable concurrency limits

#### Documentation Aggregator (`engine/agents/documentation_aggregator.py`) - **528 lines**
- **Purpose**: Advanced multi-source result aggregation
- **Features**:
  - Enhanced source ranking and filtering algorithms
  - Context window management for large result sets
  - Cross-collection result integration
  - Comprehensive summary generation with metadata

#### Agentic RAG Orchestrator (`engine/agents/orchestrator.py`) - **399 lines**
- **Purpose**: Main workflow coordinator with enhanced capabilities
- **Features**:
  - Complete workflow orchestration with 120-second timeout
  - Advanced error handling and graceful degradation
  - Performance monitoring and detailed logging
  - Health checking and comprehensive status reporting

#### Generic Data Ingestion Service (`engine/services/generic_data_ingestion.py`) - **379 lines**
- **Purpose**: Universal data ingestion across any data source types
- **Features**:
  - Configuration-driven ingestion workflows
  - Dynamic field discovery and mapping
  - Support for any registered connector type
  - Automated ingestion scheduling and monitoring

#### Document Processor Service (`engine/services/document_processor.py`) - **168 lines**
- **Purpose**: Universal document processing utilities
- **Features**:
  - Multi-format document handling
  - Content extraction and normalization
  - Metadata processing and enrichment

#### Shared OpenAI Client Service (`infrastructure/llm/openai_client.py`) - **New**
- **Purpose**: Centralized Azure OpenAI client management
- **Features**:
  - Single shared client instance across all services
  - Reduces connection overhead and resource usage
  - Centralized configuration management
  - Service wrapper for dependency injection

### Enhanced Models & Database Schema

#### Core Schemas (`engine/models/schemas.py`)
- `DataSourceType`: Dynamic enum for any data source types
- `ProjectContext`: Project detection results with area path mapping
- `ScopeAnalysisResult`: Enhanced scope analysis with code search detection
- `SourceReference`: Comprehensive source references with metadata
- `AgenticRAGRequest`: Complete request model with enhanced fields
- `AgenticRAGResponse`: Detailed response model with confidence metrics
- `WikiSummary`: Advanced wiki summarization results
- `ContextTag`: Context analysis and categorization system
- `RepositoryRecommendation`: AI-powered source selection results

#### Database Models (`app/db/models.py`)
- `DataSource`: Generic model with JSON configuration fields
- `WorkItem`: Enhanced with dynamic `additional_fields` for any field types
- Context tags and priority fields for intelligent routing
- Backward compatibility through legacy property methods

### API Integration

#### Enhanced Endpoints (`app/api/routers/analyze.py`)
- `POST /analyze`: Main ticket analysis endpoint with enhanced capabilities
- `GET /datasources/types`: List all supported data source types
- `POST /datasources`: Create data sources of any type
- Generic field discovery and ingestion endpoints

#### Service Integration
- Simplified dependency injection using `@lru_cache` decorators
- Direct service instantiation without complex service containers
- Health monitoring and status reporting integration

## Configuration

### Configuration-Driven Architecture

The system now uses a centralized configuration approach that eliminates hardcoded assumptions:

#### Data Source Registry (`engine/config/data_sources.py`)
```python
REGISTERED_DATA_SOURCES = {
    "azdo_workitems": {
        "collection_name": "work_items",
        "display_name": "Azure DevOps Work Items",
        "search_enabled": True
    },
    "azdo_wiki": {
        "collection_name": "wiki_pages", 
        "display_name": "Azure DevOps Wiki Pages",
        "search_enabled": True
    }
}
```

#### Project Configuration (Simplified)
Current Swiss project support in `ProjectDetectionService`:
```python
PROJECT_CONSTANTS = {
    "SG": {
        "area_paths": ["St. Gallen"],
        "wiki_repos": ["Wiki.SG"]
    },
    "VS": {
        "area_paths": ["Valais", "Wallis"], 
        "wiki_repos": ["Wiki.VS"]
    }
}
```

#### Dynamic Source Type Mapping

The LLM Routing Agent and Scope Analyzer dynamically map analysis results to configured data source types:

- **Configuration/Parameters** → `azdo_workitems`, project customization documents
- **Installation/Setup** → `azdo_wiki`, installation documentation  
- **Interface/Integration** → `azdo_wiki`, technical documentation
- **Code-related Issues** → All sources + code search enablement
- **Base sources**: Always includes all registered and enabled source types

## Usage

### API Request Example

```json
{
  "ticket_content": "Custom interface module throwing null pointer exception in St. Gallen deployment",
  "area_path": "St. Gallen\\Interfaces"
}
```

### Enhanced Response Structure

```json
{
  "project_context": {
    "project_code": "SG",
    "confidence": 0.95,
    "matched_patterns": ["St. Gallen"],
    "detection_source": "area_path"
  },
  "scope_analysis": {
    "scope_description": "Custom interface module in St. Gallen project deployment",
    "intent_description": "Debug and resolve null pointer exception error",
    "confidence": 0.92,
    "recommended_source_types": ["azdo_wiki", "azdo_workitems"],
    "requires_code_search": true,
    "context_tags": ["interface", "deployment", "debugging"]
  },
  "llm_routing_analysis": {
    "recommended_repositories": ["Wiki.SG", "ProjectCustomization"],
    "routing_confidence": 0.88,
    "analysis_summary": "Technical interface issue requiring wiki and work item analysis"
  },
  "final_summary": "**Problem Analysis:**\n...",
  "referenced_sources": [...],
  "overall_confidence": 0.89,
  "processing_time_ms": 4500,
  "search_metadata": {
    "cross_collection_reranking": true,
    "parallel_searches": 3,
    "total_sources_found": 15
  }
}
```

## Performance

### Current Performance Metrics
- **Total Response Time**: 4-7 seconds end-to-end (extended due to enhanced analysis)
- **Project Detection**: <50ms (simplified area path matching)
- **LLM Routing Analysis**: 1-2 seconds (new component)
- **Ticket Scope Analysis**: 1-3 seconds (enhanced with code search detection)
- **Generic Multi-Modal Search**: 1-3 seconds (cross-collection reranking)
- **Documentation Aggregation**: 2-4 seconds (enhanced filtering and ranking)

### Configuration & Optimizations
- **Orchestrator Timeout**: 120 seconds (vs. previous shorter timeouts)
- **Parallel Search Execution**: Up to 4 concurrent operations
- **Basic Caching**: Wiki summaries cached for performance
- **Cross-Collection Reranking**: Enabled for improved relevance
- **Minimum Confidence Threshold**: 0.3 for result inclusion
- **Context Window Management**: Advanced token management for large result sets
- **Result Deduplication**: Enhanced cross-source duplicate detection
- **Shared OpenAI Client**: Single client instance across all services

## Key Features

### 1. **Configuration-Driven Architecture**
- Eliminates hardcoded assumptions with dynamic configuration
- Supports ANY data source type through registry system
- Modular connector architecture reducing integration complexity
- Single configuration entry + connector approach vs. 20+ file changes

### 2. **AI-Powered Intelligent Routing**
LLM Routing Agent provides autonomous query analysis and source selection, context analysis and repository profiling, intelligent confidence-based routing decisions, and advanced query understanding with categorization.

### 3. **Enhanced Multi-Agent Architecture**
- **Project Detection**: Simplified area path-based identification
- **LLM Routing Agent**: AI-powered query analysis and routing
- **Scope Analyzer**: Enhanced analysis with code search detection
- **Generic Multi-Modal Search**: Configuration-driven search across any sources
- **Documentation Aggregator**: Advanced result integration and ranking

### 4. **Generic Multi-Source Search**
- **Current Sources**: Azure DevOps work items, project wikis, documents
- **Extensible Design**: Can add any data source type via configuration
- **Cross-Collection Reranking**: Enhanced relevance scoring
- **Parallel Processing**: Up to 4 concurrent search operations

### 5. **Advanced Intelligence Features**
- **Code Search Detection**: Identifies when code analysis is needed
- **Context Tag System**: Automated categorization and routing
- **Repository Recommendations**: AI-powered source prioritization
- **Enhanced Confidence Scoring**: Multi-level assessment with detailed metrics

### 6. **Project-Aware Intelligence**
Currently supports SG (St. Gallen) and VS (Valais) projects through area path-based detection with wiki repository mapping. Project-specific context and recommendations are provided.

### 7. **Error Handling & Monitoring**
Includes 120-second timeout for complex queries, graceful degradation with fallback responses, error logging and health checking.

## Integration Points

### Existing System Integration
- **Azure OpenAI Integration**: Uses existing credentials and configuration
- **Vector Search Infrastructure**: Integrates with existing embedding services
- **Cross-Collection Reranking**: Leverages enhanced reranking capabilities
- **Database Compatibility**: Maintains backward compatibility with existing models
- **API Pattern Consistency**: Follows established FastAPI routing patterns

### Current Limitations & Enhancement Opportunities

#### Missing Project Configurations
- **WT (Winterthur)**: Not currently configured in PROJECT_CONSTANTS
- **SH (Schaffhausen)**: Not currently configured in PROJECT_CONSTANTS
- Easy to add through area path and wiki repository mapping

#### Connector Ecosystem
- **Current**: Only Azure DevOps connectors implemented
- **Opportunity**: Generic architecture supports any data source connector
- **Implementation**: Need additional connectors for other systems

#### Code Search Integration
- **Status**: `requires_code_search` field exists but full implementation unclear
- **Opportunity**: Complete code search workflow integration
- **Architecture**: Framework exists for code analysis capabilities

#### Advanced Features Utilization
- **LLM Routing Agent**: 590-line component with intelligent routing capabilities
- **Context Tag System**: Defined but usage patterns need optimization
- **Repository Recommendations**: AI-powered features may need fine-tuning
