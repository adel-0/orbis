from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Data source types are now dynamic strings from configuration
# Common examples: "azdo_workitems", "azdo_wiki", "jira_issues", etc.

# Context tags for intelligent routing
class ContextTag(str, Enum):
    CORE_PRODUCT = "core_product"
    PROJECT_SPECIFIC = "project_specific"

# Base content model for different types of content
class BaseContent(BaseModel):
    id: str
    title: str
    content_type: str  # "workitem", "wiki_page"
    source_name: str | None = None
    organization: str | None = None
    project: str | None = None
    source_type: str | None = None  # Dynamic source type from data source
    
    def get_rerank_text(self) -> str:
        """Return text representation optimized for reranking"""
        return self.title


class Ticket(BaseContent):
    content_type: str = "workitem"
    description: str | None = None
    comments: list[str] = Field(default_factory=list)
    area_path: str | None = None  # Azure DevOps area path
    additional_fields: dict[str, Any] | None = None  # Dynamic fields from work items
    
    def get_rerank_text(self) -> str:
        """Return text representation optimized for reranking"""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.comments:
            parts.extend(self.comments[:2])  # Include first 2 comments
        return " ".join(parts)


class WikiPageContent(BaseContent):
    content_type: str = "wiki_page"
    content: str | None = None  # Markdown content
    html_content: str | None = None  # Rendered HTML
    path: str | None = None  # Wiki page path
    image_references: list[dict[str, str]] | None = None
    author: str | None = None
    last_modified: datetime | None = None
    
    def get_rerank_text(self) -> str:
        """Return text representation optimized for reranking"""
        parts = [self.title]
        if self.content:
            parts.append(self.content)
        return " ".join(parts)




class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")

    # Content type filtering
    content_types: list[str] | None = Field(None, description="Filter by content types: workitem, wiki_page")
    data_source_types: list[str] | None = Field(None, description="Filter by data source types")

    # Intelligent routing options
    context_tags: list[ContextTag] | None = Field(None, description="Manual context tags to guide search (optional - AI automatically determines optimal routing)")

    # Traditional filtering options
    source_names: list[str] | None = Field(None, description="Filter by specific data source names")
    organizations: list[str] | None = Field(None, description="Filter by specific organizations")
    projects: list[str] | None = Field(None, description="Filter by specific projects")
    area_path_contains: str | None = Field(None, description="Filter by area path containing this text")
    area_path_prefix: str | None = Field(None, description="Filter by area path starting with this prefix")


    # Wiki specific filters
    wiki_paths: list[str] | None = Field(None, description="Filter wiki pages by path prefix")

class SearchResult(BaseModel):
    content: BaseContent  # Now accepts any BaseContent subclass dynamically
    similarity_score: float
    concatenated_text: str
    rerank_score: float | None = None
    normalized_score: float | None = None  # For cross-collection normalization
    context_score: float | None = None  # For context-aware scoring
    final_score: float | None = None  # Final combined score after reranking and boosting
    content_type: str  # For easier handling in UI
    metadata: dict[str, Any] | None = Field(default_factory=dict)  # Additional metadata from vector DB
    stage1_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    blended_score: float | None = None

class SearchResponse(BaseModel):
    results: list[SearchResult]
    summary: str | None = None
    total_results: int
    query: str
    filters_applied: dict[str, Any] | None = Field(None, description="Summary of filters applied to the search")

class EmbedRequest(BaseModel):
    force_rebuild: bool = Field(default=False, description="Force rebuild embeddings")

class EmbedResponse(BaseModel):
    message: str
    total_tickets: int
    processed_tickets: int
    success: bool

class EmbedProgress(BaseModel):
    current: int
    total: int
    percentage: float
    message: str
    status: str  # "processing", "completed", "error"

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    chroma_connected: bool
    azure_openai_configured: bool
    total_tickets: int
    database_info: dict[str, Any] | None = None

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    timestamp: datetime

class EmbeddingProviderInfo(BaseModel):
    loaded: bool
    provider: str
    model_name: str | None = None
    device: str | None = None
    max_seq_length: int | str | None = None
    embedding_dimension: int | None = None

# Data ingestion schemas
class DataIngestionRequest(BaseModel):
    force_full_sync: bool = Field(default=False, description="Force full sync instead of incremental")
    skip_embedding: bool = Field(default=True, description="Skip automatic embedding generation")

class DataIngestionResponse(BaseModel):
    success: bool
    message: str | None = None
    sources: list[dict[str, Any]] | None = None  # Per-source ingestion results
    total_fetched_workitems: int | None = None
    total_saved_workitems: int | None = None
    total_workitems_across_sources: int | None = None
    execution_time_seconds: float
    timestamp: str
    embedding_result: dict[str, Any] | None = None
    error: str | None = None


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    schedule_time: str
    next_run: str | None = None
    last_run_time: str | None = None
    last_run_result: dict[str, Any] | None = None
    task_status: str  # "running" or "stopped"


# New schemas for intelligent routing
class ContextAnalysisRequest(BaseModel):
    query: str = Field(..., description="Query to analyze for context")

class ContextAnalysisResponse(BaseModel):
    detected_context_tags: list[ContextTag]
    suggested_source_types: list[str]  # Changed from suggested_repositories
    reasoning: str
    confidence_score: float

class DataSourceRecommendation(BaseModel):  # Renamed from RepositoryRecommendation
    source_type: str  # Changed from repository_name
    source_name: str  # Actual source name/identifier
    context_tags: list[ContextTag]
    relevance_score: float
    reasoning: str
    search_weight: float = 1.0  # Added search weight

class SearchPlan(BaseModel):
    """Search plan with weighted source types and filters"""
    source_types: list[str]
    source_weights: dict[str, float] = Field(default_factory=dict)  # Weight per source type
    filters: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "balanced"  # "focused", "broad", "balanced"

class IntelligentSearchResponse(SearchResponse):
    context_analysis: ContextAnalysisResponse | None = None
    source_recommendations: list[DataSourceRecommendation] | None = None  # Renamed from repository_recommendations
    search_strategy: str | None = None  # "focused", "broad", "hybrid" - derived from SearchPlan
    search_weights: dict[str, float] | None = None  # Source weights used - derived from SearchPlan

# =============================================================================
# Agentic RAG System Models
# =============================================================================

class ProjectContext(BaseModel):
    """Project context detected from ticket content and metadata"""
    project_code: str | None = Field(None, description="Project code like 'SG', 'VS', or None for general")

class ScopeAnalysisResult(BaseModel):
    """Result from Scope Analyzer's scope and intent analysis"""
    scope_description: str = Field(..., description="What the content concerns (components, interfaces, areas)")
    intent_description: str = Field(..., description="What the user wants to achieve")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in analysis")
    recommended_source_types: list[str] = Field(..., description="Source types to search")

class SourceReference(BaseModel):
    """Reference to a specific source found during search"""
    source_type: str = Field(..., description="Type of source (ticket, wiki, pdf)")
    source_name: str = Field(..., description="Name/identifier of the source")
    title: str = Field(..., description="Title or summary of the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to the query")
    snippet: str | None = Field(None, description="Relevant snippet from the source")
    url: str | None = Field(None, description="URL or path to the source")

class AgenticRAGResponse(BaseModel):
    """Complete response from the agentic RAG system"""
    project_context: ProjectContext
    scope_analysis: ScopeAnalysisResult
    final_summary: str = Field(..., description="Documentation Aggregator's aggregated summary and recommendations")
    referenced_sources: list[SourceReference] = Field(default_factory=list, description="All sources referenced")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the response")
    processing_time_ms: int | None = Field(None, description="Total processing time in milliseconds")

class AgenticRAGRequest(BaseModel):
    """Request for agentic RAG processing"""
    content: str = Field(..., description="Content to process (text, query, description, etc.)")
    area_path: str | None = Field(None, description="Area path for project detection")
    additional_context: dict[str, Any] | None = Field(None, description="Any additional context")

class WikiSummary(BaseModel):
    """Summarized wiki content for contextual analysis"""
    wiki_name: str = Field(..., description="Name of the wiki")
    summary: str = Field(..., description="High-level summary of wiki content")
    key_components: list[str] = Field(default_factory=list, description="Key components/interfaces mentioned")
    summary_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in summary quality")
    tokens_used: int = Field(..., description="Number of tokens used for summarization")

# =============================================================================
# Integration Models (formerly in infrastructure/integrators/base.py)
# =============================================================================

class IntegrationStatus(str, Enum):
    """Status of integration operations"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

class IntegrationResult(BaseModel):
    """Result of an integration operation"""
    status: IntegrationStatus
    items_processed: int
    items_successful: int
    items_failed: int
    execution_time_seconds: float
    errors: list[str]
    metadata: dict[str, Any]

class ContentItem(BaseModel):
    """Content item returned by integrators"""
    id: str
    title: str
    content: str
    content_type: str  # "workitem", "wiki_page"
    source_metadata: dict[str, Any]
    extracted_metadata: dict[str, Any]
    last_modified: datetime | None = None
    author: str | None = None

# =============================================================================
# LLM Response Models
# =============================================================================

