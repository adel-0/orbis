from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class Ticket(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    comments: List[str] = Field(default_factory=list)
    source_name: Optional[str] = None  # Which data source this ticket came from
    organization: Optional[str] = None  # DevOps organization
    project: Optional[str] = None  # DevOps project
    area_path: Optional[str] = None  # Azure DevOps area path
    iteration_path: Optional[str] = None  # Azure DevOps iteration path
    created_date: Optional[datetime] = None  # Creation date for recency boosting
    additional_fields: Optional[Dict[str, Any]] = None  # Dynamic fields from work items

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    include_summary: bool = Field(default=False, description="Whether to include AI-generated summary in the response")
    enable_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranking (blended with RRF if enabled)")
    # Filtering options
    source_names: Optional[List[str]] = Field(None, description="Filter by specific data source names")
    organizations: Optional[List[str]] = Field(None, description="Filter by specific organizations")
    projects: Optional[List[str]] = Field(None, description="Filter by specific projects")
    area_path_contains: Optional[str] = Field(None, description="Filter by area path containing this text")
    area_path_prefix: Optional[str] = Field(None, description="Filter by area path starting with this prefix")
    iteration_path_contains: Optional[str] = Field(None, description="Filter by iteration path containing this text")
    iteration_path_prefix: Optional[str] = Field(None, description="Filter by iteration path starting with this prefix")

class SearchResult(BaseModel):
    ticket: Ticket
    rerank_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    confidence_score: Optional[float] = None
    blended_score: Optional[float] = None
    recency_boost: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    summary: Optional[str] = None
    total_results: int
    query: str
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Summary of filters applied to the search")

class EmbedRequest(BaseModel):
    force_rebuild: bool = Field(default=False, description="Force rebuild embeddings")
    skip_embedding: bool = Field(default=False, description="Skip vector embedding generation")
    skip_bm25_indexing: bool = Field(default=False, description="Skip BM25 keyword index generation")

class EmbedResponse(BaseModel):
    message: str
    total_tickets: int
    processed_tickets: int
    success: bool

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    chroma_connected: bool
    azure_openai_configured: bool
    total_tickets: int
    database_info: Optional[Dict[str, Any]] = None

class EmbeddingProviderInfo(BaseModel):
    loaded: bool
    provider: str
    current_provider: str
    model_name: Optional[str] = None
    device: Optional[str] = None
    max_seq_length: Optional[Union[int, str]] = None
    embedding_dimension: Optional[int] = None
    deployment_name: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None


# Data ingestion schemas
class DataIngestionRequest(BaseModel):
    force_full_sync: bool = Field(default=False, description="Force full sync instead of incremental")
    skip_embedding: bool = Field(default=False, description="Skip automatic embedding generation")
    source_names: Optional[List[str]] = Field(None, description="Specific sources to sync (empty = all enabled sources)")

class DataIngestionResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None  # Per-source ingestion results
    total_fetched_workitems: Optional[int] = None
    total_saved_workitems: Optional[int] = None
    total_workitems_across_sources: Optional[int] = None
    execution_time_seconds: float
    timestamp: str
    embedding_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class IngestionStatusResponse(BaseModel):
    sources: Optional[List['SourceIngestionStatus']] = None
    total_sources: Optional[int] = None
    enabled_sources: Optional[int] = None
    global_last_run: Optional[str] = None
    total_workitems_across_sources: Optional[int] = None

class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    schedule_time: str
    next_run: Optional[str] = None
    last_run_time: Optional[str] = None
    last_run_result: Optional[Dict[str, Any]] = None
    task_status: str  # "running" or "stopped"

# Data source configuration schemas
class DataSourceCreateRequest(BaseModel):
    name: str = Field(..., description="Unique name for this data source")
    organization: str = Field(..., description="Azure DevOps organization name")
    project: str = Field(..., description="Azure DevOps project name")
    auth_type: str = Field(..., description="Authentication type: 'pat' or 'oauth2'")
    # PAT fields
    pat: Optional[str] = Field(None, description="Personal Access Token (required for PAT auth)")
    # OAuth2 fields  
    client_id: Optional[str] = Field(None, description="OAuth2 Client ID (required for OAuth2 auth)")
    client_secret: Optional[str] = Field(None, description="OAuth2 Client Secret (required for OAuth2 auth)")
    tenant_id: Optional[str] = Field(None, description="OAuth2 Tenant ID (required for OAuth2 auth)")
    query_ids: List[str] = Field(..., description="List of WIQL query IDs to execute")
    fields: Optional[List[str]] = Field(None, description="Azure DevOps fields to fetch (limits fetch to these fields)")
    enabled: bool = Field(default=True, description="Whether this data source is enabled")

class DataSourceUpdateRequest(BaseModel):
    organization: Optional[str] = Field(None, description="Azure DevOps organization name")
    project: Optional[str] = Field(None, description="Azure DevOps project name")
    auth_type: Optional[str] = Field(None, description="Authentication type: 'pat' or 'oauth2'")
    # PAT fields
    pat: Optional[str] = Field(None, description="Personal Access Token")
    # OAuth2 fields
    client_id: Optional[str] = Field(None, description="OAuth2 Client ID")
    client_secret: Optional[str] = Field(None, description="OAuth2 Client Secret")
    tenant_id: Optional[str] = Field(None, description="OAuth2 Tenant ID")
    query_ids: Optional[List[str]] = Field(None, description="List of WIQL query IDs to execute")
    fields: Optional[List[str]] = Field(None, description="Azure DevOps fields to fetch (limits fetch to these fields)")
    enabled: Optional[bool] = Field(None, description="Whether this data source is enabled")

class DataSourceResponse(BaseModel):
    name: str
    organization: str
    project: str
    auth_type: str
    # Masked credentials for security
    pat_masked: Optional[str] = None  # Masked PAT for PAT auth
    client_id_masked: Optional[str] = None  # Masked Client ID for OAuth2 auth
    query_ids: List[str]
    fields: Optional[List[str]] = None
    enabled: bool
    last_sync: Optional[str] = None
    total_workitems: int = 0

class DataSourceListResponse(BaseModel):
    data_sources: List[DataSourceResponse]
    total_sources: int
    enabled_sources: int

class SourceIngestionStatus(BaseModel):
    source_name: str
    last_run_timestamp: Optional[str] = None
    total_workitems: int
    workitems_date_range: Optional[Dict[str, str]] = None
    organization: str
    project: str
    enabled: bool
