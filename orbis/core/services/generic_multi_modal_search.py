"""
Generic Multi-Modal Search Service - Configuration-driven approach.
Performs searches across ANY registered data source types using configuration.
Replaces hardcoded content type assumptions with dynamic collection routing.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.config.data_sources import (
    get_all_collection_names,
    get_collection_name,
    get_data_source_config,
    is_valid_source_type,
    list_data_source_types,
)
from core.schemas import (
    BaseContent,
    ScopeAnalysisResult,
    SearchPlan,
    SearchResult,
    Ticket,
    WikiPageContent,
)
from infrastructure.storage.cross_collection_rerank_service import (
    CrossCollectionReranker,
)
from infrastructure.storage.embedding_service import EmbeddingService
from infrastructure.storage.generic_vector_service import GenericVectorService
from infrastructure.storage.rerank_service import RerankService

logger = logging.getLogger(__name__)


class GenericAggregatedSearchResult:
    """Container for aggregated search results from multiple source types"""

    def __init__(self):
        self.results_by_source_type: dict[str, list[SearchResult]] = {}
        self.reranked_results: list[SearchResult] = []
        self.total_results: int = 0
        self.search_strategies_used: list[str] = []
        self.collections_searched: list[str] = []
        self.scope_analysis = None  # Store for cross-collection reranking

    @property
    def workitem_results(self) -> list[SearchResult]:
        """Get work item results for backward compatibility"""
        return self.results_by_source_type.get('azdo_workitems', [])

    @property
    def wiki_results(self) -> list[SearchResult]:
        """Get wiki results for backward compatibility"""
        return self.results_by_source_type.get('azdo_wiki', [])

    @property
    def code_results(self) -> list[SearchResult]:
        """Get code results for backward compatibility"""
        return self.results_by_source_type.get('code', [])

    @property
    def pdf_results(self) -> list[SearchResult]:
        """Get PDF results for backward compatibility"""
        return self.results_by_source_type.get('pdf', [])


class GenericMultiModalSearch:
    """Multi-modal search across ANY registered data source types"""

    def __init__(self,
                 vector_service: GenericVectorService | None = None,
                 embedding_service: EmbeddingService | None = None,
                 rerank_service: RerankService | None = None,
                 cross_collection_reranker: CrossCollectionReranker | None = None):

        self.vector_service = vector_service or GenericVectorService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.rerank_service = rerank_service or RerankService()
        self.cross_collection_reranker = cross_collection_reranker or CrossCollectionReranker(self.rerank_service)

        # Dynamic configuration based on registered data sources
        self.config = {
            "default_top_k": 20,
            "rerank_top_k": 8,
            "max_concurrent_searches": 4
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def search(self, query: str,
                    source_types: list[str] | None = None,
                    filters: dict[str, Any] | None = None,
                    top_k: int = 10) -> GenericAggregatedSearchResult:
        """
        Perform multi-modal search across specified or all data source types.

        Args:
            query: Search query
            source_types: List of source types to search (None = search all)
            filters: Optional metadata filters
            top_k: Number of results per source type

        Returns:
            GenericAggregatedSearchResult with results from all searched types
        """
        logger.debug(f"Multi-modal search: {len(source_types) if source_types else 'all'} source types, top_k={top_k}")

        result = GenericAggregatedSearchResult()

        # Determine which source types to search
        if source_types is None:
            source_types = list_data_source_types()
        else:
            # Validate source types
            len(source_types)
            source_types = [st for st in source_types if is_valid_source_type(st)]

        if not source_types:
            logger.warning("No valid source types for search")
            return result

        logger.debug(f"Searching: {source_types}")

        # Generate query embedding once
        logger.debug("Generating query embedding")
        query_embedding = await self.embedding_service.embed_query(query)
        if not query_embedding:
            logger.error("Could not generate embedding")
            return result

        logger.debug(f"Generated {len(query_embedding)}-dim embedding")

        # Prepare search tasks for each source type
        search_tasks = []
        for source_type in source_types:
            search_tasks.append(
                self._search_source_type(query, query_embedding, source_type, filters, top_k)
            )

        # Execute searches in parallel
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            for i, search_result in enumerate(search_results):
                if isinstance(search_result, Exception):
                    logger.error(f"❌ Search task {i} failed: {search_result}")
                    continue

                source_type, results, collection_name = search_result
                if results:
                    result.results_by_source_type[source_type] = results
                    result.collections_searched.append(collection_name)
                    result.search_strategies_used.append(f"{source_type}_semantic")

        # Apply cross-collection reranking if we have results from multiple sources
        all_results = []
        for source_results in result.results_by_source_type.values():
            all_results.extend(source_results)

        if len(all_results) > 1 and self.cross_collection_reranker:
            try:
                # Pass scope_analysis if available for proper boosting
                scope = getattr(result, 'scope_analysis', None)
                reranked_results = await self._rerank_cross_collection_results(
                    all_results, query, top_k=self.config["rerank_top_k"], scope_analysis=scope
                )
                result.reranked_results = reranked_results
                result.search_strategies_used.append("cross_collection_rerank")
            except Exception as e:
                logger.error(f"Cross-collection reranking failed: {e}")

        # Calculate total results
        result.total_results = len(all_results)

        logger.info(f"Generic multi-modal search completed: {result.total_results} total results")
        logger.info(f"Collections searched: {result.collections_searched}")
        logger.info(f"Strategies used: {result.search_strategies_used}")

        return result

    async def search_by_scope_analysis(self,
                                     query: str,
                                     scope_analysis: ScopeAnalysisResult,
                                     project_code: str | None = None) -> GenericAggregatedSearchResult:
        """
        Search based on scope analysis recommendations.
        Maintains compatibility with existing scope-based search.
        """
        logger.debug(f"Scope-driven search: confidence {scope_analysis.confidence:.2f}, {len(scope_analysis.recommended_source_types)} source types")

        # Map scope analysis source types to configuration source types
        source_types = self._map_scope_to_source_types(scope_analysis.recommended_source_types)
        logger.debug(f"Source types: {source_types}")

        # Build filters from scope analysis and project code
        filters = self._build_filters_from_scope(scope_analysis, project_code)
        if filters:
            logger.debug(f"Filters: {filters}")
        if project_code:
            logger.debug(f"Project: {project_code}")

        logger.debug(f"Executing with top_k={self.config['default_top_k']}")
        result = await self.search(
            query=query,
            source_types=source_types,
            filters=filters,
            top_k=self.config["default_top_k"]
        )
        
        # Store scope_analysis for cross-collection reranking
        result.scope_analysis = scope_analysis
        return result

    async def search_with_plan(self, query: str, search_plan: SearchPlan) -> GenericAggregatedSearchResult:
        """
        Execute search using a SearchPlan with weights and strategy.

        Args:
            query: Search query
            search_plan: SearchPlan with source types, weights, and filters

        Returns:
            GenericAggregatedSearchResult with weighted results
        """
        logger.info(f"Starting weighted search with plan: strategy={search_plan.strategy}, sources={search_plan.source_types}")

        result = GenericAggregatedSearchResult()

        if not search_plan.source_types:
            logger.warning("No source types specified in search plan")
            return result

        # Generate query embedding once
        query_embedding = await self.embedding_service.embed_query(query)
        if not query_embedding:
            logger.error("Could not generate embedding for query")
            return result

        # Calculate dynamic top_k based on strategy and weights
        base_top_k = self.config["default_top_k"]
        strategy_multiplier = {
            "focused": 0.8,    # Fewer results per source, higher precision
            "balanced": 1.0,   # Standard approach
            "broad": 1.3       # More results per source, higher recall
        }

        adjusted_top_k = max(1, int(base_top_k * strategy_multiplier.get(search_plan.strategy, 1.0)))

        # Prepare weighted search tasks
        search_tasks = []
        for source_type in search_plan.source_types:
            weight = search_plan.source_weights.get(source_type, 1.0)
            # Adjust per-source top_k based on weight
            source_top_k = max(1, int(adjusted_top_k * weight))

            search_tasks.append(
                self._search_source_type_weighted(
                    query, query_embedding, source_type,
                    search_plan.filters, source_top_k, weight
                )
            )

        # Execute searches in parallel
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process weighted results
            for i, search_result in enumerate(search_results):
                if isinstance(search_result, Exception):
                    logger.error(f"Weighted search task {i} failed: {search_result}")
                    continue

                source_type, results, collection_name, source_weight = search_result
                if results:
                    # Apply source weight to similarity scores
                    for search_result in results:
                        search_result.similarity_score *= source_weight
                        # Store original weight in metadata
                        if not hasattr(search_result, 'metadata'):
                            search_result.metadata = {}
                        search_result.metadata['source_weight'] = source_weight

                    result.results_by_source_type[source_type] = results
                    result.collections_searched.append(collection_name)
                    result.search_strategies_used.append(f"{source_type}_weighted_{search_plan.strategy}")

        # Apply cross-collection reranking with strategy context
        all_results = []
        for source_results in result.results_by_source_type.values():
            all_results.extend(source_results)

        if len(all_results) > 1 and self.cross_collection_reranker:
            try:
                # Adjust rerank count based on strategy
                rerank_count = {
                    "focused": max(3, self.config["rerank_top_k"]),
                    "balanced": self.config["rerank_top_k"],
                    "broad": min(8, self.config["rerank_top_k"] + 2)
                }.get(search_plan.strategy, self.config["rerank_top_k"])

                # Pass scope_analysis if available for proper boosting
                scope = getattr(result, 'scope_analysis', None)
                reranked_results = await self._rerank_cross_collection_results(
                    all_results, query, top_k=rerank_count, scope_analysis=scope
                )
                result.reranked_results = reranked_results
                result.search_strategies_used.append(f"cross_collection_rerank_{search_plan.strategy}")
            except Exception as e:
                logger.error(f"Cross-collection reranking failed: {e}")

        # Calculate total results
        result.total_results = len(all_results)

        logger.info(f"Weighted search completed: {result.total_results} total results using {search_plan.strategy} strategy")
        logger.info(f"Source weights applied: {search_plan.source_weights}")
        logger.info(f"Collections searched: {result.collections_searched}")

        return result

    async def search_with_routing_recommendations(self,
                                                query: str,
                                                scope_analysis,
                                                routing_recommendations,
                                                project_code: str | None = None) -> GenericAggregatedSearchResult:
        """
        Execute search using routing recommendations from LLM routing agent.
        
        Args:
            query: Search query
            scope_analysis: ScopeAnalysisResult from scope analyzer
            routing_recommendations: List of DataSourceRecommendation from routing agent
            project_code: Optional project code for filtering
            
        Returns:
            GenericAggregatedSearchResult with intelligently routed results
        """
        from core.schemas import SearchPlan
        
        logger.info(f"Starting search with {len(routing_recommendations)} routing recommendations")
        
        if not routing_recommendations:
            # Fallback to scope analysis if no recommendations
            logger.warning("No routing recommendations provided, falling back to scope analysis")
            return await self.search_by_scope_analysis(query, scope_analysis, project_code)
        
        # Convert routing recommendations to SearchPlan
        # We need to map source names to types and create source-specific filters
        source_types = []
        source_weights = {}
        source_name_filters = []  # Track specific source names for filtering
        
        for rec in routing_recommendations:
            if rec.relevance_score >= 0.5:  # Only include relevant recommendations
                source_type = rec.source_type
                source_name = rec.source_name
                
                # Add to search plan
                if source_type not in source_types:
                    source_types.append(source_type)
                    source_weights[source_type] = rec.search_weight
                else:
                    # If multiple instances of same type, use higher weight
                    source_weights[source_type] = max(source_weights[source_type], rec.search_weight)
                
                # Track specific source names for filtering
                source_name_filters.append(source_name)
                
        if not source_types:
            # If no recommendations meet threshold, fallback to scope analysis
            logger.warning("No routing recommendations meet relevance threshold, falling back to scope analysis")
            return await self.search_by_scope_analysis(query, scope_analysis, project_code)
        
        # Create filters from project context AND routing recommendations
        filters = {}
        if project_code:
            filters["project"] = project_code
        
        # Add source name filtering to target specific instances
        if source_name_filters:
            filters["source_names"] = source_name_filters
            
        # Determine search strategy based on number of recommendations and confidence
        avg_confidence = sum(rec.relevance_score for rec in routing_recommendations) / len(routing_recommendations)
        if len(routing_recommendations) <= 2 and avg_confidence >= 0.8:
            strategy = "focused"
        elif len(routing_recommendations) >= 4 or avg_confidence <= 0.6:
            strategy = "broad"
        else:
            strategy = "balanced"
            
        search_plan = SearchPlan(
            source_types=source_types,
            source_weights=source_weights,
            filters=filters,
            strategy=strategy
        )
        
        logger.info(f"Created search plan: strategy={strategy}, source_types={source_types}")
        logger.info(f"Source weights from routing: {source_weights}")
        
        # Execute search with the created plan
        result = await self.search_with_plan(query, search_plan)
        
        # Store scope_analysis for cross-collection reranking
        result.scope_analysis = scope_analysis
        return result

    def _build_vector_filters(self, filters: dict[str, Any] | None, source_type: str) -> dict[str, Any] | None:
        """Convert SearchPlan filters to ChromaDB where filters"""
        if not filters:
            return None
        
        vector_filters = {}
        
        # Handle source_names filtering - filter to specific data source instances
        if "source_names" in filters and filters["source_names"]:
            source_names = filters["source_names"]
            if isinstance(source_names, list) and len(source_names) > 1:
                # Multiple source names - use $in operator
                vector_filters["source_name"] = {"$in": source_names}
            elif isinstance(source_names, list) and len(source_names) == 1:
                # Single source name
                vector_filters["source_name"] = source_names[0]
            else:
                # Single source name as string
                vector_filters["source_name"] = source_names
            
            logger.info(f"Filtering {source_type} to specific sources: {source_names}")
        
        # Handle other standard filters
        for key, value in filters.items():
            if key != "source_names":  # Skip the special source_names filter
                vector_filters[key] = value
        
        return vector_filters if vector_filters else None

    async def _search_source_type_weighted(self, query: str,
                                          query_embedding: list[float],
                                          source_type: str,
                                          filters: dict[str, Any] | None = None,
                                          top_k: int = 10,
                                          weight: float = 1.0) -> tuple[str, list[SearchResult], str, float]:
        """Search a single source type with weight information"""
        try:
            logger.debug(f"Weighted search on {source_type} with weight {weight}")

            # Use existing search logic
            source_type, results, collection_name = await self._search_source_type(
                query, query_embedding, source_type, filters, top_k
            )

            # Return results with weight
            return source_type, results, collection_name, weight

        except Exception as e:
            logger.error(f"Error in weighted search for source type {source_type}: {e}")
            return source_type, [], "", weight

    async def _search_source_type(self, query: str,
                                 query_embedding: list[float],
                                 source_type: str,
                                 filters: dict[str, Any] | None = None,
                                 top_k: int = 10) -> tuple[str, list[SearchResult], str]:
        """Search a single source type using its configured collection"""
        try:
            logger.debug(f"Searching source type: {source_type}")

            # Get collection name from connector (self-describing)
            from core.config.data_sources import get_collection_name
            collection_name = get_collection_name(source_type)

            # Process filters and convert source_names to ChromaDB where clause
            vector_filters = self._build_vector_filters(filters, source_type)

            # Search the collection
            search_results = await self.vector_service.search_content(
                query_embedding=query_embedding,
                collection_names=[collection_name],
                top_k=top_k,
                where_filters=vector_filters
            )

            # Convert to SearchResult objects
            results = []
            for item in search_results:
                if item["collection"] == collection_name:  # Should always be true for single collection search
                    # Create a BaseContent object from the raw content and metadata
                    content_obj = self._create_content_object_from_item(item, source_type)

                    search_result = SearchResult(
                        content=content_obj,
                        similarity_score=item["similarity"],
                        concatenated_text=item["content"],  # Generic content is already the searchable text
                        content_type=content_obj.content_type,  # Use the actual content type, not source_type
                        metadata=item["metadata"]
                    )
                    results.append(search_result)

            # Apply reranking if available
            if results and self.rerank_service.is_loaded():
                results = await self._rerank_results(query, results)

            logger.debug(f"Found {len(results)} results for source type {source_type}")
            return source_type, results, collection_name

        except Exception as e:
            logger.error(f"Error searching source type {source_type}: {e}")
            return source_type, [], ""

    def _create_content_object_from_item(self, item: dict[str, Any], source_type: str) -> BaseContent:
        """Create a BaseContent object from a search result item"""
        try:
            metadata = item.get("metadata", {})

            # Extract basic content info
            content_id = metadata.get("id", "unknown")
            title = metadata.get("title", "Untitled")

            # Infer content type from source type
            # Connectors are self-describing, content_type mapping is straightforward
            content_type_map = {
                "azdo_workitems": "work_item",
                "azdo_wiki": "wiki_page",
                "oncall_web_help": "oncall_web_help"
            }
            content_type = content_type_map.get(source_type, "generic")

            # Create appropriate content object based on source type
            if source_type == "azdo_workitems" or content_type == "work_item":
                # Parse additional_fields if it's a JSON string
                additional_fields = metadata.get("additional_fields", {})
                if isinstance(additional_fields, str):
                    import json
                    try:
                        additional_fields = json.loads(additional_fields)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse additional_fields JSON for {content_id}")
                        additional_fields = {}

                # Put all Azure DevOps-specific fields into additional_fields
                if not additional_fields:
                    additional_fields = {}

                # Add Azure DevOps fields to additional_fields if not already there
                azdo_fields = {
                    "state": metadata.get("state"),
                    "assigned_to": metadata.get("assigned_to"),
                    "work_item_type": metadata.get("work_item_type"),
                    "priority": metadata.get("priority"),
                    "severity": metadata.get("severity"),
                    "tags": metadata.get("tags", []),
                    "azure_created_date": metadata.get("azure_created_date"),
                    "azure_changed_date": metadata.get("azure_changed_date"),
                    "azure_resolved_date": metadata.get("azure_resolved_date"),
                    "azure_url": metadata.get("azure_url"),
                    "created_by": metadata.get("created_by")
                }
                # Only add non-None values
                for key, value in azdo_fields.items():
                    if value is not None and key not in additional_fields:
                        additional_fields[key] = value

                # Create a Ticket object with only schema-compliant fields
                return Ticket(
                    id=content_id,
                    title=title,
                    content_type="workitem",
                    description=metadata.get("description", ""),
                    comments=metadata.get("comments", []),
                    area_path=metadata.get("area_path"),
                    additional_fields=additional_fields,
                    source_name=metadata.get("source_name"),
                    organization=metadata.get("organization"),
                    project=metadata.get("project"),
                    source_type=source_type
                )
            elif source_type == "azdo_wiki" or content_type == "wiki_page":
                # Handle last_modified - convert empty strings to None
                last_modified = metadata.get("last_modified")
                if last_modified == "":
                    last_modified = None

                # Parse image_references if it's a JSON string
                image_references = metadata.get("image_references", [])
                if isinstance(image_references, str):
                    import json
                    try:
                        image_references = json.loads(image_references)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse image_references JSON for {content_id}")
                        image_references = []

                # Create a WikiPageContent object with only schema-compliant fields
                return WikiPageContent(
                    id=content_id,
                    title=title,
                    content_type="wiki_page",
                    content=metadata.get("content", ""),
                    path=metadata.get("path", ""),
                    html_content=metadata.get("html_content"),
                    image_references=image_references,
                    last_modified=last_modified,
                    author=metadata.get("author"),
                    source_name=metadata.get("source_name"),
                    organization=metadata.get("organization"),
                    project=metadata.get("project"),
                    source_type=source_type
                )
            else:
                # Create a generic BaseContent object
                return BaseContent(
                    id=content_id,
                    title=title,
                    content_type=content_type,
                    source_name=metadata.get("source_name"),
                    organization=metadata.get("organization"),
                    project=metadata.get("project"),
                    source_type=source_type
                )

        except Exception as e:
            import traceback
            logger.error(f"Error creating content object for {source_type}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            logger.debug(f"Item metadata keys: {list(item.get('metadata', {}).keys())}")
            # Return a minimal BaseContent as fallback but preserve metadata fields for URL generation
            metadata = item.get("metadata", {})
            return BaseContent(
                id=metadata.get("id", "unknown"),
                title=metadata.get("title", "Error parsing content"),
                content_type="error",
                source_type=source_type,
                source_name=metadata.get("source_name"),
                organization=metadata.get("organization"),
                project=metadata.get("project")
            )

    async def search_collections(self, query: str,
                               collection_names: list[str],
                               filters: dict[str, Any] | None = None,
                               top_k: int = 10) -> list[SearchResult]:
        """
        Search specific collections directly (bypass source type mapping).
        Useful for advanced searches or custom collection combinations.
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            if not query_embedding:
                logger.error("Could not generate embedding for collection search")
                return []

            # Search across specified collections
            search_results = await self.vector_service.search_content(
                query_embedding=query_embedding,
                collection_names=collection_names,
                top_k=top_k,
                where_filters=filters
            )

            # Convert to SearchResult objects
            results = []
            for item in search_results:
                # Create a generic BaseContent object for direct collection search
                content_obj = BaseContent(
                    id=item.get("metadata", {}).get("id", "unknown"),
                    title=item.get("metadata", {}).get("title", "Direct search result"),
                    content_type="generic",
                    source_name=item.get("metadata", {}).get("source_name"),
                    organization=item.get("metadata", {}).get("organization"),
                    project=item.get("metadata", {}).get("project"),
                    source_type="generic"
                )

                search_result = SearchResult(
                    content=content_obj,
                    similarity_score=item["similarity"],
                    concatenated_text=item["content"],
                    content_type="generic",  # Generic type for direct collection search
                    metadata=item.get("metadata", {})
                )
                # Add collection info to metadata
                search_result.metadata["collection"] = item["collection"]
                results.append(search_result)

            # Apply reranking if available
            if results and self.rerank_service.is_loaded():
                results = await self._rerank_results(query, results)

            return results

        except Exception as e:
            logger.error(f"Error searching collections {collection_names}: {e}")
            return []

    async def search_all_collections(self, query: str,
                                   filters: dict[str, Any] | None = None,
                                   top_k: int = 10) -> list[SearchResult]:
        """Search across all available collections"""
        collection_names = get_all_collection_names()
        return await self.search_collections(query, collection_names, filters, top_k)

    def _map_scope_to_source_types(self, scope_source_types: list[str]) -> list[str]:
        """Map scope analysis source types to configuration source types"""
        mapped_types = []
        for source_type in scope_source_types:
            if is_valid_source_type(source_type):
                mapped_types.append(source_type)
            else:
                logger.warning(f"Invalid source type specified: {source_type}")

        return mapped_types

    def _build_filters_from_scope(self, scope_analysis: ScopeAnalysisResult,
                                 project_code: str | None) -> dict[str, Any] | None:
        """Build search filters from scope analysis"""
        filters = {}

        # TEMPORARY: Project filtering disabled due to missing metadata fields
        # TODO: Re-enable after data re-ingestion with proper metadata
        # if project_code:
        #     filters["project"] = project_code

        # Add any specific filters from scope analysis
        if hasattr(scope_analysis, 'filters') and scope_analysis.filters:
            filters.update(scope_analysis.filters)

        return filters if filters else None

    async def _rerank_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Apply reranking to search results"""
        try:
            if not self.rerank_service.is_loaded():
                return results

            # Convert to format expected by reranker, preserving index for mapping
            rerank_input = []
            for idx, result in enumerate(results):
                rerank_input.append({
                    "text": result.concatenated_text,
                    "metadata": {**getattr(result, 'metadata', {}), "__orig_index__": idx}
                })

            # Apply reranking
            reranked_items = self.rerank_service.rerank(
                query=query,
                items=rerank_input,
                top_k=len(rerank_input)
            )

            # Attach rerank scores back to corresponding results safely
            for item in reranked_items:
                meta = item.get('metadata', {}) or {}
                idx = meta.get('__orig_index__')
                if idx is None:
                    continue
                if 0 <= idx < len(results):
                    results[idx].rerank_score = item.get('rerank_score', results[idx].similarity_score)

            # Sort by rerank scores
            results.sort(key=lambda x: getattr(x, 'rerank_score', x.similarity_score), reverse=True)

            # Cap list length to number of returned reranked items
            return results[:len(reranked_items)]

        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results

    async def _rerank_cross_collection_results(self, all_results: list[SearchResult],
                                             query: str,
                                             top_k: int = 3,
                                             scope_analysis=None) -> list[SearchResult]:
        """Apply cross-collection reranking with proper two-stage normalization"""
        try:
            # Use the cross-collection reranker if available
            if hasattr(self.cross_collection_reranker, 'rerank_cross_collection_results'):
                # Pass scope_analysis for optional boosting
                return await self.cross_collection_reranker.rerank_cross_collection_results(
                    all_results=all_results,
                    query=query,
                    scope_analysis=scope_analysis,
                    target_count=top_k
                )
            else:
                # Fallback: simple reranking
                return await self._rerank_results(query, all_results[:top_k])

        except Exception as e:
            logger.error(f"Error in cross-collection reranking: {e}")
            # Return top results by similarity score as fallback
            sorted_results = sorted(all_results, key=lambda x: x.similarity_score, reverse=True)
            return sorted_results[:top_k]

    def get_supported_source_types(self) -> list[str]:
        """Get all supported source types from configuration"""
        return list_data_source_types()

    def get_collection_for_source_type(self, source_type: str) -> str | None:
        """Get collection name for a source type"""
        try:
            return get_collection_name(source_type)
        except ValueError:
            return None

    def get_search_stats(self) -> dict[str, Any]:
        """Get search service statistics"""
        return {
            "supported_source_types": self.get_supported_source_types(),
            "available_collections": get_all_collection_names(),
            "collection_stats": self.vector_service.get_collection_stats(),
            "embedding_service_configured": self.embedding_service is not None,
            "rerank_service_configured": self.rerank_service.is_loaded() if self.rerank_service else False
        }


    def is_loaded(self) -> bool:
        """Check if the search service is properly configured"""
        return (
            self.vector_service is not None and
            self.embedding_service is not None and
            len(get_all_collection_names()) > 0
        )

    def is_configured(self) -> bool:
        """Alias for is_loaded() for compatibility with orchestrator"""
        return self.is_loaded()
