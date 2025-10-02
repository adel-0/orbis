"""
LLM-powered intelligent routing agent for OnCall Copilot.
Uses Azure OpenAI to analyze queries and route them to the most relevant data sources.
This agent autonomously decides which data sources to search based on query analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from core.config.data_sources import DataSourceConfigRegistry
from core.schemas import (
    ContextAnalysisResponse,
    DataSourceRecommendation,
    SearchPlan,
    SearchRequest,
)
from utils.constants import LLM_MAX_OUTPUT_TOKENS
from utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

@dataclass
class DataSourceProfile:
    """Profile of a data source for LLM context"""
    name: str
    source_type: str
    description: str
    context_tags: list[str]
    priority: int
    content_summary: str
    search_weight: float = 1.0

@dataclass
class RoutingPrompt:
    """Structured prompt for LLM routing"""
    query: str
    available_sources: list[DataSourceProfile]
    system_context: str

class QueryRoutingAgent:
    """Intelligent query routing agent that autonomously decides data source routing"""

    def __init__(self, openai_client_service=None, data_source_service=None):
        # Use shared OpenAI client
        from infrastructure.llm.openai_client import OpenAIClientService
        self.openai_client_service = openai_client_service or OpenAIClientService()

        # Use data source service for database access
        from infrastructure.data_processing.data_source_service import DataSourceService
        self.data_source_service = data_source_service or DataSourceService()

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

        # Routing configuration
        self.routing_config = {
            "max_sources": 3,  # Maximum data sources to recommend
            "confidence_threshold": 0.5,  # Minimum confidence for recommendations
            "enable_search_weights": True  # Enable weight-based search strategy
        }


    async def analyze_query_context(self, query: str) -> ContextAnalysisResponse:
        """Analyze a query to determine its context and routing"""
        try:
            response = await self._analyze_with_llm(query)
            return response

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            raise

    async def _analyze_with_llm(self, query: str) -> ContextAnalysisResponse:
        """Analyze query using Azure OpenAI with structured outputs"""
        try:
            # Get available data sources dynamically
            data_sources = await self._get_data_source_profiles()

            # Build dynamic data source list for prompt
            available_sources = "\n".join([
                f"- Name: {source.name}\n"
                f"  Type: {source.source_type}\n"
                f"  Description: {source.description}\n"
                f"  Priority: {source.priority}\n"
                f"  Content: {source.content_summary}\n"
                for source in data_sources
            ])

            # Get prompt from external template with dynamic injection
            prompt_variables = {
                "available_sources": available_sources,
                "query": query,
                "max_sources": self.routing_config["max_sources"],
                "confidence_threshold": self.routing_config["confidence_threshold"]
            }

            developer_prompt = self.prompt_loader.get_developer_prompt("routing_recommendations", prompt_variables)
            user_prompt = self.prompt_loader.get_user_prompt("routing_recommendations", prompt_variables)

            # Call Azure OpenAI with structured output
            response = await self._call_azure_openai_structured(developer_prompt, user_prompt)

            # Parse structured LLM response
            analysis = self._parse_structured_response(response, data_sources)

            return analysis

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise



    async def _get_data_source_profiles(self) -> list[DataSourceProfile]:
        """Get profiles of available data source instances from database"""
        profiles = []

        try:
            # Get actual data source instances from database via service
            data_sources = self.data_source_service.get_all_data_sources(enabled_only=True)
            
            for ds in data_sources:
                profile = DataSourceProfile(
                    name=ds.name,
                    source_type=ds.source_type,
                    description=self._generate_instance_description(ds),
                    context_tags=ds.context_tags or [],
                    priority=ds.priority or 3,
                    content_summary=self._generate_instance_content_summary(ds),
                    search_weight=self._get_search_weight_for_source_type(ds.source_type)
                )
                profiles.append(profile)

        except Exception as e:
            logger.error(f"Error getting data source profiles: {e}")

        return profiles

    def _generate_instance_description(self, data_source) -> str:
        """Generate description using configuration and context tags only"""
        from core.config.data_sources import get_data_source_config
        
        try:
            # Get generic source type configuration
            source_config = get_data_source_config(data_source.source_type)
            base_desc = f"{data_source.name} ({data_source.source_type})"
            
            # Add context tags - let LLM understand their meaning
            if data_source.context_tags:
                tags = ", ".join(data_source.context_tags)
                base_desc += f" | Tags: [{tags}]"
            
            return base_desc
            
        except ValueError:
            # Unknown source type, use minimal description
            base_desc = f"{data_source.name} ({data_source.source_type})"
            if data_source.context_tags:
                tags = ", ".join(data_source.context_tags)
                base_desc += f" | Tags: [{tags}]"
            return base_desc

    def _generate_instance_content_summary(self, data_source) -> str:
        """Generate content summary using configuration hints and context tags"""
        from core.config.data_sources import get_data_source_config
        
        try:
            # Use searchable_content_hint from configuration
            source_config = get_data_source_config(data_source.source_type)
            content_hint = source_config.get('searchable_content_hint', 'Various content types')
            
            # Let context tags speak for themselves - LLM will understand
            if data_source.context_tags:
                tags = ", ".join(data_source.context_tags)
                return f"{content_hint} | Specializes in: {tags}"
            else:
                return content_hint
                
        except ValueError:
            # Unknown source type, generic summary
            if data_source.context_tags:
                tags = ", ".join(data_source.context_tags)
                return f"Content related to: {tags}"
            else:
                return "Mixed content types"

    def _get_search_weight_for_source_type(self, source_type: str) -> float:
        """Get search weight for source type from configuration"""
        from core.config.data_sources import get_priority_boost_for_source_type
        return get_priority_boost_for_source_type(source_type)


    async def _call_azure_openai_structured(self, developer_prompt: str, user_prompt: str) -> str:
        """Call Azure OpenAI API with structured JSON output using shared client"""
        try:
            messages = [
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.openai_client_service.get_chat_completion(
                messages=messages,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to call Azure OpenAI via shared client: {e}")
            raise

    def _parse_structured_response(self, llm_response: str, data_sources: list[DataSourceProfile]) -> ContextAnalysisResponse:
        """Parse structured LLM response using Pydantic models"""
        try:
            # Parse JSON response directly into ContextAnalysisResponse
            structured_response = ContextAnalysisResponse.model_validate_json(llm_response)

            # The response is already in the correct format
            return structured_response

        except Exception as e:
            logger.error(f"Failed to parse structured LLM response: {e}")
            raise


    def apply_intelligent_routing(self, search_request: SearchRequest) -> SearchRequest:
        """Apply intelligent routing to modify search request"""
        # This would be called synchronously, so we need to handle async analysis
        # For now, just return the request - the async version should be used
        logger.warning("Synchronous routing not implemented - use async version")
        return search_request

    async def apply_intelligent_routing_async(self, search_request: SearchRequest) -> SearchRequest:
        """Apply intelligent routing asynchronously"""
        try:
            # Analyze the query
            analysis = await self.analyze_query_context(search_request.query)

            # If no specific sources are requested, use recommendations
            if not search_request.source_names and analysis.suggested_source_types:
                search_request.source_names = analysis.suggested_source_types

            # Add context tags if not specified
            if not search_request.context_tags:
                search_request.context_tags = analysis.detected_context_tags

            return search_request

        except Exception as e:
            logger.error(f"Error applying intelligent routing: {e}")
            return search_request

    def get_routing_config(self) -> dict[str, Any]:
        """Get current routing configuration"""
        return {
            **self.routing_config,
            "deployment_name": self.deployment_name
        }

    def update_routing_config(self, config_updates: dict[str, Any]) -> None:
        """Update routing configuration"""
        for key, value in config_updates.items():
            if key in self.routing_config:
                self.routing_config[key] = value
                logger.info(f"Updated routing config: {key} = {value}")


    async def get_detailed_source_recommendations(self, query: str) -> list[DataSourceRecommendation]:
        """Get detailed data source recommendations with explanations using structured outputs"""
        try:
            # Get data sources dynamically from database
            data_sources = await self._get_data_source_profiles()

            # Build dynamic data source list - let LLM understand from raw data
            available_sources = "\n".join([
                f"- Name: {source.name}\n"
                f"  Type: {source.source_type}\n" 
                f"  Description: {source.description}\n"
                f"  Tags: [{', '.join(source.context_tags)}]\n"
                f"  Priority: {source.priority}\n"
                f"  Content: {source.content_summary}\n"
                for source in data_sources
            ])

            # Inject dynamic data into prompt template
            prompt_variables = {
                "available_sources": available_sources,
                "query": query
            }

            developer_prompt = self.prompt_loader.get_developer_prompt("routing_recommendations", prompt_variables)
            user_prompt = self.prompt_loader.get_user_prompt("routing_recommendations", prompt_variables)

            # Call Azure OpenAI with structured output
            llm_response = await self._call_azure_openai_structured(developer_prompt, user_prompt)
            return self._parse_detailed_structured_recommendations(llm_response, data_sources)

        except Exception as e:
            logger.error(f"Error getting detailed recommendations: {e}")
            return []


    def _parse_detailed_structured_recommendations(self, llm_response: str, data_sources: list[DataSourceProfile]) -> list[DataSourceRecommendation]:
        """Parse detailed LLM recommendations using structured models"""
        try:
            # Parse JSON response directly as list of DataSourceRecommendation objects
            import json
            response_data = json.loads(llm_response)

            # Handle both list format and object with 'recommendations' field
            if isinstance(response_data, list):
                recommendations_data = response_data
            elif isinstance(response_data, dict) and 'recommendations' in response_data:
                recommendations_data = response_data['recommendations']
            else:
                recommendations_data = []

            recommendations = []
            for rec_data in recommendations_data:
                recommendation = DataSourceRecommendation.model_validate(rec_data)
                recommendations.append(recommendation)

            # Sort by relevance score
            recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            return recommendations

        except Exception as e:
            logger.error(f"Failed to parse detailed structured recommendations: {e}")
            raise

    async def generate_search_plan(self, query: str, project_context: str | None = None) -> SearchPlan:
        """Generate a comprehensive search plan with weights and filters"""
        try:
            # Get context analysis
            analysis = await self.analyze_query_context(query)

            # Get data source profiles
            data_sources = await self._get_data_source_profiles()

            # Build source weights from recommended sources and their profiles
            source_weights = {}
            recommended_sources = analysis.suggested_source_types

            if not recommended_sources:
                # Fallback to all available sources with equal weight
                recommended_sources = [source.source_type for source in data_sources]

            for source_type in recommended_sources:
                # Find matching data source profile
                matching_source = next((s for s in data_sources if s.source_type == source_type), None)
                if matching_source:
                    source_weights[source_type] = matching_source.search_weight
                else:
                    source_weights[source_type] = 1.0

            # Determine search strategy based on confidence and source count
            if analysis.confidence_score >= 0.8 and len(recommended_sources) <= 2:
                strategy = "focused"
            elif analysis.confidence_score <= 0.5 or len(recommended_sources) > 3:
                strategy = "broad"
            else:
                strategy = "balanced"

            # Build filters from project context
            filters = {}
            if project_context:
                filters["project"] = project_context

            return SearchPlan(
                source_types=recommended_sources,
                source_weights=source_weights,
                filters=filters,
                strategy=strategy
            )

        except Exception as e:
            logger.error(f"Error generating search plan: {e}")
            # Return default search plan
            return SearchPlan(
                source_types=["azdo_workitems", "azdo_wiki"],
                source_weights={"azdo_workitems": 1.3, "azdo_wiki": 0.7},
                filters={},
                strategy="balanced"
            )

