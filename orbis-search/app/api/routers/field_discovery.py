"""
API router for field discovery functionality.
Separate from data source management to keep concerns separated.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from services.field_discovery_service import FieldDiscoveryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/field-discovery", tags=["field-discovery"])


@router.get("/analyze/{source_name}")
async def analyze_source_fields(
    source_name: str,
    sample_size: Optional[int] = Query(100, description="Number of work items to analyze", ge=10, le=1000)
) -> Dict[str, Any]:
    """
    Analyze available additional fields for a data source.
    
    Args:
        source_name: Name of the data source to analyze
        sample_size: Number of work items to sample for analysis (10-1000)
    
    Returns:
        Field analysis including counts, types, and sample values
    """
    try:
        discovery_service = FieldDiscoveryService()
        field_analysis = discovery_service.discover_fields_for_source(source_name, sample_size)
        
        return {
            "success": True,
            "data": field_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze fields for source {source_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze fields for data source: {str(e)}"
        )


@router.get("/suggest/{source_name}")
async def suggest_embedding_fields(
    source_name: str,
    sample_size: Optional[int] = Query(100, description="Number of work items to analyze", ge=10, le=1000)
) -> Dict[str, Any]:
    """
    Get suggested text fields useful for embeddings/search.
    
    Args:
        source_name: Name of the data source
        sample_size: Number of work items to analyze
    
    Returns:
        Suggested fields with reasoning. Use them to update the data source 'fields' and re-ingest.
    """
    try:
        discovery_service = FieldDiscoveryService()
        
        # Get field analysis
        field_analysis = discovery_service.discover_fields_for_source(source_name, sample_size)
        
        # Generate suggestions
        suggestions = discovery_service.suggest_embedding_fields(field_analysis)
        
        return {
            "success": True,
            "source_name": source_name,
            "suggested_fields": suggestions["embedding_fields"],
            "analysis_details": {
                "total_fields_analyzed": suggestions.get("total_fields_analyzed", 0),
                "suggested_count": suggestions.get("suggested_count", 0),
                "reasoning": suggestions.get("reasoning", [])
            },
            "field_details": field_analysis.get("available_fields", {}),
            "usage_instructions": (
                "Add these to DataSource.fields and re-run POST /ingest to include them."
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to suggest embedding fields for source {source_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suggest embedding fields: {str(e)}"
        )


@router.get("/template/{source_name}")
async def get_embedding_config_template(source_name: str) -> Dict[str, Any]:
    """
    Get a discovery summary and suggested fields for a data source.
    
    Args:
        source_name: Name of the data source
    
    Returns:
        Summary containing available fields and suggested fields for ingestion.
    """
    try:
        discovery_service = FieldDiscoveryService()
        template = discovery_service.get_embedding_config_template(source_name)
        return {
            "success": True,
            "source_name": source_name,
            "summary": {
                "available_fields": template.get("available_fields", {}),
                "suggested_fields": template.get("embedding_field_config", {}).get("embedding_fields", [])
            },
            "usage_instructions": (
                "Add suggested fields to DataSource.fields and re-run POST /ingest."
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to get embedding config template for source {source_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embedding config template: {str(e)}"
        )
