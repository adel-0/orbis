"""
Service for discovering and analyzing dynamic fields from work items.
Helps users understand what additional fields are available for embedding configuration.
"""

from typing import Dict, Any
import logging
from collections import defaultdict

from services.work_item_service import WorkItemService

logger = logging.getLogger(__name__)


class FieldDiscoveryService:
    """Service to help discover and configure embedding fields for data sources"""
    
    def __init__(self):
        pass
    
    def discover_fields_for_source(self, source_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze work items from a source to discover available additional fields.
        
        Args:
            source_name: Name of the data source to analyze
            sample_size: Number of work items to sample for analysis
        
        Returns:
            Dictionary with field analysis including counts, types, and sample values
        """
        try:
            with WorkItemService() as work_service:
                work_items = work_service.get_work_items_by_source_name(source_name, limit=sample_size)
            
            if not work_items:
                return {
                    "source_name": source_name,
                    "total_work_items_analyzed": 0,
                    "available_fields": {},
                    "message": "No work items found for this data source"
                }
            
            field_analysis = defaultdict(lambda: {
                'count': 0,
                'data_types': set(),
                'sample_values': [],
                'coverage_percentage': 0.0
            })
            
            # Analyze both standard fields and additional fields from all work items
            total_items = len(work_items)
            for work_item in work_items:
                # Include important standard fields that are good for embedding
                # Use Azure DevOps field names to match user expectations
                standard_embedding_fields = {
                    'System.Title': work_item.title,
                    'System.Description': work_item.description,
                    'System.WorkItemType': work_item.work_item_type,
                    'System.State': work_item.state,
                    'Microsoft.VSTS.Common.Priority': work_item.priority,
                    'Microsoft.VSTS.Common.Severity': work_item.severity,
                    'System.AreaPath': work_item.area_path,
                    'System.IterationPath': work_item.iteration_path,
                    'System.AssignedTo': work_item.assigned_to,
                    'System.CreatedBy': work_item.created_by,
                    'System.Tags': ';'.join(work_item.tags) if work_item.tags else ''
                }
                
                # Analyze standard fields that have meaningful text content
                for field_name, field_value in standard_embedding_fields.items():
                    if field_value:  # Only include non-empty values
                        field_info = field_analysis[field_name]
                        field_info['count'] += 1
                        field_info['data_types'].add(type(field_value).__name__)
                        
                        # Collect sample values (limit to 5 unique samples)
                        value_str = str(field_value)[:200]  # Truncate long values
                        if (len(field_info['sample_values']) < 5 and 
                            value_str not in field_info['sample_values']):
                            field_info['sample_values'].append(value_str)
                
                # Analyze additional fields from work items
                if work_item.additional_fields:
                    for field_name, field_value in work_item.additional_fields.items():
                        field_info = field_analysis[field_name]
                        field_info['count'] += 1
                        field_info['data_types'].add(type(field_value).__name__)
                        
                        # Collect sample values (limit to 5 unique samples)
                        value_str = str(field_value)[:200]  # Truncate long values
                        if (len(field_info['sample_values']) < 5 and 
                            value_str not in field_info['sample_values']):
                            field_info['sample_values'].append(value_str)
            
            # Convert to serializable format and calculate coverage
            final_analysis = {}
            for field_name, info in field_analysis.items():
                final_analysis[field_name] = {
                    'count': info['count'],
                    'data_types': list(info['data_types']),
                    'sample_values': info['sample_values'],
                    'coverage_percentage': round((info['count'] / total_items) * 100, 1)
                }
            
            return {
                "source_name": source_name,
                "total_work_items_analyzed": total_items,
                "available_fields": final_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to discover fields for source {source_name}: {e}")
            raise
    
    def suggest_embedding_fields(self, field_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest which fields should be included in embeddings based on analysis.
        
        Args:
            field_analysis: Result from discover_fields_for_source
        
        Returns:
            Dictionary with suggested configuration that can be directly used for data source config
        """
        if not field_analysis.get('available_fields'):
            return {
                "embedding_fields": [],
                "reasoning": "No additional fields found to embed"
            }
        
        suggested_fields = []
        reasoning = []
        
        for field_name, info in field_analysis['available_fields'].items():
            # Criteria for suggestion:
            # 1. Text fields (str type)
            # 2. Not likely to be system/metadata fields
            
            has_text = 'str' in info['data_types']
            not_system_field = not self._is_system_field(field_name)
            
            if has_text and not_system_field:
                suggested_fields.append(field_name)
                reasoning.append(
                    f"{field_name}: {info['coverage_percentage']}% coverage, text field"
                )
            elif not has_text:
                reasoning.append(f"{field_name}: Skipped - not text data")
            elif not not_system_field:
                reasoning.append(f"{field_name}: Skipped - appears to be system/metadata field")
        
        # Return format that can be directly sent to data source configuration
        return {
            "embedding_fields": suggested_fields,
            "reasoning": reasoning,
            "total_fields_analyzed": len(field_analysis['available_fields']),
            "suggested_count": len(suggested_fields)
        }
    
    def _is_system_field(self, field_name: str) -> bool:
        """Check if a field appears to be a system/metadata field that shouldn't be embedded"""
        field_lower = field_name.lower()
        
        # Common patterns for system fields that should be excluded
        system_patterns = [
            'id', 'guid', 'url', 'uri', 'link', 'date', 'time',
            'watermark', 'revision', 'rev', 'version', 'number', 'count', 
            'index', 'order', 'sequence', 'personid', 'commentcount',
            'boardcolumndone', 'extensionmarker', 'column.done',
            'statechangedate', 'activateddate', 'closeddate', 'resolveddate',
            'authorizeddate', 'reviseddate', 'dateopened', 'dateclosed',
            'completeproductversion', 'externalreferencenumber'
        ]
        
        return any(pattern in field_lower for pattern in system_patterns)
    
    def get_embedding_config_template(self, source_name: str) -> Dict[str, Any]:
        """
        Get a template for embedding configuration that can be customized and sent to data source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Template configuration object
        """
        try:
            field_analysis = self.discover_fields_for_source(source_name)
            suggestions = self.suggest_embedding_fields(field_analysis)
            
            return {
                "embedding_field_config": {
                    "embedding_fields": suggestions["embedding_fields"],
                    "enabled": True,
                    "last_updated": None  # Will be set when applied
                },
                "analysis_summary": {
                    "total_fields_found": len(field_analysis.get('available_fields', {})),
                    "suggested_fields": suggestions["suggested_count"],
                    "work_items_analyzed": field_analysis.get('total_work_items_analyzed', 0)
                },
                "available_fields": field_analysis.get('available_fields', {}),
                "suggestion_reasoning": suggestions.get("reasoning", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get embedding config template for {source_name}: {e}")
            raise
