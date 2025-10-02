#!/usr/bin/env python3
"""
Azure DevOps Field Discovery Utility

Standalone script to analyze Azure DevOps work items and suggest optimal fields
for embedding configuration. This is specific to Azure DevOps data sources.

Usage:
    python scripts/azdo_field_discovery.py <source_name> [--sample-size 100]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.models import DataSource
from app.db.session import get_db_session
from core.services.generic_content_service import GenericContentService


class AzureDevOpsFieldDiscovery:
    """Azure DevOps specific field discovery utility"""

    def discover_work_item_fields(self, source_name: str, sample_size: int = 100) -> dict:
        """Analyze work items from an Azure DevOps source to discover available fields"""

        with GenericContentService() as content_service:
            # Get data source
            with get_db_session() as db:
                data_source = db.query(DataSource).filter(DataSource.name == source_name).first()
                if not data_source:
                    return {
                        "error": f"Data source '{source_name}' not found",
                        "available_sources": self._list_available_sources()
                    }

            # Get work item content
            content_items = content_service.get_content_by_data_source(
                data_source_id=data_source.id,
                content_type="work_item",
                limit=sample_size
            )

        if not content_items:
            return {
                "source_name": source_name,
                "total_items_analyzed": 0,
                "available_fields": {},
                "message": "No work items found for this data source"
            }

        field_analysis = defaultdict(lambda: {
            'count': 0,
            'data_types': set(),
            'sample_values': [],
            'coverage_percentage': 0.0
        })

        # Analyze Azure DevOps work item fields
        total_items = len(content_items)
        for content_item in content_items:
            metadata = content_item.content_metadata or {}

            # Standard Azure DevOps fields that might be useful for embedding
            standard_fields = {
                'System.Title': content_item.title,
                'System.Description': content_item.content,
                'System.WorkItemType': metadata.get('work_item_type'),
                'System.State': metadata.get('state'),
                'Microsoft.VSTS.Common.Priority': metadata.get('priority'),
                'Microsoft.VSTS.Common.Severity': metadata.get('severity'),
                'System.AreaPath': metadata.get('area_path'),
                'System.Tags': ';'.join(metadata.get('tags', [])) if metadata.get('tags') else ''
            }

            # Analyze standard fields with text content
            for field_name, field_value in standard_fields.items():
                if field_value and str(field_value).strip():
                    field_info = field_analysis[field_name]
                    field_info['count'] += 1
                    field_info['data_types'].add(type(field_value).__name__)

                    # Collect sample values
                    value_str = str(field_value)[:200]
                    if (len(field_info['sample_values']) < 3 and
                        value_str not in field_info['sample_values']):
                        field_info['sample_values'].append(value_str)

            # Analyze additional_fields (custom Azure DevOps fields)
            additional_fields = metadata.get('additional_fields', {})
            if additional_fields and isinstance(additional_fields, dict):
                for field_name, field_value in additional_fields.items():
                    if field_value and str(field_value).strip():
                        field_info = field_analysis[field_name]
                        field_info['count'] += 1
                        field_info['data_types'].add(type(field_value).__name__)

                        # Collect sample values
                        value_str = str(field_value)[:200]
                        if (len(field_info['sample_values']) < 3 and
                            value_str not in field_info['sample_values']):
                            field_info['sample_values'].append(value_str)

        # Convert to final format
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
            "total_items_analyzed": total_items,
            "available_fields": final_analysis
        }

    def suggest_embedding_fields(self, field_analysis: dict) -> dict:
        """Suggest which Azure DevOps fields should be included in embeddings"""

        if not field_analysis.get('available_fields'):
            return {
                "embedding_fields": [],
                "reasoning": ["No additional fields found to analyze"]
            }

        suggested_fields = []
        reasoning = []

        for field_name, info in field_analysis['available_fields'].items():
            has_text = 'str' in info['data_types']
            not_system_field = not self._is_azdo_system_field(field_name)
            good_coverage = info['coverage_percentage'] >= 20  # At least 20% coverage

            if has_text and not_system_field and good_coverage:
                suggested_fields.append(field_name)
                reasoning.append(
                    f"{field_name}: {info['coverage_percentage']}% coverage, text field"
                )
            elif not has_text:
                reasoning.append(f"{field_name}: Skipped - not text data")
            elif not not_system_field:
                reasoning.append(f"{field_name}: Skipped - system/metadata field")
            elif not good_coverage:
                reasoning.append(f"{field_name}: Skipped - low coverage ({info['coverage_percentage']}%)")

        return {
            "embedding_fields": suggested_fields,
            "reasoning": reasoning,
            "ready_to_use_config": {
                "embedding_fields": suggested_fields
            } if suggested_fields else None
        }

    def _is_azdo_system_field(self, field_name: str) -> bool:
        """Check if field is an Azure DevOps system field that shouldn't be embedded"""

        # Known Azure DevOps system fields to exclude
        system_fields = {
            'System.Id', 'System.Rev', 'System.AuthorizedDate', 'System.RevisedDate',
            'System.CreatedDate', 'System.ChangedDate', 'System.CreatedBy', 'System.ChangedBy',
            'System.Watermark', 'System.CommentCount', 'System.ExternalLinkCount',
            'System.HyperLinkCount', 'System.AttachedFileCount', 'System.NodeName',
            'System.BoardColumn', 'System.BoardColumnDone', 'System.BoardLane'
        }

        if field_name in system_fields:
            return True

        # Check for common system patterns
        field_lower = field_name.lower()
        system_indicators = [
            'id', 'guid', 'uuid', 'date', 'time', 'stamp', 'created', 'modified',
            'updated', 'changed', 'url', 'uri', 'link', 'version', 'revision',
            'watermark', 'count', 'size', 'bytes'
        ]

        for indicator in system_indicators:
            if indicator in field_lower:
                return True

        return False

    def _list_available_sources(self) -> list:
        """List available Azure DevOps data sources"""
        with get_db_session() as db:
            sources = db.query(DataSource).filter(
                DataSource.source_type.like('%azdo%')
            ).all()
            return [source.name for source in sources]


def main():
    parser = argparse.ArgumentParser(
        description="Discover and analyze Azure DevOps work item fields for embedding optimization"
    )
    parser.add_argument("source_name", help="Name of the Azure DevOps data source to analyze")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of work items to analyze (default: 100)")
    parser.add_argument("--output", help="Output file for results (default: stdout)")

    args = parser.parse_args()

    discovery = AzureDevOpsFieldDiscovery()

    print(f"Analyzing Azure DevOps source '{args.source_name}'...")
    print(f"Sample size: {args.sample_size} work items")
    print("-" * 50)

    # Discover fields
    analysis = discovery.discover_work_item_fields(args.source_name, args.sample_size)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        if "available_sources" in analysis:
            print(f"Available sources: {', '.join(analysis['available_sources'])}")
        sys.exit(1)

    # Get suggestions
    suggestions = discovery.suggest_embedding_fields(analysis)

    # Prepare results
    results = {
        "analysis": analysis,
        "suggestions": suggestions,
        "summary": {
            "total_fields_found": len(analysis.get('available_fields', {})),
            "fields_suggested": len(suggestions['embedding_fields']),
            "total_work_items": analysis.get('total_items_analyzed', 0)
        }
    }

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print("\nFIELD ANALYSIS:")
        print("=" * 50)
        for field_name, info in analysis.get('available_fields', {}).items():
            print(f"\n{field_name}:")
            print(f"  Coverage: {info['coverage_percentage']}% ({info['count']} items)")
            print(f"  Data types: {', '.join(info['data_types'])}")
            if info['sample_values']:
                print(f"  Sample: {info['sample_values'][0][:100]}...")

        print("\nSUGGESTIONS:")
        print("=" * 50)
        if suggestions['embedding_fields']:
            print("Recommended fields for embedding:")
            for field in suggestions['embedding_fields']:
                print(f"  - {field}")

            print("\nReady-to-use configuration:")
            print(json.dumps(suggestions['ready_to_use_config'], indent=2))

            print("\nApply with:")
            print(f"PUT /datasources/{args.source_name}/embedding-config")
        else:
            print("No additional fields recommended for embedding.")

        print("\nReasoning:")
        for reason in suggestions['reasoning']:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
