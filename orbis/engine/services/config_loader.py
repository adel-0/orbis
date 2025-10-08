"""
Configuration loader for data source instances.
Loads YAML configuration files and creates data source instances.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from infrastructure.data_processing.data_source_service import DataSourceService

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads data source configurations from YAML files"""

    def __init__(self, config_dir: str = "config/connector_instances"):
        self.config_dir = Path(config_dir)
        self.data_source_service = DataSourceService()

    def load_all_instances(self) -> int:
        """
        Load all data source instances from YAML files and clean up orphaned database entries.

        Returns:
            Number of instances loaded successfully
        """
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return 0

        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))

        if not yaml_files:
            logger.info(f"No YAML configuration files found in {self.config_dir}")
            return 0

        loaded_count = 0
        yaml_source_names = set()

        with get_db_session() as db:
            # First pass: Load all YAML configurations
            for yaml_file in yaml_files:
                try:
                    logger.info(f"Loading data source config from {yaml_file.name}")

                    with open(yaml_file, encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    if not config_data:
                        logger.warning(f"Empty or invalid YAML file: {yaml_file.name}")
                        continue

                    # Validate required fields
                    if not all(key in config_data for key in ['name', 'source_type', 'config']):
                        logger.error(f"Invalid config in {yaml_file.name}: missing required fields (name, source_type, config)")
                        continue

                    # Track valid source names from YAML files
                    yaml_source_names.add(config_data['name'])

                    # Create or update data source
                    self._create_or_update_data_source(db, config_data, yaml_file.name)
                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to load config from {yaml_file.name}: {e}")
                    continue

            # Second pass: Clean up orphaned database entries
            self._cleanup_orphaned_sources(db, yaml_source_names)

        logger.info(f"Successfully loaded {loaded_count} data source instances")
        return loaded_count

    def _create_or_update_data_source(self, db: Session, config_data: dict[str, Any], filename: str):
        """Create or update a data source from config data"""
        try:
            name = config_data['name']
            source_type = config_data['source_type']
            config = config_data['config']

            # Optional fields
            enabled = config_data.get('enabled', True)
            context_tags = config_data.get('context_tags', [])
            weight = config_data.get('weight', 1.0)
            # Convert weight to priority (keeping existing database field)
            priority = int(weight) if weight >= 1 else 1

            # Check if data source already exists
            existing_source = self.data_source_service.get_data_source(name)

            if existing_source:
                logger.info(f"Updating existing data source: {name}")
                self.data_source_service.update_data_source(
                    name=name,
                    source_type=source_type,
                    config=config,
                    enabled=enabled,
                    context_tags=context_tags,
                    priority=priority
                )
            else:
                logger.info(f"Creating new data source: {name}")
                self.data_source_service.create_data_source(
                    name=name,
                    source_type=source_type,
                    config=config,
                    enabled=enabled,
                    context_tags=context_tags,
                    priority=priority
                )

        except Exception as e:
            logger.error(f"Failed to create/update data source from {filename}: {e}")
            raise

    def _cleanup_orphaned_sources(self, db: Session, yaml_source_names: set[str]):
        """Remove database entries that no longer have corresponding YAML files"""
        try:
            # Get all existing data sources from database
            all_db_sources = self.data_source_service.get_all_data_sources()
            db_source_names = {source.name for source in all_db_sources}

            # Find orphaned sources (in database but not in YAML files)
            orphaned_sources = db_source_names - yaml_source_names

            if not orphaned_sources:
                logger.debug("No orphaned data sources found")
                return

            logger.info(f"Found {len(orphaned_sources)} orphaned data sources: {list(orphaned_sources)}")

            # Remove each orphaned source
            for source_name in orphaned_sources:
                try:
                    self.data_source_service.delete_data_source(source_name)
                    logger.info(f"Removed orphaned data source: {source_name}")
                except Exception as e:
                    logger.error(f"Failed to remove orphaned data source {source_name}: {e}")

        except Exception as e:
            logger.error(f"Error during orphaned source cleanup: {e}")
