"""
Project Detection Service for Orbis

Simple area path based project detection using basic string matching.
"""

import logging

from engine.schemas import ProjectContext
from utils.constants import AREA_PATH_MAPPINGS, PROJECT_WIKI_REPOS

logger = logging.getLogger(__name__)


class ProjectDetectionService:
    """Simple project detection using area path mapping"""

    def detect_project(self, area_path: str | None = None) -> ProjectContext | None:
        """
        Detect project context from area path using simple string matching.

        Args:
            area_path: Azure DevOps area path

        Returns:
            ProjectContext if project detected, None otherwise
        """
        if not area_path:
            return None

        # Check area path for project mapping
        for area_prefix, project_code in AREA_PATH_MAPPINGS.items():
            if area_path.startswith(area_prefix):
                logger.info(f"🎯 Project {project_code} detected from area path: {area_path}")
                return ProjectContext(
                    project_code=project_code
                )

        logger.info(f"🔍 No project mapping found for area path: {area_path}")
        return None

    def get_project_wiki_repos(self, project_code: str) -> list[str]:
        """Get wiki repositories for a specific project"""
        return PROJECT_WIKI_REPOS.get(project_code, [])
