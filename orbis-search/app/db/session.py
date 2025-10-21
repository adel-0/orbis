"""
Database management for Orbis Search - uses orbis-core DatabaseManager.
"""

from orbis_core.database import DatabaseManager, get_db_session
from .models import Base

# Configure orbis-core DatabaseManager to use our models
DatabaseManager.set_base(Base)

# Re-export for backward compatibility
__all__ = ["DatabaseManager", "get_db_session"]
