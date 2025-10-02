"""
Database management for OnCall Copilot.
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and initialization"""

    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    @classmethod
    def init_database(cls, database_url: Optional[str] = None) -> None:
        """Initialize database connection and create tables"""
        if database_url is None:
            # Honor DATABASE_URL if provided via environment; otherwise use default
            env_url = os.getenv("DATABASE_URL")
            if env_url:
                database_url = env_url
            else:
                # Default to SQLite database in data directory
                os.makedirs("data/database", exist_ok=True)
                database_url = "sqlite:///data/database/oncall_copilot.db"

        logger.info(f"Initializing database: {database_url}")

        # Ensure directory exists for file-backed SQLite URLs
        if database_url.startswith("sqlite"):
            # Example formats: sqlite:///path/to.db, sqlite:////absolute/path.db, sqlite:///:memory:
            if ":memory:" not in database_url:
                # Extract the path after the sqlite:/// (handle 3 or 4 slashes)
                try:
                    # split on 'sqlite:///' and also handle 'sqlite:////'
                    path_part = database_url.split("sqlite:///")[-1]
                    # For absolute windows-like paths (e.g., C:%5Cpath) leave as-is; just ensure parent exists
                    db_dir = os.path.dirname(path_part)
                    if db_dir:
                        os.makedirs(db_dir, exist_ok=True)
                except Exception as dir_exc:
                    logger.warning(f"Failed to ensure SQLite directory exists: {dir_exc}")

        # Create engine
        if database_url.startswith("sqlite"):
            # SQLite-specific configuration (file-backed DB): avoid StaticPool to reduce lock contention
            cls._engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
            # Apply pragmas for better concurrency
            try:
                with cls._engine.connect() as conn:
                    conn.execute(text("PRAGMA journal_mode=WAL"))
                    conn.execute(text("PRAGMA synchronous=NORMAL"))
            except Exception as pragma_exc:
                logger.warning(f"Failed to set SQLite PRAGMAs: {pragma_exc}")
        else:
            cls._engine = create_engine(database_url, echo=False)

        # Create session factory
        cls._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=cls._engine,
        )

        # Create all tables
        Base.metadata.create_all(bind=cls._engine)
        logger.info("Database tables created successfully")

    @classmethod
    def get_engine(cls) -> Engine:
        """Get database engine"""
        if cls._engine is None:
            cls.init_database()
        return cls._engine

    @classmethod
    def get_session_factory(cls) -> sessionmaker:
        """Get session factory"""
        if cls._session_factory is None:
            cls.init_database()
        return cls._session_factory

    @classmethod
    def get_database_info(cls) -> dict:
        """Get database information"""
        if cls._engine is None:
            return {"status": "not_initialized"}

        try:
            # Get database URL (mask sensitive info)
            url_str = str(cls._engine.url)
            if "sqlite" in url_str:
                db_type = "SQLite"
                db_path = url_str.replace("sqlite:///", "")
                db_size = "Unknown"
                if os.path.exists(db_path):
                    db_size = f"{os.path.getsize(db_path) / 1024:.1f} KB"

                return {
                    "type": db_type,
                    "path": db_path,
                    "size": db_size,
                    "status": "connected",
                }
            else:
                return {
                    "type": "Other",
                    "url": url_str,
                    "status": "connected",
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


def get_db_session() -> Session:
    """Get database session"""
    session_factory = DatabaseManager.get_session_factory()
    return session_factory()
