"""
Application setup utilities for Orbis.
Provides functions to configure CORS and register routers.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orbis_core.utils.logging import get_logger

logger = get_logger()


def setup_cors(app: FastAPI, origins: list[str], credentials: bool) -> None:
    """
    Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        origins: List of allowed origins
        credentials: Whether to allow credentials
    """
    # Credentials with wildcard origins is not allowed; disable credentials in that case
    if credentials and "*" in origins:
        logger.warning("CORS credentials cannot be used with wildcard origins. Disabling credentials.")
        credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS configured with origins: {origins}, credentials: {credentials}")


def register_routers(app: FastAPI, routers: list) -> None:
    """
    Register multiple routers with the FastAPI application.

    Args:
        app: FastAPI application instance
        routers: List of router modules with 'router' attribute
    """
    for router_module in routers:
        app.include_router(router_module.router)
        logger.debug(f"Registered router: {router_module.__name__}")

    logger.info(f"Registered {len(routers)} routers")
