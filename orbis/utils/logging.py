"""
Centralized logging utility for OnCall Copilot.
Provides consistent logging configuration across the application.
"""

import logging
import sys


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, defaults to calling module's __name__

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')

    return logging.getLogger(name)


def setup_logging(level: str = "INFO", format_string: str | None = None) -> None:
    """
    Set up application-wide logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Pre-configured logger for immediate use
logger = logging.getLogger(__name__)
