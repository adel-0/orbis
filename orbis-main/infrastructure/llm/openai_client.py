"""
Shared OpenAI client service for OnCall Copilot.
Provides a singleton Azure OpenAI client to avoid multiple client instantiations.
"""

import logging
from functools import lru_cache

from openai import AzureOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_azure_openai_client() -> AzureOpenAI:
    """
    Get a shared Azure OpenAI client instance.

    Returns:
        Configured Azure OpenAI client
    """
    logger.info("Creating shared Azure OpenAI client")

    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    )


class OpenAIClientService:
    """Service wrapper for shared OpenAI client access"""

    def __init__(self):
        self._client = None

    @property
    def client(self) -> AzureOpenAI:
        """Get the shared OpenAI client"""
        if self._client is None:
            self._client = get_azure_openai_client()
        return self._client

    @property
    def deployment_name(self) -> str:
        """Get the deployment name from settings"""
        return settings.AZURE_OPENAI_DEPLOYMENT_NAME
