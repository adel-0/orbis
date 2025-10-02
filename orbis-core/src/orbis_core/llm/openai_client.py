"""
Shared OpenAI client service for Orbis applications.
Provides a singleton Azure OpenAI client to avoid multiple client instantiations.
"""

import logging
from functools import lru_cache

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_azure_openai_client(api_key: str, api_version: str, azure_endpoint: str) -> AzureOpenAI:
    """
    Get a shared Azure OpenAI client instance.

    Args:
        api_key: Azure OpenAI API key
        api_version: Azure OpenAI API version
        azure_endpoint: Azure OpenAI endpoint URL

    Returns:
        Configured Azure OpenAI client
    """
    logger.info("Creating shared Azure OpenAI client")

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )


class OpenAIClientService:
    """Service wrapper for shared OpenAI client access"""

    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, deployment_name: str | None = None):
        """
        Initialize OpenAI client service.

        Args:
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL
            deployment_name: Optional deployment name
        """
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self._client = None

    @property
    def client(self) -> AzureOpenAI:
        """Get the shared OpenAI client"""
        if self._client is None:
            self._client = get_azure_openai_client(self.api_key, self.api_version, self.azure_endpoint)
        return self._client
