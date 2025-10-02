"""
Shared authentication module for Azure DevOps integrators and clients.
Provides common authentication functionality for PAT and OAuth2.
"""

import base64
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class AzureDevOpsAuthMixin:
    """Mixin class providing Azure DevOps authentication functionality"""

    def __init__(self):
        # These attributes should be set by the inheriting class
        self.use_oauth2: bool = False
        self.pat: str | None = None
        self.client_id: str | None = None
        self.client_secret: str | None = None
        self.tenant_id: str | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    async def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for Azure DevOps API calls"""
        if self.use_oauth2:
            if not self._access_token or self._is_token_expired():
                await self._refresh_oauth_token()
            return {"Authorization": f"Bearer {self._access_token}"}
        else:
            if not self.pat:
                raise ValueError("PAT is required when not using OAuth2")
            encoded_pat = base64.b64encode(f":{self.pat}".encode()).decode()
            return {"Authorization": f"Basic {encoded_pat}"}

    def _is_token_expired(self) -> bool:
        """Check if the OAuth2 token is expired"""
        if not self._token_expires_at:
            return True
        return datetime.now(UTC) >= self._token_expires_at

    async def _refresh_oauth_token(self):
        """Refresh OAuth2 access token using MSAL"""
        try:
            import msal
        except ImportError as e:
            raise ImportError("msal package is required for OAuth2 authentication. Install with: pip install msal") from e

        if not all([self.client_id, self.client_secret, self.tenant_id]):
            raise ValueError("OAuth2 requires client_id, client_secret, and tenant_id")

        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret
        )

        # Azure DevOps requires this specific scope for client credential flow
        result = app.acquire_token_for_client(scopes=["https://app.vssps.visualstudio.com/.default"])

        if "access_token" in result:
            self._access_token = result["access_token"]
            # Set expiration time (tokens typically expire in 1 hour)
            if "expires_in" in result:
                from datetime import timedelta
                self._token_expires_at = datetime.now(UTC) + timedelta(seconds=result["expires_in"] - 60)  # 1 minute buffer
            else:
                # Default to 50 minutes if not specified
                from datetime import timedelta
                self._token_expires_at = datetime.now(UTC) + timedelta(minutes=50)

            logger.debug("Successfully refreshed OAuth2 token")
        else:
            error = result.get("error_description", "Unknown OAuth error")
            raise Exception(f"OAuth token refresh failed: {error}")
