from collections.abc import Mapping
import base64
import json

import httpx
import openai
from httpx import Timeout

from dify_plugin.errors.model import InvokeAuthorizationError, InvokeBadRequestError, InvokeConnectionError, InvokeError, InvokeRateLimitError, InvokeServerUnavailableError


CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class CodexAuthenticationError(Exception):
    pass


class _CommonChatGPTCodex:
    def _extract_chatgpt_account_id(self, token: str | None) -> str | None:
        if not token:
            return None

        parts = token.split(".")
        if len(parts) != 3:
            return None

        try:
            payload = parts[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
        except Exception:
            return None

        if claims.get("chatgpt_account_id"):
            return claims["chatgpt_account_id"]

        auth_claims = claims.get("https://api.openai.com/auth") or {}
        if auth_claims.get("chatgpt_account_id"):
            return auth_claims["chatgpt_account_id"]

        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict) and first.get("id"):
                return first["id"]

        return None

    def _get_codex_account_id(self, credentials: Mapping, access_token: str | None = None) -> str | None:
        account_id = credentials.get("chatgpt_account_id")
        if account_id:
            return str(account_id)

        token = access_token or credentials.get("chatgpt_access_token")
        return self._extract_chatgpt_account_id(str(token) if token else None)

    def _to_credential_kwargs(self, credentials: Mapping) -> dict:
        """
        Transform credentials to kwargs for model instance

        :param credentials:
        :return:
        """
        access_token = str(credentials["chatgpt_access_token"])
        api_base = str(credentials.get("codex_api_base") or CODEX_API_BASE).rstrip("/")
        account_id = self._get_codex_account_id(credentials, access_token)

        credentials_kwargs = {
            "api_key": access_token,
            "base_url": api_base,
            "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }

        if account_id:
            credentials_kwargs["default_headers"] = {
                "ChatGPT-Account-Id": account_id,
            }

        return credentials_kwargs

    def _refresh_access_token(self, credentials: Mapping) -> dict:
        refresh_token = credentials.get("chatgpt_refresh_token")
        if not refresh_token:
            raise CodexAuthenticationError("Missing ChatGPT Codex refresh token.")

        response = httpx.post(
            f"{CODEX_OAUTH_ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": str(refresh_token),
                "client_id": CODEX_CLIENT_ID,
            },
            timeout=30.0,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CodexAuthenticationError(f"Failed to refresh ChatGPT Codex token: {exc}") from exc

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise CodexAuthenticationError("ChatGPT Codex refresh response did not include an access token.")

        return {
            "access_token": access_token,
            "refresh_token": payload.get("refresh_token") or str(refresh_token),
            "account_id": self._extract_chatgpt_account_id(payload.get("id_token"))
            or self._extract_chatgpt_account_id(access_token),
        }

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [openai.APIConnectionError, openai.APITimeoutError],
            InvokeServerUnavailableError: [openai.InternalServerError],
            InvokeRateLimitError: [openai.RateLimitError],
            InvokeAuthorizationError: [openai.AuthenticationError, openai.PermissionDeniedError, CodexAuthenticationError],
            InvokeBadRequestError: [
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ],
        }
