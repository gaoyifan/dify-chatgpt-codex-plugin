import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from models.common_chatgpt_codex import CODEX_API_BASE, CODEX_CLIENT_ID, CodexAuthenticationError, _CommonChatGPTCodex
from models.llm.llm import ChatGPTCodexLargeLanguageModel


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


class DummyCommon(_CommonChatGPTCodex):
    pass


class ChatGPTCodexCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.common = DummyCommon()

    def test_extract_chatgpt_account_id_from_direct_claim(self) -> None:
        token = _jwt({"chatgpt_account_id": "acct_direct"})
        self.assertEqual(self.common._extract_chatgpt_account_id(token), "acct_direct")

    def test_extract_chatgpt_account_id_from_nested_openai_claim(self) -> None:
        token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"}})
        self.assertEqual(self.common._extract_chatgpt_account_id(token), "acct_nested")

    def test_extract_chatgpt_account_id_from_organizations_fallback(self) -> None:
        token = _jwt({"organizations": [{"id": "acct_org"}]})
        self.assertEqual(self.common._extract_chatgpt_account_id(token), "acct_org")

    def test_to_credential_kwargs_uses_default_base_and_account_header(self) -> None:
        kwargs = self.common._to_credential_kwargs(
            {
                "chatgpt_access_token": _jwt({"chatgpt_account_id": "acct_from_token"}),
                "chatgpt_refresh_token": "refresh",
            }
        )

        self.assertEqual(kwargs["api_key"], _jwt({"chatgpt_account_id": "acct_from_token"}))
        self.assertEqual(kwargs["base_url"], CODEX_API_BASE)
        self.assertEqual(kwargs["default_headers"], {"ChatGPT-Account-Id": "acct_from_token"})

    @patch("models.common_chatgpt_codex.httpx.post")
    def test_refresh_access_token_uses_codex_client_id_and_extracts_account_id(self, post: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "access_token": _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_refresh"}}),
            "refresh_token": "refresh_new",
            "id_token": _jwt({"chatgpt_account_id": "acct_id"}),
        }
        post.return_value = response

        refreshed = self.common._refresh_access_token({"chatgpt_refresh_token": "refresh_old"})

        self.assertEqual(refreshed["access_token"], _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_refresh"}}))
        self.assertEqual(refreshed["refresh_token"], "refresh_new")
        self.assertEqual(refreshed["account_id"], "acct_id")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["data"]["client_id"], CODEX_CLIENT_ID)
        self.assertEqual(post.call_args.kwargs["data"]["grant_type"], "refresh_token")


class ChatGPTCodexLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = object.__new__(ChatGPTCodexLargeLanguageModel)

    def test_with_codex_client_refreshes_once_after_auth_error(self) -> None:
        initial_client = object()
        refreshed_client = object()

        with (
            patch("models.llm.llm.OpenAI", side_effect=[initial_client, refreshed_client]) as openai_client,
            patch.object(self.llm, "_to_credential_kwargs", side_effect=[{"api_key": "old"}, {"api_key": "new"}]),
            patch.object(
                self.llm,
                "_refresh_access_token",
                return_value={"access_token": "new_access", "refresh_token": "new_refresh", "account_id": "acct_new"},
            ) as refresh_token,
        ):
            calls = []

            def func(client, current_credentials):
                calls.append((client, dict(current_credentials)))
                if len(calls) == 1:
                    raise CodexAuthenticationError("expired")
                return current_credentials

            result = self.llm._with_codex_client(
                {
                    "chatgpt_access_token": "old_access",
                    "chatgpt_refresh_token": "old_refresh",
                },
                func,
            )

        self.assertEqual(openai_client.call_count, 2)
        refresh_token.assert_called_once()
        self.assertEqual(calls[0][0], initial_client)
        self.assertEqual(calls[1][0], refreshed_client)
        self.assertEqual(result["chatgpt_access_token"], "new_access")
        self.assertEqual(result["chatgpt_refresh_token"], "new_refresh")
        self.assertEqual(result["chatgpt_account_id"], "acct_new")

    def test_build_responses_api_params_maps_tokens_reasoning_and_json_schema(self) -> None:
        params = self.llm._build_responses_api_params(
            {
                "max_tokens": 512,
                "reasoning_effort": "high",
                "response_format": "json_schema",
                "json_schema": {
                    "name": "structured_answer",
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    "strict": True,
                },
                "verbosity": "low",
            },
            stop=["END"],
            user="user-123",
        )

        self.assertEqual(params["max_output_tokens"], 512)
        self.assertEqual(params["reasoning"], {"effort": "high"})
        self.assertEqual(params["text"]["format"]["type"], "json_schema")
        self.assertEqual(params["text"]["format"]["name"], "structured_answer")
        self.assertTrue(params["text"]["format"]["strict"])
        self.assertEqual(params["stop"], ["END"])
        self.assertEqual(params["user"], "user-123")
        self.assertEqual(params["verbosity"], "low")

    def test_validate_credentials_uses_responses_api(self) -> None:
        responses_client = SimpleNamespace(create=MagicMock())
        client = SimpleNamespace(responses=responses_client)

        with patch.object(self.llm, "_with_codex_client", side_effect=lambda credentials, func: func(client, dict(credentials))):
            self.llm.validate_credentials(
                "gpt-5.3-codex",
                {
                    "chatgpt_access_token": "token",
                    "chatgpt_refresh_token": "refresh",
                },
            )

        responses_client.create.assert_called_once_with(model="gpt-5.3-codex", input="ping", max_output_tokens=20)


class ChatGPTCodexModelListTests(unittest.TestCase):
    def test_position_file_matches_opencode_allowed_models(self) -> None:
        with open("models/llm/_position.yaml", encoding="utf-8") as f:
            models = [line.strip()[2:] for line in f if line.strip()]

        self.assertEqual(
            models,
            [
                "gpt-5.1-codex",
                "gpt-5.1-codex-max",
                "gpt-5.1-codex-mini",
                "gpt-5.2",
                "gpt-5.2-codex",
                "gpt-5.3-codex",
                "gpt-5.4",
                "gpt-5.4-mini",
            ],
        )


if __name__ == "__main__":
    unittest.main()
