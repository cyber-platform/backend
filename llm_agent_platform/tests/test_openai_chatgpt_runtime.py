import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.config import OPENAI_CHATGPT_ORIGINATOR
from llm_agent_platform.services.openai_chatgpt_api_keys import (
    OpenAIChatGPTApiKeyRegistryService,
)
from llm_agent_platform.services.provider_usage_limits import (
    OpenAIChatGptUsageLimitsAdapter,
)

SECRETS_TEST_ROOT = Path("secrets_test")


class FakeResponse:
    def __init__(
        self, status_code: int, payload: dict | None = None, text: str | None = None
    ):
        self.status_code = status_code
        self._payload = payload
        self.text = (
            text if text is not None else json.dumps(payload or {}, ensure_ascii=False)
        )
        self.headers = {"content-type": "application/json"}

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class FakeStreamResponse:
    def __init__(self, status_code: int, lines: list[str], text: str = ""):
        self.status_code = status_code
        self._lines = lines
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.text.encode("utf-8")

    def iter_lines(self):
        for line in self._lines:
            yield line


class FakeHttpClient:
    def __init__(
        self,
        *,
        post_responses: list[FakeResponse] | None = None,
        stream_responses: list[FakeStreamResponse] | None = None,
        get_response: FakeResponse | None = None,
    ):
        self.post_responses = list(post_responses or [])
        self.stream_responses = list(stream_responses or [])
        self.get_response = get_response
        self.post_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def post(self, url, headers=None, json=None, content=None):
        self.post_calls.append(
            {"url": url, "headers": headers or {}, "json": json, "content": content}
        )
        if not self.post_responses:
            raise AssertionError("Unexpected POST call")
        return self.post_responses.pop(0)

    def stream(self, method, url, headers=None, json=None):
        self.stream_calls.append(
            {"method": method, "url": url, "headers": headers or {}, "json": json}
        )
        if not self.stream_responses:
            raise AssertionError("Unexpected stream call")
        return self.stream_responses.pop(0)

    def get(self, url, headers=None):
        self.get_calls.append({"url": url, "headers": headers or {}})
        if self.get_response is None:
            raise AssertionError("Unexpected GET call")
        return self.get_response


@contextmanager
def _tmp_state_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class OpenAIChatGPTRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _openai_accounts_config(
        credentials_path: str, *, mode: str = "single", include_team_b: bool = False
    ) -> dict:
        payload = {
            "mode": mode,
            "active_account": "acct-1",
            "all_accounts": ["acct-1"] if mode == "single" else ["acct-1", "acct-2"],
            "accounts": {
                "acct-1": {"credentials_path": credentials_path},
            },
            "rotation_policy": {
                "rate_limit_threshold": 2,
                "quota_exhausted_threshold": 2,
                "rate_limit_cooldown_seconds": 5,
            },
            "quota_scope": "per_provider",
            "model_quota_resets": {"default": "07:00:00"},
            "groups": {
                "team-a": {
                    "accounts": ["acct-1"],
                    "models": ["gpt-5.4", "gpt-5.4-mini"],
                }
            },
        }
        if mode == "rounding":
            payload["accounts"]["acct-2"] = {"credentials_path": credentials_path}
            payload["groups"]["team-a"]["accounts"] = ["acct-1", "acct-2"]
        if include_team_b:
            if mode != "rounding":
                payload["accounts"]["acct-2"] = {"credentials_path": credentials_path}
            else:
                payload["groups"]["team-a"]["accounts"] = ["acct-1"]
            payload["groups"]["team-b"] = {
                "accounts": ["acct-2"],
                "models": ["gpt-5.4-mini"],
            }
        return payload

    @contextmanager
    def _patched_paths(self, tmp_dir: Path, openai_config_path: Path):
        registry_path = tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
        with (
            patch("llm_agent_platform.services.account_router.STATE_DIR", str(tmp_dir)),
            patch(
                "llm_agent_platform.services.account_state_store.STATE_DIR",
                str(tmp_dir),
            ),
            patch(
                "llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)
            ),
            patch("llm_agent_platform.config.STATE_DIR", str(tmp_dir)),
            patch(
                "llm_agent_platform.services.account_router.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(openai_config_path),
            ),
            patch(
                "llm_agent_platform.auth.credentials.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(openai_config_path),
            ),
            patch(
                "llm_agent_platform.config.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                str(registry_path),
            ),
            patch(
                "llm_agent_platform.services.openai_chatgpt_api_keys.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                str(registry_path),
            ),
        ):
            yield

    @staticmethod
    def _auth_headers(raw_api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {raw_api_key}"}

    @staticmethod
    def _sse_json_events(payload: str) -> list[dict]:
        events: list[dict] = []
        for line in payload.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                continue
            events.append(json.loads(raw))
        return events

    def test_non_stream_single_mode_uses_private_backend_and_optional_account_id(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                post_responses=[
                    FakeResponse(
                        200,
                        {
                            "id": "chatcmpl-openai-1",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "hello from openai-chatgpt",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        },
                    )
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="runtime-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=self._auth_headers(api_key),
                )

            body = json.loads(response.data.decode("utf-8"))
            usage_state = json.loads(
                (
                    tmp_dir
                    / "openai-chatgpt"
                    / "usage"
                    / "accounts"
                    / "acct-1"
                    / "limits.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["model"], "gpt-5.4")
        self.assertEqual(
            body["choices"][0]["message"]["content"], "hello from openai-chatgpt"
        )
        self.assertEqual(
            fake_client.post_calls[0]["url"],
            "https://chatgpt.com/backend-api/codex/responses",
        )
        self.assertEqual(
            fake_client.post_calls[0]["headers"]["originator"],
            OPENAI_CHATGPT_ORIGINATOR,
        )
        self.assertNotIn("ChatGPT-Account-Id", fake_client.post_calls[0]["headers"])
        self.assertEqual(usage_state["provider_id"], "openai-chatgpt")

    def test_auth_failure_triggers_single_forced_refresh_retry(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                post_responses=[
                    FakeResponse(401, {"error": "unauthorized"}),
                    FakeResponse(
                        200,
                        {
                            "id": "chatcmpl-openai-2",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "after refresh",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 2,
                                "total_tokens": 4,
                            },
                        },
                    ),
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.OpenAIChatGPTOAuthManager.force_refresh",
                    return_value={
                        "access_token": "token-456",
                        "token_type": "Bearer",
                        "account_id": "acct-xyz",
                    },
                ) as mock_refresh,
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="refresh-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "refresh me"}],
                    },
                    headers=self._auth_headers(api_key),
                )

            body = json.loads(response.data.decode("utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "after refresh")
        self.assertEqual(mock_refresh.call_count, 1)
        self.assertEqual(len(fake_client.post_calls), 2)
        self.assertEqual(
            fake_client.post_calls[1]["headers"]["Authorization"], "Bearer token-456"
        )
        self.assertEqual(
            fake_client.post_calls[1]["headers"]["ChatGPT-Account-Id"], "acct-xyz"
        )

    def test_stream_route_emits_openai_chunks_and_usage(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "account_id": "acct-1",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                stream_responses=[
                    FakeStreamResponse(
                        200,
                        [
                            'data: {"type":"response.output_text.delta","delta":"hel"}',
                            'data: {"type":"response.output_text.delta","delta":"lo"}',
                            'data: {"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"hello"}]}}',
                            'data: {"type":"response.done","response":{"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}',
                        ],
                    )
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="stream-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "stream please"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    headers=self._auth_headers(api_key),
                )
                payload = response.data.decode("utf-8")
            usage_state = json.loads(
                (
                    tmp_dir
                    / "openai-chatgpt"
                    / "usage"
                    / "accounts"
                    / "acct-1"
                    / "limits.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "hel"', payload)
        self.assertIn('"content": "lo"', payload)
        self.assertIn(
            '"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}',
            payload,
        )
        self.assertIn("data: [DONE]", payload)
        self.assertEqual(usage_state["metadata"]["usage"]["total_tokens"], 5)

    def test_stream_non_200_is_mapped_to_openai_error_event(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                stream_responses=[
                    FakeStreamResponse(
                        503,
                        [],
                        text='{"error":"backend unavailable"}',
                    )
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="stream-error-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "stream please"}],
                        "stream": True,
                    },
                    headers=self._auth_headers(api_key),
                )
                payload = response.data.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "upstream_error"', payload)
        self.assertIn("backend unavailable", payload)
        self.assertNotIn("data: [DONE]", payload)

    def test_stream_tool_call_deltas_do_not_duplicate_final_arguments(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "account_id": "acct-1",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                stream_responses=[
                    FakeStreamResponse(
                        200,
                        [
                            'data: {"type":"response.function_call_arguments.delta","call_id":"call_1","name":"read","index":0,"delta":"{\\"path\\""}',
                            'data: {"type":"response.function_call_arguments.delta","call_id":"call_1","name":"read","index":0,"delta":":\\"/tmp/a.txt\\"}"}',
                            'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"read","arguments":"{\\"path\\":\\"/tmp/a.txt\\"}"}}',
                            'data: {"type":"response.done","response":{"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}',
                        ],
                    )
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="tool-call-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "read file"}],
                        "stream": True,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "description": "Read file",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}},
                                    },
                                },
                            }
                        ],
                    },
                    headers=self._auth_headers(api_key),
                )
                payload = response.data.decode("utf-8")

        events = self._sse_json_events(payload)
        tool_events = [
            event
            for event in events
            if event.get("choices")
            and event["choices"][0].get("delta", {}).get("tool_calls")
        ]
        finish_events = [
            event
            for event in events
            if event.get("choices")
            and event["choices"][0].get("finish_reason") == "tool_calls"
        ]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(tool_events), 2)
        self.assertEqual(
            [
                item["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for item in tool_events
            ],
            ['{"path"', ':"/tmp/a.txt"}'],
        )
        self.assertEqual(len(finish_events), 1)
        self.assertIn("data: [DONE]", payload)

    def test_stream_reasoning_is_exposed_as_reasoning_text(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "account_id": "acct-1",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            fake_client = FakeHttpClient(
                stream_responses=[
                    FakeStreamResponse(
                        200,
                        [
                            'data: {"type":"response.reasoning_text.delta","delta":"thinking"}',
                            'data: {"type":"response.output_text.delta","delta":"done"}',
                            'data: {"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}}',
                        ],
                    )
                ]
            )

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.api.openai.providers.openai_chatgpt.get_http_client",
                    return_value=fake_client,
                ),
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="reasoning-test"
                )["raw_api_key"]
                response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "think"}],
                        "stream": True,
                    },
                    headers=self._auth_headers(api_key),
                )
                payload = response.data.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn('"reasoning_text": "thinking"', payload)
        self.assertNotIn('"reasoning_content"', payload)
        self.assertIn('"content": "done"', payload)

    def test_group_models_route_uses_provider_local_subset(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path), mode="rounding"),
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="models-test"
                )["raw_api_key"]
                response = self.client.get(
                    "/openai-chatgpt/team-a/v1/models",
                    headers=self._auth_headers(api_key),
                )

        body = json.loads(response.data.decode("utf-8"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["id"] for item in body["data"]}, {"gpt-5.4", "gpt-5.4-mini"}
        )

    def test_public_models_route_rejects_missing_or_malformed_bearer_token(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path, self._openai_accounts_config(str(creds_path))
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                missing_response = self.client.get("/openai-chatgpt/v1/models")
                malformed_response = self.client.get(
                    "/openai-chatgpt/v1/models",
                    headers={"Authorization": "Token not-bearer"},
                )

        missing_payload = json.loads(missing_response.data.decode("utf-8"))
        malformed_payload = json.loads(malformed_response.data.decode("utf-8"))

        self.assertEqual(missing_response.status_code, 401)
        self.assertEqual(malformed_response.status_code, 401)
        self.assertEqual(missing_payload["error"]["type"], "authentication_error")
        self.assertEqual(missing_payload["error"]["code"], "invalid_api_key")
        self.assertEqual(malformed_payload["error"]["type"], "authentication_error")
        self.assertEqual(malformed_payload["error"]["code"], "invalid_api_key")

    def test_public_routes_reject_revoked_or_out_of_scope_platform_key(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(
                    str(creds_path), mode="rounding", include_team_b=True
                ),
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                service = OpenAIChatGPTApiKeyRegistryService()
                team_a_key = service.create_key(group_id="team-a", label="team-a")[
                    "raw_api_key"
                ]
                team_b_key = service.create_key(group_id="team-b", label="team-b")[
                    "raw_api_key"
                ]
                revoked_key = service.create_key(group_id="team-a", label="revoked")[
                    "raw_api_key"
                ]
                revoked_record = service.lookup_active_key(
                    revoked_key, group_id="team-a"
                )
                if revoked_record is None:
                    self.fail("revoked_record must not be None")
                service.revoke_key(revoked_record["key_id"])

                authorized_response = self.client.get(
                    "/openai-chatgpt/v1/models",
                    headers=self._auth_headers(team_a_key),
                )
                wrong_scope_response = self.client.get(
                    "/openai-chatgpt/team-a/v1/models",
                    headers=self._auth_headers(team_b_key),
                )
                revoked_response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=self._auth_headers(revoked_key),
                )

        authorized_payload = json.loads(authorized_response.data.decode("utf-8"))
        wrong_scope_payload = json.loads(wrong_scope_response.data.decode("utf-8"))
        revoked_payload = json.loads(revoked_response.data.decode("utf-8"))

        self.assertEqual(authorized_response.status_code, 200)
        self.assertEqual(
            {item["id"] for item in authorized_payload["data"]},
            {"gpt-5.4", "gpt-5.4-mini"},
        )
        self.assertEqual(wrong_scope_response.status_code, 401)
        self.assertEqual(revoked_response.status_code, 401)
        self.assertEqual(wrong_scope_payload["error"]["code"], "invalid_api_key")
        self.assertEqual(revoked_payload["error"]["code"], "invalid_api_key")

    def test_usage_adapter_normalizes_wham_windows(self):
        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            self._write_json(
                creds_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "access_token": "token-123",
                    "refresh_token": "refresh-123",
                    "token_type": "Bearer",
                    "account_id": "acct-1",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )

            fake_client = FakeHttpClient(
                get_response=FakeResponse(
                    200,
                    {
                        "plan_type": "chatgpt-plus",
                        "rate_limit": {
                            "primary_window": {
                                "used_percent": 42,
                                "limit_window_seconds": 3600,
                                "reset_at": 1893456000,
                            },
                            "secondary_window": {
                                "used_percent": 17,
                                "limit_window_seconds": 1800,
                                "reset_at": 1893454200,
                            },
                        },
                    },
                )
            )

            with (
                patch("llm_agent_platform.config.STATE_DIR", str(tmp_dir)),
                patch(
                    "llm_agent_platform.services.provider_usage_limits.get_http_client",
                    return_value=fake_client,
                ),
            ):
                snapshot = OpenAIChatGptUsageLimitsAdapter(creds_path).fetch_snapshot()

            usage_state = json.loads(
                (
                    tmp_dir
                    / "openai-chatgpt"
                    / "usage"
                    / "accounts"
                    / "default"
                    / "limits.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(snapshot["limits"]["primary"]["used_percent"], 42.0)
        self.assertEqual(snapshot["limits"]["primary"]["window"], "60m")
        self.assertEqual(snapshot["metadata"]["plan_type"], "chatgpt-plus")
        self.assertEqual(usage_state["limits"]["secondary"]["window"], "30m")


if __name__ == "__main__":
    unittest.main()
