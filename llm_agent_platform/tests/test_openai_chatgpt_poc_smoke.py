"""OpenAI ChatGPT PoC smoke tests.

Suite: TS-OPENAI-CHATGPT-POC-SMOKE
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.services.account_router import quota_account_router
from llm_agent_platform.services.openai_chatgpt_admin_monitoring import (
    reset_monitoring_caches,
)

SECRETS_TEST_ROOT = Path("secrets_test")
REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload, ensure_ascii=False)
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, *, post_responses: list[FakeResponse] | None = None):
        self.post_responses = list(post_responses or [])
        self.post_calls: list[dict] = []

    def post(self, url, headers=None, json=None, content=None):
        self.post_calls.append(
            {"url": url, "headers": headers or {}, "json": json, "content": content}
        )
        if not self.post_responses:
            raise AssertionError("Unexpected POST call")
        return self.post_responses.pop(0)


@contextmanager
def _tmp_state_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class OpenAIChatGPTPoCSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        quota_account_router._state.clear()
        quota_account_router._preferred_account_overrides.clear()
        reset_monitoring_caches()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _openai_accounts_config(credentials_path: str) -> dict:
        return {
            "mode": "rounding",
            "active_account": "acct-1",
            "all_accounts": ["acct-1", "acct-2"],
            "accounts": {
                "acct-1": {"credentials_path": credentials_path},
                "acct-2": {"credentials_path": credentials_path},
            },
            "rotation_policy": {
                "rate_limit_threshold": 2,
                "quota_exhausted_threshold": 2,
                "rate_limit_cooldown_seconds": 60,
            },
            "quota_scope": "per_provider",
            "model_quota_resets": {"default": "07:00:00"},
            "groups": {
                "team-a": {
                    "accounts": ["acct-1", "acct-2"],
                    "models": ["gpt-5.4", "gpt-5.4-mini"],
                }
            },
        }

    @staticmethod
    def _service_behavior_config() -> str:
        return """monitoring:\n  background_poll_interval_seconds: 3600\n  refresh_concurrency: 2\n  per_account_refresh_timeout_seconds: 2\n  refresh_run_timeout_seconds: 10\n  refresh_run_retention_seconds: 300\nfrontend:\n  page_poll_interval_ms: 10000\n  manual_refresh_status_poll_interval_ms: 1000\n"""

    @staticmethod
    def _auth_headers(raw_api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {raw_api_key}"}

    @contextmanager
    def _patched_paths(self, tmp_dir: Path, accounts_config_path: Path):
        registry_path = tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
        service_behavior_path = tmp_dir / "service_behavior_config.yaml"
        service_behavior_path.write_text(
            self._service_behavior_config(), encoding="utf-8"
        )
        with (
            patch("llm_agent_platform.config.STATE_DIR", str(tmp_dir)),
            patch("llm_agent_platform.services.account_router.STATE_DIR", str(tmp_dir)),
            patch(
                "llm_agent_platform.services.account_state_store.STATE_DIR",
                str(tmp_dir),
            ),
            patch(
                "llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)
            ),
            patch(
                "llm_agent_platform.services.openai_chatgpt_admin_monitoring.STATE_DIR",
                str(tmp_dir),
            ),
            patch(
                "llm_agent_platform.services.account_router.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(accounts_config_path),
            ),
            patch(
                "llm_agent_platform.auth.credentials.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(accounts_config_path),
            ),
            patch(
                "llm_agent_platform.services.service_behavior_config.SERVICE_BEHAVIOR_CONFIG_PATH",
                str(service_behavior_path),
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

    def _seed_monitoring_files(self, tmp_dir: Path) -> None:
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-1" / "usage_windows.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "short_window": {"used_percent": 12.5, "window_minutes": 60},
                "long_window": {"used_percent": 44.0, "window_minutes": 10080},
                "refresh": {
                    "last_refreshed_at": "2026-01-01T00:00:00Z",
                    "next_refresh_at": "2026-01-01T00:05:00Z",
                    "refresh_interval_seconds": 300,
                    "status": "fresh",
                    "last_error": None,
                },
            },
        )
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-2" / "usage_windows.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "short_window": {"used_percent": 3.0, "window_minutes": 60},
                "long_window": {"used_percent": 10.0, "window_minutes": 10080},
                "refresh": {
                    "last_refreshed_at": "2026-01-01T00:00:00Z",
                    "next_refresh_at": "2026-01-01T00:05:00Z",
                    "refresh_interval_seconds": 300,
                    "status": "fresh",
                    "last_error": None,
                },
            },
        )
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-1" / "request_usage.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "request_counters": {
                    "total_requests": 2,
                    "successful_requests": 2,
                    "failed_requests": 0,
                    "last_request_at": "2026-01-01T00:00:00Z",
                },
            },
        )

    def test_admin_key_lifecycle_smoke_covers_public_models_chat_and_revocation(self):
        """Runs the critical PoC path from admin key issuance to public-route revoke.

        Test case: TC-OPENAI-CHATGPT-POC-SMOKE-001
        Requirement: demo smoke proves create -> models/chat -> revoke -> 401 invalid_api_key.
        """

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
                self._openai_accounts_config(str(creds_path)),
            )
            fake_client = FakeHttpClient(
                post_responses=[
                    FakeResponse(
                        200,
                        {
                            "id": "chatcmpl-openai-smoke",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "smoke ok",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 2,
                                "total_tokens": 6,
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
                create_response = self.client.post(
                    "/admin/api-keys/openai-chatgpt",
                    json={"group_id": "team-a", "label": "demo-smoke"},
                )
                created_payload = create_response.get_json()
                raw_api_key = created_payload["raw_api_key"]

                models_response = self.client.get(
                    "/openai-chatgpt/v1/models",
                    headers=self._auth_headers(raw_api_key),
                )
                chat_response = self.client.post(
                    "/openai-chatgpt/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=self._auth_headers(raw_api_key),
                )
                revoke_response = self.client.post(
                    f"/admin/api-keys/openai-chatgpt/{created_payload['key_id']}/revoke"
                )
                revoked_models_response = self.client.get(
                    "/openai-chatgpt/v1/models",
                    headers=self._auth_headers(raw_api_key),
                )

        models_payload = json.loads(models_response.data.decode("utf-8"))
        chat_payload = json.loads(chat_response.data.decode("utf-8"))
        revoked_payload = json.loads(revoked_models_response.data.decode("utf-8"))

        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(created_payload["provider_id"], "openai-chatgpt")
        self.assertEqual(models_response.status_code, 200)
        self.assertEqual(
            {item["id"] for item in models_payload["data"]},
            {"gpt-5.4", "gpt-5.4-mini"},
        )
        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(chat_payload["choices"][0]["message"]["content"], "smoke ok")
        self.assertEqual(revoke_response.status_code, 200)
        self.assertEqual(revoke_response.get_json()["status"], "revoked")
        self.assertEqual(revoked_models_response.status_code, 401)
        self.assertEqual(revoked_payload["error"]["code"], "invalid_api_key")

    def test_monitoring_page_smoke_exposes_frontend_critical_contract_fields(self):
        """Checks the provider page fields used by the PoC operator UI.

        Test case: TC-OPENAI-CHATGPT-POC-SMOKE-002
        Requirement: monitoring payload keeps schema-critical fields for refresh, drawer and activate flows.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            self._write_json(creds_path, {"version": 1})
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )
            self._seed_monitoring_files(tmp_dir)

            with self._patched_paths(tmp_dir, accounts_config_path):
                activate_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/groups/team-a/accounts/acct-1/activate"
                )
                page_response = self.client.get("/admin/monitoring/openai-chatgpt")

        activate_payload = activate_response.get_json()
        page_payload = page_response.get_json()
        group_payload = page_payload["groups"][0]
        account_payload = group_payload["accounts"][0]

        self.assertEqual(activate_response.status_code, 200)
        self.assertEqual(
            set(activate_payload),
            {"ok", "provider_id", "group_id", "account_name", "activate_semantics"},
        )
        self.assertEqual(page_response.status_code, 200)
        self.assertEqual(page_payload["provider_id"], "openai-chatgpt")
        self.assertIn("capabilities", page_payload)
        self.assertIn("security_boundary", page_payload)
        self.assertTrue(page_payload["capabilities"]["supports_manual_refresh"])
        self.assertEqual(
            page_payload["security_boundary"]["scope"], "local_single_user_poc"
        )
        self.assertEqual(group_payload["group_id"], "team-a")
        self.assertIn("refresh", group_payload)
        self.assertIn("accounts", group_payload)
        self.assertEqual(account_payload["account_name"], "acct-1")
        self.assertTrue(account_payload["is_preferred_for_session"])
        self.assertIn("short_window", account_payload)
        self.assertIn("long_window", account_payload)
        self.assertIn("drawer", account_payload)
        self.assertEqual(
            account_payload["actions"]["activate_semantics"],
            "session_scoped_in_memory_preferred_account_override",
        )

    def test_delivery_configs_keep_frontend_local_and_public_api_scoped(self):
        """Verifies the checked-in PoC delivery boundary configuration.

        Test case: TC-OPENAI-CHATGPT-POC-SMOKE-003
        Requirement: checked-in delivery config keeps frontend/admin local and exposes only the provider public namespace.
        """

        dev_server_config = json.loads(
            (REPO_ROOT / "services/frontend/config/dev-server.json").read_text(
                encoding="utf-8"
            )
        )
        runtime_config = json.loads(
            (REPO_ROOT / "services/frontend/public/runtime-config.json").read_text(
                encoding="utf-8"
            )
        )
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertEqual(dev_server_config["backendTarget"], "http://127.0.0.1:4000")
        self.assertEqual(
            set(dev_server_config["proxyPaths"]), {"/admin", "/openai-chatgpt"}
        )
        self.assertEqual(runtime_config["adminApiBasePath"], "/admin")
        self.assertEqual(runtime_config["publicApiBasePath"], "/openai-chatgpt")
        self.assertIn('"127.0.0.1:4000:4000"', compose_text)
        self.assertIn('"127.0.0.1:4173:80"', compose_text)
        self.assertIn(
            "./services/frontend/public/runtime-config.json:/usr/share/nginx/html/runtime-config.json:ro",
            compose_text,
        )


if __name__ == "__main__":
    unittest.main()
