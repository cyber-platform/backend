import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from contextlib import contextmanager

from llm_agent_platform.__main__ import app
from llm_agent_platform.auth.credentials import AuthAvailability
from llm_agent_platform.services.account_router import (
    AllAccountsExhaustedError,
    GeminiAccount,
    SelectedAccount,
)
from llm_agent_platform.services.account_state_store import state_writer
from llm_agent_platform.services.openai_chatgpt_api_keys import (
    OpenAIChatGPTApiKeyRegistryService,
)

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


@contextmanager
def _patched_state_dir(tmp_dir: Path):
    with (
        patch("llm_agent_platform.services.account_router.STATE_DIR", str(tmp_dir)),
        patch(
            "llm_agent_platform.services.account_state_store.STATE_DIR", str(tmp_dir)
        ),
    ):
        try:
            yield
        finally:
            state_writer.flush_once()


def _ensure_dummy_credentials(path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.write_text("{}", encoding="utf-8")


def _seed_credentials_from_config(cfg: dict) -> None:
    accounts = cfg.get("accounts") or {}
    if not isinstance(accounts, dict):
        return
    for account in accounts.values():
        if not isinstance(account, dict):
            continue
        path = account.get("credentials_path")
        if isinstance(path, str) and path:
            _ensure_dummy_credentials(path)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict | None = None,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.text = (
            text
            if text is not None
            else (
                json.dumps(payload, ensure_ascii=False) if payload is not None else ""
            )
        )
        self.content = self.text.encode("utf-8")
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


class FakeStreamResponse(FakeResponse):
    def __init__(
        self, status_code: int, lines: list[str | bytes], text: str | None = None
    ):
        super().__init__(status_code=status_code, payload=None, text=text)
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line

    def read(self):
        return self.content


class FakeHttpClient:
    def __init__(
        self,
        post_response: FakeResponse | None = None,
        stream_response: FakeStreamResponse | None = None,
    ):
        self.post_response = post_response
        self.stream_response = stream_response
        self.last_post_call: dict | None = None
        self.last_stream_call: dict | None = None

    def post(self, url, headers=None, content=None, params=None):
        self.last_post_call = {
            "url": url,
            "headers": headers,
            "content": content,
            "params": params,
        }
        return self.post_response

    def stream(self, method, url, headers=None, content=None, params=None):
        self.last_stream_call = {
            "method": method,
            "url": url,
            "headers": headers,
            "content": content,
            "params": params,
        }
        return self.stream_response


class RefactorP2RoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        _seed_credentials_from_config(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _selected_gemini_account() -> SelectedAccount:
        return SelectedAccount(
            provider="gemini",
            mode="single",
            account=GeminiAccount(
                name="acct-1",
                credentials_path="secrets_test/accounts/acct-1.json",
                project_id="demo-project",
            ),
            pool=["acct-1"],
            model="gemini-3-flash-preview",
        )

    def test_smoke_import_and_models_endpoint(self):
        self.assertIsNotNone(app)

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            openai_path = tmp_dir / "openai_accounts_config.json"
            qwen_path = tmp_dir / "qwen_accounts_config.json"
            openai_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            gemini_cfg = {
                "mode": "single",
                "active_account": "acct-1",
                "all_accounts": ["acct-1"],
                "accounts": {
                    "acct-1": {
                        "credentials_path": "secrets_test/accounts/gemini_acct-1.json",
                        "project_id": "demo-project",
                    }
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
            }
            qwen_cfg = {
                "mode": "single",
                "active_account": "acct-1",
                "all_accounts": ["acct-1"],
                "accounts": {
                    "acct-1": {
                        "credentials_path": "secrets_test/accounts/qwen_acct-1.json"
                    },
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
            }
            openai_cfg = {
                "mode": "single",
                "active_account": "acct-1",
                "all_accounts": ["acct-1"],
                "accounts": {
                    "acct-1": {
                        "credentials_path": "secrets_test/accounts/openai_acct-1.json"
                    }
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
                "groups": {
                    "team-a": {
                        "accounts": ["acct-1"],
                        "models": ["gpt-5.4", "gpt-5.4-mini"],
                    }
                },
            }
            _seed_credentials_from_config(gemini_cfg)
            _seed_credentials_from_config(qwen_cfg)
            _seed_credentials_from_config(openai_cfg)
            self._write_json(gemini_path, gemini_cfg)
            self._write_json(qwen_path, qwen_cfg)
            self._write_json(openai_path, openai_cfg)

            with (
                patch(
                    "llm_agent_platform.services.account_router.GEMINI_ACCOUNTS_CONFIG_PATH",
                    str(gemini_path),
                ),
                patch(
                    "llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH",
                    str(qwen_path),
                ),
                patch(
                    "llm_agent_platform.services.account_router.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                    str(openai_path),
                ),
                patch(
                    "llm_agent_platform.auth.credentials.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                    str(openai_path),
                ),
                patch(
                    "llm_agent_platform.config.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                    str(openai_registry_path),
                ),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_api_keys.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                    str(openai_registry_path),
                ),
                _patched_state_dir(tmp_dir),
            ):
                openai_api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="models-smoke"
                )["raw_api_key"]
                gemini_response = self.client.get("/gemini-cli/v1/models")
                self.assertEqual(gemini_response.status_code, 200)
                gemini_payload = json.loads(gemini_response.data.decode("utf-8"))
                gemini_ids = {item["id"] for item in gemini_payload["data"]}
                self.assertIn("gemini-3-flash-preview", gemini_ids)

                qwen_response = self.client.get("/qwen-code/v1/models")
                self.assertEqual(qwen_response.status_code, 200)
                qwen_payload = json.loads(qwen_response.data.decode("utf-8"))
                qwen_ids = {item["id"] for item in qwen_payload["data"]}
                self.assertIn("coder-model", qwen_ids)

                openai_response = self.client.get(
                    "/openai-chatgpt/v1/models",
                    headers={"Authorization": f"Bearer {openai_api_key}"},
                )
                self.assertEqual(openai_response.status_code, 200)
                openai_payload = json.loads(openai_response.data.decode("utf-8"))
                openai_ids = {item["id"] for item in openai_payload["data"]}
                self.assertIn("gpt-5.4", openai_ids)

    def test_group_aware_models_endpoint(self):
        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            gemini_cfg = {
                "mode": "single",
                "active_account": "lisa",
                "all_accounts": ["lisa"],
                "accounts": {
                    "lisa": {
                        "credentials_path": "secrets_test/accounts/gemini_lisa.json",
                        "project_id": "project-lisa",
                    }
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
            }
            qwen_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "groups": {
                    "g1": {
                        "accounts": ["lisa"],
                        "models": ["coder-model"],
                    },
                    "g2": {
                        "accounts": ["petr"],
                        "models": ["coder-model"],
                    },
                },
                "accounts": {
                    "lisa": {
                        "credentials_path": "secrets_test/accounts/qwen_lisa.json"
                    },
                    "petr": {
                        "credentials_path": "secrets_test/accounts/qwen_petr.json"
                    },
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
            }

            _seed_credentials_from_config(gemini_cfg)
            _seed_credentials_from_config(qwen_cfg)

            self._write_json(gemini_path, gemini_cfg)
            self._write_json(qwen_path, qwen_cfg)

            with (
                patch(
                    "llm_agent_platform.services.account_router.GEMINI_ACCOUNTS_CONFIG_PATH",
                    str(gemini_path),
                ),
                patch(
                    "llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH",
                    str(qwen_path),
                ),
                _patched_state_dir(tmp_dir),
            ):
                default_response = self.client.get("/qwen-code/v1/models")
                self.assertEqual(default_response.status_code, 200)
                default_payload = json.loads(default_response.data.decode("utf-8"))
                default_ids = {item["id"] for item in default_payload["data"]}
                self.assertEqual(default_ids, {"coder-model"})

                g1_response = self.client.get("/qwen-code/g1/v1/models")
                self.assertEqual(g1_response.status_code, 200)
                g1_payload = json.loads(g1_response.data.decode("utf-8"))
                g1_ids = {item["id"] for item in g1_payload["data"]}
                self.assertEqual(g1_ids, {"coder-model"})

                g2_response = self.client.get("/qwen-code/g2/v1/models")
                self.assertEqual(g2_response.status_code, 200)
                g2_payload = json.loads(g2_response.data.decode("utf-8"))
                g2_ids = {item["id"] for item in g2_payload["data"]}
                self.assertEqual(g2_ids, {"coder-model"})

    def test_unknown_provider_models_returns_contract_error(self):
        response = self.client.get("/missing-provider/v1/models")
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.data.decode("utf-8"))
        self.assertIn("Unknown provider", payload["error"]["message"])

    def test_unknown_group_returns_contract_error(self):
        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_cfg = {
                "mode": "rounding",
                "active_account": "acct-1",
                "all_accounts": ["acct-1"],
                "groups": {
                    "g1": {
                        "accounts": ["acct-1"],
                        "models": ["coder-model"],
                    }
                },
                "accounts": {
                    "acct-1": {
                        "credentials_path": "secrets_test/accounts/qwen_acct-1.json"
                    },
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                },
            }
            _seed_credentials_from_config(qwen_cfg)
            self._write_json(qwen_path, qwen_cfg)

            with (
                patch(
                    "llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH",
                    str(qwen_path),
                ),
                _patched_state_dir(tmp_dir),
            ):
                response = self.client.get("/qwen-code/missing/v1/models")

        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.data.decode("utf-8"))
        self.assertIn("Unknown group", payload["error"]["message"])

    @patch("llm_agent_platform.api.gemini.routes.quota_account_router.select_account")
    @patch(
        "llm_agent_platform.api.gemini.routes.get_gemini_access_token_from_file",
        return_value="token-123",
    )
    @patch(
        "llm_agent_platform.api.gemini.routes.get_auth_lock",
        return_value=threading.Lock(),
    )
    @patch("llm_agent_platform.api.gemini.routes.send_generate")
    def test_gemini_native_nonstream_quota_success(
        self,
        mock_send_generate,
        _mock_lock,
        _mock_token,
        mock_select_account,
    ):
        mock_select_account.return_value = self._selected_gemini_account()
        mock_send_generate.return_value = FakeResponse(
            200,
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "hello from native gemini"},
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 8,
                    },
                }
            },
        )

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            self._write_json(
                gemini_path,
                {
                    "mode": "single",
                    "active_account": "acct-1",
                    "all_accounts": ["acct-1"],
                    "accounts": {
                        "acct-1": {
                            "credentials_path": "secrets_test/accounts/acct-1.json",
                            "project_id": "demo-project",
                        }
                    },
                    "rotation_policy": {
                        "rate_limit_threshold": 2,
                        "quota_exhausted_threshold": 2,
                        "rate_limit_cooldown_seconds": 5,
                    },
                    "model_quota_resets": {
                        "default": "00:00:00",
                    },
                },
            )
            _seed_credentials_from_config(
                {
                    "accounts": {
                        "acct-1": {
                            "credentials_path": "secrets_test/accounts/acct-1.json",
                        }
                    }
                }
            )
            with (
                patch(
                    "llm_agent_platform.services.account_router.GEMINI_ACCOUNTS_CONFIG_PATH",
                    str(gemini_path),
                ),
                _patched_state_dir(tmp_dir),
            ):
                response = self.client.post(
                    "/v1/models/gemini-3-flash-preview-quota:generateContent",
                    json={
                        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.data.decode("utf-8"))
        self.assertIn("candidates", body)
        self.assertEqual(
            body["candidates"][0]["content"]["parts"][0]["text"],
            "hello from native gemini",
        )

    @patch(
        "llm_agent_platform.api.gemini.routes.quota_account_router.select_account",
        side_effect=AllAccountsExhaustedError("all_accounts_exceed_quota"),
    )
    def test_gemini_native_quota_all_accounts_exhausted(self, _mock_select_account):
        response = self.client.post(
            "/v1/models/gemini-3-flash-preview-quota:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

        self.assertEqual(response.status_code, 429)
        body = json.loads(response.data.decode("utf-8"))
        self.assertEqual(body["error"]["message"], "all_accounts_exceed_quota")
        self.assertEqual(body["error"]["code"], 429)

    @patch("llm_agent_platform.api.gemini.routes.quota_account_router.select_account")
    @patch("llm_agent_platform.api.gemini.routes.quota_account_router.register_success")
    @patch(
        "llm_agent_platform.api.gemini.routes.get_gemini_access_token_from_file",
        return_value="token-123",
    )
    @patch(
        "llm_agent_platform.api.gemini.routes.get_auth_lock",
        return_value=threading.Lock(),
    )
    @patch("llm_agent_platform.api.gemini.routes.stream_generate_lines")
    def test_gemini_native_stream_quota_yields_done(
        self,
        mock_stream_generate_lines,
        _mock_lock,
        _mock_token,
        mock_register_success,
        mock_select_account,
    ):
        mock_select_account.return_value = self._selected_gemini_account()
        mock_stream_generate_lines.return_value = iter(
            [
                'data: {"response":{"candidates":[{"content":{"parts":[{"text":"hel"}]}}]}}',
                'data: {"response":{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}]}}',
                "data: [DONE]",
            ]
        )

        response = self.client.post(
            "/v1/models/gemini-3-flash-preview-quota:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.data.decode("utf-8")
        self.assertIn("data: [DONE]", payload)
        self.assertIn('"candidates"', payload)
        mock_register_success.assert_called_once()

    @patch("llm_agent_platform.api.parity.routes.get_http_client")
    def test_parity_relay_nonstream_forwards_body_and_params(
        self, mock_get_http_client
    ):
        fake_client = FakeHttpClient(
            post_response=FakeResponse(
                200,
                payload={"ok": True},
                headers={"content-type": "application/json"},
            )
        )
        mock_get_http_client.return_value = fake_client

        response = self.client.post(
            "/parity-relay/v1internal:generateContent?alt=json",
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
            headers={"Authorization": "Bearer parity-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data.decode("utf-8")), {"ok": True})
        self.assertIsNotNone(fake_client.last_post_call)
        last_post_call = fake_client.last_post_call
        if last_post_call is None:
            self.fail("last_post_call must not be None")
        self.assertEqual(last_post_call["params"], {"alt": "json"})
        self.assertEqual(last_post_call["content"], b'{"message": "hello"}')

    @patch("llm_agent_platform.api.parity.routes.get_http_client")
    def test_parity_relay_stream_mode_returns_sse(self, mock_get_http_client):
        fake_client = FakeHttpClient(
            stream_response=FakeStreamResponse(
                200,
                lines=[
                    'data: {"delta":"hel"}',
                    'data: {"delta":"lo"}',
                    "data: [DONE]",
                ],
            )
        )
        mock_get_http_client.return_value = fake_client

        response = self.client.post(
            "/parity-relay/v1internal:streamGenerateContent?alt=sse",
            data="{}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")

        payload = response.data.decode("utf-8")
        self.assertIn('data: {"delta":"hel"}', payload)
        self.assertIn("data: [DONE]", payload)
        last_stream_call = fake_client.last_stream_call
        if last_stream_call is None:
            self.fail("last_stream_call must not be None")
        self.assertEqual(last_stream_call["method"], "POST")


if __name__ == "__main__":
    unittest.main()
