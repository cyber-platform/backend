import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from contextlib import contextmanager

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


from llm_agent_platform.__main__ import app
from llm_agent_platform.services.account_router import BaseAccount, GeminiAccount, SelectedAccount
from llm_agent_platform.services.account_router import AccountRouterError
from llm_agent_platform.services.account_state_store import AccountStatePaths, state_writer
from llm_agent_platform.api.openai.providers.base import ProviderRuntimeCreds
from llm_agent_platform.api.openai.providers.qwen_code import QwenCodeProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload, ensure_ascii=False) if payload is not None else "")
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def _parse_sse_json_chunks(raw_stream: str) -> list[dict]:
    chunks: list[dict] = []
    for frame in raw_stream.split("\n\n"):
        frame = frame.strip()
        if not frame.startswith("data: "):
            continue
        payload = frame[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return chunks


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


class OpenAIContractTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    @contextmanager
    def _patched_paths(tmp_dir: Path, gemini_path: Path | None = None, qwen_path: Path | None = None):
        patches = []
        if gemini_path is not None:
            patches.append(
                patch("llm_agent_platform.services.account_router.GEMINI_ACCOUNTS_CONFIG_PATH", str(gemini_path))
            )
        if qwen_path is not None:
            patches.append(
                patch("llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH", str(qwen_path))
            )
        patches.append(patch("llm_agent_platform.services.account_router.STATE_DIR", str(tmp_dir)))
        patches.append(patch("llm_agent_platform.services.account_state_store.STATE_DIR", str(tmp_dir)))
        patches.append(
            patch(
                "llm_agent_platform.services.account_router.AccountStatePaths",
                new=lambda provider_id, account_name, root_dir=Path("."): AccountStatePaths(
                    provider_id=provider_id,
                    account_name=account_name,
                    root_dir=tmp_dir,
                ),
            )
        )
        if not patches:
            yield
            return

        with patches[0]:
            for ctx in patches[1:]:
                ctx.__enter__()
            try:
                yield
            finally:
                state_writer.flush_once()
                for ctx in reversed(patches[1:]):
                    ctx.__exit__(None, None, None)

    @staticmethod
    def _write_json(path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _gemini_config() -> dict:
        return {
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
        }

    @staticmethod
    def _qwen_config() -> dict:
        return {
            "mode": "single",
            "active_account": "acct-1",
            "all_accounts": ["acct-1"],
            "accounts": {
                "acct-1": {"credentials_path": "secrets_test/accounts/acct-1.json"},
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

    @staticmethod
    def _gemini_selected_account(mode: str = "single", name: str = "acct-1", project_id: str = "demo-project") -> SelectedAccount:
        account = GeminiAccount(
            name=name,
            credentials_path=f"secrets_test/accounts/{name}.json",
            project_id=project_id,
        )
        pool = ["acct-1"] if mode == "single" else ["acct-1", "acct-2"]
        return SelectedAccount(
            provider="gemini",
            mode=mode,
            account=account,
            pool=pool,
        )

    @staticmethod
    def _qwen_selected_account(mode: str = "single", name: str = "acct-1") -> SelectedAccount:
        account = BaseAccount(
            name=name,
            credentials_path=f"secrets_test/accounts/{name}.json",
        )
        pool = ["acct-1"] if mode == "single" else ["acct-1", "acct-2"]
        return SelectedAccount(
            provider="qwen",
            mode=mode,
            account=account,
            pool=pool,
        )

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_gemini_access_token_from_file", return_value="token-123")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_auth_lock", return_value=threading.Lock())
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.send_generate")
    def test_non_stream_text_usage_contract(self, mock_send_generate, _mock_lock, _mock_token, mock_select_account):
        mock_select_account.return_value = self._gemini_selected_account()
        mock_send_generate.return_value = FakeResponse(
            200,
            {
                "response": {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "hello from gemini"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 20,
                    },
                }
            },
        )

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            gemini_cfg = self._gemini_config()
            _seed_credentials_from_config(gemini_cfg)
            self._write_json(gemini_path, gemini_cfg)
            with self._patched_paths(tmp_dir, gemini_path=gemini_path, qwen_path=gemini_path):
                response = self.client.post(
                    "/gemini-cli/v1/chat/completions",
                    json={
                        "model": "gemini-3-flash-preview",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                )

                self.assertEqual(response.status_code, 200)
                body = json.loads(response.data.decode("utf-8"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "hello from gemini")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["prompt_tokens"], 12)
        self.assertEqual(body["usage"]["completion_tokens"], 8)
        self.assertEqual(body["usage"]["total_tokens"], 20)

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_gemini_access_token_from_file", return_value="token-123")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_auth_lock", return_value=threading.Lock())
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.stream_generate_lines")
    def test_stream_usage_chunk_and_done(self, mock_stream_generate_lines, _mock_lock, _mock_token, mock_select_account):
        mock_select_account.return_value = self._gemini_selected_account()
        mock_stream_generate_lines.return_value = iter(
            [
                'data: {"response":{"candidates":[{"content":{"parts":[{"text":"hel"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}}',
                'data: {"response":{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}}',
                "data: [DONE]",
            ]
        )

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            gemini_cfg = self._gemini_config()
            _seed_credentials_from_config(gemini_cfg)
            self._write_json(gemini_path, gemini_cfg)
            with self._patched_paths(tmp_dir, gemini_path=gemini_path, qwen_path=gemini_path):
                response = self.client.post(
                    "/gemini-cli/v1/chat/completions",
                    json={
                        "model": "gemini-3-flash-preview",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.data.decode("utf-8")
                self.assertIn("data: [DONE]", payload)

        chunks = _parse_sse_json_chunks(payload)
        content = ""
        usage_chunk = None
        for chunk in chunks:
            choices = chunk.get("choices", [])
            if not choices and "usage" in chunk:
                usage_chunk = chunk
                continue
            if choices:
                content += choices[0].get("delta", {}).get("content", "")

        self.assertEqual(content, "hello")
        self.assertIsNotNone(usage_chunk)
        self.assertEqual(usage_chunk["usage"]["total_tokens"], 5)

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_gemini_access_token_from_file", return_value="token-123")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_auth_lock", return_value=threading.Lock())
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.send_generate")
    def test_non_stream_upstream_error_mapped_to_openai_shape(self, mock_send_generate, _mock_lock, _mock_token, mock_select_account):
        mock_select_account.return_value = self._gemini_selected_account()
        mock_send_generate.return_value = FakeResponse(503, payload=None, text="backend unavailable")

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            gemini_cfg = self._gemini_config()
            _seed_credentials_from_config(gemini_cfg)
            self._write_json(gemini_path, gemini_cfg)
            with self._patched_paths(tmp_dir, gemini_path=gemini_path, qwen_path=gemini_path):
                response = self.client.post(
                    "/gemini-cli/v1/chat/completions",
                    json={
                        "model": "gemini-3-flash-preview",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": False,
                    },
                )

                self.assertEqual(response.status_code, 503)
                body = json.loads(response.data.decode("utf-8"))
        self.assertIn("error", body)
        self.assertEqual(body["error"]["type"], "upstream_error")
        self.assertEqual(body["error"]["code"], 503)

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch(
        "llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.register_event",
        return_value=type(
            "EventResult",
            (),
            {"switched": True, "all_exhausted": False, "all_cooldown": False},
        )(),
    )
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_gemini_access_token_from_file", return_value="token-123")
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.get_auth_lock", return_value=threading.Lock())
    @patch("llm_agent_platform.api.openai.providers.gemini_cli.send_generate")
    def test_non_stream_quota_rotation_contract(
        self,
        mock_send_generate,
        _mock_lock,
        _mock_token,
        _mock_register_event,
        mock_select_account,
    ):
        mock_select_account.side_effect = [
            self._gemini_selected_account(mode="rounding", name="acct-1", project_id="p1"),
            self._gemini_selected_account(mode="rounding", name="acct-2", project_id="p2"),
        ]
        mock_send_generate.side_effect = [
            FakeResponse(429, payload=None, text="quota exceeded"),
            FakeResponse(
                200,
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "rotated hello"}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 4,
                            "totalTokenCount": 14,
                        },
                    }
                },
            ),
        ]

        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            gemini_cfg = self._gemini_config()
            _seed_credentials_from_config(gemini_cfg)
            self._write_json(gemini_path, gemini_cfg)
            with self._patched_paths(tmp_dir, gemini_path=gemini_path, qwen_path=gemini_path):
                response = self.client.post(
                    "/gemini-cli/v1/chat/completions",
                    json={
                        "model": "gemini-3-flash-preview",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                )

                self.assertEqual(response.status_code, 200)
                body = json.loads(response.data.decode("utf-8"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "rotated hello")
        self.assertEqual(body["usage"]["total_tokens"], 14)
        self.assertEqual(mock_send_generate.call_count, 2)
        self.assertEqual(mock_select_account.call_count, 2)

    @patch(
        "llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account",
        side_effect=AccountRouterError("all accounts on cooldown please wait 4"),
    )
    def test_all_cooldown_message_contract(self, _mock_select_account):
        with _secrets_test_dir() as tmp_dir:
            gemini_path = tmp_dir / "gemini_accounts_config.json"
            gemini_cfg = self._gemini_config()
            _seed_credentials_from_config(gemini_cfg)
            self._write_json(gemini_path, gemini_cfg)
            with self._patched_paths(tmp_dir, gemini_path=gemini_path, qwen_path=gemini_path):
                response = self.client.post(
                    "/gemini-cli/v1/chat/completions",
                    json={
                        "model": "gemini-3-flash-preview",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": False,
                    },
                )

                self.assertEqual(response.status_code, 429)
                body = json.loads(response.data.decode("utf-8"))
        self.assertIn("error", body)
        self.assertIn("all accounts on cooldown please wait 4", body["error"]["message"])

    @patch("llm_agent_platform.api.openai.providers.qwen_code.read_qwen_credentials")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.refresh_qwen_credentials_file")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.load_last_used_at")
    def test_qwen_idle_refresh_triggers_refresh(
        self,
        mock_load_last_used,
        mock_refresh,
        mock_read,
    ):
        account = BaseAccount(name="acct-1", credentials_path="secrets_test/accounts/qwen.json")
        last_used = datetime.now(tz=timezone.utc) - timedelta(seconds=1000)
        mock_load_last_used.return_value = last_used
        mock_refresh.return_value = {
            "access_token": "qwen-token",
            "resource_url": "https://dashscope.aliyuncs.com/compatible-mode",
        }
        mock_read.side_effect = AssertionError("read_qwen_credentials should not be called")

        _ensure_dummy_credentials("secrets_test/accounts/qwen.json")

        provider = QwenCodeProvider()
        creds = provider.load_runtime_credentials(account)

        self.assertEqual(creds.token, "qwen-token")
        self.assertEqual(creds.resource_url, "https://dashscope.aliyuncs.com/compatible-mode")
        self.assertEqual(mock_refresh.call_count, 1)

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch(
        "llm_agent_platform.api.openai.providers.qwen_code.QwenCodeProvider.load_runtime_credentials",
        return_value=ProviderRuntimeCreds(
            token="qwen-token",
            resource_url="https://dashscope.aliyuncs.com/compatible-mode",
        ),
    )
    @patch("llm_agent_platform.api.openai.providers.qwen_code.stream_generate_lines_from_url")
    def test_stream_quota_rotation_usage_contract(
        self,
        mock_stream_generate_lines,
        _mock_load_credentials,
        mock_select_account,
    ):
        mock_select_account.return_value = self._qwen_selected_account(mode="rounding")
        mock_stream_generate_lines.return_value = iter(
            [
                'data: {"id":"chatcmpl-q1","choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
                'data: {"id":"chatcmpl-q1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                "data: [DONE]",
            ]
        )

        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_cfg = self._qwen_config()
            _seed_credentials_from_config(qwen_cfg)
            self._write_json(qwen_path, qwen_cfg)
            with self._patched_paths(tmp_dir, qwen_path=qwen_path, gemini_path=qwen_path):
                response = self.client.post(
                    "/qwen-code/v1/chat/completions",
                    json={
                        "model": "coder-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.data.decode("utf-8")
                self.assertIn("data: [DONE]", payload)

        chunks = _parse_sse_json_chunks(payload)
        content = ""
        usage_chunk = None
        for chunk in chunks:
            choices = chunk.get("choices", [])
            if not choices and "usage" in chunk:
                usage_chunk = chunk
                continue
            if choices:
                content += choices[0].get("delta", {}).get("content", "")

        self.assertEqual(content, "hello")
        self.assertIsNotNone(usage_chunk)
        self.assertEqual(usage_chunk["usage"]["total_tokens"], 5)

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.save_last_used_at")
    @patch(
        "llm_agent_platform.api.openai.providers.qwen_code.QwenCodeProvider.load_runtime_credentials",
        return_value=ProviderRuntimeCreds(
            token="qwen-token",
            resource_url="https://dashscope.aliyuncs.com/compatible-mode",
        ),
    )
    @patch("llm_agent_platform.api.openai.providers.qwen_code.stream_generate_lines_from_url")
    def test_qwen_stream_persists_last_used(
        self,
        mock_stream_generate_lines,
        _mock_load_credentials,
        mock_save_last_used,
        mock_select_account,
    ):
        mock_select_account.return_value = self._qwen_selected_account(mode="single")
        mock_stream_generate_lines.return_value = iter(
            [
                'data: {"id":"chatcmpl-q1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                "data: [DONE]",
            ]
        )

        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_cfg = self._qwen_config()
            _seed_credentials_from_config(qwen_cfg)
            self._write_json(qwen_path, qwen_cfg)
            with self._patched_paths(tmp_dir, qwen_path=qwen_path, gemini_path=qwen_path):
                response = self.client.post(
                    "/qwen-code/v1/chat/completions",
                    json={
                        "model": "coder-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )

                payload_stream = response.data.decode("utf-8")
                self.assertTrue(payload_stream)
                self.assertEqual(response.status_code, 200)

        mock_save_last_used.assert_called_once()
        saved_paths = mock_save_last_used.call_args[0][0]
        self.assertEqual(saved_paths.provider_id, "qwen_code")
        self.assertEqual(saved_paths.account_name, "acct-1")

    @patch("llm_agent_platform.api.openai.strategies.rotate_on_429_rounding.quota_account_router.select_account")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.save_last_used_at")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.send_generate_to_url")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.read_qwen_credentials")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.refresh_qwen_credentials_file")
    @patch("llm_agent_platform.api.openai.providers.qwen_code.load_last_used_at")
    def test_qwen_refresh_and_retry_on_401(
        self,
        mock_load_last_used,
        mock_refresh,
        mock_read,
        mock_send_generate,
        _mock_touch,
        mock_select_account,
    ):
        mock_select_account.return_value = self._qwen_selected_account(mode="single")
        mock_load_last_used.return_value = datetime.now(tz=timezone.utc)
        mock_read.return_value = {
            "access_token": "qwen-token",
            "resource_url": "https://dashscope.aliyuncs.com/compatible-mode",
        }
        mock_refresh.return_value = {
            "access_token": "qwen-token-2",
            "resource_url": "https://dashscope.aliyuncs.com/compatible-mode",
        }
        mock_send_generate.side_effect = [
            FakeResponse(401, payload={"error": "unauthorized"}),
            FakeResponse(
                200,
                payload={
                    "id": "chatcmpl-q1",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        ]

        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_cfg = self._qwen_config()
            _seed_credentials_from_config(qwen_cfg)
            self._write_json(qwen_path, qwen_cfg)
            with self._patched_paths(tmp_dir, qwen_path=qwen_path, gemini_path=qwen_path):
                response = self.client.post(
                    "/qwen-code/v1/chat/completions",
                    json={
                        "model": "coder-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = json.loads(response.data.decode("utf-8"))
        self.assertEqual(payload["model"], "coder-model")
        self.assertEqual(mock_send_generate.call_count, 2)
        self.assertEqual(mock_refresh.call_count, 1)
