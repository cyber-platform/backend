import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.services.provider_registry import ProviderRegistry

SECRETS_TEST_ROOT = Path("secrets_test")


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def get(self, url, headers=None):
        self.calls.append({"url": url, "headers": headers})
        if self.error is not None:
            raise self.error
        return self.response


@contextmanager
def _state_dir() -> Path:
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class ProviderCatalogTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_static_provider_ignores_snapshot_and_uses_bootstrap(self):
        with _state_dir() as tmp_dir:
            snapshot_path = tmp_dir / "gemini-cli" / "catalog" / "models.json"
            self._write_json(
                snapshot_path,
                {
                    "version": 1,
                    "provider_id": "gemini-cli",
                    "as_of": "2026-03-21T00:00:00Z",
                    "source": "discovery",
                    "models": [
                        {
                            "model_id": "unexpected-model",
                            "display_name": "Unexpected",
                            "capabilities": ["chat"],
                            "lifecycle": "ga",
                            "upstream_id": "unexpected-model",
                        }
                    ],
                },
            )

            with patch("llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)):
                registry = ProviderRegistry.load()
                snapshot = registry.load_catalog("gemini-cli")

        self.assertEqual(snapshot.source, "bootstrap")
        self.assertIn("gemini-3-flash-preview", {model.model_id for model in snapshot.models})
        self.assertNotIn("unexpected-model", {model.model_id for model in snapshot.models})

    def test_openai_chatgpt_uses_bootstrap_without_oauth_state(self):
        fake_client = FakeHttpClient(
            response=FakeResponse(
                200,
                {"data": [{"id": "gpt-5-mini", "owned_by": "openai"}]},
            )
        )

        with _state_dir() as tmp_dir:
            with (
                patch("llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)),
                patch("llm_agent_platform.services.provider_registry.get_http_client", return_value=fake_client),
                patch.dict("os.environ", {"OPENAI_CHATGPT_DISCOVERY_BASE_URL": "https://discovery.example"}, clear=False),
            ):
                registry = ProviderRegistry.load()
                snapshot = registry.load_catalog("openai-chatgpt")

        self.assertEqual(snapshot.source, "bootstrap")
        self.assertEqual(fake_client.calls, [])
        self.assertIn("gpt-5.4", {model.model_id for model in snapshot.models})

    def test_openai_chatgpt_ignores_snapshot_and_discovery_env_when_static(self):
        fake_client = FakeHttpClient(
            response=FakeResponse(
                200,
                {
                    "data": [
                        {"id": "gpt-5.4", "owned_by": "openai"},
                        {"id": "gpt-5-mini", "owned_by": "openai"},
                    ]
                },
            )
        )

        with _state_dir() as tmp_dir:
            snapshot_path = tmp_dir / "openai-chatgpt" / "catalog" / "models.json"
            self._write_json(
                snapshot_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-03-21T00:00:00Z",
                    "source": "discovery",
                    "models": [
                        {
                            "model_id": "unexpected-model",
                            "display_name": "Unexpected",
                            "capabilities": ["chat"],
                            "lifecycle": "ga",
                            "upstream_id": "unexpected-model",
                        }
                    ],
                },
            )

            with (
                patch("llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)),
                patch("llm_agent_platform.services.provider_registry.get_http_client", return_value=fake_client),
                patch.dict("os.environ", {"OPENAI_CHATGPT_DISCOVERY_BASE_URL": "https://discovery.example"}, clear=False),
            ):
                registry = ProviderRegistry.load()
                snapshot = registry.load_catalog("openai-chatgpt")

        self.assertEqual(snapshot.source, "bootstrap")
        self.assertEqual(fake_client.calls, [])
        self.assertIn("gpt-5.4-mini", {model.model_id for model in snapshot.models})
        self.assertNotIn("unexpected-model", {model.model_id for model in snapshot.models})


if __name__ == "__main__":
    unittest.main()
