import json
import tempfile
import unittest
from contextlib import contextmanager
from typing import Iterator
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.services.openai_chatgpt_model_capabilities import (
    ModelCapabilitiesRegistryError,
    OpenAIChatGPTModelCapabilitiesService,
)
from llm_agent_platform.services.provider_registry import ProviderRegistry

SECRETS_TEST_ROOT = Path("secrets_test")


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(
        self, response: FakeResponse | None = None, error: Exception | None = None
    ):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def get(self, url, headers=None):
        self.calls.append({"url": url, "headers": headers})
        if self.error is not None:
            raise self.error
        return self.response


@contextmanager
def _state_dir() -> Iterator[Path]:
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class ProviderCatalogTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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

            with patch(
                "llm_agent_platform.services.provider_registry.STATE_DIR", str(tmp_dir)
            ):
                registry = ProviderRegistry.load()
                snapshot = registry.load_catalog("gemini-cli")

        self.assertEqual(snapshot.source, "bootstrap")
        self.assertIn(
            "gemini-3-flash-preview", {model.model_id for model in snapshot.models}
        )
        self.assertNotIn(
            "unexpected-model", {model.model_id for model in snapshot.models}
        )

    def test_openai_chatgpt_uses_bootstrap_without_oauth_state(self):
        fake_client = FakeHttpClient(
            response=FakeResponse(
                200,
                {"data": [{"id": "gpt-5-mini", "owned_by": "openai"}]},
            )
        )

        with _state_dir() as tmp_dir:
            with (
                patch(
                    "llm_agent_platform.services.provider_registry.STATE_DIR",
                    str(tmp_dir),
                ),
                patch(
                    "llm_agent_platform.services.provider_registry.get_http_client",
                    return_value=fake_client,
                ),
                patch.dict(
                    "os.environ",
                    {"OPENAI_CHATGPT_DISCOVERY_BASE_URL": "https://discovery.example"},
                    clear=False,
                ),
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
                patch(
                    "llm_agent_platform.services.provider_registry.STATE_DIR",
                    str(tmp_dir),
                ),
                patch(
                    "llm_agent_platform.services.provider_registry.get_http_client",
                    return_value=fake_client,
                ),
                patch.dict(
                    "os.environ",
                    {"OPENAI_CHATGPT_DISCOVERY_BASE_URL": "https://discovery.example"},
                    clear=False,
                ),
            ):
                registry = ProviderRegistry.load()
                snapshot = registry.load_catalog("openai-chatgpt")

        self.assertEqual(snapshot.source, "bootstrap")
        self.assertEqual(fake_client.calls, [])
        self.assertIn("gpt-5.4-mini", {model.model_id for model in snapshot.models})
        self.assertNotIn(
            "unexpected-model", {model.model_id for model in snapshot.models}
        )


class OpenAIChatGPTModelCapabilitiesTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def test_capability_overlay_returns_merged_record_for_catalog_model(self):
        payload = {
            "version": 1,
            "provider_id": "openai-chatgpt",
            "models": {
                "gpt-5.4": {
                    "parameters": {
                        "reasoning_effort": {
                            "supported": True,
                            "values": ["none", "low", "medium", "high", "xhigh"],
                            "default": "none",
                            "ui_label": "Reasoning level",
                        }
                    },
                    "raw": {"source": "test"},
                }
            },
        }

        with _state_dir() as tmp_dir:
            overlay_path = (
                tmp_dir / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            self._write_json(overlay_path, payload)

            service = OpenAIChatGPTModelCapabilitiesService(
                registry_path=overlay_path,
                provider_registry=ProviderRegistry.load(),
            )
            record = service.get_model_capabilities("gpt-5.4")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.provider_id, "openai-chatgpt")
        self.assertEqual(record.model_id, "gpt-5.4")
        self.assertEqual(record.display_name, "GPT-5.4")
        self.assertEqual(record.catalog_model.display_name, "GPT-5.4")
        self.assertEqual(
            record.parameters["reasoning_effort"].values,
            ("none", "low", "medium", "high", "xhigh"),
        )
        self.assertEqual(record.parameters["reasoning_effort"].default, "none")
        self.assertEqual(
            record.to_admin_payload(),
            {
                "provider_id": "openai-chatgpt",
                "model_id": "gpt-5.4",
                "display_name": "GPT-5.4",
                "parameters": {
                    "reasoning_effort": {
                        "supported": True,
                        "values": ["none", "low", "medium", "high", "xhigh"],
                        "default": "none",
                        "ui_label": "Reasoning level",
                    }
                },
                "drawer": {"raw_capability_payload": {"source": "test"}},
            },
        )

    def test_capability_overlay_missing_file_and_missing_record_return_none(self):
        with _state_dir() as tmp_dir:
            overlay_path = (
                tmp_dir / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            service = OpenAIChatGPTModelCapabilitiesService(
                registry_path=overlay_path,
                provider_registry=ProviderRegistry.load(),
            )
            self.assertIsNone(service.get_model_capabilities("gpt-5.4"))

            self._write_json(
                overlay_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "models": {
                        "gpt-5.3-codex": {
                            "parameters": {
                                "reasoning_effort": {
                                    "supported": True,
                                    "values": ["low", "medium", "high", "xhigh"],
                                    "default": "medium",
                                }
                            },
                            "raw": {"source": "test"},
                        }
                    },
                },
            )

            self.assertIsNone(service.get_model_capabilities("gpt-5.4-mini"))

    def test_capability_overlay_rejects_unknown_model_and_malformed_parameters(self):
        with _state_dir() as tmp_dir:
            overlay_path = (
                tmp_dir / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            service = OpenAIChatGPTModelCapabilitiesService(
                registry_path=overlay_path,
                provider_registry=ProviderRegistry.load(),
            )

            self._write_json(
                overlay_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "models": {
                        "unknown-model": {
                            "parameters": {
                                "reasoning_effort": {
                                    "supported": True,
                                    "values": ["none"],
                                    "default": "none",
                                }
                            },
                            "raw": {"source": "test"},
                        }
                    },
                },
            )
            with self.assertRaisesRegex(
                ModelCapabilitiesRegistryError, "unknown provider_registry model_id"
            ):
                service.get_model_capabilities("unknown-model")

            self._write_json(
                overlay_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "models": {
                        "gpt-5.4": {
                            "parameters": {
                                "reasoning_effort": {
                                    "supported": True,
                                    "values": ["none", "low"],
                                    "default": "medium",
                                }
                            },
                            "raw": {"source": "test"},
                        }
                    },
                },
            )
            with self.assertRaisesRegex(
                ModelCapabilitiesRegistryError, "default that exists in values"
            ):
                service.get_model_capabilities("gpt-5.4")


if __name__ == "__main__":
    unittest.main()
