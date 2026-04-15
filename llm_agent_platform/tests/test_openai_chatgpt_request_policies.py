"""OpenAI ChatGPT request policy registry tests.

Suite: TS-OPENAI-CHATGPT-REQUEST-POLICIES
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.services.openai_chatgpt_api_keys import (
    InvalidGroupError,
    OpenAIChatGPTApiKeyRegistryService,
)
from llm_agent_platform.services.openai_chatgpt_model_capabilities import (
    OpenAIChatGPTModelCapabilitiesService,
)
from llm_agent_platform.services.openai_chatgpt_request_policies import (
    OpenAIChatGPTRequestPolicyRegistryService,
    RequestPolicyRegistryError,
)
from llm_agent_platform.services.provider_registry import ProviderRegistry

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class OpenAIChatGPTRequestPolicyRegistryTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _openai_accounts_config(credentials_path: str) -> dict:
        return {
            "mode": "single",
            "active_account": "acct-1",
            "all_accounts": ["acct-1", "acct-2"],
            "accounts": {
                "acct-1": {"credentials_path": credentials_path},
                "acct-2": {"credentials_path": credentials_path},
            },
            "rotation_policy": {
                "rate_limit_threshold": 2,
                "quota_exhausted_threshold": 2,
                "rate_limit_cooldown_seconds": 5,
            },
            "quota_scope": "per_provider",
            "model_quota_resets": {"default": "07:00:00"},
            "groups": {
                "team-a": {"accounts": ["acct-1"], "models": ["gpt-5.4"]},
                "team-b": {"accounts": ["acct-2"], "models": ["gpt-5.4-mini"]},
            },
        }

    @contextmanager
    def _patched_paths(
        self,
        *,
        api_key_registry_path: Path,
        policy_registry_path: Path,
        accounts_config_path: Path,
        model_capabilities_path: Path,
    ):
        with (
            patch(
                "llm_agent_platform.services.account_router.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(accounts_config_path),
            ),
            patch(
                "llm_agent_platform.config.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                str(api_key_registry_path),
            ),
            patch(
                "llm_agent_platform.services.openai_chatgpt_api_keys.OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH",
                str(api_key_registry_path),
            ),
            patch(
                "llm_agent_platform.config.OPENAI_CHATGPT_REQUEST_POLICY_REGISTRY_PATH",
                str(policy_registry_path),
            ),
            patch(
                "llm_agent_platform.services.openai_chatgpt_request_policies.OPENAI_CHATGPT_REQUEST_POLICY_REGISTRY_PATH",
                str(policy_registry_path),
            ),
            patch(
                "llm_agent_platform.config.OPENAI_CHATGPT_MODEL_CAPABILITIES_PATH",
                str(model_capabilities_path),
            ),
            patch(
                "llm_agent_platform.services.openai_chatgpt_model_capabilities.OPENAI_CHATGPT_MODEL_CAPABILITIES_PATH",
                str(model_capabilities_path),
            ),
        ):
            yield

    def _build_service(
        self,
        *,
        api_key_registry_path: Path,
        policy_registry_path: Path,
        model_capabilities_path: Path,
        capability_validation_enabled: bool = True,
    ) -> OpenAIChatGPTRequestPolicyRegistryService:
        return OpenAIChatGPTRequestPolicyRegistryService(
            registry_path=policy_registry_path,
            api_key_registry=OpenAIChatGPTApiKeyRegistryService(
                registry_path=api_key_registry_path
            ),
            model_capabilities_service=OpenAIChatGPTModelCapabilitiesService(
                registry_path=model_capabilities_path,
                provider_registry=ProviderRegistry.load(),
            ),
            capability_validation_enabled=capability_validation_enabled,
        )

    def test_get_policy_returns_explicit_no_policy_state_for_known_key(self):
        """Reads a missing record as a valid pass-through state.

        Test case: TC-OPENAI-CHATGPT-REQUEST-POLICIES-001
        Requirement: missing key-scoped request policy returns explicit no-policy payload.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "llm_agent_platform" / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            credentials_path = (
                tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            )
            accounts_config_path = tmp_dir / "openai-chatgpt" / "accounts_config.json"
            credentials_path.parent.mkdir(parents=True, exist_ok=True)
            credentials_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(credentials_path)),
            )

            with self._patched_paths(
                api_key_registry_path=api_key_registry_path,
                policy_registry_path=policy_registry_path,
                accounts_config_path=accounts_config_path,
                model_capabilities_path=model_capabilities_path,
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="policy-read"
                )
                service = self._build_service(
                    api_key_registry_path=api_key_registry_path,
                    policy_registry_path=policy_registry_path,
                    model_capabilities_path=model_capabilities_path,
                )

                payload = service.get_policy(api_key["key_id"]).to_payload()

        self.assertEqual(
            payload,
            {
                "provider_id": "openai-chatgpt",
                "key_id": api_key["key_id"],
                "group_id": "team-a",
                "model_overrides": {},
            },
        )

    def test_upsert_and_delete_round_trip_persists_one_key_policy(self):
        """Stores and clears one key-scoped policy record.

        Test case: TC-OPENAI-CHATGPT-REQUEST-POLICIES-002
        Requirement: service supports read, full upsert, and delete/reset semantics.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "llm_agent_platform" / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            credentials_path = (
                tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            )
            accounts_config_path = tmp_dir / "openai-chatgpt" / "accounts_config.json"
            credentials_path.parent.mkdir(parents=True, exist_ok=True)
            credentials_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(credentials_path)),
            )
            self._write_json(
                model_capabilities_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "models": {
                        "gpt-5.4": {
                            "parameters": {
                                "reasoning_effort": {
                                    "supported": True,
                                    "values": [
                                        "none",
                                        "low",
                                        "medium",
                                        "high",
                                        "xhigh",
                                    ],
                                    "default": "none",
                                }
                            },
                            "raw": {"source": "test"},
                        }
                    },
                },
            )

            with self._patched_paths(
                api_key_registry_path=api_key_registry_path,
                policy_registry_path=policy_registry_path,
                accounts_config_path=accounts_config_path,
                model_capabilities_path=model_capabilities_path,
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="policy-upsert"
                )
                service = self._build_service(
                    api_key_registry_path=api_key_registry_path,
                    policy_registry_path=policy_registry_path,
                    model_capabilities_path=model_capabilities_path,
                )

                created = service.upsert_policy(
                    key_id=api_key["key_id"],
                    group_id="team-a",
                    model_overrides={
                        "gpt-5.4": {
                            "reasoning_effort": {
                                "mode": "force",
                                "value": "medium",
                            }
                        }
                    },
                )
                fetched = service.get_policy(api_key["key_id"])
                deleted = service.delete_policy(api_key["key_id"])

                persisted_payload = json.loads(
                    policy_registry_path.read_text(encoding="utf-8")
                )

        self.assertEqual(created.group_id, "team-a")
        self.assertIsNotNone(created.created_at)
        self.assertIsNotNone(created.updated_at)
        self.assertEqual(fetched.to_payload(), created.to_payload())
        self.assertEqual(
            persisted_payload,
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "policies": [],
            },
        )
        self.assertEqual(
            deleted.to_payload(),
            {
                "provider_id": "openai-chatgpt",
                "key_id": api_key["key_id"],
                "group_id": "team-a",
                "model_overrides": {},
            },
        )

    def test_upsert_rejects_group_mismatch_and_invalid_capability_value(self):
        """Enforces key/group consistency and overlay-backed value validation.

        Test case: TC-OPENAI-CHATGPT-REQUEST-POLICIES-003
        Requirement: service validates group ownership and rejects unsupported parameter values.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "llm_agent_platform" / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            credentials_path = (
                tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            )
            accounts_config_path = tmp_dir / "openai-chatgpt" / "accounts_config.json"
            credentials_path.parent.mkdir(parents=True, exist_ok=True)
            credentials_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(credentials_path)),
            )
            self._write_json(
                model_capabilities_path,
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "models": {
                        "gpt-5.4": {
                            "parameters": {
                                "reasoning_effort": {
                                    "supported": True,
                                    "values": [
                                        "none",
                                        "low",
                                        "medium",
                                        "high",
                                        "xhigh",
                                    ],
                                    "default": "none",
                                }
                            },
                            "raw": {"source": "test"},
                        }
                    },
                },
            )

            with self._patched_paths(
                api_key_registry_path=api_key_registry_path,
                policy_registry_path=policy_registry_path,
                accounts_config_path=accounts_config_path,
                model_capabilities_path=model_capabilities_path,
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="policy-invalid"
                )
                service = self._build_service(
                    api_key_registry_path=api_key_registry_path,
                    policy_registry_path=policy_registry_path,
                    model_capabilities_path=model_capabilities_path,
                )

                with self.assertRaisesRegex(
                    RequestPolicyRegistryError, "unsupported value"
                ):
                    service.upsert_policy(
                        key_id=api_key["key_id"],
                        group_id="team-a",
                        model_overrides={
                            "gpt-5.4": {
                                "reasoning_effort": {
                                    "mode": "force",
                                    "value": "turbo",
                                }
                            }
                        },
                    )

                with self.assertRaisesRegex(InvalidGroupError, "does not match key"):
                    service.upsert_policy(
                        key_id=api_key["key_id"],
                        group_id="team-b",
                        model_overrides={
                            "gpt-5.4": {
                                "reasoning_effort": {
                                    "mode": "force",
                                    "value": "medium",
                                }
                            }
                        },
                    )

    def test_capability_validation_toggle_can_be_disabled_without_skipping_structure(
        self,
    ):
        """Allows structurally valid policies when overlay validation is disabled.

        Test case: TC-OPENAI-CHATGPT-REQUEST-POLICIES-004
        Requirement: capability validation is toggleable, while structural checks still apply.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "llm_agent_platform" / "provider_configuration" / "openai-chatgpt" / "models.json"
            )
            credentials_path = (
                tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            )
            accounts_config_path = tmp_dir / "openai-chatgpt" / "accounts_config.json"
            credentials_path.parent.mkdir(parents=True, exist_ok=True)
            credentials_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(credentials_path)),
            )

            with self._patched_paths(
                api_key_registry_path=api_key_registry_path,
                policy_registry_path=policy_registry_path,
                accounts_config_path=accounts_config_path,
                model_capabilities_path=model_capabilities_path,
            ):
                api_key = OpenAIChatGPTApiKeyRegistryService().create_key(
                    group_id="team-a", label="policy-toggle"
                )
                service = self._build_service(
                    api_key_registry_path=api_key_registry_path,
                    policy_registry_path=policy_registry_path,
                    model_capabilities_path=model_capabilities_path,
                    capability_validation_enabled=False,
                )

                created = service.upsert_policy(
                    key_id=api_key["key_id"],
                    group_id="team-a",
                    model_overrides={
                        "future-model": {
                            "future_parameter": {
                                "mode": "default_if_absent",
                                "value": "beta",
                            }
                        }
                    },
                )

                self.assertEqual(
                    created.to_payload()["model_overrides"],
                    {
                        "future-model": {
                            "future_parameter": {
                                "mode": "default_if_absent",
                                "value": "beta",
                            }
                        }
                    },
                )

                with self.assertRaisesRegex(
                    RequestPolicyRegistryError, "must not be empty"
                ):
                    service.upsert_policy(
                        key_id=api_key["key_id"],
                        group_id="team-a",
                        model_overrides={},
                    )


if __name__ == "__main__":
    unittest.main()
