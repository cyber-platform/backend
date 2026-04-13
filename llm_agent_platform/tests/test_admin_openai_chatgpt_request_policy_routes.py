"""Admin routes for OpenAI ChatGPT capabilities and request policies.

Suite: TS-ADMIN-OPENAI-CHATGPT-POLICY-ROUTES
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.services.openai_chatgpt_api_keys import (
    OpenAIChatGPTApiKeyRegistryService,
)
from llm_agent_platform.tests.admin_auth_test_utils import install_admin_client_auth

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class AdminOpenAIChatGPTPolicyRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        install_admin_client_auth(self, self.client)

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

    def test_admin_routes_support_model_capability_read_and_policy_lifecycle(self):
        """Exposes provider-scoped capability read and key policy CRUD.

        Test case: TC-ADMIN-OPENAI-CHATGPT-POLICY-ROUTES-001
        Requirement: admin API materializes capability read plus request policy read/upsert/delete semantics.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "provider_configuration" / "openai-chatgpt" / "models.json"
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
                    group_id="team-a", label="policy-route"
                )

                capabilities_response = self.client.get(
                    "/admin/model-capabilities/openai-chatgpt/models/gpt-5.4"
                )
                initial_policy_response = self.client.get(
                    f"/admin/request-policies/openai-chatgpt/keys/{api_key['key_id']}"
                )
                upsert_response = self.client.put(
                    f"/admin/request-policies/openai-chatgpt/keys/{api_key['key_id']}",
                    json={
                        "group_id": "team-a",
                        "model_overrides": {
                            "gpt-5.4": {
                                "reasoning_effort": {
                                    "mode": "force",
                                    "value": "medium",
                                }
                            }
                        },
                    },
                )
                read_after_upsert_response = self.client.get(
                    f"/admin/request-policies/openai-chatgpt/keys/{api_key['key_id']}"
                )
                delete_response = self.client.delete(
                    f"/admin/request-policies/openai-chatgpt/keys/{api_key['key_id']}"
                )

        self.assertEqual(capabilities_response.status_code, 200)
        self.assertEqual(
            capabilities_response.get_json(),
            {
                "provider_id": "openai-chatgpt",
                "model_id": "gpt-5.4",
                "display_name": "GPT-5.4",
                "parameters": {
                    "reasoning_effort": {
                        "supported": True,
                        "values": ["none", "low", "medium", "high", "xhigh"],
                        "default": "none",
                    }
                },
                "drawer": {"raw_capability_payload": {"source": "test"}},
            },
        )
        self.assertEqual(initial_policy_response.status_code, 200)
        self.assertEqual(
            initial_policy_response.get_json(),
            {
                "provider_id": "openai-chatgpt",
                "key_id": api_key["key_id"],
                "group_id": "team-a",
                "model_overrides": {},
            },
        )
        self.assertEqual(upsert_response.status_code, 200)
        upsert_payload = upsert_response.get_json()
        self.assertEqual(upsert_payload["group_id"], "team-a")
        self.assertIn("created_at", upsert_payload)
        self.assertIn("updated_at", upsert_payload)
        self.assertEqual(read_after_upsert_response.status_code, 200)
        self.assertEqual(read_after_upsert_response.get_json(), upsert_payload)
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(
            delete_response.get_json(),
            {
                "provider_id": "openai-chatgpt",
                "key_id": api_key["key_id"],
                "group_id": "team-a",
                "model_overrides": {},
            },
        )

    def test_admin_routes_distinguish_missing_resources_and_invalid_policy_input(self):
        """Maps missing key/model and invalid payloads to predictable admin errors.

        Test case: TC-ADMIN-OPENAI-CHATGPT-POLICY-ROUTES-002
        Requirement: admin API distinguishes missing key/model from invalid request policy input.
        """

        with _secrets_test_dir() as tmp_dir:
            api_key_registry_path = (
                tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
            )
            policy_registry_path = (
                tmp_dir / "openai-chatgpt" / "policy_registry" / "registry.json"
            )
            model_capabilities_path = (
                tmp_dir / "provider_configuration" / "openai-chatgpt" / "models.json"
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
                                    "values": ["none", "low", "medium"],
                                    "default": "none",
                                }
                            },
                            "raw": {},
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
                missing_model_response = self.client.get(
                    "/admin/model-capabilities/openai-chatgpt/models/unknown-model"
                )
                missing_key_read_response = self.client.get(
                    "/admin/request-policies/openai-chatgpt/keys/key_missing"
                )
                missing_key_upsert_response = self.client.put(
                    "/admin/request-policies/openai-chatgpt/keys/key_missing",
                    json={"group_id": "team-a", "model_overrides": {}},
                )
                invalid_upsert_response = self.client.put(
                    f"/admin/request-policies/openai-chatgpt/keys/{api_key['key_id']}",
                    json={"group_id": "team-a", "model_overrides": {}},
                )

        self.assertEqual(missing_model_response.status_code, 404)
        self.assertEqual(
            missing_model_response.get_json(),
            {"error": "Unknown model_id 'unknown-model'"},
        )
        self.assertEqual(missing_key_read_response.status_code, 404)
        self.assertEqual(
            missing_key_read_response.get_json(),
            {"error": "Unknown key_id 'key_missing'"},
        )
        self.assertEqual(missing_key_upsert_response.status_code, 404)
        self.assertEqual(
            missing_key_upsert_response.get_json(),
            {"error": "Unknown key_id 'key_missing'"},
        )
        self.assertEqual(invalid_upsert_response.status_code, 400)
        self.assertIn(
            "model_overrides must not be empty",
            invalid_upsert_response.get_json()["error"],
        )


if __name__ == "__main__":
    unittest.main()
