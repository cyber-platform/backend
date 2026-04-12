"""Admin API key registry tests.

Suite: TS-ADMIN-API-KEYS
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

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class AdminApiKeyTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

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
    def _patched_paths(self, registry_path: Path, accounts_config_path: Path):
        with (
            patch(
                "llm_agent_platform.services.account_router.OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH",
                str(accounts_config_path),
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

    def test_service_persists_hash_only_and_revoked_key_is_not_lookupable(self):
        """Stores only the key hash and excludes revoked keys from lookup.

        Test case: TC-ADMIN-API-KEYS-001
        Requirement: registry stores hashes only and revocation disables future authorization.
        """

        with _secrets_test_dir() as tmp_dir:
            registry_path = tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
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

            with self._patched_paths(registry_path, accounts_config_path):
                service = OpenAIChatGPTApiKeyRegistryService()
                created = service.create_key(group_id="team-a", label="demo-default")

                registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
                self.assertEqual(registry_payload["provider_id"], "openai-chatgpt")
                self.assertEqual(len(registry_payload["keys"]), 1)
                self.assertNotIn("raw_api_key", registry_payload["keys"][0])
                self.assertNotIn(created["raw_api_key"], json.dumps(registry_payload))
                self.assertTrue(
                    registry_payload["keys"][0]["key_hash"].startswith("sha256:")
                )

                lookup_before_revoke = service.lookup_active_key(
                    created["raw_api_key"], group_id="team-a"
                )
                self.assertIsNotNone(lookup_before_revoke)
                if lookup_before_revoke is None:
                    self.fail("lookup_before_revoke must not be None")
                self.assertEqual(lookup_before_revoke["key_id"], created["key_id"])

                revoke_response = service.revoke_key(created["key_id"])
                self.assertEqual(
                    revoke_response,
                    {"ok": True, "key_id": created["key_id"], "status": "revoked"},
                )

                lookup_after_revoke = service.lookup_active_key(
                    created["raw_api_key"], group_id="team-a"
                )
                self.assertIsNone(lookup_after_revoke)

    def test_admin_routes_support_create_list_and_revoke(self):
        """Implements the admin CRUD flow for provider-scoped API keys.

        Test case: TC-ADMIN-API-KEYS-002
        Requirement: admin API exposes list, create, and revoke payloads compatible with PoC contracts.
        """

        with _secrets_test_dir() as tmp_dir:
            registry_path = tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
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

            with self._patched_paths(registry_path, accounts_config_path):
                create_response = self.client.post(
                    "/admin/api-keys/openai-chatgpt",
                    json={"group_id": "team-a", "label": "frontend-demo"},
                )
                self.assertEqual(create_response.status_code, 200)
                created_payload = create_response.get_json()
                self.assertEqual(created_payload["provider_id"], "openai-chatgpt")
                self.assertEqual(created_payload["group_id"], "team-a")
                self.assertEqual(created_payload["status"], "active")
                self.assertIn("raw_api_key", created_payload)

                other_group_create = self.client.post(
                    "/admin/api-keys/openai-chatgpt",
                    json={"group_id": "team-b", "label": "other-group"},
                )
                self.assertEqual(other_group_create.status_code, 200)

                list_response = self.client.get(
                    "/admin/api-keys/openai-chatgpt",
                    query_string={"group_id": "team-a"},
                )
                self.assertEqual(list_response.status_code, 200)
                list_payload = list_response.get_json()
                self.assertEqual(list_payload["provider_id"], "openai-chatgpt")
                self.assertEqual(list_payload["group_id"], "team-a")
                self.assertEqual(len(list_payload["keys"]), 1)
                self.assertEqual(
                    list_payload["keys"][0]["key_id"], created_payload["key_id"]
                )
                self.assertNotIn("raw_api_key", list_payload["keys"][0])

                revoke_response = self.client.post(
                    f"/admin/api-keys/openai-chatgpt/{created_payload['key_id']}/revoke"
                )
                self.assertEqual(revoke_response.status_code, 200)
                self.assertEqual(
                    revoke_response.get_json(),
                    {
                        "ok": True,
                        "key_id": created_payload["key_id"],
                        "status": "revoked",
                    },
                )

                list_after_revoke = self.client.get(
                    "/admin/api-keys/openai-chatgpt",
                    query_string={"group_id": "team-a"},
                )
                self.assertEqual(list_after_revoke.status_code, 200)
                self.assertEqual(
                    list_after_revoke.get_json()["keys"][0]["status"], "revoked"
                )

    def test_admin_routes_reject_unknown_group(self):
        """Rejects create and list requests for undeclared provider groups.

        Test case: TC-ADMIN-API-KEYS-003
        Requirement: admin API must not create keys for unknown group_id values.
        """

        with _secrets_test_dir() as tmp_dir:
            registry_path = tmp_dir / "openai-chatgpt" / "api-keys" / "registry.json"
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

            with self._patched_paths(registry_path, accounts_config_path):
                create_response = self.client.post(
                    "/admin/api-keys/openai-chatgpt",
                    json={"group_id": "missing-group", "label": "invalid"},
                )
                self.assertEqual(create_response.status_code, 400)
                create_payload = create_response.get_json()
                self.assertIsNotNone(create_payload)
                self.assertIn("Unknown group_id", create_payload["error"])

                list_response = self.client.get(
                    "/admin/api-keys/openai-chatgpt",
                    query_string={"group_id": "missing-group"},
                )
                self.assertEqual(list_response.status_code, 400)
                list_payload = list_response.get_json()
                self.assertIsNotNone(list_payload)
                self.assertIn("Unknown group_id", list_payload["error"])
