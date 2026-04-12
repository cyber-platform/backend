"""Admin monitoring read-model tests.

Suite: TS-ADMIN-MONITORING-READ-MODEL
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.services.account_router import quota_account_router
from llm_agent_platform.services.openai_chatgpt_admin_monitoring import (
    reset_monitoring_caches,
)

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _tmp_state_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class AdminMonitoringReadModelTests(unittest.TestCase):
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
    def _iso_now(offset_seconds: int = 0) -> str:
        return (
            (datetime.now(tz=timezone.utc) + timedelta(seconds=offset_seconds))
            .isoformat()
            .replace("+00:00", "Z")
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

    @contextmanager
    def _patched_paths(self, tmp_dir: Path, accounts_config_path: Path):
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
                "llm_agent_platform.services.service_behavior_config.SERVICE_BEHAVIOR_CONFIG_PATH",
                str(service_behavior_path),
            ),
        ):
            yield

    def _seed_monitoring_files(self, tmp_dir: Path) -> None:
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-1" / "usage_windows.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "account_id": "upstream-acct-1",
                "short_window": {
                    "used_percent": 12.5,
                    "window_minutes": 60,
                    "reset_at": self._iso_now(900),
                    "source": "wham.primary_window",
                },
                "long_window": {
                    "used_percent": 44.0,
                    "window_minutes": 10080,
                    "reset_at": self._iso_now(7200),
                    "source": "wham.secondary_window",
                },
                "refresh": {
                    "last_refreshed_at": self._iso_now(-30),
                    "next_refresh_at": self._iso_now(270),
                    "refresh_interval_seconds": 300,
                    "status": "ok",
                    "last_error": None,
                },
            },
        )
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-2" / "usage_windows.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "short_window": {
                    "used_percent": 3.0,
                    "window_minutes": 60,
                },
                "long_window": {
                    "used_percent": 10.0,
                    "window_minutes": 10080,
                },
                "refresh": {
                    "last_refreshed_at": self._iso_now(-20),
                    "next_refresh_at": self._iso_now(280),
                    "refresh_interval_seconds": 300,
                    "status": "ok",
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
                    "total_requests": 9,
                    "successful_requests": 8,
                    "failed_requests": 1,
                    "last_request_at": self._iso_now(-15),
                },
                "aggregate_tokens": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "last_request_summary": {
                    "requested_model": "gpt-5.4",
                    "status": "success",
                    "requested_at": self._iso_now(-15),
                    "tokens": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                },
            },
        )
        self._write_json(
            tmp_dir / "openai-chatgpt" / "accounts" / "acct-2" / "request_usage.json",
            {
                "version": 1,
                "provider_id": "openai-chatgpt",
                "request_counters": {
                    "total_requests": 2,
                    "successful_requests": 2,
                    "failed_requests": 0,
                    "last_request_at": self._iso_now(-10),
                },
            },
        )

    def test_provider_list_returns_only_openai_chatgpt_for_poc(self):
        """Exposes the single-provider monitoring shell for the current PoC.

        Test case: TC-ADMIN-MONITORING-READ-MODEL-001
        Requirement: provider list for the PoC returns only openai-chatgpt.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            creds_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                response = self.client.get("/admin/monitoring/providers")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["providers"]), 1)
        self.assertEqual(payload["providers"][0]["provider_id"], "openai-chatgpt")
        self.assertEqual(payload["providers"][0]["account_count"], 2)
        self.assertTrue(payload["providers"][0]["has_monitoring"])

    def test_provider_page_materializes_openai_chatgpt_read_model(self):
        """Builds the provider page from runtime state, group config, and monitoring snapshots.

        Test case: TC-ADMIN-MONITORING-READ-MODEL-002
        Requirement: openai-chatgpt provider page matches the provider-specific admin contract.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            creds_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )
            self._seed_monitoring_files(tmp_dir)
            self._write_json(
                tmp_dir
                / "openai-chatgpt"
                / "accounts"
                / "acct-1"
                / "account_state.json",
                {"version": 1, "last_used_at": self._iso_now(-300)},
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                response = self.client.get("/admin/monitoring/openai-chatgpt")

        payload = response.get_json()
        account_rows = payload["groups"][0]["accounts"]
        acct_1 = next(item for item in account_rows if item["account_name"] == "acct-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["provider_id"], "openai-chatgpt")
        self.assertTrue(payload["capabilities"]["supports_activate"])
        self.assertTrue(payload["capabilities"]["supports_manual_refresh"])
        self.assertEqual(payload["security_boundary"]["auth_mode"], "none")
        self.assertEqual(payload["groups"][0]["group_id"], "team-a")
        self.assertEqual(payload["groups"][0]["refresh"]["status"], "idle")
        self.assertEqual(acct_1["short_window"]["used_percent"], 12.5)
        self.assertEqual(acct_1["long_window"]["window_minutes"], 10080)
        self.assertEqual(
            acct_1["drawer"]["raw_request_usage_payload"]["request_counters"][
                "total_requests"
            ],
            9,
        )
        self.assertEqual(acct_1["routing"]["state"], "ready")
        self.assertFalse(acct_1.get("is_preferred_for_session", False))

    def test_activate_marks_preferred_account_without_bypassing_cooldown(self):
        """Stores a session-scoped preferred account override while keeping routing safeguards active.

        Test case: TC-ADMIN-MONITORING-READ-MODEL-003
        Requirement: activate marks the preferred account for the session and does not bypass cooldown safeguards.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            creds_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )
            self._seed_monitoring_files(tmp_dir)
            self._write_json(
                tmp_dir
                / "openai-chatgpt"
                / "accounts"
                / "acct-1"
                / "account_state.json",
                {
                    "version": 1,
                    "cooldown": {"last_cooldown_at": self._iso_now(-5)},
                },
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                activate_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/groups/team-a/accounts/acct-1/activate"
                )
                page_response = self.client.get("/admin/monitoring/openai-chatgpt")
                selected = quota_account_router.select_account(
                    "openai-chatgpt", group_id="team-a"
                )

        activate_payload = activate_response.get_json()
        page_payload = page_response.get_json()
        account_rows = page_payload["groups"][0]["accounts"]
        acct_1 = next(item for item in account_rows if item["account_name"] == "acct-1")

        self.assertEqual(activate_response.status_code, 200)
        self.assertEqual(activate_payload["group_id"], "team-a")
        self.assertEqual(
            activate_payload["activate_semantics"],
            "session_scoped_in_memory_preferred_account_override",
        )
        self.assertEqual(page_response.status_code, 200)
        self.assertTrue(acct_1["is_preferred_for_session"])
        self.assertEqual(acct_1["status"], "cooldown")
        self.assertEqual(acct_1["routing"]["block_reason"], "rate_limit_cooldown")
        self.assertEqual(selected.account.name, "acct-2")

    def test_provider_page_keeps_runtime_snapshot_after_persisted_files_change(self):
        """Keeps serving hydrated runtime state instead of rereading persisted files.

        Test case: TC-ADMIN-MONITORING-READ-MODEL-004
        Requirement: admin monitoring page reads process-local runtime state after startup hydration.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            creds_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )
            self._seed_monitoring_files(tmp_dir)

            with self._patched_paths(tmp_dir, accounts_config_path):
                first_response = self.client.get("/admin/monitoring/openai-chatgpt")
                self._write_json(
                    tmp_dir
                    / "openai-chatgpt"
                    / "accounts"
                    / "acct-1"
                    / "usage_windows.json",
                    {
                        "version": 1,
                        "provider_id": "openai-chatgpt",
                        "short_window": {"used_percent": 99.0, "window_minutes": 60},
                        "long_window": {"used_percent": 88.0, "window_minutes": 10080},
                        "refresh": {
                            "last_refreshed_at": self._iso_now(-5),
                            "next_refresh_at": self._iso_now(300),
                            "refresh_interval_seconds": 300,
                            "status": "ok",
                            "last_error": None,
                        },
                    },
                )
                self._write_json(
                    tmp_dir
                    / "openai-chatgpt"
                    / "accounts"
                    / "acct-1"
                    / "request_usage.json",
                    {
                        "version": 1,
                        "provider_id": "openai-chatgpt",
                        "request_counters": {
                            "total_requests": 999,
                            "successful_requests": 999,
                            "failed_requests": 0,
                            "last_request_at": self._iso_now(-1),
                        },
                    },
                )
                second_response = self.client.get("/admin/monitoring/openai-chatgpt")

        first_payload = first_response.get_json()
        second_payload = second_response.get_json()
        first_acct_1 = next(
            item
            for item in first_payload["groups"][0]["accounts"]
            if item["account_name"] == "acct-1"
        )
        second_acct_1 = next(
            item
            for item in second_payload["groups"][0]["accounts"]
            if item["account_name"] == "acct-1"
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(first_acct_1["short_window"]["used_percent"], 12.5)
        self.assertEqual(second_acct_1["short_window"]["used_percent"], 12.5)
        self.assertEqual(
            second_acct_1["drawer"]["raw_request_usage_payload"]["request_counters"][
                "total_requests"
            ],
            9,
        )

    def test_provider_page_ignores_legacy_limits_files_on_read_path(self):
        """Does not use legacy limits.json as a live monitoring input.

        Test case: TC-ADMIN-MONITORING-READ-MODEL-005
        Requirement: admin monitoring read-model does not depend on legacy limits.json.
        """

        with _tmp_state_dir() as tmp_dir:
            creds_path = tmp_dir / "openai-chatgpt" / "auth" / "oauth-account.json"
            accounts_config_path = tmp_dir / "openai_accounts_config.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            creds_path.write_text("{}", encoding="utf-8")
            self._write_json(
                accounts_config_path,
                self._openai_accounts_config(str(creds_path)),
            )
            self._write_json(
                tmp_dir
                / "openai-chatgpt"
                / "usage"
                / "accounts"
                / "acct-1"
                / "limits.json",
                {
                    "account_id": "legacy-acct-1",
                    "as_of": self._iso_now(-30),
                    "limits": {
                        "primary": {"used_percent": 77, "window": "60m"},
                        "secondary": {"used_percent": 55, "window": "10080m"},
                    },
                    "metadata": {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "total_tokens": 3,
                        }
                    },
                },
            )

            with self._patched_paths(tmp_dir, accounts_config_path):
                response = self.client.get("/admin/monitoring/openai-chatgpt")

        payload = response.get_json()
        acct_1 = next(
            item
            for item in payload["groups"][0]["accounts"]
            if item["account_name"] == "acct-1"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(acct_1["short_window"]["used_percent"], 0.0)
        self.assertEqual(acct_1["long_window"]["used_percent"], 0.0)
        self.assertEqual(
            acct_1["drawer"]["raw_request_usage_payload"]["request_counters"][
                "total_requests"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
