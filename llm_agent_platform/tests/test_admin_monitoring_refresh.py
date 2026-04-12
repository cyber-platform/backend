"""Admin monitoring refresh tests.

Suite: TS-ADMIN-MONITORING-REFRESH
"""

import json
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.services.account_router import quota_account_router
from llm_agent_platform.services.openai_chatgpt_admin_monitoring import (
    get_usage_windows,
    reset_monitoring_caches,
)

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _tmp_state_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


class AdminMonitoringRefreshTests(unittest.TestCase):
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
    def _openai_accounts_config(credentials_a: str, credentials_b: str) -> dict:
        return {
            "mode": "rounding",
            "active_account": "acct-1",
            "all_accounts": ["acct-1", "acct-2"],
            "accounts": {
                "acct-1": {"credentials_path": credentials_a},
                "acct-2": {"credentials_path": credentials_b},
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
        return """monitoring:\n  background_poll_interval_seconds: 3600\n  refresh_concurrency: 2\n  per_account_refresh_timeout_seconds: 1\n  refresh_run_timeout_seconds: 5\n  refresh_run_retention_seconds: 300\nfrontend:\n  page_poll_interval_ms: 10000\n  manual_refresh_status_poll_interval_ms: 1000\n"""

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

    def _seed_files(self, tmp_dir: Path) -> Path:
        creds_a = tmp_dir / "openai-chatgpt" / "auth" / "acct-1.json"
        creds_b = tmp_dir / "openai-chatgpt" / "auth" / "acct-2.json"
        creds_a.parent.mkdir(parents=True, exist_ok=True)
        creds_a.write_text("{}", encoding="utf-8")
        creds_b.write_text("{}", encoding="utf-8")
        accounts_config_path = tmp_dir / "openai_accounts_config.json"
        self._write_json(
            accounts_config_path,
            self._openai_accounts_config(str(creds_a), str(creds_b)),
        )
        return accounts_config_path

    def _poll_terminal_status(
        self, refresh_id: str, timeout_seconds: float = 2.0
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = self.client.get(
                f"/admin/monitoring/openai-chatgpt/refresh/{refresh_id}"
            )
            payload = response.get_json()
            if payload["status"] in {"completed", "completed_with_errors", "error"}:
                return payload
            time.sleep(0.02)
        self.fail("Refresh run did not reach terminal state")

    def test_start_manual_refresh_reuses_active_run_id(self):
        """Reuses the active refresh run for duplicate manual refresh requests.

        Test case: TC-ADMIN-MONITORING-REFRESH-001
        Requirement: repeated POST /refresh during an active run returns the same refresh_id.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)
            release_refresh = False

            def fetch_snapshot(adapter_self):
                nonlocal release_refresh
                while not release_refresh:
                    time.sleep(0.01)
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "account_id": adapter_self.credentials_path,
                    "as_of": "2026-01-01T00:00:00Z",
                    "limits": {
                        "primary": {"used_percent": 12, "window": "60m"},
                        "secondary": {"used_percent": 24, "window": "10080m"},
                    },
                }

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                first_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/refresh"
                )
                second_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/refresh"
                )
                release_refresh = True
                terminal_payload = self._poll_terminal_status(
                    first_response.get_json()["refresh_id"]
                )

        self.assertEqual(first_response.status_code, 202)
        self.assertEqual(second_response.status_code, 202)
        self.assertEqual(
            first_response.get_json()["refresh_id"],
            second_response.get_json()["refresh_id"],
        )
        self.assertEqual(terminal_payload["status"], "completed")

    def test_refresh_status_progresses_to_completed(self):
        """Exposes running progress before the refresh reaches a terminal status.

        Test case: TC-ADMIN-MONITORING-REFRESH-002
        Requirement: refresh status endpoint progresses from queued or running to a terminal status.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)
            release_refresh = False

            def fetch_snapshot(_adapter_self):
                nonlocal release_refresh
                while not release_refresh:
                    time.sleep(0.01)
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-01-01T00:00:00Z",
                    "limits": {
                        "primary": {"used_percent": 5, "window": "60m"},
                        "secondary": {"used_percent": 10, "window": "10080m"},
                    },
                }

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                start_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/refresh"
                )
                refresh_id = start_response.get_json()["refresh_id"]
                status_response = self.client.get(
                    f"/admin/monitoring/openai-chatgpt/refresh/{refresh_id}"
                )
                release_refresh = True
                terminal_payload = self._poll_terminal_status(refresh_id)

        running_payload = status_response.get_json()
        self.assertEqual(start_response.status_code, 202)
        self.assertIn(running_payload["status"], {"queued", "running"})
        self.assertEqual(terminal_payload["status"], "completed")
        self.assertEqual(terminal_payload["successful_accounts"], 2)

    def test_refresh_completed_with_errors_when_subset_fails(self):
        """Aggregates mixed per-account outcomes into completed_with_errors.

        Test case: TC-ADMIN-MONITORING-REFRESH-003
        Requirement: refresh run returns completed_with_errors when only a subset of accounts fail.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)

            def fetch_snapshot(adapter_self):
                if str(adapter_self.credentials_path).endswith("acct-2.json"):
                    raise RuntimeError("upstream usage failed")
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-01-01T00:00:00Z",
                    "limits": {
                        "primary": {"used_percent": 8, "window": "60m"},
                        "secondary": {"used_percent": 16, "window": "10080m"},
                    },
                }

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                response = self.client.post("/admin/monitoring/openai-chatgpt/refresh")
                payload = self._poll_terminal_status(response.get_json()["refresh_id"])
                acct_2_usage = get_usage_windows("acct-2")

        self.assertEqual(payload["status"], "completed_with_errors")
        self.assertEqual(payload["successful_accounts"], 1)
        self.assertEqual(payload["failed_accounts"], 1)
        self.assertEqual(acct_2_usage["refresh"]["status"], "error")

    def test_refresh_returns_error_when_all_accounts_fail(self):
        """Marks the whole run as error when every account refresh fails.

        Test case: TC-ADMIN-MONITORING-REFRESH-004
        Requirement: refresh run returns error when all accounts fail.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)

            def fetch_snapshot(_adapter_self):
                raise RuntimeError("all accounts failed")

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                response = self.client.post("/admin/monitoring/openai-chatgpt/refresh")
                payload = self._poll_terminal_status(response.get_json()["refresh_id"])

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["successful_accounts"], 0)
        self.assertEqual(payload["failed_accounts"], 2)

    def test_provider_page_exposes_manual_refresh_capability_and_group_summary(self):
        """Materializes provider page fields needed by the operator short-polling UX.

        Test case: TC-ADMIN-MONITORING-REFRESH-005
        Requirement: provider page exposes supports_manual_refresh and grouped refresh summary fields.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)

            def fetch_snapshot(_adapter_self):
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-01-01T00:00:00Z",
                    "limits": {
                        "primary": {"used_percent": 6, "window": "60m"},
                        "secondary": {"used_percent": 12, "window": "10080m"},
                    },
                }

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                start_response = self.client.post(
                    "/admin/monitoring/openai-chatgpt/refresh"
                )
                refresh_id = start_response.get_json()["refresh_id"]
                self._poll_terminal_status(refresh_id)
                page_response = self.client.get("/admin/monitoring/openai-chatgpt")

        payload = page_response.get_json()
        self.assertEqual(page_response.status_code, 200)
        self.assertTrue(payload["capabilities"]["supports_manual_refresh"])
        self.assertEqual(payload["groups"][0]["refresh"]["refresh_id"], refresh_id)
        self.assertEqual(payload["groups"][0]["refresh"]["status"], "completed")
        self.assertEqual(payload["groups"][0]["refresh"]["successful_accounts"], 2)

    def test_refresh_normalizes_single_upstream_window_without_falling_back_to_stale(
        self,
    ):
        """Preserves single-window upstream snapshots instead of collapsing to stale defaults.

        Test case: TC-ADMIN-MONITORING-REFRESH-006
        Requirement: a partial usage-limits snapshot still materializes a fresh monitoring payload.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)

            def fetch_snapshot(_adapter_self):
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-01-01T00:00:00Z",
                    "limits": {
                        "primary": {
                            "used_percent": 18,
                            "window": "10080m",
                            "metadata": {"limit_window_seconds": 604800},
                        }
                    },
                }

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
            ):
                response = self.client.post("/admin/monitoring/openai-chatgpt/refresh")
                self._poll_terminal_status(response.get_json()["refresh_id"])
                acct_1_usage = get_usage_windows("acct-1")

        self.assertEqual(acct_1_usage["refresh"]["status"], "fresh")
        self.assertEqual(acct_1_usage["short_window"]["window_minutes"], 1)
        self.assertEqual(acct_1_usage["long_window"]["window_minutes"], 10080)
        self.assertEqual(acct_1_usage["long_window"]["used_percent"], 18.0)

    def test_refresh_updates_runtime_before_persistence_flush(self):
        """Keeps live runtime state fresh even when the durability flush fails.

        Test case: TC-ADMIN-MONITORING-REFRESH-007
        Requirement: refresh mutates memory first and persistence remains secondary.
        """

        with _tmp_state_dir() as tmp_dir:
            accounts_config_path = self._seed_files(tmp_dir)
            self._write_json(
                tmp_dir
                / "openai-chatgpt"
                / "accounts"
                / "acct-1"
                / "usage_windows.json",
                {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "short_window": {"used_percent": 7.0, "window_minutes": 60},
                    "long_window": {"used_percent": 14.0, "window_minutes": 10080},
                    "refresh": {
                        "last_refreshed_at": "2026-01-01T00:00:00Z",
                        "next_refresh_at": "2026-01-01T00:10:00Z",
                        "refresh_interval_seconds": 600,
                        "status": "ok",
                        "last_error": None,
                    },
                },
            )

            def fetch_snapshot(_adapter_self):
                return {
                    "version": 1,
                    "provider_id": "openai-chatgpt",
                    "as_of": "2026-01-02T00:00:00Z",
                    "limits": {
                        "primary": {"used_percent": 61, "window": "60m"},
                        "secondary": {"used_percent": 91, "window": "10080m"},
                    },
                }

            def fail_usage_windows_write(path, payload):
                if path.name == "usage_windows.json":
                    raise OSError("disk flush failed")
                self.fail(f"unexpected persisted write: {path}")

            with (
                self._patched_paths(tmp_dir, accounts_config_path),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring.OpenAIChatGptUsageLimitsAdapter.fetch_snapshot",
                    new=fetch_snapshot,
                ),
                patch(
                    "llm_agent_platform.services.openai_chatgpt_admin_monitoring._write_json",
                    new=fail_usage_windows_write,
                ),
            ):
                response = self.client.post("/admin/monitoring/openai-chatgpt/refresh")
                payload = self._poll_terminal_status(response.get_json()["refresh_id"])
                runtime_usage = get_usage_windows("acct-1")
                page_response = self.client.get("/admin/monitoring/openai-chatgpt")

            persisted_usage = json.loads(
                (
                    tmp_dir
                    / "openai-chatgpt"
                    / "accounts"
                    / "acct-1"
                    / "usage_windows.json"
                ).read_text(encoding="utf-8")
            )
        page_payload = page_response.get_json()
        acct_1 = next(
            item
            for item in page_payload["groups"][0]["accounts"]
            if item["account_name"] == "acct-1"
        )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(runtime_usage["short_window"]["used_percent"], 61.0)
        self.assertEqual(runtime_usage["refresh"]["status"], "fresh")
        self.assertEqual(acct_1["short_window"]["used_percent"], 61.0)
        self.assertEqual(persisted_usage["short_window"]["used_percent"], 7.0)


if __name__ == "__main__":
    unittest.main()
