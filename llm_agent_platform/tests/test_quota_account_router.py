import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from zoneinfo import ZoneInfo

from llm_agent_platform.services.account_router import (
    AccountRouterError,
    AllAccountsExhaustedError,
    QuotaAccountRouter,
    RotationEvent,
)
from llm_agent_platform.services.account_state_store import (
    AsyncStateWriter,
    AccountStatePaths,
    save_quota_exhausted_at,
    save_last_used_at,
    state_writer,
)

SECRETS_TEST_ROOT = Path("secrets_test")


@contextmanager
def _secrets_test_dir():
    SECRETS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SECRETS_TEST_ROOT) as tmp:
        yield Path(tmp)


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


def _write_json(path: Path, payload: dict) -> None:
    _seed_credentials_from_config(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class QuotaAccountRouterTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _patched_paths(base_dir: Path, gemini_path: Path, qwen_path: Path):
        with (
            patch("llm_agent_platform.services.account_router.GEMINI_ACCOUNTS_CONFIG_PATH", str(gemini_path)),
            patch("llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH", str(qwen_path)),
            patch("llm_agent_platform.services.account_router.STATE_DIR", str(base_dir)),
            patch("llm_agent_platform.services.account_state_store.STATE_DIR", str(base_dir)),
            patch(
                "llm_agent_platform.services.account_router.AccountStatePaths",
                new=lambda provider_id, account_name, root_dir=Path("."): AccountStatePaths(
                    provider_id=provider_id,
                    account_name=account_name,
                    root_dir=base_dir,
                ),
            ),
        ):
            yield

    def _make_router(
        self,
        base_dir: Path,
        gemini_cfg: dict | None = None,
        qwen_cfg: dict | None = None,
    ) -> QuotaAccountRouter:
        gemini_cfg = gemini_cfg or {
            "mode": "single",
            "active_account": "lisa",
            "all_accounts": ["lisa", "petr"],
            "accounts": {
                "lisa": {
                    "credentials_path": "secrets_test/accounts/gemini_lisa.json",
                    "project_id": "project-lisa",
                },
                "petr": {
                    "credentials_path": "secrets_test/accounts/gemini_petr.json",
                    "project_id": "project-petr",
                },
            },
            "rotation_policy": {
                "rate_limit_threshold": 2,
                "quota_exhausted_threshold": 2,
                "rate_limit_cooldown_seconds": 5,
            },
            "model_quota_resets": {
                "default": "00:00:00",
                "gemini-3-flash-preview": "00:10:00",
            },
        }
        qwen_cfg = qwen_cfg or {
            "mode": "single",
            "active_account": "lisa",
            "all_accounts": ["lisa", "petr"],
            "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
            "rotation_policy": {
                "rate_limit_threshold": 2,
                "quota_exhausted_threshold": 2,
                "rate_limit_cooldown_seconds": 5,
            },
            "model_quota_resets": {
                "default": "00:00:00",
                "qwen-coder-model": "00:08:00",
            },
        }

        gemini_path = base_dir / "gemini_accounts_config.json"
        qwen_path = base_dir / "qwen_accounts_config.json"
        _seed_credentials_from_config(gemini_cfg)
        _seed_credentials_from_config(qwen_cfg)
        _write_json(gemini_path, gemini_cfg)
        _write_json(qwen_path, qwen_cfg)

        with self._patched_paths(base_dir, gemini_path, qwen_path):
            router = QuotaAccountRouter()
        return router

    def test_single_mode_uses_active_account(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            gemini_path = tmp_dir / "gemini_accounts_config.json"
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            with self._patched_paths(tmp_dir, gemini_path, qwen_path):
                gemini_selected = router.select_account("gemini_cli")
                qwen_selected = router.select_account("qwen_code")

            self.assertEqual(gemini_selected.account.name, "lisa")
            self.assertEqual(gemini_selected.pool, ["lisa"])
            self.assertEqual(getattr(gemini_selected.account, "project_id", None), "project-lisa")

            self.assertEqual(qwen_selected.account.name, "lisa")
            self.assertEqual(qwen_selected.pool, ["lisa"])

    def test_rounding_rotates_after_two_limit_errors(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                selected = router.select_account("qwen_code", "qwen-coder-model")
            self.assertEqual(selected.account.name, "lisa")

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                first_event = router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                )
                self.assertFalse(first_event.switched)

                second_event = router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                )
                self.assertTrue(second_event.switched)
                self.assertFalse(second_event.all_exhausted)

                next_selected = router.select_account("qwen_code", "qwen-coder-model")
            self.assertEqual(next_selected.account.name, "petr")

    def test_rounding_marks_exhausted_only_for_quota_exhausted(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                selected = router.select_account("qwen_code", "qwen-coder-model")

                router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                second = router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                self.assertTrue(second.switched)
                self.assertFalse(second.all_exhausted)

                next_selected = router.select_account("qwen_code", "qwen-coder-model")
            self.assertEqual(next_selected.account.name, "petr")

    def test_all_accounts_exceed_quota_error(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                selected = router.select_account("qwen_code", "qwen-coder-model")

                router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                router.register_event(
                    provider="qwen_code",
                    account_name="petr",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                router.register_event(
                    provider="qwen_code",
                    account_name="petr",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )

                self.assertTrue(
                    router.all_accounts_exhausted("qwen_code", selected.pool, "qwen-coder-model")
                )

                with self.assertRaises(AllAccountsExhaustedError):
                    router.select_account("qwen_code", "qwen-coder-model")

    def test_model_reset_timestamp_uses_vladivostok_schedule(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "qwen-coder-model": "00:08:00",
                    "default": "00:00:00",
                },
            }
            _write_json(qwen_path, cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                selected = router.select_account("qwen_code", "qwen-coder-model")

                router.register_event(
                    provider="qwen_code",
                    account_name=selected.account.name,
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                router.register_event(
                    provider="qwen_code",
                    account_name=selected.account.name,
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )

                self.assertTrue(
                    router.all_accounts_exhausted("qwen_code", [selected.account.name], "qwen-coder-model")
                )

                now_local = datetime.now(ZoneInfo("Asia/Vladivostok"))
                self.assertIsNotNone(now_local)

    def test_period_based_reset_uses_days_hours_minutes(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "01:02:30",
                },
            }
            _write_json(qwen_path, cfg)

            exhausted_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            with patch("llm_agent_platform.services.account_router.QWEN_ACCOUNTS_CONFIG_PATH", str(qwen_path)):
                provider_cfg = router._load_provider_config("qwen_code")

            until_ts = router._quota_reset_timestamp(
                provider_cfg,
                "qwen-coder-model",
                exhausted_at=exhausted_at,
            )
            expected = exhausted_at + timedelta(days=1, hours=2, minutes=30)
            self.assertEqual(until_ts, expected.timestamp())

    def test_persisted_exhausted_state_survives_restart(self):
        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:01:00",
                },
            }
            _write_json(qwen_path, cfg)

            exhausted_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            state_paths = AccountStatePaths(
                provider_id="qwen_code",
                account_name="lisa",
                root_dir=tmp_dir,
            )
            save_quota_exhausted_at(state_paths, "qwen-coder-model", exhausted_at)
            state_writer.flush_once()

            now_ts = exhausted_at.timestamp() + 30
            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.time.time", return_value=now_ts),
            ):
                router = QuotaAccountRouter()
                with self.assertRaises(AllAccountsExhaustedError):
                    router.select_account("qwen_code", "qwen-coder-model")

    def test_per_provider_scope_uses_provider_sentinel(self):
        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa"],
                "quota_scope": "per_provider",
                "accounts": {
                    "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 1,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:10:00",
                },
            }
            _write_json(qwen_path, cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                router = QuotaAccountRouter()
                selected = router.select_account("qwen_code", "qwen-coder-model")
                result = router.register_event(
                    provider="qwen_code",
                    account_name=selected.account.name,
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )
                state_writer.flush_once()

            self.assertTrue(result.all_exhausted)
            state_payload = json.loads(
                (tmp_dir / "qwen_code" / "accounts" / "lisa" / "account_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                sorted(state_payload["quota_exhausted"]["keys"].keys()),
                ["__provider__"],
            )

    def test_group_snapshot_is_written_to_state_dir(self):
        with _secrets_test_dir() as tmp_dir:
            qwen_path = tmp_dir / "qwen_accounts_config.json"

            cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "groups": {
                    "g1": {
                        "accounts": ["lisa", "petr"],
                        "models": ["qwen-coder-model"],
                    },
                },
                "accounts": {
                    "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                    "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
                },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                },
                "model_quota_resets": {
                    "default": "00:10:00",
                    "qwen-coder-model": "00:10:00",
                },
            }
            _write_json(qwen_path, cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                router = QuotaAccountRouter()
                selected = router.select_account("qwen_code", "qwen-coder-model", group_id="g1")
                router.register_event(
                    provider="qwen_code",
                    account_name=selected.account.name,
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                    group_id="g1",
                )
                state_writer.flush_once()

            snapshot_path = tmp_dir / "qwen_code" / "groups" / "g1" / "quota_state.json"
            self.assertTrue(snapshot_path.exists())
            snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot_payload["group_id"], "g1")
            self.assertEqual(snapshot_payload["total_accounts"], 2)
            self.assertEqual(snapshot_payload["cooldown_accounts"], 1)
            self.assertIn("models", snapshot_payload)

    def test_async_state_writer_close_flushes_pending_payload(self):
        with _secrets_test_dir() as tmp_dir:
            writer = AsyncStateWriter(flush_interval_seconds=3600, max_pending_files=16)
            paths = AccountStatePaths(
                provider_id="qwen_code",
                account_name="lisa",
                root_dir=tmp_dir,
            )

            save_last_used_at(
                paths,
                datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                writer=writer,
            )
            writer.close()

            payload = json.loads(paths.account_state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_used_at"], "2026-01-01T00:00:00Z")

    def test_rounding_random_order_switches_using_rng(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr", "ivan", "kate"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
                "ivan": {"credentials_path": "secrets_test/accounts/qwen_ivan.json"},
                "kate": {"credentials_path": "secrets_test/accounts/qwen_kate.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": True,
                    "rotate_after_n_successes": 0,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.random.choice", return_value="ivan") as mock_choice,
            ):
                selected = router.select_account("qwen_code", "qwen-coder-model")
            self.assertEqual(selected.account.name, "ivan")
            self.assertTrue(mock_choice.called)

            router._state[("qwen_code", "g0")].has_selected = False
            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.random.choice", return_value="kate") as mock_choice,
            ):
                event = router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                )
                self.assertTrue(event.switched)
                self.assertTrue(mock_choice.called)
                next_selected = router.select_account("qwen_code", "qwen-coder-model")

            self.assertEqual(next_selected.account.name, "kate")

    def test_rounding_random_order_excludes_exhausted_accounts(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr", "ivan", "kate"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
                "ivan": {"credentials_path": "secrets_test/accounts/qwen_ivan.json"},
                "kate": {"credentials_path": "secrets_test/accounts/qwen_kate.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 1,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": True,
                    "rotate_after_n_successes": 0,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.random.choice", return_value="lisa") as mock_choice,
            ):
                selected = router.select_account("qwen_code", "qwen-coder-model")
            self.assertEqual(selected.account.name, "lisa")
            self.assertTrue(mock_choice.called)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                router.register_event(
                    provider="qwen_code",
                    account_name="ivan",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.QUOTA_EXHAUSTED,
                    model="qwen-coder-model",
                )

            def _assert_choice(candidates: list[str]) -> str:
                self.assertNotIn("ivan", candidates)
                return "kate"

            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.random.choice", side_effect=_assert_choice),
            ):
                event = router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=selected.mode,
                    pool=selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                )
                self.assertTrue(event.switched)
                next_selected = router.select_account("qwen_code", "qwen-coder-model")

            self.assertEqual(next_selected.account.name, "kate")

    def test_rounding_rotate_after_n_successes_switches_account(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 2,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": False,
                    "rotate_after_n_successes": 2,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                selected = router.select_account("qwen_code", "qwen-coder-model", group_id="g0")

            self.assertEqual(selected.account.name, "lisa")
            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                router.register_success("qwen_code", "lisa", group_id="g0")
                router.register_success("qwen_code", "lisa", group_id="g0")
                next_selected = router.select_account("qwen_code", "qwen-coder-model", group_id="g0")

            self.assertEqual(next_selected.account.name, "petr")

    def test_group_isolation_in_rounding(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr", "ivan", "kate"],
                "groups": {
                    "g1": {
                        "accounts": ["lisa", "petr"],
                        "models": ["qwen-coder-model-quota"],
                    },
                    "g2": {
                        "accounts": ["ivan", "kate"],
                        "models": ["qwen-coder-model-quota"],
                    },
                },
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
                "ivan": {"credentials_path": "secrets_test/accounts/qwen_ivan.json"},
                "kate": {"credentials_path": "secrets_test/accounts/qwen_kate.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": False,
                    "rotate_after_n_successes": 0,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.random.choice", return_value="ivan") as mock_choice,
            ):
                g1_selected = router.select_account("qwen_code", "qwen-coder-model", group_id="g1")
                g2_selected = router.select_account("qwen_code", "qwen-coder-model", group_id="g2")

                self.assertEqual(g1_selected.account.name, "lisa")
                self.assertEqual(g2_selected.account.name, "ivan")

                router.register_event(
                    provider="qwen_code",
                    account_name="lisa",
                    mode=g1_selected.mode,
                    pool=g1_selected.pool,
                    event=RotationEvent.RATE_LIMIT,
                    model="qwen-coder-model",
                    group_id="g1",
                )

                g1_next = router.select_account("qwen_code", "qwen-coder-model", group_id="g1")
                g2_still = router.select_account("qwen_code", "qwen-coder-model", group_id="g2")

            self.assertEqual(g1_next.account.name, "petr")
            self.assertEqual(g2_still.account.name, "ivan")

    def test_disjoint_group_validation(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "groups": {
                    "g1": {
                        "accounts": ["lisa"],
                        "models": ["qwen-g1-model-quota"],
                    },
                    "g2": {
                        "accounts": ["lisa"],
                        "models": ["qwen-g2-model-quota"],
                    },
                },
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": False,
                    "rotate_after_n_successes": 0,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                with self.assertRaises(AccountRouterError) as ctx:
                    router.select_account("qwen_code", "qwen-coder-model", group_id="g1")

            self.assertIn("appears in multiple groups", str(ctx.exception))

    def test_all_cooldown_error_includes_wait_seconds(self):
        with _secrets_test_dir() as tmp_dir:
            router = self._make_router(tmp_dir)

            qwen_path = tmp_dir / "qwen_accounts_config.json"
            qwen_rounding_cfg = {
                "mode": "rounding",
                "active_account": "lisa",
                "all_accounts": ["lisa", "petr"],
                "accounts": {
                "lisa": {"credentials_path": "secrets_test/accounts/qwen_lisa.json"},
                "petr": {"credentials_path": "secrets_test/accounts/qwen_petr.json"},
            },
                "rotation_policy": {
                    "rate_limit_threshold": 1,
                    "quota_exhausted_threshold": 2,
                    "rate_limit_cooldown_seconds": 5,
                    "random_order": False,
                    "rotate_after_n_successes": 0,
                },
                "model_quota_resets": {
                    "default": "00:00:00",
                    "qwen-coder-model": "00:08:00",
                },
            }
            _write_json(qwen_path, qwen_rounding_cfg)

            with self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path):
                router.select_account("qwen_code", "qwen-coder-model", group_id="g0")

            state = router._state[("qwen_code", "g0")]
            state.cooldown_until = {"lisa": 1005.0, "petr": 1010.0}

            with (
                self._patched_paths(tmp_dir, tmp_dir / "gemini_accounts_config.json", qwen_path),
                patch("llm_agent_platform.services.account_router.time.time", return_value=1001.0),
            ):
                with self.assertRaises(AccountRouterError) as ctx:
                    router.select_account("qwen_code", "qwen-coder-model", group_id="g0")

            self.assertIn("all accounts on cooldown", str(ctx.exception))
            self.assertIn("please wait 4", str(ctx.exception))
