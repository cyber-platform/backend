from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_agent_platform.config import STATE_DIR
from llm_agent_platform.services.account_router import (
    AccountRouterError,
    quota_account_router,
)
from llm_agent_platform.services.provider_registry import get_provider_registry
from llm_agent_platform.services.provider_usage_limits import (
    OpenAIChatGptUsageLimitsAdapter,
)
from llm_agent_platform.services.service_behavior_config import (
    MonitoringBehaviorConfig,
    load_service_behavior_config,
)

PROVIDER_ID = "openai-chatgpt"
ACTIVATE_SEMANTICS = "session_scoped_in_memory_preferred_account_override"
DEFAULT_REFRESH_INTERVAL_SECONDS = 10

_cache_lock = threading.Lock()


@dataclass(slots=True)
class _MonitoringRuntimeState:
    initialized: bool = False
    hydrated_accounts: set[str] = field(default_factory=set)
    usage_windows_by_account: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_usage_by_account: dict[str, dict[str, Any]] = field(default_factory=dict)


_runtime_state = _MonitoringRuntimeState()


class MonitoringReadModelError(RuntimeError):
    """Raised when monitoring read-model state cannot be built."""


class RefreshRunNotFoundError(MonitoringReadModelError):
    """Raised when a monitoring refresh run cannot be found."""


ACTIVE_REFRESH_STATUSES = {"queued", "running"}
TERMINAL_REFRESH_STATUSES = {"completed", "completed_with_errors", "error"}


def _monitoring_behavior() -> MonitoringBehaviorConfig:
    return load_service_behavior_config().monitoring


def _refresh_interval_seconds() -> int:
    return _monitoring_behavior().background_poll_interval_seconds


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _normalize_refresh_status(refresh: dict[str, Any]) -> str:
    raw_status = str(refresh.get("status") or "").strip().lower()
    status_map = {
        "ok": "fresh",
        "fresh": "fresh",
        "missing": "stale",
        "stale": "stale",
        "refreshing": "refreshing",
        "error": "error",
    }
    status = status_map.get(raw_status, "")
    if not status:
        status = "error" if refresh.get("last_error") else "fresh"
    if status == "fresh":
        next_refresh_at = _parse_datetime(refresh.get("next_refresh_at"))
        if next_refresh_at is not None and next_refresh_at <= _utc_now():
            return "stale"
    return status


def _normalize_refresh_payload(
    refresh: dict[str, Any], *, default_status: str
) -> dict[str, Any]:
    now = _utc_now()
    interval_seconds = _refresh_interval_seconds()
    raw_interval = refresh.get("refresh_interval_seconds")
    if isinstance(raw_interval, (int, float, str)):
        try:
            interval_seconds = max(1, int(raw_interval))
        except ValueError:
            interval_seconds = _refresh_interval_seconds()
    normalized = dict(refresh)
    normalized.setdefault("last_refreshed_at", _isoformat(now))
    normalized.setdefault(
        "next_refresh_at",
        _isoformat(now + timedelta(seconds=interval_seconds)),
    )
    normalized["refresh_interval_seconds"] = interval_seconds
    normalized.setdefault("last_error", None)
    if not normalized.get("status"):
        normalized["status"] = default_status
    normalized["status"] = _normalize_refresh_status(normalized)
    return normalized


@dataclass(slots=True)
class _RefreshAccountState:
    account_name: str
    status: str = "pending"
    last_error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "account_name": self.account_name,
            "status": self.status,
        }
        payload["last_error"] = self.last_error
        return payload


@dataclass(slots=True)
class _RefreshRun:
    refresh_id: str
    provider_id: str
    trigger: str
    accounts_by_group: dict[str, list[str]]
    accounts: OrderedDict[str, _RefreshAccountState]
    status: str = "queued"
    created_at: str = field(default_factory=lambda: _isoformat(_utc_now()))
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def total_accounts(self) -> int:
        return len(self.accounts)

    @property
    def processed_accounts(self) -> int:
        return sum(
            1
            for account in self.accounts.values()
            if account.status in {"success", "error", "skipped"}
        )

    @property
    def successful_accounts(self) -> int:
        return sum(
            1 for account in self.accounts.values() if account.status == "success"
        )

    @property
    def failed_accounts(self) -> int:
        return sum(1 for account in self.accounts.values() if account.status == "error")

    def to_start_payload(self) -> dict[str, Any]:
        return {
            "refresh_id": self.refresh_id,
            "provider_id": self.provider_id,
            "status": self.status
            if self.status in ACTIVE_REFRESH_STATUSES
            else "running",
            "processed_accounts": self.processed_accounts,
            "total_accounts": self.total_accounts,
            "successful_accounts": self.successful_accounts,
            "failed_accounts": self.failed_accounts,
        }

    def to_status_payload(self) -> dict[str, Any]:
        return {
            **self.to_start_payload(),
            "status": self.status,
            "accounts": [account.to_payload() for account in self.accounts.values()],
        }


class _MonitoringRefreshManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs_by_id: dict[str, _RefreshRun] = {}
        self._active_run_ids: dict[str, str] = {}
        self._latest_run_ids: dict[str, str] = {}
        self._poller_thread: threading.Thread | None = None
        self._poller_stop = threading.Event()

    def stop(self) -> None:
        self._poller_stop.set()
        thread = self._poller_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)

    def ensure_background_poller(self) -> None:
        with self._lock:
            if self._poller_thread is not None and self._poller_thread.is_alive():
                return
            self._poller_stop = threading.Event()
            self._poller_thread = threading.Thread(
                target=self._poll_forever,
                name=f"{PROVIDER_ID}-monitoring-poller",
                daemon=True,
            )
            self._poller_thread.start()

    def start_manual_refresh(self, provider_id: str) -> dict[str, Any]:
        run = self._start_run(provider_id, trigger="manual")
        return run.to_start_payload()

    def get_run_status(self, refresh_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire_runs_locked()
            run = self._runs_by_id.get(refresh_id)
            if run is None:
                raise RefreshRunNotFoundError(
                    f"Unknown monitoring refresh '{refresh_id}'"
                )
            return run.to_status_payload()

    def get_latest_run_summary(
        self, provider_id: str, group_id: str, account_names: list[str]
    ) -> dict[str, Any]:
        with self._lock:
            self._expire_runs_locked()
            run_id = self._latest_run_ids.get(provider_id)
            run = self._runs_by_id.get(run_id or "") if run_id else None
            if run is None:
                return {
                    "status": "idle",
                    "processed_accounts": 0,
                    "total_accounts": len(account_names),
                    "successful_accounts": 0,
                    "failed_accounts": 0,
                }
            return self._group_summary_for_run(run, group_id, account_names)

    def account_status_overrides(self, provider_id: str) -> dict[str, str]:
        with self._lock:
            run_id = self._active_run_ids.get(provider_id)
            run = self._runs_by_id.get(run_id or "") if run_id else None
            if run is None:
                return {}
            return {
                account_name: account.status
                for account_name, account in run.accounts.items()
                if account.status in {"pending", "refreshing"}
            }

    def _group_summary_for_run(
        self, run: _RefreshRun, group_id: str, account_names: list[str]
    ) -> dict[str, Any]:
        relevant_names = set(run.accounts_by_group.get(group_id) or account_names)
        processed = 0
        successful = 0
        failed = 0
        for account_name in account_names:
            account = run.accounts.get(account_name)
            if account_name not in relevant_names or account is None:
                continue
            if account.status in {"success", "error", "skipped"}:
                processed += 1
            if account.status == "success":
                successful += 1
            if account.status == "error":
                failed += 1
        payload = {
            "refresh_id": run.refresh_id,
            "status": run.status,
            "processed_accounts": processed,
            "total_accounts": len(account_names),
            "successful_accounts": successful,
            "failed_accounts": failed,
        }
        if run.status not in ACTIVE_REFRESH_STATUSES | TERMINAL_REFRESH_STATUSES:
            payload["status"] = "idle"
        return payload

    def _poll_forever(self) -> None:
        while not self._poller_stop.wait(_refresh_interval_seconds()):
            try:
                self._start_run(PROVIDER_ID, trigger="background")
            except Exception:
                continue

    def _start_run(self, provider_id: str, *, trigger: str) -> _RefreshRun:
        cfg = quota_account_router.try_load_provider_config(provider_id)
        if cfg is None:
            raise MonitoringReadModelError(
                f"{provider_id} provider config is not available"
            )
        accounts_by_group = self._accounts_by_group(cfg)
        ordered_accounts: OrderedDict[str, _RefreshAccountState] = OrderedDict()
        for group_accounts in accounts_by_group.values():
            for account_name in group_accounts:
                ordered_accounts.setdefault(
                    account_name,
                    _RefreshAccountState(account_name=account_name),
                )
        with self._lock:
            self._expire_runs_locked()
            active_id = self._active_run_ids.get(provider_id)
            active_run = self._runs_by_id.get(active_id or "") if active_id else None
            if active_run is not None and active_run.status in ACTIVE_REFRESH_STATUSES:
                return active_run
            run = _RefreshRun(
                refresh_id=uuid4().hex,
                provider_id=provider_id,
                trigger=trigger,
                accounts_by_group=accounts_by_group,
                accounts=ordered_accounts,
            )
            self._runs_by_id[run.refresh_id] = run
            self._active_run_ids[provider_id] = run.refresh_id
            self._latest_run_ids[provider_id] = run.refresh_id
        thread = threading.Thread(
            target=self._execute_run,
            args=(run.refresh_id,),
            name=f"{provider_id}-refresh-{run.refresh_id[:8]}",
            daemon=True,
        )
        thread.start()
        return run

    def _accounts_by_group(self, cfg: Any) -> dict[str, list[str]]:
        if cfg.groups:
            return {
                group_id: list(group.accounts) for group_id, group in cfg.groups.items()
            }
        return {"g0": list(cfg.all_accounts)}

    def _execute_run(self, refresh_id: str) -> None:
        with self._lock:
            run = self._runs_by_id.get(refresh_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = _isoformat(_utc_now())
        cfg = quota_account_router.try_load_provider_config(run.provider_id)
        if cfg is None:
            self._finish_run(refresh_id, "error")
            return
        behavior = _monitoring_behavior()
        deadline = time.monotonic() + behavior.refresh_run_timeout_seconds
        futures: list[tuple[str, Any]] = []
        with ThreadPoolExecutor(max_workers=behavior.refresh_concurrency) as executor:
            for account_name in run.accounts:
                futures.append(
                    (
                        account_name,
                        executor.submit(
                            self._refresh_account,
                            cfg.accounts[account_name].credentials_path,
                            account_name,
                        ),
                    )
                )
            for account_name, future in futures:
                with self._lock:
                    account = self._runs_by_id[refresh_id].accounts[account_name]
                    if account.status == "pending":
                        account.status = "refreshing"
                remaining = min(
                    behavior.per_account_refresh_timeout_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
                if remaining <= 0:
                    self._mark_account_result(
                        refresh_id,
                        account_name,
                        success=False,
                        error_message="refresh run timed out",
                    )
                    continue
                try:
                    future.result(timeout=remaining)
                    self._mark_account_result(refresh_id, account_name, success=True)
                except FutureTimeoutError:
                    self._mark_account_result(
                        refresh_id,
                        account_name,
                        success=False,
                        error_message="account refresh timed out",
                    )
                except Exception as exc:
                    self._mark_account_result(
                        refresh_id,
                        account_name,
                        success=False,
                        error_message=str(exc),
                    )
        with self._lock:
            current = self._runs_by_id.get(refresh_id)
            if current is None:
                return
            if current.successful_accounts == current.total_accounts:
                terminal_status = "completed"
            elif current.successful_accounts > 0:
                terminal_status = "completed_with_errors"
            else:
                terminal_status = "error"
        self._finish_run(refresh_id, terminal_status)

    def _refresh_account(self, credentials_path: str, account_name: str) -> None:
        snapshot = OpenAIChatGptUsageLimitsAdapter(credentials_path).fetch_snapshot()
        normalized = _store_runtime_usage_windows(account_name, snapshot)
        try:
            _persist_usage_windows(account_name, normalized)
        except Exception:
            pass

    def _mark_account_result(
        self,
        refresh_id: str,
        account_name: str,
        *,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        if not success:
            self._persist_refresh_error(account_name, error_message or "refresh failed")
        with self._lock:
            run = self._runs_by_id.get(refresh_id)
            if run is None:
                return
            account = run.accounts[account_name]
            account.status = "success" if success else "error"
            account.last_error = (
                None if success else (error_message or "refresh failed")
            )

    def _persist_refresh_error(self, account_name: str, error_message: str) -> None:
        payload = _get_runtime_usage_windows(account_name)
        refresh = dict(payload.get("refresh") or {})
        refresh["status"] = "error"
        refresh["last_error"] = error_message
        refresh["next_refresh_at"] = _isoformat(
            _utc_now() + timedelta(seconds=_refresh_interval_seconds())
        )
        payload["refresh"] = refresh
        normalized = _store_runtime_usage_windows(account_name, payload)
        try:
            _persist_usage_windows(account_name, normalized)
        except Exception:
            pass

    def _finish_run(self, refresh_id: str, status: str) -> None:
        with self._lock:
            run = self._runs_by_id.get(refresh_id)
            if run is None:
                return
            run.status = status
            run.finished_at = _isoformat(_utc_now())
            active_id = self._active_run_ids.get(run.provider_id)
            if active_id == refresh_id:
                self._active_run_ids.pop(run.provider_id, None)
            self._expire_runs_locked()

    def _expire_runs_locked(self) -> None:
        retention_seconds = _monitoring_behavior().refresh_run_retention_seconds
        cutoff = _utc_now() - timedelta(seconds=retention_seconds)
        for refresh_id, run in list(self._runs_by_id.items()):
            if run.status in ACTIVE_REFRESH_STATUSES:
                continue
            finished_at = _parse_datetime(run.finished_at)
            if finished_at is None or finished_at >= cutoff:
                continue
            self._runs_by_id.pop(refresh_id, None)
            latest_id = self._latest_run_ids.get(run.provider_id)
            if latest_id == refresh_id:
                self._latest_run_ids.pop(run.provider_id, None)


_refresh_manager_lock = threading.Lock()
_refresh_manager: _MonitoringRefreshManager | None = None


def _get_refresh_manager() -> _MonitoringRefreshManager:
    global _refresh_manager
    with _refresh_manager_lock:
        if _refresh_manager is None:
            _refresh_manager = _MonitoringRefreshManager()
        return _refresh_manager


def _reset_refresh_runtime() -> None:
    global _refresh_manager
    with _refresh_manager_lock:
        if _refresh_manager is not None:
            _refresh_manager.stop()
        _refresh_manager = None


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _state_root() -> Path:
    return Path(STATE_DIR) / PROVIDER_ID


def _account_dir(account_name: str) -> Path:
    return _state_root() / "accounts" / account_name


def _usage_windows_path(account_name: str) -> Path:
    return _account_dir(account_name) / "usage_windows.json"


def _request_usage_path(account_name: str) -> Path:
    return _account_dir(account_name) / "request_usage.json"


def _legacy_limits_path(account_name: str) -> Path:
    return _state_root() / "usage" / "accounts" / account_name / "limits.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def _window_minutes(window_payload: dict[str, Any]) -> int:
    raw_minutes = window_payload.get("window_minutes")
    if isinstance(raw_minutes, int) and raw_minutes >= 1:
        return raw_minutes
    raw_window = window_payload.get("window")
    if isinstance(raw_window, str) and raw_window.endswith("m"):
        try:
            parsed = int(raw_window[:-1])
        except ValueError:
            parsed = 0
        if parsed >= 1:
            return parsed
    metadata = window_payload.get("metadata")
    if isinstance(metadata, dict):
        raw_seconds = metadata.get("limit_window_seconds")
        if isinstance(raw_seconds, (int, float, str)):
            try:
                seconds = int(raw_seconds)
            except ValueError:
                seconds = 0
        else:
            seconds = 0
        if seconds >= 60:
            return max(1, seconds // 60)
    return 1


def _normalize_window(window_payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "used_percent": float(window_payload.get("used_percent", 0) or 0),
        "window_minutes": _window_minutes(window_payload),
    }
    reset_at = window_payload.get("reset_at")
    if isinstance(reset_at, str) and reset_at:
        normalized["reset_at"] = reset_at
    source = window_payload.get("source")
    if isinstance(source, str) and source:
        normalized["source"] = source
    metadata = window_payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        normalized["metadata"] = dict(metadata)
    return normalized


def _default_usage_windows_payload() -> dict[str, Any]:
    now = _utc_now()
    return {
        "version": 1,
        "provider_id": PROVIDER_ID,
        "short_window": {"used_percent": 0.0, "window_minutes": 1},
        "long_window": {"used_percent": 0.0, "window_minutes": 1},
        "refresh": {
            "last_refreshed_at": _isoformat(now),
            "next_refresh_at": _isoformat(
                now + timedelta(seconds=_refresh_interval_seconds())
            ),
            "refresh_interval_seconds": _refresh_interval_seconds(),
            "status": "stale",
            "last_error": None,
        },
    }


def _default_request_usage_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "provider_id": PROVIDER_ID,
        "request_counters": {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
        },
    }


def _normalize_usage_windows_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "short_window" in payload and "long_window" in payload and "refresh" in payload:
        normalized = dict(payload)
        normalized["version"] = 1
        normalized["provider_id"] = PROVIDER_ID
        normalized["short_window"] = _normalize_window(
            dict(payload.get("short_window") or {})
        )
        normalized["long_window"] = _normalize_window(
            dict(payload.get("long_window") or {})
        )
        refresh = _normalize_refresh_payload(
            dict(payload.get("refresh") or {}),
            default_status="fresh",
        )
        normalized["refresh"] = refresh
        return normalized

    limits = payload.get("limits")
    if not isinstance(limits, dict):
        return _default_usage_windows_payload()
    primary = limits.get("primary")
    secondary = limits.get("secondary")
    if not isinstance(primary, dict) and not isinstance(secondary, dict):
        return _default_usage_windows_payload()

    short_window = {"used_percent": 0.0, "window_minutes": 1}
    long_window = {"used_percent": 0.0, "window_minutes": 1}
    normalized_primary = (
        _normalize_window(primary) if isinstance(primary, dict) else None
    )
    normalized_secondary = (
        _normalize_window(secondary) if isinstance(secondary, dict) else None
    )

    if normalized_primary is not None and normalized_secondary is not None:
        short_window = normalized_primary
        long_window = normalized_secondary
    elif normalized_primary is not None:
        if int(normalized_primary.get("window_minutes", 0) or 0) >= 1440:
            long_window = normalized_primary
        else:
            short_window = normalized_primary
    elif normalized_secondary is not None:
        long_window = normalized_secondary

    as_of = (
        payload.get("as_of")
        if isinstance(payload.get("as_of"), str)
        else _isoformat(_utc_now())
    )
    return {
        "version": 1,
        "provider_id": PROVIDER_ID,
        "account_id": payload.get("account_id"),
        "short_window": short_window,
        "long_window": long_window,
        "refresh": {
            "last_refreshed_at": as_of,
            "next_refresh_at": _isoformat(
                _utc_now() + timedelta(seconds=_refresh_interval_seconds())
            ),
            "refresh_interval_seconds": _refresh_interval_seconds(),
            "status": "fresh",
            "last_error": None,
        },
        "metadata": dict(payload.get("metadata") or {}),
    }


def _normalize_request_usage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "request_counters" in payload:
        normalized = dict(payload)
        normalized["version"] = 1
        normalized["provider_id"] = PROVIDER_ID
        return normalized

    metadata = payload.get("metadata")
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return _default_request_usage_payload()

    requested_at = (
        payload.get("as_of")
        if isinstance(payload.get("as_of"), str)
        else _isoformat(_utc_now())
    )
    tokens = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    return {
        "version": 1,
        "provider_id": PROVIDER_ID,
        "account_id": payload.get("account_id"),
        "request_counters": {
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "last_request_at": requested_at,
        },
        "aggregate_tokens": tokens,
        "last_request_summary": {
            "status": "success",
            "requested_at": requested_at,
            "tokens": tokens,
            "metadata": dict(metadata or {}),
        },
        "metadata": dict(metadata or {}),
    }


def _configured_account_names() -> list[str]:
    cfg = quota_account_router.try_load_provider_config(PROVIDER_ID)
    if cfg is None:
        return []
    return list(cfg.accounts.keys())


def _load_persisted_usage_windows(account_name: str) -> dict[str, Any]:
    payload = _read_json(_usage_windows_path(account_name)) or {}
    return _normalize_usage_windows_payload(payload)


def _load_persisted_request_usage(account_name: str) -> dict[str, Any]:
    payload = _read_json(_request_usage_path(account_name)) or {}
    return _normalize_request_usage_payload(payload)


def initialize_monitoring_runtime() -> None:
    account_names = _configured_account_names()
    if not account_names:
        return

    with _cache_lock:
        missing_accounts = [
            account_name
            for account_name in account_names
            if account_name not in _runtime_state.hydrated_accounts
        ]
        if _runtime_state.initialized and not missing_accounts:
            return

    hydrated_usage = {
        account_name: _load_persisted_usage_windows(account_name)
        for account_name in missing_accounts
    }
    hydrated_request_usage = {
        account_name: _load_persisted_request_usage(account_name)
        for account_name in missing_accounts
    }

    with _cache_lock:
        for account_name, payload in hydrated_usage.items():
            _runtime_state.usage_windows_by_account[account_name] = deepcopy(payload)
            _runtime_state.hydrated_accounts.add(account_name)
        for account_name, payload in hydrated_request_usage.items():
            _runtime_state.request_usage_by_account[account_name] = deepcopy(payload)
        _runtime_state.initialized = True


def _get_runtime_usage_windows(account_name: str) -> dict[str, Any]:
    with _cache_lock:
        payload = _runtime_state.usage_windows_by_account.get(account_name)
        if payload is None:
            raise MonitoringReadModelError(
                f"Monitoring runtime state for account '{account_name}' is not hydrated"
            )
        return deepcopy(payload)


def _get_runtime_request_usage(account_name: str) -> dict[str, Any]:
    with _cache_lock:
        payload = _runtime_state.request_usage_by_account.get(account_name)
        if payload is None:
            raise MonitoringReadModelError(
                f"Request usage runtime state for account '{account_name}' is not hydrated"
            )
        return deepcopy(payload)


def _store_runtime_usage_windows(
    account_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    normalized = _normalize_usage_windows_payload(payload)
    with _cache_lock:
        _runtime_state.usage_windows_by_account[account_name] = deepcopy(normalized)
        _runtime_state.hydrated_accounts.add(account_name)
        _runtime_state.initialized = True
    return normalized


def _store_runtime_request_usage(
    account_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    normalized = _normalize_request_usage_payload(payload)
    with _cache_lock:
        _runtime_state.request_usage_by_account[account_name] = deepcopy(normalized)
        _runtime_state.hydrated_accounts.add(account_name)
        _runtime_state.initialized = True
    return normalized


def _persist_usage_windows(account_name: str, payload: dict[str, Any]) -> None:
    _write_json(_usage_windows_path(account_name), payload)


def _persist_request_usage(account_name: str, payload: dict[str, Any]) -> None:
    _write_json(_request_usage_path(account_name), payload)


def get_usage_windows(account_name: str) -> dict[str, Any]:
    initialize_monitoring_runtime()
    return _get_runtime_usage_windows(account_name)


def save_usage_windows(account_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _store_runtime_usage_windows(account_name, payload)
    _persist_usage_windows(account_name, normalized)
    return normalized


def get_request_usage(account_name: str) -> dict[str, Any]:
    initialize_monitoring_runtime()
    return _get_runtime_request_usage(account_name)


def save_request_usage(account_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _store_runtime_request_usage(account_name, payload)
    _persist_request_usage(account_name, normalized)
    return normalized


def reset_monitoring_caches() -> None:
    with _cache_lock:
        _runtime_state.initialized = False
        _runtime_state.hydrated_accounts.clear()
        _runtime_state.usage_windows_by_account.clear()
        _runtime_state.request_usage_by_account.clear()
    _reset_refresh_runtime()


def _build_routing_payload(runtime_status: dict[str, Any]) -> dict[str, Any]:
    routing: dict[str, Any] = {"state": str(runtime_status.get("state", "ready"))}
    quota_blocked_until = runtime_status.get("quota_blocked_until")
    if isinstance(quota_blocked_until, str) and quota_blocked_until:
        routing["quota_blocked_until"] = quota_blocked_until
    block_reason = runtime_status.get("block_reason")
    if isinstance(block_reason, str) and block_reason:
        routing["block_reason"] = block_reason
    metadata = runtime_status.get("metadata")
    if isinstance(metadata, dict) and metadata:
        routing["metadata"] = dict(metadata)
    return routing


def _account_status(runtime_status: dict[str, Any]) -> str:
    state = str(runtime_status.get("state", "ready"))
    if state == "quota_blocked":
        return "quota_blocked"
    if state == "cooldown":
        return "cooldown"
    return "ready"


class OpenAIChatGPTAdminMonitoringService:
    def list_providers(self) -> dict[str, Any]:
        initialize_monitoring_runtime()
        _get_refresh_manager().ensure_background_poller()
        cfg = quota_account_router.try_load_provider_config(PROVIDER_ID)
        if cfg is None:
            return {"providers": []}
        provider = get_provider_registry().get_provider(PROVIDER_ID)
        return {
            "providers": [
                {
                    "provider_id": PROVIDER_ID,
                    "label": provider.display_name,
                    "account_count": len(cfg.accounts),
                    "has_monitoring": True,
                    "capabilities": {"supports_provider_page": True},
                }
            ]
        }

    def get_provider_page(self) -> dict[str, Any]:
        initialize_monitoring_runtime()
        refresh_manager = _get_refresh_manager()
        refresh_manager.ensure_background_poller()
        cfg = quota_account_router.try_load_provider_config(PROVIDER_ID)
        if cfg is None:
            raise MonitoringReadModelError(
                "openai-chatgpt provider config is not available"
            )

        provider = get_provider_registry().get_provider(PROVIDER_ID)
        groups_payload: list[dict[str, Any]] = []
        refresh_interval_seconds = _refresh_interval_seconds()
        refresh_overrides = refresh_manager.account_status_overrides(PROVIDER_ID)

        group_ids = list(cfg.groups.keys()) if cfg.groups else ["g0"]
        for group_id in group_ids:
            described = quota_account_router.describe_group(PROVIDER_ID, group_id)
            accounts_payload: list[dict[str, Any]] = []
            preferred_account_name = str(described.get("preferred_account") or "")
            described_accounts = described.get("accounts")
            if not isinstance(described_accounts, list):
                described_accounts = []

            for account_status in described_accounts:
                account_name = str(account_status.get("account_name") or "")
                usage_windows = get_usage_windows(account_name)
                request_usage = get_request_usage(account_name)
                refresh = dict(usage_windows.get("refresh") or {})
                account_refresh_state = refresh_overrides.get(account_name)
                if account_refresh_state in {"pending", "refreshing"}:
                    refresh["status"] = "refreshing"
                    refresh["last_error"] = None
                raw_refresh_interval = refresh.get(
                    "refresh_interval_seconds", refresh_interval_seconds
                )
                if isinstance(raw_refresh_interval, (int, float, str)):
                    try:
                        refresh_interval_seconds = int(raw_refresh_interval)
                    except ValueError:
                        pass
                accounts_payload.append(
                    {
                        "account_name": account_name,
                        "is_preferred_for_session": preferred_account_name
                        == account_name,
                        "status": _account_status(account_status),
                        "short_window": dict(usage_windows.get("short_window") or {}),
                        "long_window": dict(usage_windows.get("long_window") or {}),
                        "refresh": refresh,
                        "routing": _build_routing_payload(account_status),
                        "actions": {
                            "can_activate": True,
                            "activate_endpoint": f"/admin/monitoring/{PROVIDER_ID}/groups/{group_id}/accounts/{account_name}/activate",
                            "activate_semantics": ACTIVATE_SEMANTICS,
                        },
                        "drawer": {
                            "raw_monitoring_payload": usage_windows,
                            "raw_request_usage_payload": request_usage,
                            "raw_account_state_payload": dict(
                                account_status.get("raw_account_state") or {}
                            ),
                        },
                    }
                )

            groups_payload.append(
                {
                    "group_id": group_id,
                    "display_name": group_id,
                    "runtime_group_key": f"{PROVIDER_ID}:{group_id}",
                    "refresh": refresh_manager.get_latest_run_summary(
                        PROVIDER_ID,
                        group_id,
                        [
                            str(account.get("account_name") or "")
                            for account in described_accounts
                            if isinstance(account, dict)
                        ],
                    ),
                    "accounts": accounts_payload,
                }
            )

        return {
            "provider_id": PROVIDER_ID,
            "display_name": provider.display_name,
            "capabilities": {
                "supports_activate": True,
                "supports_drawer": True,
                "supports_usage_windows": True,
                "supports_manual_refresh": True,
            },
            "security_boundary": {
                "auth_mode": "none",
                "scope": "local_single_user_poc",
                "shared_deployment_allowed": False,
            },
            "refresh_interval_seconds": refresh_interval_seconds,
            "groups": groups_payload,
        }

    def start_refresh(self) -> dict[str, Any]:
        initialize_monitoring_runtime()
        refresh_manager = _get_refresh_manager()
        refresh_manager.ensure_background_poller()
        return refresh_manager.start_manual_refresh(PROVIDER_ID)

    def get_refresh_status(self, refresh_id: str) -> dict[str, Any]:
        initialize_monitoring_runtime()
        refresh_manager = _get_refresh_manager()
        refresh_manager.ensure_background_poller()
        return refresh_manager.get_run_status(refresh_id)

    def activate_account(self, group_id: str, account_name: str) -> dict[str, Any]:
        quota_account_router.set_preferred_account(PROVIDER_ID, group_id, account_name)
        return {
            "ok": True,
            "provider_id": PROVIDER_ID,
            "group_id": group_id,
            "account_name": account_name,
            "activate_semantics": ACTIVATE_SEMANTICS,
        }
