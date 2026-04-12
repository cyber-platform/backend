from __future__ import annotations

import atexit
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from llm_agent_platform.config import (
    STATE_DIR,
    STATE_FLUSH_INTERVAL_SECONDS,
    STATE_WRITER_MAX_PENDING_FILES,
)
from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.services.credentials_paths import normalize_provider_storage_id

logger = get_logger(__name__)

ACCOUNT_STATE_FILENAME: Final[str] = "account_state.json"
GROUP_QUOTA_STATE_FILENAME: Final[str] = "quota_state.json"
PROVIDER_QUOTA_SENTINEL: Final[str] = "__provider__"


@dataclass(slots=True)
class AccountState:
    last_used_at: datetime | None = None
    last_cooldown_at: datetime | None = None
    quota_exhausted_at: dict[str, datetime] = field(default_factory=dict)


@dataclass(slots=True)
class AccountStatePaths:
    provider_id: str
    account_name: str
    root_dir: Path

    @property
    def account_dir(self) -> Path:
        return Path(self.root_dir) / normalize_provider_storage_id(self.provider_id) / "accounts" / self.account_name

    @property
    def account_state_path(self) -> Path:
        return self.account_dir / ACCOUNT_STATE_FILENAME

    def group_quota_state_path(self, group_id: str) -> Path:
        return (
            Path(self.root_dir)
            / normalize_provider_storage_id(self.provider_id)
            / "groups"
            / group_id
            / GROUP_QUOTA_STATE_FILENAME
        )


class AsyncStateWriter:
    def __init__(self, flush_interval_seconds: float, max_pending_files: int) -> None:
        self._flush_interval_seconds = flush_interval_seconds
        self._max_pending_files = max_pending_files
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self._thread: threading.Thread | None = None

    def enqueue_write(self, file_path: Path, payload: dict[str, Any]) -> None:
        normalized = Path(file_path)
        with self._lock:
            self._ensure_thread_unlocked()
            if normalized in self._pending:
                self._pending.pop(normalized, None)
            elif len(self._pending) >= self._max_pending_files:
                dropped_path, _ = self._pending.popitem(last=False)
                logger.warning(
                    "[state] Writer pending overflow, drop-oldest applied for %s",
                    dropped_path,
                )
            self._pending[normalized] = payload

    def flush_once(self) -> None:
        with self._lock:
            if not self._pending:
                return
            to_flush = self._pending
            self._pending = OrderedDict()

        failed: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        for file_path, payload in to_flush.items():
            if not _write_json(file_path, payload):
                failed[file_path] = payload

        if not failed:
            return

        with self._lock:
            for file_path, payload in failed.items():
                if file_path in self._pending:
                    continue
                self._pending[file_path] = payload

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._flush_interval_seconds * 2, 1.0))
        self.flush_once()

    def _ensure_thread_unlocked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="state-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._flush_interval_seconds):
            self.flush_once()


state_writer = AsyncStateWriter(
    flush_interval_seconds=STATE_FLUSH_INTERVAL_SECONDS,
    max_pending_files=STATE_WRITER_MAX_PENDING_FILES,
)
atexit.register(state_writer.close)


def default_state_root() -> Path:
    return Path(STATE_DIR)


def load_account_state(paths: AccountStatePaths) -> AccountState:
    payload = _read_json(paths.account_state_path)
    if not payload:
        return AccountState()

    state = AccountState()

    raw_last_used = payload.get("last_used_at")
    if isinstance(raw_last_used, str):
        state.last_used_at = _parse_datetime(raw_last_used)

    cooldown_payload = payload.get("cooldown")
    if isinstance(cooldown_payload, dict):
        raw_last_cooldown = cooldown_payload.get("last_cooldown_at")
        if isinstance(raw_last_cooldown, str):
            state.last_cooldown_at = _parse_datetime(raw_last_cooldown)

    quota_payload = payload.get("quota_exhausted")
    quota_keys = quota_payload.get("keys") if isinstance(quota_payload, dict) else None
    if isinstance(quota_keys, dict):
        for key, value in quota_keys.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            parsed = _parse_datetime(value)
            if parsed is not None:
                state.quota_exhausted_at[key] = parsed

    return state


def save_account_state(
    paths: AccountStatePaths,
    state: AccountState,
    *,
    writer: AsyncStateWriter | None = state_writer,
) -> None:
    payload: dict[str, Any] = {"version": 1}
    if state.last_used_at is not None:
        payload["last_used_at"] = _format_datetime(state.last_used_at)
    if state.last_cooldown_at is not None:
        payload["cooldown"] = {
            "last_cooldown_at": _format_datetime(state.last_cooldown_at),
        }
    if state.quota_exhausted_at:
        payload["quota_exhausted"] = {
            "keys": {
                key: _format_datetime(value)
                for key, value in sorted(state.quota_exhausted_at.items())
            }
        }
    _persist_json(paths.account_state_path, payload, writer=writer)


def load_last_used_at(paths: AccountStatePaths) -> datetime | None:
    return load_account_state(paths).last_used_at


def save_last_used_at(
    paths: AccountStatePaths,
    when: datetime,
    *,
    writer: AsyncStateWriter | None = state_writer,
) -> None:
    state = load_account_state(paths)
    state.last_used_at = when
    save_account_state(paths, state, writer=writer)


def load_last_cooldown_at(paths: AccountStatePaths) -> datetime | None:
    return load_account_state(paths).last_cooldown_at


def save_last_cooldown_at(
    paths: AccountStatePaths,
    when: datetime,
    *,
    writer: AsyncStateWriter | None = state_writer,
) -> None:
    state = load_account_state(paths)
    state.last_cooldown_at = when
    save_account_state(paths, state, writer=writer)


def load_quota_exhausted_at(paths: AccountStatePaths, model: str) -> datetime | None:
    return load_account_state(paths).quota_exhausted_at.get(model)


def save_quota_exhausted_at(
    paths: AccountStatePaths,
    model: str,
    when: datetime,
    *,
    writer: AsyncStateWriter | None = state_writer,
) -> None:
    state = load_account_state(paths)
    state.quota_exhausted_at[model] = when
    save_account_state(paths, state, writer=writer)


def write_group_quota_state(
    provider_id: str,
    group_id: str,
    payload: dict[str, Any],
    *,
    writer: AsyncStateWriter | None = state_writer,
) -> None:
    file_path = (
        default_state_root()
        / normalize_provider_storage_id(provider_id)
        / "groups"
        / group_id
        / GROUP_QUOTA_STATE_FILENAME
    )
    _persist_json(file_path, payload, writer=writer)


def _persist_json(
    file_path: Path,
    payload: dict[str, Any],
    *,
    writer: AsyncStateWriter | None,
) -> None:
    if writer is None:
        _write_json(file_path, payload)
        return
    writer.enqueue_write(file_path, payload)


def _write_json(file_path: Path, payload: dict[str, Any]) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        tmp_path.replace(file_path)
        return True
    except Exception as exc:
        logger.warning(f"[state] Failed to write state file {file_path}: {exc}")
        return False


def _read_json(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(f"[state] Failed to read state file {file_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime | None:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
