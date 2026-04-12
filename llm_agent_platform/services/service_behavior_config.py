from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_agent_platform.config import SERVICE_BEHAVIOR_CONFIG_PATH


@dataclass(frozen=True, slots=True)
class MonitoringBehaviorConfig:
    background_poll_interval_seconds: int = 10
    refresh_concurrency: int = 5
    per_account_refresh_timeout_seconds: int = 10
    refresh_run_timeout_seconds: int = 60
    refresh_run_retention_seconds: int = 300


@dataclass(frozen=True, slots=True)
class FrontendBehaviorConfig:
    page_poll_interval_ms: int = 10000
    manual_refresh_status_poll_interval_ms: int = 1000


@dataclass(frozen=True, slots=True)
class ServiceBehaviorConfig:
    monitoring: MonitoringBehaviorConfig = MonitoringBehaviorConfig()
    frontend: FrontendBehaviorConfig = FrontendBehaviorConfig()


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_section_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if indent == 0:
            if raw_value.strip():
                root[key] = _parse_scalar(raw_value)
                current_section = None
                current_section_indent = 0
            else:
                section: dict[str, Any] = {}
                root[key] = section
                current_section = section
                current_section_indent = indent
            continue
        if current_section is None or indent <= current_section_indent:
            continue
        current_section[key] = _parse_scalar(raw_value)
    return root


def _read_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    return _parse_simple_yaml(text)


def load_service_behavior_config(
    path: str | Path | None = None,
) -> ServiceBehaviorConfig:
    config_path = Path(path) if path is not None else Path(SERVICE_BEHAVIOR_CONFIG_PATH)
    payload = _read_raw_config(config_path)
    monitoring_payload = payload.get("monitoring")
    frontend_payload = payload.get("frontend")
    if not isinstance(monitoring_payload, dict):
        monitoring_payload = {}
    if not isinstance(frontend_payload, dict):
        frontend_payload = {}
    monitoring = MonitoringBehaviorConfig(
        background_poll_interval_seconds=_coerce_positive_int(
            monitoring_payload.get("background_poll_interval_seconds"), 10
        ),
        refresh_concurrency=_coerce_positive_int(
            monitoring_payload.get("refresh_concurrency"), 5
        ),
        per_account_refresh_timeout_seconds=_coerce_positive_int(
            monitoring_payload.get("per_account_refresh_timeout_seconds"), 10
        ),
        refresh_run_timeout_seconds=_coerce_positive_int(
            monitoring_payload.get("refresh_run_timeout_seconds"), 60
        ),
        refresh_run_retention_seconds=_coerce_positive_int(
            monitoring_payload.get("refresh_run_retention_seconds"), 300
        ),
    )
    frontend = FrontendBehaviorConfig(
        page_poll_interval_ms=_coerce_positive_int(
            frontend_payload.get("page_poll_interval_ms"), 10000
        ),
        manual_refresh_status_poll_interval_ms=_coerce_positive_int(
            frontend_payload.get("manual_refresh_status_poll_interval_ms"), 1000
        ),
    )
    return ServiceBehaviorConfig(monitoring=monitoring, frontend=frontend)
