from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Any

from llm_agent_platform.auth.openai_chatgpt_oauth import OpenAIChatGPTOAuthManager
from llm_agent_platform.config import OPENAI_CHATGPT_USAGE_URL
from llm_agent_platform.services.credentials_paths import resolve_credentials_path_ref
from llm_agent_platform.services.http_pool import get_http_client
from llm_agent_platform.services.runtime_state_paths import resolve_runtime_state_paths


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch_seconds_to_iso(value: Any) -> str | None:
    try:
        seconds = int(value)
    except Exception:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class ProviderUsageLimitsPort(Protocol):
    def fetch_snapshot(self) -> dict[str, Any]:
        """Fetch and normalize provider usage limits for monitoring."""


@dataclass(slots=True)
class OpenAIChatGptUsageLimitsAdapter:
    credentials_path: str | Path

    def fetch_snapshot(self) -> dict[str, Any]:
        manager = OpenAIChatGPTOAuthManager(self.credentials_path)
        state = manager.get_valid_state()
        token = str(state.get("access_token") or "").strip()
        token_type = str(state.get("token_type") or "Bearer") or "Bearer"
        response = get_http_client().get(
            OPENAI_CHATGPT_USAGE_URL,
            headers={
                "Authorization": f"{token_type} {token}",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OpenAI ChatGPT usage endpoint must return a JSON object")

        rate_limit = payload.get("rate_limit") or {}
        limits: dict[str, Any] = {}
        for source_name, target_name in (("primary_window", "primary"), ("secondary_window", "secondary")):
            source = rate_limit.get(source_name)
            if not isinstance(source, dict):
                continue
            entry = {
                "used_percent": float(source.get("used_percent", 0)),
                "source": f"wham.{source_name}",
                "metadata": {},
            }
            reset_at = _epoch_seconds_to_iso(source.get("reset_at"))
            if reset_at:
                entry["reset_at"] = reset_at
            window_seconds = source.get("limit_window_seconds")
            if window_seconds is not None:
                entry["metadata"]["limit_window_seconds"] = int(window_seconds)
                entry["window"] = f"{int(window_seconds) // 60}m"
            limits[target_name] = entry

        if not limits:
            raise ValueError("Usage payload does not contain primary or secondary windows")

        snapshot = {
            "version": 1,
            "provider_id": "openai-chatgpt",
            "account_id": state.get("account_id"),
            "as_of": _iso_now(),
            "limits": limits,
            "metadata": {
                "plan_type": payload.get("plan_type"),
                "provider_usage_source": "wham",
            },
        }
        self.write_snapshot(snapshot)
        return snapshot

    def write_snapshot(self, payload: dict[str, Any]) -> Path:
        creds_ref = resolve_credentials_path_ref("openai-chatgpt", self.credentials_path)
        usage_path = resolve_runtime_state_paths(
            creds_ref.provider_id,
            account_name=creds_ref.account_name,
        ).usage_snapshot_path
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return usage_path
