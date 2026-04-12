from __future__ import annotations

import hashlib
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_agent_platform.config import OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH
from llm_agent_platform.services.account_router import quota_account_router

PROVIDER_ID = "openai-chatgpt"


class ApiKeyRegistryError(RuntimeError):
    """Raised when the API key registry cannot be read or written."""


class InvalidGroupError(ApiKeyRegistryError):
    """Raised when the requested provider group is invalid."""


class ApiKeyNotFoundError(ApiKeyRegistryError):
    """Raised when the requested key_id is absent from the registry."""


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_key_hash(raw_api_key: str) -> str:
    digest = hashlib.sha256(raw_api_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _build_key_id() -> str:
    return f"key_{secrets.token_hex(4)}"


def _build_raw_api_key() -> str:
    return f"sk-poc-{secrets.token_urlsafe(24)}"


def _mask_api_key(raw_api_key: str) -> str:
    return f"{raw_api_key[:7]}...{raw_api_key[-4:]}"


class OpenAIChatGPTApiKeyRegistryService:
    def __init__(self, registry_path: str | Path | None = None) -> None:
        self._registry_path = Path(
            registry_path or OPENAI_CHATGPT_API_KEYS_REGISTRY_PATH
        )
        self._lock = threading.Lock()

    def list_keys(self, group_id: str) -> dict[str, Any]:
        normalized_group_id = self._require_group(group_id)
        with self._lock:
            payload = self._load_registry_unlocked()
        return {
            "provider_id": PROVIDER_ID,
            "group_id": normalized_group_id,
            "keys": [
                self._to_list_item(entry)
                for entry in payload["keys"]
                if entry["group_id"] == normalized_group_id
            ],
        }

    def create_key(self, *, group_id: str, label: str) -> dict[str, Any]:
        normalized_group_id = self._require_group(group_id)
        normalized_label = label.strip()
        if not normalized_label:
            raise ApiKeyRegistryError("label is required")

        raw_api_key = _build_raw_api_key()
        entry = {
            "key_id": _build_key_id(),
            "label": normalized_label,
            "group_id": normalized_group_id,
            "status": "active",
            "key_hash": _compute_key_hash(raw_api_key),
            "masked_preview": _mask_api_key(raw_api_key),
            "created_at": _utc_now(),
        }

        with self._lock:
            payload = self._load_registry_unlocked()
            payload["keys"].append(entry)
            self._write_registry_unlocked(payload)

        return {
            "key_id": entry["key_id"],
            "provider_id": PROVIDER_ID,
            "group_id": normalized_group_id,
            "label": normalized_label,
            "status": "active",
            "masked_preview": entry["masked_preview"],
            "created_at": entry["created_at"],
            "raw_api_key": raw_api_key,
        }

    def revoke_key(self, key_id: str) -> dict[str, Any]:
        normalized_key_id = key_id.strip()
        if not normalized_key_id:
            raise ApiKeyRegistryError("key_id is required")

        with self._lock:
            payload = self._load_registry_unlocked()
            target = next(
                (
                    entry
                    for entry in payload["keys"]
                    if entry["key_id"] == normalized_key_id
                ),
                None,
            )
            if target is None:
                raise ApiKeyNotFoundError(f"Unknown key_id '{normalized_key_id}'")
            if target["status"] != "revoked":
                target["status"] = "revoked"
                target["revoked_at"] = _utc_now()
                self._write_registry_unlocked(payload)

        return {"ok": True, "key_id": normalized_key_id, "status": "revoked"}

    def lookup_active_key(
        self, raw_api_key: str, *, group_id: str | None = None
    ) -> dict[str, Any] | None:
        normalized_raw_api_key = raw_api_key.strip()
        if not normalized_raw_api_key:
            return None

        expected_group_id = group_id.strip() if group_id is not None else None
        key_hash = _compute_key_hash(normalized_raw_api_key)
        with self._lock:
            payload = self._load_registry_unlocked()

        for entry in payload["keys"]:
            if entry["status"] != "active":
                continue
            if expected_group_id is not None and entry["group_id"] != expected_group_id:
                continue
            if entry["key_hash"] != key_hash:
                continue
            return {
                "key_id": entry["key_id"],
                "provider_id": PROVIDER_ID,
                "group_id": entry["group_id"],
                "label": entry["label"],
                "status": entry["status"],
                "masked_preview": entry["masked_preview"],
                "created_at": entry["created_at"],
            }
        return None

    def _require_group(self, group_id: str) -> str:
        normalized_group_id = group_id.strip()
        if not normalized_group_id:
            raise InvalidGroupError("group_id is required")

        provider_cfg = quota_account_router.try_load_provider_config(PROVIDER_ID)
        if provider_cfg is None or not provider_cfg.groups:
            raise InvalidGroupError(
                "openai-chatgpt groups are unavailable in accounts config"
            )
        if normalized_group_id not in provider_cfg.groups:
            raise InvalidGroupError(
                f"Unknown group_id '{normalized_group_id}' for provider '{PROVIDER_ID}'"
            )
        return normalized_group_id

    def _load_registry_unlocked(self) -> dict[str, Any]:
        if not self._registry_path.exists():
            return self._empty_registry()
        try:
            with self._registry_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise ApiKeyRegistryError(
                f"Failed to read API key registry {self._registry_path}: {exc}"
            ) from exc
        return self._validate_registry_payload(payload)

    def _write_registry_unlocked(self, payload: dict[str, Any]) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._registry_path.with_suffix(self._registry_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(self._registry_path)
        except Exception as exc:
            raise ApiKeyRegistryError(
                f"Failed to write API key registry {self._registry_path}: {exc}"
            ) from exc

    def _validate_registry_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ApiKeyRegistryError(
                f"API key registry {self._registry_path} must contain a JSON object"
            )
        version = payload.get("version")
        provider_id = payload.get("provider_id")
        keys = payload.get("keys")
        if version != 1 or provider_id != PROVIDER_ID or not isinstance(keys, list):
            raise ApiKeyRegistryError(
                f"API key registry {self._registry_path} has invalid top-level shape"
            )

        normalized_keys: list[dict[str, Any]] = []
        for index, item in enumerate(keys, start=1):
            if not isinstance(item, dict):
                raise ApiKeyRegistryError(
                    f"API key entry #{index} in {self._registry_path} must be an object"
                )
            entry = dict(item)
            required_fields = {
                "key_id",
                "label",
                "group_id",
                "status",
                "key_hash",
                "masked_preview",
                "created_at",
            }
            if not required_fields.issubset(entry):
                raise ApiKeyRegistryError(
                    f"API key entry #{index} in {self._registry_path} is missing required fields"
                )
            if entry["status"] not in {"active", "revoked"}:
                raise ApiKeyRegistryError(
                    f"API key entry #{index} in {self._registry_path} has invalid status"
                )
            normalized_keys.append(entry)

        return {"version": 1, "provider_id": PROVIDER_ID, "keys": normalized_keys}

    def _empty_registry(self) -> dict[str, Any]:
        return {"version": 1, "provider_id": PROVIDER_ID, "keys": []}

    @staticmethod
    def _to_list_item(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "key_id": entry["key_id"],
            "label": entry["label"],
            "status": entry["status"],
            "masked_preview": entry["masked_preview"],
            "created_at": entry["created_at"],
        }
