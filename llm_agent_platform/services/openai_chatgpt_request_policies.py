from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_agent_platform.config import (
    OPENAI_CHATGPT_REQUEST_POLICY_CAPABILITY_VALIDATION_ENABLED,
    OPENAI_CHATGPT_REQUEST_POLICY_REGISTRY_PATH,
)
from llm_agent_platform.services.openai_chatgpt_api_keys import (
    InvalidGroupError,
    OpenAIChatGPTApiKeyRegistryService,
)
from llm_agent_platform.services.openai_chatgpt_model_capabilities import (
    OpenAIChatGPTModelCapabilitiesService,
)

PROVIDER_ID = "openai-chatgpt"
ALLOWED_OVERRIDE_MODES = {"force", "default_if_absent"}


class RequestPolicyRegistryError(RuntimeError):
    """Raised when the request policy registry cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class RequestParameterOverride:
    mode: str
    value: str

    def to_payload(self) -> dict[str, str]:
        return {"mode": self.mode, "value": self.value}


@dataclass(frozen=True, slots=True)
class OpenAIChatGPTRequestPolicyRecord:
    provider_id: str
    key_id: str
    group_id: str
    model_overrides: dict[str, dict[str, RequestParameterOverride]]
    created_at: str | None = None
    updated_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_id": self.provider_id,
            "key_id": self.key_id,
            "group_id": self.group_id,
            "model_overrides": {
                model_id: {
                    parameter_name: override.to_payload()
                    for parameter_name, override in parameters.items()
                }
                for model_id, parameters in self.model_overrides.items()
            },
        }
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


class OpenAIChatGPTRequestPolicyRegistryService:
    def __init__(
        self,
        registry_path: str | Path | None = None,
        *,
        api_key_registry: OpenAIChatGPTApiKeyRegistryService | None = None,
        model_capabilities_service: OpenAIChatGPTModelCapabilitiesService | None = None,
        capability_validation_enabled: bool | None = None,
    ) -> None:
        self._registry_path = Path(
            registry_path or OPENAI_CHATGPT_REQUEST_POLICY_REGISTRY_PATH
        )
        self._api_key_registry = (
            api_key_registry or OpenAIChatGPTApiKeyRegistryService()
        )
        self._model_capabilities_service = (
            model_capabilities_service or OpenAIChatGPTModelCapabilitiesService()
        )
        self._capability_validation_enabled = (
            OPENAI_CHATGPT_REQUEST_POLICY_CAPABILITY_VALIDATION_ENABLED
            if capability_validation_enabled is None
            else capability_validation_enabled
        )
        self._lock = threading.Lock()

    def get_policy(self, key_id: str) -> OpenAIChatGPTRequestPolicyRecord:
        key_record = self._get_key_record(key_id)
        normalized_key_id = key_record["key_id"]
        with self._lock:
            policies = self._load_registry_unlocked()
        record = policies.get(normalized_key_id)
        if record is not None:
            return record
        return OpenAIChatGPTRequestPolicyRecord(
            provider_id=PROVIDER_ID,
            key_id=normalized_key_id,
            group_id=key_record["group_id"],
            model_overrides={},
        )

    def upsert_policy(
        self,
        *,
        key_id: str,
        group_id: str,
        model_overrides: Any,
    ) -> OpenAIChatGPTRequestPolicyRecord:
        key_record = self._get_key_record(key_id)
        normalized_group_id = self._require_non_empty_string(
            group_id, field_name="group_id"
        )
        if key_record["group_id"] != normalized_group_id:
            raise InvalidGroupError(
                f"group_id '{normalized_group_id}' does not match key '{key_record['key_id']}' group '{key_record['group_id']}'"
            )

        normalized_overrides = self._normalize_model_overrides(
            model_overrides,
            field_name="model_overrides",
            require_non_empty=True,
        )
        now = _utc_now()

        with self._lock:
            policies = self._load_registry_unlocked()
            existing = policies.get(key_record["key_id"])
            record = OpenAIChatGPTRequestPolicyRecord(
                provider_id=PROVIDER_ID,
                key_id=key_record["key_id"],
                group_id=normalized_group_id,
                model_overrides=normalized_overrides,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
            )
            policies[key_record["key_id"]] = record
            self._write_registry_unlocked(policies)
        return record

    def delete_policy(self, key_id: str) -> OpenAIChatGPTRequestPolicyRecord:
        key_record = self._get_key_record(key_id)
        normalized_key_id = key_record["key_id"]

        with self._lock:
            policies = self._load_registry_unlocked()
            if normalized_key_id in policies:
                del policies[normalized_key_id]
                self._write_registry_unlocked(policies)

        return OpenAIChatGPTRequestPolicyRecord(
            provider_id=PROVIDER_ID,
            key_id=normalized_key_id,
            group_id=key_record["group_id"],
            model_overrides={},
        )

    def _get_key_record(self, key_id: str) -> dict[str, Any]:
        return self._api_key_registry.get_key_record(key_id)

    def _load_registry_unlocked(self) -> dict[str, OpenAIChatGPTRequestPolicyRecord]:
        if not self._registry_path.exists():
            return {}
        try:
            with self._registry_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise RequestPolicyRegistryError(
                f"Failed to read request policy registry {self._registry_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise RequestPolicyRegistryError(
                f"Request policy registry {self._registry_path} must contain a JSON object"
            )

        policies_payload = self._validate_registry_top_level(payload)
        normalized_policies: dict[str, OpenAIChatGPTRequestPolicyRecord] = {}
        for index, item in enumerate(policies_payload, start=1):
            record = self._parse_policy_record(item, index=index)
            if record.key_id in normalized_policies:
                raise RequestPolicyRegistryError(
                    f"Request policy registry {self._registry_path} contains duplicate key_id '{record.key_id}'"
                )
            key_record = self._get_key_record(record.key_id)
            if key_record["group_id"] != record.group_id:
                raise InvalidGroupError(
                    f"Stored policy for key '{record.key_id}' references group '{record.group_id}' but API key registry declares '{key_record['group_id']}'"
                )
            normalized_policies[record.key_id] = record
        return normalized_policies

    def _write_registry_unlocked(
        self, payload: dict[str, OpenAIChatGPTRequestPolicyRecord]
    ) -> None:
        serialized = {
            "version": 1,
            "provider_id": PROVIDER_ID,
            "policies": [
                payload[key_id].to_payload() for key_id in sorted(payload.keys())
            ],
        }
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._registry_path.with_suffix(self._registry_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(serialized, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(self._registry_path)
        except Exception as exc:
            raise RequestPolicyRegistryError(
                f"Failed to write request policy registry {self._registry_path}: {exc}"
            ) from exc

    def _validate_registry_top_level(self, payload: dict[str, Any]) -> list[Any]:
        version = payload.get("version")
        provider_id = payload.get("provider_id")
        policies = payload.get("policies")
        if version != 1:
            raise RequestPolicyRegistryError(
                f"Request policy registry {self._registry_path} must declare version=1"
            )
        if provider_id != PROVIDER_ID:
            raise RequestPolicyRegistryError(
                f"Request policy registry {self._registry_path} must declare provider_id='{PROVIDER_ID}'"
            )
        if not isinstance(policies, list):
            raise RequestPolicyRegistryError(
                f"Request policy registry {self._registry_path} must declare policies as an array"
            )
        return policies

    def _parse_policy_record(
        self, payload: Any, *, index: int
    ) -> OpenAIChatGPTRequestPolicyRecord:
        if not isinstance(payload, dict):
            raise RequestPolicyRegistryError(
                f"Request policy entry #{index} in {self._registry_path} must be an object"
            )
        key_id = self._require_non_empty_string(
            payload.get("key_id"), field_name=f"policies[{index}].key_id"
        )
        group_id = self._require_non_empty_string(
            payload.get("group_id"), field_name=f"policies[{index}].group_id"
        )
        model_overrides = self._normalize_model_overrides(
            payload.get("model_overrides"),
            field_name=f"policies[{index}].model_overrides",
            require_non_empty=False,
        )
        updated_at = self._normalize_timestamp(
            payload.get("updated_at"), field_name=f"policies[{index}].updated_at"
        )
        created_at = payload.get("created_at")
        normalized_created_at = (
            self._normalize_timestamp(
                created_at, field_name=f"policies[{index}].created_at"
            )
            if created_at is not None
            else None
        )
        return OpenAIChatGPTRequestPolicyRecord(
            provider_id=PROVIDER_ID,
            key_id=key_id,
            group_id=group_id,
            model_overrides=model_overrides,
            created_at=normalized_created_at,
            updated_at=updated_at,
        )

    def _normalize_model_overrides(
        self,
        value: Any,
        *,
        field_name: str,
        require_non_empty: bool,
    ) -> dict[str, dict[str, RequestParameterOverride]]:
        if not isinstance(value, dict):
            raise RequestPolicyRegistryError(f"{field_name} must be an object")
        if require_non_empty and not value:
            raise RequestPolicyRegistryError(f"{field_name} must not be empty")

        normalized: dict[str, dict[str, RequestParameterOverride]] = {}
        for model_id, raw_parameter_overrides in value.items():
            normalized_model_id = self._require_non_empty_string(
                model_id, field_name=f"{field_name} model key"
            )
            if not isinstance(raw_parameter_overrides, dict):
                raise RequestPolicyRegistryError(
                    f"{field_name}.{normalized_model_id} must be an object"
                )
            if require_non_empty and not raw_parameter_overrides:
                raise RequestPolicyRegistryError(
                    f"{field_name}.{normalized_model_id} must not be empty"
                )
            normalized_parameters: dict[str, RequestParameterOverride] = {}
            for parameter_name, raw_override in raw_parameter_overrides.items():
                normalized_parameter_name = self._require_non_empty_string(
                    parameter_name,
                    field_name=f"{field_name}.{normalized_model_id} parameter key",
                )
                normalized_parameters[normalized_parameter_name] = (
                    self._normalize_parameter_override(
                        raw_override,
                        field_name=(
                            f"{field_name}.{normalized_model_id}.{normalized_parameter_name}"
                        ),
                    )
                )
            self._validate_capabilities(
                model_id=normalized_model_id,
                parameter_overrides=normalized_parameters,
                field_name=f"{field_name}.{normalized_model_id}",
            )
            normalized[normalized_model_id] = normalized_parameters
        return normalized

    def _normalize_parameter_override(
        self, value: Any, *, field_name: str
    ) -> RequestParameterOverride:
        if not isinstance(value, dict):
            raise RequestPolicyRegistryError(f"{field_name} must be an object")
        mode = self._require_non_empty_string(
            value.get("mode"), field_name=f"{field_name}.mode"
        )
        if mode not in ALLOWED_OVERRIDE_MODES:
            raise RequestPolicyRegistryError(
                f"{field_name}.mode must be one of: force, default_if_absent"
            )
        normalized_value = self._require_non_empty_string(
            value.get("value"), field_name=f"{field_name}.value"
        )
        return RequestParameterOverride(mode=mode, value=normalized_value)

    def _validate_capabilities(
        self,
        *,
        model_id: str,
        parameter_overrides: dict[str, RequestParameterOverride],
        field_name: str,
    ) -> None:
        if not self._capability_validation_enabled:
            return
        capability_record = self._model_capabilities_service.get_model_capabilities(
            model_id
        )
        if capability_record is None:
            raise RequestPolicyRegistryError(
                f"{field_name} references model '{model_id}' without a capability overlay record"
            )
        for parameter_name, override in parameter_overrides.items():
            capability = capability_record.parameters.get(parameter_name)
            if capability is None or not capability.supported:
                raise RequestPolicyRegistryError(
                    f"{field_name} references unsupported parameter '{parameter_name}' for model '{model_id}'"
                )
            if override.value not in capability.values:
                raise RequestPolicyRegistryError(
                    f"{field_name}.{parameter_name} uses unsupported value '{override.value}' for model '{model_id}'"
                )

    @staticmethod
    def _require_non_empty_string(value: Any, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RequestPolicyRegistryError(f"{field_name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _normalize_timestamp(value: Any, *, field_name: str) -> str:
        normalized = (
            OpenAIChatGPTRequestPolicyRegistryService._require_non_empty_string(
                value,
                field_name=field_name,
            )
        )
        candidate = (
            normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
        )
        try:
            datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise RequestPolicyRegistryError(
                f"{field_name} must be a valid ISO 8601 datetime"
            ) from exc
        return normalized


_request_policy_registry_service_singleton: (
    OpenAIChatGPTRequestPolicyRegistryService | None
) = None


def get_openai_chatgpt_request_policy_registry_service() -> (
    OpenAIChatGPTRequestPolicyRegistryService
):
    global _request_policy_registry_service_singleton
    if _request_policy_registry_service_singleton is None:
        _request_policy_registry_service_singleton = (
            OpenAIChatGPTRequestPolicyRegistryService()
        )
    return _request_policy_registry_service_singleton
