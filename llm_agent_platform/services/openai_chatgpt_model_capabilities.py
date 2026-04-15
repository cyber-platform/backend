from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_agent_platform.config import OPENAI_CHATGPT_MODEL_CAPABILITIES_PATH
from llm_agent_platform.services.provider_registry import (
    ProviderModelDescriptor,
    ProviderRegistry,
    get_provider_registry,
)

PROVIDER_ID = "openai-chatgpt"


class ModelCapabilitiesRegistryError(RuntimeError):
    """Raised when the capability overlay cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class SupportedParameterCapability:
    supported: bool
    values: tuple[str, ...] = ()
    default: str | None = None
    ui_label: str | None = None
    description: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"supported": self.supported}
        if self.supported:
            payload["values"] = list(self.values)
            payload["default"] = self.default
        if self.ui_label is not None:
            payload["ui_label"] = self.ui_label
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class CapabilityConstraintRule:
    kind: str
    when: dict[str, tuple[str, ...]]
    forbid_parameters: tuple[str, ...]
    description: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "when": {name: list(values) for name, values in self.when.items()},
            "forbid_parameters": list(self.forbid_parameters),
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class OpenAIChatGPTModelCapabilitiesRecord:
    provider_id: str
    model_id: str
    display_name: str
    catalog_model: ProviderModelDescriptor
    parameters: dict[str, SupportedParameterCapability]
    constraints: tuple[CapabilityConstraintRule, ...]
    raw: dict[str, Any]

    def to_admin_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "parameters": {
                name: capability.to_payload()
                for name, capability in self.parameters.items()
            },
        }
        if self.raw:
            payload["drawer"] = {"raw_capability_payload": self.raw}
        return payload


class OpenAIChatGPTModelCapabilitiesService:
    def __init__(
        self,
        registry_path: str | Path | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self._registry_path = Path(
            registry_path or OPENAI_CHATGPT_MODEL_CAPABILITIES_PATH
        )
        self._provider_registry = provider_registry or get_provider_registry()

    def get_model_capabilities(
        self, model_id: str
    ) -> OpenAIChatGPTModelCapabilitiesRecord | None:
        normalized_model_id = self._require_non_empty_string(
            model_id, field_name="model_id"
        )
        return self._load_registry().get(normalized_model_id)

    def list_model_capabilities(
        self,
    ) -> tuple[OpenAIChatGPTModelCapabilitiesRecord, ...]:
        records = self._load_registry()
        return tuple(records[model_id] for model_id in sorted(records))

    def _load_registry(self) -> dict[str, OpenAIChatGPTModelCapabilitiesRecord]:
        if not self._registry_path.exists():
            return {}

        try:
            with self._registry_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise ModelCapabilitiesRegistryError(
                f"Failed to read capability overlay {self._registry_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay {self._registry_path} must contain a JSON object"
            )

        models_payload = self._validate_registry_top_level(payload)
        catalog_by_id = self._catalog_models_by_id()
        records: dict[str, OpenAIChatGPTModelCapabilitiesRecord] = {}
        for model_id, model_payload in models_payload.items():
            catalog_model = catalog_by_id.get(model_id)
            if catalog_model is None:
                raise ModelCapabilitiesRegistryError(
                    f"Capability overlay {self._registry_path} references unknown provider_registry model_id '{model_id}'"
                )
            records[model_id] = self._parse_model_record(
                model_id=model_id,
                payload=model_payload,
                catalog_model=catalog_model,
            )
        return records

    def _catalog_models_by_id(self) -> dict[str, ProviderModelDescriptor]:
        catalog = self._provider_registry.load_catalog(PROVIDER_ID)
        return {model.model_id: model for model in catalog.models}

    def _validate_registry_top_level(self, payload: dict[str, Any]) -> dict[str, Any]:
        version = payload.get("version")
        provider_id = payload.get("provider_id")
        models = payload.get("models")
        if version != 1:
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay {self._registry_path} must declare version=1"
            )
        if provider_id != PROVIDER_ID:
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay {self._registry_path} must declare provider_id='{PROVIDER_ID}'"
            )
        if not isinstance(models, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay {self._registry_path} must declare models as an object"
            )
        return models

    def _parse_model_record(
        self,
        *,
        model_id: str,
        payload: Any,
        catalog_model: ProviderModelDescriptor,
    ) -> OpenAIChatGPTModelCapabilitiesRecord:
        if not isinstance(payload, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay model '{model_id}' in {self._registry_path} must be an object"
            )

        display_name = payload.get("display_name")
        parameters_payload = payload.get("parameters")
        raw_payload = payload.get("raw")
        constraints_payload = payload.get("constraints", [])

        if not isinstance(parameters_payload, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay model '{model_id}' in {self._registry_path} must define parameters as an object"
            )
        if not isinstance(raw_payload, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay model '{model_id}' in {self._registry_path} must define raw as an object"
            )
        if not isinstance(constraints_payload, list):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay model '{model_id}' in {self._registry_path} must define constraints as an array"
            )

        parameters = self._parse_parameters(
            model_id=model_id, payload=parameters_payload
        )
        constraints = self._parse_constraints(
            model_id=model_id,
            payload=constraints_payload,
            declared_parameters=set(parameters),
        )
        normalized_display_name = (
            self._require_non_empty_string(
                display_name, field_name=f"models.{model_id}.display_name"
            )
            if display_name is not None
            else catalog_model.display_name
        )

        return OpenAIChatGPTModelCapabilitiesRecord(
            provider_id=PROVIDER_ID,
            model_id=model_id,
            display_name=normalized_display_name,
            catalog_model=catalog_model,
            parameters=parameters,
            constraints=constraints,
            raw=dict(raw_payload),
        )

    def _parse_parameters(
        self, *, model_id: str, payload: dict[str, Any]
    ) -> dict[str, SupportedParameterCapability]:
        parameters: dict[str, SupportedParameterCapability] = {}
        for parameter_name, parameter_payload in payload.items():
            normalized_name = self._require_non_empty_string(
                parameter_name,
                field_name=f"models.{model_id}.parameters key",
            )
            parameters[normalized_name] = self._parse_parameter_capability(
                model_id=model_id,
                parameter_name=normalized_name,
                payload=parameter_payload,
            )
        return parameters

    def _parse_parameter_capability(
        self,
        *,
        model_id: str,
        parameter_name: str,
        payload: Any,
    ) -> SupportedParameterCapability:
        if not isinstance(payload, dict):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay parameter '{parameter_name}' for model '{model_id}' in {self._registry_path} must be an object"
            )

        supported = payload.get("supported")
        if not isinstance(supported, bool):
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay parameter '{parameter_name}' for model '{model_id}' in {self._registry_path} must define boolean supported"
            )

        values_payload = payload.get("values")
        default_value = payload.get("default")
        ui_label = payload.get("ui_label")
        description = payload.get("description")

        if supported:
            values = self._parse_string_list(
                values_payload,
                field_name=f"models.{model_id}.parameters.{parameter_name}.values",
            )
            normalized_default = self._require_non_empty_string(
                default_value,
                field_name=f"models.{model_id}.parameters.{parameter_name}.default",
            )
            if normalized_default not in values:
                raise ModelCapabilitiesRegistryError(
                    f"Capability overlay parameter '{parameter_name}' for model '{model_id}' in {self._registry_path} must use a default that exists in values"
                )
            return SupportedParameterCapability(
                supported=True,
                values=values,
                default=normalized_default,
                ui_label=(
                    self._require_non_empty_string(
                        ui_label,
                        field_name=f"models.{model_id}.parameters.{parameter_name}.ui_label",
                    )
                    if ui_label is not None
                    else None
                ),
                description=(
                    self._require_non_empty_string(
                        description,
                        field_name=f"models.{model_id}.parameters.{parameter_name}.description",
                    )
                    if description is not None
                    else None
                ),
            )

        if values_payload is not None or default_value is not None:
            raise ModelCapabilitiesRegistryError(
                f"Capability overlay parameter '{parameter_name}' for model '{model_id}' in {self._registry_path} cannot define values/default when supported=false"
            )
        return SupportedParameterCapability(
            supported=False,
            ui_label=(
                self._require_non_empty_string(
                    ui_label,
                    field_name=f"models.{model_id}.parameters.{parameter_name}.ui_label",
                )
                if ui_label is not None
                else None
            ),
            description=(
                self._require_non_empty_string(
                    description,
                    field_name=f"models.{model_id}.parameters.{parameter_name}.description",
                )
                if description is not None
                else None
            ),
        )

    def _parse_constraints(
        self,
        *,
        model_id: str,
        payload: list[Any],
        declared_parameters: set[str],
    ) -> tuple[CapabilityConstraintRule, ...]:
        constraints: list[CapabilityConstraintRule] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ModelCapabilitiesRegistryError(
                    f"Capability overlay constraint #{index} for model '{model_id}' in {self._registry_path} must be an object"
                )
            kind = item.get("kind")
            if kind != "forbid_parameters_when":
                raise ModelCapabilitiesRegistryError(
                    f"Capability overlay constraint #{index} for model '{model_id}' in {self._registry_path} has unsupported kind"
                )
            when_payload = item.get("when")
            if not isinstance(when_payload, dict) or not when_payload:
                raise ModelCapabilitiesRegistryError(
                    f"Capability overlay constraint #{index} for model '{model_id}' in {self._registry_path} must define non-empty when"
                )
            forbid_parameters = self._parse_string_list(
                item.get("forbid_parameters"),
                field_name=f"models.{model_id}.constraints[{index}].forbid_parameters",
            )

            when: dict[str, tuple[str, ...]] = {}
            for parameter_name, values_payload in when_payload.items():
                normalized_name = self._require_non_empty_string(
                    parameter_name,
                    field_name=f"models.{model_id}.constraints[{index}].when key",
                )
                if normalized_name not in declared_parameters:
                    raise ModelCapabilitiesRegistryError(
                        f"Capability overlay constraint #{index} for model '{model_id}' in {self._registry_path} references undeclared trigger parameter '{normalized_name}'"
                    )
                when[normalized_name] = self._parse_string_list(
                    values_payload,
                    field_name=f"models.{model_id}.constraints[{index}].when.{normalized_name}",
                )

            description = item.get("description")
            constraints.append(
                CapabilityConstraintRule(
                    kind="forbid_parameters_when",
                    when=when,
                    forbid_parameters=forbid_parameters,
                    description=(
                        self._require_non_empty_string(
                            description,
                            field_name=f"models.{model_id}.constraints[{index}].description",
                        )
                        if description is not None
                        else None
                    ),
                )
            )
        return tuple(constraints)

    @staticmethod
    def _parse_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ModelCapabilitiesRegistryError(
                f"{field_name} must be a non-empty array"
            )
        normalized_values: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized_item = (
                OpenAIChatGPTModelCapabilitiesService._require_non_empty_string(
                    item,
                    field_name=field_name,
                )
            )
            if normalized_item in seen:
                raise ModelCapabilitiesRegistryError(
                    f"{field_name} must not contain duplicates"
                )
            seen.add(normalized_item)
            normalized_values.append(normalized_item)
        return tuple(normalized_values)

    @staticmethod
    def _require_non_empty_string(value: Any, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ModelCapabilitiesRegistryError(
                f"{field_name} must be a non-empty string"
            )
        return value.strip()


_model_capabilities_service_singleton: OpenAIChatGPTModelCapabilitiesService | None = (
    None
)


def get_openai_chatgpt_model_capabilities_service() -> (
    OpenAIChatGPTModelCapabilitiesService
):
    global _model_capabilities_service_singleton
    if _model_capabilities_service_singleton is None:
        _model_capabilities_service_singleton = OpenAIChatGPTModelCapabilitiesService()
    return _model_capabilities_service_singleton
