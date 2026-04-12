from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_agent_platform.auth.credentials import AuthAvailability
from llm_agent_platform.config import STATE_DIR
from llm_agent_platform.services.http_pool import get_http_client


class ProviderRegistryError(Exception):
    """Raised when provider registry bootstrap data is invalid."""


@dataclass(frozen=True, slots=True)
class ProviderModelDescriptor:
    model_id: str
    display_name: str
    capabilities: tuple[str, ...]
    lifecycle: str
    upstream_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    provider_id: str
    route_name: str
    display_name: str
    auth_kind: str
    runtime_adapter: str
    discovery_enabled: bool
    snapshot_enabled: bool
    bootstrap_models: tuple[ProviderModelDescriptor, ...]
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderCatalogSnapshot:
    provider_id: str
    source: str
    models: tuple[ProviderModelDescriptor, ...]
    as_of: str | None = None


def _registry_root() -> Path:
    return Path(__file__).resolve().parents[1] / "provider_registry"


def _registry_file() -> Path:
    return _registry_root() / "registry.json"


def _provider_models_from_payload(payload: list[dict[str, Any]]) -> tuple[ProviderModelDescriptor, ...]:
    models: list[ProviderModelDescriptor] = []
    for item in payload:
        model_id = str(item["model_id"])
        models.append(
            ProviderModelDescriptor(
                model_id=model_id,
                display_name=str(item["display_name"]),
                capabilities=tuple(str(cap) for cap in item.get("capabilities", [])),
                lifecycle=str(item["lifecycle"]),
                upstream_id=str(item["upstream_id"]),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return tuple(models)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ProviderRegistryError(f"Expected JSON object in {path}")
    return payload


def _load_descriptor(path: Path) -> ProviderDescriptor:
    payload = _load_json(path)
    provider_id = str(payload["provider_id"])
    route_name = str(payload["route_name"])
    if provider_id != route_name:
        raise ProviderRegistryError(
            f"provider_id and route_name must match for {path}: {provider_id!r} != {route_name!r}"
        )

    auth = payload.get("auth") or {}
    transport = payload.get("transport") or {}
    catalog = payload.get("catalog") or {}
    bootstrap = catalog.get("bootstrap") or {}
    discovery = catalog.get("discovery") or {}
    cache = catalog.get("cache") or {}

    return ProviderDescriptor(
        provider_id=provider_id,
        route_name=route_name,
        display_name=str(payload["display_name"]),
        auth_kind=str(auth.get("kind", "")),
        runtime_adapter=str(transport.get("runtime_adapter", "")),
        discovery_enabled=bool(discovery.get("enabled", False)),
        snapshot_enabled=bool(cache.get("snapshot_enabled", False)),
        bootstrap_models=_provider_models_from_payload(list(bootstrap.get("models") or [])),
        raw_payload=payload,
    )


def _snapshot_path(provider_id: str) -> Path:
    return Path(STATE_DIR) / provider_id / "catalog" / "models.json"


def _provider_snapshot_from_payload(payload: dict[str, Any]) -> ProviderCatalogSnapshot:
    provider_id = str(payload["provider_id"])
    models = _provider_models_from_payload(list(payload.get("models") or []))
    if not models:
        raise ProviderRegistryError(f"Provider catalog snapshot for {provider_id} must contain non-empty models")
    as_of = payload.get("as_of")
    return ProviderCatalogSnapshot(
        provider_id=provider_id,
        source=str(payload.get("source", "discovery")),
        models=models,
        as_of=str(as_of) if as_of else None,
    )


def _provider_snapshot_to_payload(snapshot: ProviderCatalogSnapshot) -> dict[str, Any]:
    return {
        "version": 1,
        "provider_id": snapshot.provider_id,
        "as_of": snapshot.as_of or datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": snapshot.source,
        "models": [
            {
                "model_id": model.model_id,
                "display_name": model.display_name,
                "capabilities": list(model.capabilities),
                "lifecycle": model.lifecycle,
                "upstream_id": model.upstream_id,
                "metadata": model.metadata,
            }
            for model in snapshot.models
        ],
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_snapshot(provider: ProviderDescriptor) -> ProviderCatalogSnapshot | None:
    if not (provider.discovery_enabled and provider.snapshot_enabled):
        return None

    snapshot_path = _snapshot_path(provider.provider_id)
    if not snapshot_path.exists():
        return None

    try:
        payload = _load_json(snapshot_path)
        snapshot = _provider_snapshot_from_payload(payload)
    except Exception:
        return None
    if snapshot.provider_id != provider.provider_id:
        return None
    return snapshot


def _oauth_state_path(provider: ProviderDescriptor) -> Path | None:
    auth_metadata = (provider.raw_payload.get("auth") or {}).get("metadata") or {}
    state_path = auth_metadata.get("oauth_state_path")
    if not isinstance(state_path, str) or not state_path:
        return None
    return Path(STATE_DIR) / state_path


def _load_oauth_token(provider: ProviderDescriptor) -> tuple[str, str] | None:
    state_path = _oauth_state_path(provider)
    if state_path is None or not state_path.exists():
        return None

    try:
        payload = _load_json(state_path)
    except Exception:
        return None

    token = str(payload.get("access_token", "")).strip()
    token_type = str(payload.get("token_type", "Bearer")).strip() or "Bearer"
    if not token:
        return None
    return token_type, token


def _transport_metadata(provider: ProviderDescriptor) -> dict[str, Any]:
    return dict((provider.raw_payload.get("transport") or {}).get("metadata") or {})


def _discovery_url(provider: ProviderDescriptor) -> str | None:
    metadata = _transport_metadata(provider)
    env_name = metadata.get("discovery_base_url_env")
    if not isinstance(env_name, str) or not env_name:
        return None
    base_url = os.environ.get(env_name, "").strip().rstrip("/")
    if not base_url:
        return None
    path = str(metadata.get("discovery_models_path", "/v1/models") or "/v1/models")
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base_url}{path}"


def _bootstrap_models_by_id(provider: ProviderDescriptor) -> dict[str, ProviderModelDescriptor]:
    return {model.model_id: model for model in provider.bootstrap_models}


def _discover_openai_models(provider: ProviderDescriptor) -> ProviderCatalogSnapshot | None:
    auth = _load_oauth_token(provider)
    url = _discovery_url(provider)
    if auth is None or url is None:
        return None

    token_type, token = auth
    response = get_http_client().get(
        url,
        headers={
            "Authorization": f"{token_type} {token}",
            "Accept": "application/json",
        },
    )
    if getattr(response, "status_code", 500) != 200:
        return None

    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None

    bootstrap_by_id = _bootstrap_models_by_id(provider)
    models: list[ProviderModelDescriptor] = []
    seen_model_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)

        bootstrap_model = bootstrap_by_id.get(model_id)
        metadata = dict(bootstrap_model.metadata) if bootstrap_model is not None else {}
        owned_by = item.get("owned_by")
        if isinstance(owned_by, str) and owned_by:
            metadata["owned_by"] = owned_by
        metadata["discovered"] = True

        if bootstrap_model is not None:
            models.append(
                ProviderModelDescriptor(
                    model_id=bootstrap_model.model_id,
                    display_name=bootstrap_model.display_name,
                    capabilities=bootstrap_model.capabilities,
                    lifecycle=bootstrap_model.lifecycle,
                    upstream_id=bootstrap_model.upstream_id,
                    metadata=metadata,
                )
            )
            continue

        models.append(
            ProviderModelDescriptor(
                model_id=model_id,
                display_name=model_id,
                capabilities=("chat", "streaming", "tools"),
                lifecycle="ga",
                upstream_id=model_id,
                metadata=metadata,
            )
        )

    if not models:
        return None

    return ProviderCatalogSnapshot(
        provider_id=provider.provider_id,
        source="discovery",
        models=tuple(models),
        as_of=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _refresh_catalog(provider: ProviderDescriptor) -> ProviderCatalogSnapshot | None:
    discovery = (provider.raw_payload.get("catalog") or {}).get("discovery") or {}
    strategy = str(discovery.get("strategy", "")).strip()
    if strategy != "openai-models-list":
        return None

    try:
        return _discover_openai_models(provider)
    except Exception:
        return None


def _persist_snapshot(snapshot: ProviderCatalogSnapshot) -> None:
    _write_json_atomic(_snapshot_path(snapshot.provider_id), _provider_snapshot_to_payload(snapshot))


def _auth_available_for_provider(provider: ProviderDescriptor, availability: AuthAvailability) -> bool:
    kind = provider.auth_kind
    if kind == "gemini-oauth":
        return availability.gemini_quota
    if kind == "qwen-oauth":
        return availability.qwen_quota
    if kind == "openai-oauth":
        return _load_oauth_token(provider) is not None
    if kind == "gcp-service-account":
        return availability.vertex
    if kind == "openai-api-key":
        return False
    return False


class ProviderRegistry:
    def __init__(self, providers: tuple[ProviderDescriptor, ...]):
        self.providers = providers
        self._by_id = {provider.provider_id: provider for provider in providers}

    @classmethod
    def load(cls) -> "ProviderRegistry":
        registry_payload = _load_json(_registry_file())
        providers_payload = registry_payload.get("providers") or []
        if not isinstance(providers_payload, list) or not providers_payload:
            raise ProviderRegistryError("Provider registry must contain non-empty providers list")

        providers: list[ProviderDescriptor] = []
        seen_provider_ids: set[str] = set()
        seen_route_names: set[str] = set()
        root = _registry_root()

        for item in providers_payload:
            if not isinstance(item, dict):
                raise ProviderRegistryError("Each provider registry entry must be an object")
            descriptor_rel_path = item.get("descriptor_path")
            if not isinstance(descriptor_rel_path, str) or not descriptor_rel_path:
                raise ProviderRegistryError("Each provider registry entry must define descriptor_path")
            descriptor = _load_descriptor(root / descriptor_rel_path)
            if descriptor.provider_id in seen_provider_ids:
                raise ProviderRegistryError(f"Duplicate provider_id in registry: {descriptor.provider_id}")
            if descriptor.route_name in seen_route_names:
                raise ProviderRegistryError(f"Duplicate route_name in registry: {descriptor.route_name}")
            seen_provider_ids.add(descriptor.provider_id)
            seen_route_names.add(descriptor.route_name)
            providers.append(descriptor)

        return cls(tuple(providers))

    def get_provider(self, provider_id: str) -> ProviderDescriptor:
        try:
            return self._by_id[provider_id]
        except KeyError as exc:
            raise ProviderRegistryError(f"Unknown provider_id: {provider_id}") from exc

    def load_catalog(self, provider_id: str) -> ProviderCatalogSnapshot:
        provider = self.get_provider(provider_id)
        snapshot = _load_snapshot(provider)
        if provider.discovery_enabled:
            refreshed_snapshot = _refresh_catalog(provider)
            if refreshed_snapshot is not None:
                if provider.snapshot_enabled:
                    _persist_snapshot(refreshed_snapshot)
                return refreshed_snapshot
            if snapshot is not None:
                return snapshot

        return ProviderCatalogSnapshot(
            provider_id=provider.provider_id,
            source="bootstrap",
            models=provider.bootstrap_models,
        )

    def list_models_for_availability(self, availability: AuthAvailability) -> list[str]:
        model_ids: list[str] = []
        seen: set[str] = set()

        for provider in self.providers:
            if not _auth_available_for_provider(provider, availability):
                continue
            for model in self.load_catalog(provider.provider_id).models:
                if model.model_id in seen:
                    continue
                seen.add(model.model_id)
                model_ids.append(model.model_id)

        return model_ids


_provider_registry_singleton: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    global _provider_registry_singleton
    if _provider_registry_singleton is None:
        _provider_registry_singleton = ProviderRegistry.load()
    return _provider_registry_singleton
