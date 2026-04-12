from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def normalize_provider_storage_id(provider_id: str) -> str:
    normalized = provider_id.strip()
    aliases = {
        "gemini": "gemini-cli",
        "gemini_cli": "gemini-cli",
        "gemini-cli": "gemini-cli",
        "openai_chatgpt": "openai-chatgpt",
        "openai-chatgpt": "openai-chatgpt",
        "qwen": "qwen_code",
        "qwen-code": "qwen_code",
        "qwen_code": "qwen_code",
    }
    return aliases.get(normalized, normalized)


def infer_account_name_from_credentials_path(credentials_path: str | Path, *, default: str = "default") -> str:
    candidate = Path(credentials_path)
    stem = candidate.stem.strip()
    if not stem or stem in {"oauth-account", "user_credentials"}:
        return default
    return stem


@dataclass(slots=True)
class CredentialsPathRef:
    provider_id: str
    credentials_path: Path
    account_name: str


def resolve_credentials_path_ref(
    provider_id: str,
    credentials_path: str | Path,
    *,
    account_name: str | None = None,
) -> CredentialsPathRef:
    resolved_path = Path(credentials_path)
    resolved_account_name = (account_name or "").strip() or infer_account_name_from_credentials_path(resolved_path)
    return CredentialsPathRef(
        provider_id=normalize_provider_storage_id(provider_id),
        credentials_path=resolved_path,
        account_name=resolved_account_name,
    )
