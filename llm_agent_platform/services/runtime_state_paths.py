from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_agent_platform import config
from llm_agent_platform.services.credentials_paths import normalize_provider_storage_id

ACCOUNT_STATE_FILENAME = "account_state.json"
GROUP_QUOTA_STATE_FILENAME = "quota_state.json"
USAGE_LIMITS_FILENAME = "limits.json"


@dataclass(slots=True)
class RuntimeStatePaths:
    provider_id: str
    account_name: str
    root_dir: Path

    @property
    def provider_dir(self) -> Path:
        return Path(self.root_dir) / self.provider_id

    @property
    def account_dir(self) -> Path:
        return self.provider_dir / "accounts" / self.account_name

    @property
    def account_state_path(self) -> Path:
        return self.account_dir / ACCOUNT_STATE_FILENAME

    @property
    def usage_account_dir(self) -> Path:
        return self.provider_dir / "usage" / "accounts" / self.account_name

    @property
    def usage_snapshot_path(self) -> Path:
        return self.usage_account_dir / USAGE_LIMITS_FILENAME

    def group_quota_state_path(self, group_id: str) -> Path:
        return self.provider_dir / "groups" / group_id / GROUP_QUOTA_STATE_FILENAME


def default_state_root() -> Path:
    return Path(config.STATE_DIR)


def resolve_runtime_state_paths(
    provider_id: str,
    *,
    account_name: str,
    root_dir: str | Path | None = None,
) -> RuntimeStatePaths:
    return RuntimeStatePaths(
        provider_id=normalize_provider_storage_id(provider_id),
        account_name=account_name.strip() or "default",
        root_dir=Path(root_dir) if root_dir is not None else default_state_root(),
    )
