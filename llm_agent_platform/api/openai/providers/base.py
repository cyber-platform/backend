from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable, Any

from llm_agent_platform.api.openai.types import ChatRequestContext, UpstreamRequestContext
from llm_agent_platform.services.account_router import BaseAccount


@dataclass(frozen=True)
class ProviderRuntimeCreds:
    token: str | None
    resource_url: str | None = None
    token_type: str = "Bearer"
    account_id: str | None = None
    metadata: dict | None = None


class Provider(Protocol):
    """Protocol for LLM provider integration."""

    id: str

    def load_runtime_credentials(self, account: BaseAccount | None) -> ProviderRuntimeCreds:
        """Load runtime creds from files/env (no bootstrap creation here)."""

    def prepare_upstream(
        self,
        ctx: ChatRequestContext,
        creds: ProviderRuntimeCreds,
        account: BaseAccount | None,
        model_override: str | None = None,
    ) -> UpstreamRequestContext:
        """Build upstream request context with headers/url/payload."""

    def execute_non_stream(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> tuple[Any, int]:
        """Execute non-stream upstream call and return (data, status_code)."""

    def stream_lines(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> Iterable[str | bytes]:
        """Stream upstream lines for SSE processing."""
