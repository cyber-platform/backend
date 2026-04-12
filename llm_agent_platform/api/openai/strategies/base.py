from __future__ import annotations

from typing import Protocol, Iterable

from llm_agent_platform.api.openai.providers.base import Provider
from llm_agent_platform.api.openai.types import ChatRequestContext


class ExecutionStrategy(Protocol):
    id: str

    def execute_non_stream(self, ctx: ChatRequestContext, provider: Provider) -> tuple[str, int]:
        """Return OpenAI-compatible JSON string and status code."""

    def stream(self, ctx: ChatRequestContext, provider: Provider) -> Iterable[str]:
        """Yield SSE lines for OpenAI streaming responses."""
