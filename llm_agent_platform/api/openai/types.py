from __future__ import annotations

from dataclasses import dataclass

from llm_agent_platform.services.account_router import SelectedAccount


@dataclass
class ChatRequestContext:
    provider_id: str
    provider_runtime_adapter: str
    provider_transport_kind: str
    provider_auth_kind: str
    provider_mode: str | None
    uses_account_router: bool
    allowed_model_ids: tuple[str, ...] | None
    data: dict
    raw_model: str
    target_model: str
    messages: list
    stream: bool
    include_usage: bool
    contents: list
    system_instruction: str | None
    gemini_config: dict
    gemini_tools: list
    is_qwen_quota_mode: bool
    is_gemini_quota_mode: bool
    is_quota_mode: bool
    session_id: str | None
    user_prompt_id: str | None
    group_id: str


@dataclass
class UpstreamRequestContext:
    token: str | None
    payload: dict
    url: str
    headers: dict
    params: dict
    selected_account: SelectedAccount | None
    quota_request_payload: dict | None
    session_id: str | None
    user_prompt_id: str | None
    account_name: str | None = None
    credentials_path: str | None = None


class UpstreamPreparationError(Exception):
    def __init__(self, message: str, error_type: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
