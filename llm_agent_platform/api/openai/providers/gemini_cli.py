from __future__ import annotations

from typing import Iterable, Any

from llm_agent_platform.api.openai.providers.base import Provider, ProviderRuntimeCreds
from llm_agent_platform.api.openai.types import ChatRequestContext, UpstreamRequestContext, UpstreamPreparationError
from llm_agent_platform.auth.credentials import get_auth_lock, get_gemini_access_token_from_file
from llm_agent_platform.config import CLOUD_CODE_ENDPOINT
from llm_agent_platform.services.account_router import GeminiAccount
from llm_agent_platform.services.quota_transport import (
    build_quota_payload,
    send_generate,
    stream_generate_lines,
)


class GeminiCliProvider(Provider):
    id = "gemini_cli"

    def load_runtime_credentials(self, account) -> ProviderRuntimeCreds:
        if not isinstance(account, GeminiAccount):
            raise UpstreamPreparationError("Invalid Gemini account configuration", "config_error", 500)
        with get_auth_lock():
            token = get_gemini_access_token_from_file(account.credentials_path)
        return ProviderRuntimeCreds(token=token)

    def prepare_upstream(
        self,
        ctx: ChatRequestContext,
        creds: ProviderRuntimeCreds,
        account,
        model_override: str | None = None,
    ) -> UpstreamRequestContext:
        if not isinstance(account, GeminiAccount):
            raise UpstreamPreparationError("Invalid Gemini account configuration", "config_error", 500)

        quota_request_payload = {
            "contents": ctx.contents,
            "generationConfig": ctx.gemini_config,
        }
        if ctx.gemini_tools:
            quota_request_payload["tools"] = ctx.gemini_tools
        if ctx.system_instruction:
            quota_request_payload["systemInstruction"] = {
                "parts": [{"text": ctx.system_instruction}]
            }

        payload = build_quota_payload(
            model=model_override or ctx.target_model,
            project=account.project_id,
            request_payload=quota_request_payload,
            user_prompt_id=ctx.user_prompt_id,
            session_id=ctx.session_id,
        )
        url = f"{CLOUD_CODE_ENDPOINT}:{'streamGenerateContent' if ctx.stream else 'generateContent'}"
        params = {"alt": "sse"} if ctx.stream else {}
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }
        return UpstreamRequestContext(
            token=creds.token,
            payload=payload,
            url=url,
            headers=headers,
            params=params,
            selected_account=None,
            quota_request_payload=quota_request_payload,
            session_id=ctx.session_id,
            user_prompt_id=ctx.user_prompt_id,
            account_name=account.name,
            credentials_path=account.credentials_path,
        )

    def execute_non_stream(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> tuple[Any, int]:
        response = send_generate(
            upstream.token,
            upstream.payload,
        )
        try:
            data = response.json()
        except Exception:
            data = response.text
        return data, response.status_code

    def stream_lines(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> Iterable[str | bytes]:
        return stream_generate_lines(
            upstream.token,
            upstream.payload,
        )
