from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Any

from llm_agent_platform.api.openai.providers.base import Provider, ProviderRuntimeCreds
from llm_agent_platform.api.openai.types import ChatRequestContext, UpstreamRequestContext, UpstreamPreparationError
from llm_agent_platform.auth.qwen_oauth import refresh_qwen_credentials_file, read_qwen_credentials
from llm_agent_platform.config import QWEN_REFRESH_IDLE_THRESHOLD_SECONDS
from llm_agent_platform.services.account_state_store import (
    AccountStatePaths,
    default_state_root,
    load_last_used_at,
    save_last_used_at,
)
from llm_agent_platform.services.account_router import BaseAccount
from llm_agent_platform.services.quota_transport import send_generate_to_url, stream_generate_lines_from_url


def _qwen_completion_url(resource_url: str) -> str:
    normalized = resource_url.strip()
    if not normalized.startswith('http://') and not normalized.startswith('https://'):
        normalized = f'https://{normalized}'
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    if not normalized.endswith('/v1'):
        normalized = f'{normalized}/v1'
    return f'{normalized}/chat/completions'


def _qwen_payload_from_openai(data: dict, target_model: str) -> dict:
    payload = {
        "model": target_model,
        "messages": data.get('messages', []),
        "stream": bool(data.get('stream', False)),
    }

    for field in [
        'temperature',
        'top_p',
        'max_tokens',
        'max_completion_tokens',
        'presence_penalty',
        'frequency_penalty',
        'stop',
        'tools',
        'tool_choice',
        'response_format',
        'stream_options',
    ]:
        if field in data:
            payload[field] = data[field]

    return payload


class QwenCodeProvider(Provider):
    id = "qwen_code"

    def load_runtime_credentials(self, account) -> ProviderRuntimeCreds:
        if not isinstance(account, BaseAccount):
            raise UpstreamPreparationError("Invalid Qwen account configuration", "config_error", 500)

        paths = AccountStatePaths(
            provider_id=self.id,
            account_name=account.name,
            root_dir=default_state_root(),
        )
        should_refresh = _should_refresh_credentials(paths)

        try:
            if should_refresh:
                qwen_creds = refresh_qwen_credentials_file(account.credentials_path)
            else:
                qwen_creds = read_qwen_credentials(account.credentials_path)
        except Exception as exc:
            raise UpstreamPreparationError(f"Qwen OAuth refresh failed: {exc}", "auth_error", 401)

        token = qwen_creds.get("access_token")
        resource_url = qwen_creds.get("resource_url")
        if not token or not resource_url:
            raise UpstreamPreparationError(
                "Qwen OAuth credentials must include access_token and resource_url",
                "auth_error",
                401,
            )

        return ProviderRuntimeCreds(token=token, resource_url=resource_url)

    def prepare_upstream(
        self,
        ctx: ChatRequestContext,
        creds: ProviderRuntimeCreds,
        account,
        model_override: str | None = None,
    ) -> UpstreamRequestContext:
        payload = _qwen_payload_from_openai(ctx.data, model_override or ctx.target_model)
        url = _qwen_completion_url(creds.resource_url or "")
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }
        return UpstreamRequestContext(
            token=creds.token,
            payload=payload,
            url=url,
            headers=headers,
            params={},
            selected_account=None,
            quota_request_payload=None,
            session_id=ctx.session_id,
            user_prompt_id=ctx.user_prompt_id,
            account_name=account.name if isinstance(account, BaseAccount) else None,
            credentials_path=account.credentials_path if isinstance(account, BaseAccount) else None,
        )

    def execute_non_stream(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> tuple[Any, int]:
        response = send_generate_to_url(
            upstream.token,
            upstream.payload,
            upstream.url,
        )
        if response.status_code in {401, 403} and upstream.credentials_path:
            retry = _refresh_and_retry(upstream)
            if retry is not None:
                response = retry
        try:
            data = response.json()
        except Exception:
            data = response.text
        if response.status_code not in {401, 403}:
            _touch_last_used(upstream)
        return data, response.status_code

    def stream_lines(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> Iterable[str | bytes]:
        had_activity = False
        try:
            for line in stream_generate_lines_from_url(
                upstream.token,
                upstream.payload,
                upstream.url,
                params=None,
            ):
                had_activity = True
                yield line
        except Exception as exc:
            message = str(exc)
            if upstream.credentials_path and (message.startswith("401:") or message.startswith("403:")):
                retry = _refresh_and_retry(upstream, stream=True)
                if retry is not None:
                    for line in retry:
                        had_activity = True
                        yield line
                    return
            raise
        finally:
            if had_activity:
                _touch_last_used(upstream)


def _state_paths(upstream: UpstreamRequestContext) -> AccountStatePaths | None:
    if not upstream.account_name:
        return None
    return AccountStatePaths(
        provider_id="qwen_code",
        account_name=upstream.account_name,
        root_dir=default_state_root(),
    )


def _should_refresh_credentials(paths: AccountStatePaths) -> bool:
    last_used = load_last_used_at(paths)
    if last_used is None:
        return True
    elapsed = (datetime.now(tz=timezone.utc) - last_used).total_seconds()
    return elapsed > QWEN_REFRESH_IDLE_THRESHOLD_SECONDS


def _touch_last_used(upstream: UpstreamRequestContext) -> None:
    paths = _state_paths(upstream)
    if not paths:
        return
    save_last_used_at(paths, datetime.now(tz=timezone.utc))


def _refresh_and_retry(
    upstream: UpstreamRequestContext,
    *,
    stream: bool = False,
) -> Any | None:
    try:
        refreshed = refresh_qwen_credentials_file(upstream.credentials_path)
    except Exception:
        return None

    token = refreshed.get("access_token")
    resource_url = refreshed.get("resource_url")
    if not token or not resource_url:
        return None

    url = _qwen_completion_url(resource_url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    upstream.token = token
    upstream.url = url
    upstream.headers = headers

    if stream:
        return stream_generate_lines_from_url(
            upstream.token,
            upstream.payload,
            upstream.url,
            params=None,
            extra_headers=headers,
        )

    return send_generate_to_url(
        upstream.token,
        upstream.payload,
        upstream.url,
        extra_headers=headers,
    )
