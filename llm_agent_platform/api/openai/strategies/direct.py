from __future__ import annotations

import json

from llm_agent_platform.core.utils import create_openai_error
from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.api.openai.providers.base import Provider
from llm_agent_platform.api.openai.types import ChatRequestContext
from llm_agent_platform.api.openai.response_shaper import shape_gemini_nonstream_response
from llm_agent_platform.api.openai.streaming import gemini_chunk_to_sse_events, build_usage_stream_chunk, parse_vertex_stream_line
from llm_agent_platform.config import STRICT_CLI_PARITY
from llm_agent_platform.core.utils import sanitize_data
from llm_agent_platform.services.account_router import AccountRouterError, AllAccountsExhaustedError, quota_account_router

logger = get_logger(__name__)


class DirectStrategy:
    id = "direct"

    @staticmethod
    def _select_account(ctx: ChatRequestContext):
        if not ctx.uses_account_router:
            return None
        return quota_account_router.select_account(
            ctx.provider_runtime_adapter,
            ctx.target_model,
            group_id=ctx.group_id,
        )

    def _build_fallback_chain(self, ctx: ChatRequestContext) -> list[str]:
        fallback_chain = [ctx.target_model]
        if not (ctx.is_quota_mode and STRICT_CLI_PARITY) and 'gemini-3-flash' in ctx.target_model:
            fallback_chain.extend(['gemini-2.5-flash', 'gemini-2.5-flash-lite'])
        elif not (ctx.is_quota_mode and STRICT_CLI_PARITY) and 'gemini-3.1-pro' in ctx.target_model:
            fallback_chain.append('gemini-2.5-pro')
        return fallback_chain

    def execute_non_stream(self, ctx: ChatRequestContext, provider: Provider) -> tuple[str, int]:
        fallback_chain = self._build_fallback_chain(ctx)
        last_status = None
        last_text = ""

        try:
            selected_account = self._select_account(ctx)
        except AllAccountsExhaustedError:
            return create_openai_error("all_accounts_exceed_quota", "quota_exhausted", 429), 429
        except AccountRouterError as router_error:
            if "all accounts on cooldown" in str(router_error):
                return create_openai_error(str(router_error), "rate_limit_error", 429), 429
            raise

        for attempt in range(len(fallback_chain)):
            current_model = fallback_chain[min(attempt, len(fallback_chain) - 1)]
            account = selected_account.account if selected_account is not None else None
            creds = provider.load_runtime_credentials(account)
            upstream = provider.prepare_upstream(ctx, creds, account, model_override=current_model)
            data, status_code = provider.execute_non_stream(ctx, upstream)

            if isinstance(data, dict):
                data_text = json.dumps(data, ensure_ascii=False)
            else:
                data_text = str(data)

            last_status = status_code
            last_text = data_text

            if status_code == 429 and "capacity" in data_text.lower() and len(fallback_chain) > 1:
                logger.warning(f"[WARN] Model {current_model} exhausted (capacity). Trying fallback...")
                continue

            if status_code != 200:
                return create_openai_error(
                    f"Upstream Error: {data_text}",
                    "upstream_error",
                    status_code,
                ), status_code

            if isinstance(data, dict) and data.get("object") == "chat.completion":
                if selected_account is not None:
                    quota_account_router.register_success(
                        ctx.provider_runtime_adapter,
                        selected_account.account.name,
                        group_id=ctx.group_id,
                        model=current_model,
                    )
                return json.dumps(sanitize_data(data), ensure_ascii=False), 200

            if not isinstance(data, dict):
                return create_openai_error(
                    f"Upstream Error: {data_text}",
                    "upstream_error",
                    502,
                ), 502

            return shape_gemini_nonstream_response(data, ctx.raw_model)

        if last_status is None:
            return create_openai_error("Upstream Error (No attempts executed)", "upstream_error", 500), 500

        return create_openai_error(
            f"Upstream Error (All fallbacks exhausted): {last_text}",
            "upstream_error",
            last_status,
        ), last_status

    def stream(self, ctx: ChatRequestContext, provider: Provider):
        usage_accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        saw_usage_chunk = False

        try:
            selected_account = self._select_account(ctx)
            account = selected_account.account if selected_account is not None else None
            creds = provider.load_runtime_credentials(account)
            upstream = provider.prepare_upstream(ctx, creds, account)

            for line in provider.stream_lines(ctx, upstream):
                if not line:
                    continue

                normalized = line.decode(errors="ignore") if isinstance(line, bytes) else str(line)
                fallback_line = line
                if normalized.startswith("data: "):
                    payload = normalized[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(payload)
                    except json.JSONDecodeError:
                        chunk_data = None
                    if isinstance(chunk_data, dict) and chunk_data.get("object") == "chat.completion.chunk":
                        usage = chunk_data.get("usage")
                        if isinstance(usage, dict):
                            usage_accumulated.update(usage)
                            saw_usage_chunk = True
                        yield f"data: {json.dumps(sanitize_data(chunk_data), ensure_ascii=False)}\n\n"
                        continue
                    fallback_line = payload

                chunk_data = parse_vertex_stream_line(fallback_line)
                if not chunk_data:
                    continue

                events, policy_error_payload = gemini_chunk_to_sse_events(
                    chunk_data=chunk_data,
                    raw_model=ctx.raw_model,
                    usage_accumulated=usage_accumulated,
                )
                for event in events:
                    yield event
                if policy_error_payload:
                    yield f"data: {policy_error_payload}\n\n"
                    return

            if selected_account is not None:
                quota_account_router.register_success(
                    ctx.provider_runtime_adapter,
                    selected_account.account.name,
                    group_id=ctx.group_id,
                    model=ctx.target_model,
                )

            if ctx.include_usage and not saw_usage_chunk:
                yield build_usage_stream_chunk(ctx.raw_model, usage_accumulated)

            yield "data: [DONE]\n\n"

        except Exception as exc:
            from llm_agent_platform.services.quota_transport import parse_stream_exception

            status_code, body = parse_stream_exception(exc)
            if status_code is not None:
                logger.error(f"[ERROR] Stream API Error {status_code}: {body}")
                yield f"data: {create_openai_error(f'Upstream API Error: {body}', 'upstream_error', status_code)}\n\n"
                return

            logger.error(f"[ERROR] Stream Exception: {exc}")
            yield f"data: {create_openai_error(str(exc), 'stream_exception')}\n\n"
