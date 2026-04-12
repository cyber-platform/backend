from __future__ import annotations

import json

from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.core.utils import create_openai_error, sanitize_data
from llm_agent_platform.api.openai.providers.openai_chatgpt import OpenAIChatGPTProvider
from llm_agent_platform.api.openai.providers.base import Provider
from llm_agent_platform.api.openai.providers.qwen_code import QwenCodeProvider
from llm_agent_platform.api.openai.types import ChatRequestContext
from llm_agent_platform.api.openai.response_shaper import shape_gemini_nonstream_response
from llm_agent_platform.api.openai.streaming import gemini_chunk_to_sse_events, build_usage_stream_chunk
from llm_agent_platform.services.account_router import (
    quota_account_router,
    RotationEvent,
    AllAccountsExhaustedError,
    AccountRouterError,
    SelectedAccount,
)
from llm_agent_platform.services.quota_transport import (
    classify_429_response,
    classify_429_exception,
    classify_429_error_payload,
    Quota429Type,
    parse_cloud_code_sse_line,
    unwrap_cloud_code_response,
)

logger = get_logger(__name__)


class RotateOn429RoundingStrategy:
    id = "rotate_on_429_rounding"

    @staticmethod
    def _openai_chunk_from_qwen(parsed: dict, raw_model: str) -> dict:
        choice = (parsed.get("choices") or [{}])[0]
        delta = choice.get("delta", {})

        if not delta and "message" in choice:
            msg = choice.get("message", {})
            delta = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
            }

        return {
            "id": parsed.get("id"),
            "object": "chat.completion.chunk",
            "created": parsed.get("created"),
            "model": raw_model,
            "choices": [{
                "index": choice.get("index", 0),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }],
        }

    def _select_account(self, provider: str, model: str, group_id: str) -> SelectedAccount:
        return quota_account_router.select_account(provider, model, group_id=group_id)

    @staticmethod
    def _uses_openai_chunks(ctx: ChatRequestContext) -> bool:
        return ctx.provider_runtime_adapter in {QwenCodeProvider.id, OpenAIChatGPTProvider.id}

    def execute_non_stream(self, ctx: ChatRequestContext, provider: Provider) -> tuple[str, int]:
        provider_id = ctx.provider_runtime_adapter
        group_id = ctx.group_id
        try:
            selected_account = self._select_account(provider_id, ctx.target_model, group_id)
        except AllAccountsExhaustedError:
            return create_openai_error("all_accounts_exceed_quota", "quota_exhausted", 429), 429
        except AccountRouterError as router_error:
            if str(router_error) == "all_accounts_on_cooldown":
                return create_openai_error(str(router_error), "rate_limit_error", 429), 429
            if "all accounts on cooldown" in str(router_error):
                return create_openai_error(str(router_error), "rate_limit_error", 429), 429
            raise

        creds = provider.load_runtime_credentials(selected_account.account)
        upstream = provider.prepare_upstream(ctx, creds, selected_account.account)

        fallback_chain = [ctx.target_model]
        max_attempts = len(fallback_chain)
        if selected_account.mode == "rounding":
            max_attempts = max(max_attempts, len(selected_account.pool))

        attempt = 0
        last_status = None
        last_text = ""
        while attempt < max_attempts:
            current_model = fallback_chain[min(attempt, len(fallback_chain) - 1)]
            upstream = provider.prepare_upstream(ctx, creds, selected_account.account, model_override=current_model)

            data, status_code = provider.execute_non_stream(ctx, upstream)
            last_status = status_code
            data_text = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
            last_text = data_text

            error_kind = classify_429_response(status_code, data_text)
            if error_kind in {Quota429Type.RATE_LIMIT, Quota429Type.QUOTA_EXHAUSTED}:
                event = (
                    RotationEvent.RATE_LIMIT
                    if error_kind == Quota429Type.RATE_LIMIT
                    else RotationEvent.QUOTA_EXHAUSTED
                )
                event_result = quota_account_router.register_event(
                    provider=provider_id,
                    account_name=selected_account.account.name,
                    mode=selected_account.mode,
                    pool=selected_account.pool,
                    event=event,
                    model=current_model,
                    group_id=group_id,
                )
                if event_result.all_exhausted:
                    return create_openai_error("all_accounts_exceed_quota", "quota_exhausted", 429), 429
                if event_result.all_cooldown:
                    wait_seconds = quota_account_router.cooldown_wait_seconds(
                        provider_id,
                        selected_account.pool,
                        group_id=group_id,
                    )
                    return create_openai_error(
                        f"all accounts on cooldown please wait {wait_seconds}",
                        "rate_limit_error",
                        429,
                    ), 429

                if event_result.switched and selected_account.mode == "rounding":
                    old_account = selected_account.account.name
                    logger.info(f"[{provider_id}] Account rotation triggered in non-stream, switching from {old_account}")
                    try:
                        selected_account = quota_account_router.select_account(
                            provider_id,
                            current_model,
                            group_id=group_id,
                        )
                    except AllAccountsExhaustedError:
                        return create_openai_error("all_accounts_exceed_quota", "quota_exhausted", 429), 429
                    except AccountRouterError as router_error:
                        if "all accounts on cooldown" in str(router_error):
                            return create_openai_error(str(router_error), "rate_limit_error", 429), 429
                        raise

                    creds = provider.load_runtime_credentials(selected_account.account)
                    attempt += 1
                    continue

            if status_code != 200:
                return create_openai_error(f"Upstream Error: {data_text}", "upstream_error", status_code), status_code

            if isinstance(data, dict) and data.get("object") == "chat.completion":
                data["model"] = ctx.raw_model
                quota_account_router.register_success(
                    provider_id,
                    selected_account.account.name,
                    group_id=group_id,
                    model=current_model,
                )
                return json.dumps(sanitize_data(data), ensure_ascii=False), 200

            if ctx.is_qwen_quota_mode:
                if isinstance(data, dict):
                    data["model"] = ctx.raw_model
                quota_account_router.register_success(provider_id, selected_account.account.name, group_id=group_id)
                return json.dumps(sanitize_data(data), ensure_ascii=False), 200

            if not isinstance(data, dict):
                return create_openai_error(f"Upstream Error: {data_text}", "upstream_error", 502), 502

            data = unwrap_cloud_code_response(data)
            quota_account_router.register_success(provider_id, selected_account.account.name, group_id=group_id)
            return shape_gemini_nonstream_response(data, ctx.raw_model)

        if last_status is None:
            return create_openai_error("Upstream Error (No attempts executed)", "upstream_error", 500), 500

        return create_openai_error(
            f"Upstream Error (All fallbacks exhausted): {last_text}",
            "upstream_error",
            last_status,
        ), last_status

    def stream(self, ctx: ChatRequestContext, provider: Provider):
        provider_id = ctx.provider_runtime_adapter
        group_id = ctx.group_id
        try:
            selected_account = self._select_account(provider_id, ctx.target_model, group_id)
        except AllAccountsExhaustedError:
            yield f"data: {create_openai_error('all_accounts_exceed_quota', 'quota_exhausted', 429)}\n\n"
            return
        except AccountRouterError as router_error:
            if "all accounts on cooldown" in str(router_error):
                yield f"data: {create_openai_error(str(router_error), 'rate_limit_error', 429)}\n\n"
                return
            raise

        usage_accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            creds = provider.load_runtime_credentials(selected_account.account)
            upstream = provider.prepare_upstream(ctx, creds, selected_account.account)

            for line in provider.stream_lines(ctx, upstream):
                if not line:
                    continue

                if self._uses_openai_chunks(ctx) and isinstance(line, str) and line.startswith("data: "):
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        quota_account_router.register_success(provider_id, selected_account.account.name, group_id=group_id)
                        break
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        continue

                    if isinstance(parsed, dict) and parsed.get("error"):
                        error_text = json.dumps(parsed.get("error"), ensure_ascii=False)
                        if selected_account.mode == "rounding":
                            error_kind = classify_429_error_payload(parsed.get("error"))
                            event = RotationEvent.QUOTA_EXHAUSTED
                            if error_kind == Quota429Type.RATE_LIMIT:
                                event = RotationEvent.RATE_LIMIT
                            event_result = quota_account_router.register_event(
                                provider=provider_id,
                                account_name=selected_account.account.name,
                                mode=selected_account.mode,
                                pool=selected_account.pool,
                                event=event,
                                model=ctx.target_model,
                                group_id=group_id,
                            )
                            if event_result.all_exhausted:
                                yield f"data: {create_openai_error('all_accounts_exceed_quota', 'quota_exhausted', 429)}\n\n"
                                return
                            if event_result.all_cooldown:
                                wait_seconds = quota_account_router.cooldown_wait_seconds(
                                    provider_id,
                                    selected_account.pool,
                                    group_id=group_id,
                                )
                                yield f"data: {create_openai_error(f'all accounts on cooldown please wait {wait_seconds}', 'rate_limit_error', 429)}\n\n"
                                return
                            if event_result.switched:
                                old_account = selected_account.account.name
                                logger.info(f"[{provider_id}] Account rotation triggered in stream, switching from {old_account}")
                                try:
                                    selected_account = quota_account_router.select_account(
                                        provider_id,
                                        ctx.target_model,
                                        group_id=group_id,
                                    )
                                    creds = provider.load_runtime_credentials(selected_account.account)
                                    upstream = provider.prepare_upstream(ctx, creds, selected_account.account)
                                except AllAccountsExhaustedError:
                                    yield f"data: {create_openai_error('all_accounts_exceed_quota', 'quota_exhausted', 429)}\n\n"
                                    return
                                except AccountRouterError as router_error:
                                    if "all accounts on cooldown" in str(router_error):
                                        yield f"data: {create_openai_error(str(router_error), 'rate_limit_error', 429)}\n\n"
                                        return
                                    raise

                        yield f"data: {create_openai_error(f'Upstream API Error: {error_text}', 'upstream_error', 429)}\n\n"
                        return

                    if parsed.get("object") == "chat.completion.chunk":
                        openai_chunk = parsed
                        openai_chunk["model"] = ctx.raw_model
                    else:
                        openai_chunk = self._openai_chunk_from_qwen(parsed, ctx.raw_model)
                    yield f"data: {json.dumps(sanitize_data(openai_chunk), ensure_ascii=False)}\n\n"

                    usage = parsed.get("usage") or openai_chunk.get("usage")
                    if usage:
                        usage_accumulated["prompt_tokens"] = usage.get("prompt_tokens", 0)
                        usage_accumulated["completion_tokens"] = usage.get("completion_tokens", 0)
                        usage_accumulated["total_tokens"] = usage.get("total_tokens", 0)
                        if usage.get("completion_tokens_details"):
                            usage_accumulated["completion_tokens_details"] = usage.get("completion_tokens_details")
                    continue

                chunk_data = parse_cloud_code_sse_line(line)
                if not chunk_data:
                    continue
                if chunk_data.get("done"):
                    break

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

            if ctx.include_usage:
                yield build_usage_stream_chunk(ctx.raw_model, usage_accumulated)

            if not ctx.is_qwen_quota_mode:
                quota_account_router.register_success(provider_id, selected_account.account.name, group_id=group_id)

            yield "data: [DONE]\n\n"

        except Exception as exc:
            error_kind = classify_429_exception(exc)
            if error_kind in {Quota429Type.RATE_LIMIT, Quota429Type.QUOTA_EXHAUSTED}:
                event = (
                    RotationEvent.RATE_LIMIT
                    if error_kind == Quota429Type.RATE_LIMIT
                    else RotationEvent.QUOTA_EXHAUSTED
                )
                event_result = quota_account_router.register_event(
                    provider=provider_id,
                    account_name=selected_account.account.name,
                    mode=selected_account.mode,
                    pool=selected_account.pool,
                    event=event,
                    model=ctx.target_model,
                    group_id=group_id,
                )
                if event_result.all_exhausted:
                    yield f"data: {create_openai_error('all_accounts_exceed_quota', 'quota_exhausted', 429)}\n\n"
                    return
                if event_result.all_cooldown:
                    wait_seconds = quota_account_router.cooldown_wait_seconds(
                        provider_id,
                        selected_account.pool,
                        group_id=group_id,
                    )
                    yield f"data: {create_openai_error(f'all accounts on cooldown please wait {wait_seconds}', 'rate_limit_error', 429)}\n\n"
                    return

            logger.error(f"[ERROR] Stream Exception: {exc}")
            yield f"data: {create_openai_error(str(exc), 'stream_exception')}\n\n"
