from __future__ import annotations

from flask import request

from llm_agent_platform.api.openai.transform import transform_openai_to_gemini
from llm_agent_platform.api.openai.types import ChatRequestContext, UpstreamPreparationError
from llm_agent_platform.config import STRICT_CLI_PARITY
from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.core.utils import sanitize_data, sanitize_string, clean_gemini_schema
from llm_agent_platform.services.quota_transport import generate_session_id, generate_user_prompt_id
from llm_agent_platform.api.openai.providers.google_vertex import GoogleVertexProvider
from llm_agent_platform.api.openai.providers.gemini_cli import GeminiCliProvider
from llm_agent_platform.api.openai.providers.openai_chatgpt import OpenAIChatGPTProvider
from llm_agent_platform.api.openai.providers.qwen_code import QwenCodeProvider
from llm_agent_platform.api.openai.strategies.direct import DirectStrategy
from llm_agent_platform.api.openai.strategies.registry import get_strategy
from llm_agent_platform.api.openai.strategies.rotate_on_429_rounding import RotateOn429RoundingStrategy
from llm_agent_platform.services.account_router import AccountRouterError, quota_account_router
from llm_agent_platform.services.provider_registry import ProviderDescriptor, get_provider_registry

logger = get_logger(__name__)

_RUNTIME_ADAPTER_PROVIDERS = {
    GeminiCliProvider.id: GeminiCliProvider,
    GoogleVertexProvider.id: GoogleVertexProvider,
    OpenAIChatGPTProvider.id: OpenAIChatGPTProvider,
    QwenCodeProvider.id: QwenCodeProvider,
}

_LEGACY_QUOTA_RUNTIME_ADAPTERS = {
    GeminiCliProvider.id,
    QwenCodeProvider.id,
}

def _is_qwen_quota_model(raw_model: str) -> bool:
    return 'qwen' in raw_model and 'quota' in raw_model


def _is_gemini_quota_model(raw_model: str) -> bool:
    return 'quota' in raw_model and not _is_qwen_quota_model(raw_model)


def _build_gemini_config(data: dict) -> dict:
    max_completion_tokens = data.get('max_completion_tokens')
    max_tokens = max_completion_tokens if max_completion_tokens is not None else data.get('max_tokens')

    gemini_config = {
        "temperature": data.get('temperature', 0.7),
        "topP": data.get('top_p', 1.0),
        "topK": data.get('top_k', 40),
    }

    if max_tokens is not None and max_tokens != -1:
        gemini_config["maxOutputTokens"] = int(max_tokens)

    stop_sequences = data.get('stop')
    if stop_sequences:
        if isinstance(stop_sequences, str):
            gemini_config["stopSequences"] = [stop_sequences]
        elif isinstance(stop_sequences, list):
            gemini_config["stopSequences"] = stop_sequences

    reasoning_effort = data.get('reasoning_effort')
    max_thinking_tokens = data.get('modelMaxThinkingTokens') or data.get('max_completion_tokens')
    if reasoning_effort or max_thinking_tokens or data.get('enableReasoningEffort'):
        thinking_config = {
            "includeThoughts": True
        }
        if max_thinking_tokens:
            thinking_config["thinkingBudget"] = int(max_thinking_tokens)

        gemini_config["thinkingConfig"] = thinking_config

    return gemini_config


def _build_gemini_tools(data: dict) -> list:
    tools = data.get('tools', [])
    if not tools:
        return []

    declarations = []
    for tool in tools:
        if tool.get('type') == 'function':
            func = tool.get('function', {})
            declarations.append({
                "name": func.get('name'),
                "description": func.get('description'),
                "parameters": clean_gemini_schema(func.get('parameters'))
            })

    if not declarations:
        return []
    return [{"function_declarations": declarations}]


def _resolve_group_models(provider: ProviderDescriptor, group_id: str | None) -> tuple[str, tuple[str, ...] | None]:
    cfg = quota_account_router.try_load_provider_config(provider.runtime_adapter)
    if cfg is None:
        return (group_id or "g0"), None

    if not cfg.groups:
        if group_id is not None and group_id != "g0":
            raise UpstreamPreparationError(
                f"Unknown group '{group_id}' for provider '{provider.route_name}'",
                "invalid_request_error",
                400,
            )
        return "g0", None

    if group_id is None:
        default_group_id, default_group = next(iter(cfg.groups.items()))
        return default_group_id, tuple(default_group.models)

    group = cfg.groups.get(group_id)
    if group is None:
        raise UpstreamPreparationError(
            f"Unknown group '{group_id}' for provider '{provider.route_name}'",
            "invalid_request_error",
            400,
        )
    return group_id, tuple(group.models)


def resolve_route_provider(provider_name: str, group_name: str | None = None) -> tuple[ProviderDescriptor, str, tuple[str, ...] | None]:
    registry = get_provider_registry()
    try:
        provider = registry.get_provider(provider_name)
    except Exception as exc:
        raise UpstreamPreparationError(
            f"Unknown provider '{provider_name}'",
            "invalid_request_error",
            400,
        ) from exc

    resolved_group_id, allowed_model_ids = _resolve_group_models(provider, group_name)
    return provider, resolved_group_id, allowed_model_ids


def _validate_target_model(
    provider: ProviderDescriptor,
    target_model: str,
    resolved_group_id: str,
    allowed_model_ids: tuple[str, ...] | None,
) -> None:
    catalog = get_provider_registry().load_catalog(provider.provider_id)
    catalog_model_ids = {model.model_id for model in catalog.models}
    if target_model not in catalog_model_ids:
        raise UpstreamPreparationError(
            f"Model '{target_model}' is not available for provider '{provider.route_name}'",
            "invalid_request_error",
            400,
        )

    if allowed_model_ids is not None and target_model not in allowed_model_ids:
        raise UpstreamPreparationError(
            f"Model '{target_model}' is not available for group '{resolved_group_id}'",
            "invalid_request_error",
            400,
        )


def build_request_context(provider_name: str, group_name: str | None = None) -> ChatRequestContext:
    data = request.json
    if not isinstance(data, dict):
        data = {}

    provider, resolved_group_id, allowed_model_ids = resolve_route_provider(provider_name, group_name)

    raw_model = data.get('model')
    if not isinstance(raw_model, str) or not raw_model.strip():
        raise UpstreamPreparationError("Field 'model' is required", "invalid_request_error", 400)
    target_model = raw_model.strip()
    _validate_target_model(provider, target_model, resolved_group_id, allowed_model_ids)

    messages = data.get('messages', [])
    stream = data.get('stream', False)
    stream_options = data.get('stream_options', {})
    include_usage = stream_options.get('include_usage', False) if stream else False

    logger.info(
        f"[REQ] Provider: {provider.provider_id} | Group: {resolved_group_id} | "
        f"Model: {raw_model} | Stream: {stream} | Usage: {include_usage}"
    )

    contents, system_instruction = transform_openai_to_gemini(messages)
    contents = sanitize_data(contents)
    if system_instruction:
        system_instruction = sanitize_string(system_instruction)

    gemini_config = _build_gemini_config(data)
    gemini_tools = _build_gemini_tools(data)

    provider_cfg = quota_account_router.try_load_provider_config(provider.runtime_adapter)
    provider_mode = provider_cfg.mode if provider_cfg is not None else None
    uses_account_router = provider_cfg is not None
    is_qwen_quota_mode = uses_account_router and provider.runtime_adapter == QwenCodeProvider.id
    is_gemini_quota_mode = uses_account_router and provider.runtime_adapter == GeminiCliProvider.id
    is_quota_mode = uses_account_router and provider_mode == "rounding"

    session_id = data.get("session_id") or request.headers.get("x-session-id") or generate_session_id()
    user_prompt_id = data.get("user_prompt_id") or request.headers.get("x-user-prompt-id") or generate_user_prompt_id()

    return ChatRequestContext(
        provider_id=provider.provider_id,
        provider_runtime_adapter=provider.runtime_adapter,
        provider_transport_kind=provider.raw_payload.get("transport", {}).get("kind", ""),
        provider_auth_kind=provider.auth_kind,
        provider_mode=provider_mode,
        uses_account_router=uses_account_router,
        allowed_model_ids=allowed_model_ids,
        data=data,
        raw_model=raw_model,
        target_model=target_model,
        messages=messages,
        stream=stream,
        include_usage=include_usage,
        contents=contents,
        system_instruction=system_instruction,
        gemini_config=gemini_config,
        gemini_tools=gemini_tools,
        is_qwen_quota_mode=is_qwen_quota_mode,
        is_gemini_quota_mode=is_gemini_quota_mode,
        is_quota_mode=is_quota_mode,
        session_id=session_id,
        user_prompt_id=user_prompt_id,
        group_id=resolved_group_id,
    )


def resolve_provider(ctx: ChatRequestContext):
    provider_cls = _RUNTIME_ADAPTER_PROVIDERS.get(ctx.provider_runtime_adapter)
    if provider_cls is None:
        raise UpstreamPreparationError(
            f"Runtime adapter '{ctx.provider_runtime_adapter}' is not implemented for provider '{ctx.provider_id}'",
            "not_implemented_error",
            501,
        )
    return provider_cls()


def resolve_strategy(ctx: ChatRequestContext):
    if ctx.provider_runtime_adapter in _LEGACY_QUOTA_RUNTIME_ADAPTERS:
        return get_strategy(RotateOn429RoundingStrategy.id)
    if ctx.uses_account_router and ctx.provider_mode == "rounding":
        return get_strategy(RotateOn429RoundingStrategy.id)
    return get_strategy(DirectStrategy.id)


def build_fallback_chain(ctx: ChatRequestContext) -> list[str]:
    fallback_chain = [ctx.target_model]
    if not (ctx.is_quota_mode and STRICT_CLI_PARITY) and 'gemini-3-flash' in ctx.target_model:
        fallback_chain.extend(['gemini-2.5-flash', 'gemini-2.5-flash-lite'])
    elif not (ctx.is_quota_mode and STRICT_CLI_PARITY) and 'gemini-3.1-pro' in ctx.target_model:
        fallback_chain.append('gemini-2.5-pro')
    return fallback_chain
