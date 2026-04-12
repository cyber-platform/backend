from __future__ import annotations

import json

from flask import request

from llm_agent_platform.api.openai.pipeline import resolve_route_provider
from llm_agent_platform.api.openai.types import UpstreamPreparationError
from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.core.utils import sanitize_data
from llm_agent_platform.services.openai_chatgpt_api_keys import (
    PROVIDER_ID as OPENAI_CHATGPT_PROVIDER_ID,
    ApiKeyRegistryError,
    OpenAIChatGPTApiKeyRegistryService,
)

logger = get_logger(__name__)

_INVALID_API_KEY_MESSAGE = "Incorrect API key provided"


class InvalidPlatformApiKeyError(RuntimeError):
    """Raised when the public OpenAI-compatible route is unauthorized."""


def create_invalid_api_key_error() -> str:
    return json.dumps(
        sanitize_data(
            {
                "error": {
                    "message": _INVALID_API_KEY_MESSAGE,
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            }
        ),
        ensure_ascii=False,
    )


def authorize_public_route(
    provider_name: str, group_name: str | None = None
) -> tuple[object, str, tuple[str, ...] | None]:
    provider, resolved_group_id, allowed_model_ids = resolve_route_provider(
        provider_name, group_name
    )
    if provider.provider_id != OPENAI_CHATGPT_PROVIDER_ID:
        return provider, resolved_group_id, allowed_model_ids

    raw_api_key = _extract_bearer_token(request.headers.get("Authorization"))
    if raw_api_key is None:
        logger.info(
            "[AUTH] Missing or malformed platform API key for provider=%s group=%s",
            provider.provider_id,
            resolved_group_id,
        )
        raise InvalidPlatformApiKeyError(_INVALID_API_KEY_MESSAGE)

    try:
        key_record = OpenAIChatGPTApiKeyRegistryService().lookup_active_key(
            raw_api_key, group_id=resolved_group_id
        )
    except ApiKeyRegistryError as exc:
        raise UpstreamPreparationError(
            "Platform API key registry is unavailable",
            "internal_error",
            500,
        ) from exc

    if key_record is None:
        logger.info(
            "[AUTH] Rejected platform API key for provider=%s group=%s",
            provider.provider_id,
            resolved_group_id,
        )
        raise InvalidPlatformApiKeyError(_INVALID_API_KEY_MESSAGE)

    return provider, resolved_group_id, allowed_model_ids


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    if not isinstance(authorization_header, str):
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    normalized_token = token.strip()
    if not normalized_token or " " in normalized_token:
        return None
    return normalized_token
