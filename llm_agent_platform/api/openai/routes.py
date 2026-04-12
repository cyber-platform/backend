import json
from flask import Blueprint, Response, stream_with_context

from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.core.utils import create_openai_error
from llm_agent_platform.api.openai.auth_guard import (
    InvalidPlatformApiKeyError,
    authorize_public_route,
    create_invalid_api_key_error,
)
from llm_agent_platform.api.openai.pipeline import (
    build_request_context,
    resolve_provider,
    resolve_route_provider,
    resolve_strategy,
)
from llm_agent_platform.services.provider_registry import get_provider_registry
from llm_agent_platform.api.openai.types import UpstreamPreparationError

openai_bp = Blueprint("openai", __name__)
logger = get_logger(__name__)


@openai_bp.route("/<provider_name>/v1/chat/completions", methods=["POST"])
@openai_bp.route("/<provider_name>/<group_name>/v1/chat/completions", methods=["POST"])
def chat_completions(provider_name: str, group_name: str | None = None):
    try:
        authorize_public_route(provider_name, group_name)
        request_ctx = build_request_context(provider_name, group_name)
        provider = resolve_provider(request_ctx)
        strategy = resolve_strategy(request_ctx)

        if request_ctx.stream:
            return Response(
                stream_with_context(strategy.stream(request_ctx, provider)),
                mimetype="text/event-stream",
            )

        return strategy.execute_non_stream(request_ctx, provider)

    except InvalidPlatformApiKeyError:
        return create_invalid_api_key_error(), 401
    except UpstreamPreparationError as prep_error:
        return create_openai_error(
            prep_error.message, prep_error.error_type, prep_error.status_code
        ), prep_error.status_code
    except Exception as e:
        logger.exception("[ERROR] Internal Gateway Error in chat_completions")
        return create_openai_error(
            f"Internal Gateway Error: {str(e)}", "internal_error", 500
        ), 500


def _serialize_models(model_ids: list[str]) -> str:
    return json.dumps(
        {"data": [{"id": model_id, "object": "model"} for model_id in model_ids]}
    )


@openai_bp.route("/<provider_name>/v1/models", methods=["GET"])
@openai_bp.route("/<provider_name>/<group_name>/v1/models", methods=["GET"])
def list_models(provider_name: str, group_name: str | None = None):
    try:
        provider, _resolved_group_id, allowed_model_ids = authorize_public_route(
            provider_name, group_name
        )
        catalog = get_provider_registry().load_catalog(provider.provider_id)
        models = [model.model_id for model in catalog.models]
        if allowed_model_ids is not None:
            allowed = set(allowed_model_ids)
            models = [model_id for model_id in models if model_id in allowed]
        return _serialize_models(models)
    except InvalidPlatformApiKeyError:
        return create_invalid_api_key_error(), 401
    except UpstreamPreparationError as prep_error:
        return create_openai_error(
            prep_error.message, prep_error.error_type, prep_error.status_code
        ), prep_error.status_code
