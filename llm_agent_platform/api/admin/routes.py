from flask import Blueprint, jsonify, request

from llm_agent_platform.services.openai_chatgpt_api_keys import (
    ApiKeyNotFoundError,
    ApiKeyRegistryError,
    InvalidGroupError,
    OpenAIChatGPTApiKeyRegistryService,
)
from llm_agent_platform.services.openai_chatgpt_admin_monitoring import (
    MonitoringReadModelError,
    OpenAIChatGPTAdminMonitoringService,
    RefreshRunNotFoundError,
)
from llm_agent_platform.services.account_router import AccountRouterError

admin_bp = Blueprint("admin", __name__)


def _registry_service() -> OpenAIChatGPTApiKeyRegistryService:
    return OpenAIChatGPTApiKeyRegistryService()


def _monitoring_service() -> OpenAIChatGPTAdminMonitoringService:
    return OpenAIChatGPTAdminMonitoringService()


@admin_bp.route("/admin/api-keys/openai-chatgpt", methods=["GET"])
def list_openai_chatgpt_api_keys():
    group_id = request.args.get("group_id", "")
    try:
        return jsonify(_registry_service().list_keys(group_id))
    except InvalidGroupError as exc:
        return jsonify({"error": str(exc)}), 400
    except ApiKeyRegistryError as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/admin/api-keys/openai-chatgpt", methods=["POST"])
def create_openai_chatgpt_api_key():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        return jsonify(
            _registry_service().create_key(
                group_id=str(payload.get("group_id", "")),
                label=str(payload.get("label", "")),
            )
        )
    except InvalidGroupError as exc:
        return jsonify({"error": str(exc)}), 400
    except ApiKeyRegistryError as exc:
        return jsonify({"error": str(exc)}), 400


@admin_bp.route("/admin/api-keys/openai-chatgpt/<key_id>/revoke", methods=["POST"])
def revoke_openai_chatgpt_api_key(key_id: str):
    try:
        return jsonify(_registry_service().revoke_key(key_id))
    except ApiKeyNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ApiKeyRegistryError as exc:
        return jsonify({"error": str(exc)}), 400


@admin_bp.route("/admin/monitoring/providers", methods=["GET"])
def list_monitoring_providers():
    try:
        return jsonify(_monitoring_service().list_providers())
    except MonitoringReadModelError as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/admin/monitoring/openai-chatgpt", methods=["GET"])
def get_openai_chatgpt_monitoring_page():
    try:
        return jsonify(_monitoring_service().get_provider_page())
    except MonitoringReadModelError as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/admin/monitoring/openai-chatgpt/refresh", methods=["POST"])
def start_openai_chatgpt_monitoring_refresh():
    try:
        return jsonify(_monitoring_service().start_refresh()), 202
    except MonitoringReadModelError as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route(
    "/admin/monitoring/openai-chatgpt/refresh/<refresh_id>", methods=["GET"]
)
def get_openai_chatgpt_monitoring_refresh_status(refresh_id: str):
    try:
        return jsonify(_monitoring_service().get_refresh_status(refresh_id))
    except RefreshRunNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except MonitoringReadModelError as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route(
    "/admin/monitoring/openai-chatgpt/groups/<group_id>/accounts/<account_name>/activate",
    methods=["POST"],
)
def activate_openai_chatgpt_account(group_id: str, account_name: str):
    try:
        return jsonify(_monitoring_service().activate_account(group_id, account_name))
    except AccountRouterError as exc:
        return jsonify({"error": str(exc)}), 404
    except MonitoringReadModelError as exc:
        return jsonify({"error": str(exc)}), 500
