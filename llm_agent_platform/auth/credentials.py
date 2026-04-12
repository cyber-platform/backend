import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials

from llm_agent_platform.config import (
    GEMINI_ACCOUNTS_CONFIG_PATH,
    GEMINI_CLI_CLIENT_ID,
    GEMINI_CLI_CLIENT_SECRET,
    OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH,
    QWEN_ACCOUNTS_CONFIG_PATH,
    SERVICE_ACCOUNT_PATH,
)
from llm_agent_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class AuthAvailability:
    gemini_quota: bool
    qwen_quota: bool
    openai_chatgpt: bool
    vertex: bool
    diagnostics: list[str]

    def has_any(self) -> bool:
        return self.gemini_quota or self.qwen_quota or self.openai_chatgpt or self.vertex


_auth_state = {
    "auth_lock": threading.Lock(),
}


def get_auth_lock() -> threading.Lock:
    """Return global auth lock for quota token access.

    Returns:
        threading.Lock: shared lock.
    """
    return _auth_state["auth_lock"]


def _load_gemini_refresh_token(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    if "refresh_token" not in info:
        raise ValueError(f"Missing 'refresh_token' in {file_path}")
    return info


def _load_json_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {file_path}")
    return payload


def _credentials_has_refresh_token(credentials_path: str | Path) -> bool:
    file_path = Path(credentials_path)
    if not file_path.exists():
        return False
    try:
        info = _load_json_file(file_path)
    except Exception:
        return False
    return bool(info.get("refresh_token"))


def _provider_has_valid_account_data(config_path: str | Path, provider: str) -> bool:
    file_path = Path(config_path)
    if not file_path.exists():
        return False

    try:
        payload = _load_json_file(file_path)
    except Exception:
        return False

    mode = payload.get("mode", "single")
    if mode not in {"single", "rounding"}:
        return False

    accounts = payload.get("accounts")
    if not isinstance(accounts, dict) or not accounts:
        return False

    candidate_names: list[str] = []
    if mode == "single":
        active_account = payload.get("active_account")
        if not active_account:
            return False
        candidate_names = [active_account]
    else:
        all_accounts = payload.get("all_accounts") or []
        if not isinstance(all_accounts, list) or not all_accounts:
            return False
        candidate_names = [name for name in all_accounts if isinstance(name, str) and name]
        if not candidate_names:
            return False

    for name in candidate_names:
        account = accounts.get(name)
        if not isinstance(account, dict):
            continue
        credentials_path = account.get("credentials_path")
        if not credentials_path or not _credentials_has_refresh_token(credentials_path):
            continue
        if provider == "gemini" and not account.get("project_id"):
            continue
        return True

    return False


def get_auth_availability() -> AuthAvailability:
    """Inspect local auth data and report available auth sources.

    Returns:
        AuthAvailability: availability flags and diagnostics.
    """
    diagnostics: list[str] = []

    gemini_quota = _provider_has_valid_account_data(GEMINI_ACCOUNTS_CONFIG_PATH, provider="gemini")
    if not gemini_quota:
        diagnostics.append(
            "Gemini quota auth unavailable: provide valid accounts config "
            f"with refresh_token and project_id in '{GEMINI_ACCOUNTS_CONFIG_PATH}'."
        )

    qwen_quota = _provider_has_valid_account_data(QWEN_ACCOUNTS_CONFIG_PATH, provider="qwen")
    if not qwen_quota:
        diagnostics.append(
            "Qwen quota auth unavailable: provide valid accounts config "
            f"with refresh_token in '{QWEN_ACCOUNTS_CONFIG_PATH}'."
        )

    openai_chatgpt = _provider_has_valid_account_data(
        OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH,
        provider="openai-chatgpt",
    )
    if not openai_chatgpt:
        diagnostics.append(
            "OpenAI ChatGPT auth unavailable: provide valid accounts config "
            f"with refresh_token in '{OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH}'."
        )

    vertex_project_id = os.environ.get("VERTEX_PROJECT_ID", "").strip()
    vertex_service_account = Path(SERVICE_ACCOUNT_PATH)
    vertex = bool(vertex_project_id) and vertex_service_account.exists()
    if not vertex:
        diagnostics.append(
            "Vertex auth unavailable: set VERTEX_PROJECT_ID and provide service account file "
            f"at '{SERVICE_ACCOUNT_PATH}'."
        )

    return AuthAvailability(
        gemini_quota=gemini_quota,
        qwen_quota=qwen_quota,
        openai_chatgpt=openai_chatgpt,
        vertex=vertex,
        diagnostics=diagnostics,
    )


def _refresh_gemini_token(info: dict) -> str:
    client_id = (info.get("client_id") or GEMINI_CLI_CLIENT_ID or "").strip()
    client_secret = (info.get("client_secret") or GEMINI_CLI_CLIENT_SECRET or "").strip()

    if not client_id:
        raise ValueError(
            "Gemini OAuth client_id is missing. Set GEMINI_CLI_CLIENT_ID in env "
            "or provide client_id in credentials JSON."
        )
    if not client_secret:
        raise ValueError(
            "Gemini OAuth client_secret is missing. Set GEMINI_CLI_CLIENT_SECRET in env "
            "or provide client_secret in credentials JSON."
        )

    creds = Credentials(
        token=None,
        refresh_token=info["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )
    creds.refresh(Request())

    if not creds.token:
        raise RuntimeError("Google refresh returned empty access token")
    return creds.token


def get_gemini_access_token_from_file(credentials_path: str | Path) -> str:
    """Load and refresh Gemini OAuth token from credentials file.

    Args:
        credentials_path: Path to OAuth authorized_user credentials JSON.

    Returns:
        str: Fresh Gemini access token.
    """
    file_path = Path(credentials_path)
    info = _load_gemini_refresh_token(file_path)
    return _refresh_gemini_token(info)


def initialize_auth() -> bool:
    """Validate that at least one auth source is available.

    Returns:
        bool: True when any auth source is available.
    """
    availability = get_auth_availability()
    if not availability.has_any():
        logger.warning("[AUTH] No authentication data found. Configure at least one source:")
        for message in availability.diagnostics:
            logger.warning(f"[AUTH] - {message}")
        return False

    if availability.gemini_quota:
        logger.info("[AUTH] Gemini quota auth detected.")
        if not GEMINI_CLI_CLIENT_ID:
            logger.warning(
                "[AUTH] - Warning: GEMINI_CLI_CLIENT_ID is not set in env. "
                "Token refresh will rely on client_id from credentials files."
            )
        if not GEMINI_CLI_CLIENT_SECRET:
            logger.warning(
                "[AUTH] - Warning: GEMINI_CLI_CLIENT_SECRET is not set in env. "
                "Token refresh will rely on client_secret from credentials files."
            )
    if availability.qwen_quota:
        logger.info("[AUTH] Qwen quota auth detected.")
        logger.info("[AUTH] - Qwen runtime refresh uses client_id from credentials file.")
    if availability.openai_chatgpt:
        logger.info("[AUTH] OpenAI ChatGPT auth detected.")
        logger.info("[AUTH] - OpenAI ChatGPT runtime refresh uses client_id from credentials file.")
    if availability.vertex:
        logger.info("[AUTH] Vertex auth detected.")
    return True


def get_vertex_token() -> str | None:
    """Gets an access token for Vertex AI using Service Account.

    Returns:
        str | None: Vertex access token.
    """
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        return None

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds.token
