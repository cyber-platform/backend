from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from llm_agent_platform.services.http_pool import get_http_client

logger = logging.getLogger(__name__)
from llm_agent_platform.config import (
    QWEN_DEFAULT_RESOURCE_URL,
    QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_DEVICE_CODE_ENDPOINT,
    QWEN_OAUTH_SCOPE,
    QWEN_OAUTH_TOKEN_ENDPOINT,
    USER_QWEN_CREDS_PATH,
)


class QwenOAuthError(RuntimeError):
    """Qwen OAuth flow runtime error."""


def _require_qwen_client_id() -> str:
    client_id = (QWEN_OAUTH_CLIENT_ID or "").strip()
    if client_id:
        return client_id
    raise QwenOAuthError(
        "QWEN_OAUTH_CLIENT_ID is not set. Configure it via environment before running Qwen OAuth flows."
    )


def _body_preview(text: str, limit: int = 400) -> str:
    compact = (text or "").replace("\n", "\\n").replace("\r", "")
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _parse_json_dict(response, context: str) -> dict:
    """Parse JSON response and provide actionable diagnostics on failure."""
    content_type = response.headers.get("content-type", "<missing>")
    try:
        data = response.json()
    except Exception as exc:
        raise QwenOAuthError(
            f"{context}: expected JSON but got non-JSON response "
            f"(status={response.status_code}, content-type={content_type}, "
            f"body_preview='{_body_preview(response.text)}')."
        ) from exc

    if not isinstance(data, dict):
        raise QwenOAuthError(
            f"{context}: expected JSON object but got {type(data).__name__} "
            f"(status={response.status_code}, content-type={content_type})."
        )
    return data


def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE verifier/challenge pair.

    Returns:
        tuple[str, str]: code_verifier, code_challenge.
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


def request_device_authorization(code_challenge: str) -> dict:
    """Request device authorization from Qwen OAuth server.

    Args:
        code_challenge: PKCE code challenge.

    Returns:
        Device authorization response dict.

    Raises:
        QwenOAuthError: On HTTP error or missing device_code.
    """
    logger.info(f"Requesting device authorization from {QWEN_OAUTH_DEVICE_CODE_ENDPOINT}")
    client = get_http_client()
    client_id = _require_qwen_client_id()
    payload = {
        "client_id": client_id,
        "scope": QWEN_OAUTH_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    request_id = str(uuid4())
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": "qwen-code",
        "x-request-id": request_id,
    }

    try:
        response = client.post(
            QWEN_OAUTH_DEVICE_CODE_ENDPOINT,
            headers=headers,
            content=urlencode(payload),
        )
    except Exception as exc:
        logger.error(f"HTTP error during device authorization: {type(exc).__name__}: {exc}")
        raise QwenOAuthError(f"Device authorization HTTP error: {type(exc).__name__}: {exc}") from exc

    if response.status_code != 200:
        content_type = response.headers.get("content-type", "<missing>")
        error_msg = (
            f"Device authorization failed: "
            f"status={response.status_code}, content-type={content_type}, "
            f"body_preview='{_body_preview(response.text)}'"
        )
        logger.error(error_msg)
        raise QwenOAuthError(error_msg)

    data = _parse_json_dict(response, "Device authorization response parse failed")
    if "device_code" not in data:
        logger.error(f"Device authorization response missing device_code: {data}")
        raise QwenOAuthError(f"Device authorization error: {data}")

    logger.info(f"Device authorization successful, device_code={data['device_code'][:20]}...")
    return data


def poll_device_token(device_code: str, code_verifier: str, timeout_seconds: int = 600) -> dict:
    """Poll token endpoint until authorization completes or times out.

    Args:
        device_code: Device code from authorization request.
        code_verifier: PKCE code verifier.
        timeout_seconds: Maximum time to wait for authorization.

    Returns:
        Token response dict with access_token.

    Raises:
        QwenOAuthError: On timeout or unexpected error.
    """
    client = get_http_client()
    client_id = _require_qwen_client_id()
    started = time.time()
    interval = 2.0
    attempt = 0

    logger.info(f"Starting token polling, timeout={timeout_seconds}s")

    while True:
        attempt += 1
        elapsed = time.time() - started

        if elapsed > timeout_seconds:
            logger.error(f"Polling timed out after {elapsed:.1f}s ({attempt} attempts)")
            raise QwenOAuthError(f"Qwen OAuth device flow timed out after {timeout_seconds}s")

        payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": client_id,
            "device_code": device_code,
            "code_verifier": code_verifier,
        }
        request_id = str(uuid4())
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "qwen-code",
            "x-request-id": request_id,
        }

        logger.debug(f"Poll attempt #{attempt}, elapsed={elapsed:.1f}s, interval={interval:.1f}s")

        try:
            response = client.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                headers=headers,
                content=urlencode(payload),
            )
        except Exception as exc:
            # Важно: логируем любые HTTP ошибки для диагностики
            logger.error(f"HTTP error on attempt #{attempt}: {type(exc).__name__}: {exc}")
            raise QwenOAuthError(f"HTTP request failed: {type(exc).__name__}: {exc}") from exc

        if response.status_code == 200:
            data = _parse_json_dict(response, "Device token response parse failed")
            if data.get("access_token"):
                logger.info(f"Authorization successful after {elapsed:.1f}s ({attempt} attempts)")
                return data
            raise QwenOAuthError(f"Token response missing access_token: {data}")

        data = {}
        try:
            data = response.json()
        except Exception:
            pass

        if response.status_code == 400 and data.get("error") == "authorization_pending":
            logger.debug(f"Authorization pending (attempt #{attempt})")
            time.sleep(interval)
            continue

        if response.status_code == 429 and data.get("error") == "slow_down":
            interval = min(interval * 1.5, 10.0)
            logger.warning(f"Rate limited, increasing interval to {interval:.1f}s")
            time.sleep(interval)
            continue

        content_type = response.headers.get("content-type", "<missing>")
        error_msg = (
            f"Device token polling failed: "
            f"status={response.status_code}, error={data.get('error', 'unknown')}, "
            f"content-type={content_type}, body='{_body_preview(response.text)}'"
        )
        logger.error(error_msg)
        raise QwenOAuthError(error_msg)


def refresh_access_token(refresh_token: str, client_id: str | None = None) -> dict:
    client = get_http_client()
    effective_client_id = (client_id or "").strip()
    if not effective_client_id:
        raise QwenOAuthError(
            "Qwen token refresh requires client_id from credentials file. "
            "Re-run scripts/get_qwen-code_credentials.py to regenerate credentials."
        )
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": effective_client_id,
    }
    request_id = str(uuid4())
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": "qwen-code",
        "x-request-id": request_id,
    }
    response = client.post(
        QWEN_OAUTH_TOKEN_ENDPOINT,
        headers=headers,
        content=urlencode(payload),
    )
    if response.status_code != 200:
        content_type = response.headers.get("content-type", "<missing>")
        raise QwenOAuthError(
            "Token refresh failed: "
            f"status={response.status_code}, content-type={content_type}, "
            f"body_preview='{_body_preview(response.text)}'"
        )

    data = _parse_json_dict(response, "Token refresh response parse failed")
    if "access_token" not in data:
        raise QwenOAuthError(f"Refresh response missing access_token: {data}")
    return data


def normalize_qwen_credentials(token_data: dict, *, fallback_client_id: str | None = None) -> dict:
    expires_in = int(token_data.get("expires_in", 3600))
    resolved_client_id = (token_data.get("client_id") or fallback_client_id or "").strip()
    if not resolved_client_id:
        raise QwenOAuthError(
            "Qwen OAuth credentials missing client_id. "
            "Provide client_id in token response or explicit fallback_client_id."
        )
    return {
        "client_id": resolved_client_id,
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_type": token_data.get("token_type", "Bearer"),
        "resource_url": token_data.get("resource_url", QWEN_DEFAULT_RESOURCE_URL),
        "expiry_date": int(time.time() * 1000) + expires_in * 1000,
    }


def read_qwen_credentials(path: str | Path = USER_QWEN_CREDS_PATH) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise QwenOAuthError(f"Qwen credentials file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "refresh_token" not in data:
        raise QwenOAuthError(f"Missing refresh_token in {file_path}")
    return data


def write_qwen_credentials(data: dict, path: str | Path = USER_QWEN_CREDS_PATH) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return file_path


def refresh_qwen_credentials_file(path: str | Path = USER_QWEN_CREDS_PATH) -> dict:
    creds = read_qwen_credentials(path)
    credentials_client_id = (creds.get("client_id") or "").strip()
    if not credentials_client_id:
        raise QwenOAuthError(
            f"Missing client_id in {Path(path)}. "
            "Re-run scripts/get_qwen-code_credentials.py to regenerate credentials."
        )

    refreshed = refresh_access_token(
        creds["refresh_token"],
        client_id=credentials_client_id,
    )
    normalized = normalize_qwen_credentials(
        {
            **creds,
            **refreshed,
            "client_id": refreshed.get("client_id") or credentials_client_id,
            "refresh_token": refreshed.get("refresh_token") or creds.get("refresh_token"),
            "resource_url": refreshed.get("resource_url") or creds.get("resource_url"),
        }
    )
    write_qwen_credentials(normalized, path)
    return normalized
