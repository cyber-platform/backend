from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from llm_agent_platform.config import (
    OPENAI_CHATGPT_OAUTH_AUTHORIZE_ENDPOINT,
    OPENAI_CHATGPT_OAUTH_CALLBACK_HOST,
    OPENAI_CHATGPT_OAUTH_CALLBACK_PATH,
    OPENAI_CHATGPT_OAUTH_CALLBACK_PORT,
    OPENAI_CHATGPT_OAUTH_CLIENT_ID,
    OPENAI_CHATGPT_OAUTH_SCOPE,
    OPENAI_CHATGPT_OAUTH_TOKEN_ENDPOINT,
    OPENAI_CHATGPT_ORIGINATOR,
    OPENAI_CHATGPT_REFRESH_BUFFER_SECONDS,
    OPENAI_CHATGPT_USER_AGENT,
    USER_OPENAI_CHATGPT_CREDS_PATH,
)
from llm_agent_platform.services.http_pool import get_http_client


class OpenAIChatGPTOAuthError(RuntimeError):
    """Raised when OpenAI ChatGPT OAuth operations fail."""


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_iso8601(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    return datetime.fromisoformat(candidate)


def _require_client_id(candidate: str | None = None) -> str:
    client_id = (candidate or OPENAI_CHATGPT_OAUTH_CLIENT_ID or "").strip()
    if client_id:
        return client_id
    raise OpenAIChatGPTOAuthError(
        "OPENAI_CHATGPT_OAUTH_CLIENT_ID is not set. Configure it before running the OAuth bootstrap or refresh flow."
    )


def _decode_token_claims(token: str | None) -> dict:
    if not token or token.count(".") < 2:
        return {}
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(f"{payload_segment}{padding}")
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_account_id(token_data: dict) -> str | None:
    """Extract ChatGPT account id from token claims on a best-effort basis.

    Args:
        token_data: OAuth token payload or persisted state.

    Returns:
        Optional account identifier for `ChatGPT-Account-Id` propagation.
    """

    for token_name in ("id_token", "access_token"):
        claims = _decode_token_claims(token_data.get(token_name))
        nested_auth = claims.get("https://api.openai.com/auth")
        nested_account_id = None
        if isinstance(nested_auth, dict):
            candidate = nested_auth.get("chatgpt_account_id")
            if isinstance(candidate, str) and candidate.strip():
                nested_account_id = candidate.strip()
        for candidate in (
            claims.get("chatgpt_account_id"),
            claims.get("https://api.openai.com/auth.chatgpt_account_id"),
            nested_account_id,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict):
                candidate = first.get("id")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return None


def generate_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")


def generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def build_redirect_uri(
    host: str = OPENAI_CHATGPT_OAUTH_CALLBACK_HOST,
    port: int = OPENAI_CHATGPT_OAUTH_CALLBACK_PORT,
    path: str = OPENAI_CHATGPT_OAUTH_CALLBACK_PATH,
) -> str:
    callback_path = path if path.startswith("/") else f"/{path}"
    return f"http://{host}:{port}{callback_path}"


def build_authorization_url(
    *,
    code_challenge: str,
    state: str,
    redirect_uri: str | None = None,
    client_id: str | None = None,
) -> str:
    params = {
        "response_type": "code",
        "client_id": _require_client_id(client_id),
        "redirect_uri": redirect_uri or build_redirect_uri(),
        "scope": OPENAI_CHATGPT_OAUTH_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "originator": OPENAI_CHATGPT_ORIGINATOR,
        "codex_cli_simplified_flow": "true",
    }
    return f"{OPENAI_CHATGPT_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(params)}"


def _token_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": OPENAI_CHATGPT_USER_AGENT,
        "x-request-id": str(uuid4()),
    }


def _parse_token_response(response, context: str) -> dict:
    content_type = response.headers.get("content-type", "<missing>")
    try:
        payload = response.json()
    except Exception as exc:
        raise OpenAIChatGPTOAuthError(
            f"{context}: expected JSON response (status={response.status_code}, content-type={content_type})."
        ) from exc
    if not isinstance(payload, dict):
        raise OpenAIChatGPTOAuthError(
            f"{context}: expected JSON object but got {type(payload).__name__}."
        )
    return payload


def exchange_code_for_tokens(
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client_id: str | None = None,
) -> dict:
    payload = {
        "grant_type": "authorization_code",
        "client_id": _require_client_id(client_id),
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    response = get_http_client().post(
        OPENAI_CHATGPT_OAUTH_TOKEN_ENDPOINT,
        headers=_token_headers(),
        content=urlencode(payload),
    )
    if response.status_code != 200:
        body = _parse_token_response(response, "Authorization code exchange failed")
        raise OpenAIChatGPTOAuthError(
            f"Authorization code exchange failed: status={response.status_code}, error={body.get('error', 'unknown')}"
        )
    return _parse_token_response(response, "Authorization code exchange failed")


def refresh_access_token(refresh_token: str, *, client_id: str | None = None) -> dict:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": _require_client_id(client_id),
    }
    response = get_http_client().post(
        OPENAI_CHATGPT_OAUTH_TOKEN_ENDPOINT,
        headers=_token_headers(),
        content=urlencode(payload),
    )
    if response.status_code != 200:
        body = _parse_token_response(response, "Token refresh failed")
        raise OpenAIChatGPTOAuthError(
            f"Token refresh failed: status={response.status_code}, error={body.get('error', 'unknown')}"
        )
    return _parse_token_response(response, "Token refresh failed")


def normalize_token_payload(token_data: dict, *, previous_state: dict | None = None) -> dict:
    expires_in = int(token_data.get("expires_in", 3600))
    now = _utc_now()
    account_id = extract_account_id(token_data) or (previous_state or {}).get("account_id")
    refresh_token = str(token_data.get("refresh_token") or (previous_state or {}).get("refresh_token") or "").strip()
    if not refresh_token:
        raise OpenAIChatGPTOAuthError(
            "OpenAI ChatGPT OAuth token payload does not contain refresh_token. "
            "Strict runtime requires refreshable OAuth state; re-run the bootstrap flow and persist a refresh-capable account file."
        )
    scopes_raw = token_data.get("scope") or token_data.get("scopes") or OPENAI_CHATGPT_OAUTH_SCOPE
    if isinstance(scopes_raw, str):
        scopes = [item for item in scopes_raw.split() if item]
    elif isinstance(scopes_raw, list):
        scopes = [str(item) for item in scopes_raw if str(item).strip()]
    else:
        scopes = []

    normalized = {
        "version": 1,
        "provider_id": "openai-chatgpt",
        "client_id": _require_client_id(token_data.get("client_id") or (previous_state or {}).get("client_id")),
        "access_token": token_data.get("access_token"),
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "account_id": account_id,
        "expires_at": _to_iso8601(now + timedelta(seconds=expires_in)),
        "obtained_at": _to_iso8601(now),
        "scopes": scopes,
        "metadata": {
            "originator": OPENAI_CHATGPT_ORIGINATOR,
            "user_agent": OPENAI_CHATGPT_USER_AGENT,
        },
    }

    id_token = token_data.get("id_token") or (previous_state or {}).get("id_token")
    if id_token:
        normalized["id_token"] = id_token
    return normalized


def read_openai_chatgpt_oauth_state(path: str | Path = USER_OPENAI_CHATGPT_CREDS_PATH) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise OpenAIChatGPTOAuthError(f"OAuth state file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise OpenAIChatGPTOAuthError(f"OAuth state must be a JSON object: {file_path}")
    return payload


def write_openai_chatgpt_oauth_state(payload: dict, path: str | Path = USER_OPENAI_CHATGPT_CREDS_PATH) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return file_path


@dataclass(slots=True)
class OpenAIChatGPTOAuthManager:
    credentials_path: str | Path = USER_OPENAI_CHATGPT_CREDS_PATH
    refresh_lock: threading.Lock = field(default_factory=threading.Lock)

    def load_state(self) -> dict:
        state = read_openai_chatgpt_oauth_state(self.credentials_path)
        refresh_token = str(state.get("refresh_token") or "").strip()
        if not refresh_token:
            raise OpenAIChatGPTOAuthError(
                f"OAuth state {self.credentials_path} does not contain refresh_token. Re-run the bootstrap flow."
            )
        changed = False
        if state.get("token_type") != "Bearer":
            state["token_type"] = "Bearer"
            changed = True
        if not state.get("account_id"):
            extracted_account_id = extract_account_id(state)
            if extracted_account_id:
                state["account_id"] = extracted_account_id
                changed = True
        if changed:
            self.save_state(state)
        return state

    def save_state(self, state: dict) -> Path:
        return write_openai_chatgpt_oauth_state(state, self.credentials_path)

    def is_expired(self, state: dict, *, buffer_seconds: int = OPENAI_CHATGPT_REFRESH_BUFFER_SECONDS) -> bool:
        expires_at = _parse_datetime(state.get("expires_at"))
        if expires_at is None:
            return True
        return expires_at <= _utc_now() + timedelta(seconds=buffer_seconds)

    def refresh_state(self, *, force: bool = False) -> dict:
        with self.refresh_lock:
            state = self.load_state()
            if not force and not self.is_expired(state):
                return state

            refresh_token = str(state.get("refresh_token") or "").strip()
            if not refresh_token:
                raise OpenAIChatGPTOAuthError(
                    f"OAuth state {self.credentials_path} does not contain refresh_token. Re-run the bootstrap flow."
                )

            refreshed = refresh_access_token(
                refresh_token,
                client_id=state.get("client_id"),
            )
            normalized = normalize_token_payload(
                {**state, **refreshed, "refresh_token": refreshed.get("refresh_token") or refresh_token},
                previous_state=state,
            )
            self.save_state(normalized)
            return normalized

    def get_valid_state(self) -> dict:
        state = self.load_state()
        if self.is_expired(state):
            return self.refresh_state(force=True)
        return state

    def force_refresh(self) -> dict:
        return self.refresh_state(force=True)
