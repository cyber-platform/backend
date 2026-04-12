from __future__ import annotations

import json
import os
import re
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from llm_agent_platform.config import CLOUD_CODE_ENDPOINT
from llm_agent_platform.services.http_pool import get_http_client

_CAPTURE_DIR = Path(os.environ.get("PARITY_CAPTURE_DIR", "artifacts/parity"))
_CAPTURE_ENABLED = os.environ.get("PARITY_CAPTURE_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class Quota429Type(str, Enum):
    """Semantic classification for HTTP 429 responses."""

    NOT_429 = "not_429"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    UNKNOWN_429 = "unknown_429"


def build_quota_payload(
    *,
    model: str,
    project: str,
    request_payload: dict[str, Any],
    user_prompt_id: str,
    session_id: str | None,
) -> dict[str, Any]:
    request_data = dict(request_payload)
    if session_id:
        request_data["session_id"] = session_id

    return {
        "model": model,
        "project": project,
        "user_prompt_id": user_prompt_id,
        "request": request_data,
    }


def generate_user_prompt_id() -> str:
    return str(uuid.uuid4())


def generate_session_id() -> str:
    return str(uuid.uuid4())


def quota_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def quota_url(stream: bool) -> str:
    return f"{CLOUD_CODE_ENDPOINT}:{'streamGenerateContent' if stream else 'generateContent'}"


def send_generate(token: str, payload: dict[str, Any]):
    return send_generate_to_url(
        token,
        payload,
        quota_url(stream=False),
    )


def send_generate_to_url(
    token: str,
    payload: dict[str, Any],
    url: str,
    *,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
):
    client = get_http_client()
    headers = quota_headers(token)
    if extra_headers:
        headers.update(extra_headers)

    _capture("proxy_request_generate", payload)
    response = client.post(
        url,
        headers=headers,
        json=payload,
        params=params,
    )
    _capture_response("proxy_response_generate", response.status_code, response.text)
    return response


def stream_generate_lines(token: str, payload: dict[str, Any]) -> Iterable[str]:
    return stream_generate_lines_from_url(
        token,
        payload,
        quota_url(stream=True),
        params={"alt": "sse"},
    )


def stream_generate_lines_from_url(
    token: str,
    payload: dict[str, Any],
    url: str,
    *,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Iterable[str]:
    client = get_http_client()
    headers = quota_headers(token)
    if extra_headers:
        headers.update(extra_headers)

    _capture("proxy_request_stream", payload)
    with client.stream(
        "POST",
        url,
        headers=headers,
        json=payload,
        params=params,
    ) as response:
        if response.status_code != 200:
            body = response.read().decode(errors="replace")
            _capture_response("proxy_response_stream_error", response.status_code, body)
            raise RuntimeError(f"{response.status_code}:{body}")

        for line in response.iter_lines():
            if not line:
                continue
            text_line = line.decode() if isinstance(line, bytes) else line
            _capture_line("proxy_response_stream_line", text_line)
            yield text_line


def unwrap_cloud_code_response(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("response", data)


def parse_cloud_code_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None

    raw = line[6:].strip()
    if raw == "[DONE]":
        return {"done": True}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return unwrap_cloud_code_response(parsed)


_RATE_LIMIT_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"too many requests",
        r"rate[_ ]?limit",
        r"slow[_ ]?down",
        r"retry[_ -]?after",
        r"requests? per (minute|second)",
        r"rpm",
    ]
]

_QUOTA_EXHAUSTED_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"insufficient[_ ]?quota",
        r"quota",
        r"model_capacity_exhausted",
        r"resource[_ ]?exhausted",
        r"capacity",
        r"daily[_ ]?limit",
        r"usage[_ ]?limit",
        r"quota[_ ]?exceeded",
        r"exceeded.*quota",
        r"insufficient[_ ]?quota",
    ]
]


def _contains_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_429_response(status_code: int, response_text: str) -> Quota429Type:
    """Classify HTTP response by semantic 429 category."""
    if status_code != 429:
        return Quota429Type.NOT_429

    body = response_text or ""
    if _contains_any(body, _RATE_LIMIT_PATTERNS):
        return Quota429Type.RATE_LIMIT
    if _contains_any(body, _QUOTA_EXHAUSTED_PATTERNS):
        return Quota429Type.QUOTA_EXHAUSTED
    return Quota429Type.UNKNOWN_429


def classify_429_error_payload(error_payload: Any) -> Quota429Type:
    """Classify SSE/JSON error payloads that may contain 429 semantics."""
    if error_payload is None:
        return Quota429Type.UNKNOWN_429

    status_code: int | None = None
    body_text: str
    if isinstance(error_payload, dict):
        raw_status = error_payload.get("code") or error_payload.get("status")
        if isinstance(raw_status, int):
            status_code = raw_status
        elif isinstance(raw_status, str) and raw_status.isdigit():
            status_code = int(raw_status)
        body_text = json.dumps(error_payload, ensure_ascii=False)
    else:
        body_text = str(error_payload)

    if status_code is not None and status_code != 429:
        return Quota429Type.NOT_429

    return classify_429_response(429, body_text)


def parse_stream_exception(error: Exception) -> tuple[int | None, str]:
    """Extract status/body from stream exceptions like '429:...'."""
    err_text = str(error)
    status_code: int | None = None
    body = err_text

    if ":" in err_text:
        prefix, rest = err_text.split(":", 1)
        if prefix.isdigit():
            status_code = int(prefix)
            body = rest

    return status_code, body


def classify_429_exception(error: Exception) -> Quota429Type:
    """Classify stream exception into semantic 429 category."""
    status_code, body = parse_stream_exception(error)
    if status_code is None:
        return Quota429Type.NOT_429
    return classify_429_response(status_code, body)


def is_quota_limit_response(status_code: int, response_text: str) -> bool:
    """Detect quota/capacity limit errors for account rotation logic."""
    kind = classify_429_response(status_code, response_text)
    return kind in {Quota429Type.RATE_LIMIT, Quota429Type.QUOTA_EXHAUSTED}


def _capture(kind: str, payload: dict[str, Any]) -> None:
    if not _CAPTURE_ENABLED:
        return

    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    file = _CAPTURE_DIR / f"{int(time.time() * 1000)}_{kind}.json"
    with file.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)


def _capture_response(kind: str, status: int, body: str) -> None:
    if not _CAPTURE_ENABLED:
        return

    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    file = _CAPTURE_DIR / f"{int(time.time() * 1000)}_{kind}.json"
    payload = {
        "status": status,
        "body": body,
    }
    with file.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)


def _capture_line(kind: str, line: str) -> None:
    if not _CAPTURE_ENABLED:
        return

    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    file = _CAPTURE_DIR / f"{int(time.time() * 1000)}_{kind}.txt"
    with file.open("w", encoding="utf-8") as f:
        f.write(line)


def _sanitize(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower in {"authorization", "token", "access_token", "refresh_token", "client_secret"}:
                out[key] = "***redacted***"
            else:
                out[key] = _sanitize(value)
        return out

    if isinstance(data, list):
        return [_sanitize(v) for v in data]

    return data
