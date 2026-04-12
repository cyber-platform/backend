from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, request, stream_with_context

from llm_agent_platform.services.http_pool import get_http_client

parity_bp = Blueprint("parity", __name__)

_RELAY_TARGET_ENDPOINT = os.environ.get(
    "PARITY_RELAY_TARGET_ENDPOINT",
    "https://cloudcode-pa.googleapis.com",
).rstrip("/")
_CAPTURE_DIR = Path(os.environ.get("PARITY_CAPTURE_DIR", "artifacts/parity"))
_CAPTURE_ENABLED = os.environ.get("PARITY_CAPTURE_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _sanitize(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if key.lower() in {
                "authorization",
                "token",
                "access_token",
                "refresh_token",
                "client_secret",
            }:
                out[key] = "***redacted***"
            else:
                out[key] = _sanitize(value)
        return out

    if isinstance(data, list):
        return [_sanitize(v) for v in data]

    return data


def _capture(name: str, payload: Any) -> None:
    if not _CAPTURE_ENABLED:
        return

    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CAPTURE_DIR / f"{int(time.time() * 1000)}_{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)


@parity_bp.route("/parity-relay/<path:method_path>", methods=["POST"])
def cloud_code_parity_relay(method_path: str):
    """Capture-and-relay endpoint for gemini-cli CODE_ASSIST_ENDPOINT parity runs."""
    target_url = f"{_RELAY_TARGET_ENDPOINT}/{method_path}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    body_bytes = request.get_data(cache=False)
    try:
        body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        body_json = {"raw": body_bytes.decode("utf-8", errors="replace")}

    _capture(
        "cli_request",
        {
            "url": target_url,
            "args": request.args.to_dict(flat=True),
            "headers": headers,
            "body": body_json,
        },
    )

    client = get_http_client()

    stream_mode = request.args.get("alt") == "sse" or method_path.endswith("streamGenerateContent")

    if stream_mode:
        def stream_back():
            with client.stream(
                "POST",
                target_url,
                headers=headers,
                content=body_bytes,
                params=request.args.to_dict(flat=True),
            ) as upstream:
                if upstream.status_code != 200:
                    err_body = upstream.read().decode(errors="replace")
                    _capture(
                        "cli_response_error",
                        {"status": upstream.status_code, "body": err_body},
                    )
                    yield f"data: {json.dumps({'error': {'code': upstream.status_code, 'message': err_body}})}\n\n"
                    return

                for line in upstream.iter_lines():
                    if not line:
                        continue
                    text = line.decode() if isinstance(line, bytes) else line
                    if _CAPTURE_ENABLED:
                        _capture("cli_response_stream_line", {"line": text})
                    yield text + "\n"

        return Response(stream_with_context(stream_back()), mimetype="text/event-stream")

    upstream = client.post(
        target_url,
        headers=headers,
        content=body_bytes,
        params=request.args.to_dict(flat=True),
    )

    try:
        resp_json = upstream.json()
    except Exception:
        resp_json = {"raw": upstream.text}

    _capture(
        "cli_response",
        {
            "status": upstream.status_code,
            "headers": dict(upstream.headers),
            "body": resp_json,
        },
    )

    return Response(
        upstream.content,
        status=upstream.status_code,
        mimetype=upstream.headers.get("content-type", "application/json"),
    )
