from __future__ import annotations

import atexit
import os
import threading

import httpx

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _build_client() -> httpx.Client:
    timeout = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "60"))
    max_connections = int(os.environ.get("HTTP_MAX_CONNECTIONS", "10"))
    max_keepalive = int(os.environ.get("HTTP_MAX_KEEPALIVE_CONNECTIONS", "5"))

    return httpx.Client(
        timeout=timeout,
        http2=True,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
    )


def get_http_client() -> httpx.Client:
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            _client = _build_client()

    return _client


def close_http_client() -> None:
    global _client

    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


atexit.register(close_http_client)
