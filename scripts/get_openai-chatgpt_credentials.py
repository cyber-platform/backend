from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "secrets" / "openai-chatgpt" / "accounts" / "user_credentials.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_file(path: Path, *, override: bool) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed_value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = parsed_value

_load_env_file(PROJECT_ROOT / ".env", override=False)
_load_env_file(PROJECT_ROOT / ".env.oauth", override=True)

from llm_agent_platform.auth.openai_chatgpt_oauth import (
    OpenAIChatGPTOAuthError,
    build_authorization_url,
    build_redirect_uri,
    exchange_code_for_tokens,
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    normalize_token_payload,
    write_openai_chatgpt_oauth_state,
)
from llm_agent_platform.config import USER_OPENAI_CHATGPT_CREDS_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap OpenAI ChatGPT OAuth credentials via Authorization Code + PKCE.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for persisted OAuth state JSON.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open the authorization URL in a browser automatically.",
    )
    return parser.parse_args()


class _ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _start_callback_server(expected_state: str, redirect_uri: str) -> tuple[_ReusableHTTPServer, dict[str, str], Exception | None]:
    parsed_redirect = urlparse(redirect_uri)
    callback_path = parsed_redirect.path or "/"
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                return

            query = parse_qs(parsed.query)
            state = (query.get("state") or [""])[0]
            if state != expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"OAuth state mismatch")
                return

            code = (query.get("code") or [""])[0]
            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            print("Callback received on local listener, authorization code captured.", flush=True)
            result["code"] = code
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization completed. You can close this tab.")

        def log_message(self, format, *args):  # noqa: A003
            return

    bind_error: Exception | None = None
    server: _ReusableHTTPServer | None = None
    try:
        server = _ReusableHTTPServer(
            (parsed_redirect.hostname or "127.0.0.1", parsed_redirect.port or 1455),
            CallbackHandler,
        )
    except Exception as exc:
        bind_error = exc
    return server, result, bind_error


def _wait_for_callback(server: _ReusableHTTPServer, result: dict[str, str]) -> str:
    try:
        while "code" not in result:
            server.handle_request()
    finally:
        server.server_close()
    return result["code"]


def main() -> int:
    args = _parse_args()
    redirect_uri = build_redirect_uri()
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = generate_state()
    authorization_url = build_authorization_url(
        code_challenge=code_challenge,
        state=state,
        redirect_uri=redirect_uri,
    )

    server, callback_result, bind_error = _start_callback_server(state, redirect_uri)
    if bind_error is not None or server is None:
        print(f"Redirect URI: {redirect_uri}")
        print(
            "OpenAI ChatGPT OAuth bootstrap failed: "
            f"cannot bind callback server on {redirect_uri}. "
            f"Underlying error: {bind_error}"
        )
        return 1

    print(f"Redirect URI: {redirect_uri}")
    print(f"Authorization URL: {authorization_url}")
    if not args.no_browser:
        webbrowser.open(authorization_url)

    try:
        code = _wait_for_callback(server, callback_result)
        print("Exchanging authorization code for tokens...", flush=True)
        token_payload = exchange_code_for_tokens(
            code=code,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )
        print("Token response received, normalizing OAuth state...", flush=True)
        normalized = normalize_token_payload(token_payload)
        output_path = write_openai_chatgpt_oauth_state(normalized, Path(args.output))
    except OpenAIChatGPTOAuthError as exc:
        print(f"OpenAI ChatGPT OAuth bootstrap failed: {exc}")
        return 1
    except Exception as exc:
        print(f"OpenAI ChatGPT OAuth bootstrap failed with unexpected error: {exc}")
        return 1

    print(f"OAuth state written to {output_path}")
    if normalized.get("account_id"):
        print(f"Detected account_id: {normalized['account_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
