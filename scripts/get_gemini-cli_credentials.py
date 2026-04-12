from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from google_auth_oauthlib.flow import InstalledAppFlow

# Позволяет запускать скрипт напрямую: `python scripts/get_gemini-cli_credentials.py`
# и `uv run scripts/get_gemini-cli_credentials.py`.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import GEMINI_CLI_CLIENT_ID, GEMINI_CLI_CLIENT_SECRET, USER_GEMINI_CREDS_PATH

# Scopes required for Cloud Code API
SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]


def _is_true_env(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_callback_port() -> int:
    port_env = os.environ.get('OAUTH_CALLBACK_PORT')
    if port_env:
        try:
            port = int(port_env)
        except ValueError as exc:
            raise ValueError(f"Invalid OAUTH_CALLBACK_PORT: {port_env}") from exc

        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid OAUTH_CALLBACK_PORT: {port_env}")
        return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return int(sock.getsockname()[1])


def _get_bind_addr() -> str:
    # Как и в gemini-cli: redirect URI остаётся loopback (127.0.0.1),
    # но bind-адрес callback сервера можно переопределить env-переменной.
    return os.environ.get('OAUTH_CALLBACK_HOST', '127.0.0.1')


def _extract_code_from_input(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError('Пустой ввод.')

    if raw.startswith('http://') or raw.startswith('https://'):
        parsed = urlparse(raw)
        query = parse_qs(parsed.query)
        code = query.get('code', [None])[0]
        if not code:
            raise ValueError('В callback URL не найден параметр code.')
        return code

    return raw


def _authorize_manual(flow: InstalledAppFlow, open_browser: bool) -> object:
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
    )

    print('\n[MANUAL] Откройте ссылку ниже и завершите авторизацию:')
    print(auth_url)

    if open_browser:
        print('[MANUAL] Если браузер открылся автоматически, дождитесь редиректа и скопируйте URL из адресной строки.')

    print('[MANUAL] Вставьте полный callback URL (http://127.0.0.1:PORT/?code=...) или только значение code:')
    callback_or_code = input('> ').strip()
    code = _extract_code_from_input(callback_or_code)

    flow.fetch_token(code=code)
    return flow.credentials

def main():
    # Fix for "Scope has changed" error: Google adds 'openid' scope automatically
    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
    
    print("=== Google OAuth 2.0 Credentials Generator (Gemini CLI Emulation) ===")
    print("\n[INFO] This script will authenticate you using the official Gemini CLI Client ID.")
    print("       This is required to access the 1500 req/day user quota.")

    if not GEMINI_CLI_CLIENT_ID:
        raise RuntimeError(
            "GEMINI_CLI_CLIENT_ID is not set. Configure it in environment before running OAuth script."
        )
    if not GEMINI_CLI_CLIENT_SECRET:
        raise RuntimeError(
            "GEMINI_CLI_CLIENT_SECRET is not set. Configure it in environment before running OAuth script."
        )

    callback_port = _get_callback_port()
    bind_addr = _get_bind_addr()
    no_browser = _is_true_env('NO_BROWSER')
    redirect_uri = f"http://127.0.0.1:{callback_port}/"
    
    # Create client config dictionary manually instead of loading from file
    client_config = {
        "installed": {
            "client_id": GEMINI_CLI_CLIENT_ID,
            "client_secret": GEMINI_CLI_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri]
        }
    }

    try:
        # Initialize flow with the manual config
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        flow.redirect_uri = redirect_uri
        
        print("\n[WAIT] Ожидание подтверждения в браузере...")
        print(f"[INFO] Callback bind: {bind_addr}:{callback_port} | Redirect URI: {redirect_uri}")
        if no_browser:
            print("[INFO] NO_BROWSER=true: ссылка будет выведена в терминал, откройте её вручную.")

        if no_browser:
            auth_creds = _authorize_manual(flow, open_browser=False)
        else:
            try:
                auth_creds = flow.run_local_server(
                    host='127.0.0.1',
                    bind_addr=bind_addr,
                    port=callback_port,
                    open_browser=True,
                    timeout_seconds=300,
                    access_type='offline',
                    prompt='consent',
                    success_message='Авторизация успешна! Вы можете закрыть это окно и вернуться в терминал.'
                )
            except TimeoutError:
                print("\n[WARN] Локальный callback не был получен за 300 секунд. Перехожу в ручной режим.")
                auth_creds = _authorize_manual(flow, open_browser=False)

        if not auth_creds.refresh_token:
            raise RuntimeError(
                'Google не вернул refresh_token. Попробуйте отозвать доступ приложения "Google Cloud Code" '
                'в Google Account permissions и пройти авторизацию заново.'
            )
        
        # Save credentials in the format expected by main.py
        creds = {
            "client_id": GEMINI_CLI_CLIENT_ID,
            "client_secret": GEMINI_CLI_CLIENT_SECRET,
            "refresh_token": auth_creds.refresh_token,
            "type": "authorized_user"
        }

        target_path = USER_GEMINI_CREDS_PATH
        target_dir = os.path.dirname(target_path) or '.'
        os.makedirs(target_dir, exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(creds, f, indent=2)
            
        print(f"\n[SUCCESS] Credentials saved to '{target_path}'")
        print("You can now restart your Docker container to apply changes.")

    except KeyboardInterrupt:
        print("\n[INFO] Авторизация прервана пользователем.")
    except TimeoutError:
        print("\n[ERROR] OAuth flow timed out (300s). Попробуйте снова или используйте NO_BROWSER=true.")
    except Exception as e:
        print(f"\n[ERROR] OAuth flow failed: {e}")

if __name__ == '__main__':
    main()
