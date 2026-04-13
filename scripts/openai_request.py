from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Any

# Add project root to sys.path for direct script execution
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import httpx


def send_openai_request(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    timeout: int = 120,
) -> dict[str, Any] | None:
    """
    Send chat completion request to OpenAI Compatible API endpoint.

    Args:
        base_url: Base API endpoint URL (without /v1/chat/completions)
        model: Model identifier to use
        prompt: User prompt text
        api_key: Optional API key for authentication
        temperature: Sampling temperature 0.0 - 2.0
        max_tokens: Maximum response tokens
        stream: Enable streaming response
        timeout: Request timeout in seconds

    Returns:
        Parsed API response or None on failure
    """
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    try:
        with httpx.Client(timeout=timeout, http2=True) as client:
            if stream:
                with client.stream(
                    "POST", endpoint, headers=headers, json=payload
                ) as response:
                    # Check status before reading stream
                    if response.status_code >= 400:
                        response.read()
                        response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    delta = chunk["choices"][0]["delta"]
                                    if "content" in delta:
                                        print(delta["content"], end="", flush=True)
                                except json.JSONDecodeError:
                                    pass
                    print()
                    return None
            else:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error {e.response.status_code}: {e}", file=sys.stderr)
        # Read response explicitly for streaming cases
        if not e.response.is_closed:
            e.response.read()
        print(f"Response body: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"HTTP Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send request to OpenAI Compatible API endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--base-url", help="Base API endpoint URL")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument("--prompt", help="User prompt text")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key (or use API_KEY env var)",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Maximum response tokens")
    parser.add_argument(
        "--stream", action="store_true", default=None, help="Stream response"
    )
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")
    parser.add_argument(
        "--raw", action="store_true", default=None, help="Output raw JSON response"
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge values: command line args override config, config overrides defaults
    params = {
        "base_url": args.base_url or config.get("base_url"),
        "model": args.model or config.get("model"),
        "prompt": args.prompt or config.get("prompt"),
        "api_key": args.api_key or config.get("api_key"),
        "temperature": args.temperature
        if args.temperature is not None
        else config.get("temperature", 0.7),
        "max_tokens": args.max_tokens
        if args.max_tokens is not None
        else config.get("max_tokens", 1024),
        "stream": args.stream
        if args.stream is not None
        else config.get("stream", False),
        "timeout": args.timeout
        if args.timeout is not None
        else config.get("timeout", 120),
        "raw": args.raw if args.raw is not None else config.get("raw", False),
    }

    # Validate required fields
    required_fields = ["base_url", "model", "prompt"]
    missing = [f for f in required_fields if not params[f]]
    if missing:
        print(f"Missing required parameters: {', '.join(missing)}", file=sys.stderr)
        print("Provide them via config file or command line arguments", file=sys.stderr)
        sys.exit(1)

    result = send_openai_request(
        base_url=params["base_url"],
        model=params["model"],
        prompt=params["prompt"],
        api_key=params["api_key"],
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        stream=params["stream"],
        timeout=params["timeout"],
    )

    if result is not None:
        if params["raw"]:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result["choices"][0]["message"]["content"].strip())

    parser.add_argument("--base-url", required=True, help="Base API endpoint URL")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key (or use API_KEY env var)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum response tokens"
    )
    parser.add_argument("--stream", action="store_true", help="Stream response")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds"
    )
    parser.add_argument("--raw", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    result = send_openai_request(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=args.stream,
        timeout=args.timeout,
    )

    if result is not None:
        if args.raw:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    main()
