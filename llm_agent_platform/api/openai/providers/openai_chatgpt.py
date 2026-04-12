from __future__ import annotations

import json
import time
from typing import Any, Iterable

from llm_agent_platform.api.openai.providers.base import Provider, ProviderRuntimeCreds
from llm_agent_platform.api.openai.streaming import build_usage_stream_chunk
from llm_agent_platform.api.openai.types import (
    ChatRequestContext,
    UpstreamPreparationError,
    UpstreamRequestContext,
)
from llm_agent_platform.auth.openai_chatgpt_oauth import (
    OpenAIChatGPTOAuthError,
    OpenAIChatGPTOAuthManager,
    extract_account_id,
)
from llm_agent_platform.config import (
    OPENAI_CHATGPT_BACKEND_BASE_URL,
    OPENAI_CHATGPT_ORIGINATOR,
    OPENAI_CHATGPT_RESPONSES_PATH,
    OPENAI_CHATGPT_USER_AGENT,
    USER_OPENAI_CHATGPT_CREDS_PATH,
)
from llm_agent_platform.services.credentials_paths import resolve_credentials_path_ref
from llm_agent_platform.services.account_router import BaseAccount
from llm_agent_platform.services.http_pool import get_http_client
from llm_agent_platform.services.runtime_state_paths import resolve_runtime_state_paths


def _join_url(base_url: str, path: str) -> str:
    normalized_base = (base_url or OPENAI_CHATGPT_BACKEND_BASE_URL).rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{normalized_base}{normalized_path}"


def _user_message_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "input_text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append({"type": "input_text", "text": str(item.get("text", ""))})
            elif item_type in {"input_text", "input_image", "image_url"}:
                parts.append(dict(item))
        return parts
    return [{"type": "input_text", "text": json.dumps(content, ensure_ascii=False)}]


def _assistant_message_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "output_text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "output_text"}:
                parts.append({"type": "output_text", "text": str(item.get("text", ""))})
            elif item_type == "refusal":
                parts.append({"type": "refusal", "text": str(item.get("text", ""))})
        return parts
    return [{"type": "output_text", "text": json.dumps(content, ensure_ascii=False)}]


def _messages_to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))

        if role == "user":
            content = _user_message_parts(message.get("content", ""))
            if content:
                input_items.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            content = _assistant_message_parts(message.get("content", ""))
            if content:
                input_items.append({"role": "assistant", "content": content})
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function_payload = tool_call.get("function") or {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id")
                        or f"call_{int(time.time())}_{len(input_items)}",
                        "name": function_payload.get("name")
                        or tool_call.get("name")
                        or "",
                        "arguments": function_payload.get("arguments")
                        or tool_call.get("arguments")
                        or "{}",
                    }
                )
            continue

        if role == "tool":
            tool_output = message.get("content", "")
            if not isinstance(tool_output, str):
                tool_output = json.dumps(tool_output, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id")
                    or message.get("id")
                    or f"call_{int(time.time())}_{len(input_items)}",
                    "output": tool_output,
                }
            )
            continue

        content = _user_message_parts(message.get("content", ""))
        if content:
            input_items.append({"role": role, "content": content})
    return input_items


def _map_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function_payload = item.get("function") or {}
        mapped.append(
            {
                "type": "function",
                "name": function_payload.get("name"),
                "description": function_payload.get("description"),
                "parameters": function_payload.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return mapped


def _ensure_all_required(schema: Any) -> Any:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return schema

    result = dict(schema)
    if result.get("additionalProperties") is not False:
        result["additionalProperties"] = False

    properties = result.get("properties")
    if isinstance(properties, dict):
        all_keys = list(properties.keys())
        result["required"] = all_keys
        normalized_properties: dict[str, Any] = {}
        for key, value in properties.items():
            if (
                isinstance(value, dict)
                and value.get("type") == "array"
                and isinstance(value.get("items"), dict)
            ):
                normalized_properties[key] = {
                    **value,
                    "items": _ensure_all_required(value["items"]),
                }
            else:
                normalized_properties[key] = _ensure_all_required(value)
        result["properties"] = normalized_properties

    return result


def _map_codex_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function_payload = item.get("function") or {}
        parameters = function_payload.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        mapped.append(
            {
                "type": "function",
                "name": function_payload.get("name"),
                "description": function_payload.get("description"),
                "parameters": _ensure_all_required(parameters),
                "strict": True,
            }
        )
    return mapped


def _map_reasoning_effort(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    if normalized in {"none", "disable", "disabled", "off"}:
        return None
    return None


def _usage_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens")
    normalized: dict[str, Any] = {
        "prompt_tokens": int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        ),
        "completion_tokens": int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        ),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    if reasoning_tokens is not None:
        normalized["completion_tokens_details"] = {
            "reasoning_tokens": int(reasoning_tokens)
        }
    return normalized


def _shape_non_stream_response(
    payload: dict[str, Any], raw_model: str
) -> dict[str, Any]:
    if payload.get("object") == "chat.completion":
        payload["model"] = raw_model
        return payload

    output_text = payload.get("output_text")
    if not isinstance(output_text, str):
        output_text = ""

    reasoning_text = ""
    tool_calls: list[dict[str, Any]] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    output_text += str(content.get("text", ""))
                if content.get("type") == "reasoning":
                    reasoning_text += str(
                        content.get("summary", content.get("text", ""))
                    )
        elif item_type in {"function_call", "tool_call"}:
            tool_calls.append(
                {
                    "id": item.get("call_id")
                    or f"call_{int(time.time())}_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": json.dumps(
                            item.get("arguments") or item.get("input") or {},
                            ensure_ascii=False,
                        ),
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": output_text}
    finish_reason = "tool_calls" if tool_calls else "stop"
    if reasoning_text:
        message["reasoning_text"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": payload.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": raw_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage_from_payload(payload),
    }


def _stream_chunk(
    raw_model: str, delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": raw_model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _tool_state_key(payload: dict[str, Any]) -> str | None:
    call_id = payload.get("call_id") or payload.get("tool_call_id") or payload.get("id")
    if isinstance(call_id, str) and call_id:
        return call_id
    return None


def _stream_events_from_backend(
    payload: dict[str, Any], raw_model: str, state: dict[str, Any]
) -> list[str]:
    if payload.get("object") == "chat.completion.chunk":
        payload["model"] = raw_model
        return [f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"]

    event_type = str(payload.get("type", ""))
    if event_type in {"response.text.delta", "response.output_text.delta"}:
        delta = str(payload.get("delta", ""))
        if not delta:
            return []
        state["text_delta_seen"] = True
        return [
            f"data: {json.dumps(_stream_chunk(raw_model, {'content': delta}), ensure_ascii=False)}\n\n"
        ]

    if event_type in {
        "response.reasoning.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary.delta",
        "response.reasoning_summary_text.delta",
    }:
        delta = str(payload.get("delta", ""))
        if not delta:
            return []
        state["reasoning_delta_seen"] = True
        return [
            f"data: {json.dumps(_stream_chunk(raw_model, {'reasoning_text': delta}), ensure_ascii=False)}\n\n"
        ]

    if event_type == "response.refusal.delta":
        delta = str(payload.get("delta", ""))
        if not delta:
            return []
        return [
            f"data: {json.dumps(_stream_chunk(raw_model, {'content': f'[Refusal] {delta}'}), ensure_ascii=False)}\n\n"
        ]

    if event_type in {
        "response.tool_call_arguments.delta",
        "response.function_call_arguments.delta",
    }:
        call_id = _tool_state_key(payload)
        name = payload.get("name") or payload.get("function_name") or ""
        arguments_delta = payload.get("delta") or payload.get("arguments") or ""
        if not isinstance(call_id, str) or not call_id:
            return []
        tool_calls_state = state.setdefault("tool_calls", {})
        tool_state = tool_calls_state.setdefault(
            call_id,
            {
                "index": int(payload.get("index", 0) or 0),
                "name": name,
                "saw_delta": False,
            },
        )
        state["has_tool_call"] = True
        tool_state["name"] = name or tool_state.get("name") or ""
        tool_state["saw_delta"] = True
        tool_calls = [
            {
                "index": int(tool_state.get("index", 0) or 0),
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_state.get("name") or "",
                    "arguments": str(arguments_delta),
                },
            }
        ]
        return [
            f"data: {json.dumps(_stream_chunk(raw_model, {'tool_calls': tool_calls}), ensure_ascii=False)}\n\n"
        ]

    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = payload.get("item")
        if not isinstance(item, dict):
            return []
        if event_type == "response.output_item.added":
            return []

        events: list[str] = []
        item_type = item.get("type")
        if (
            item_type == "text"
            and item.get("text")
            and not state.get("text_delta_seen")
        ):
            events.append(
                f"data: {json.dumps(_stream_chunk(raw_model, {'content': str(item['text'])}), ensure_ascii=False)}\n\n"
            )
        elif (
            item_type == "reasoning"
            and item.get("text")
            and not state.get("reasoning_delta_seen")
        ):
            events.append(
                f"data: {json.dumps(_stream_chunk(raw_model, {'reasoning_text': str(item['text'])}), ensure_ascii=False)}\n\n"
            )
        elif item_type == "message":
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                content_type = content.get("type")
                if (
                    content_type in {"text", "output_text"}
                    and content.get("text")
                    and not state.get("text_delta_seen")
                ):
                    events.append(
                        f"data: {json.dumps(_stream_chunk(raw_model, {'content': str(content['text'])}), ensure_ascii=False)}\n\n"
                    )
                elif content_type == "reasoning" and not state.get(
                    "reasoning_delta_seen"
                ):
                    reasoning_text = content.get("summary") or content.get("text")
                    if reasoning_text:
                        events.append(
                            f"data: {json.dumps(_stream_chunk(raw_model, {'reasoning_text': str(reasoning_text)}), ensure_ascii=False)}\n\n"
                        )
        elif (
            item_type in {"function_call", "tool_call"}
            and event_type == "response.output_item.done"
        ):
            call_id = _tool_state_key(item)
            if isinstance(call_id, str) and call_id:
                tool_calls_state = state.setdefault("tool_calls", {})
                tool_state = tool_calls_state.setdefault(
                    call_id,
                    {
                        "index": int(item.get("index", 0) or 0),
                        "name": item.get("name") or item.get("function_name") or "",
                        "saw_delta": False,
                    },
                )
                state["has_tool_call"] = True
                if not tool_state.get("saw_delta"):
                    tool_calls = [
                        {
                            "index": int(tool_state.get("index", 0) or 0),
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_state.get("name")
                                or item.get("name")
                                or item.get("function_name")
                                or "",
                                "arguments": str(
                                    item.get("arguments")
                                    or item.get("function_arguments")
                                    or "{}"
                                ),
                            },
                        }
                    ]
                    events.append(
                        f"data: {json.dumps(_stream_chunk(raw_model, {'tool_calls': tool_calls}), ensure_ascii=False)}\n\n"
                    )
        return events

    if event_type in {"response.completed", "response.done"}:
        response_payload = (
            payload.get("response")
            if isinstance(payload.get("response"), dict)
            else payload
        )
        if not isinstance(response_payload, dict):
            response_payload = {}
        usage = _usage_from_payload(response_payload)
        finish_reason = "tool_calls" if state.get("has_tool_call") else "stop"
        return [
            f"data: {json.dumps(_stream_chunk(raw_model, {}, finish_reason), ensure_ascii=False)}\n\n",
            build_usage_stream_chunk(raw_model, usage),
        ]

    return []


def _write_usage_state(
    credentials_path: str, payload: dict[str, Any], *, account_name: str | None = None
) -> None:
    credentials_file = credentials_path.strip()
    if not credentials_file:
        return
    creds_ref = resolve_credentials_path_ref(
        "openai-chatgpt", credentials_file, account_name=account_name
    )
    output_path = resolve_runtime_state_paths(
        creds_ref.provider_id,
        account_name=creds_ref.account_name,
    ).usage_snapshot_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


class OpenAIChatGPTProvider(Provider):
    id = "openai_chatgpt"

    def load_runtime_credentials(
        self, account: BaseAccount | None
    ) -> ProviderRuntimeCreds:
        state_path = (
            account.credentials_path
            if isinstance(account, BaseAccount)
            else USER_OPENAI_CHATGPT_CREDS_PATH
        )
        manager = OpenAIChatGPTOAuthManager(state_path)
        try:
            state = manager.get_valid_state()
        except OpenAIChatGPTOAuthError as exc:
            raise UpstreamPreparationError(
                f"OpenAI ChatGPT OAuth load failed: {exc}", "auth_error", 401
            ) from exc

        token = str(state.get("access_token") or "").strip()
        if not token:
            raise UpstreamPreparationError(
                "OpenAI ChatGPT OAuth state must contain access_token",
                "auth_error",
                401,
            )

        account_id = (
            state.get("account_id")
            if isinstance(state.get("account_id"), str)
            else None
        )
        if not account_id:
            account_id = extract_account_id(state)

        return ProviderRuntimeCreds(
            token=token,
            token_type="Bearer",
            account_id=account_id,
            metadata={"state_path": str(state_path)},
        )

    def prepare_upstream(
        self,
        ctx: ChatRequestContext,
        creds: ProviderRuntimeCreds,
        account: BaseAccount | None,
        model_override: str | None = None,
    ) -> UpstreamRequestContext:
        instructions = (ctx.system_instruction or "You are Kilo Code.").strip()
        reasoning_effort = _map_reasoning_effort(ctx.data.get("reasoning_effort"))
        payload: dict[str, Any] = {
            "model": model_override or ctx.target_model,
            "stream": bool(ctx.stream),
            "store": False,
            "instructions": instructions,
            "input": _messages_to_input(ctx.messages),
        }

        if reasoning_effort:
            payload["include"] = ["reasoning.encrypted_content"]
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "summary": "auto",
            }

        tools = _map_codex_tools(ctx.data.get("tools") or [])
        if tools:
            payload["tools"] = tools
        if "tool_choice" in ctx.data:
            payload["tool_choice"] = ctx.data["tool_choice"]
        if "parallel_tool_calls" in ctx.data:
            payload["parallel_tool_calls"] = bool(ctx.data["parallel_tool_calls"])

        headers = {
            "Authorization": f"{creds.token_type} {creds.token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if ctx.stream else "application/json",
            "User-Agent": OPENAI_CHATGPT_USER_AGENT,
            "originator": OPENAI_CHATGPT_ORIGINATOR,
            "session_id": ctx.session_id or "",
        }
        if creds.account_id:
            headers["ChatGPT-Account-Id"] = creds.account_id

        state_path = None
        if isinstance(creds.metadata, dict):
            state_path = creds.metadata.get("state_path")

        return UpstreamRequestContext(
            token=creds.token,
            payload=payload,
            url=_join_url(
                OPENAI_CHATGPT_BACKEND_BASE_URL, OPENAI_CHATGPT_RESPONSES_PATH
            ),
            headers=headers,
            params={},
            selected_account=None,
            quota_request_payload=None,
            session_id=ctx.session_id,
            user_prompt_id=ctx.user_prompt_id,
            account_name=account.name if isinstance(account, BaseAccount) else None,
            credentials_path=str(state_path or USER_OPENAI_CHATGPT_CREDS_PATH),
        )

    def _force_refresh(self, upstream: UpstreamRequestContext) -> bool:
        try:
            refreshed = OpenAIChatGPTOAuthManager(
                upstream.credentials_path or USER_OPENAI_CHATGPT_CREDS_PATH
            ).force_refresh()
        except OpenAIChatGPTOAuthError:
            return False

        token = str(refreshed.get("access_token") or "").strip()
        if not token:
            return False

        upstream.token = token
        upstream.headers["Authorization"] = f"Bearer {token}"
        account_id = refreshed.get("account_id")
        if isinstance(account_id, str) and account_id.strip():
            upstream.headers["ChatGPT-Account-Id"] = account_id.strip()
        else:
            upstream.headers.pop("ChatGPT-Account-Id", None)
        return True

    @staticmethod
    def _response_payload(response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text

    def execute_non_stream(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> tuple[Any, int]:
        client = get_http_client()
        response = client.post(
            upstream.url, headers=upstream.headers, json=upstream.payload
        )
        if (
            response.status_code in {401, 403}
            and upstream.credentials_path
            and self._force_refresh(upstream)
        ):
            response = client.post(
                upstream.url, headers=upstream.headers, json=upstream.payload
            )

        payload = self._response_payload(response)
        if response.status_code != 200:
            return payload, response.status_code
        if not isinstance(payload, dict):
            return payload, 502
        usage_payload = {
            "version": 1,
            "provider_id": "openai-chatgpt",
            "account_id": upstream.headers.get("ChatGPT-Account-Id"),
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "limits": {},
            "metadata": {"provider_usage_source": "runtime-response"},
        }
        _write_usage_state(
            upstream.credentials_path or USER_OPENAI_CHATGPT_CREDS_PATH,
            usage_payload,
            account_name=upstream.account_name,
        )
        return _shape_non_stream_response(payload, ctx.raw_model), response.status_code

    def stream_lines(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> Iterable[str | bytes]:
        yielded = False
        attempts = 0
        usage_snapshot: dict[str, Any] | None = None
        stream_state = {
            "has_tool_call": False,
            "reasoning_delta_seen": False,
            "text_delta_seen": False,
            "tool_calls": {},
        }
        while attempts < 2:
            attempts += 1
            with get_http_client().stream(
                "POST",
                upstream.url,
                headers=upstream.headers,
                json=upstream.payload,
            ) as response:
                if (
                    response.status_code in {401, 403}
                    and attempts == 1
                    and upstream.credentials_path
                ):
                    if self._force_refresh(upstream):
                        continue
                if response.status_code != 200:
                    try:
                        response.read()
                        body = response.text
                    except Exception:
                        body = ""
                    raise RuntimeError(f"{response.status_code}: {body}")

                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        decoded = line.decode("utf-8", errors="ignore")
                    else:
                        decoded = str(line)
                    if not decoded.strip():
                        continue

                    if decoded.startswith("data: "):
                        raw_payload = decoded[6:].strip()
                        if raw_payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            parsed = json.loads(raw_payload)
                        except Exception:
                            continue
                        if parsed.get("type") in {
                            "response.completed",
                            "response.done",
                        }:
                            response_payload = (
                                parsed.get("response")
                                if isinstance(parsed.get("response"), dict)
                                else parsed
                            )
                            usage_snapshot = {
                                "version": 1,
                                "provider_id": "openai-chatgpt",
                                "account_id": upstream.headers.get(
                                    "ChatGPT-Account-Id"
                                ),
                                "as_of": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                ),
                                "limits": {},
                                "metadata": {
                                    "provider_usage_source": "runtime-response",
                                    "usage": _usage_from_payload(response_payload),
                                },
                            }
                        for event in _stream_events_from_backend(
                            parsed, ctx.raw_model, stream_state
                        ):
                            if not ctx.include_usage and '"usage"' in event:
                                continue
                            yielded = True
                            yield event
                        continue

                    try:
                        parsed = json.loads(decoded)
                    except Exception:
                        continue
                    for event in _stream_events_from_backend(
                        parsed, ctx.raw_model, stream_state
                    ):
                        if not ctx.include_usage and '"usage"' in event:
                            continue
                        yielded = True
                        yield event
                break

        if usage_snapshot is not None:
            _write_usage_state(
                upstream.credentials_path or USER_OPENAI_CHATGPT_CREDS_PATH,
                usage_snapshot,
                account_name=upstream.account_name,
            )

        if not yielded and ctx.include_usage:
            yield build_usage_stream_chunk(
                ctx.raw_model,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        yield "data: [DONE]\n\n"
