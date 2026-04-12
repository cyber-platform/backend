from __future__ import annotations

import json
import time

from llm_agent_platform.core.utils import sanitize_data, create_openai_error


def _map_gemini_finish_reason(gemini_finish: str | None) -> str | None:
    if gemini_finish == "STOP":
        return "stop"
    if gemini_finish == "MAX_TOKENS":
        return "length"
    return None


def _build_openai_stream_chunk(
    raw_model: str,
    delta: dict,
    finish_reason: str | None,
    index: int = 0,
) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": raw_model,
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


def convert_gemini_stream_chunk(chunk_data: dict, raw_model: str) -> tuple[list[dict], str | None]:
    candidates = chunk_data.get('candidates', [])
    if not candidates:
        return [], None

    candidate = candidates[0]
    gemini_finish = candidate.get('finishReason')
    if gemini_finish in ["SAFETY", "RECITATION", "OTHER"]:
        return [], f"Gemini stream interrupted by safety filters or other reason: {gemini_finish}"

    openai_finish = _map_gemini_finish_reason(gemini_finish)
    content = candidate.get('content', {})
    parts = content.get('parts', [])
    if not parts:
        return [], None

    openai_chunks: list[dict] = []
    current_tool_calls = []
    for part in parts:
        text = part.get('text', '')
        thought_val = part.get('thought')
        is_thought = thought_val is not None

        if is_thought:
            thought_text = text if text else (thought_val if isinstance(thought_val, str) else "")
            if thought_text:
                openai_chunks.append(
                    _build_openai_stream_chunk(
                        raw_model=raw_model,
                        delta={"reasoning_content": thought_text},
                        finish_reason=openai_finish,
                    )
                )
            continue

        if text:
            openai_chunks.append(
                _build_openai_stream_chunk(
                    raw_model=raw_model,
                    delta={"content": text},
                    finish_reason=openai_finish,
                )
            )

        if part.get('functionCall'):
            current_tool_calls.append(part.get('functionCall'))

    if current_tool_calls:
        openai_tool_calls = []
        for idx, fn in enumerate(current_tool_calls):
            openai_tool_calls.append({
                "index": idx,
                "id": fn.get('id') or f"call_{int(time.time())}_{idx}",
                "type": "function",
                "function": {
                    "name": fn.get('name'),
                    "arguments": json.dumps(fn.get('args', {}))
                }
            })

        openai_chunks.append(
            _build_openai_stream_chunk(
                raw_model=raw_model,
                delta={"tool_calls": openai_tool_calls},
                finish_reason=openai_finish or "tool_calls",
            )
        )

    return openai_chunks, None


def extract_usage_update_from_gemini_chunk(chunk_data: dict) -> dict | None:
    usage_meta = chunk_data.get('usageMetadata', {})
    if not usage_meta:
        return None

    usage_update = {
        "prompt_tokens": usage_meta.get('promptTokenCount', 0),
        "completion_tokens": usage_meta.get('candidatesTokenCount', 0),
        "total_tokens": usage_meta.get('totalTokenCount', 0),
    }
    thoughts_token_count = usage_meta.get('thoughtsTokenCount')
    if thoughts_token_count is not None:
        usage_update["reasoning_tokens"] = thoughts_token_count

    return usage_update


def merge_usage_accumulated(usage_accumulated: dict, usage_update: dict) -> None:
    usage_accumulated["prompt_tokens"] = usage_update.get("prompt_tokens", 0)
    usage_accumulated["completion_tokens"] = usage_update.get("completion_tokens", 0)
    usage_accumulated["total_tokens"] = usage_update.get("total_tokens", 0)

    reasoning_tokens = usage_update.get("reasoning_tokens")
    if reasoning_tokens is not None:
        usage_accumulated.setdefault("completion_tokens_details", {})
        usage_accumulated["completion_tokens_details"]["reasoning_tokens"] = reasoning_tokens


def gemini_chunk_to_sse_events(
    chunk_data: dict,
    raw_model: str,
    usage_accumulated: dict,
) -> tuple[list[str], str | None]:
    openai_chunks, policy_error = convert_gemini_stream_chunk(chunk_data, raw_model)
    if policy_error:
        return [], create_openai_error(policy_error, 'policy_violation', 400)

    usage_update = extract_usage_update_from_gemini_chunk(chunk_data)
    if usage_update:
        merge_usage_accumulated(usage_accumulated, usage_update)

    events = [
        f"data: {json.dumps(sanitize_data(chunk), ensure_ascii=False)}\n\n"
        for chunk in openai_chunks
    ]
    return events, None


def build_usage_stream_chunk(raw_model: str, usage_accumulated: dict) -> str:
    usage_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": raw_model,
        "choices": [],
        "usage": usage_accumulated,
    }
    return f"data: {json.dumps(sanitize_data(usage_chunk), ensure_ascii=False)}\n\n"


def parse_vertex_stream_line(line: str | bytes) -> dict | None:
    if isinstance(line, bytes):
        line = line.decode(errors="ignore")

    clean_line = line.strip().strip(',').strip('[').strip(']')
    if not clean_line:
        return None

    try:
        chunk_data = json.loads(clean_line)
    except Exception:
        return None

    if not isinstance(chunk_data, dict):
        return None
    return chunk_data
