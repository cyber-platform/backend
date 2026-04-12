from __future__ import annotations

import json
import time

from llm_agent_platform.core.utils import sanitize_data, create_openai_error


def shape_gemini_nonstream_response(resp_data: dict, raw_model: str) -> tuple[str, int]:
    candidates = resp_data.get('candidates', [])
    text = ""
    reasoning_text = ""
    all_function_calls = []
    finish_reason = "stop"

    if candidates:
        candidate = candidates[0]
        gemini_finish = candidate.get('finishReason')

        if gemini_finish in ["SAFETY", "RECITATION", "OTHER"]:
            return create_openai_error(f"Gemini blocked request: {gemini_finish}", "policy_violation", 400), 400

        content = candidate.get('content', {})
        parts = content.get('parts', [])

        for part in parts:
            p_text = part.get('text', '')
            thought = part.get('thought')
            if thought is not None:
                reasoning_text += p_text if p_text else (thought if isinstance(thought, str) else "")
            else:
                if p_text:
                    text += p_text
                if part.get('functionCall'):
                    all_function_calls.append(part.get('functionCall'))

        if gemini_finish == "MAX_TOKENS":
            finish_reason = "length"
        elif all_function_calls:
            finish_reason = "tool_calls"

    usage = resp_data.get('usageMetadata', {})

    message_content = {
        "role": "assistant",
        "content": text,
    }
    if reasoning_text:
        message_content["reasoning_content"] = reasoning_text

    if all_function_calls:
        message_content["tool_calls"] = []
        for idx, fn in enumerate(all_function_calls):
            message_content["tool_calls"].append({
                "id": fn.get('id') or f"call_{int(time.time())}_{idx}",
                "type": "function",
                "function": {
                    "name": fn.get('name'),
                    "arguments": json.dumps(fn.get('args', {})),
                }
            })

    openai_response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": raw_model,
        "choices": [{
            "index": 0,
            "message": message_content,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.get('promptTokenCount', 0),
            "completion_tokens": usage.get('candidatesTokenCount', 0),
            "total_tokens": usage.get('totalTokenCount', 0),
        }
    }

    if 'thoughtsTokenCount' in usage:
        openai_response["usage"]["completion_tokens_details"] = {
            "reasoning_tokens": usage['thoughtsTokenCount']
        }

    return json.dumps(sanitize_data(openai_response), ensure_ascii=False), 200
