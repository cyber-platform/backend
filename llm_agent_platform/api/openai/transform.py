import json

def transform_openai_to_gemini(messages):
    """
    Преобразует сообщения из формата OpenAI в формат Gemini.
    Поддерживает:
    - system/developer (переносится в system_instruction)
    - user (текст, массивы контента, image_url)
    - assistant (текст, tool_calls)
    - tool (functionResponse)
    """
    contents = []
    system_instruction = None
    
    for m in messages:
        role = m.get('role')
        raw_content = m.get('content')
        gemini_parts = []
        
        # 1. Обработка контента (текст и изображения)
        if isinstance(raw_content, str):
            if raw_content:
                gemini_parts.append({"text": raw_content})
        elif isinstance(raw_content, list):
            for part in raw_content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        gemini_parts.append({"text": part.get("text", "")})
                    elif ptype == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:"):
                            try:
                                # data:image/png;base64,iVBOR...
                                mime_part, data_part = img_url.split(",", 1)
                                mime_type = mime_part.split(":")[1].split(";")[0]
                                gemini_parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": data_part
                                    }
                                })
                            except (ValueError, IndexError, KeyError):
                                # Ignore malformed data URLs and continue processing other parts
                                pass
                elif isinstance(part, str):
                    gemini_parts.append({"text": part})

        # 2. Обработка вызовов инструментов (assistant)
        if role == 'assistant' and m.get('tool_calls'):
            for tc in m.get('tool_calls'):
                func = tc.get('function', {})
                try:
                    args = json.loads(func.get('arguments', '{}'))

                    function_call_part = {
                        "functionCall": {
                            "name": func.get('name'),
                            "args": args
                        }
                    }

                    gemini_parts.append(function_call_part)
                except (json.JSONDecodeError, TypeError):
                    # Ignore malformed tool args and continue processing other calls
                    pass

        # 3. Обработка ответов инструментов (tool)
        if role == 'tool':
            # Ответ инструмента в OpenAI - это одна строка контента
            # В Gemini - это functionResponse с именем и объектом ответа
            try:
                # Пытаемся распарсить контент как JSON, если это возможно
                resp_obj = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
            except (json.JSONDecodeError, TypeError):
                resp_obj = raw_content
            
            # Gemini требует, чтобы response был объектом
            if not isinstance(resp_obj, dict):
                resp_obj = {"result": resp_obj}

            f_resp = {
                "name": m.get('name') or "unknown_tool",
                "response": resp_obj
            }
            
            # Пробрасываем ID вызова, если он есть (важно для Gemini 2.0+)
            if m.get('tool_call_id'):
                f_resp["id"] = m.get('tool_call_id')

            gemini_parts.append({"functionResponse": f_resp})

        # 4. Распределение по ролям Gemini (только user и model)
        if role == 'system' or role == 'developer':
            # Накапливаем системные инструкции
            current_text = "".join([p["text"] for p in gemini_parts if "text" in p])
            if system_instruction:
                system_instruction += "\n" + current_text
            else:
                system_instruction = current_text
        else:
            # Ответы инструментов (tool) мапятся на 'user'
            gemini_role = 'user' if role in ['user', 'tool'] else 'model'
            if gemini_parts:
                contents.append({
                    "role": gemini_role,
                    "parts": gemini_parts
                })
            
    return contents, system_instruction
