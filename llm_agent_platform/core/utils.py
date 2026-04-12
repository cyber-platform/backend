import json

def sanitize_string(s):
    """Removes surrogate characters that can't be encoded in UTF-8"""
    if not isinstance(s, str):
        return s
    return s.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def sanitize_data(data):
    """Recursively sanitizes strings in dictionaries and lists"""
    if isinstance(data, str):
        return sanitize_string(data)
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data

def clean_gemini_schema(schema):
    """
    Очищает JSON Schema для совместимости с Gemini API (Google Cloud / Vertex AI).
    1. Преобразует type: ["string", "null"] -> type: "string"
    2. Удаляет $schema
    3. Удаляет additionalProperties (если вызывает проблемы, но пока оставим, если false)
    """
    if not isinstance(schema, dict):
        return schema
    
    new_schema = schema.copy()
    
    # 1. Fix type being a list
    if 'type' in new_schema:
        if isinstance(new_schema['type'], list):
            # Take the first non-null type
            types = [t for t in new_schema['type'] if t != 'null']
            if types:
                new_schema['type'] = types[0]
            else:
                # Fallback if only null or empty
                new_schema['type'] = 'STRING'
    
    # 2. Remove $schema
    if '$schema' in new_schema:
        del new_schema['$schema']
        
    # Recurse
    if 'properties' in new_schema:
        new_schema['properties'] = {k: clean_gemini_schema(v) for k, v in new_schema['properties'].items()}
    
    if 'items' in new_schema:
        new_schema['items'] = clean_gemini_schema(new_schema['items'])
        
    return new_schema

def create_openai_error(message, type="server_error", code=500):
    """Формирует ошибку в формате OpenAI"""
    return json.dumps(sanitize_data({
        "error": {
            "message": message,
            "type": type,
            "param": None,
            "code": code
        }
    }), ensure_ascii=False)