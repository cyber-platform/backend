def map_model_name(name):
    """Maps proxy model names to real Google model IDs (January 2026)"""
    mapping = {
        # Quota Group (OAuth)
        'gemini-3.1-pro-preview-quota': 'gemini-3.1-pro-preview',
        'gemini-3-flash-preview-quota': 'gemini-3-flash-preview',
        'gemini-2.5-pro-quota': 'gemini-2.5-pro',
        'gemini-2.5-flash-quota': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite-quota': 'gemini-2.5-flash-lite',
        'qwen-coder-model-quota': 'coder-model',
        
        # Vertex Group (Credits)
        'gemini-3.1-pro-preview-vertex': 'gemini-3.1-pro-preview',
        'gemini-3-flash-preview-vertex': 'gemini-3-flash-preview',
        'gemini-2.5-pro-vertex': 'gemini-2.5-pro',
        'gemini-2.5-flash-vertex': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite-vertex': 'gemini-2.5-flash-lite',
        
        # Specialized
        'gemini-3-pro-image-vertex': 'gemini-3-pro-image-preview',
        'gemini-2.5-flash-image-vertex': 'gemini-2.5-flash-image',
        'nano-banana': 'gemini-2.5-flash-image'
    }
    return mapping.get(name, name)
