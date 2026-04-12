import os

from llm_agent_platform.core.logging import get_logger

logger = get_logger(__name__)

# Global state for project ID
project_id_cache = None

def discover_project_id(token):
    """
    Discovers the Google Cloud project ID associated with the user's account.
    Prioritizes environment variables, then attempts to fetch user info.
    Does NOT use Cloud Code API for discovery to avoid circular dependency on API activation.
    """
    global project_id_cache
    if project_id_cache:
        return project_id_cache

    # Check env var first
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        project_id_cache = os.environ.get("GOOGLE_CLOUD_PROJECT")
        logger.info(f"[AUTH] Using Project ID from env: {project_id_cache}")
        return project_id_cache

    # Fallback to a hardcoded default if known, or try to infer from user info
    # For now, we strongly rely on the env var because the Cloud Code API discovery endpoint
    # itself requires the API to be enabled, creating a chicken-and-egg problem.
    
    logger.warning("[AUTH] Warning: GOOGLE_CLOUD_PROJECT env var not set. Using fallback project discovery.")
    
    # Try to get project from userinfo (not standard, but sometimes helpful for debugging)
    # or just return a placeholder that might work if the user has a default project set
    # in their gcloud config (though we can't access gcloud config here directly).
    
    # Returning None here will cause an error downstream, prompting the user to set the env var.
    return None
