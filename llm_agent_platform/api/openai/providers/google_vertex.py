from __future__ import annotations

import os
from typing import Iterable, Any

from llm_agent_platform.api.openai.providers.base import Provider, ProviderRuntimeCreds
from llm_agent_platform.api.openai.types import ChatRequestContext, UpstreamRequestContext, UpstreamPreparationError
from llm_agent_platform.services.http_pool import get_http_client
from llm_agent_platform.auth.credentials import get_vertex_token


class GoogleVertexProvider(Provider):
    id = "google_vertex"

    def load_runtime_credentials(self, account) -> ProviderRuntimeCreds:
        token = get_vertex_token()
        if not token:
            raise UpstreamPreparationError("Vertex AI Service Account not found.", "auth_error", 500)
        return ProviderRuntimeCreds(token=token)

    def prepare_upstream(
        self,
        ctx: ChatRequestContext,
        creds: ProviderRuntimeCreds,
        account,
        model_override: str | None = None,
    ) -> UpstreamRequestContext:
        project_id = os.environ.get('VERTEX_PROJECT_ID')
        location = os.environ.get('VERTEX_LOCATION', 'us-central1')
        model = model_override or ctx.target_model

        payload = {
            "contents": ctx.contents,
            "generationConfig": ctx.gemini_config,
        }
        if ctx.gemini_tools:
            payload["tools"] = ctx.gemini_tools
        if ctx.system_instruction:
            payload["system_instruction"] = {"parts": [{"text": ctx.system_instruction}]}

        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/publishers/google/models/{model}"
            f":{'streamGenerateContent' if ctx.stream else 'generateContent'}"
        )
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }
        return UpstreamRequestContext(
            token=creds.token,
            payload=payload,
            url=url,
            headers=headers,
            params={},
            selected_account=None,
            quota_request_payload=None,
            session_id=ctx.session_id,
            user_prompt_id=ctx.user_prompt_id,
        )

    def execute_non_stream(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> tuple[Any, int]:
        client = get_http_client()
        response = client.post(upstream.url, headers=upstream.headers, json=upstream.payload)
        try:
            data = response.json()
        except Exception:
            data = response.text
        return data, response.status_code

    def stream_lines(
        self,
        ctx: ChatRequestContext,
        upstream: UpstreamRequestContext,
    ) -> Iterable[str | bytes]:
        client = get_http_client()
        with client.stream(
            "POST",
            upstream.url,
            headers=upstream.headers,
            json=upstream.payload,
            params=upstream.params,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode(errors="replace")
                raise RuntimeError(f"{response.status_code}:{body}")

            for line in response.iter_lines():
                yield line
