FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY llm_agent_platform ./llm_agent_platform

EXPOSE 4000

CMD ["uv", "run", "-m", "llm_agent_platform"]
