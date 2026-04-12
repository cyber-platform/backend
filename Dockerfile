FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

WORKDIR /app

# Копируем только описание среды и lock-файл.
# Код приложения в dev-режиме приходит через bind mount из docker-compose.
COPY pyproject.toml uv.lock ./

# Устанавливаем только среду выполнения по lock-файлу.
# При изменении зависимостей/версий нужен rebuild образа.
RUN uv sync --frozen --no-dev

# Открываем порт
EXPOSE 4000

# Запускаем локальный код, смонтированный в /app/llm_agent_platform
CMD ["uv", "run", "-m", "llm_agent_platform"]
