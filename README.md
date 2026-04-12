# Backend Service

`services/backend` — backend-граница provider-centric платформы для LLM-провайдеров.

Сервис отдает два основных класса HTTP-маршрутов:

- provider-scoped OpenAI-compatible API, например `/openai-chatgpt/v1/*`;
- admin API, например `/admin/monitoring/*` и `/admin/api-keys/*`.

## Что находится в этом репозитории

- `llm_agent_platform/` — основной Flask runtime package.
- `llm_agent_platform/tests/` — тесты маршрутизации, контрактов, мониторинга и API keys.
- `scripts/` — вспомогательные OAuth bootstrap-скрипты.
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `.env.example` — локальные run/build assets.

## Текущая роль сервиса

- отдавать provider-local OpenAI-compatible endpoints;
- загружать provider descriptors из registry;
- выполнять provider-specific runtime adapters и стратегии вызова;
- поддерживать runtime state по аккаунтам и группам;
- отдавать admin monitoring read-model и operator actions;
- выпускать и проверять platform API keys для публичного namespace `openai-chatgpt`.

## Что не входит в scope этого сервиса

- браузерный UI;
- end-user identity management;
- внешний продуктовый frontend.

## Локальная разработка

Требования:

- Python 3.13
- `uv`

Установка зависимостей:

```bash
uv sync
```

Запуск backend локально:

```bash
uv run python -m llm_agent_platform
```

Локальный адрес по умолчанию:

- `http://127.0.0.1:4000`

Основные runtime inputs обычно приходят из env vars или mounted files:

- `OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH`
- `GEMINI_ACCOUNTS_CONFIG_PATH`
- `QWEN_ACCOUNTS_CONFIG_PATH`
- `STATE_DIR`
- `LOG_DIR`

## Полезные команды

Запуск тестов:

```bash
uv run python -m unittest discover -s llm_agent_platform/tests -p "test_*.py"
```

Compile-check пакета:

```bash
uv run python -m compileall llm_agent_platform
```

Пример OAuth bootstrap-скрипта:

```bash
uv run python scripts/get_openai-chatgpt_credentials.py
```

## Основные семейства маршрутов

- публичные provider routes: `/<provider_name>/v1/models`, `/<provider_name>/v1/chat/completions`
- grouped public routes: `/<provider_name>/<group_name>/v1/*`
- admin monitoring: `/admin/monitoring/*`
- admin API keys: `/admin/api-keys/*`

## Текущие ограничения

- балансировка и routing сейчас provider-local, а не cross-provider;
- текущая стратегия внутри группы — `round robin` по зарегистрированным аккаунтам/пользователям провайдера;
- статистика и monitoring ведутся per provider, per group и per account;
- усиление admin auth и RBAC планируется отдельным этапом.

## Связанные материалы

- основной репозиторий платформы: https://github.com/cyber-platform/llm_agent_platform
