# Backend service

`Backend service` — автономная service-local граница для machine-facing provider API и admin-facing backend runtime.

## Что живет в этом repo boundary

- `llm_agent_platform/` — runtime package backend сервиса.
- `llm_agent_platform/tests/` — backend tests и fixtures.
- `scripts/` — OAuth bootstrap scripts, завязанные на backend auth/config layer.
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `.env.example` — service-local dev/run assets.

## Что остается в root repo

- `docs/` — system-level Source of Truth для всей assembled системы `llm_agent_platform`.
- `docker-compose.yml` — local multi-service assembly для `frontend` + `backend`.
- `operational_scope/` и `project/` — root execution/context layer.

## Команды

- `uv sync`
- `uv run python -m llm_agent_platform`
- `uv run python -m unittest discover -s llm_agent_platform/tests -p "test_*.py"`
- `uv run python -m compileall llm_agent_platform`

## Boundary note

Этот каталог materialized как будущий autonomous repo root. Его локальная документация intentionally короткая и не дублирует system architecture из root `docs/`.
