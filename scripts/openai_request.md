# OpenAI Compatible API Request Script

Скрипт для отправки запросов на любой эндпоинт совместимый с OpenAI Chat Completions API.

## Использование

Запуск из директории `services/backend/`:

```bash
uv run scripts/openai_request.py [опции]
```

## Режимы работы

Скрипт поддерживает два режима конфигурации:

1.  **Через аргументы командной строки**
2.  **Через JSON конфигурационный файл**

Аргументы командной строки всегда имеют приоритет над значениями из конфига.

## Параметры

| Параметр          | Обязательный | По умолчанию | Описание
|-------------------|--------------|--------------|----------
| `--config`        | ❌           | -            | Путь к JSON конфигурационному файлу
| `--base-url`      | ✅           | -            | Базовый URL API эндпоинта
| `--model`         | ✅           | -            | Идентификатор модели
| `--prompt`        | ✅           | -            | Текст пользовательского запроса
| `--api-key`       | ❌           | `$API_KEY`   | API ключ (также берется из переменной окружения)
| `--temperature`   | ❌           | `0.7`        | Температура семплирования 0.0 - 2.0
| `--max-tokens`    | ❌           | `1024`       | Максимальное количество токенов в ответе
| `--stream`        | ❌           | `false`      | Включить стриминг ответа
| `--timeout`       | ❌           | `120`        | Таймаут запроса в секундах
| `--raw`           | ❌           | `false`      | Вывести полный JSON ответ API

## Пример конфигурационного файла

`request_config.json`:
```json
{
  "base_url": "http://localhost:8000",
  "model": "llama-3.1-8b-instruct",
  "prompt": "Напиши пример простой функции на Python",
  "temperature": 0.3,
  "max_tokens": 512,
  "stream": true,
  "raw": false
}
```

Запуск с конфигом:
```bash
uv run scripts/openai_request.py --config request_config.json
```

Переопределение отдельных параметров:
```bash
uv run scripts/openai_request.py --config request_config.json --temperature 0.8 --prompt "Другой запрос"
```

## Примеры запуска

### Обычный запрос
```bash
uv run scripts/openai_request.py \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --prompt "Привет, как дела?"
```

### Стриминг ответа
```bash
uv run scripts/openai_request.py \
  --base-url http://localhost:8000 \
  --model local-model \
  --prompt "Расскажи про HTTP/2" \
  --stream
```

### Сырой JSON вывод
```bash
uv run scripts/openai_request.py \
  --base-url https://api.openai.com/v1 \
  --model gpt-3.5-turbo \
  --prompt "Тест" \
  --raw
```

## Особенности

✅ Полная совместимость со стандартом OpenAI API
✅ Поддержка HTTP/2
✅ Стриминг ответов
✅ Автоматическое чтение API ключа из окружения
✅ Обработка ошибок и кодов статуса
✅ Валидация обязательных параметров
✅ Следует всем соглашениям кодовой базы проекта
