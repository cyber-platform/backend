# Backend Release Notes v0.0.3: `2026-04-15-openai-chatgpt-policy-editor-release`

## Version

- Backend release image tag: `v0.0.3`

## Delivered changes

- Added provider capability overlay loading for `openai-chatgpt`.
- Added key-scoped request policy registry service.
- Added admin routes for capability read and request policy CRUD.
- Materialized pipeline policy resolution for `force` and `default_if_absent`.
- Kept `pass-through by default` when no policy record exists.
- Completed adapter-side `reasoning_effort` propagation for `gpt-5.4*` and `gpt-5.3-codex` baseline.

## Canonical references

- [`provider-request-policy-overrides.md`](../../../../docs/architecture/provider-request-policy-overrides.md)
- [`openai-chatgpt.md`](../../../../docs/providers/openai-chatgpt.md)
- [`openai-chatgpt-model-capabilities-registry.schema.json`](../../../../docs/contracts/config/openai-chatgpt-model-capabilities-registry.schema.json)
- [`openai-chatgpt-request-policy-registry.schema.json`](../../../../docs/contracts/config/openai-chatgpt-request-policy-registry.schema.json)
