from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_agent_platform.config import (
    GEMINI_ACCOUNTS_CONFIG_PATH,
    OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH,
    QWEN_ACCOUNTS_CONFIG_PATH,
    STATE_DIR,
)
from llm_agent_platform.core.logging import get_logger
from llm_agent_platform.services.account_state_store import (
    PROVIDER_QUOTA_SENTINEL,
    AccountStatePaths,
    load_account_state,
    save_account_state,
    write_group_quota_state,
)

logger = get_logger(__name__)


class AccountRouterError(RuntimeError):
    """Configuration or state error in account router."""


class AllAccountsExhaustedError(AccountRouterError):
    """Raised when all accounts in rounding mode are marked exhausted."""


@dataclass(slots=True)
class BaseAccount:
    name: str
    credentials_path: str


@dataclass(slots=True)
class GeminiAccount(BaseAccount):
    project_id: str


@dataclass(slots=True)
class ProviderGroup:
    accounts: list[str]
    models: list[str]


@dataclass(slots=True)
class ProviderConfig:
    provider: str
    mode: str
    active_account: str
    all_accounts: list[str]
    accounts: dict[str, BaseAccount]
    rate_limit_threshold: int
    quota_exhausted_threshold: int
    rate_limit_cooldown_seconds: int
    random_order: bool
    rotate_after_n_successes: int
    model_quota_resets: dict[str, str]
    quota_scope: str
    groups: dict[str, ProviderGroup]


@dataclass(slots=True)
class SelectedAccount:
    provider: str
    mode: str
    account: BaseAccount
    pool: list[str]
    model: str | None = None


@dataclass(slots=True)
class _ProviderState:
    next_index: int = 0
    has_selected: bool = False
    consecutive_rate_limit_errors: dict[str, int] = field(default_factory=dict)
    consecutive_quota_exhausted_errors: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, float] = field(default_factory=dict)
    quota_exhausted_at: dict[str, dict[str, datetime]] = field(default_factory=dict)
    successes_on_account: dict[str, int] = field(default_factory=dict)
    hydrated_accounts: set[str] = field(default_factory=set)


class RotationEvent:
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"


class RotationDecision:
    RETRY_SAME = "retry_same"
    SWITCH_ACCOUNT = "switch_account"
    ALL_EXHAUSTED = "all_exhausted"
    ALL_COOLDOWN = "all_cooldown"
    NO_ACTION = "no_action"


@dataclass(slots=True)
class EventResult:
    decision: str
    switched: bool
    all_exhausted: bool
    all_cooldown: bool


class QuotaAccountRouter:
    """Provider-aware account router for quota flows."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[tuple[str, str], _ProviderState] = {}
        self._preferred_account_overrides: dict[tuple[str, str], str] = {}

    def select_account(
        self,
        provider: str,
        model: str | None = None,
        group_id: str = "g0",
    ) -> SelectedAccount:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        pool = self._resolve_pool(cfg, group_id)
        if cfg.mode == "single":
            with self._lock:
                state = self._state.setdefault((provider, group_id), _ProviderState())
                self._ensure_pool_hydrated_unlocked(cfg, state, pool)
                self._cleanup_state_unlocked(state, cfg)
                selected_account_name = cfg.active_account
                preferred_account = self._preferred_account_overrides.get(
                    (provider, group_id)
                )
                if (
                    preferred_account in pool
                    and not self._is_exhausted(state, preferred_account, cfg, model)
                    and not self._is_on_cooldown(state, preferred_account)
                ):
                    selected_account_name = preferred_account
                account = self._require_account(cfg, selected_account_name)
                return SelectedAccount(
                    provider=provider,
                    mode=cfg.mode,
                    account=account,
                    pool=[selected_account_name],
                    model=model,
                )

        if not pool:
            raise AccountRouterError(
                f"Provider '{provider}' has mode=rounding but pool is empty"
            )

        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)

            if self._all_exhausted(state, pool, cfg, model):
                logger.warning(
                    f"[{provider}] rounding: all accounts exhausted (model={model}|group={group_id})"
                )
                raise AllAccountsExhaustedError("all_accounts_exceed_quota")

            if self._all_cooldown(state, pool):
                wait_seconds = self._cooldown_wait_seconds(state, pool)
                logger.warning(
                    f"[{provider}] rounding: all accounts on cooldown "
                    f"(model={model}|group={group_id}|wait={wait_seconds}s)"
                )
                raise AccountRouterError(
                    f"all accounts on cooldown please wait {wait_seconds}"
                )

            preferred_account = self._preferred_account_overrides.get(
                (provider, group_id)
            )
            if (
                preferred_account in pool
                and not self._is_exhausted(state, preferred_account, cfg, model)
                and not self._is_on_cooldown(state, preferred_account)
            ):
                state.next_index = pool.index(preferred_account)

            if cfg.random_order:
                if not state.has_selected:
                    chosen = self._choose_random_available(state, pool, cfg, model)
                    if chosen is not None:
                        state.next_index = pool.index(chosen)
                    state.has_selected = True
                else:
                    current_name = pool[state.next_index % len(pool)]
                    if self._is_exhausted(
                        state, current_name, cfg, model
                    ) or self._is_on_cooldown(state, current_name):
                        chosen = self._choose_random_available(state, pool, cfg, model)
                        if chosen is not None:
                            state.next_index = pool.index(chosen)

            start = state.next_index % len(pool)
            for offset in range(len(pool)):
                idx = (start + offset) % len(pool)
                candidate = pool[idx]
                if self._is_exhausted(state, candidate, cfg, model):
                    continue
                if self._is_on_cooldown(state, candidate):
                    continue
                state.next_index = idx
                account = self._require_account(cfg, candidate)
                reason = "rotation" if offset > 0 else "initial"
                logger.info(
                    f"[{provider}] rounding: selected account={candidate} "
                    f"(reason={reason}|attempt={offset + 1}|model={model}|group={group_id})"
                )
                self._enqueue_group_snapshot_unlocked(cfg, state, pool, group_id, model)
                return SelectedAccount(
                    provider=provider,
                    mode=cfg.mode,
                    account=account,
                    pool=list(pool),
                    model=model,
                )

        wait_seconds = self._cooldown_wait_seconds(state, pool)
        raise AccountRouterError(f"all accounts on cooldown please wait {wait_seconds}")

    def register_success(
        self,
        provider: str,
        account_name: str,
        group_id: str = "g0",
        model: str | None = None,
    ) -> None:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        pool = self._resolve_pool(cfg, group_id)
        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)
            state.consecutive_rate_limit_errors[account_name] = 0
            state.consecutive_quota_exhausted_errors[account_name] = 0
            state.cooldown_until.pop(account_name, None)
            self._persist_account_state_unlocked(
                cfg,
                state,
                account_name,
                last_used_at=datetime.now(tz=timezone.utc),
                clear_cooldown=True,
            )
            self._enqueue_group_snapshot_unlocked(cfg, state, pool, group_id, model)

            if cfg.mode != "rounding" or cfg.rotate_after_n_successes <= 0:
                return

            current = state.successes_on_account.get(account_name, 0) + 1
            state.successes_on_account[account_name] = current
            if current < cfg.rotate_after_n_successes:
                return

            state.successes_on_account[account_name] = 0
            if self._set_next_available_unlocked(
                state,
                pool,
                account_name,
                cfg,
                model=None,
                random_order=cfg.random_order,
            ):
                next_account = pool[state.next_index]
                logger.info(
                    f"[{provider}] rounding: switching {account_name} -> {next_account} "
                    f"(trigger=BY_N|group={group_id}|threshold={cfg.rotate_after_n_successes})"
                )

    def register_quota_limit(
        self,
        provider: str,
        account_name: str,
        mode: str,
        pool: list[str],
        group_id: str = "g0",
    ) -> bool:
        result = self.register_event(
            provider=provider,
            account_name=account_name,
            mode=mode,
            pool=pool,
            event=RotationEvent.QUOTA_EXHAUSTED,
            model=None,
            group_id=group_id,
        )
        return result.switched

    def register_event(
        self,
        *,
        provider: str,
        account_name: str,
        mode: str,
        pool: list[str],
        event: str,
        model: str | None,
        group_id: str = "g0",
    ) -> EventResult:
        provider = self._normalize_provider_id(provider)
        if mode != "rounding":
            return EventResult(
                decision=RotationDecision.NO_ACTION,
                switched=False,
                all_exhausted=False,
                all_cooldown=False,
            )

        cfg = self._load_provider_config(provider)

        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)

            if event == RotationEvent.RATE_LIMIT:
                current = state.consecutive_rate_limit_errors.get(account_name, 0) + 1
                state.consecutive_rate_limit_errors[account_name] = current
                state.consecutive_quota_exhausted_errors[account_name] = 0
                state.successes_on_account.pop(account_name, None)

                logger.info(
                    f"[{provider}] rounding: rate_limit error for {account_name} "
                    f"(consecutive={current}/{cfg.rate_limit_threshold}|model={model}|group={group_id})"
                )

                if current < cfg.rate_limit_threshold:
                    return EventResult(
                        decision=RotationDecision.RETRY_SAME,
                        switched=False,
                        all_exhausted=False,
                        all_cooldown=False,
                    )

                state.consecutive_rate_limit_errors[account_name] = 0
                cooldown_started_at = datetime.now(tz=timezone.utc)
                state.cooldown_until[account_name] = (
                    cooldown_started_at.timestamp() + cfg.rate_limit_cooldown_seconds
                )
                self._persist_account_state_unlocked(
                    cfg,
                    state,
                    account_name,
                    last_cooldown_at=cooldown_started_at,
                )
                self._enqueue_group_snapshot_unlocked(cfg, state, pool, group_id, model)

                logger.warning(
                    f"[{provider}] rounding: account {account_name} on cooldown "
                    f"(trigger=RATE_LIMIT|duration={cfg.rate_limit_cooldown_seconds}s|group={group_id})"
                )

                if self._set_next_available_unlocked(
                    state,
                    pool,
                    account_name,
                    cfg,
                    model,
                    random_order=cfg.random_order,
                ):
                    next_account = pool[state.next_index]
                    logger.info(
                        f"[{provider}] rounding: switching {account_name} -> {next_account} "
                        f"(trigger=RATE_LIMIT|consecutive_errors={current}|group={group_id})"
                    )
                    return EventResult(
                        decision=RotationDecision.SWITCH_ACCOUNT,
                        switched=True,
                        all_exhausted=False,
                        all_cooldown=False,
                    )

                logger.error(
                    f"[{provider}] rounding: all accounts on cooldown "
                    f"(trigger=RATE_LIMIT|model={model}|group={group_id})"
                )
                return EventResult(
                    decision=RotationDecision.ALL_COOLDOWN,
                    switched=True,
                    all_exhausted=False,
                    all_cooldown=True,
                )

            current = state.consecutive_quota_exhausted_errors.get(account_name, 0) + 1
            state.consecutive_quota_exhausted_errors[account_name] = current
            state.consecutive_rate_limit_errors[account_name] = 0
            state.successes_on_account.pop(account_name, None)

            logger.info(
                f"[{provider}] rounding: quota_exhausted error for {account_name} "
                f"(consecutive={current}/{cfg.quota_exhausted_threshold}|model={model}|group={group_id})"
            )

            if current < cfg.quota_exhausted_threshold:
                return EventResult(
                    decision=RotationDecision.RETRY_SAME,
                    switched=False,
                    all_exhausted=False,
                    all_cooldown=False,
                )

            state.consecutive_quota_exhausted_errors[account_name] = 0
            exhausted_at = datetime.now(tz=timezone.utc)
            exhausted_key = self._quota_key(cfg, model)
            exhausted_for_account = state.quota_exhausted_at.setdefault(
                account_name, {}
            )
            exhausted_for_account[exhausted_key] = exhausted_at
            self._persist_account_state_unlocked(
                cfg,
                state,
                account_name,
                exhausted_key=exhausted_key,
                exhausted_at=exhausted_at,
            )
            self._enqueue_group_snapshot_unlocked(cfg, state, pool, group_id, model)

            until = self._quota_reset_timestamp(cfg, model, exhausted_at=exhausted_at)
            reset_time = datetime.fromtimestamp(until, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            logger.warning(
                f"[{provider}] rounding: account {account_name} exhausted for model {model} "
                f"(trigger=QUOTA_EXHAUSTED|reset_at={reset_time}|group={group_id})"
            )

            if self._all_exhausted(state, pool, cfg, model):
                logger.error(
                    f"[{provider}] rounding: all accounts exhausted "
                    f"(trigger=QUOTA_EXHAUSTED|model={model}|group={group_id})"
                )
                return EventResult(
                    decision=RotationDecision.ALL_EXHAUSTED,
                    switched=True,
                    all_exhausted=True,
                    all_cooldown=False,
                )

            if self._set_next_available_unlocked(
                state,
                pool,
                account_name,
                cfg,
                model,
                random_order=cfg.random_order,
            ):
                next_account = pool[state.next_index]
                logger.info(
                    f"[{provider}] rounding: switching {account_name} -> {next_account} "
                    f"(trigger=QUOTA_EXHAUSTED|consecutive_errors={current}|group={group_id})"
                )
                return EventResult(
                    decision=RotationDecision.SWITCH_ACCOUNT,
                    switched=True,
                    all_exhausted=False,
                    all_cooldown=False,
                )

            logger.error(
                f"[{provider}] rounding: all accounts exhausted or on cooldown "
                f"(trigger=QUOTA_EXHAUSTED|model={model}|group={group_id})"
            )
            return EventResult(
                decision=RotationDecision.ALL_COOLDOWN,
                switched=True,
                all_exhausted=False,
                all_cooldown=True,
            )

    def all_accounts_exhausted(
        self,
        provider: str,
        pool: list[str],
        model: str | None = None,
        group_id: str = "g0",
    ) -> bool:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)
            return self._all_exhausted(state, pool, cfg, model)

    def all_accounts_on_cooldown(
        self,
        provider: str,
        pool: list[str],
        group_id: str = "g0",
    ) -> bool:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)
            return self._all_cooldown(state, pool)

    def cooldown_wait_seconds(
        self,
        provider: str,
        pool: list[str],
        group_id: str = "g0",
    ) -> int:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)
            return self._cooldown_wait_seconds(state, pool)

    def set_preferred_account(
        self, provider: str, group_id: str, account_name: str
    ) -> None:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        pool = self._resolve_pool(cfg, group_id)
        if account_name not in pool:
            raise AccountRouterError(
                f"Account '{account_name}' is not available for group '{group_id}'"
            )
        with self._lock:
            self._preferred_account_overrides[(provider, group_id)] = account_name

    def get_preferred_account(self, provider: str, group_id: str) -> str | None:
        provider = self._normalize_provider_id(provider)
        with self._lock:
            return self._preferred_account_overrides.get((provider, group_id))

    def describe_group(
        self,
        provider: str,
        group_id: str = "g0",
        model: str | None = None,
    ) -> dict[str, object]:
        provider = self._normalize_provider_id(provider)
        cfg = self._load_provider_config(provider)
        pool = self._resolve_pool(cfg, group_id)
        with self._lock:
            state = self._state.setdefault((provider, group_id), _ProviderState())
            self._ensure_pool_hydrated_unlocked(cfg, state, pool)
            self._cleanup_state_unlocked(state, cfg)
            preferred_account = self._preferred_account_overrides.get(
                (provider, group_id)
            )
            accounts_payload: list[dict[str, object]] = []
            for account_name in pool:
                raw_state = load_account_state(
                    self._state_paths(cfg.provider, account_name)
                )
                account_payload: dict[str, object] = {
                    "account_name": account_name,
                    "state": "ready",
                    "metadata": {
                        "mode": cfg.mode,
                        "quota_scope": cfg.quota_scope,
                    },
                    "raw_account_state": {
                        "version": 1,
                    },
                }
                if raw_state.last_used_at is not None:
                    account_payload["raw_account_state"]["last_used_at"] = (
                        raw_state.last_used_at.astimezone(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                if raw_state.last_cooldown_at is not None:
                    account_payload["raw_account_state"]["cooldown"] = {
                        "last_cooldown_at": raw_state.last_cooldown_at.astimezone(
                            timezone.utc
                        )
                        .isoformat()
                        .replace("+00:00", "Z")
                    }
                if raw_state.quota_exhausted_at:
                    account_payload["raw_account_state"]["quota_exhausted"] = {
                        "keys": {
                            key: value.astimezone(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                            for key, value in raw_state.quota_exhausted_at.items()
                        }
                    }
                if preferred_account == account_name:
                    account_payload["metadata"]["preferred_for_session"] = True

                if self._is_on_cooldown(state, account_name):
                    cooldown_until = datetime.fromtimestamp(
                        state.cooldown_until[account_name], tz=timezone.utc
                    )
                    account_payload["state"] = "cooldown"
                    account_payload["quota_blocked_until"] = (
                        cooldown_until.isoformat().replace("+00:00", "Z")
                    )
                    account_payload["block_reason"] = "rate_limit_cooldown"
                elif self._is_exhausted(state, account_name, cfg, model):
                    quota_key = self._quota_key(cfg, model)
                    exhausted_at = state.quota_exhausted_at.get(account_name, {}).get(
                        quota_key
                    )
                    if exhausted_at is not None:
                        quota_until = datetime.fromtimestamp(
                            self._quota_reset_timestamp(
                                cfg, model, exhausted_at=exhausted_at
                            ),
                            tz=timezone.utc,
                        )
                        account_payload["state"] = "quota_blocked"
                        account_payload["quota_blocked_until"] = (
                            quota_until.isoformat().replace("+00:00", "Z")
                        )
                        account_payload["block_reason"] = "quota_exhausted"

                accounts_payload.append(account_payload)

            return {
                "provider": provider,
                "group_id": group_id,
                "mode": cfg.mode,
                "quota_scope": cfg.quota_scope,
                "preferred_account": preferred_account,
                "accounts": accounts_payload,
            }

    def _all_exhausted(
        self,
        state: _ProviderState,
        pool: list[str],
        cfg: ProviderConfig,
        model: str | None,
    ) -> bool:
        if not pool:
            return False
        return all(self._is_exhausted(state, account, cfg, model) for account in pool)

    def _all_cooldown(self, state: _ProviderState, pool: list[str]) -> bool:
        if not pool:
            return False
        return all(self._is_on_cooldown(state, account) for account in pool)

    def _is_on_cooldown(self, state: _ProviderState, account_name: str) -> bool:
        until = state.cooldown_until.get(account_name)
        if until is None:
            return False
        return until > time.time()

    def _is_exhausted(
        self,
        state: _ProviderState,
        account_name: str,
        cfg: ProviderConfig,
        model: str | None,
    ) -> bool:
        exhausted_for_account = state.quota_exhausted_at.get(account_name, {})
        key = self._quota_key(cfg, model)
        exhausted_at = exhausted_for_account.get(key)
        if exhausted_at is None:
            return False
        return self._is_exhausted_key(cfg, key, exhausted_at, model)

    def _set_next_available_unlocked(
        self,
        state: _ProviderState,
        pool: list[str],
        current_account: str,
        cfg: ProviderConfig,
        model: str | None,
        random_order: bool = False,
    ) -> bool:
        if not pool:
            return False

        if random_order:
            candidates = [
                account
                for account in pool
                if account != current_account
                and not self._is_exhausted(state, account, cfg, model)
                and not self._is_on_cooldown(state, account)
            ]
            if not candidates:
                return False
            chosen = random.choice(candidates)
            state.next_index = pool.index(chosen)
            return True

        idx = pool.index(current_account)
        for offset in range(1, len(pool) + 1):
            candidate_idx = (idx + offset) % len(pool)
            candidate = pool[candidate_idx]
            if self._is_exhausted(state, candidate, cfg, model):
                continue
            if self._is_on_cooldown(state, candidate):
                continue
            state.next_index = candidate_idx
            return True
        return False

    def _cleanup_state_unlocked(
        self, state: _ProviderState, cfg: ProviderConfig
    ) -> None:
        now = time.time()
        for account_name, until in list(state.cooldown_until.items()):
            if until <= now:
                state.cooldown_until.pop(account_name, None)

        for account_name, exhausted_for_account in list(
            state.quota_exhausted_at.items()
        ):
            for exhausted_key, exhausted_at in list(exhausted_for_account.items()):
                reference_model = (
                    None if exhausted_key == PROVIDER_QUOTA_SENTINEL else exhausted_key
                )
                if not self._is_exhausted_key(
                    cfg, exhausted_key, exhausted_at, reference_model
                ):
                    exhausted_for_account.pop(exhausted_key, None)
            if not exhausted_for_account:
                state.quota_exhausted_at.pop(account_name, None)

    def _quota_reset_timestamp(
        self,
        cfg: ProviderConfig,
        model: str | None,
        exhausted_at: datetime | None = None,
    ) -> float:
        reset_time = cfg.model_quota_resets.get(
            model or ""
        ) or cfg.model_quota_resets.get("default", "00:00:00")
        days, hours, minutes = _parse_period(reset_time)
        base = exhausted_at or datetime.now(tz=timezone.utc)
        reset_dt = base + timedelta(days=days, hours=hours, minutes=minutes)
        return reset_dt.timestamp()

    def _model_key(self, model: str | None) -> str:
        return model or "<unknown-model>"

    def _quota_key(self, cfg: ProviderConfig, model: str | None) -> str:
        if cfg.quota_scope == "per_provider":
            return PROVIDER_QUOTA_SENTINEL
        return self._model_key(model)

    def _is_exhausted_key(
        self,
        cfg: ProviderConfig,
        exhausted_key: str,
        exhausted_at: datetime,
        requested_model: str | None,
    ) -> bool:
        model_for_reset = requested_model
        if exhausted_key != PROVIDER_QUOTA_SENTINEL:
            model_for_reset = exhausted_key
        return (
            self._quota_reset_timestamp(cfg, model_for_reset, exhausted_at=exhausted_at)
            > time.time()
        )

    def _normalize_provider_id(self, provider: str) -> str:
        if provider == "gemini":
            return "gemini-cli"
        if provider == "qwen":
            return "qwen_code"
        if provider == "openai-chatgpt":
            return "openai-chatgpt"
        return provider

    def try_load_provider_config(self, provider: str) -> ProviderConfig | None:
        try:
            return self._load_provider_config(provider)
        except AccountRouterError:
            return None

    def _resolve_pool(self, cfg: ProviderConfig, group_id: str) -> list[str]:
        if not cfg.groups:
            return cfg.all_accounts
        if group_id in cfg.groups:
            return list(cfg.groups[group_id].accounts)
        if group_id == "g0":
            return cfg.all_accounts
        raise AccountRouterError(
            f"Unknown group_id '{group_id}' for provider '{cfg.provider}'"
        )

    def _cooldown_wait_seconds(self, state: _ProviderState, pool: list[str]) -> int:
        now = time.time()
        until_values = [state.cooldown_until.get(account) for account in pool]
        active = [value for value in until_values if value is not None and value > now]
        if not active:
            return 0
        return max(0, math.ceil(min(active) - now))

    def _choose_random_available(
        self,
        state: _ProviderState,
        pool: list[str],
        cfg: ProviderConfig,
        model: str | None,
    ) -> str | None:
        candidates = [
            account
            for account in pool
            if not self._is_exhausted(state, account, cfg, model)
            and not self._is_on_cooldown(state, account)
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    def _ensure_pool_hydrated_unlocked(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        pool: list[str],
    ) -> None:
        for account_name in pool:
            self._hydrate_account_state_unlocked(cfg, state, account_name)

    def _hydrate_account_state_unlocked(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        account_name: str,
    ) -> None:
        if account_name in state.hydrated_accounts:
            return

        paths = self._state_paths(cfg.provider, account_name)
        persisted = load_account_state(paths)
        now = time.time()

        if persisted.last_cooldown_at is not None:
            cooldown_until = (
                persisted.last_cooldown_at.timestamp() + cfg.rate_limit_cooldown_seconds
            )
            if cooldown_until > now:
                state.cooldown_until[account_name] = cooldown_until

        if persisted.quota_exhausted_at:
            account_exhausted = state.quota_exhausted_at.setdefault(account_name, {})
            for exhausted_key, exhausted_at in persisted.quota_exhausted_at.items():
                if (
                    cfg.quota_scope == "per_provider"
                    and exhausted_key != PROVIDER_QUOTA_SENTINEL
                ):
                    continue
                if (
                    cfg.quota_scope == "per_model"
                    and exhausted_key == PROVIDER_QUOTA_SENTINEL
                ):
                    continue
                account_exhausted[exhausted_key] = exhausted_at

        state.hydrated_accounts.add(account_name)

    def _persist_account_state_unlocked(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        account_name: str,
        *,
        last_used_at: datetime | None = None,
        last_cooldown_at: datetime | None = None,
        clear_cooldown: bool = False,
        exhausted_key: str | None = None,
        exhausted_at: datetime | None = None,
    ) -> None:
        paths = self._state_paths(cfg.provider, account_name)
        persisted = load_account_state(paths)

        if last_used_at is not None:
            persisted.last_used_at = last_used_at

        if clear_cooldown:
            persisted.last_cooldown_at = None
        elif last_cooldown_at is not None:
            persisted.last_cooldown_at = last_cooldown_at
        elif not self._is_on_cooldown(state, account_name):
            persisted.last_cooldown_at = None

        active_exhausted: dict[str, datetime] = {}
        for key, value in state.quota_exhausted_at.get(account_name, {}).items():
            reference_model = None if key == PROVIDER_QUOTA_SENTINEL else key
            if self._is_exhausted_key(cfg, key, value, reference_model):
                active_exhausted[key] = value
        if exhausted_key is not None and exhausted_at is not None:
            active_exhausted[exhausted_key] = exhausted_at
        persisted.quota_exhausted_at = active_exhausted

        save_account_state(paths, persisted)

    def _enqueue_group_snapshot_unlocked(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        pool: list[str],
        group_id: str,
        model: str | None,
    ) -> None:
        payload = self._build_group_snapshot_unlocked(cfg, state, pool, group_id, model)
        write_group_quota_state(cfg.provider, group_id, payload)

    def _build_group_snapshot_unlocked(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        pool: list[str],
        group_id: str,
        model: str | None,
    ) -> dict[str, object]:
        total_accounts = len(pool)
        cooldown_accounts = sum(
            1 for account in pool if self._is_on_cooldown(state, account)
        )
        payload: dict[str, object] = {
            "version": 1,
            "provider_id": cfg.provider,
            "group_id": group_id,
            "quota_scope": cfg.quota_scope,
            "as_of": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_accounts": total_accounts,
            "cooldown_accounts": cooldown_accounts,
            "cooldown_ratio": 0.0
            if total_accounts == 0
            else cooldown_accounts / total_accounts,
        }

        if cfg.quota_scope == "per_provider":
            exhausted_accounts = sum(
                1 for account in pool if self._is_exhausted(state, account, cfg, model)
            )
            payload["provider"] = {
                "exhausted_accounts": exhausted_accounts,
                "not_exhausted_ratio": 0.0
                if total_accounts == 0
                else (total_accounts - exhausted_accounts) / total_accounts,
            }
            return payload

        models = self._snapshot_models(cfg, state, group_id, model)
        payload["models"] = {
            model_name: {
                "exhausted_accounts": sum(
                    1
                    for account in pool
                    if self._is_exhausted(state, account, cfg, model_name)
                ),
                "not_exhausted_ratio": 0.0
                if total_accounts == 0
                else (
                    total_accounts
                    - sum(
                        1
                        for account in pool
                        if self._is_exhausted(state, account, cfg, model_name)
                    )
                )
                / total_accounts,
            }
            for model_name in models
        }
        return payload

    def _snapshot_models(
        self,
        cfg: ProviderConfig,
        state: _ProviderState,
        group_id: str,
        model: str | None,
    ) -> list[str]:
        ordered: list[str] = []

        def add(candidate: str | None) -> None:
            if not candidate or candidate == PROVIDER_QUOTA_SENTINEL:
                return
            if candidate not in ordered:
                ordered.append(candidate)

        group = cfg.groups.get(group_id)
        if group is not None:
            for group_model in group.models:
                add(group_model)
        add(model)
        for configured_model in cfg.model_quota_resets:
            if configured_model != "default":
                add(configured_model)
        for exhausted_for_account in state.quota_exhausted_at.values():
            for exhausted_key in exhausted_for_account:
                add(exhausted_key)
        if not ordered:
            add(self._model_key(model))
        return ordered

    def _load_provider_config(self, provider: str) -> ProviderConfig:
        provider = self._normalize_provider_id(provider)
        file_path = self._config_path_for_provider(provider)
        if not file_path.exists():
            raise AccountRouterError(
                f"Accounts config for provider '{provider}' not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        mode = payload.get("mode", "single")
        if mode not in {"single", "rounding"}:
            raise AccountRouterError(
                f"Invalid mode '{mode}' in {file_path}. Allowed: single|rounding"
            )

        active_account = payload.get("active_account")
        if not active_account:
            raise AccountRouterError(f"Missing active_account in {file_path}")

        all_accounts = payload.get("all_accounts") or []
        if not isinstance(all_accounts, list):
            raise AccountRouterError(f"all_accounts must be list in {file_path}")

        policy_payload = payload.get("rotation_policy") or {}
        if not isinstance(policy_payload, dict):
            raise AccountRouterError(f"rotation_policy must be object in {file_path}")
        rate_limit_threshold = int(policy_payload.get("rate_limit_threshold", 2))
        quota_exhausted_threshold = int(
            policy_payload.get("quota_exhausted_threshold", 2)
        )
        rate_limit_cooldown_seconds = int(
            policy_payload.get("rate_limit_cooldown_seconds", 5)
        )
        random_order = bool(policy_payload.get("random_order", False))
        rotate_after_n_successes = int(
            policy_payload.get("rotate_after_n_successes", 0)
        )

        model_quota_resets = payload.get("model_quota_resets") or {
            "default": "00:00:00"
        }
        if not isinstance(model_quota_resets, dict):
            raise AccountRouterError(
                f"model_quota_resets must be object in {file_path}"
            )
        model_quota_resets = self._validate_model_quota_resets(
            model_quota_resets, file_path
        )
        if "default" not in model_quota_resets:
            raise AccountRouterError(
                f"model_quota_resets must include 'default' in {file_path}"
            )

        quota_scope = str(payload.get("quota_scope", "per_model"))
        if quota_scope not in {"per_model", "per_provider"}:
            raise AccountRouterError(
                f"Invalid quota_scope '{quota_scope}' in {file_path}. Allowed: per_model|per_provider"
            )

        accounts_payload = payload.get("accounts") or {}
        if not isinstance(accounts_payload, dict):
            raise AccountRouterError(f"accounts must be object in {file_path}")

        accounts: dict[str, BaseAccount] = {}
        for account_name, account_data in accounts_payload.items():
            if not isinstance(account_data, dict):
                raise AccountRouterError(
                    f"Account '{account_name}' config must be object in {file_path}"
                )
            creds_path = account_data.get("credentials_path")
            if not creds_path:
                raise AccountRouterError(
                    f"Missing credentials_path for account '{account_name}' in {file_path}"
                )

            if provider == "gemini_cli":
                project_id = account_data.get("project_id")
                if not project_id:
                    raise AccountRouterError(
                        f"Missing project_id for gemini account '{account_name}' in {file_path}"
                    )
                accounts[account_name] = GeminiAccount(
                    name=account_name,
                    credentials_path=creds_path,
                    project_id=project_id,
                )
            else:
                accounts[account_name] = BaseAccount(
                    name=account_name,
                    credentials_path=creds_path,
                )

        groups_payload = payload.get("groups") or {}
        if not isinstance(groups_payload, dict):
            raise AccountRouterError(f"groups must be object in {file_path}")
        groups: dict[str, ProviderGroup] = {}
        for group_key, group_data in groups_payload.items():
            if not isinstance(group_data, dict):
                raise AccountRouterError(
                    f"Group '{group_key}' must be object in {file_path}"
                )
            group_accounts = group_data.get("accounts") or []
            if not isinstance(group_accounts, list):
                raise AccountRouterError(
                    f"Group '{group_key}' accounts must be list in {file_path}"
                )
            group_models = group_data.get("models") or []
            if not isinstance(group_models, list):
                raise AccountRouterError(
                    f"Group '{group_key}' models must be list in {file_path}"
                )
            groups[group_key] = ProviderGroup(
                accounts=[str(item) for item in group_accounts],
                models=[str(item) for item in group_models],
            )

        self._validate_disjoint_groups(groups, file_path)

        return ProviderConfig(
            provider=provider,
            mode=mode,
            active_account=active_account,
            all_accounts=all_accounts,
            accounts=accounts,
            rate_limit_threshold=rate_limit_threshold,
            quota_exhausted_threshold=quota_exhausted_threshold,
            rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
            random_order=random_order,
            rotate_after_n_successes=rotate_after_n_successes,
            model_quota_resets={str(k): str(v) for k, v in model_quota_resets.items()},
            quota_scope=quota_scope,
            groups=groups,
        )

    def _validate_model_quota_resets(
        self, model_quota_resets: dict, file_path: Path
    ) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for key, value in model_quota_resets.items():
            value_str = str(value)
            _parse_period(value_str, file_path=file_path)
            parsed[str(key)] = value_str
        return parsed

    def _validate_disjoint_groups(
        self, groups: dict[str, ProviderGroup], file_path: Path
    ) -> None:
        if not groups:
            return
        seen: dict[str, str] = {}
        for group_key, group in groups.items():
            for account_name in group.accounts:
                previous = seen.get(account_name)
                if previous and previous != group_key:
                    raise AccountRouterError(
                        f"Account '{account_name}' appears in multiple groups ({previous}, {group_key}) in {file_path}"
                    )
                seen[account_name] = group_key

    def _require_account(self, cfg: ProviderConfig, account_name: str) -> BaseAccount:
        account = cfg.accounts.get(account_name)
        if account is None:
            raise AccountRouterError(
                f"Account '{account_name}' is not declared in accounts section"
            )
        return account

    def _config_path_for_provider(self, provider: str) -> Path:
        if provider in {"gemini_cli", "gemini-cli", "gemini"}:
            return Path(GEMINI_ACCOUNTS_CONFIG_PATH)
        if provider in {"qwen_code", "qwen"}:
            return Path(QWEN_ACCOUNTS_CONFIG_PATH)
        if provider in {"openai_chatgpt", "openai-chatgpt"}:
            return Path(OPENAI_CHATGPT_ACCOUNTS_CONFIG_PATH)
        raise AccountRouterError(f"Unsupported provider: {provider}")

    def _state_paths(self, provider_id: str, account_name: str) -> AccountStatePaths:
        return AccountStatePaths(
            provider_id=provider_id,
            account_name=account_name,
            root_dir=Path(STATE_DIR),
        )


def _parse_period(value: str, file_path: Path | None = None) -> tuple[int, int, int]:
    try:
        day_str, hour_str, minute_str = value.split(":", 2)
        days = int(day_str)
        hours = int(hour_str)
        minutes = int(minute_str)
    except Exception as exc:
        source = f" in {file_path}" if file_path else ""
        raise AccountRouterError(
            f"Invalid reset period '{value}'{source}. Use DD:HH:MM"
        ) from exc

    if days < 0 or not (0 <= hours <= 23) or not (0 <= minutes <= 59):
        source = f" in {file_path}" if file_path else ""
        raise AccountRouterError(
            f"Invalid reset period '{value}'{source}. "
            "Days must be >= 0, hours 0-23, minutes 0-59."
        )
    return days, hours, minutes


quota_account_router = QuotaAccountRouter()
