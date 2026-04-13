from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import g, jsonify, request
from jose import ExpiredSignatureError, JWTError, jwt

from llm_agent_platform.config import JWT_ALGORITHM, JWT_ISSUER, JWT_SHARED_SECRET


@dataclass(frozen=True)
class AdminPrincipal:
    subject: str
    user_id: str
    source_roles: tuple[str, ...]
    mapped_roles: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return "admin" in self.mapped_roles


class AdminAuthError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str,
        error_type: str,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.error_type = error_type


def authorize_admin_request() -> AdminPrincipal:
    raw_token = _extract_bearer_token(request.headers.get("Authorization"))
    if raw_token is None:
        raise AdminAuthError(
            "Missing or malformed bearer token",
            status_code=401,
            code="missing_bearer_token",
            error_type="authentication_error",
        )

    if not JWT_SHARED_SECRET:
        raise AdminAuthError(
            "Admin authentication is not configured",
            status_code=503,
            code="auth_not_configured",
            error_type="configuration_error",
        )

    try:
        payload = jwt.decode(
            raw_token,
            JWT_SHARED_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )
    except ExpiredSignatureError as exc:
        raise AdminAuthError(
            "Access token expired",
            status_code=401,
            code="token_expired",
            error_type="authentication_error",
        ) from exc
    except JWTError as exc:
        raise AdminAuthError(
            "Invalid access token",
            status_code=401,
            code="invalid_token",
            error_type="authentication_error",
        ) from exc

    principal = _principal_from_claims(payload)
    if not principal.is_admin:
        raise AdminAuthError(
            "Admin access required",
            status_code=403,
            code="insufficient_role",
            error_type="authorization_error",
        )

    g.admin_principal = principal
    return principal


def create_admin_auth_error(error: AdminAuthError):
    return (
        jsonify(
            {
                "error": {
                    "message": error.message,
                    "type": error.error_type,
                    "code": error.code,
                }
            }
        ),
        error.status_code,
    )


def _principal_from_claims(payload: dict[str, Any]) -> AdminPrincipal:
    subject = payload.get("sub")
    user_id = payload.get("user_id")
    if not isinstance(subject, str) or not subject.strip():
        raise AdminAuthError(
            "Invalid access token",
            status_code=401,
            code="invalid_token",
            error_type="authentication_error",
        )
    if not isinstance(user_id, str) or not user_id.strip():
        raise AdminAuthError(
            "Invalid access token",
            status_code=401,
            code="invalid_token",
            error_type="authentication_error",
        )

    source_roles = _extract_roles(payload)
    mapped_roles = tuple(_map_role(role) for role in source_roles)
    return AdminPrincipal(
        subject=subject,
        user_id=user_id,
        source_roles=source_roles,
        mapped_roles=mapped_roles,
    )


def _extract_roles(payload: dict[str, Any]) -> tuple[str, ...]:
    roles: list[str] = []

    direct_role = payload.get("role")
    if isinstance(direct_role, str) and direct_role.strip():
        roles.append(direct_role.strip())

    roles_claim = payload.get("roles")
    if isinstance(roles_claim, list):
        for value in roles_claim:
            if isinstance(value, str) and value.strip():
                roles.append(value.strip())

    deduplicated_roles = tuple(dict.fromkeys(roles))
    if deduplicated_roles:
        return deduplicated_roles

    raise AdminAuthError(
        "Invalid access token",
        status_code=401,
        code="invalid_token",
        error_type="authentication_error",
    )


def _map_role(role: str) -> str:
    if role == "developer":
        return "admin"
    return role


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    if not isinstance(authorization_header, str):
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    normalized_token = token.strip()
    if not normalized_token or " " in normalized_token:
        return None
    return normalized_token
