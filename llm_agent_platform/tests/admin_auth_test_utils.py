from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence
from unittest.mock import patch

from jose import jwt

TEST_JWT_SHARED_SECRET = "test-jwt-shared-secret"
TEST_JWT_ISSUER = "user_service"
TEST_JWT_ALGORITHM = "HS256"


def patch_admin_auth_config(testcase) -> None:
    for target, value in (
        (
            "llm_agent_platform.api.admin.auth_guard.JWT_SHARED_SECRET",
            TEST_JWT_SHARED_SECRET,
        ),
        ("llm_agent_platform.api.admin.auth_guard.JWT_ISSUER", TEST_JWT_ISSUER),
        ("llm_agent_platform.api.admin.auth_guard.JWT_ALGORITHM", TEST_JWT_ALGORITHM),
    ):
        patcher = patch(target, value)
        patcher.start()
        testcase.addCleanup(patcher.stop)


def build_admin_token(
    *,
    subject: str = "alice",
    user_id: str = "user-123",
    role: str = "developer",
    roles: Sequence[str] | None = None,
    expires_delta: timedelta = timedelta(minutes=5),
) -> str:
    now = datetime.now(tz=timezone.utc)
    normalized_roles = list(roles) if roles is not None else [role]
    payload = {
        "sub": subject,
        "user_id": user_id,
        "role": role,
        "roles": normalized_roles,
        "iss": TEST_JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, TEST_JWT_SHARED_SECRET, algorithm=TEST_JWT_ALGORITHM)


def build_admin_headers(**kwargs) -> dict[str, str]:
    return {"Authorization": f"Bearer {build_admin_token(**kwargs)}"}


def install_admin_client_auth(testcase, client) -> None:
    patch_admin_auth_config(testcase)
    client.environ_base["HTTP_AUTHORIZATION"] = build_admin_headers()["Authorization"]
