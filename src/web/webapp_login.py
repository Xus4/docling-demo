from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from fastapi import HTTPException


class _AuthUserLike(Protocol):
    username: str
    role: str


class _RequestLike(Protocol):
    session: dict[str, str]


class _ConfigLike(Protocol):
    oa_auth_enabled: bool
    auth_admin_username: str
    access_token_secret: str
    access_token_ttl_sec: int


class _AuthStoreLike(Protocol):
    def authenticate(self, username: str, password: str) -> _AuthUserLike | None: ...


class _ListJobsPayloadLike(Protocol):
    def __call__(
        self,
        user: _AuthUserLike,
        owner: str | None,
        status: str | None,
        q: str | None,
        page: int,
        page_size: int,
    ) -> dict[str, object]: ...


def handle_login(
    *,
    username: str,
    password: str,
    request: _RequestLike,
    config: _ConfigLike,
    auth_store: _AuthStoreLike,
    authenticate_with_oa: Callable[[_ConfigLike, str, str], _AuthUserLike | None],
    create_access_token: Callable[..., str],
) -> dict[str, object]:
    user = None
    if config.oa_auth_enabled:
        admin_name = config.auth_admin_username.strip()
        is_env_admin = bool(admin_name and username.strip().lower() == admin_name.lower())
        if is_env_admin:
            # OA 模式下，env 管理员走本地认证兜底
            user = auth_store.authenticate(admin_name, password)
        else:
            user = authenticate_with_oa(config, username, password)
    else:
        user = auth_store.authenticate(username, password)

    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    request.session["username"] = user.username
    request.session["role"] = user.role
    token = create_access_token(
        username=user.username,
        role=user.role,
        secret=config.access_token_secret,
        ttl_seconds=config.access_token_ttl_sec,
    )
    return {
        "username": user.username,
        "role": user.role,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": config.access_token_ttl_sec,
        "message": f"登录成功，欢迎 {user.username}",
    }


def auth_bootstrap_payload(
    *,
    user: _AuthUserLike,
    list_jobs_payload: _ListJobsPayloadLike,
) -> dict[str, object]:
    return {
        "username": user.username,
        "role": user.role,
        "jobs": list_jobs_payload(user, None, None, None, 1, 100),
    }

