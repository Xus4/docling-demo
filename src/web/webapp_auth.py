from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException, Request

from src.core.access_token import (
    ExpiredSignatureError,
    InvalidTokenError,
    decode_access_token,
)
from src.core.auth import AuthUser, JobRecord


def require_auth_user(
    request: Request,
    *,
    access_token_secret: str,
    oa_auth_enabled: bool,
    get_user: Callable[[str], AuthUser | None],
) -> AuthUser:
    auth_hdr = request.headers.get("Authorization")
    if auth_hdr and auth_hdr.startswith("Bearer "):
        raw = auth_hdr[7:].strip()
        if raw:
            try:
                return decode_access_token(raw, access_token_secret)
            except ExpiredSignatureError:
                raise HTTPException(status_code=401, detail="登录已过期，请重新登录") from None
            except InvalidTokenError:
                raise HTTPException(status_code=401, detail="无效或已失效的凭证") from None

    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")

    if oa_auth_enabled:
        role = request.session.get("role")
        if role not in ("admin", "user"):
            request.session.clear()
            raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
        return AuthUser(username=str(username).strip(), role=str(role))

    user = get_user(str(username))
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
    return user


def is_admin(user: AuthUser) -> bool:
    return user.role == "admin"


def can_access_job(user: AuthUser, job: JobRecord) -> bool:
    return is_admin(user) or job.owner_username == user.username

