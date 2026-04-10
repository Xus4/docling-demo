"""JWT 访问凭证（与前端 localStorage + Authorization: Bearer 配合）。"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import jwt

from auth import AuthUser

# 与 PyJWT 异常类型一致，供 webapp 捕获
ExpiredSignatureError = jwt.ExpiredSignatureError
InvalidTokenError = jwt.InvalidTokenError


def _jwt_hs256_key(secret: str) -> bytes:
    """将任意长度口令派生为 32 字节密钥，满足 HS256 建议长度并消除弱密钥告警。"""
    return hashlib.sha256(secret.encode("utf-8")).digest()


def create_access_token(
    *,
    username: str,
    role: str,
    secret: str,
    ttl_seconds: int,
) -> str:
    if role not in ("admin", "user"):
        role = "user"
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": username.strip(),
        "role": role,
        "iat": now,
        "exp": now + max(60, int(ttl_seconds)),
    }
    return jwt.encode(payload, _jwt_hs256_key(secret), algorithm="HS256")


def decode_access_token(token: str, secret: str) -> AuthUser:
    data = jwt.decode(token, _jwt_hs256_key(secret), algorithms=["HS256"])
    username = data.get("sub")
    role = data.get("role")
    if not username or not isinstance(username, str):
        raise jwt.InvalidTokenError("missing sub")
    if role not in ("admin", "user"):
        raise jwt.InvalidTokenError("invalid role")
    return AuthUser(username=username.strip(), role=str(role))
