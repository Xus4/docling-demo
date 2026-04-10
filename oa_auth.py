"""公司 OA / 管理后台登录对接（与 Postman 导出行为对齐；不写本地 users 表）。"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException, Timeout

from auth import AuthUser
from config import AppConfig

log = logging.getLogger("oa_auth")

_DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
)


def _origin_from_login_url(login_url: str) -> str:
    p = urlparse(login_url.strip())
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}"
    return ""


def _oa_http_success(payload: dict[str, Any]) -> bool:
    if payload.get("success") is False:
        return False
    code = payload.get("code")
    if code is None:
        d = payload.get("data")
        return d is not None and d != ""
    if isinstance(code, bool):
        return code
    return code in (0, 200) or str(code) in {"0", "200"}


def _pick_username_from_data(data: dict[str, Any], fallback: str) -> str:
    u = data.get("user")
    user_obj = u if isinstance(u, dict) else {}
    for key in ("userName", "username", "nickName", "loginName"):
        v = user_obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v2 = data.get("username")
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return fallback.strip()


def _is_admin_from_payload(
    data: dict[str, Any],
    login_username: str,
    cfg: AppConfig,
) -> bool:
    if login_username.strip().lower() == cfg.auth_admin_username.strip().lower():
        return True
    u = data.get("user")
    user_obj = u if isinstance(u, dict) else {}
    if user_obj.get("admin") is True:
        return True
    for roles_key in ("roles", "roleIds"):
        roles = data.get(roles_key) or user_obj.get(roles_key)
        if not isinstance(roles, list):
            continue
        for r in roles:
            if isinstance(r, str) and "admin" in r.lower():
                return True
            if r == 1 or r == "1":
                return True
    return False


def authenticate_with_oa(
    cfg: AppConfig,
    username: str,
    password: str,
) -> AuthUser | None:
    """
    调用 OA 登录接口（requests + 与 Postman 一致的头与 JSON 体）。
    默认 OA_AUTH_TRUST_ENV=false，避免 HTTP(S)_PROXY 把内网地址交给公司代理导致超时。
    """
    url = (cfg.oa_auth_login_url or "").strip()
    if not url:
        log.warning("OA 已启用但未配置 OA_AUTH_LOGIN_URL")
        return None

    body: dict[str, Any] = {
        "tenantName": cfg.oa_auth_tenant_name,
        "username": username.strip(),
        "password": password,
        "rememberMe": bool(cfg.oa_auth_remember_me),
    }
    derived_origin = _origin_from_login_url(url)
    origin = cfg.oa_auth_origin.strip() or derived_origin
    referer = cfg.oa_auth_referer.strip() or (
        f"{origin}/login?redirect=/index" if origin else ""
    )
    user_agent = cfg.oa_auth_user_agent.strip() or _DEFAULT_BROWSER_UA

    headers: dict[str, str] = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "tenant-id": cfg.oa_auth_tenant_id,
        "User-Agent": user_agent,
    }
    if origin:
        headers["Origin"] = origin
    if referer:
        headers["Referer"] = referer
    if cfg.oa_auth_cookie:
        headers["Cookie"] = cfg.oa_auth_cookie

    payload_str = json.dumps(body, ensure_ascii=False)
    connect_s = min(10.0, max(2.0, cfg.oa_auth_timeout_sec / 3))
    read_s = max(connect_s, cfg.oa_auth_timeout_sec)

    session = requests.Session()
    session.trust_env = cfg.oa_auth_trust_env

    try:
        resp = session.post(
            url,
            data=payload_str.encode("utf-8"),
            headers=headers,
            timeout=(connect_s, read_s),
            verify=bool(cfg.oa_auth_verify_ssl),
            allow_redirects=True,
        )
    except Timeout:
        log.warning(
            "OA 登录超时 connect=%.1fs read=%.1fs: %s — 若 Postman 正常而此处超时，"
            "多为系统代理（可保持 OA_AUTH_TRUST_ENV=false）或网络不可达",
            connect_s,
            read_s,
            url,
        )
        return None
    except RequestException as exc:
        log.warning("OA 登录请求失败: %s url=%s", exc, url)
        return None

    if resp.status_code != 200:
        log.warning(
            "OA 登录 HTTP 非 200: status=%s body=%s",
            resp.status_code,
            (resp.text or "")[:500],
        )
        return None

    try:
        payload = resp.json()
    except ValueError:
        log.warning("OA 响应非 JSON")
        return None

    if not isinstance(payload, dict) or not _oa_http_success(payload):
        msg = payload.get("msg") if isinstance(payload, dict) else None
        log.info("OA 登录业务失败: msg=%s", msg)
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}

    resolved_name = _pick_username_from_data(data, username)
    if not resolved_name:
        return None

    role = (
        "admin"
        if _is_admin_from_payload(data, resolved_name, cfg)
        else "user"
    )
    return AuthUser(username=resolved_name, role=role)
