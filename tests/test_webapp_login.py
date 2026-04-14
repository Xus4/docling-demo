import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from src.core.auth import AuthUser
from src.web.webapp_login import auth_bootstrap_payload, handle_login


class _StubRequest:
    def __init__(self) -> None:
        self.session: dict[str, str] = {}


class _FakeAuthStore:
    def __init__(self, *, user: AuthUser | None = None) -> None:
        self.user = user
        self.calls: list[tuple[str, str]] = []

    def authenticate(self, username: str, password: str):
        self.calls.append((username, password))
        return self.user


class TestWebappLogin(unittest.TestCase):
    def test_local_login_success(self) -> None:
        cfg = SimpleNamespace(
            oa_auth_enabled=False,
            auth_admin_username="admin",
            access_token_secret="sec",
            access_token_ttl_sec=123,
        )
        req = _StubRequest()
        store = _FakeAuthStore(user=AuthUser(username="u1", role="user"))
        out = handle_login(
            username="u1",
            password="p1",
            request=req,
            config=cfg,
            auth_store=store,
            authenticate_with_oa=lambda *_: None,
            create_access_token=lambda **_: "tok",
        )
        self.assertEqual(store.calls, [("u1", "p1")])
        self.assertEqual(req.session["username"], "u1")
        self.assertEqual(out["access_token"], "tok")

    def test_oa_env_admin_uses_local_auth(self) -> None:
        cfg = SimpleNamespace(
            oa_auth_enabled=True,
            auth_admin_username="admin",
            access_token_secret="sec",
            access_token_ttl_sec=123,
        )
        req = _StubRequest()
        store = _FakeAuthStore(user=AuthUser(username="admin", role="admin"))
        oa_calls: list[tuple[str, str]] = []

        out = handle_login(
            username="Admin",
            password="p2",
            request=req,
            config=cfg,
            auth_store=store,
            authenticate_with_oa=lambda _cfg, u, p: oa_calls.append((u, p)),
            create_access_token=lambda **_: "tok2",
        )
        self.assertEqual(store.calls, [("admin", "p2")])
        self.assertEqual(oa_calls, [])
        self.assertEqual(out["role"], "admin")

    def test_oa_normal_user_uses_oa(self) -> None:
        cfg = SimpleNamespace(
            oa_auth_enabled=True,
            auth_admin_username="admin",
            access_token_secret="sec",
            access_token_ttl_sec=123,
        )
        req = _StubRequest()
        store = _FakeAuthStore(user=None)
        oa_calls: list[tuple[str, str]] = []

        out = handle_login(
            username="u2",
            password="p3",
            request=req,
            config=cfg,
            auth_store=store,
            authenticate_with_oa=lambda _cfg, u, p: (
                oa_calls.append((u, p)) or AuthUser(username=u, role="user")
            ),
            create_access_token=lambda **_: "tok3",
        )
        self.assertEqual(store.calls, [])
        self.assertEqual(oa_calls, [("u2", "p3")])
        self.assertEqual(out["username"], "u2")

    def test_login_failed_raises_401(self) -> None:
        cfg = SimpleNamespace(
            oa_auth_enabled=False,
            auth_admin_username="admin",
            access_token_secret="sec",
            access_token_ttl_sec=123,
        )
        req = _StubRequest()
        store = _FakeAuthStore(user=None)
        with self.assertRaises(HTTPException) as ctx:
            handle_login(
                username="u1",
                password="bad",
                request=req,
                config=cfg,
                auth_store=store,
                authenticate_with_oa=lambda *_: None,
                create_access_token=lambda **_: "tok",
            )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_auth_bootstrap_payload(self) -> None:
        user = AuthUser(username="u1", role="user")
        out = auth_bootstrap_payload(
            user=user,
            list_jobs_payload=lambda *_: {"items": [], "total": 0},
        )
        self.assertEqual(out["username"], "u1")
        self.assertEqual(out["role"], "user")
        self.assertEqual(out["jobs"]["total"], 0)


if __name__ == "__main__":
    unittest.main()


