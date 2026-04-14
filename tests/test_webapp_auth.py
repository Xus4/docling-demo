import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from src.core.auth import AuthUser, JobRecord
from src.web.webapp_auth import (
    ExpiredSignatureError,
    InvalidTokenError,
    can_access_job,
    is_admin,
    require_auth_user,
)


class _StubRequest:
    def __init__(self, headers: dict[str, str] | None = None, session: dict | None = None) -> None:
        self.headers = headers or {}
        self.session = session or {}


def _build_job(owner_username: str) -> JobRecord:
    return JobRecord(
        job_id="b" * 32,
        owner_username=owner_username,
        role_snapshot="user",
        original_filename="x.pdf",
        status="queued",
        input_file="/tmp/in.pdf",
        output_file="/tmp/out.md",
        error_message=None,
        cancel_requested=0,
        attempt_count=1,
        created_at="2026-01-01T00:00:00+00:00",
        started_at=None,
        finished_at=None,
    )


class TestWebappAuth(unittest.TestCase):
    def test_require_auth_user_accepts_bearer(self) -> None:
        req = _StubRequest(headers={"Authorization": "Bearer token"}, session={})
        get_user = Mock(return_value=None)
        with patch("src.web.webapp_auth.decode_access_token", return_value=AuthUser(username="u1", role="user")):
            user = require_auth_user(
                req, access_token_secret="s", oa_auth_enabled=False, get_user=get_user
            )
        self.assertEqual(user.username, "u1")
        get_user.assert_not_called()

    def test_require_auth_user_bearer_invalid(self) -> None:
        req = _StubRequest(headers={"Authorization": "Bearer token"}, session={})
        get_user = Mock(return_value=None)
        with patch(
            "src.web.webapp_auth.decode_access_token",
            side_effect=InvalidTokenError("bad token"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                require_auth_user(
                    req, access_token_secret="s", oa_auth_enabled=False, get_user=get_user
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_require_auth_user_bearer_expired(self) -> None:
        req = _StubRequest(headers={"Authorization": "Bearer token"}, session={})
        get_user = Mock(return_value=None)
        with patch(
            "src.web.webapp_auth.decode_access_token",
            side_effect=ExpiredSignatureError("expired"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                require_auth_user(
                    req, access_token_secret="s", oa_auth_enabled=False, get_user=get_user
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_require_auth_user_no_session(self) -> None:
        req = _StubRequest(headers={}, session={})
        with self.assertRaises(HTTPException) as ctx:
            require_auth_user(req, access_token_secret="s", oa_auth_enabled=False, get_user=lambda _: None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_require_auth_user_oa_mode_uses_session_role(self) -> None:
        req = _StubRequest(session={"username": "u1", "role": "admin"})
        user = require_auth_user(req, access_token_secret="s", oa_auth_enabled=True, get_user=lambda _: None)
        self.assertEqual(user.role, "admin")

    def test_require_auth_user_oa_mode_invalid_role(self) -> None:
        sess = {"username": "u1", "role": "bad"}
        req = _StubRequest(session=sess)
        with self.assertRaises(HTTPException):
            require_auth_user(req, access_token_secret="s", oa_auth_enabled=True, get_user=lambda _: None)
        self.assertEqual(req.session, {})

    def test_require_auth_user_local_mode_missing_user(self) -> None:
        sess = {"username": "u1"}
        req = _StubRequest(session=sess)
        with self.assertRaises(HTTPException):
            require_auth_user(req, access_token_secret="s", oa_auth_enabled=False, get_user=lambda _: None)
        self.assertEqual(req.session, {})

    def test_require_auth_user_local_mode_success(self) -> None:
        req = _StubRequest(session={"username": "u1"})
        user = require_auth_user(
            req,
            access_token_secret="s",
            oa_auth_enabled=False,
            get_user=lambda _: AuthUser(username="u1", role="user"),
        )
        self.assertEqual(user.username, "u1")

    def test_is_admin_and_can_access_job(self) -> None:
        admin = AuthUser(username="a", role="admin")
        user = AuthUser(username="u1", role="user")
        job_own = _build_job("u1")
        job_other = _build_job("u2")
        self.assertTrue(is_admin(admin))
        self.assertFalse(is_admin(user))
        self.assertTrue(can_access_job(admin, job_other))
        self.assertTrue(can_access_job(user, job_own))
        self.assertFalse(can_access_job(user, job_other))


if __name__ == "__main__":
    unittest.main()


