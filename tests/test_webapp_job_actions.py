import tempfile
import unittest
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from src.web.webapp_job_actions import (
    cancel_job_payload,
    delete_job_payload,
    get_job_detail_payload,
    retry_job_payload,
)


def _job(
    *,
    status: str,
    owner: str = "u1",
    is_directory: int = 0,
    output_root: str | None = None,
    output_file: str | None = None,
    original_filename: str = "a.pdf",
):
    return SimpleNamespace(
        job_id="a" * 32,
        owner_username=owner,
        status=status,
        is_directory=is_directory,
        output_root=output_root,
        output_file=output_file,
        original_filename=original_filename,
    )


class _FakeAuthStore:
    def __init__(self, job=None) -> None:
        self.job = job
        self.cancel_ok = True
        self.reset_ok = True
        self.delete_ok = True
        self.reset_args = None

    def get_job(self, _jid: str):
        return self.job

    def try_mark_job_cancelled(self, _jid: str):
        return self.cancel_ok

    def try_reset_job_queued(self, jid: str, output_file: str | None = None):
        self.reset_args = (jid, output_file)
        return self.reset_ok

    def delete_job(self, _jid: str):
        return self.delete_ok


class _FakeWorker:
    def __init__(self) -> None:
        self.cancelled = []
        self.enqueued = []

    def cancel(self, jid: str):
        self.cancelled.append(jid)

    def enqueue(self, jid: str):
        self.enqueued.append(jid)


class TestWebappJobActions(unittest.TestCase):
    def test_get_job_detail_forbidden(self) -> None:
        user = SimpleNamespace(username="u1", role="user")
        store = _FakeAuthStore(job=_job(status="queued", owner="u2"))
        with self.assertRaises(HTTPException) as ctx:
            get_job_detail_payload(
                jid="a" * 32,
                user=user,
                auth_store=store,
                can_access_job=lambda _u, _j: False,
                job_to_api_dict=lambda _j: {},
            )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_cancel_running_success(self) -> None:
        user = SimpleNamespace(username="u1")
        store = _FakeAuthStore(job=_job(status="running", owner="u1"))
        worker = _FakeWorker()
        out = cancel_job_payload(
            jid="a" * 32,
            user=user,
            auth_store=store,
            can_access_job=lambda _u, _j: True,
            job_worker=worker,
            log=logging.getLogger("test_webapp_job_actions"),
        )
        self.assertEqual(out["status"], "cancelled")
        self.assertEqual(worker.cancelled, ["a" * 32])

    def test_cancel_conflict(self) -> None:
        user = SimpleNamespace(username="u1")
        store = _FakeAuthStore(job=_job(status="queued", owner="u1"))
        store.cancel_ok = False
        worker = _FakeWorker()
        with self.assertRaises(HTTPException) as ctx:
            cancel_job_payload(
                jid="a" * 32,
                user=user,
                auth_store=store,
                can_access_job=lambda _u, _j: True,
                job_worker=worker,
                log=logging.getLogger("test_webapp_job_actions"),
            )
        self.assertEqual(ctx.exception.status_code, 409)

    def test_retry_success_for_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            user = SimpleNamespace(username="u1")
            store = _FakeAuthStore(
                job=_job(
                    status="failed",
                    owner="u1",
                    is_directory=0,
                    output_file=None,
                    original_filename="x.pdf",
                )
            )
            worker = _FakeWorker()
            out = retry_job_payload(
                jid="a" * 32,
                user=user,
                auth_store=store,
                can_access_job=lambda _u, _j: True,
                job_worker=worker,
                output_dir_root=out_root,
                log=logging.getLogger("test_webapp_job_actions"),
            )
            self.assertEqual(out["status"], "queued")
            self.assertEqual(worker.enqueued, ["a" * 32])
            assert store.reset_args is not None
            self.assertEqual(store.reset_args[0], "a" * 32)
            self.assertTrue(str(store.reset_args[1]).endswith("x.md"))

    def test_retry_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            user = SimpleNamespace(username="u1")
            store = _FakeAuthStore(job=_job(status="failed", owner="u1"))
            store.reset_ok = False
            worker = _FakeWorker()
            with self.assertRaises(HTTPException) as ctx:
                retry_job_payload(
                    jid="a" * 32,
                    user=user,
                    auth_store=store,
                    can_access_job=lambda _u, _j: True,
                    job_worker=worker,
                    output_dir_root=out_root,
                    log=logging.getLogger("test_webapp_job_actions"),
                )
            self.assertEqual(ctx.exception.status_code, 409)

    def test_delete_running_rejected(self) -> None:
        user = SimpleNamespace(username="u1")
        store = _FakeAuthStore(job=_job(status="running", owner="u1"))
        cleaned = []
        with self.assertRaises(HTTPException) as ctx:
            delete_job_payload(
                jid="a" * 32,
                user=user,
                auth_store=store,
                can_access_job=lambda _u, _j: True,
                remove_job_workspace=lambda jid: cleaned.append(jid),
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(cleaned, [])


if __name__ == "__main__":
    unittest.main()


