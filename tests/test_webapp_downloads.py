import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.web.webapp_downloads import (
    build_batch_download_response,
    build_single_download_response,
)


class _FakeBackgroundTasks:
    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, fn):
        self.tasks.append(fn)


class _FakeAuthStore:
    def __init__(self, jobs: dict[str, object]) -> None:
        self.jobs = jobs

    def get_job(self, jid: str):
        return self.jobs.get(jid)


def _job(
    *,
    status: str = "succeeded",
    output_root: str | None = None,
    output_file: str | None = None,
    owner: str = "u1",
    name: str = "doc.pdf",
):
    return SimpleNamespace(
        job_id="a" * 32,
        owner_username=owner,
        status=status,
        output_root=output_root,
        output_file=output_file,
        original_filename=name,
    )


class TestWebappDownloads(unittest.TestCase):
    def test_single_missing_job_404(self) -> None:
        tasks = _FakeBackgroundTasks()
        with self.assertRaises(HTTPException) as ctx:
            build_single_download_response(
                jid="a" * 32,
                user=SimpleNamespace(username="u1", role="user"),
                auth_store=_FakeAuthStore({}),
                can_access_job=lambda _u, _j: True,
                zip_job_output_folder=lambda _d: Path("x.zip"),
                short_job_id=lambda x: x[:8],
                log=SimpleNamespace(exception=lambda *a, **k: None),
                background_tasks=tasks,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_single_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            out = d / "out"
            out.mkdir()
            (out / "a.md").write_text("ok", encoding="utf-8")
            z = d / "x.zip"
            z.write_bytes(b"zip")
            tasks = _FakeBackgroundTasks()
            resp = build_single_download_response(
                jid="a" * 32,
                user=SimpleNamespace(username="u1", role="user"),
                auth_store=_FakeAuthStore({"a" * 32: _job(output_root=str(out))}),
                can_access_job=lambda _u, _j: True,
                zip_job_output_folder=lambda _d: z,
                short_job_id=lambda x: x[:8],
                log=SimpleNamespace(exception=lambda *a, **k: None),
                background_tasks=tasks,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(len(tasks.tasks), 1)

    def test_batch_limit_exceeded(self) -> None:
        tasks = _FakeBackgroundTasks()
        raw_ids = [f"{i:032x}" for i in range(501)]
        with self.assertRaises(HTTPException) as ctx:
            build_batch_download_response(
                raw_ids=raw_ids,
                user=SimpleNamespace(username="u1", role="user"),
                normalize_job_id=lambda x: x,
                auth_store=_FakeAuthStore({}),
                can_access_job=lambda _u, _j: True,
                zip_job_output_folder=lambda _d: Path("x.zip"),
                log=SimpleNamespace(exception=lambda *a, **k: None),
                background_tasks=tasks,
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_batch_forbidden(self) -> None:
        tasks = _FakeBackgroundTasks()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            jobs = {"a" * 32: _job(output_root=str(out))}
            with self.assertRaises(HTTPException) as ctx:
                build_batch_download_response(
                    raw_ids=["a" * 32],
                    user=SimpleNamespace(username="u1", role="user"),
                    normalize_job_id=lambda x: x,
                    auth_store=_FakeAuthStore(jobs),
                    can_access_job=lambda _u, _j: False,
                    zip_job_output_folder=lambda _d: Path("x.zip"),
                    log=SimpleNamespace(exception=lambda *a, **k: None),
                    background_tasks=tasks,
                )
            self.assertEqual(ctx.exception.status_code, 403)

    def test_batch_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out1 = root / "o1"
            out1.mkdir()
            (out1 / "a.md").write_text("ok", encoding="utf-8")
            out2 = root / "o2"
            out2.mkdir()
            (out2 / "b.md").write_text("ok", encoding="utf-8")
            z = root / "z.zip"
            z.write_bytes(b"zip")
            tasks = _FakeBackgroundTasks()
            jobs = {
                "a" * 32: _job(output_root=str(out1), name="a.pdf"),
                "b" * 32: _job(output_root=str(out2), name="b.pdf"),
            }
            resp = build_batch_download_response(
                raw_ids=["a" * 32, "b" * 32],
                user=SimpleNamespace(username="u1", role="user"),
                normalize_job_id=lambda x: x,
                auth_store=_FakeAuthStore(jobs),
                can_access_job=lambda _u, _j: True,
                zip_job_output_folder=lambda _d: z,
                log=SimpleNamespace(exception=lambda *a, **k: None),
                background_tasks=tasks,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(len(tasks.tasks), 2)


if __name__ == "__main__":
    unittest.main()


