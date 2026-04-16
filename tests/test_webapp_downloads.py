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
    job_id: str | None = None,
    status: str = "succeeded",
    output_root: str | None = None,
    output_file: str | None = None,
    owner: str = "u1",
    name: str = "doc.pdf",
):
    jid = job_id or ("a" * 32)
    return SimpleNamespace(
        job_id=jid,
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
                workspace_output_root=None,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(len(tasks.tasks), 1)

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
                    workspace_output_root=None,
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
                "a" * 32: _job(job_id="a" * 32, output_root=str(out1), name="a.pdf"),
                "b" * 32: _job(job_id="b" * 32, output_root=str(out2), name="b.pdf"),
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
                workspace_output_root=None,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(len(tasks.tasks), 2)

    def test_batch_falls_back_to_workspace_output_dir(self) -> None:
        """库中绝对路径失效时，使用当前 output 根目录 + job_id 定位任务目录。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "server_output"
            jid = "c" * 32
            real_dir = ws / jid
            real_dir.mkdir(parents=True)
            (real_dir / "out.md").write_text("x", encoding="utf-8")
            z = root / "pack.zip"
            z.write_bytes(b"zip")
            tasks = _FakeBackgroundTasks()
            jobs = {
                jid: _job(
                    job_id=jid,
                    output_root=str(root / "stale_missing_root"),
                    output_file=str(root / "stale_missing" / "out.md"),
                    name="legacy.pdf",
                ),
            }
            resp = build_batch_download_response(
                raw_ids=[jid],
                user=SimpleNamespace(username="u1", role="user"),
                normalize_job_id=lambda x: x,
                auth_store=_FakeAuthStore(jobs),
                can_access_job=lambda _u, _j: True,
                zip_job_output_folder=lambda _d: z,
                log=SimpleNamespace(exception=lambda *a, **k: None),
                background_tasks=tasks,
                workspace_output_root=ws,
            )
            self.assertIsInstance(resp, FileResponse)


if __name__ == "__main__":
    unittest.main()


