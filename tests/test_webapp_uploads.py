import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from src.core.auth import AuthUser
from src.core.service import ConversionJobPaths
from src.web.webapp_uploads import ingest_uploads_create_job


class _StubRequest:
    def __init__(self) -> None:
        self.headers = {}
        self.session = {}

    async def is_disconnected(self) -> bool:
        return False


class _StubUpload:
    def __init__(self, filename: str, chunks: list[bytes]) -> None:
        self.filename = filename
        self._chunks = list(chunks)
        self._idx = 0
        self.closed = False

    async def read(self, _size: int = -1) -> bytes:
        if self._idx >= len(self._chunks):
            return b""
        out = self._chunks[self._idx]
        self._idx += 1
        return out

    async def close(self) -> None:
        self.closed = True


class _FakeService:
    def __init__(self, root: Path, *, support_files: bool = True) -> None:
        self.root = root
        self.support_files = support_files
        self._counter = 0

    def validate_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in {"pdf", "docx", "xlsx", "png"}:
            raise RuntimeError("bad ext")

    def create_job_paths(self, entry_name: str, *, is_directory: bool = False) -> ConversionJobPaths:
        self._counter += 1
        job_id = f"job{self._counter:04d}"
        input_root = self.root / "input" / job_id
        output_root = self.root / "output" / job_id
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        safe_name = Path(entry_name).name or "upload"
        if is_directory:
            input_entry = input_root / safe_name
            output_entry = output_root / safe_name
        else:
            stem = Path(safe_name).stem or "result"
            input_entry = input_root / safe_name
            output_entry = output_root / f"{stem}.md"
        return ConversionJobPaths(
            job_id=job_id,
            input_root=input_root,
            output_root=output_root,
            input_entry=input_entry,
            output_entry=output_entry,
            is_directory=is_directory,
        )

    def iter_supported_files(self, input_root: Path):
        if not self.support_files:
            return iter(())
        files = [p for p in input_root.rglob("*") if p.is_file()]
        return iter(files)


class _FakeAuthStore:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    def insert_job(self, **kwargs) -> None:
        self.inserted.append(kwargs)


class _FakeJobWorker:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


class TestWebappUploads(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_single_file_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = _FakeService(root)
            auth_store = _FakeAuthStore()
            worker = _FakeJobWorker()
            req = _StubRequest()
            up = _StubUpload("a.pdf", [b"hello"])
            cleaned: list[str] = []

            job_id = self._run(
                ingest_uploads_create_job(
                    req,
                    file=up,
                    files=None,
                    relative_paths=None,
                    upload_kind="file",
                    root_name="",
                    require_auth_user=lambda _r: AuthUser(username="u1", role="user"),
                    safe_rel_path=lambda s: Path(s),
                    remove_job_workspace=lambda jid: cleaned.append(jid),
                    service=service,
                    auth_store=auth_store,
                    job_worker=worker,
                    max_file_size_bytes=1024 * 1024,
                    log=SimpleNamespace(
                        log=lambda *a, **k: None,
                        info=lambda *a, **k: None,
                        exception=lambda *a, **k: None,
                    ),
                    short_job_id=lambda x: x[:8],
                )
            )

            self.assertEqual(job_id, "job0001")
            self.assertEqual(worker.enqueued, ["job0001"])
            self.assertEqual(len(auth_store.inserted), 1)
            self.assertTrue(up.closed)
            self.assertEqual(cleaned, [])

    def test_folder_without_relative_paths_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = _FakeService(root)
            auth_store = _FakeAuthStore()
            worker = _FakeJobWorker()
            req = _StubRequest()
            up = _StubUpload("a.pdf", [b"hello"])

            with self.assertRaises(HTTPException) as ctx:
                self._run(
                    ingest_uploads_create_job(
                        req,
                        file=None,
                        files=[up],
                        relative_paths=None,
                        upload_kind="folder",
                        root_name="x",
                        require_auth_user=lambda _r: AuthUser(username="u1", role="user"),
                        safe_rel_path=lambda s: Path(s),
                        remove_job_workspace=lambda _jid: None,
                        service=service,
                        auth_store=auth_store,
                        job_worker=worker,
                        max_file_size_bytes=1024 * 1024,
                        log=SimpleNamespace(
                        log=lambda *a, **k: None,
                        info=lambda *a, **k: None,
                        exception=lambda *a, **k: None,
                    ),
                        short_job_id=lambda x: x[:8],
                    )
                )
            self.assertEqual(ctx.exception.status_code, 400)

    def test_folder_no_supported_files_triggers_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = _FakeService(root, support_files=False)
            auth_store = _FakeAuthStore()
            worker = _FakeJobWorker()
            req = _StubRequest()
            up = _StubUpload("a.pdf", [b"hello"])
            cleaned: list[str] = []

            with self.assertRaises(HTTPException) as ctx:
                self._run(
                    ingest_uploads_create_job(
                        req,
                        file=None,
                        files=[up],
                        relative_paths=["a.pdf"],
                        upload_kind="folder",
                        root_name="x",
                        require_auth_user=lambda _r: AuthUser(username="u1", role="user"),
                        safe_rel_path=lambda s: Path(s),
                        remove_job_workspace=lambda jid: cleaned.append(jid),
                        service=service,
                        auth_store=auth_store,
                        job_worker=worker,
                        max_file_size_bytes=1024 * 1024,
                        log=SimpleNamespace(
                        log=lambda *a, **k: None,
                        info=lambda *a, **k: None,
                        exception=lambda *a, **k: None,
                    ),
                        short_job_id=lambda x: x[:8],
                    )
                )
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(cleaned, ["job0001"])


if __name__ == "__main__":
    unittest.main()


