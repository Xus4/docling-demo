import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.web.webapp_preview import (
    build_output_preview_response,
    build_source_preview_response,
)


class _FakeAuthStore:
    def __init__(self, jobs: dict[str, object]) -> None:
        self.jobs = jobs

    def get_job(self, jid: str):
        return self.jobs.get(jid)


def _job(
    *,
    job_id: str | None = None,
    status: str = "succeeded",
    input_file: str = "",
    input_root: str | None = None,
    output_file: str | None = None,
    output_root: str | None = None,
    owner: str = "u1",
    name: str = "doc.pdf",
):
    return SimpleNamespace(
        job_id=job_id or ("a" * 32),
        owner_username=owner,
        status=status,
        input_file=input_file,
        input_root=input_root,
        output_file=output_file,
        output_root=output_root,
        original_filename=name,
    )


class TestWebappPreview(unittest.TestCase):
    def test_source_preview_success_uses_job_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.pdf"
            src.write_bytes(b"%PDF")
            resp = build_source_preview_response(
                jid="a" * 32,
                user=SimpleNamespace(username="u1", role="user"),
                auth_store=_FakeAuthStore({"a" * 32: _job(input_file=str(src))}),
                can_access_job=lambda _u, _j: True,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(Path(resp.path), src)

    def test_output_preview_requires_succeeded_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.md"
            out.write_text("# pending", encoding="utf-8")
            with self.assertRaises(HTTPException) as ctx:
                build_output_preview_response(
                    jid="a" * 32,
                    user=SimpleNamespace(username="u1", role="user"),
                    auth_store=_FakeAuthStore(
                        {"a" * 32: _job(status="running", output_file=str(out))}
                    ),
                    can_access_job=lambda _u, _j: True,
                    workspace_output_root=None,
                )
            self.assertEqual(ctx.exception.status_code, 400)

    def test_output_preview_success_uses_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.md"
            out.write_text("# ok", encoding="utf-8")
            resp = build_output_preview_response(
                jid="a" * 32,
                user=SimpleNamespace(username="u1", role="user"),
                auth_store=_FakeAuthStore({"a" * 32: _job(output_file=str(out))}),
                can_access_job=lambda _u, _j: True,
                workspace_output_root=None,
            )
            self.assertIsInstance(resp, FileResponse)
            self.assertEqual(Path(resp.path), out)

    def test_output_preview_rejects_relative_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "safe.md").write_text("# safe", encoding="utf-8")
            (root / "secret.md").write_text("# secret", encoding="utf-8")
            with self.assertRaises(HTTPException) as ctx:
                build_output_preview_response(
                    jid="a" * 32,
                    user=SimpleNamespace(username="u1", role="user"),
                    auth_store=_FakeAuthStore(
                        {"a" * 32: _job(output_root=str(out_dir), output_file=None)}
                    ),
                    can_access_job=lambda _u, _j: True,
                    workspace_output_root=None,
                    relative_path="../secret.md",
                )
            self.assertEqual(ctx.exception.status_code, 400)

    def test_output_preview_can_select_nested_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            nested = out_dir / "nested" / "result.md"
            nested.parent.mkdir(parents=True)
            nested.write_text("# nested", encoding="utf-8")
            resp = build_output_preview_response(
                jid="a" * 32,
                user=SimpleNamespace(username="u1", role="user"),
                auth_store=_FakeAuthStore(
                    {"a" * 32: _job(output_root=str(out_dir), output_file=None)}
                ),
                can_access_job=lambda _u, _j: True,
                workspace_output_root=None,
                relative_path="nested/result.md",
            )
            self.assertEqual(Path(resp.path), nested)


if __name__ == "__main__":
    unittest.main()
