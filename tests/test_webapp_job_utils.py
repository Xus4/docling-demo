import unittest

from fastapi import HTTPException

from src.core.auth import JobRecord
from src.web.webapp_job_utils import (
    job_to_api_dict,
    normalize_job_id,
    safe_rel_path,
)


def _build_job(**overrides) -> JobRecord:
    base = JobRecord(
        job_id="a" * 32,
        owner_username="u1",
        role_snapshot="user",
        original_filename="a.pdf",
        status="queued",
        input_file="/tmp/in.pdf",
        output_file="/tmp/out.md",
        error_message=None,
        cancel_requested=0,
        attempt_count=1,
        created_at="2026-01-01T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        progress_percent=None,
        progress_note=None,
        progress_pages_done=None,
        progress_pages_total=None,
        current_file_name=None,
        result_extra=None,
        is_directory=0,
        input_root=None,
        output_root=None,
        total_files=1,
        processed_files=0,
        succeeded_files=0,
        failed_files=0,
        mineru_backend=None,
        mineru_task_id=None,
        processing_stage=None,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestWebappJobUtils(unittest.TestCase):
    def test_normalize_job_id_valid(self) -> None:
        self.assertEqual(normalize_job_id("a" * 32), "a" * 32)

    def test_normalize_job_id_invalid(self) -> None:
        with self.assertRaises(HTTPException):
            normalize_job_id("bad-id")

    def test_safe_rel_path_allows_nested_relative(self) -> None:
        p = safe_rel_path("foo/bar/a.pdf")
        self.assertEqual(p.as_posix(), "foo/bar/a.pdf")

    def test_safe_rel_path_rejects_parent(self) -> None:
        with self.assertRaises(HTTPException):
            safe_rel_path("../x.pdf")

    def test_job_to_api_dict_extracts_failed_files_preview(self) -> None:
        job = _build_job(
            status="succeeded",
            result_extra=(
                '{"failed_files":[{"file":"a.pdf","error":"boom"},'
                '{"file":"b.pdf","error":"oops"}]}'
            ),
        )
        out = job_to_api_dict(job)
        preview = out["failed_files_preview"]
        self.assertIsInstance(preview, list)
        assert isinstance(preview, list)
        self.assertEqual(preview[0]["file"], "a.pdf")
        self.assertEqual(preview[1]["error"], "oops")

    def test_job_to_api_dict_processing_stage_running(self) -> None:
        job = _build_job(
            status="running",
            progress_percent=50,
            progress_note="正在解析中…",
            processing_stage="mineru_remote",
        )
        out = job_to_api_dict(job)
        self.assertEqual(out["processing_stage"], "mineru_remote")

    def test_job_to_api_dict_processing_stage_cleared_when_succeeded(self) -> None:
        job = _build_job(status="succeeded", processing_stage="semantic_enhance")
        out = job_to_api_dict(job)
        self.assertIsNone(out["processing_stage"])


if __name__ == "__main__":
    unittest.main()


