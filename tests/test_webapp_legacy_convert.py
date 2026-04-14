import asyncio
import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from src.web.webapp_legacy_convert import handle_legacy_convert


class TestWebappLegacyConvert(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_legacy_convert_success(self) -> None:
        async def _ingest(*_args, **_kwargs):
            return "a" * 32

        def _wait(_jid, _get):
            return SimpleNamespace(status="succeeded", error_message=None)

        def _get(_jid):
            return SimpleNamespace(original_filename="x.pdf")

        out = self._run(
            handle_legacy_convert(
                request=SimpleNamespace(),
                file=SimpleNamespace(),
                ingest_uploads_create_job=_ingest,
                wait_job_terminal=_wait,
                get_job=_get,
            )
        )
        self.assertEqual(out["job_id"], "a" * 32)
        self.assertEqual(out["filename"], "x.pdf")
        self.assertEqual(out["download_url"], f"/download/{'a' * 32}")

    def test_legacy_convert_cancelled(self) -> None:
        async def _ingest(*_args, **_kwargs):
            return "b" * 32

        def _wait(_jid, _get):
            return SimpleNamespace(status="cancelled", error_message="任务已取消")

        with self.assertRaises(HTTPException) as ctx:
            self._run(
                handle_legacy_convert(
                    request=SimpleNamespace(),
                    file=SimpleNamespace(),
                    ingest_uploads_create_job=_ingest,
                    wait_job_terminal=_wait,
                    get_job=lambda _jid: None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("任务已取消", str(ctx.exception.detail))

    def test_legacy_convert_failed(self) -> None:
        async def _ingest(*_args, **_kwargs):
            return "c" * 32

        def _wait(_jid, _get):
            return SimpleNamespace(status="failed", error_message="转换失败: bad")

        with self.assertRaises(HTTPException) as ctx:
            self._run(
                handle_legacy_convert(
                    request=SimpleNamespace(),
                    file=SimpleNamespace(),
                    ingest_uploads_create_job=_ingest,
                    wait_job_terminal=_wait,
                    get_job=lambda _jid: None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("转换失败", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()


