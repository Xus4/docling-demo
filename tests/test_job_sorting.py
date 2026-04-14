import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from src.core.auth import AuthStore


class TestJobSorting(unittest.TestCase):
    def _create_store(self) -> AuthStore:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "jobs.db"
        return AuthStore(f"sqlite:///{db_path}")

    def tearDown(self) -> None:
        tmp = getattr(self, "_tmpdir", None)
        if tmp is not None:
            tmp.cleanup()
        super().tearDown()

    def test_list_jobs_sorts_by_status_specific_time_desc(self) -> None:
        store = self._create_store()

        def add_job(job_id: str, created_at: str, status: str, started_at: str | None, finished_at: str | None) -> None:
            store.insert_job(
                job_id=job_id,
                owner_username="u1",
                role_snapshot="user",
                original_filename=f"{job_id}.pdf",
                input_file=f"/tmp/{job_id}.pdf",
                output_file=f"/tmp/{job_id}.zip",
            )
            with store.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE jobs
                        SET status = :status, created_at = :created_at, started_at = :started_at, finished_at = :finished_at
                        WHERE job_id = :job_id
                        """
                    ),
                    {
                        "job_id": job_id,
                        "status": status,
                        "created_at": created_at,
                        "started_at": started_at,
                        "finished_at": finished_at,
                    },
                )

        add_job("queued_old", "2026-01-01T10:00:00+00:00", "queued", None, None)
        add_job("queued_new", "2026-01-01T11:00:00+00:00", "queued", None, None)
        add_job("running_old", "2026-01-01T08:00:00+00:00", "running", "2026-01-01T09:00:00+00:00", None)
        add_job("running_new", "2026-01-01T07:00:00+00:00", "running", "2026-01-01T10:30:00+00:00", None)
        add_job("succeeded_old", "2026-01-01T06:00:00+00:00", "succeeded", "2026-01-01T06:05:00+00:00", "2026-01-01T12:00:00+00:00")
        add_job("succeeded_new", "2026-01-01T05:00:00+00:00", "succeeded", "2026-01-01T05:05:00+00:00", "2026-01-01T13:00:00+00:00")
        add_job("failed_old", "2026-01-01T04:00:00+00:00", "failed", "2026-01-01T04:05:00+00:00", "2026-01-01T14:00:00+00:00")
        add_job("failed_new", "2026-01-01T03:00:00+00:00", "failed", "2026-01-01T03:05:00+00:00", "2026-01-01T15:00:00+00:00")
        add_job("cancelled_old", "2026-01-01T02:00:00+00:00", "cancelled", "2026-01-01T02:05:00+00:00", "2026-01-01T16:00:00+00:00")
        add_job("cancelled_new", "2026-01-01T01:00:00+00:00", "cancelled", "2026-01-01T01:05:00+00:00", "2026-01-01T17:00:00+00:00")

        all_items, _ = store.list_jobs(
            viewer_username="admin",
            viewer_role="admin",
            limit=50,
            offset=0,
        )
        all_ids = [j.job_id for j in all_items]
        self.assertEqual(
            all_ids,
            [
                "cancelled_new",
                "cancelled_old",
                "failed_new",
                "failed_old",
                "succeeded_new",
                "succeeded_old",
                "queued_new",
                "running_new",
                "queued_old",
                "running_old",
            ],
        )

        done_items, _ = store.list_jobs(
            viewer_username="admin",
            viewer_role="admin",
            status_filter="succeeded",
            limit=50,
            offset=0,
        )
        self.assertEqual([j.job_id for j in done_items], ["succeeded_new", "succeeded_old"])

        queued_items, _ = store.list_jobs(
            viewer_username="admin",
            viewer_role="admin",
            status_filter="queued",
            limit=50,
            offset=0,
        )
        self.assertEqual([j.job_id for j in queued_items], ["queued_new", "queued_old"])


if __name__ == "__main__":
    unittest.main()
