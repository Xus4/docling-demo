import unittest
import logging
from types import SimpleNamespace

from src.core.auth import AuthUser
from src.web.webapp_jobs_view import list_jobs_payload


class _FakeAuthStore:
    def __init__(self) -> None:
        self.last_owner_filter = None
        self.raise_status_counts = False
        self.raise_queue_positions = False

    def list_jobs(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter,
        status_filter,
        query,
        limit: int,
        offset: int,
    ):
        _ = (viewer_username, viewer_role, status_filter, query, limit, offset)
        self.last_owner_filter = owner_filter
        items = [
            SimpleNamespace(job_id="j1", status="queued"),
            SimpleNamespace(job_id="j2", status="succeeded"),
        ]
        return items, 2

    def count_jobs_by_status(self, **kwargs):
        _ = kwargs
        if self.raise_status_counts:
            raise RuntimeError("status fail")
        return {"queued": 1, "succeeded": 1}

    def get_queue_positions(self, job_ids):
        _ = job_ids
        if self.raise_queue_positions:
            raise RuntimeError("queue fail")
        return {"j1": 3}, 10

    def count_queued_jobs(self):
        return 10


class TestWebappJobsView(unittest.TestCase):
    def test_owner_filter_only_for_admin(self) -> None:
        store = _FakeAuthStore()
        user = AuthUser(username="u1", role="user")
        payload = list_jobs_payload(
            user=user,
            owner="u2",
            status=None,
            q=None,
            page=1,
            page_size=20,
            auth_store=store,
            is_admin=lambda u: u.role == "admin",
            job_to_api_dict=lambda j, queue_position, queue_total: {
                "job_id": j.job_id,
                "queue_position": queue_position,
                "queue_total": queue_total,
            },
            log=logging.getLogger("test_webapp_jobs_view"),
        )
        self.assertEqual(store.last_owner_filter, None)
        self.assertEqual(payload["total"], 2)

    def test_status_counts_failure_fallback(self) -> None:
        store = _FakeAuthStore()
        store.raise_status_counts = True
        admin = AuthUser(username="admin", role="admin")
        payload = list_jobs_payload(
            user=admin,
            owner="u2",
            status=None,
            q=None,
            page=1,
            page_size=20,
            auth_store=store,
            is_admin=lambda u: u.role == "admin",
            job_to_api_dict=lambda j, queue_position, queue_total: {
                "job_id": j.job_id,
                "queue_position": queue_position,
                "queue_total": queue_total,
            },
            log=logging.getLogger("test_webapp_jobs_view"),
        )
        self.assertEqual(store.last_owner_filter, "u2")
        self.assertEqual(payload["status_counts"], {"all": 0})

    def test_queue_failure_fallback(self) -> None:
        store = _FakeAuthStore()
        store.raise_queue_positions = True
        admin = AuthUser(username="admin", role="admin")
        payload = list_jobs_payload(
            user=admin,
            owner=None,
            status=None,
            q=None,
            page=1,
            page_size=20,
            auth_store=store,
            is_admin=lambda u: u.role == "admin",
            job_to_api_dict=lambda j, queue_position, queue_total: {
                "job_id": j.job_id,
                "queue_position": queue_position,
                "queue_total": queue_total,
            },
            log=logging.getLogger("test_webapp_jobs_view"),
        )
        items = payload["items"]
        assert isinstance(items, list)
        self.assertEqual(items[0]["job_id"], "j1")
        self.assertEqual(items[0]["queue_position"], None)
        self.assertEqual(items[0]["queue_total"], 0)


if __name__ == "__main__":
    unittest.main()


