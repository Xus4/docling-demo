from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from src.core.auth import AuthUser
from src.logging_utils import log_event


class _JobLike(Protocol):
    job_id: str
    status: str


class _AuthStoreLike(Protocol):
    def list_jobs(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None,
        status_filter: str | None,
        query: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[_JobLike], int]: ...

    def count_jobs_by_status(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None,
        query: str | None,
    ) -> dict[str, int]: ...

    def get_queue_positions(self, job_ids: list[str]) -> tuple[dict[str, int], int]: ...

    def count_queued_jobs(self) -> int: ...


class _JobToApiDictLike(Protocol):
    def __call__(
        self,
        job: _JobLike,
        *,
        queue_position: int | None,
        queue_total: int,
    ) -> dict[str, object | None]: ...


def list_jobs_payload(
    *,
    user: AuthUser,
    owner: str | None,
    status: str | None,
    q: str | None,
    page: int,
    page_size: int,
    auth_store: _AuthStoreLike,
    is_admin: Callable[[AuthUser], bool],
    job_to_api_dict: _JobToApiDictLike,
    log: logging.Logger,
) -> dict[str, object]:
    offset = (page - 1) * page_size
    items, total = auth_store.list_jobs(
        viewer_username=user.username,
        viewer_role=user.role,
        owner_filter=owner if is_admin(user) else None,
        status_filter=status,
        query=q,
        limit=page_size,
        offset=offset,
    )
    try:
        status_counts = auth_store.count_jobs_by_status(
            viewer_username=user.username,
            viewer_role=user.role,
            owner_filter=owner if is_admin(user) else None,
            query=q,
        )
        status_counts["all"] = sum(status_counts.values())
    except Exception as e:  # noqa: BLE001
        log_event(
            log,
            logging.WARNING,
            "jobs.status_counts.failed",
            err=repr(e),
        )
        status_counts = {"all": 0}

    queued_ids = [j.job_id for j in items if j.status == "queued"]
    q_map: dict[str, int] = {}
    q_total = 0
    try:
        if queued_ids:
            q_map, q_total = auth_store.get_queue_positions(queued_ids)
        else:
            q_total = auth_store.count_queued_jobs()
    except Exception as e:  # noqa: BLE001
        log_event(
            log,
            logging.WARNING,
            "jobs.queue_snapshot.failed",
            err=repr(e),
        )
        q_map, q_total = {}, 0
    return {
        "items": [
            job_to_api_dict(
                j,
                queue_position=q_map.get(j.job_id) if j.status == "queued" else None,
                queue_total=q_total,
            )
            for j in items
        ],
        "status_counts": status_counts,
        "total": total,
        "page": page,
        "page_size": page_size,
    }

