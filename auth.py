from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _hash_password(password: str, salt: str | None = None) -> str:
    """使用 PBKDF2 对密码做安全哈希（不存明文）。"""
    if salt is None:
        salt = secrets.token_hex(16)
    iterations = 210000
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        iterations,
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt, digest = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        ).hex()
        return hmac.compare_digest(calc, digest)
    except Exception:
        return False


@dataclass
class AuthUser:
    username: str
    role: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class JobRecord:
    job_id: str
    owner_username: str
    role_snapshot: str | None
    original_filename: str
    status: str
    input_file: str
    output_file: str | None
    error_message: str | None
    cancel_requested: int
    attempt_count: int
    created_at: str
    started_at: str | None
    finished_at: str | None


class AuthStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_username TEXT NOT NULL,
                    role_snapshot TEXT,
                    original_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_file TEXT NOT NULL,
                    output_file TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_owner_created "
                "ON jobs (owner_username, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)"
            )
            conn.commit()

    def bootstrap_users(
        self,
        users: list[str],
        initial_password: str,
        admin_username: str,
    ) -> None:
        all_users = {u.strip() for u in users if u.strip()}
        all_users.add(admin_username.strip())
        with self._connect() as conn:
            for username in sorted(all_users):
                role = "admin" if username == admin_username else "user"
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, _hash_password(initial_password), role),
                )
            conn.commit()

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, role FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if not row:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return AuthUser(username=row["username"], role=row["role"])

    def get_user(self, username: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, role FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if not row:
            return None
        return AuthUser(username=row["username"], role=row["role"])

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            owner_username=row["owner_username"],
            role_snapshot=row["role_snapshot"],
            original_filename=row["original_filename"],
            status=row["status"],
            input_file=row["input_file"],
            output_file=row["output_file"],
            error_message=row["error_message"],
            cancel_requested=int(row["cancel_requested"] or 0),
            attempt_count=int(row["attempt_count"] or 0),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def insert_job(
        self,
        job_id: str,
        owner_username: str,
        role_snapshot: str | None,
        original_filename: str,
        input_file: str,
        output_file: str,
        status: str = "queued",
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_username, role_snapshot, original_filename,
                    status, input_file, output_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    owner_username.strip(),
                    role_snapshot,
                    original_filename,
                    status,
                    input_file,
                    output_file,
                    now,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def try_claim_job_running(self, job_id: str) -> bool:
        """仅当状态为 queued 时置为 running，返回是否成功认领。"""
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'running',
                    started_at = ?,
                    attempt_count = attempt_count + 1
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_job_succeeded(self, job_id: str, output_file: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'succeeded',
                    output_file = ?,
                    finished_at = ?,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (output_file, now, job_id),
            )
            conn.commit()

    def mark_job_failed(self, job_id: str, error_message: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'failed',
                    finished_at = ?,
                    error_message = ?
                WHERE job_id = ?
                """,
                (now, error_message[:4000], job_id),
            )
            conn.commit()

    def mark_job_cancelled_finished(self, job_id: str, message: str | None = None) -> None:
        now = _utc_now_iso()
        msg = message or "已取消"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'cancelled',
                    finished_at = ?,
                    error_message = ?,
                    output_file = NULL
                WHERE job_id = ?
                """,
                (now, msg, job_id),
            )
            conn.commit()

    def try_mark_job_cancelled_queued(self, job_id: str) -> bool:
        """排队中任务直接标记为已取消。"""
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'cancelled',
                    finished_at = ?,
                    error_message = '已取消'
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def set_job_cancel_requested(self, job_id: str) -> bool:
        """运行中任务请求软取消。"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET cancel_requested = 1
                WHERE job_id = ? AND status = 'running'
                """,
                (job_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def refresh_job_cancel_requested(self, job_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return int(row["cancel_requested"]) if row else 0

    def list_usernames(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username FROM users ORDER BY username COLLATE NOCASE"
            ).fetchall()
        return [str(r["username"]) for r in rows]

    def list_jobs(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None = None,
        status_filter: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        is_admin = viewer_role == "admin"
        where: list[str] = []
        params: list[object] = []

        if not is_admin:
            where.append("owner_username = ?")
            params.append(viewer_username)
        elif owner_filter and owner_filter.strip():
            where.append("owner_username = ?")
            params.append(owner_filter.strip())

        if status_filter and status_filter.strip():
            where.append("status = ?")
            params.append(status_filter.strip())

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS c FROM jobs{where_sql}", params
            ).fetchone()
            total = int(total_row["c"]) if total_row else 0
            rows = conn.execute(
                f"""
                SELECT * FROM jobs{where_sql}
                ORDER BY datetime(created_at) DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()

        return [self._row_to_job(r) for r in rows], total

    def list_queued_job_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM jobs
                WHERE status = 'queued'
                ORDER BY datetime(created_at) ASC
                """
            ).fetchall()
        return [str(r["job_id"]) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0
