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

    progress_percent: int | None = None
    progress_note: str | None = None
    progress_pages_done: int | None = None
    progress_pages_total: int | None = None
    current_file_name: str | None = None
    result_extra: str | None = None

    # 新增：目录 / 批量任务支持
    is_directory: int = 0
    input_root: str | None = None
    output_root: str | None = None
    total_files: int | None = None
    processed_files: int | None = None
    succeeded_files: int | None = None
    failed_files: int | None = None


class AuthStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"无法创建 SQLite 目录: {self.db_path.parent}") from exc
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        except sqlite3.OperationalError as exc:
            raise sqlite3.OperationalError(f"{exc} (db_path={self.db_path})") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
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
            self._migrate_jobs_progress_columns(conn)
            self._migrate_jobs_folder_columns(conn)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
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

    def ensure_env_admin_user(self, admin_username: str, initial_password: str) -> None:
        """OA 模式下仅维护一条本地管理员账号（密码来自 INITIAL_PASSWORD，与 bootstrap 哈希方式一致）。"""
        uname = admin_username.strip()
        if not uname:
            return
        h = _hash_password(initial_password)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username FROM users WHERE username = ?",
                (uname,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE users SET password_hash = ?, role = 'admin' WHERE username = ?",
                    (h, uname),
                )
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
                    (uname, h),
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

    def _migrate_jobs_progress_columns(self, conn: sqlite3.Connection) -> None:
        info = conn.execute("PRAGMA table_info(jobs)").fetchall()
        colnames = {str(r[1]) for r in info}
        stmts: list[str] = []
        if "progress_percent" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN progress_percent INTEGER")
        if "progress_note" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN progress_note TEXT")
        if "progress_pages_done" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN progress_pages_done INTEGER")
        if "progress_pages_total" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN progress_pages_total INTEGER")
        if "result_extra" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN result_extra TEXT")
        if "current_file_name" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN current_file_name TEXT")
        for sql in stmts:
            conn.execute(sql)

    def _migrate_jobs_folder_columns(self, conn: sqlite3.Connection) -> None:
        info = conn.execute("PRAGMA table_info(jobs)").fetchall()
        colnames = {str(r[1]) for r in info}
        stmts: list[str] = []
        if "is_directory" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN is_directory INTEGER NOT NULL DEFAULT 0")
        if "input_root" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN input_root TEXT")
        if "output_root" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN output_root TEXT")
        if "total_files" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN total_files INTEGER")
        if "processed_files" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN processed_files INTEGER")
        if "succeeded_files" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN succeeded_files INTEGER")
        if "failed_files" not in colnames:
            stmts.append("ALTER TABLE jobs ADD COLUMN failed_files INTEGER")
        for sql in stmts:
            conn.execute(sql)

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        keys = set(row.keys())
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
            progress_percent=(
                int(row["progress_percent"])
                if "progress_percent" in keys and row["progress_percent"] is not None
                else None
            ),
            progress_note=(
                str(row["progress_note"])
                if "progress_note" in keys and row["progress_note"] is not None
                else None
            ),
            progress_pages_done=(
                int(row["progress_pages_done"])
                if "progress_pages_done" in keys and row["progress_pages_done"] is not None
                else None
            ),
            progress_pages_total=(
                int(row["progress_pages_total"])
                if "progress_pages_total" in keys and row["progress_pages_total"] is not None
                else None
            ),
            current_file_name=(
                str(row["current_file_name"])
                if "current_file_name" in keys and row["current_file_name"] is not None
                else None
            ),
            result_extra=(
                str(row["result_extra"])
                if "result_extra" in keys and row["result_extra"] is not None
                else None
            ),
            is_directory=(
                int(row["is_directory"])
                if "is_directory" in keys and row["is_directory"] is not None
                else 0
            ),
            input_root=(
                str(row["input_root"])
                if "input_root" in keys and row["input_root"] is not None
                else None
            ),
            output_root=(
                str(row["output_root"])
                if "output_root" in keys and row["output_root"] is not None
                else None
            ),
            total_files=(
                int(row["total_files"])
                if "total_files" in keys and row["total_files"] is not None
                else None
            ),
            processed_files=(
                int(row["processed_files"])
                if "processed_files" in keys and row["processed_files"] is not None
                else None
            ),
            succeeded_files=(
                int(row["succeeded_files"])
                if "succeeded_files" in keys and row["succeeded_files"] is not None
                else None
            ),
            failed_files=(
                int(row["failed_files"])
                if "failed_files" in keys and row["failed_files"] is not None
                else None
            ),
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
        *,
        is_directory: int = 0,
        input_root: str | None = None,
        output_root: str | None = None,
        total_files: int | None = None,
        processed_files: int | None = None,
        succeeded_files: int | None = None,
        failed_files: int | None = None,
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_username, role_snapshot, original_filename,
                    status, input_file, output_file, created_at,
                    is_directory, input_root, output_root,
                    total_files, processed_files, succeeded_files, failed_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    int(is_directory),
                    input_root,
                    output_root,
                    total_files,
                    processed_files,
                    succeeded_files,
                    failed_files,
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
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'running',
                    started_at = ?,
                    attempt_count = attempt_count + 1,
                    progress_percent = 0,
                    progress_note = '开始处理…',
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = NULL,
                    processed_files = COALESCE(processed_files, 0),
                    succeeded_files = COALESCE(succeeded_files, 0),
                    failed_files = COALESCE(failed_files, 0)
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_job_succeeded(
        self,
        job_id: str,
        output_file: str,
        *,
        result_extra: str | None = None,
    ) -> bool:
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'succeeded',
                    output_file = ?,
                    finished_at = ?,
                    error_message = NULL,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = ?
                WHERE job_id = ? AND status = 'running'
                """,
                (output_file, now, result_extra, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_job_failed(self, job_id: str, error_message: str) -> bool:
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'failed',
                    finished_at = ?,
                    error_message = ?,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = NULL
                WHERE job_id = ? AND status = 'running'
                """,
                (now, error_message[:4000], job_id),
            )
            conn.commit()
            return cur.rowcount > 0

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
                    output_file = NULL,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = NULL
                WHERE job_id = ?
                """,
                (now, msg, job_id),
            )
            conn.commit()

    def try_reset_job_queued(self, job_id: str) -> bool:
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'queued',
                    cancel_requested = 0,
                    started_at = NULL,
                    finished_at = NULL,
                    error_message = NULL,
                    created_at = ?,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    current_file_name = NULL,
                    result_extra = NULL,
                    processed_files = 0,
                    succeeded_files = 0,
                    failed_files = 0
                WHERE job_id = ? AND status IN ('failed', 'cancelled', 'succeeded')
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def try_mark_job_cancelled_queued(self, job_id: str) -> bool:
        return self.try_mark_job_cancelled(job_id)

    def try_mark_job_cancelled(self, job_id: str) -> bool:
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'cancelled',
                    finished_at = ?,
                    error_message = '已取消',
                    output_file = NULL,
                    cancel_requested = 0,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = NULL
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def reset_orphan_running_jobs_to_queued(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET
                    status = 'queued',
                    started_at = NULL,
                    cancel_requested = 0,
                    progress_percent = NULL,
                    progress_note = NULL,
                    progress_pages_done = NULL,
                    progress_pages_total = NULL,
                    result_extra = NULL
                WHERE status = 'running'
                """
            )
            conn.commit()
            return int(cur.rowcount) if cur.rowcount is not None else 0

    def set_job_cancel_requested(self, job_id: str) -> bool:
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
        query: str | None = None,
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

        if query and query.strip():
            where.append("LOWER(original_filename) LIKE ?")
            params.append("%" + query.strip().lower() + "%")

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

    def count_queued_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
            ).fetchone()
        return int(row["c"]) if row else 0

    def get_queue_position(self, job_id: str) -> tuple[int | None, int]:
        total = self.count_queued_jobs()
        job = self.get_job(job_id)
        if not job or job.status != "queued":
            return None, total
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) + 1 AS pos FROM jobs AS j2
                WHERE j2.status = 'queued' AND (
                    j2.created_at < ? OR (j2.created_at = ? AND j2.job_id < ?)
                )
                """,
                (job.created_at, job.created_at, job_id),
            ).fetchone()
        pos = int(row["pos"]) if row else 1
        return pos, total

    def update_job_progress(
        self,
        job_id: str,
        *,
        percent: int | None = None,
        note: str | None = None,
        pages_done: int | None = None,
        pages_total: int | None = None,
        current_file_name: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[object] = []
        if percent is not None:
            sets.append("progress_percent = ?")
            params.append(max(0, min(100, int(percent))))
        if note is not None:
            sets.append("progress_note = ?")
            params.append(str(note)[:500])
        if pages_done is not None:
            sets.append("progress_pages_done = ?")
            params.append(int(pages_done))
        if pages_total is not None:
            sets.append("progress_pages_total = ?")
            params.append(int(pages_total))
        if current_file_name is not None:
            sets.append("current_file_name = ?")
            params.append(str(current_file_name)[:500])
        if not sets:
            return
        params.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
                params,
            )
            conn.commit()

    def update_job_file_counts(
        self,
        job_id: str,
        *,
        total_files: int | None = None,
        processed_files: int | None = None,
        succeeded_files: int | None = None,
        failed_files: int | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[object] = []
        if total_files is not None:
            sets.append("total_files = ?")
            params.append(int(total_files))
        if processed_files is not None:
            sets.append("processed_files = ?")
            params.append(int(processed_files))
        if succeeded_files is not None:
            sets.append("succeeded_files = ?")
            params.append(int(succeeded_files))
        if failed_files is not None:
            sets.append("failed_files = ?")
            params.append(int(failed_files))
        if not sets:
            return
        params.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
                params,
            )
            conn.commit()

    def list_queued_job_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM jobs
                WHERE status = 'queued'
                AND output_file IS NOT NULL
                AND output_file <> ''
                ORDER BY datetime(created_at) ASC
                """
            ).fetchall()
        return [str(r["job_id"]) for r in rows]

    def get_queue_positions(self, job_ids: list[str]) -> tuple[dict[str, int], int]:
        wanted = {str(x) for x in job_ids if x}
        if not wanted:
            return {}, self.count_queued_jobs()
        with self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
            ).fetchone()
            total = int(total_row["c"]) if total_row else 0
            if total <= 0:
                return {}, 0
            rows = conn.execute(
                """
                SELECT job_id FROM jobs
                WHERE status = 'queued'
                ORDER BY datetime(created_at) ASC, job_id ASC
                """
            ).fetchall()
        out: dict[str, int] = {}
        pos = 0
        for r in rows:
            jid = str(r["job_id"])
            pos += 1
            if jid in wanted:
                out[jid] = pos
                if len(out) == len(wanted):
                    break
        return out, total

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0
