from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool


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
    mineru_backend: str | None = None
    mineru_task_id: str | None = None
    # 运行中细分子阶段（MinerU 解析 + 语义增强等），终态任务为 None
    processing_stage: str | None = None


class AuthStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        if not self.database_url:
            raise RuntimeError("DATABASE_URL 不能为空")
        self.engine = self._create_engine(self.database_url)
        self.dialect = str(self.engine.dialect.name or "").lower()
        self._ensure_sqlite_parent_dir()
        self._init_db()

    def _create_engine(self, database_url: str) -> Engine:
        if database_url.startswith("sqlite:"):
            return create_engine(
                database_url,
                poolclass=NullPool,
                connect_args={"timeout": 10, "check_same_thread": False},
            )
        return create_engine(database_url, pool_pre_ping=True)

    def _ensure_sqlite_parent_dir(self) -> None:
        if not self.database_url.startswith("sqlite:///"):
            return
        raw = self.database_url[len("sqlite:///") :]
        if raw == ":memory:":
            return
        try:
            p = Path(raw).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"无法创建数据库目录: {raw}") from exc

    def _init_db(self) -> None:
        with self.engine.begin() as conn:
            if self.dialect == "mysql":
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            username VARCHAR(255) NOT NULL UNIQUE,
                            password_hash TEXT NOT NULL,
                            role VARCHAR(32) NOT NULL DEFAULT 'user',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS jobs (
                            job_id VARCHAR(64) PRIMARY KEY,
                            owner_username VARCHAR(255) NOT NULL,
                            role_snapshot VARCHAR(32),
                            original_filename TEXT NOT NULL,
                            status VARCHAR(32) NOT NULL,
                            input_file TEXT NOT NULL,
                            output_file TEXT,
                            error_message TEXT,
                            cancel_requested INTEGER NOT NULL DEFAULT 0,
                            attempt_count INTEGER NOT NULL DEFAULT 0,
                            created_at VARCHAR(32) NOT NULL,
                            started_at VARCHAR(32),
                            finished_at VARCHAR(32),
                            progress_percent INTEGER,
                            progress_note TEXT,
                            progress_pages_done INTEGER,
                            progress_pages_total INTEGER,
                            result_extra TEXT,
                            current_file_name TEXT,
                            is_directory INTEGER NOT NULL DEFAULT 0,
                            input_root TEXT,
                            output_root TEXT,
                            total_files INTEGER,
                            processed_files INTEGER,
                            succeeded_files INTEGER,
                            failed_files INTEGER
                            ,
                            mineru_backend VARCHAR(64),
                            mineru_task_id VARCHAR(128),
                            processing_stage VARCHAR(64)
                        )
                        """
                    )
                )
            else:
                conn.execute(
                    text(
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
                )
                conn.execute(
                    text(
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
                            finished_at TEXT,
                            progress_percent INTEGER,
                            progress_note TEXT,
                            progress_pages_done INTEGER,
                            progress_pages_total INTEGER,
                            result_extra TEXT,
                            current_file_name TEXT,
                            is_directory INTEGER NOT NULL DEFAULT 0,
                            input_root TEXT,
                            output_root TEXT,
                            total_files INTEGER,
                            processed_files INTEGER,
                            succeeded_files INTEGER,
                            failed_files INTEGER
                            ,
                            mineru_backend TEXT,
                            mineru_task_id TEXT,
                            processing_stage TEXT
                        )
                        """
                    )
                )

            self._ensure_indexes(conn)
            self._migrate_jobs_columns(conn)

    def _ensure_indexes(self, conn) -> None:
        try:
            conn.execute(
                text(
                    "CREATE INDEX idx_jobs_owner_created ON jobs (owner_username, created_at)"
                )
            )
        except Exception:
            pass
        try:
            conn.execute(text("CREATE INDEX idx_jobs_status ON jobs (status)"))
        except Exception:
            pass

    def _migrate_jobs_columns(self, conn) -> None:
        insp = inspect(conn)
        try:
            cols = {c["name"] for c in insp.get_columns("jobs")}
        except Exception:
            cols = set()
        stmts: list[str] = []
        for name, ddl in [
            ("progress_percent", "ALTER TABLE jobs ADD COLUMN progress_percent INTEGER"),
            ("progress_note", "ALTER TABLE jobs ADD COLUMN progress_note TEXT"),
            ("progress_pages_done", "ALTER TABLE jobs ADD COLUMN progress_pages_done INTEGER"),
            ("progress_pages_total", "ALTER TABLE jobs ADD COLUMN progress_pages_total INTEGER"),
            ("result_extra", "ALTER TABLE jobs ADD COLUMN result_extra TEXT"),
            ("current_file_name", "ALTER TABLE jobs ADD COLUMN current_file_name TEXT"),
            ("is_directory", "ALTER TABLE jobs ADD COLUMN is_directory INTEGER NOT NULL DEFAULT 0"),
            ("input_root", "ALTER TABLE jobs ADD COLUMN input_root TEXT"),
            ("output_root", "ALTER TABLE jobs ADD COLUMN output_root TEXT"),
            ("total_files", "ALTER TABLE jobs ADD COLUMN total_files INTEGER"),
            ("processed_files", "ALTER TABLE jobs ADD COLUMN processed_files INTEGER"),
            ("succeeded_files", "ALTER TABLE jobs ADD COLUMN succeeded_files INTEGER"),
            ("failed_files", "ALTER TABLE jobs ADD COLUMN failed_files INTEGER"),
            ("mineru_backend", "ALTER TABLE jobs ADD COLUMN mineru_backend TEXT"),
            ("mineru_task_id", "ALTER TABLE jobs ADD COLUMN mineru_task_id TEXT"),
            ("processing_stage", "ALTER TABLE jobs ADD COLUMN processing_stage TEXT"),
        ]:
            if name not in cols:
                stmts.append(ddl)
        for sql in stmts:
            try:
                conn.execute(text(sql))
            except Exception:
                pass

    def bootstrap_users(
        self,
        users: list[str],
        initial_password: str,
        admin_username: str,
    ) -> None:
        all_users = {u.strip() for u in users if u.strip()}
        all_users.add(admin_username.strip())
        with self.engine.begin() as conn:
            for username in sorted(all_users):
                role = "admin" if username == admin_username else "user"
                exists = conn.execute(
                    text("SELECT 1 FROM users WHERE username = :u"),
                    {"u": username},
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    text(
                        "INSERT INTO users (username, password_hash, role) "
                        "VALUES (:u, :ph, :r)"
                    ),
                    {"u": username, "ph": _hash_password(initial_password), "r": role},
                )

    def ensure_env_admin_user(self, admin_username: str, initial_password: str) -> None:
        """OA 模式下仅维护一条本地管理员账号（密码来自 INITIAL_PASSWORD，与 bootstrap 哈希方式一致）。"""
        uname = admin_username.strip()
        if not uname:
            return
        h = _hash_password(initial_password)
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT 1 FROM users WHERE username = :u"),
                {"u": uname},
            ).fetchone()
            if row:
                conn.execute(
                    text(
                        "UPDATE users SET password_hash = :ph, role = 'admin' "
                        "WHERE username = :u"
                    ),
                    {"ph": h, "u": uname},
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO users (username, password_hash, role) "
                        "VALUES (:u, :ph, 'admin')"
                    ),
                    {"u": uname, "ph": h},
                )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT username, password_hash, role FROM users WHERE username = :u"
                    ),
                    {"u": username.strip()},
                )
                .mappings()
                .fetchone()
            )
        if not row:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return AuthUser(username=row["username"], role=row["role"])

    def get_user(self, username: str) -> AuthUser | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT username, role FROM users WHERE username = :u"),
                    {"u": username.strip()},
                )
                .mappings()
                .fetchone()
            )
        if not row:
            return None
        return AuthUser(username=row["username"], role=row["role"])

    def _row_to_job(self, row: Mapping[str, Any]) -> JobRecord:
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
            mineru_backend=(
                str(row["mineru_backend"])
                if "mineru_backend" in keys and row["mineru_backend"] is not None
                else None
            ),
            mineru_task_id=(
                str(row["mineru_task_id"])
                if "mineru_task_id" in keys and row["mineru_task_id"] is not None
                else None
            ),
            processing_stage=(
                str(row["processing_stage"]).strip()
                if "processing_stage" in keys and row["processing_stage"] is not None
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
        mineru_backend: str | None = None,
        mineru_task_id: str | None = None,
    ) -> None:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO jobs (
                        job_id, owner_username, role_snapshot, original_filename,
                        status, input_file, output_file, created_at,
                        is_directory, input_root, output_root,
                        total_files, processed_files, succeeded_files, failed_files
                        , mineru_backend, mineru_task_id
                    ) VALUES (
                        :job_id, :owner_username, :role_snapshot, :original_filename,
                        :status, :input_file, :output_file, :created_at,
                        :is_directory, :input_root, :output_root,
                        :total_files, :processed_files, :succeeded_files, :failed_files
                        , :mineru_backend, :mineru_task_id
                    )
                    """
                ),
                {
                    "job_id": job_id,
                    "owner_username": owner_username.strip(),
                    "role_snapshot": role_snapshot,
                    "original_filename": original_filename,
                    "status": status,
                    "input_file": input_file,
                    "output_file": output_file,
                    "created_at": now,
                    "is_directory": int(is_directory),
                    "input_root": input_root,
                    "output_root": output_root,
                    "total_files": total_files,
                    "processed_files": processed_files,
                    "succeeded_files": succeeded_files,
                    "failed_files": failed_files,
                    "mineru_backend": mineru_backend,
                    "mineru_task_id": mineru_task_id,
                },
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM jobs WHERE job_id = :job_id"),
                    {"job_id": job_id},
                )
                .mappings()
                .fetchone()
            )
        if not row:
            return None
        return self._row_to_job(row)

    def try_claim_job_running(self, job_id: str) -> bool:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'running',
                        started_at = :started_at,
                        attempt_count = attempt_count + 1,
                        progress_percent = 0,
                        progress_note = '开始处理…',
                        processing_stage = 'mineru_prepare',
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL,
                        processed_files = COALESCE(processed_files, 0),
                        succeeded_files = COALESCE(succeeded_files, 0),
                        failed_files = COALESCE(failed_files, 0)
                    WHERE job_id = :job_id AND status = 'queued'
                    """
                ),
                {"started_at": now, "job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def mark_job_succeeded(
        self,
        job_id: str,
        output_file: str,
        *,
        result_extra: str | None = None,
    ) -> bool:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'succeeded',
                        output_file = :output_file,
                        finished_at = :finished_at,
                        error_message = NULL,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = :result_extra
                    WHERE job_id = :job_id AND status = 'running'
                    """
                ),
                {
                    "output_file": output_file,
                    "finished_at": now,
                    "result_extra": result_extra,
                    "job_id": job_id,
                },
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def mark_job_failed(self, job_id: str, error_message: str) -> bool:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'failed',
                        finished_at = :finished_at,
                        error_message = :error_message,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL
                    WHERE job_id = :job_id AND status = 'running'
                    """
                ),
                {"finished_at": now, "error_message": error_message[:4000], "job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def mark_job_cancelled_finished(self, job_id: str, message: str | None = None) -> None:
        now = _utc_now_iso()
        msg = message or "已取消"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'cancelled',
                        finished_at = :finished_at,
                        error_message = :error_message,
                        output_file = NULL,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL
                    WHERE job_id = :job_id
                    """
                ),
                {"finished_at": now, "error_message": msg, "job_id": job_id},
            )

    def try_reset_job_queued(self, job_id: str, output_file: str | None = None) -> bool:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'queued',
                        cancel_requested = 0,
                        started_at = NULL,
                        finished_at = NULL,
                        error_message = NULL,
                        created_at = :created_at,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        current_file_name = NULL,
                        result_extra = NULL,
                        output_file = COALESCE(:output_file, output_file),
                        mineru_task_id = NULL,
                        processed_files = 0,
                        succeeded_files = 0,
                        failed_files = 0
                    WHERE job_id = :job_id AND status IN ('failed', 'cancelled', 'succeeded')
                    """
                ),
                {"created_at": now, "job_id": job_id, "output_file": output_file},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def set_job_mineru_task_id(self, job_id: str, task_id: str) -> None:
        tid = str(task_id or "").strip()
        if not tid:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE jobs SET mineru_task_id = :task_id
                    WHERE job_id = :job_id
                    """
                ),
                {"task_id": tid[:128], "job_id": job_id},
            )

    def try_mark_job_cancelled_queued(self, job_id: str) -> bool:
        return self.try_mark_job_cancelled(job_id)

    def try_mark_job_cancelled(self, job_id: str) -> bool:
        now = _utc_now_iso()
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'cancelled',
                        finished_at = :finished_at,
                        error_message = '已取消',
                        output_file = NULL,
                        cancel_requested = 0,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL
                    WHERE job_id = :job_id AND status IN ('queued', 'running')
                    """
                ),
                {"finished_at": now, "job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def reset_orphan_running_jobs_to_queued(self) -> int:
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'queued',
                        started_at = NULL,
                        cancel_requested = 0,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL
                    WHERE status = 'running'
                    """
                )
            )
            return int(cur.rowcount) if cur.rowcount is not None else 0

    def requeue_running_job_for_resume(self, job_id: str) -> bool:
        """将被中断的 running 任务回退到 queued，保留远端 task_id 以便恢复轮询。"""
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET
                        status = 'queued',
                        started_at = NULL,
                        finished_at = NULL,
                        cancel_requested = 0,
                        error_message = NULL,
                        progress_percent = NULL,
                        progress_note = NULL,
                        processing_stage = NULL,
                        progress_pages_done = NULL,
                        progress_pages_total = NULL,
                        result_extra = NULL
                    WHERE job_id = :job_id AND status = 'running'
                    """
                ),
                {"job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def set_job_cancel_requested(self, job_id: str) -> bool:
        with self.engine.begin() as conn:
            cur = conn.execute(
                text(
                    """
                    UPDATE jobs SET cancel_requested = 1
                    WHERE job_id = :job_id AND status = 'running'
                    """
                ),
                {"job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    def refresh_job_cancel_requested(self, job_id: str) -> int:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT cancel_requested FROM jobs WHERE job_id = :job_id"),
                    {"job_id": job_id},
                )
                .mappings()
                .fetchone()
            )
        return int(row["cancel_requested"]) if row else 0

    def list_usernames(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(text("SELECT username FROM users ORDER BY username"))
                .mappings()
                .fetchall()
            )
        return [str(r["username"]) for r in rows]

    def list_job_owner_usernames(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT DISTINCT owner_username
                        FROM jobs
                        WHERE owner_username IS NOT NULL
                          AND owner_username <> ''
                        ORDER BY owner_username
                        """
                    )
                )
                .mappings()
                .fetchall()
            )
        return [str(r["owner_username"]) for r in rows if r.get("owner_username")]

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
        params: dict[str, object] = {}

        if not is_admin:
            where.append("owner_username = :viewer_username")
            params["viewer_username"] = viewer_username
        elif owner_filter and owner_filter.strip():
            where.append("owner_username = :owner_filter")
            params["owner_filter"] = owner_filter.strip()

        if status_filter and status_filter.strip():
            where.append("status = :status_filter")
            params["status_filter"] = status_filter.strip()

        if query and query.strip():
            where.append("LOWER(original_filename) LIKE :q")
            params["q"] = "%" + query.strip().lower() + "%"

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with self.engine.connect() as conn:
            total_row = (
                conn.execute(text(f"SELECT COUNT(*) AS c FROM jobs{where_sql}"), params)
                .mappings()
                .fetchone()
            )
            total = int(total_row["c"]) if total_row else 0
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT * FROM jobs{where_sql}
                        ORDER BY
                            CASE
                                WHEN status = 'succeeded' THEN COALESCE(finished_at, created_at)
                                WHEN status = 'failed' THEN COALESCE(finished_at, started_at, created_at)
                                WHEN status = 'cancelled' THEN COALESCE(finished_at, started_at, created_at)
                                WHEN status = 'running' THEN COALESCE(started_at, created_at)
                                WHEN status = 'queued' THEN created_at
                                ELSE COALESCE(finished_at, started_at, created_at)
                            END DESC,
                            created_at DESC,
                            job_id DESC
                        LIMIT :limit OFFSET :offset
                        """
                    ),
                    {**params, "limit": limit, "offset": offset},
                )
                .mappings()
                .fetchall()
            )

        return [self._row_to_job(r) for r in rows], total

    def count_jobs_by_status(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None = None,
        query: str | None = None,
    ) -> dict[str, int]:
        is_admin = viewer_role == "admin"
        where: list[str] = []
        params: dict[str, object] = {}

        if not is_admin:
            where.append("owner_username = :viewer_username")
            params["viewer_username"] = viewer_username
        elif owner_filter and owner_filter.strip():
            where.append("owner_username = :owner_filter")
            params["owner_filter"] = owner_filter.strip()

        if query and query.strip():
            where.append("LOWER(original_filename) LIKE :q")
            params["q"] = "%" + query.strip().lower() + "%"

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT status, COUNT(*) AS c
                        FROM jobs{where_sql}
                        GROUP BY status
                        """
                    ),
                    params,
                )
                .mappings()
                .fetchall()
            )

        out: dict[str, int] = {}
        for r in rows:
            s = str(r.get("status") or "").strip()
            if not s:
                continue
            out[s] = int(r.get("c") or 0)
        return out

    def _job_viewer_filters(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None,
        query: str | None,
    ) -> tuple[list[str], dict[str, object]]:
        """与 list_jobs 一致的可见性与文件名筛选（不含 status）。"""
        is_admin = viewer_role == "admin"
        where: list[str] = []
        params: dict[str, object] = {}

        if not is_admin:
            where.append("owner_username = :viewer_username")
            params["viewer_username"] = viewer_username
        elif owner_filter and owner_filter.strip():
            where.append("owner_username = :owner_filter")
            params["owner_filter"] = owner_filter.strip()

        if query and query.strip():
            where.append("LOWER(original_filename) LIKE :q")
            params["q"] = "%" + query.strip().lower() + "%"

        return where, params

    def count_succeeded_jobs_in_time_range(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None = None,
        query: str | None = None,
        time_field: str,
        time_start: str,
        time_end: str,
    ) -> int:
        """统计已完成任务在指定时间字段区间内的数量（与列表筛选一致）。"""
        if time_field not in ("created_at", "finished_at"):
            raise ValueError("time_field 必须为 created_at 或 finished_at")
        where, params = self._job_viewer_filters(
            viewer_username=viewer_username,
            viewer_role=viewer_role,
            owner_filter=owner_filter,
            query=query,
        )
        where.append("status = 'succeeded'")
        where.append(f"{time_field} >= :time_start")
        where.append(f"{time_field} <= :time_end")
        params["time_start"] = time_start
        params["time_end"] = time_end
        if time_field == "finished_at":
            where.append("finished_at IS NOT NULL")
        where_sql = " WHERE " + " AND ".join(where)
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text(f"SELECT COUNT(*) AS c FROM jobs{where_sql}"),
                    params,
                )
                .mappings()
                .fetchone()
            )
        return int(row["c"]) if row else 0

    def list_succeeded_job_ids_in_time_range(
        self,
        *,
        viewer_username: str,
        viewer_role: str,
        owner_filter: str | None = None,
        query: str | None = None,
        time_field: str,
        time_start: str,
        time_end: str,
        limit: int | None = None,
    ) -> list[str]:
        """列出时间区间内已完成任务的 job_id，按时间字段降序；limit 为 None 时不限制条数。"""
        if time_field not in ("created_at", "finished_at"):
            raise ValueError("time_field 必须为 created_at 或 finished_at")
        where, params = self._job_viewer_filters(
            viewer_username=viewer_username,
            viewer_role=viewer_role,
            owner_filter=owner_filter,
            query=query,
        )
        where.append("status = 'succeeded'")
        where.append(f"{time_field} >= :time_start")
        where.append(f"{time_field} <= :time_end")
        params["time_start"] = time_start
        params["time_end"] = time_end
        if time_field == "finished_at":
            where.append("finished_at IS NOT NULL")
        where_sql = " WHERE " + " AND ".join(where)
        order_sql = f"SELECT job_id FROM jobs{where_sql} ORDER BY {time_field} DESC, job_id DESC"
        exec_params: dict[str, object] = {**params}
        if limit is not None:
            lim = max(1, int(limit))
            order_sql += " LIMIT :lim"
            exec_params["lim"] = lim
        with self.engine.connect() as conn:
            rows = (
                conn.execute(text(order_sql), exec_params).mappings().fetchall()
            )
        return [str(r["job_id"]) for r in rows]

    def count_queued_jobs(self) -> int:
        with self.engine.connect() as conn:
            row = (
                conn.execute(text("SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"))
                .mappings()
                .fetchone()
            )
        return int(row["c"]) if row else 0

    def get_queue_position(self, job_id: str) -> tuple[int | None, int]:
        total = self.count_queued_jobs()
        job = self.get_job(job_id)
        if not job or job.status != "queued":
            return None, total
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT COUNT(*) + 1 AS pos FROM jobs AS j2
                        WHERE j2.status = 'queued' AND (
                            j2.created_at < :created_at OR (j2.created_at = :created_at AND j2.job_id < :job_id)
                        )
                        """
                    ),
                    {"created_at": job.created_at, "job_id": job_id},
                )
                .mappings()
                .fetchone()
            )
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
        params: dict[str, object] = {}
        if percent is not None:
            sets.append("progress_percent = :progress_percent")
            params["progress_percent"] = max(0, min(100, int(percent)))
        if note is not None:
            sets.append("progress_note = :progress_note")
            params["progress_note"] = str(note)[:500]
        if pages_done is not None:
            sets.append("progress_pages_done = :progress_pages_done")
            params["progress_pages_done"] = int(pages_done)
        if pages_total is not None:
            sets.append("progress_pages_total = :progress_pages_total")
            params["progress_pages_total"] = int(pages_total)
        if current_file_name is not None:
            sets.append("current_file_name = :current_file_name")
            params["current_file_name"] = str(current_file_name)[:500]
        if not sets:
            return
        params["job_id"] = job_id
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = :job_id"),
                params,
            )

    def set_job_processing_stage(self, job_id: str, stage: str | None) -> None:
        """仅更新 ``processing_stage``（运行中任务）；``stage`` 为 None 时写入 NULL。"""
        raw = str(stage).strip()[:64] if stage else None
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE jobs SET processing_stage = :st "
                    "WHERE job_id = :job_id AND status = 'running'"
                ),
                {"st": raw, "job_id": job_id},
            )

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
        params: dict[str, object] = {}
        if total_files is not None:
            sets.append("total_files = :total_files")
            params["total_files"] = int(total_files)
        if processed_files is not None:
            sets.append("processed_files = :processed_files")
            params["processed_files"] = int(processed_files)
        if succeeded_files is not None:
            sets.append("succeeded_files = :succeeded_files")
            params["succeeded_files"] = int(succeeded_files)
        if failed_files is not None:
            sets.append("failed_files = :failed_files")
            params["failed_files"] = int(failed_files)
        if not sets:
            return
        params["job_id"] = job_id
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = :job_id"),
                params,
            )

    def list_queued_job_ids(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT job_id FROM jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        """
                    )
                )
                .mappings()
                .fetchall()
            )
        return [str(r["job_id"]) for r in rows]

    def get_queue_positions(self, job_ids: list[str]) -> tuple[dict[str, int], int]:
        wanted = {str(x) for x in job_ids if x}
        if not wanted:
            return {}, self.count_queued_jobs()
        with self.engine.connect() as conn:
            total_row = (
                conn.execute(text("SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"))
                .mappings()
                .fetchone()
            )
            total = int(total_row["c"]) if total_row else 0
            if total <= 0:
                return {}, 0
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT job_id FROM jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC, job_id ASC
                        """
                    )
                )
                .mappings()
                .fetchall()
            )
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
        with self.engine.begin() as conn:
            cur = conn.execute(
                text("DELETE FROM jobs WHERE job_id = :job_id"),
                {"job_id": job_id},
            )
            return bool(cur.rowcount and cur.rowcount > 0)
