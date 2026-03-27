from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from dataclasses import dataclass
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
