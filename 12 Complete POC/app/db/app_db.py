"""
app/db/app_db.py — Application SQLite Database

Stores: employees (users), projects
Future: Replace with PostgreSQL by swapping get_app_db_connection()
        and the SQL dialect in init_app_db().

Database file: data/app.db
"""

import sqlite3
from contextlib import contextmanager
from app.config import APP_DB_PATH


@contextmanager
def get_app_db_connection():
    """Context manager for app.db connections. Use WAL mode for concurrency."""
    conn = sqlite3.connect(APP_DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_app_db() -> None:
    """
    Create tables if they don't exist.
    Safe to call on every startup — uses IF NOT EXISTS.
    """
    with get_app_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id   TEXT    NOT NULL UNIQUE,   -- e.g. "emp_001"
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                department    TEXT    NOT NULL,
                role          TEXT    NOT NULL,          -- employee | manager | admin | hr
                joining_date  TEXT    NOT NULL,          -- ISO date string YYYY-MM-DD
                password_hash TEXT    NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                start_date  TEXT    NOT NULL,            -- ISO date string
                end_date    TEXT,                        -- NULL if ongoing
                status      TEXT    NOT NULL DEFAULT 'active',  -- active | completed | on-hold
                assignee_id INTEGER REFERENCES employees(id) ON DELETE SET NULL
            );
        """)


# ── CRUD Helpers ────────────────────────────────────────────


def get_employee_by_email(email: str) -> dict | None:
    with get_app_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM employees WHERE email = ? AND is_active = 1", (email,)
        ).fetchone()
        return dict(row) if row else None


def get_employee_by_id(employee_id: str) -> dict | None:
    with get_app_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ? AND is_active = 1", (employee_id,)
        ).fetchone()
        return dict(row) if row else None


def list_employees() -> list[dict]:
    with get_app_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, employee_id, name, email, department, role, joining_date, is_active "
            "FROM employees ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]


def list_projects() -> list[dict]:
    with get_app_db_connection() as conn:
        rows = conn.execute("""
            SELECT p.id, p.name, p.start_date, p.end_date, p.status,
                   e.name AS assignee_name, e.employee_id AS assignee_employee_id
            FROM projects p
            LEFT JOIN employees e ON p.assignee_id = e.id
            ORDER BY p.start_date DESC
        """).fetchall()
        return [dict(r) for r in rows]


def insert_employee(employee_id: str, name: str, email: str, department: str,
                    role: str, joining_date: str, password_hash: str) -> None:
    with get_app_db_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO employees
                (employee_id, name, email, department, role, joining_date, password_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (employee_id, name, email, department, role, joining_date, password_hash))


def insert_project(name: str, start_date: str, end_date: str | None,
                   status: str, assignee_employee_id: str | None) -> None:
    with get_app_db_connection() as conn:
        assignee_db_id = None
        if assignee_employee_id:
            row = conn.execute(
                "SELECT id FROM employees WHERE employee_id = ?", (assignee_employee_id,)
            ).fetchone()
            assignee_db_id = row["id"] if row else None

        conn.execute("""
            INSERT INTO projects (name, start_date, end_date, status, assignee_id)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM projects WHERE name = ?)
        """, (name, start_date, end_date, status, assignee_db_id, name))
