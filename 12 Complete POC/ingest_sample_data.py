"""
ingest_sample_data.py — Seed Documents, Employees, and Projects

Run ONCE after setting up the project (or re-run safely — all inserts are idempotent).

Usage:
    python ingest_sample_data.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from passlib.context import CryptContext

from app.db.app_db import init_app_db, insert_employee, insert_project
from app.db.rate_limit_db import init_rate_limit_db
from app.db.sessions_db import init_sessions_db
from app.rag.ingestion import ingest_file
from app.models.schemas import AccessLevel

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


# ── Sample Employees ─────────────────────────────────────────────────────────

SAMPLE_EMPLOYEES = [
    {
        "employee_id": "emp_001",
        "name":        "Alice Johnson",
        "email":       "alice@company.com",
        "department":  "Engineering",
        "role":        "employee",
        "joining_date": "2022-03-15",
        "password":    "password123",
    },
    {
        "employee_id": "emp_002",
        "name":        "Bob Smith",
        "email":       "bob@company.com",
        "department":  "Engineering",
        "role":        "employee",
        "joining_date": "2021-07-01",
        "password":    "password123",
    },
    {
        "employee_id": "mgr_001",
        "name":        "Carol Williams",
        "email":       "carol@company.com",
        "department":  "Engineering",
        "role":        "manager",
        "joining_date": "2019-11-20",
        "password":    "password123",
    },
    {
        "employee_id": "mgr_002",
        "name":        "David Brown",
        "email":       "david@company.com",
        "department":  "Finance",
        "role":        "manager",
        "joining_date": "2020-04-10",
        "password":    "password123",
    },
    {
        "employee_id": "adm_001",
        "name":        "Eve Davis",
        "email":       "eve@company.com",
        "department":  "IT",
        "role":        "admin",
        "joining_date": "2018-06-05",
        "password":    "password123",
    },
    {
        "employee_id": "hr_001",
        "name":        "Frank Miller",
        "email":       "frank@company.com",
        "department":  "HR",
        "role":        "hr",
        "joining_date": "2020-01-15",
        "password":    "password123",
    },
]

# ── Sample Projects ──────────────────────────────────────────────────────────

SAMPLE_PROJECTS = [
    {
        "name":                "AI Knowledge Assistant",
        "start_date":          "2024-01-01",
        "end_date":            None,
        "status":              "active",
        "assignee_employee_id": "mgr_001",
    },
    {
        "name":                "Finance System Upgrade",
        "start_date":          "2023-06-01",
        "end_date":            "2024-03-31",
        "status":              "completed",
        "assignee_employee_id": "mgr_002",
    },
    {
        "name":                "Security Compliance Audit",
        "start_date":          "2024-03-01",
        "end_date":            None,
        "status":              "active",
        "assignee_employee_id": "adm_001",
    },
    {
        "name":                "Employee Onboarding Portal",
        "start_date":          "2024-02-15",
        "end_date":            "2024-12-31",
        "status":              "on-hold",
        "assignee_employee_id": "hr_001",
    },
]

# ── Sample Documents ─────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "file_path":    "data/documents/hr_handbook.txt",
        "title":        "HR Handbook 2024",
        "department":   "HR",
        "access_level": AccessLevel.PUBLIC,
        "doc_id":       "doc_hr_handbook",
    },
    {
        "file_path":    "data/documents/it_security_policy.txt",
        "title":        "IT Security Policy 2024",
        "department":   "IT",
        "access_level": AccessLevel.PUBLIC,
        "doc_id":       "doc_it_security",
    },
    {
        "file_path":    "data/documents/finance_policy.txt",
        "title":        "Finance and Expense Policy",
        "department":   "Finance",
        "access_level": AccessLevel.MANAGER,
        "doc_id":       "doc_finance_policy",
    },
    {
        "file_path":    "data/documents/executive_compensation.txt",
        "title":        "Executive Compensation Report FY2024",
        "department":   "Finance",
        "access_level": AccessLevel.CONFIDENTIAL,
        "doc_id":       "doc_exec_comp",
    },
]


def seed_databases():
    print("\n── Initialising databases ──────────────────────────────────")
    init_app_db()
    init_rate_limit_db()
    init_sessions_db()
    print("  app.db, rate_limit.db, sessions.db ready")

    print("\n── Seeding employees ───────────────────────────────────────")
    for emp in SAMPLE_EMPLOYEES:
        insert_employee(
            employee_id=emp["employee_id"],
            name=emp["name"],
            email=emp["email"],
            department=emp["department"],
            role=emp["role"],
            joining_date=emp["joining_date"],
            password_hash=hash_password(emp["password"]),
        )
        print(f"  {emp['employee_id']:10s}  {emp['name']:20s}  {emp['role']}")

    print("\n── Seeding projects ────────────────────────────────────────")
    for proj in SAMPLE_PROJECTS:
        insert_project(
            name=proj["name"],
            start_date=proj["start_date"],
            end_date=proj["end_date"],
            status=proj["status"],
            assignee_employee_id=proj["assignee_employee_id"],
        )
        print(f"  [{proj['status']:10s}]  {proj['name']}")


def seed_documents():
    print("\n── Ingesting documents ─────────────────────────────────────")
    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        print(f"  Ingesting: '{doc['title']}'")
        try:
            result = ingest_file(
                file_path=doc["file_path"],
                title=doc["title"],
                department=doc["department"],
                access_level=doc["access_level"],
                doc_id=doc["doc_id"],
            )
            print(f"    Chunks: {result['chunks_created']}  ✓")
            total_chunks += result["chunks_created"]
        except FileNotFoundError:
            print(f"    WARNING: File not found at {doc['file_path']} — skipping")
        except Exception as e:
            print(f"    ERROR: {e}")
    print(f"\n  Total chunks stored: {total_chunks}")


def main():
    print("=" * 60)
    print("  AI Knowledge Assistant — Data Seeding")
    print("=" * 60)

    seed_databases()
    seed_documents()

    print("\n" + "=" * 60)
    print("  Seeding complete!")
    print()
    print("  Test credentials:")
    for emp in SAMPLE_EMPLOYEES:
        print(f"    {emp['email']:30s}  password: {emp['password']}")
    print()
    print("  Next step: start the server with:")
    print("  uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
