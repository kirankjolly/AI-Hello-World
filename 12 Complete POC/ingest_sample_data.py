"""
ingest_sample_data.py — Load sample documents into ChromaDB

Run this ONCE after setting up the project to populate the vector database
with the sample company documents.

Usage:
    python ingest_sample_data.py
"""

import sys
import os

# Make sure we can import from the app package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.ingestion import ingest_file
from app.models.schemas import AccessLevel


def main():
    print("=" * 60)
    print("  AI Knowledge Assistant — Document Ingestion")
    print("=" * 60)
    print()

    # ──────────────────────────────────────────────
    # Documents to ingest
    # ──────────────────────────────────────────────
    documents = [
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
            "access_level": AccessLevel.MANAGER,    # Only managers can see this
            "doc_id":       "doc_finance_policy",
        },
        {
            "file_path":    "data/documents/executive_compensation.txt",
            "title":        "Executive Compensation Report FY2024",
            "department":   "Finance",
            "access_level": AccessLevel.CONFIDENTIAL,  # Only admins
            "doc_id":       "doc_exec_comp",
        },
    ]

    # ──────────────────────────────────────────────
    # Ingest each document
    # ──────────────────────────────────────────────
    total_chunks = 0

    for doc in documents:
        print(f"Ingesting: '{doc['title']}'")
        print(f"  Department:   {doc['department']}")
        print(f"  Access Level: {doc['access_level'].value}")

        try:
            result = ingest_file(
                file_path=doc["file_path"],
                title=doc["title"],
                department=doc["department"],
                access_level=doc["access_level"],
                doc_id=doc["doc_id"],
            )
            print(f"  Chunks Created: {result['chunks_created']}")
            print(f"  Status: OK")
            total_chunks += result['chunks_created']

        except FileNotFoundError:
            print(f"  WARNING: File not found at {doc['file_path']}")
            print(f"  Skipping this document.")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print("=" * 60)
    print(f"  Ingestion complete!")
    print(f"  Total chunks stored: {total_chunks}")
    print(f"  Vector DB: ./data/chroma_db")
    print()
    print("  Next step: start the server with:")
    print("  uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
