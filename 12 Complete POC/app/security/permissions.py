"""
app/security/permissions.py — User Permission Model

[Concept: Security - Permission Aware RAG]

────────────────────────────────────────────────────────────────
WHY PERMISSION-BASED RETRIEVAL IS CRITICAL
────────────────────────────────────────────────────────────────
In a naive RAG system, ALL documents are retrieved without any
access control. This creates serious security problems:

  BAD EXAMPLE (naive RAG):
    Employee asks: "What is the company's financial forecast?"
    System retrieves: confidential board-level financial documents
    LLM answers with: sensitive financial data the employee shouldn't see

This is a data breach — caused entirely by the RAG pipeline
ignoring permissions.

CORRECT APPROACH:
  Before querying the vector database, determine which document
  access levels the user is allowed to see, then pass that as
  a FILTER to the vector search.

  This means even if a confidential document is semantically
  similar to the query, it will NEVER be returned to an
  unauthorized user.

Access Level Hierarchy:
  employee → can read: [public]
  manager  → can read: [public, manager]
  admin    → can read: [public, manager, confidential]
────────────────────────────────────────────────────────────────
"""

from typing import List, Optional, Dict
from app.models.schemas import UserRole, AccessLevel
from app.observability.logger import log_permission_denied, log_workflow_step


# ──────────────────────────────────────────────
# Simulated User Database
# In production this would be your auth system (JWT, OAuth, LDAP...)
# ──────────────────────────────────────────────

USERS: Dict[str, Dict] = {
    "emp_001": {"name": "Alice Johnson",  "role": UserRole.EMPLOYEE},
    "emp_002": {"name": "Bob Smith",      "role": UserRole.EMPLOYEE},
    "mgr_001": {"name": "Carol Williams", "role": UserRole.MANAGER},
    "mgr_002": {"name": "David Brown",    "role": UserRole.MANAGER},
    "adm_001": {"name": "Eve Davis",      "role": UserRole.ADMIN},
}


# ──────────────────────────────────────────────
# Permission Mapping
# Maps a user role to the list of access levels they can see
# ──────────────────────────────────────────────

ROLE_ACCESS_MAP: Dict[UserRole, List[str]] = {
    UserRole.EMPLOYEE: [AccessLevel.PUBLIC.value],
    UserRole.MANAGER:  [AccessLevel.PUBLIC.value, AccessLevel.MANAGER.value],
    UserRole.ADMIN:    [AccessLevel.PUBLIC.value, AccessLevel.MANAGER.value,
                        AccessLevel.CONFIDENTIAL.value],
}


def get_user(user_id: str) -> Optional[Dict]:
    """Retrieve user info. Returns None if user not found."""
    return USERS.get(user_id)


def get_user_role(user_id: str) -> Optional[UserRole]:
    """Return the role for a given user_id."""
    user = get_user(user_id)
    if user is None:
        return None
    return user["role"]


def get_allowed_access_levels(user_role: UserRole) -> List[str]:
    """
    Return the list of document access levels a user role can see.

    This list is used as a FILTER when querying ChromaDB.
    Only chunks with matching access_level metadata are returned.

    Example:
        get_allowed_access_levels(UserRole.MANAGER)
        → ["public", "manager"]
    """
    return ROLE_ACCESS_MAP.get(user_role, [])


def can_user_access_document(user_role: UserRole, doc_access_level: str) -> bool:
    """
    Check whether a specific role can access a specific document level.

    Used for post-retrieval validation to double-check results.
    """
    allowed = get_allowed_access_levels(user_role)
    allowed_check = doc_access_level in allowed

    if not allowed_check:
        log_permission_denied("unknown", user_role.value, doc_access_level)

    return allowed_check


def validate_user(user_id: str) -> tuple[bool, Optional[UserRole], str]:
    """
    Validate a user exists and return their role.

    Returns:
        (is_valid, role, error_message)
    """
    user = get_user(user_id)
    if user is None:
        return False, None, f"User '{user_id}' not found. Valid users: {list(USERS.keys())}"
    return True, user["role"], ""
