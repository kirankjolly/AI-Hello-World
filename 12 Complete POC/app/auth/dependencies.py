"""
app/auth/dependencies.py — FastAPI Auth Dependencies

Use as: current_user: dict = Depends(get_current_user)

The returned dict contains all JWT claims plus a fresh employee record
from the database (to pick up role/department changes without re-login).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth.jwt_handler import decode_token
from app.db.app_db import get_employee_by_id

bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Extract and validate JWT from Authorization: Bearer <token> header.

    Returns the employee record (augmented with JWT claims) as a dict.
    Raises HTTP 401 if token is invalid.
    Raises HTTP 403 if employee is inactive.
    """
    token = credentials.credentials
    claims = decode_token(token)

    user_id = claims.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user_id claim",
        )

    employee = get_employee_by_id(user_id)
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User '{user_id}' not found or inactive",
        )

    if not employee.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Merge JWT claims + fresh DB record
    return {**claims, **employee}
