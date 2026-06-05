"""
app/auth/jwt_handler.py — JWT Token Management

Production-grade implementation using python-jose.
Token expiry is controlled by JWT_EXPIRE_DAYS (default 30 days for dev/testing).
In production, set JWT_EXPIRE_DAYS=1 or less and rotate JWT_SECRET_KEY regularly.

JWT Claims carried:
  user_id      — matches employee.employee_id (e.g. "emp_001")
  employee_id  — same as user_id (explicit alias for clarity)
  email        — employee email
  name         — full name
  role         — UserRole value (employee | manager | admin | hr)
  department   — employee department
  iat          — issued-at (Unix timestamp)
  exp          — expiry (Unix timestamp)
  jti          — unique token ID (UUID, allows future token revocation)
"""

import uuid
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import HTTPException, status

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_DAYS


# Warn loudly if secret is too short (weak)
if len(JWT_SECRET_KEY) < 32:
    import warnings
    warnings.warn(
        f"JWT_SECRET_KEY is only {len(JWT_SECRET_KEY)} chars. "
        "Use at least 32 characters in production.",
        stacklevel=1,
    )


def create_access_token(payload: dict) -> str:
    """
    Create a signed JWT token.

    Args:
        payload: dict with user claims (user_id, email, role, department, name, employee_id)

    Returns:
        Signed JWT string
    """
    now = datetime.now(timezone.utc)
    data = payload.copy()
    data.update({
        "iat": now,
        "exp": now + timedelta(days=JWT_EXPIRE_DAYS),
        "jti": str(uuid.uuid4()),
    })
    return jwt.encode(data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Raises:
        HTTPException(401) if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
