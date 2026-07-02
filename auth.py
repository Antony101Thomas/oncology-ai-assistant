# auth.py

import os
import bcrypt
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from database import get_user_by_id, get_user_by_email

# ── Config ────────────────────────────────────────────────────────
SECRET_KEY         = os.getenv("JWT_SECRET", "change-this-in-production")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")

# ── Password hashing ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password[:72].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain[:72].encode("utf-8"), hashed.encode("utf-8"))

# ── JWT ───────────────────────────────────────────────────────────
def create_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Password-reset tokens are single-purpose and must never be usable
        # as a regular auth token.
        if payload.get("purpose") is not None:
            raise JWTError("Not a valid auth token")
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ── Password reset tokens ──────────────────────────────────────────
# Short-lived, single-purpose JWTs used for the "forgot password" flow.
# No separate database table is needed — the token itself carries the
# user id and an expiry, and is only ever valid for password resets
# (never accepted as a normal auth token, see get_current_user above).
RESET_TOKEN_EXPIRE_MINUTES = 30

def create_reset_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "purpose": "password_reset", "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

def verify_reset_token(token: str) -> int:
    """Decode a password-reset token and return the user_id, or raise ValueError."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise ValueError("This reset link is invalid or has expired.")
    if payload.get("purpose") != "password_reset":
        raise ValueError("This reset link is invalid.")
    try:
        return int(payload.get("sub"))
    except (TypeError, ValueError):
        raise ValueError("This reset link is invalid.")

# ── Google OAuth ──────────────────────────────────────────────────
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)