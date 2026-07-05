# auth.py

import os
import secrets
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

# ── Two-step verification (2FA) tokens ─────────────────────────────
# "login_2fa" tokens are handed to the frontend right after a correct
# password, before the OTP is confirmed — they only allow calling
# /login/verify-otp, never /ask or anything else.
LOGIN_2FA_TOKEN_EXPIRE_MINUTES = 10

def create_login_2fa_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=LOGIN_2FA_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "purpose": "login_2fa", "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

def verify_login_2fa_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise ValueError("This verification session has expired. Please sign in again.")
    if payload.get("purpose") != "login_2fa":
        raise ValueError("Invalid verification session.")
    try:
        return int(payload.get("sub"))
    except (TypeError, ValueError):
        raise ValueError("Invalid verification session.")

# "login_approval" tokens are embedded in the "Approve this sign-in" email
# link. They carry both the user id and the login_id of the pending
# browser session so main.py can mark the right one approved.
LOGIN_APPROVAL_TOKEN_EXPIRE_MINUTES = 15

def create_login_approval_token(user_id: int, login_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=LOGIN_APPROVAL_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "purpose": "login_approval", "login_id": login_id, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

def verify_login_approval_token(token: str) -> tuple[int, str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise ValueError("This approval link is invalid or has expired.")
    if payload.get("purpose") != "login_approval":
        raise ValueError("This approval link is invalid.")
    login_id = payload.get("login_id")
    if not login_id:
        raise ValueError("This approval link is invalid.")
    try:
        return int(payload.get("sub")), str(login_id)
    except (TypeError, ValueError):
        raise ValueError("This approval link is invalid.")

# ── OTP helpers ─────────────────────────────────────────────────────
# Used alongside (or instead of) the reset token above for a one-time-code
# style "forgot password" flow, e.g. emailing a 6-digit code the user types
# in rather than clicking a link.
OTP_EXPIRE_MINUTES = 10
OTP_MAX_ATTEMPTS   = 5

def generate_otp() -> str:
    """6-digit numeric OTP, cryptographically random."""
    return f"{secrets.randbelow(1_000_000):06d}"

def hash_otp(otp: str) -> str:
    """We never store the raw OTP — only its hash, same principle as passwords."""
    import hashlib
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()

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