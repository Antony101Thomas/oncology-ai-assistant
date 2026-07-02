# models.py

from pydantic import BaseModel

# ── Auth models ──────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_email: str

# ── Chat models ──────────────────────────────────────────────────
class ChatRecord(BaseModel):
    question: str
    answer: str
    confidence: str
    citations: str      # JSON string
    validated: bool
    route: str
    timestamp: str
    