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

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str

class VerifyRegistrationOtpRequest(BaseModel):
    email: str
    otp: str

class ResendRegistrationOtpRequest(BaseModel):
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_email: str

# ── Profile & 2FA models ─────────────────────────────────────────
class UpdateNameRequest(BaseModel):
    name: str

class UpdatePhotoRequest(BaseModel):
    photo: str | None = None   # base64 data URL, or None to remove

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class TwoFactorSettingsRequest(BaseModel):
    enabled: bool
    method: str  # "otp" or "link"

class VerifyLoginOtpRequest(BaseModel):
    temp_token: str
    otp: str

# ── Chat models ──────────────────────────────────────────────────
class ChatRecord(BaseModel):
    question: str
    answer: str
    confidence: str
    citations: str      # JSON string
    validated: bool
    route: str
    timestamp: str
    