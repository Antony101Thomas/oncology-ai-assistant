import json
import os
import secrets as pysecrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from starlette.middleware.sessions import SessionMiddleware

from auth import (create_reset_token, create_token, generate_otp, get_current_user,
                  hash_otp, hash_password, oauth, verify_password, verify_reset_token,
                  OTP_EXPIRE_MINUTES, OTP_MAX_ATTEMPTS,
                  create_login_2fa_token, verify_login_2fa_token,
                  create_login_approval_token, verify_login_approval_token)
from bm25_search import build_bm25_index
from chunker import chunk_pages
from database import (create_user, get_user_by_email, get_reset_otp,
                      get_user_by_google_id, get_user_by_id, save_chat, get_user_history,
                      set_reset_otp, increment_otp_attempts, clear_reset_otp,
                      update_user_password, update_user_name, update_user_photo,
                      update_two_factor_settings, set_login_otp, get_login_otp,
                      increment_login_otp_attempts, clear_login_otp,
                      set_registration_otp, get_registration_otp,
                      increment_registration_otp_attempts, mark_email_verified)
from embedder import embed_texts
from models import (ForgotPasswordRequest, LoginRequest, RegisterRequest,
                    ResetPasswordRequest, Token, VerifyOtpRequest,
                    UpdateNameRequest, UpdatePhotoRequest, ChangePasswordRequest,
                    TwoFactorSettingsRequest, VerifyLoginOtpRequest,
                    VerifyRegistrationOtpRequest, ResendRegistrationOtpRequest)
from pdf_extractor import extract_pdf_pages
from rag_agent import execute_rag_query

load_dotenv()

APP_DIR        = Path(__file__).resolve().parent
UPLOAD_DIR     = APP_DIR / "uploads"
AUTO_INDEX_DIR = APP_DIR / "auto_index"
DEFAULT_PDF    = APP_DIR / "test.pdf"
COLLECTION     = "oncology_docs"
VECTOR_SIZE    = 384

FRONTEND_URL = "https://antony101thomas.github.io/oncology-ai-assistant/oncology_ui.html"
FRONTEND_ORIGIN = "https://antony101thomas.github.io"

# ── Password-reset email (Resend) ────────────────────────────────────────
# If RESEND_API_KEY isn't set, reset links are just printed to the server
# logs instead of emailed — handy for local development before Resend is
# configured, without blocking the rest of the feature.
RESEND_API_KEY    = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "ONCO AI <onboarding@resend.dev>")

app    = FastAPI(title="ONCO AI")
qdrant = QdrantClient(":memory:")

indexed_chunks:  list[dict[str, Any]] = []
indexed_sources: list[str]            = []

# ── SessionMiddleware FIRST, then CORS ───────────────────────────────────────
app.add_middleware(SessionMiddleware,
                   secret_key=os.getenv("JWT_SECRET", "change-this"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


# In-memory store for pending "approve this sign-in" logins (link-based 2FA).
# Keyed by a random login_id the browser tab polls on. Fine for a single
# free-tier instance; entries are short-lived (created + polled within
# LOGIN_APPROVAL_TOKEN_EXPIRE_MINUTES) and are popped once consumed.
pending_logins: dict[str, dict[str, Any]] = {}


def _clear_pending_logins_for_user(user_id: int) -> None:
    """
    Only one link-based sign-in challenge should be "live" per user at a
    time. If a new one is requested (e.g. the person tries Google sign-in,
    then tries password sign-in before finishing the first), drop any
    earlier pending entries so they don't end up with two valid emailed
    links racing each other.
    """
    stale = [lid for lid, entry in pending_logins.items() if entry["user_id"] == user_id]
    for lid in stale:
        pending_logins.pop(lid, None)


# ── Collection helpers ────────────────────────────────────────────────────────

def ensure_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION in existing:
        qdrant.delete_collection(collection_name=COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def index_pdf_paths(pdf_paths: list[Path]) -> dict[str, Any]:
    global indexed_chunks, indexed_sources

    seen, unique_paths = set(), []
    for p in pdf_paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            unique_paths.append(r)

    pages = []
    for pdf_path in unique_paths:
        pages.extend(extract_pdf_pages(str(pdf_path)))

    if not pages:
        raise ValueError("No readable text found in the uploaded PDF files.")

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No searchable chunks could be created from the PDF files.")

    ensure_collection()

    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    from qdrant_client.models import PointStruct
    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=embeddings[i],
            payload={"source": chunk["source"], "page": chunk["page"], "text": chunk["text"]},
        )
        for i, chunk in enumerate(chunks)
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)
    build_bm25_index(chunks)

    indexed_chunks  = chunks
    indexed_sources = [p.name for p in unique_paths]
    print(f"Indexed {len(points)} chunks from {len(unique_paths)} PDF(s).")

    return {"indexed_files": indexed_sources, "pages": len(pages), "chunks": len(chunks)}


def get_auto_index_pdfs() -> list[Path]:
    AUTO_INDEX_DIR.mkdir(exist_ok=True)
    return sorted(AUTO_INDEX_DIR.glob("*.pdf"))


def index_auto_folder() -> dict[str, Any]:
    pdf_paths = get_auto_index_pdfs()
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {AUTO_INDEX_DIR}.")
    return index_pdf_paths(pdf_paths)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    try:
        stats = index_auto_folder()
        print(f"Auto-index ready: {stats}")
    except ValueError as exc:
        ensure_collection()
        build_bm25_index([])
        print(f"Backend ready without PDFs. {exc}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "message": "ONCO AI is running",
        "indexed_files": indexed_sources,
        "chunks": len(indexed_chunks),
        "auto_index_folder": str(AUTO_INDEX_DIR),
        "auto_index_files": [p.name for p in get_auto_index_pdfs()],
    }


# ── Auth routes ───────────────────────────────────────────────────────────────

def send_registration_otp_email(to_email: str, otp: str) -> None:
    if not RESEND_API_KEY:
        print(f"[Register] RESEND_API_KEY not set. Verification code for {to_email}: {otp}")
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Verify your ONCO AI account",
                "html": (
                    "<p>Welcome to ONCO AI! Enter this code to verify your email "
                    "and finish creating your account:</p>"
                    f"<p style=\"font-size:28px;font-weight:700;letter-spacing:4px;\">{otp}</p>"
                    "<p>This code expires in 10 minutes. If you didn't request this, "
                    "you can safely ignore this email.</p>"
                ),
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[Register] Failed to send verification email: {exc}")


@app.post("/register")
def register(req: RegisterRequest) -> dict[str, Any]:
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(req.password)
    user = create_user(req.email, hashed, req.name, provider="local")
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Local accounts start unverified. Send an OTP and hand the frontend a
    # "please verify" signal instead of an access token — mirrors the
    # requires_2fa shape used by /login below, just at signup time.
    otp = generate_otp()
    otp_hash = hash_otp(otp)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    set_registration_otp(user["id"], otp_hash, expires_at)
    send_registration_otp_email(user["email"], otp)

    return {
        "requires_verification": True,
        "email": user["email"],
        "message": "A verification code has been sent to your email.",
    }


@app.post("/verify-registration-otp", response_model=Token)
def verify_registration_otp(req: VerifyRegistrationOtpRequest):
    generic_error = "Invalid or expired code."
    user = get_user_by_email(req.email)
    if not user or user.get("provider") != "local":
        raise HTTPException(status_code=400, detail=generic_error)

    if user.get("email_verified"):
        # Already verified (e.g. a double submit) — just sign them in
        # rather than erroring on a stale/reused code.
        token = create_token(user["id"])
        return Token(access_token=token, user_name=user["name"], user_email=user["email"])

    record = get_registration_otp(user["id"])
    if not record or not record.get("reg_otp_hash"):
        raise HTTPException(status_code=400, detail=generic_error)

    if record["reg_otp_attempts"] >= OTP_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Please request a new code.")

    expires_at = datetime.fromisoformat(record["reg_otp_expires"])
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail=generic_error)

    if hash_otp(req.otp.strip()) != record["reg_otp_hash"]:
        increment_registration_otp_attempts(user["id"])
        raise HTTPException(status_code=400, detail=generic_error)

    mark_email_verified(user["id"])
    token = create_token(user["id"])
    return Token(access_token=token, user_name=user["name"], user_email=user["email"])


@app.post("/register/resend-otp")
def resend_registration_otp(req: ResendRegistrationOtpRequest) -> dict[str, Any]:
    user = get_user_by_email(req.email)
    if not user or user.get("provider") != "local":
        # Don't reveal whether the account exists.
        return {"message": "If that account needs verification, a new code has been sent."}
    if user.get("email_verified"):
        return {"message": "This account is already verified."}

    otp = generate_otp()
    otp_hash = hash_otp(otp)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    set_registration_otp(user["id"], otp_hash, expires_at)
    send_registration_otp_email(user["email"], otp)
    return {"message": "A new verification code has been sent."}


@app.post("/login")
def login(req: LoginRequest) -> dict[str, Any]:
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user.get("provider") == "local" and not user.get("email_verified"):
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before signing in.",
        )

    if not user.get("two_factor_enabled"):
        token = create_token(user["id"])
        return {
            "requires_2fa": False,
            "access_token": token,
            "token_type": "bearer",
            "user_name": user["name"],
            "user_email": user["email"],
        }

    method = user.get("two_factor_method") or "otp"

    if method == "link":
        _clear_pending_logins_for_user(user["id"])
        login_id = pysecrets.token_urlsafe(16)
        pending_logins[login_id] = {
            "user_id": user["id"],
            "approved": False,
            "created": datetime.now(timezone.utc),
        }
        approval_token = create_login_approval_token(user["id"], login_id)
        approve_url = f"https://onco-ai-api.onrender.com/login/approve?token={approval_token}"
        send_login_approval_email(user["email"], approve_url)
        return {"requires_2fa": True, "method": "link", "login_id": login_id}

    # Default / "otp" method
    otp = generate_otp()
    otp_hash = hash_otp(otp)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    set_login_otp(user["id"], otp_hash, expires_at)
    send_login_otp_email(user["email"], otp)
    return {
        "requires_2fa": True,
        "method": "otp",
        "temp_token": create_login_2fa_token(user["id"]),
    }


def send_login_otp_email(to_email: str, otp: str) -> None:
    if not RESEND_API_KEY:
        print(f"[Login 2FA] RESEND_API_KEY not set. Login OTP for {to_email}: {otp}")
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Your ONCO AI sign-in code",
                "html": (
                    "<p>Someone is signing in to your ONCO AI account. "
                    "Enter this code to continue:</p>"
                    f"<p style=\"font-size:28px;font-weight:700;letter-spacing:4px;\">{otp}</p>"
                    "<p>This code expires in 10 minutes. If this wasn't you, "
                    "you can safely ignore this email — your account is still secure.</p>"
                ),
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[Login 2FA] Failed to send login OTP email: {exc}")


def send_login_approval_email(to_email: str, approve_url: str) -> None:
    if not RESEND_API_KEY:
        print(f"[Login 2FA] RESEND_API_KEY not set. Approval link for {to_email}: {approve_url}")
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Approve your ONCO AI sign-in",
                "html": (
                    "<p>Someone is signing in to your ONCO AI account. "
                    "If this was you, click below to approve it:</p>"
                    f"<p><a href=\"{approve_url}\" "
                    "style=\"display:inline-block;padding:12px 24px;background:#7c3aed;"
                    "color:#fff;text-decoration:none;border-radius:8px;font-weight:600;\">"
                    "Approve this sign-in</a></p>"
                    "<p>This link expires in 15 minutes. If this wasn't you, ignore this "
                    "email and consider changing your password.</p>"
                ),
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[Login 2FA] Failed to send approval email: {exc}")


@app.post("/login/verify-otp", response_model=Token)
def login_verify_otp(req: VerifyLoginOtpRequest):
    generic_error = "Invalid or expired code."
    try:
        user_id = verify_login_2fa_token(req.temp_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = get_login_otp(user_id)
    if not record or not record.get("login_otp_hash"):
        raise HTTPException(status_code=400, detail=generic_error)

    if record["login_otp_attempts"] >= OTP_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Please sign in again.")

    expires_at = datetime.fromisoformat(record["login_otp_expires"])
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail=generic_error)

    if hash_otp(req.otp.strip()) != record["login_otp_hash"]:
        increment_login_otp_attempts(user_id)
        raise HTTPException(status_code=400, detail=generic_error)

    clear_login_otp(user_id)
    user = get_user_by_id(user_id)
    token = create_token(user_id)
    return Token(access_token=token, user_name=user["name"], user_email=user["email"])


@app.get("/login/approve")
def login_approve(token: str) -> HTMLResponse:
    try:
        user_id, login_id = verify_login_approval_token(token)
    except ValueError as exc:
        return HTMLResponse(f"<html><body><p>{exc}</p></body></html>", status_code=400)

    entry = pending_logins.get(login_id)
    if not entry or entry["user_id"] != user_id:
        return HTMLResponse(
            "<html><body><p>This sign-in request has expired or was already used. "
            "You can close this tab.</p></body></html>",
            status_code=400,
        )

    entry["approved"] = True
    return HTMLResponse(
        "<html><body style=\"font-family:sans-serif;text-align:center;padding:60px 20px;\">"
        "<h2>Sign-in approved ✓</h2><p>You can close this tab and return to ONCO AI.</p>"
        "</body></html>"
    )


@app.get("/login/status/{login_id}")
def login_status(login_id: str) -> dict[str, Any]:
    entry = pending_logins.get(login_id)
    if not entry:
        raise HTTPException(status_code=404, detail="This sign-in request has expired.")

    if not entry["approved"]:
        return {"approved": False}

    user = get_user_by_id(entry["user_id"])
    pending_logins.pop(login_id, None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    token = create_token(user["id"])
    return {
        "approved": True,
        "access_token": token,
        "token_type": "bearer",
        "user_name": user["name"],
        "user_email": user["email"],
    }


def send_otp_email(to_email: str, otp: str) -> None:
    if not RESEND_API_KEY:
        print(f"[Password Reset] RESEND_API_KEY not set. OTP for {to_email}: {otp}")
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Your ONCO AI password reset code",
                "html": (
                    "<p>We received a request to reset your ONCO AI password.</p>"
                    f"<p style=\"font-size:28px;font-weight:700;letter-spacing:4px;\">{otp}</p>"
                    "<p>This code expires in 10 minutes. If you didn't request this, "
                    "you can safely ignore this email.</p>"
                ),
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[Password Reset] Failed to send OTP email via Resend: {exc}")


@app.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest) -> dict[str, Any]:
    user = get_user_by_email(req.email)

    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email.")

    if user.get("provider") != "local":
        raise HTTPException(
            status_code=400,
            detail="This account signs in with Google — there's no password to reset.",
        )

    otp        = generate_otp()
    otp_hash   = hash_otp(otp)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    set_reset_otp(user["id"], otp_hash, expires_at)
    send_otp_email(user["email"], otp)
    return {"message": "A reset code has been sent to your email."}


@app.post("/verify-otp")
def verify_otp(req: VerifyOtpRequest) -> dict[str, Any]:
    generic_error = "Invalid or expired code."
    user = get_user_by_email(req.email)
    if not user or user.get("provider") != "local":
        raise HTTPException(status_code=400, detail=generic_error)

    record = get_reset_otp(user["id"])
    if not record or not record.get("reset_otp_hash"):
        raise HTTPException(status_code=400, detail=generic_error)

    if record["reset_otp_attempts"] >= OTP_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Please request a new code.")

    expires_at = datetime.fromisoformat(record["reset_otp_expires"])
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail=generic_error)

    if hash_otp(req.otp.strip()) != record["reset_otp_hash"]:
        increment_otp_attempts(user["id"])
        raise HTTPException(status_code=400, detail=generic_error)

    clear_reset_otp(user["id"])
    token = create_reset_token(user["id"])
    return {"token": token}


@app.post("/reset-password")
def reset_password(req: ResetPasswordRequest) -> dict[str, Any]:
    try:
        user_id = verify_reset_token(req.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Don't allow "resetting" to the same password the account already has.
    if user.get("hashed_password") and verify_password(req.new_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="New password must be different from your old password.")

    hashed  = hash_password(req.new_password)
    updated = update_user_password(user_id, hashed)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")

    return {"message": "Password updated successfully. You can now sign in."}


@app.get("/auth/google")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(
        request, os.getenv("GOOGLE_REDIRECT_URI",
        "https://onco-ai-api.onrender.com/auth/callback")
    )


@app.get("/auth/callback")
async def google_callback(request: Request):
    token_data = await oauth.google.authorize_access_token(request)
    info = token_data.get("userinfo")

    user = get_user_by_google_id(info["sub"])
    if not user:
        user = get_user_by_email(info["email"])
        if not user:
            user = create_user(
                email=info["email"],
                hashed_password=None,
                name=info["name"],
                provider="google",
                google_id=info["sub"]
            )

    # ── Two-step verification also applies to Google sign-in ──────────────
    # Build a JSON payload describing what the frontend should do next:
    # either a normal token (2FA off), or a "please verify" instruction
    # (2FA on) using whichever method the user picked in their profile.
    if user.get("two_factor_enabled"):
        method = user.get("two_factor_method") or "otp"

        if method == "link":
            _clear_pending_logins_for_user(user["id"])
            login_id = pysecrets.token_urlsafe(16)
            pending_logins[login_id] = {
                "user_id": user["id"],
                "approved": False,
                "created": datetime.now(timezone.utc),
            }
            approval_token = create_login_approval_token(user["id"], login_id)
            approve_url = f"https://onco-ai-api.onrender.com/login/approve?token={approval_token}"
            send_login_approval_email(user["email"], approve_url)
            payload = {
                "type": "onco-google-auth",
                "requires_2fa": True,
                "method": "link",
                "login_id": login_id,
                "email": user["email"],
            }
        else:
            otp = generate_otp()
            otp_hash = hash_otp(otp)
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
            set_login_otp(user["id"], otp_hash, expires_at)
            send_login_otp_email(user["email"], otp)
            payload = {
                "type": "onco-google-auth",
                "requires_2fa": True,
                "method": "otp",
                "temp_token": create_login_2fa_token(user["id"]),
                "email": user["email"],
            }
    else:
        token = create_token(user["id"])
        payload = {
            "type": "onco-google-auth",
            "requires_2fa": False,
            "token": token,
            "user_name": user["name"],
            "user_email": user["email"],
        }

    payload_json = json.dumps(payload)

    # Build the no-popup fallback redirect URL's query string from the same payload.
    fallback_params = {"requires_2fa": "1" if payload["requires_2fa"] else "0"}
    if payload["requires_2fa"]:
        fallback_params["method"] = payload["method"]
        fallback_params["email"] = payload["email"]
        if payload["method"] == "otp":
            fallback_params["temp_token"] = payload["temp_token"]
        else:
            fallback_params["login_id"] = payload["login_id"]
    else:
        fallback_params["token"] = payload["token"]
    fallback_query = "&".join(f"{k}={v}" for k, v in fallback_params.items())

    # This endpoint is normally opened in a small popup window by the
    # frontend (see googleLogin() in oncology_ui.html), so the user's main
    # tab never navigates to Google and nothing shows up in its back-history.
    # We hand the payload back to that opener window via postMessage and
    # close the popup. If there's no opener (e.g. popups were blocked and
    # the frontend fell back to a full-page redirect), we fall back to a
    # redirect with the same information carried in the URL instead.
    html = f"""<!DOCTYPE html>
<html>
<body>
<script>
  const payload = {payload_json};
  if (window.opener) {{
    window.opener.postMessage(payload, "{FRONTEND_ORIGIN}");
    // Cross-window postMessage delivery is asynchronous — closing the
    // popup on the very next line risks the opener seeing "popup closed"
    // before it has actually processed the message (a real race that broke
    // the 2FA hand-off). A short delay guarantees the message lands first.
    setTimeout(() => window.close(), 300);
  }} else {{
    window.location.replace("{FRONTEND_URL}?{fallback_query}");
  }}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    records = get_user_history(current_user["id"])
    return {"history": records, "user": current_user["name"]}


# ── Profile routes ────────────────────────────────────────────────────────────

MAX_PHOTO_BASE64_CHARS = 2_000_000  # roughly ~1.5MB image, generous for a demo

@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "name": current_user["name"],
        "email": current_user["email"],
        "provider": current_user["provider"],
        "profile_pic": current_user.get("profile_pic"),
        "two_factor_enabled": bool(current_user.get("two_factor_enabled")),
        "two_factor_method": current_user.get("two_factor_method") or "otp",
    }


@app.put("/profile/name")
def put_profile_name(req: UpdateNameRequest,
                     current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    update_user_name(current_user["id"], name)
    return {"message": "Name updated.", "name": name}


@app.put("/profile/photo")
def put_profile_photo(req: UpdatePhotoRequest,
                      current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if req.photo and len(req.photo) > MAX_PHOTO_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="Image is too large. Please use a smaller photo.")
    update_user_photo(current_user["id"], req.photo)
    return {"message": "Profile photo updated." if req.photo else "Profile photo removed."}


@app.post("/profile/change-password")
def post_change_password(req: ChangePasswordRequest,
                         current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if current_user.get("provider") != "local":
        raise HTTPException(
            status_code=400,
            detail="This account signs in with Google — there's no password to change.",
        )
    if not verify_password(req.current_password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")
    if verify_password(req.new_password, current_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="New password must be different from your current password.")
    update_user_password(current_user["id"], hash_password(req.new_password))
    return {"message": "Password updated successfully."}


@app.put("/profile/2fa")
def put_two_factor_settings(req: TwoFactorSettingsRequest,
                            current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if req.method not in ("otp", "link"):
        raise HTTPException(status_code=400, detail="Method must be 'otp' or 'link'.")
    update_two_factor_settings(current_user["id"], req.enabled, req.method)
    return {"message": "Two-step verification settings updated.", "enabled": req.enabled, "method": req.method}


# ── PDF routes ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    UPLOAD_DIR.mkdir(exist_ok=True)
    saved_paths: list[Path] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
        safe_name   = Path(file.filename).name
        destination = UPLOAD_DIR / safe_name
        content     = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{safe_name} is empty.")
        destination.write_bytes(content)
        saved_paths.append(destination)

    try:
        stats = index_pdf_paths(saved_paths)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"status": "indexed", **stats}


@app.post("/index-default")
def index_default_pdf() -> dict[str, Any]:
    if not DEFAULT_PDF.exists():
        raise HTTPException(status_code=404, detail="test.pdf was not found.")
    try:
        stats = index_pdf_paths([DEFAULT_PDF])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "indexed", **stats}


@app.post("/index-auto")
def index_auto_pdfs() -> dict[str, Any]:
    try:
        stats = index_auto_folder()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "indexed", **stats}


@app.get("/pdf/{filename}")
def serve_pdf(filename: str) -> FileResponse:
    for folder in [AUTO_INDEX_DIR, UPLOAD_DIR, APP_DIR]:
        path = folder / filename
        if path.exists() and path.suffix.lower() == ".pdf":
            return FileResponse(
                path=str(path),
                media_type="application/pdf",
                headers={"Content-Disposition": f"inline; filename={filename}"}
            )
    raise HTTPException(status_code=404, detail=f"{filename} not found.")


# ── Ask route ─────────────────────────────────────────────────────────────────

@app.post("/ask")
def ask_question(request: QuestionRequest,
                 current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if not indexed_chunks:
        raise HTTPException(
            status_code=409,
            detail="No PDFs indexed yet. Upload PDFs first.",
        )

    result = execute_rag_query(
        question=request.question,
        qdrant=qdrant,
        indexed_chunks=indexed_chunks,
    )

    save_chat(
        user_id=current_user["id"],
        question=request.question,
        answer=result["answer"],
        confidence=result["confidence"],
        citations=json.dumps(result.get("citations", [])),
        validated=result.get("validated", False),
        route=result.get("route", "conceptual")
    )

    return result


# ── Guest ask route (no auth, limited on the frontend to N free questions) ────

@app.post("/ask-guest")
def ask_question_guest(request: QuestionRequest) -> dict[str, Any]:
    """
    Same pipeline as /ask, but does not require a logged-in user and does not
    persist anything to chat_history. The free-question limit is enforced by
    the guest frontend (guest.html); this endpoint just answers the question.
    """
    if not indexed_chunks:
        raise HTTPException(
            status_code=409,
            detail="No PDFs indexed yet. Upload PDFs first.",
        )

    result = execute_rag_query(
        question=request.question,
        qdrant=qdrant,
        indexed_chunks=indexed_chunks,
    )

    return result