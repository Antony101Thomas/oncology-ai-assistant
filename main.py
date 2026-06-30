import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from starlette.middleware.sessions import SessionMiddleware

from auth import create_token, get_current_user, hash_password, oauth, verify_password
from bm25_search import build_bm25_index
from chunker import chunk_pages
from database import (create_user, get_user_by_email,
                      get_user_by_google_id, save_chat, get_user_history)
from embedder import embed_texts
from models import LoginRequest, RegisterRequest, Token
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

@app.post("/register", response_model=Token)
def register(req: RegisterRequest):
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(req.password)
    user = create_user(req.email, hashed, req.name, provider="local")
    token = create_token(user["id"])
    return Token(access_token=token, user_name=user["name"], user_email=user["email"])


@app.post("/login", response_model=Token)
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["id"])
    return Token(access_token=token, user_name=user["name"], user_email=user["email"])


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

    token = create_token(user["id"])
    return RedirectResponse(url=f"{FRONTEND_URL}?token={token}")


@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    records = get_user_history(current_user["id"])
    return {"history": records, "user": current_user["name"]}


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