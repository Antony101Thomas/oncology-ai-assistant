"""
database.py
-----------
PostgreSQL (Supabase) database layer for ONCO AI, via SQLAlchemy Core.

Why this replaced the old sqlite3 version:
Render's free web services have an ephemeral filesystem — any local file
(including a SQLite .db file) is wiped on every deploy and restart. Supabase
gives a persistent, free, hosted Postgres database so user accounts and chat
history survive deploys.

Required environment variable:
  DATABASE_URL = postgresql+psycopg2://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres
  (Get this from Supabase: Project Settings -> Database -> Connection string -> URI)

Same public functions as before, same return shapes (dicts / lists of dicts),
so auth.py and main.py do not need to change.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from sqlalchemy.exc import IntegrityError

# ── Database connection ───────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add your Supabase Postgres connection string "
        "as an environment variable named DATABASE_URL (see top of this file)."
    )

# pool_pre_ping avoids errors from Supabase's connection pooler dropping idle
# connections (which happens more after a free-tier project un-pauses).
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def _row_to_dict(row: Row | None) -> dict | None:
    return dict(row._mapping) if row is not None else None


# ── Table creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables if they don't already exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id               SERIAL PRIMARY KEY,
                email            TEXT    NOT NULL UNIQUE,
                hashed_password  TEXT,                      -- NULL for Google users
                name             TEXT    NOT NULL,
                provider         TEXT    NOT NULL DEFAULT 'local',  -- 'local' or 'google'
                google_id        TEXT    UNIQUE,            -- only for Google users
                created_at       TEXT    NOT NULL
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id           SERIAL PRIMARY KEY,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                question     TEXT    NOT NULL,
                answer       TEXT    NOT NULL,
                confidence   TEXT    NOT NULL DEFAULT 'low',
                citations    TEXT    NOT NULL DEFAULT '[]',  -- stored as JSON string
                validated    BOOLEAN NOT NULL DEFAULT FALSE,
                route        TEXT    NOT NULL DEFAULT 'conceptual',
                timestamp    TEXT    NOT NULL
            )
        """))

    print("[DB] Tables ready on Supabase Postgres")


# ── User operations ───────────────────────────────────────────────────────────

def create_user(email: str, hashed_password: str | None,
                name: str, provider: str = "local",
                google_id: str | None = None) -> dict | None:
    """
    Insert a new user. Returns the created user as a dict, or None if
    the email already exists.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO users (email, hashed_password, name, provider, google_id, created_at)
                VALUES (:email, :hashed_password, :name, :provider, :google_id, :created_at)
            """), {
                "email": email,
                "hashed_password": hashed_password,
                "name": name,
                "provider": provider,
                "google_id": google_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        return get_user_by_email(email)
    except IntegrityError:
        return None  # email (or google_id) already registered


def get_user_by_email(email: str) -> dict | None:
    """Return a user dict by email, or None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email = :email"), {"email": email}
        ).fetchone()
    return _row_to_dict(row)


def get_user_by_id(user_id: int) -> dict | None:
    """Return a user dict by ID, or None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": user_id}
        ).fetchone()
    return _row_to_dict(row)


def get_user_by_google_id(google_id: str) -> dict | None:
    """Return a user dict by Google ID, or None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE google_id = :google_id"), {"google_id": google_id}
        ).fetchone()
    return _row_to_dict(row)


# ── Chat history operations ───────────────────────────────────────────────────

def save_chat(user_id: int, question: str, answer: str,
              confidence: str, citations: str,
              validated: bool, route: str) -> int:
    """
    Save a Q&A exchange to chat_history.
    citations should be a JSON string.
    Returns the new row ID.
    """
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO chat_history
                (user_id, question, answer, confidence, citations, validated, route, timestamp)
            VALUES (:user_id, :question, :answer, :confidence, :citations, :validated, :route, :timestamp)
            RETURNING id
        """), {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "citations": citations,
            "validated": bool(validated),
            "route": route,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        row_id = result.scalar_one()
    return row_id


def get_user_history(user_id: int, limit: int = 50) -> list[dict]:
    """
    Return the most recent `limit` chat records for a user,
    newest first.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT * FROM chat_history
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"user_id": user_id, "limit": limit}).fetchall()
    return [dict(r._mapping) for r in rows]


def delete_user_history(user_id: int) -> int:
    """Delete all chat history for a user. Returns number of rows deleted."""
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM chat_history WHERE user_id = :user_id"), {"user_id": user_id}
        )
    return result.rowcount


# ── Startup call ──────────────────────────────────────────────────────────────
# This runs automatically when database.py is imported,
# ensuring tables always exist before anything else runs.
create_tables()


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n── ONCO AI Database Self-Test (Supabase Postgres) ──\n")

    user = create_user(
        email="test@oncai.com",
        hashed_password="hashed_pw_placeholder",
        name="Test User",
        provider="local",
    )
    print(f"Created user: {user}")

    fetched = get_user_by_email("test@oncai.com")
    print(f"Fetched user: {fetched}")

    citations = json.dumps([{"title": "NCCN Breast Cancer — Page 5", "type": "pdf"}])
    chat_id = save_chat(
        user_id=fetched["id"],
        question="What is HER2-low metastatic breast cancer?",
        answer="HER2-low is defined by IHC scores of 1+ or 2+ with negative ISH.",
        confidence="high",
        citations=citations,
        validated=True,
        route="conceptual",
    )
    print(f"Saved chat ID: {chat_id}")

    history = get_user_history(fetched["id"])
    print(f"History ({len(history)} records): {history}")

    deleted = delete_user_history(fetched["id"])
    print(f"Deleted {deleted} history record(s)")

    print("\nAll tests passed!")
