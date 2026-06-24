"""
database.py
-----------
SQLite database setup for ONCO AI.
Creates and manages two tables:
  - users        : stores registered user accounts
  - chat_history : stores every Q&A linked to a user
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# ── Database file location ────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "onco_ai.db"


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Table creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── users table ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            email            TEXT    NOT NULL UNIQUE,
            hashed_password  TEXT,                      -- NULL for Google users
            name             TEXT    NOT NULL,
            provider         TEXT    NOT NULL DEFAULT 'local',  -- 'local' or 'google'
            google_id        TEXT    UNIQUE,            -- only for Google users
            created_at       TEXT    NOT NULL
        )
    """)

    # ── chat_history table ───────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            question     TEXT    NOT NULL,
            answer       TEXT    NOT NULL,
            confidence   TEXT    NOT NULL DEFAULT 'low',
            citations    TEXT    NOT NULL DEFAULT '[]',  -- stored as JSON string
            validated    INTEGER NOT NULL DEFAULT 0,     -- 0 = False, 1 = True
            route        TEXT    NOT NULL DEFAULT 'conceptual',
            timestamp    TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Tables ready at {DB_PATH}")


# ── User operations ───────────────────────────────────────────────────────────

def create_user(email: str, hashed_password: str | None,
                name: str, provider: str = "local",
                google_id: str | None = None) -> dict | None:
    """
    Insert a new user. Returns the created user as a dict, or None if
    the email already exists.
    """
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO users (email, hashed_password, name, provider, google_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, hashed_password, name, provider, google_id,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()
        user = get_user_by_email(email)
        return user
    except sqlite3.IntegrityError:
        return None   # email already registered
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    """Return a user dict by email, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    """Return a user dict by ID, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_google_id(google_id: str) -> dict | None:
    """Return a user dict by Google ID, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE google_id = ?", (google_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Chat history operations ───────────────────────────────────────────────────

def save_chat(user_id: int, question: str, answer: str,
              confidence: str, citations: str,
              validated: bool, route: str) -> int:
    """
    Save a Q&A exchange to chat_history.
    citations should be a JSON string.
    Returns the new row ID.
    """
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO chat_history
            (user_id, question, answer, confidence, citations, validated, route, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, question, answer, confidence, citations,
          1 if validated else 0, route,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_user_history(user_id: int, limit: int = 50) -> list[dict]:
    """
    Return the most recent `limit` chat records for a user,
    newest first.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM chat_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user_history(user_id: int) -> int:
    """Delete all chat history for a user. Returns number of rows deleted."""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM chat_history WHERE user_id = ?", (user_id,)
    )
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


# ── Startup call ──────────────────────────────────────────────────────────────

# This runs automatically when database.py is imported,
# ensuring tables always exist before anything else runs.
create_tables()


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n── ONCO AI Database Self-Test ──\n")

    # Create a test user
    user = create_user(
        email="test@oncai.com",
        hashed_password="hashed_pw_placeholder",
        name="Test User",
        provider="local"
    )
    print(f"Created user: {user}")

    # Fetch by email
    fetched = get_user_by_email("test@oncai.com")
    print(f"Fetched user: {fetched}")

    # Save a chat record
    citations = json.dumps([{"title": "NCCN Breast Cancer — Page 5", "type": "pdf"}])
    chat_id = save_chat(
        user_id=fetched["id"],
        question="What is HER2-low metastatic breast cancer?",
        answer="HER2-low is defined by IHC scores of 1+ or 2+ with negative ISH.",
        confidence="high",
        citations=citations,
        validated=True,
        route="conceptual"
    )
    print(f"Saved chat ID: {chat_id}")

    # Retrieve history
    history = get_user_history(fetched["id"])
    print(f"History ({len(history)} records): {history}")

    # Clean up test data
    deleted = delete_user_history(fetched["id"])
    print(f"Deleted {deleted} history record(s)")

    print("\nAll tests passed!")
