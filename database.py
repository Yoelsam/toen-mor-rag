import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "/home/rag/history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email  TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            sources     TEXT,
            created_at  TEXT    NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON history(user_email)")
    conn.commit()
    conn.close()


def save_search(user_email, question, answer, sources):
    conn = _connect()
    conn.execute(
        "INSERT INTO history (user_email, question, answer, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            user_email,
            question,
            answer,
            json.dumps(sources or [], ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return last_id


def get_history(user_email, limit=50):
    conn = _connect()
    rows = conn.execute(
        "SELECT id, question, answer, sources, created_at "
        "FROM history WHERE user_email = ? ORDER BY id DESC LIMIT ?",
        (user_email, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "sources": json.loads(r["sources"]) if r["sources"] else [],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_entry(user_email, entry_id):
    conn = _connect()
    cur = conn.execute(
        "DELETE FROM history WHERE id = ? AND user_email = ?",
        (entry_id, user_email),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted > 0
