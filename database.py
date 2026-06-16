
import json
import sqlite3
import time
from pathlib import Path

from config_loader import cfg

DB_PATH = Path(cfg["db_path"])

# Pending updates expire after 30 minutes of inactivity.
# If the user doesn't respond to a confirmation prompt within this window
# (e.g., closes the browser and returns later), the pending update is
# silently discarded so it doesn't intercept unrelated future messages.
PENDING_TTL_SECONDS: int = 1800

FIELD_LABELS = {
    "name":        "Name",
    "level":       "Academic Level",
    "field":       "Field of Study",
    "gpa":         "GPA",
    "domicile":    "Domicile / Province",
    "nationality": "Nationality / Citizenship",
}


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    id              INTEGER PRIMARY KEY,
                    name            TEXT    DEFAULT 'Student',
                    level           TEXT,
                    field           TEXT,
                    nationality     TEXT,
                    domicile        TEXT,
                    gpa             REAL,
                    pending_updates TEXT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_history (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id              TEXT,
                    role                    TEXT,
                    message                 TEXT,
                    intent                  TEXT,
                    scholarships_referenced TEXT,
                    timestamp               TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migrate: add pending_updates column if it was somehow missing
            try:
                conn.execute("ALTER TABLE user_profile ADD COLUMN pending_updates TEXT")
                conn.commit()
            except Exception:
                pass
            # Seed a default profile row if the table is empty
            if conn.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0] == 0:
                conn.execute("INSERT INTO user_profile (id, name) VALUES (1, 'Student')")
                conn.commit()

    def get_profile(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM user_profile WHERE id=1").fetchone()
            return dict(row) if row else {}

    def update_profile(self, **kwargs):
        allowed = {"name", "level", "field", "nationality", "domicile", "gpa"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE user_profile SET {set_clause}, updated_at=CURRENT_TIMESTAMP WHERE id=1",
                list(updates.values()),
            )
            conn.commit()

    def get_pending_updates(self) -> dict:
        """
        Return pending profile updates, respecting the TTL.

        Handles two storage formats for backward compatibility:
          - Old format: {"name": "Ali", ...}          (plain field dict)
          - New format: {"data": {...}, "expires_at": timestamp}

        If the new format is present and the TTL has elapsed, the pending
        update is auto-cleared and an empty dict is returned.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT pending_updates FROM user_profile WHERE id=1"
            ).fetchone()
            if not (row and row[0]):
                return {}
            try:
                payload = json.loads(row[0])
            except Exception:
                return {}

        if not isinstance(payload, dict):
            return {}

        # New format: has expiry timestamp
        if "data" in payload and "expires_at" in payload:
            if time.time() > payload["expires_at"]:
                # Expired — auto-clear so it never blocks a future session
                self.clear_pending_updates()
                print("[db] Pending profile update expired and was auto-cleared.")
                return {}
            return payload["data"]

        # Old format: plain field dict — return as-is (no TTL applied)
        return payload

    def set_pending_updates(self, updates: dict):
        """
        Store pending updates with a TTL expiry timestamp.
        The update will be auto-discarded after PENDING_TTL_SECONDS if not confirmed.
        """
        payload = {
            "data":       updates,
            "expires_at": time.time() + PENDING_TTL_SECONDS,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE user_profile SET pending_updates=? WHERE id=1",
                (json.dumps(payload),),
            )
            conn.commit()

    def clear_pending_updates(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE user_profile SET pending_updates=NULL WHERE id=1")
            conn.commit()

    def save_message(self, session_id, role, message, intent=None, scholarships=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, message, intent, scholarships_referenced) "
                "VALUES (?,?,?,?,?)",
                (session_id, role, message, intent, json.dumps(scholarships or [])),
            )
            conn.commit()

    def get_recent_history(self, session_id: str, n: int = 4) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, message FROM chat_history "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, n * 2),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_last_scholarships(self, session_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT scholarships_referenced FROM chat_history "
                "WHERE session_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        return []
