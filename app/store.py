"""SQLite schema and the conversation repository. Every query on user data is scoped by user_id."""
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, text TEXT NOT NULL, embedder TEXT NOT NULL, vector BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""


@contextmanager
def connection(path: str):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")  # SQLite ignores foreign keys unless asked, per connection
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(path: str) -> None:
    with connection(path) as c:
        for table in ("conversations", "documents"):
            cols = [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]
            if cols and "user_id" not in cols:
                raise RuntimeError(f"{path} was created by an older version (no user_id on {table}); delete it or migrate it")
        c.executescript(SCHEMA)


class Store:
    def __init__(self, path: str):
        self.path = path
        init_db(path)

    def _conn(self):
        return connection(self.path)

    def exists(self, user_id: str, cid: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).fetchone() is not None

    def save_exchange(self, user_id: str, cid: str, title: str, user_text: str, assistant_text: str) -> None:
        """Persist one user/assistant pair atomically, creating the conversation on first use."""
        now = time.time()
        with self._conn() as c:  # one transaction: both messages are saved, or neither
            c.execute("INSERT OR IGNORE INTO conversations VALUES (?,?,?,?)", (cid, user_id, title, now))
            owner = c.execute("SELECT user_id FROM conversations WHERE id=?", (cid,)).fetchone()["user_id"]
            if owner != user_id:  # defence in depth: never write into someone else's conversation
                raise PermissionError("conversation belongs to another user")
            c.executemany(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
                [(cid, "user", user_text, now), (cid, "assistant", assistant_text, now)],
            )

    def messages(self, user_id: str, cid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT m.role, m.content FROM messages m JOIN conversations v ON v.id = m.conversation_id "
                "WHERE m.conversation_id=? AND v.user_id=? ORDER BY m.id", (cid, user_id))
            return [dict(r) for r in rows]

    def list(self, user_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, title, created_at FROM conversations WHERE user_id=? ORDER BY created_at DESC", (user_id,))
            return [dict(r) for r in rows]

    def delete(self, user_id: str, cid: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).rowcount > 0
