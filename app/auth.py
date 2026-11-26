"""Users, password hashing (scrypt) and opaque server-side sessions. See docs/AUTH_SPEC.md."""
import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid

from .store import connection

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 3  # OWASP-listed combination; stored with each hash so it can be raised later
LOCAL_USER = {"id": "local", "username": "local"}


class UserExists(Exception):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=2**28)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, expected = stored.split("$")
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), maxmem=2**28)
    except ValueError:  # malformed hash, e.g. the built-in local user, which can never log in
        return False
    return hmac.compare_digest(h.hex(), expected)


_dummy: str | None = None


def create_user(path: str, username: str, password: str) -> dict:
    uid = uuid.uuid4().hex
    try:
        with connection(path) as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?)", (uid, username, hash_password(password), time.time()))
    except sqlite3.IntegrityError:
        raise UserExists(username)
    return {"id": uid, "username": username}


def authenticate(path: str, username: str, password: str) -> dict | None:
    global _dummy
    with connection(path) as c:
        row = c.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,)).fetchone()
    if row is None:  # still do the expensive hash so timing does not reveal whether the username exists
        _dummy = _dummy or hash_password("dummy password")
        verify_password(password, _dummy)
        return None
    return {"id": row["id"], "username": row["username"]} if verify_password(password, row["password_hash"]) else None


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()  # token is 256-bit random, so a fast hash is fine


def create_session(path: str, user_id: str, days: int) -> str:
    token, now = secrets.token_urlsafe(32), time.time()
    with connection(path) as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (_digest(token), user_id, now, now + days * 86400))
    return token


def user_for_token(path: str, token: str) -> dict | None:
    with connection(path) as c:
        row = c.execute(
            "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=? AND s.expires_at > ?", (_digest(token), time.time())).fetchone()
    return dict(row) if row else None


def revoke_token(path: str, token: str) -> None:
    with connection(path) as c:
        c.execute("DELETE FROM sessions WHERE token_hash=?", (_digest(token),))


def ensure_local_user(path: str) -> None:
    with connection(path) as c:
        c.execute("INSERT OR IGNORE INTO users VALUES ('local','local','!',?)", (time.time(),))
