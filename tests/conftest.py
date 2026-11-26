import pytest
from fastapi.testclient import TestClient

from app import auth
from app.main import create_app
from app.rag import HashEmbedder, KnowledgeBase
from app.store import Store

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(auth, "SCRYPT_N", 2**10)  # production cost is 2**15; tests need speed, not strength


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "t.db")


def signup(client, name="alice"):
    r = client.post("/api/auth/register", json={"username": name, "password": PASSWORD})
    assert r.status_code == 201, r.text
    client.headers["Authorization"] = "Bearer " + r.json()["token"]
    return r.json()["user"]


@pytest.fixture
def make_client(db_path):
    """Factory for independent API clients sharing one database (to test multi-user behaviour)."""
    return lambda: TestClient(create_app(Store(db_path), KnowledgeBase(db_path, HashEmbedder())))


@pytest.fixture
def client(make_client):
    c = make_client()
    signup(c)
    return c
