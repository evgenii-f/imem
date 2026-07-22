"""
Integration tests for the HTTP API (imem/api/app.py).

Backs the routes with an in-memory Qdrant store and a FakeEncoder returning
controlled embeddings — same approach as tests/test_query.py — so these run
fast without a live server, network, or GPU. The app's lifespan only populates
app.state fields that aren't already set, so we pre-seed encoder / client /
store_factory and the real model is never loaded.

Run with: pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("qdrant_client")
pytest.importorskip("blake3")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from imem.api.app import create_app  # noqa: E402
from imem.vector_store import ImageVectorStore  # noqa: E402

EMBEDDING_DIM = 8
COLLECTION = "test_api_collection"


class FakeEncoder:
    """Fixed-embedding stand-in for ImageTextEncoder (cf. tests/test_query.py)."""

    embedding_dim = EMBEDDING_DIM

    def __init__(self, text_vec=None, image_vec=None):
        self._text_vec = text_vec
        self._image_vec = image_vec

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._text_vec for _ in list(texts)], dtype=np.float32)

    def encode_images(
        self, image_paths: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        paths = list(image_paths)
        return np.array([self._image_vec for _ in paths], dtype=np.float32), paths


def _unit_embeddings(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs


def _make_images(directory: Path, names: Sequence[str]) -> List[str]:
    paths = []
    for i, name in enumerate(names):
        color = ((i * 53) % 256, (i * 97) % 256, (i * 151) % 256)
        p = directory / name
        Image.new("RGB", (16, 16), color=color).save(p)
        paths.append(str(p))
    return paths


@pytest.fixture
def seeded(tmp_path: Path):
    """An in-memory store seeded with 3 images and their embeddings, plus the
    paths and embeddings so a test can steer the fake encoder toward a known
    point."""
    store = ImageVectorStore(
        collection_name=COLLECTION, embedding_dim=EMBEDDING_DIM, path=":memory:"
    )
    paths = _make_images(tmp_path, ["a.png", "b.png", "c.png"])
    embeddings = _unit_embeddings(3)
    store.upsert_images(embeddings, paths, show_progress=False)
    return store, paths, embeddings


def _client(store, encoder) -> TestClient:
    app = create_app()
    app.state.encoder = encoder
    app.state.qdrant_client = store.client  # catalog reads run on the same client
    app.state.store_factory = lambda collection, dim: store
    return TestClient(app)


# ---- /health, /config, /collections ----

def test_health_ok(seeded):
    store, _, _ = seeded
    with _client(store, FakeEncoder()) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["qdrant_reachable"] is True


def test_config_reports_defaults(seeded):
    store, _, _ = seeded
    with _client(store, FakeEncoder()) as c:
        body = c.get("/config").json()
    assert body["default_top_k"] >= 1
    assert ".png" in body["extensions"]
    assert body["model_id"]


def test_collections_lists_seeded(seeded):
    store, paths, _ = seeded
    with _client(store, FakeEncoder()) as c:
        body = c.get("/collections").json()
    entry = {row["name"]: row["count"] for row in body["collections"]}
    assert entry.get(COLLECTION) == len(paths)


# ---- /query (JSON) ----

def test_query_text_returns_nearest(seeded):
    store, paths, embeddings = seeded
    encoder = FakeEncoder(text_vec=embeddings[1])
    with _client(store, encoder) as c:
        body = c.post(
            "/query", json={"collection": COLLECTION, "query": "anything", "top_k": 1}
        ).json()
    assert body["mode"] == "text"
    assert body["results"][0]["path"] == paths[1]
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_query_respects_top_k(seeded):
    store, _, _ = seeded
    encoder = FakeEncoder(text_vec=_unit_embeddings(1)[0])
    with _client(store, encoder) as c:
        body = c.post(
            "/query", json={"collection": COLLECTION, "query": "x", "top_k": 2}
        ).json()
    assert len(body["results"]) == 2


def test_query_missing_collection_404(seeded):
    store, _, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.post("/query", json={"collection": "nope", "query": "x"})
    assert resp.status_code == 404


def test_query_invalid_mode_422(seeded):
    store, _, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.post(
            "/query", json={"collection": COLLECTION, "query": "x", "mode": "bogus"}
        )
    assert resp.status_code == 422


# ---- /query/upload (multipart) ----

def test_query_upload_returns_results(seeded, tmp_path):
    store, paths, embeddings = seeded
    encoder = FakeEncoder(image_vec=embeddings[0])
    upload = tmp_path / "ref.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(upload)
    with _client(store, encoder) as c:
        with open(upload, "rb") as fh:
            resp = c.post(
                "/query/upload",
                data={"collection": COLLECTION, "top_k": 1},
                files={"file": ("ref.png", fh, "image/png")},
            )
    body = resp.json()
    assert body["mode"] == "image"
    assert body["results"][0]["path"] == paths[0]


def test_query_upload_missing_collection_404(seeded, tmp_path):
    store, _, _ = seeded
    upload = tmp_path / "ref.png"
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(upload)
    with _client(store, FakeEncoder(image_vec=_unit_embeddings(1)[0])) as c:
        with open(upload, "rb") as fh:
            resp = c.post(
                "/query/upload",
                data={"collection": "nope"},
                files={"file": ("ref.png", fh, "image/png")},
            )
    assert resp.status_code == 404


# ---- /image ----

def test_image_serves_original(seeded):
    store, paths, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.get("/image", params={"path": paths[0]})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_image_thumb_returns_jpeg(seeded):
    store, paths, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.get("/image", params={"path": paths[0], "thumb": 8})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(resp.content)) as img:
        assert max(img.size) <= 8


def test_image_download_sets_attachment(seeded):
    store, paths, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.get("/image", params={"path": paths[0], "download": "true"})
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


def test_image_non_image_path_404(seeded, tmp_path):
    store, _, _ = seeded
    txt = tmp_path / "note.txt"
    txt.write_text("not an image")
    with _client(store, FakeEncoder()) as c:
        resp = c.get("/image", params={"path": str(txt)})
    assert resp.status_code == 404


def test_image_missing_file_404(seeded, tmp_path):
    store, _, _ = seeded
    with _client(store, FakeEncoder()) as c:
        resp = c.get("/image", params={"path": str(tmp_path / "ghost.png")})
    assert resp.status_code == 404
