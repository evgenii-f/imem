"""
Smoke tests for the Qdrant vector store wrapper (src/vector_store.py).

Uses Qdrant's in-memory mode (`path=":memory:"`) so these tests are fast,
require no Docker/server, and don't touch disk.

Run with: pytest tests/test_vector_store.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from src.vector_store import ImageVectorStore  # noqa: E402

EMBEDDING_DIM = 8


@pytest.fixture
def store() -> ImageVectorStore:
    return ImageVectorStore(
        collection_name="test_collection",
        embedding_dim=EMBEDDING_DIM,
        path=":memory:",
    )


def _random_unit_embeddings(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs


def test_collection_created_empty(store: ImageVectorStore):
    assert store.count() == 0


def test_upsert_and_count(store: ImageVectorStore):
    embeddings = _random_unit_embeddings(3)
    paths = ["a.jpg", "b.jpg", "c.jpg"]

    n_written = store.upsert_images(embeddings, paths)

    assert n_written == 3
    assert store.count() == 3


def test_image_exists(store: ImageVectorStore):
    embeddings = _random_unit_embeddings(1)
    store.upsert_images(embeddings, ["known.jpg"])

    assert store.image_exists("known.jpg") is True
    assert store.image_exists("unknown.jpg") is False


def test_images_exist_batch(store: ImageVectorStore):
    embeddings = _random_unit_embeddings(2)
    store.upsert_images(embeddings, ["known1.jpg", "known2.jpg"])

    result = store.images_exist(["known1.jpg", "known2.jpg", "unknown.jpg"])

    assert result == {
        "known1.jpg": True,
        "known2.jpg": True,
        "unknown.jpg": False,
    }


def test_images_exist_empty_input_returns_empty_dict(store: ImageVectorStore):
    assert store.images_exist([]) == {}


def test_filter_new_paths(store: ImageVectorStore):
    embeddings = _random_unit_embeddings(2)
    store.upsert_images(embeddings, ["existing1.jpg", "existing2.jpg"])

    candidates = ["existing1.jpg", "existing2.jpg", "new1.jpg", "new2.jpg"]
    new_paths = store.filter_new_paths(candidates)

    assert sorted(new_paths) == ["new1.jpg", "new2.jpg"]


def test_upsert_images_respects_batch_size(store: ImageVectorStore):
    n = 10
    embeddings = _random_unit_embeddings(n)
    paths = [f"batch_img_{i}.jpg" for i in range(n)]

    n_written = store.upsert_images(embeddings, paths, batch_size=3)

    assert n_written == n
    assert store.count() == n
    assert all(store.images_exist(paths).values())


def test_upsert_images_is_idempotent_on_same_path(store: ImageVectorStore):
    # Upserting the same path twice (even with a different embedding) should
    # update the existing point rather than create a duplicate, since point
    # ids are derived deterministically from the path.
    embeddings1 = _random_unit_embeddings(1)
    embeddings2 = _random_unit_embeddings(1)

    store.upsert_images(embeddings1, ["same.jpg"])
    store.upsert_images(embeddings2, ["same.jpg"])

    assert store.count() == 1


def test_search_returns_nearest_neighbor(store: ImageVectorStore):
    embeddings = _random_unit_embeddings(5)
    paths = [f"img_{i}.jpg" for i in range(5)]
    store.upsert_images(embeddings, paths)

    # Querying with the exact embedding of point 2 should return it as the top hit.
    results = store.search(embeddings[2], top_k=1)

    assert len(results.points) == 1
    assert results.points[0].payload["path"] == "img_2.jpg"
    assert results.points[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_empty_collection_returns_no_points(store: ImageVectorStore):
    query = _random_unit_embeddings(1)[0]
    results = store.search(query, top_k=5)
    assert results.points == []

