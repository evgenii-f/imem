"""
Smoke tests for the Qdrant vector store wrapper (src/vector_store.py).

Uses Qdrant's in-memory mode (`path=":memory:"`) so these tests are fast,
require no Docker/server, and don't touch disk (beyond the tmp image files
each test writes, which the store reads to compute BLAKE3 file hashes).

Point ids are path-based, while `file_hash` (BLAKE3 of the raw file bytes) is a
separate indexed payload field: `path_exist(s)` check presence by path,
`image_exist(s)` check for duplicate file content (possibly under another path).

Run with: pytest tests/test_vector_store.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("qdrant_client")
pytest.importorskip("blake3")

from src.vector_store import ImageVectorStore, file_hash  # noqa: E402

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


def _make_images(directory: Path, names: Sequence[str]) -> List[str]:
    """
    Writes a distinct solid-color PNG per name (distinct pixels => distinct file
    bytes => distinct hash) and returns their paths as strings.
    """
    paths = []
    for i, name in enumerate(names):
        color = ((i * 53) % 256, (i * 97) % 256, (i * 151) % 256)
        p = directory / name
        Image.new("RGB", (16, 16), color=color).save(p)
        paths.append(str(p))
    return paths


def _copy_bytes(src: Path, dst: Path) -> None:
    """Byte-for-byte copy so the two files hash identically under a new path."""
    dst.write_bytes(src.read_bytes())


def test_collection_created_empty(store: ImageVectorStore):
    assert store.count() == 0


def test_upsert_and_count(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, ["a.png", "b.png", "c.png"])
    n_written = store.upsert_images(_random_unit_embeddings(3), paths)

    assert n_written == 3
    assert store.count() == 3


def test_payload_carries_path_and_file_hash(store: ImageVectorStore, tmp_path: Path):
    (path,) = _make_images(tmp_path, ["x.png"])
    store.upsert_images(_random_unit_embeddings(1), [path])

    payload = store.search(_random_unit_embeddings(1)[0], top_k=1).points[0].payload

    assert payload["path"] == path
    assert payload["file_hash"] == file_hash(path)
    assert len(payload["file_hash"]) == 64  # 32-byte digest as hex


# ---- path-based presence checks ----

def test_path_exist(store: ImageVectorStore, tmp_path: Path):
    known, unknown = _make_images(tmp_path, ["known.png", "unknown.png"])
    store.upsert_images(_random_unit_embeddings(1), [known])

    assert store.path_exist(known) is True
    assert store.path_exist(unknown) is False


def test_path_exists_batch(store: ImageVectorStore, tmp_path: Path):
    k1, k2, u = _make_images(tmp_path, ["k1.png", "k2.png", "u.png"])
    store.upsert_images(_random_unit_embeddings(2), [k1, k2])

    assert store.path_exists([k1, k2, u]) == {k1: True, k2: True, u: False}


def test_path_exists_empty_input_returns_empty_dict(store: ImageVectorStore):
    assert store.path_exists([]) == {}


# ---- content-based duplicate checks (by file bytes) ----

def test_image_exist(store: ImageVectorStore, tmp_path: Path):
    indexed, other = _make_images(tmp_path, ["indexed.png", "other.png"])
    store.upsert_images(_random_unit_embeddings(1), [indexed])

    assert store.image_exist(indexed) is True
    assert store.image_exist(other) is False


def test_image_exists_batch(store: ImageVectorStore, tmp_path: Path):
    i1, i2, other = _make_images(tmp_path, ["i1.png", "i2.png", "other.png"])
    store.upsert_images(_random_unit_embeddings(2), [i1, i2])

    assert store.image_exists([i1, i2, other]) == {i1: True, i2: True, other: False}


def test_image_exists_empty_input_returns_empty_dict(store: ImageVectorStore):
    assert store.image_exists([]) == {}


def test_image_exists_unreadable_reported_as_missing(store: ImageVectorStore, tmp_path: Path):
    missing = str(tmp_path / "does_not_exist.png")
    assert store.image_exists([missing]) == {missing: False}


def test_content_check_is_independent_of_path(store: ImageVectorStore, tmp_path: Path):
    # A byte-identical file under a different name is detected by content, but
    # its path is still considered new.
    src = tmp_path / "orig.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(src)
    copy = tmp_path / "copy.png"
    _copy_bytes(src, copy)

    store.upsert_images(_random_unit_embeddings(1), [str(src)])

    assert store.image_exist(str(copy)) is True   # same bytes already present
    assert store.path_exist(str(copy)) is False   # but this path is not


# ---- id semantics (path-based) ----

def test_same_path_is_idempotent(store: ImageVectorStore, tmp_path: Path):
    (path,) = _make_images(tmp_path, ["same.png"])

    store.upsert_images(_random_unit_embeddings(1), [path])
    store.upsert_images(_random_unit_embeddings(1), [path])

    assert store.count() == 1


def test_duplicate_content_at_different_paths_both_stored(store: ImageVectorStore, tmp_path: Path):
    # Ids are path-based, so a byte-identical file under two paths coexists.
    src = tmp_path / "orig.png"
    Image.new("RGB", (16, 16), color=(40, 50, 60)).save(src)
    copy = tmp_path / "copy.png"
    _copy_bytes(src, copy)

    store.upsert_images(_random_unit_embeddings(2), [str(src), str(copy)])

    assert store.count() == 2


def test_filter_new_paths(store: ImageVectorStore, tmp_path: Path):
    e1, e2, n1, n2 = _make_images(tmp_path, ["e1.png", "e2.png", "n1.png", "n2.png"])
    store.upsert_images(_random_unit_embeddings(2), [e1, e2])

    new_paths = store.filter_new_paths([e1, e2, n1, n2])

    assert sorted(new_paths) == sorted([n1, n2])


def test_upsert_images_respects_batch_size(store: ImageVectorStore, tmp_path: Path):
    n = 10
    paths = _make_images(tmp_path, [f"batch_img_{i}.png" for i in range(n)])

    n_written = store.upsert_images(_random_unit_embeddings(n), paths, batch_size=3)

    assert n_written == n
    assert store.count() == n
    assert all(store.path_exists(paths).values())


def test_search_returns_nearest_neighbor(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, [f"img_{i}.png" for i in range(5)])
    embeddings = _random_unit_embeddings(5)
    store.upsert_images(embeddings, paths)

    # Querying with the exact embedding of point 2 should return it as the top hit.
    results = store.search(embeddings[2], top_k=1)

    assert len(results.points) == 1
    assert results.points[0].payload["path"] == paths[2]
    assert results.points[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_empty_collection_returns_no_points(store: ImageVectorStore):
    query = _random_unit_embeddings(1)[0]
    results = store.search(query, top_k=5)
    assert results.points == []
