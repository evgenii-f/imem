"""
Integration tests for the query pipeline (src/query.py).

Uses Qdrant's in-memory mode and a FakeEncoder returning controlled
embeddings instead of loading a real model, so these run fast without
network/GPU access — same approach as tests/test_indexer.py. query_images()
only relies on encoder.encode_text/encode_images and the real store search,
so a fake stand-in exercises the actual orchestration.

Run with: pytest tests/test_query.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("qdrant_client")
pytest.importorskip("blake3")

from src.query import looks_like_image_path, query_images  # noqa: E402
from src.vector_store import ImageVectorStore  # noqa: E402

EMBEDDING_DIM = 8


class FakeEncoder:
    """
    Stands in for ImageTextEncoder: encode_text/encode_images return fixed,
    pre-set embeddings so a test can steer the query toward a known point.
    """

    embedding_dim = EMBEDDING_DIM

    def __init__(self, text_vec=None, image_vec=None, drop_images: bool = False):
        self._text_vec = text_vec
        self._image_vec = image_vec
        self._drop_images = drop_images

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._text_vec for _ in list(texts)], dtype=np.float32)

    def encode_images(
        self, image_paths: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        paths = list(image_paths)
        if self._drop_images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
        return np.array([self._image_vec for _ in paths], dtype=np.float32), paths


@pytest.fixture
def store() -> ImageVectorStore:
    return ImageVectorStore(
        collection_name="test_query_collection",
        embedding_dim=EMBEDDING_DIM,
        path=":memory:",
    )


def _random_unit_embeddings(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
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


# ---- looks_like_image_path ----

def test_looks_like_image_path_true_for_existing_image(tmp_path: Path):
    (path,) = _make_images(tmp_path, ["a.png"])
    assert looks_like_image_path(path) is True


def test_looks_like_image_path_false_for_plain_text():
    assert looks_like_image_path("a red cat on a sofa") is False


def test_looks_like_image_path_false_for_nonexistent_file(tmp_path: Path):
    assert looks_like_image_path(str(tmp_path / "nope.png")) is False


# ---- query_images ----

def test_query_by_text_returns_nearest(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, ["a.png", "b.png", "c.png"])
    embeddings = _random_unit_embeddings(3)
    store.upsert_images(embeddings, paths)

    # Steer the text query embedding to exactly match point 1.
    encoder = FakeEncoder(text_vec=embeddings[1])
    results = query_images("anything", store, encoder, top_k=1)

    assert results[0][0] == paths[1]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_query_by_image_autodetected(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, ["a.png", "b.png"])
    embeddings = _random_unit_embeddings(2)
    store.upsert_images(embeddings, paths)

    # A real image file path -> auto-detected as image mode.
    (ref,) = _make_images(tmp_path, ["ref.png"])
    encoder = FakeEncoder(image_vec=embeddings[0])
    results = query_images(ref, store, encoder, top_k=1)

    assert results[0][0] == paths[0]


def test_query_force_text_on_image_path(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, ["a.png", "b.png"])
    embeddings = _random_unit_embeddings(2)
    store.upsert_images(embeddings, paths)

    # The query is a real image file, but mode="text" must use encode_text.
    (ref,) = _make_images(tmp_path, ["ref.png"])
    encoder = FakeEncoder(text_vec=embeddings[1], image_vec=embeddings[0])
    results = query_images(ref, store, encoder, top_k=1, mode="text")

    assert results[0][0] == paths[1]  # text_vec match, not image_vec match


def test_query_respects_top_k(store: ImageVectorStore, tmp_path: Path):
    paths = _make_images(tmp_path, [f"img_{i}.png" for i in range(5)])
    store.upsert_images(_random_unit_embeddings(5), paths)

    encoder = FakeEncoder(text_vec=_random_unit_embeddings(1)[0])
    results = query_images("anything", store, encoder, top_k=3)

    assert len(results) == 3


def test_query_unreadable_image_raises(store: ImageVectorStore, tmp_path: Path):
    encoder = FakeEncoder(drop_images=True)
    with pytest.raises(ValueError, match="Could not read query image"):
        query_images(str(tmp_path / "broken.png"), store, encoder, mode="image")


def test_query_unknown_mode_raises(store: ImageVectorStore):
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="Unknown query mode"):
        query_images("anything", store, encoder, mode="bogus")


def test_query_empty_collection_returns_empty(store: ImageVectorStore):
    encoder = FakeEncoder(text_vec=_random_unit_embeddings(1)[0])
    assert query_images("anything", store, encoder) == []
