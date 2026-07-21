"""
Unit tests for instance-level collection operations (imem/catalog.py).

Uses a bare in-memory QdrantClient with collections created directly (no
ImageVectorStore, no images) so these stay focused on the catalog behavior:
listing, existence, counting, deletion.

Run with: pytest tests/test_catalog.py -v
"""

from __future__ import annotations

from typing import Dict

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from imem.catalog import (  # noqa: E402
    collection_count,
    collection_exists,
    drop_collection,
    list_collections,
)

DIM = 4


def _make_client(collections: Dict[str, int]) -> QdrantClient:
    """Builds an in-memory client with the given {collection_name: num_points}."""
    client = QdrantClient(":memory:")
    for name, n_points in collections.items():
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        if n_points:
            points = [
                PointStruct(id=i, vector=[1.0, 0.0, 0.0, float(i)]) for i in range(n_points)
            ]
            client.upsert(collection_name=name, points=points)
    return client


# ---- list_collections ----

def test_list_collections_returns_names_and_counts():
    client = _make_client({"images": 3, "personal": 1})
    assert list_collections(client) == [("images", 3), ("personal", 1)]


def test_list_collections_sorted_by_name():
    client = _make_client({"zebra": 0, "alpha": 0, "mango": 0})
    names = [name for name, _ in list_collections(client)]
    assert names == ["alpha", "mango", "zebra"]


def test_list_collections_empty():
    client = QdrantClient(":memory:")
    assert list_collections(client) == []


# ---- collection_exists / collection_count ----

def test_collection_exists():
    client = _make_client({"images": 2})
    assert collection_exists(client, "images") is True
    assert collection_exists(client, "missing") is False


def test_collection_count():
    client = _make_client({"images": 5})
    assert collection_count(client, "images") == 5


# ---- drop_collection ----

def test_drop_collection_removes_existing():
    client = _make_client({"images": 2, "keep": 1})

    deleted = drop_collection(client, "images")

    assert deleted is True
    assert collection_exists(client, "images") is False
    assert collection_exists(client, "keep") is True


def test_drop_collection_missing_returns_false():
    client = _make_client({"keep": 1})

    deleted = drop_collection(client, "nope")

    assert deleted is False
    assert collection_exists(client, "keep") is True
