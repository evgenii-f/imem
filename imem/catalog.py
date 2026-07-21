"""
catalog.py — instance-level Qdrant operations that span collections
(listing, existence, deletion), as opposed to ImageVectorStore which is bound
to a single collection.

These take a bare QdrantClient rather than an ImageVectorStore: constructing a
store get-or-creates its collection as a side effect, which is wrong for
read-only catalog queries. Keeping them as free functions over the client
mirrors indexer/query and stays testable against an in-memory client.
"""

from __future__ import annotations

from typing import List, Tuple

from qdrant_client import QdrantClient


def list_collections(client: QdrantClient) -> List[Tuple[str, int]]:
    """Returns (name, point_count) for every collection, sorted by name."""
    names = sorted(c.name for c in client.get_collections().collections)
    return [(name, client.count(collection_name=name).count) for name in names]


def collection_exists(client: QdrantClient, name: str) -> bool:
    """Whether a collection with the given name exists."""
    return name in {c.name for c in client.get_collections().collections}


def collection_count(client: QdrantClient, name: str) -> int:
    """Number of points in the named collection."""
    return client.count(collection_name=name).count


def drop_collection(client: QdrantClient, name: str) -> bool:
    """
    Deletes the named collection if it exists. Returns True if a collection was
    deleted, False if there was nothing to delete.
    """
    if not collection_exists(client, name):
        return False
    client.delete_collection(collection_name=name)
    return True
