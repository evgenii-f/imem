from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

DEFAULT_UPSERT_BATCH_SIZE = 256


def point_id_from_path(path: str) -> int:
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:16], 16)


class ImageVectorStore:
    """
    Thin wrapper around Qdrant for storing/searching image embeddings.
    Each point carries a `path` payload field used for de-duplication and display.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6333,
        recreate: bool = False,
    ):
        if host is not None:
            self.client = QdrantClient(host=host, port=port)
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        existing = {c.name for c in self.client.get_collections().collections}

        if recreate and collection_name in existing:
            self.client.delete_collection(collection_name)
            existing.discard(collection_name)

        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def image_exists(self, path: str) -> bool:
        hits, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="path",
                        match=MatchValue(value=path),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return len(hits) > 0

    def images_exist(self, paths: Sequence[str]) -> Dict[str, bool]:
        """
        Batched existence check: looks up all given paths in a single
        Qdrant query (via MatchAny) instead of one round-trip per path.
        Returns a dict mapping each input path to whether it's indexed.
        """
        paths = list(paths)
        if not paths:
            return {}

        hits, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="path",
                        match=MatchAny(any=paths),
                    )
                ]
            ),
            limit=len(paths),
            with_payload=True,
            with_vectors=False,
        )
        existing_paths = {hit.payload["path"] for hit in hits}
        return {path: path in existing_paths for path in paths}

    def filter_new_paths(self, paths: Sequence[str]) -> List[str]:
        existence = self.images_exist(paths)
        return [p for p in paths if not existence[p]]

    def upsert_images(
        self,
        embeddings: np.ndarray,
        paths: Sequence[str],
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> int:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

        if len(embeddings) != len(paths):
            raise ValueError(
                f"Number of embeddings and paths must match: "
                f"{len(embeddings)} != {len(paths)}"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        total_written = 0
        batch_starts = range(0, len(paths), batch_size)
        for start in tqdm(
            batch_starts,
            desc="Upserting images",
            unit="batch",
            disable=not show_progress or len(paths) <= batch_size,
        ):
            end = start + batch_size
            batch_embeddings = embeddings[start:end]
            batch_paths = paths[start:end]

            points = [
                PointStruct(
                    id=point_id_from_path(path),
                    vector=batch_embeddings[i].tolist(),
                    payload={"path": path},
                )
                for i, path in enumerate(batch_paths)
            ]

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                total_written += len(points)

        return total_written

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = np.asarray(query_embedding)

        if query_embedding.ndim != 1:
            raise ValueError(
                f"Expected a single 1D query embedding, got shape {query_embedding.shape}"
            )

        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {query_embedding.shape[0]}"
            )

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection_name,
        ).count
