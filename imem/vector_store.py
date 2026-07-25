from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
from blake3 import blake3
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from .config import CONFIG

DEFAULT_UPSERT_BATCH_SIZE = CONFIG.vector_store.upsert_batch_size

# Indexed payload fields used for de-duplication lookups.
FILE_HASH_FIELD = "file_hash"  # BLAKE3 of the raw file bytes
PATH_FIELD = "path"            # source file path

# Bytes read per chunk when streaming a file through the hasher.
_HASH_CHUNK = 1 << 20  # 1 MiB

# Maps the string metric from config.yml to Qdrant's Distance enum.
_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
    "manhattan": Distance.MANHATTAN,
}


def file_hash(path: str) -> str:
    """
    Returns the BLAKE3 hex digest of a file's raw bytes.

    Cheap (no image decode) and exact: it detects byte-identical duplicate
    files regardless of their path. It does not recognize the same picture
    re-encoded to a different format/quality as a duplicate.
    """
    hasher = blake3()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def point_id_from_path(path: str) -> str:
    """
    Deterministic Qdrant point id (UUID) derived from the source path. The id is
    intentionally path-based, not content-based: re-indexing the same path
    updates its point in place, while the same file under a different path is
    stored as a separate point.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class ImageVectorStore:
    """
    Thin wrapper around Qdrant for storing/searching image embeddings.

    Each point is keyed by a path-derived id and carries two indexed payload
    fields: `path` (the source file, for display/retrieval — the image itself is
    not stored) and `file_hash` (BLAKE3 of the file's bytes, for detecting
    duplicate files regardless of path).
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = CONFIG.qdrant.port,
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
                    distance=_DISTANCE_MAP[CONFIG.vector_store.distance.lower()],
                ),
            )
            # Index both lookup fields so duplicate checks match quickly. Payload
            # indexes only take effect on a real Qdrant server, so skip the no-op
            # (and its warning) in local/in-memory mode.
            if host is not None:
                for field in (FILE_HASH_FIELD, PATH_FIELD):
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )

    # ---- content-based duplicate checks (by file bytes) ----

    def image_exist(self, path: str) -> bool:
        """Whether a file with the same bytes as `path` is already indexed."""
        digest = file_hash(path)
        hits, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key=FILE_HASH_FIELD, match=MatchValue(value=digest))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(hits) > 0

    def image_exists(self, paths: Sequence[str]) -> Dict[str, bool]:
        """
        Batched content check: BLAKE3-hashes each file and resolves all distinct
        hashes in a single indexed query. Returns a dict mapping each input path
        to whether a file with that content is indexed. Unreadable files are
        reported as not-indexed.
        """
        paths = list(paths)
        if not paths:
            return {}

        path_to_hash: Dict[str, Optional[str]] = {}
        for p in paths:
            try:
                path_to_hash[p] = file_hash(p)
            except Exception as e:
                print(f"Failed to hash {p}: {e}")
                path_to_hash[p] = None

        digests = sorted({h for h in path_to_hash.values() if h is not None})
        present: set = set()
        if digests:
            hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key=FILE_HASH_FIELD, match=MatchAny(any=digests))]
                ),
                limit=len(digests),
                with_payload=[FILE_HASH_FIELD],
                with_vectors=False,
            )
            present = {hit.payload[FILE_HASH_FIELD] for hit in hits}

        return {p: (h is not None and h in present) for p, h in path_to_hash.items()}

    # ---- path-based presence checks (by source path) ----

    def path_exist(self, path: str) -> bool:
        """Whether the given source path is already indexed."""
        hits, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key=PATH_FIELD, match=MatchValue(value=path))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(hits) > 0

    def path_exists(self, paths: Sequence[str]) -> Dict[str, bool]:
        """
        Batched path check: looks up all given paths in a single indexed query.
        Returns a dict mapping each input path to whether it's indexed.
        """
        paths = list(paths)
        if not paths:
            return {}

        hits, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key=PATH_FIELD, match=MatchAny(any=paths))]
            ),
            limit=len(paths),
            with_payload=[PATH_FIELD],
            with_vectors=False,
        )
        present = {hit.payload[PATH_FIELD] for hit in hits}
        return {p: p in present for p in paths}

    def filter_new_paths(self, paths: Sequence[str]) -> List[str]:
        """Returns the subset of `paths` not yet present in the store (by path)."""
        existence = self.path_exists(paths)
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
        n = len(paths)
        n_batches = (n + batch_size - 1) // batch_size
        # Progress in images with the batch index as a postfix, mirroring
        # ImageTextEncoder.encode_images.
        with tqdm(
            total=n,
            desc="Upserting images",
            unit="img",
            disable=not show_progress or n <= batch_size,
        ) as pbar:
            for batch_idx, start in enumerate(range(0, n, batch_size), 1):
                end = start + batch_size
                batch_embeddings = embeddings[start:end]
                batch_paths = paths[start:end]

                points = []
                for i, path in enumerate(batch_paths):
                    # Hash the file bytes for the payload. Skip unreadable files
                    # rather than aborting the whole batch.
                    try:
                        digest = file_hash(path)
                    except Exception as e:
                        print(f"Failed to hash {path}: {e}")
                        continue
                    points.append(
                        PointStruct(
                            id=point_id_from_path(path),
                            vector=batch_embeddings[i].tolist(),
                            payload={PATH_FIELD: path, FILE_HASH_FIELD: digest},
                        )
                    )

                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    total_written += len(points)

                elapsed = pbar.format_dict["elapsed"]
                batch_rate = batch_idx / elapsed if elapsed else 0.0
                pbar.set_postfix(
                    {"batch": f"{batch_idx}/{n_batches}", "batch/s": f"{batch_rate:.2f}"},
                    refresh=False,
                )
                pbar.update(len(batch_paths))

        return total_written

    def search(self, query_embedding: np.ndarray, top_k: int = CONFIG.vector_store.top_k):
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
