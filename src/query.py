"""
query.py — text/image query against an indexed collection.

Encodes a query (either free text or a reference image) into the shared
SigLIP embedding space and returns the nearest indexed image paths. Pure
orchestration over ImageTextEncoder + ImageVectorStore, mirroring
indexer.index_folders — the CLI (src/cli.py) wires argument parsing on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import CONFIG
from .encoder import ImageTextEncoder
from .vector_store import ImageVectorStore

# Extensions that mark a query string as an image path (shared with the indexer).
IMAGE_EXTENSIONS = CONFIG.indexer.extensions


def looks_like_image_path(query: str) -> bool:
    """Whether `query` points at an existing file with a known image extension."""
    path = Path(query)
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()


def query_images(
    query: str,
    store: ImageVectorStore,
    encoder: ImageTextEncoder,
    top_k: int = CONFIG.vector_store.top_k,
    mode: str = "auto",
) -> List[Tuple[str, float]]:
    """
    Encodes `query` and returns the top_k nearest indexed images as
    (path, score) tuples, best match first.

    `mode` selects how `query` is interpreted:
        "auto"  — image if it points at a readable image file, else text
        "text"  — always treat `query` as free text
        "image" — always treat `query` as an image file path
    """
    if mode not in {"auto", "text", "image"}:
        raise ValueError(f"Unknown query mode: {mode!r}")

    if mode == "auto":
        mode = "image" if looks_like_image_path(query) else "text"

    if mode == "image":
        embeddings, valid_paths = encoder.encode_images([query], show_progress=False)
        if len(valid_paths) == 0:
            raise ValueError(f"Could not read query image: {query}")
        query_embedding = embeddings[0]
    else:
        query_embedding = encoder.encode_text([query])[0]

    result = store.search(query_embedding, top_k=top_k)
    return [(point.payload["path"], point.score) for point in result.points]
