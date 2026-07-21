"""
indexer.py — recursively discovers image files under one or more folders and
indexes them into a Qdrant collection via ImageVectorStore.

Orchestrates the existing pieces rather than reinventing them: file discovery
here, de-duplication and payload (path + file_hash) handled by
ImageVectorStore, embedding computed by ImageTextEncoder.

Usage:
    from imem.indexer import index_folders
    from imem.encoder import ImageTextEncoder
    from imem.vector_store import ImageVectorStore

    encoder = ImageTextEncoder()
    store = ImageVectorStore("images", encoder.embedding_dim, host="localhost")
    report = index_folders(["/path/to/photos"], store, encoder)

Driven from the command line via `python -m imem.cli add` (see imem/cli.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Union

from .config import CONFIG
from .encoder import ImageTextEncoder
from .vector_store import ImageVectorStore

DEFAULT_EXTENSIONS = CONFIG.indexer.extensions


def iter_image_paths(
    folders: Sequence[Union[str, Path]],
    extensions: FrozenSet[str] = DEFAULT_EXTENSIONS,
) -> List[str]:
    """
    Recursively walks each folder and returns the paths of files whose
    extension (case-insensitive) is in `extensions`, deduplicated and sorted
    for a stable order across runs.
    """
    found = set()
    for folder in folders:
        for path in Path(folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                found.add(str(path))
    return sorted(found)


@dataclass
class IndexReport:
    """Summary of an index_folders() run."""

    found: int
    skipped_existing: int
    indexed: int
    failed: List[str] = field(default_factory=list)


def index_folders(
    folders: Sequence[Union[str, Path]],
    store: ImageVectorStore,
    encoder: ImageTextEncoder,
    extensions: FrozenSet[str] = DEFAULT_EXTENSIONS,
) -> IndexReport:
    """
    Discovers images under `folders`, skips paths already present in `store`,
    encodes and upserts the rest. Returns a report of what happened.
    """
    paths = iter_image_paths(folders, extensions)
    new_paths = store.filter_new_paths(paths)

    indexed = 0
    failed: List[str] = []
    if new_paths:
        embeddings, valid_paths = encoder.encode_images(new_paths)
        failed = [p for p in new_paths if p not in set(valid_paths)]
        if valid_paths:
            indexed = store.upsert_images(embeddings, valid_paths)

    return IndexReport(
        found=len(paths),
        skipped_existing=len(paths) - len(new_paths),
        indexed=indexed,
        failed=failed,
    )


def _parse_extensions(raw: Optional[str]) -> FrozenSet[str]:
    """Parses a comma-separated `--extensions` value, tolerating a missing leading dot."""
    if raw is None:
        return DEFAULT_EXTENSIONS
    extensions = set()
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        extensions.add(part if part.startswith(".") else f".{part}")
    return frozenset(extensions)
