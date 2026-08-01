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
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .config import CONFIG
from .encoder import ImageTextEncoder
from .vector_store import ImageVectorStore

DEFAULT_EXTENSIONS = CONFIG.indexer.extensions
DEFAULT_CHUNK_SIZE = CONFIG.indexer.chunk_size
DEFAULT_MIN_CHANNELS = CONFIG.indexer.min_channels
DEFAULT_MIN_HEIGHT = CONFIG.indexer.min_height
DEFAULT_MIN_WIDTH = CONFIG.indexer.min_width


def iter_image_paths(
    folders: Sequence[Union[str, Path]],
    extensions: FrozenSet[str] = DEFAULT_EXTENSIONS,
) -> List[str]:
    """
    Recursively walks each folder and returns the paths of files whose
    extension (case-insensitive) is in `extensions`, deduplicated and sorted
    for a stable order across runs.

    Paths are made absolute (via ``Path.absolute()`` — no symlink resolution)
    so a collection is portable regardless of the working directory it was
    indexed from. The API serves images by these stored paths, so relative
    paths would only resolve when the server ran from the indexing directory.
    """
    found = set()
    for folder in folders:
        for path in Path(folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                found.add(str(path.absolute()))
    return sorted(found)


@dataclass
class IndexReport:
    """Summary of an index_folders() run."""

    found: int
    skipped_existing: int
    indexed: int
    skipped_small: int = 0
    failed: List[str] = field(default_factory=list)


def _filter_min_resolution(
    paths: Sequence[str],
    min_channels: int,
    min_height: int,
    min_width: int,
) -> Tuple[List[str], List[str]]:
    """
    Partitions `paths` into (kept, skipped_small) by minimum channels/height/width.

    Reads only image headers (no pixel decode), counting channels as image bands
    (grayscale/2-D = 1, RGB = 3, RGBA = 4). Unreadable files are kept and left for
    the encoder to report as failures, so "too small" stays distinct from "broken".
    """
    kept: List[str] = []
    skipped: List[str] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                channels = len(img.getbands())
        except Exception:
            kept.append(path)  # defer to the encoder, which reports it as failed
            continue
        if channels >= min_channels and height >= min_height and width >= min_width:
            kept.append(path)
        else:
            skipped.append(path)
    return kept, skipped


def index_folders(
    folders: Sequence[Union[str, Path]],
    store: ImageVectorStore,
    encoder: ImageTextEncoder,
    extensions: FrozenSet[str] = DEFAULT_EXTENSIONS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_channels: int = DEFAULT_MIN_CHANNELS,
    min_height: int = DEFAULT_MIN_HEIGHT,
    min_width: int = DEFAULT_MIN_WIDTH,
) -> IndexReport:
    """
    Discovers images under `folders`, skips paths already present in `store`,
    then encodes and upserts the rest in chunks of `chunk_size`.

    Each chunk is encoded and immediately upserted before the next is read, so
    the collection is populated incrementally (an interrupted run keeps every
    completed chunk) and peak memory stays bounded to one chunk of embeddings.

    Images below `min_channels` / `min_height` / `min_width` are skipped and
    counted in the report. The check reads only image headers and is skipped
    entirely when all minimums are at their trivial default of 1 (no overhead).
    """
    paths = iter_image_paths(folders, extensions)
    new_paths = store.filter_new_paths(paths)

    filter_active = min_channels > 1 or min_height > 1 or min_width > 1

    indexed = 0
    skipped_small = 0
    failed: List[str] = []
    n_chunks = (len(new_paths) + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(range(0, len(new_paths), chunk_size), 1):
        chunk = new_paths[start : start + chunk_size]

        if filter_active:
            chunk, too_small = _filter_min_resolution(chunk, min_channels, min_height, min_width)
            skipped_small += len(too_small)
        if not chunk:
            continue

        desc = "Encoding images" if n_chunks == 1 else f"Encoding images [chunk {chunk_idx}/{n_chunks}]"
        embeddings, valid_paths = encoder.encode_images(chunk, desc=desc)
        failed.extend(p for p in chunk if p not in set(valid_paths))
        if valid_paths:
            indexed += store.upsert_images(embeddings, valid_paths, show_progress=False)

    return IndexReport(
        found=len(paths),
        skipped_existing=len(paths) - len(new_paths),
        indexed=indexed,
        skipped_small=skipped_small,
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
