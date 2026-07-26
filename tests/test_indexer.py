"""
Integration tests for the indexing pipeline (imem/indexer.py).

Uses Qdrant's in-memory mode and a FakeEncoder (random embeddings) instead of
loading a real model, so these run fast without network/GPU access — same
approach as tests/test_vector_store.py. index_folders() only relies on
encoder.embedding_dim and encoder.encode_images(), so a fake stand-in is
enough to exercise the real orchestration logic (discovery, skip-existing,
upsert, failure reporting).

Run with: pytest tests/test_indexer.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("qdrant_client")
pytest.importorskip("blake3")

from conftest import EMBEDDING_DIM, FakeEncoder  # noqa: E402
from imem.indexer import IndexReport, _parse_extensions, index_folders, iter_image_paths  # noqa: E402
from imem.vector_store import ImageVectorStore  # noqa: E402


@pytest.fixture
def store() -> ImageVectorStore:
    return ImageVectorStore(
        collection_name="test_indexer_collection",
        embedding_dim=EMBEDDING_DIM,
        path=":memory:",
    )


def _write_image(path: Path, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


# ---- iter_image_paths ----

def test_iter_image_paths_finds_nested_images(tmp_path: Path):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "sub" / "b.jpg")
    _write_image(tmp_path / "sub" / "deeper" / "c.jpeg")

    found = iter_image_paths([str(tmp_path)])

    assert sorted(found) == sorted(
        str(p) for p in [tmp_path / "a.png", tmp_path / "sub" / "b.jpg", tmp_path / "sub" / "deeper" / "c.jpeg"]
    )


def test_iter_image_paths_ignores_non_matching_extensions(tmp_path: Path):
    _write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("not an image")

    found = iter_image_paths([str(tmp_path)])

    assert found == [str(tmp_path / "a.png")]


def test_iter_image_paths_extension_match_is_case_insensitive(tmp_path: Path):
    _write_image(tmp_path / "a.PNG")

    found = iter_image_paths([str(tmp_path)])

    assert found == [str(tmp_path / "a.PNG")]


def test_iter_image_paths_dedups_overlapping_folders(tmp_path: Path):
    _write_image(tmp_path / "sub" / "a.png")

    found = iter_image_paths([str(tmp_path), str(tmp_path / "sub")])

    assert found == [str(tmp_path / "sub" / "a.png")]


def test_iter_image_paths_respects_custom_extensions(tmp_path: Path):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.jpg")

    found = iter_image_paths([str(tmp_path)], extensions=frozenset({".png"}))

    assert found == [str(tmp_path / "a.png")]


def test_iter_image_paths_empty_folder_returns_empty_list(tmp_path: Path):
    assert iter_image_paths([str(tmp_path)]) == []


def test_iter_image_paths_are_absolute_for_relative_folder(tmp_path: Path, monkeypatch):
    # Indexing from a relative folder arg must still store absolute paths, so the
    # collection is portable regardless of the CWD it was indexed from.
    _write_image(tmp_path / "a.png")
    monkeypatch.chdir(tmp_path)
    found = iter_image_paths(["."])
    assert found and all(Path(p).is_absolute() for p in found)


# ---- index_folders ----

def test_index_folders_indexes_new_images(store: ImageVectorStore, tmp_path: Path):
    _write_image(tmp_path / "a.png", color=(1, 2, 3))
    _write_image(tmp_path / "b.png", color=(4, 5, 6))

    report = index_folders([str(tmp_path)], store, FakeEncoder())

    assert report == IndexReport(found=2, skipped_existing=0, indexed=2, failed=[])
    assert store.count() == 2


def test_index_folders_streams_in_chunks(store: ImageVectorStore, tmp_path: Path):
    # 5 images with chunk_size=2 -> 3 encode+upsert rounds, all persisted.
    for i in range(5):
        _write_image(tmp_path / f"c_{i}.png", color=(i, i * 2, i * 3))
    encoder = FakeEncoder()

    report = index_folders([str(tmp_path)], store, encoder, chunk_size=2)

    assert encoder.calls == 3  # ceil(5 / 2)
    assert report.indexed == 5
    assert store.count() == 5


def test_index_folders_single_chunk_when_under_chunk_size(store: ImageVectorStore, tmp_path: Path):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.png", color=(9, 9, 9))
    encoder = FakeEncoder()

    index_folders([str(tmp_path)], store, encoder, chunk_size=100)

    assert encoder.calls == 1


def test_index_folders_skips_already_indexed_paths(store: ImageVectorStore, tmp_path: Path):
    _write_image(tmp_path / "a.png")
    index_folders([str(tmp_path)], store, FakeEncoder())

    report = index_folders([str(tmp_path)], store, FakeEncoder())

    assert report == IndexReport(found=1, skipped_existing=1, indexed=0, failed=[])
    assert store.count() == 1


def test_index_folders_only_encodes_new_paths(store: ImageVectorStore, tmp_path: Path):
    _write_image(tmp_path / "a.png")
    index_folders([str(tmp_path)], store, FakeEncoder())

    _write_image(tmp_path / "b.png")
    report = index_folders([str(tmp_path)], store, FakeEncoder())

    assert report == IndexReport(found=2, skipped_existing=1, indexed=1, failed=[])
    assert store.count() == 2


def test_index_folders_reports_failed_paths(store: ImageVectorStore, tmp_path: Path):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    _write_image(good)
    _write_image(bad)

    report = index_folders([str(tmp_path)], store, FakeEncoder(drop_paths=[str(bad)]))

    assert report.found == 2
    assert report.indexed == 1
    assert report.failed == [str(bad)]
    assert store.count() == 1


def test_index_folders_no_images_found(store: ImageVectorStore, tmp_path: Path):
    report = index_folders([str(tmp_path)], store, FakeEncoder())

    assert report == IndexReport(found=0, skipped_existing=0, indexed=0, failed=[])
    assert store.count() == 0


# ---- _parse_extensions ----

def test_parse_extensions_none_returns_default():
    from imem.indexer import DEFAULT_EXTENSIONS

    assert _parse_extensions(None) == DEFAULT_EXTENSIONS


def test_parse_extensions_adds_missing_dot():
    assert _parse_extensions("jpg,png") == frozenset({".jpg", ".png"})


def test_parse_extensions_normalizes_case_and_whitespace():
    assert _parse_extensions(" .JPG , Png ") == frozenset({".jpg", ".png"})
