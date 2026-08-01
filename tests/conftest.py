"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

EMBEDDING_DIM = 8


class FakeEncoder:
    """
    Shared test double for ImageTextEncoder — no real model, GPU, or network.

    - ``text_vec`` / ``image_vec``: a fixed embedding to return, so a test can
      steer a query toward a known point. When ``None``, random embeddings are
      returned instead.
    - ``drop_paths``: image paths to treat as unreadable (exercises the indexer's
      failure reporting).
    - ``drop_images``: treat every image path as unreadable.
    - ``.calls``: number of ``encode_images`` calls, to assert chunked streaming.
    """

    embedding_dim = EMBEDDING_DIM

    def __init__(
        self,
        text_vec=None,
        image_vec=None,
        drop_paths: Sequence[str] = (),
        drop_images: bool = False,
    ):
        self._text_vec = text_vec
        self._image_vec = image_vec
        self._drop_paths = set(drop_paths)
        self._drop_images = drop_images
        self.calls = 0

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        n = len(list(texts))
        if self._text_vec is None:
            return np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        return np.array([self._text_vec for _ in range(n)], dtype=np.float32)

    def encode_images(
        self, image_paths: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        self.calls += 1
        if self._drop_images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
        valid_paths = [p for p in image_paths if p not in self._drop_paths]
        if self._image_vec is None:
            embeddings = np.random.randn(len(valid_paths), EMBEDDING_DIM).astype(np.float32)
        else:
            embeddings = np.array([self._image_vec for _ in valid_paths], dtype=np.float32)
        return embeddings, valid_paths


@pytest.fixture
def sample_images(tmp_path: Path) -> List[str]:
    """
    Creates a handful of small synthetic solid-color images on disk and
    returns their paths as strings. Used as lightweight stand-ins for real
    photos in encoder smoke tests, so tests don't depend on the test
    datasets being downloaded first.
    """
    colors = {
        "red.png": (220, 20, 60),
        "green.png": (34, 139, 34),
        "blue.png": (30, 60, 220),
    }
    paths = []
    for name, color in colors.items():
        img = Image.new("RGB", (64, 64), color=color)
        path = tmp_path / name
        img.save(path)
        paths.append(str(path))
    return paths

