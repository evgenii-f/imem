"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from PIL import Image


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

