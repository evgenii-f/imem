"""Tests for config loading (imem/config.py)."""

from __future__ import annotations

import importlib


def test_base_dir_env_override(monkeypatch):
    # IMEM_BASE_DIR overrides the tracked config.ini [api] base_dir.
    monkeypatch.setenv("IMEM_BASE_DIR", "/tmp/imem-base")
    import imem.config as cfg

    try:
        importlib.reload(cfg)
        assert cfg.CONFIG.api.base_dir == "/tmp/imem-base"
    finally:
        # Restore the module to its env-free state for other tests.
        monkeypatch.delenv("IMEM_BASE_DIR", raising=False)
        importlib.reload(cfg)


def test_base_dir_default_empty():
    import imem.config as cfg

    assert cfg.CONFIG.api.base_dir == ""


def test_indexer_min_resolution_defaults():
    # Single home for the production minimums; test_indexer exercises the filter
    # logic with its own small thresholds rather than depending on these numbers.
    import imem.config as cfg

    idx = cfg.CONFIG.indexer
    assert idx.min_channels == 1
    assert idx.min_height == 128
    assert idx.min_width == 128
