"""
Smoke tests for the image/text encoder pipeline (imem/encoder.py).

These load the real SigLIP2 model (auto-downloaded from Hugging Face on
first run), so they require network access + the ML dependencies from
requirements.txt. They're intentionally basic: the goal is to catch
pipeline breakage (wrong shapes, broken normalization, crashes on bad
input) rather than to assert on retrieval quality.

Run with: pytest tests/test_encoder.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from imem.encoder import ImageTextEncoder, get_device, l2_normalize  # noqa: E402


def test_get_device_returns_valid_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}


def test_l2_normalize_produces_unit_vectors():
    x = torch.randn(4, 16)
    normed = l2_normalize(x)
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


@pytest.fixture(scope="module")
def encoder() -> ImageTextEncoder:
    """Loads the encoder once and reuses it across tests in this module."""
    return ImageTextEncoder(device=torch.device("cpu"))


def test_encoder_embedding_dim_matches_model_config(encoder: ImageTextEncoder):
    assert encoder.embedding_dim == encoder.model.config.text_config.hidden_size
    assert encoder.embedding_dim > 0


def test_encode_images_shape_and_normalization(encoder: ImageTextEncoder, sample_images):
    embeddings, valid_paths = encoder.encode_images(sample_images)

    assert embeddings.shape == (len(sample_images), encoder.embedding_dim)
    assert valid_paths == sample_images

    norms = np.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, np.ones(len(sample_images)), atol=1e-4)


def test_encode_images_skips_unreadable_files(encoder: ImageTextEncoder, sample_images, tmp_path):
    bad_path = tmp_path / "not_an_image.txt"
    bad_path.write_text("this is not an image")

    paths = sample_images + [str(bad_path)]
    embeddings, valid_paths = encoder.encode_images(paths)

    assert str(bad_path) not in valid_paths
    assert len(valid_paths) == len(sample_images)
    assert embeddings.shape == (len(sample_images), encoder.embedding_dim)


def test_encode_images_empty_list_returns_empty(encoder: ImageTextEncoder):
    embeddings, valid_paths = encoder.encode_images([])
    assert valid_paths == []
    assert embeddings.size == 0


def test_encode_text_shape_and_normalization(encoder: ImageTextEncoder):
    texts = ["a red square", "a green square", "a blue square"]
    embeddings = encoder.encode_text(texts)

    assert embeddings.shape == (len(texts), encoder.embedding_dim)

    norms = np.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-4)


def test_encode_text_is_deterministic(encoder: ImageTextEncoder):
    emb1 = encoder.encode_text(["a photo of a cat"])
    emb2 = encoder.encode_text(["a photo of a cat"])
    np.testing.assert_allclose(emb1, emb2, atol=1e-6)

