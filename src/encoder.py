"""
encoder.py — thin wrapper around a CLIP-like dual encoder model (default:
SigLIP2), providing L2-normalized image/text embeddings for retrieval.

Kept deliberately small and dependency-light so it can be unit-tested
without a GPU/MPS device and without network access for the pure-Python
logic (device selection, batching, normalization), while still allowing
full model-loading smoke tests when network/model download is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_BATCH_SIZE = 32


def get_device() -> torch.device:
    """Picks the best available device: Apple Silicon (MPS) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalizes embeddings along the last dimension."""
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


class ImageTextEncoder:
    """
    Loads a CLIP-like dual encoder (default: SigLIP2) and exposes
    L2-normalized image/text embedding helpers sharing a single space,
    so text and image embeddings can be compared directly (cosine sim).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[torch.device] = None):
        self.model_id = model_id
        self.device = device or get_device()
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @property
    def embedding_dim(self) -> int:
        """Joint embedding dimension (text/image feature size)."""
        return self.model.config.text_config.hidden_size

    def encode_images(
        self,
        image_paths: Sequence[Union[str, Path]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encodes a list of image paths into L2-normalized embeddings.
        Skips unreadable files rather than raising. Returns
        (embeddings [N, D], valid_paths) — valid_paths may be shorter than
        image_paths if some files failed to load.
        """
        embeddings = []
        valid_paths_all: List[str] = []

        batch_starts = range(0, len(image_paths), batch_size)
        for i in tqdm(
            batch_starts,
            desc="Encoding images",
            unit="batch",
            disable=not show_progress or len(image_paths) <= batch_size,
        ):
            batch_paths = image_paths[i : i + batch_size]

            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(str(path))
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            img_emb = l2_normalize(outputs.pooler_output)
            embeddings.append(img_emb.cpu().numpy())
            valid_paths_all.extend(valid_paths)

        if not embeddings:
            return np.array([]), []

        return np.vstack(embeddings), valid_paths_all

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encodes a list of strings into L2-normalized embeddings [N, D]."""
        inputs = self.processor(text=list(texts), padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        text_emb = l2_normalize(outputs.pooler_output)
        return text_emb.cpu().numpy()

