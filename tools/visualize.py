"""visualize.py — small matplotlib helpers for eyeballing retrieval results in notebooks."""

from __future__ import annotations

import numpy as np
from PIL import Image


def show_results(results, cols: int = 3, figsize_per_image: float = 4) -> None:
    """Displays a grid of Qdrant search result images with their similarity scores."""
    import matplotlib.pyplot as plt

    points = results.points
    n = len(points)

    if n == 0:
        print("No results")
        return

    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))

    for i, hit in enumerate(points):
        img_path = hit.payload["path"]
        score = hit.score
        img = Image.open(img_path).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"#{i + 1} | score: {score:.4f}", fontsize=10)

    plt.tight_layout()
    plt.show()

