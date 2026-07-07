"""
dataloader.py — fetches small, well-known image datasets for testing the
iMem retrieval pipeline and materializes them as plain image files on disk
(so the rest of the pipeline can treat every dataset the same way: a flat
list of image paths + optional labels).

Supported test datasets:
    - Olivetti Faces (via scikit-learn)   -> 400 grayscale 64x64 face personal_images
    - Caltech-101 (via torchvision)       -> ~9k personal_images across 101 object categories

Downloaded/generated data is written under `data/` which is git-ignored.

Usage:
    from src.dataloader import load_olivetti_faces, load_caltech101, ImageRecord

    records = load_olivetti_faces()
    records = load_caltech101()

    for r in records[:5]:
        print(r.path, r.label, r.dataset)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

# Root directory for all locally materialized/downloaded test data.
# Git-ignored via the `data/` entry in .gitignore.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ImageRecord:
    """A single image sample with optional label metadata."""

    path: Path
    label: str
    dataset: str


def load_olivetti_faces(data_dir: Path = DATA_DIR, force: bool = False) -> List[ImageRecord]:
    """
    Fetches the Olivetti Faces dataset (via scikit-learn, cached under
    `data/raw/`) and materializes each sample as a PNG file under
    `data/olivetti/<person_id>/<sample_id>.png`.

    Returns a list of ImageRecord (path, label=person id, dataset="olivetti").
    """
    from sklearn.datasets import fetch_olivetti_faces
    from PIL import Image
    import numpy as np

    out_dir = data_dir / "olivetti"
    if out_dir.exists() and not force:
        return _scan_labeled_dir(out_dir, dataset="olivetti")

    raw_cache = data_dir / "raw"
    raw_cache.mkdir(parents=True, exist_ok=True)

    bunch = fetch_olivetti_faces(data_home=raw_cache, shuffle=False)
    images = bunch.images  # (400, 64, 64) floats in [0, 1]
    targets = bunch.target  # (400,) person ids 0..39

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[ImageRecord] = []
    for idx, (img, person_id) in enumerate(tqdm(list(zip(images, targets)), desc="Writing Olivetti personal_images")):
        person_dir = out_dir / f"person_{person_id:02d}"
        person_dir.mkdir(exist_ok=True)
        arr = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="L").convert("RGB")
        img_path = person_dir / f"sample_{idx:03d}.png"
        pil_img.save(img_path)
        records.append(ImageRecord(path=img_path, label=f"person_{person_id:02d}", dataset="olivetti"))

    return records


def load_caltech101(data_dir: Path = DATA_DIR, download: bool = True) -> List[ImageRecord]:
    """
    Fetches Caltech-101 (via torchvision, cached under `data/raw/`) and
    returns a flat list of ImageRecord pointing directly at the extracted
    category folders (no re-copying needed — torchvision already lays the
    dataset out as `<root>/caltech101/101_ObjectCategories/<class>/*.jpg`).
    """
    from torchvision.datasets import Caltech101

    raw_cache = data_dir / "raw"
    raw_cache.mkdir(parents=True, exist_ok=True)

    # Triggers download+extraction on first run; subsequent runs just
    # validate the cache is present.
    dataset = Caltech101(root=str(raw_cache), download=download)

    categories_root = raw_cache / "caltech101" / "101_ObjectCategories"
    records: List[ImageRecord] = []
    for img_path in tqdm(sorted(categories_root.glob("*/*.jpg")), desc="Indexing Caltech-101 personal_images"):
        label = img_path.parent.name
        records.append(ImageRecord(path=img_path, label=label, dataset="caltech101"))

    return records


def load_all_test_data(data_dir: Path = DATA_DIR) -> List[ImageRecord]:
    """Convenience helper: loads both Olivetti and Caltech-101 and concatenates them."""
    return load_olivetti_faces(data_dir) + load_caltech101(data_dir)


def _scan_labeled_dir(root: Path, dataset: str) -> List[ImageRecord]:
    """Rebuilds ImageRecord list from an already-materialized `<root>/<label>/*` layout."""
    records: List[ImageRecord] = []
    for img_path in sorted(root.glob("*/*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        records.append(ImageRecord(path=img_path, label=img_path.parent.name, dataset=dataset))
    return records


if __name__ == "__main__":
    olivetti = load_olivetti_faces()
    print(f"Olivetti: {len(olivetti)} personal_images")

    caltech = load_caltech101()
    print(f"Caltech-101: {len(caltech)} personal_images")

