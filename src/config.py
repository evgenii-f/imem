"""
config.py — single source of truth for iMem's tunable constants.

Application/model constants live in ``config.yml`` (nested, human-editable);
Qdrant connection & infra settings live in ``config.ini``. Both files sit at
the project root. This module loads them once at import time and exposes them
through a frozen ``CONFIG`` object, so the rest of the codebase never hardcodes
these values.

    from src.config import CONFIG
    CONFIG.encoder.model_id      # "google/siglip2-base-patch16-224"
    CONFIG.qdrant.port           # 6333
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YML = PROJECT_ROOT / "config.yml"
CONFIG_INI = PROJECT_ROOT / "config.ini"


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str
    batch_size: int


@dataclass(frozen=True)
class VectorStoreConfig:
    upsert_batch_size: int
    top_k: int
    distance: str  # cosine | euclid | dot | manhattan


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: Path


@dataclass(frozen=True)
class IndexerConfig:
    extensions: frozenset


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    grpc_port: int
    collection: str
    storage: str


@dataclass(frozen=True)
class Config:
    encoder: EncoderConfig
    vector_store: VectorStoreConfig
    dataset: DatasetConfig
    indexer: IndexerConfig
    qdrant: QdrantConfig


def _load() -> Config:
    with open(CONFIG_YML, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    ini = configparser.ConfigParser()
    if not ini.read(CONFIG_INI, encoding="utf-8"):
        raise FileNotFoundError(f"Config file not found: {CONFIG_INI}")
    q = ini["qdrant"]

    return Config(
        encoder=EncoderConfig(
            model_id=str(y["encoder"]["model_id"]),
            batch_size=int(y["encoder"]["batch_size"]),
        ),
        vector_store=VectorStoreConfig(
            upsert_batch_size=int(y["vector_store"]["upsert_batch_size"]),
            top_k=int(y["vector_store"]["top_k"]),
            distance=str(y["vector_store"]["distance"]),
        ),
        dataset=DatasetConfig(
            # Resolve relative to the project root so callers get an absolute path.
            data_dir=(PROJECT_ROOT / str(y["dataset"]["data_dir"])).resolve(),
        ),
        indexer=IndexerConfig(
            extensions=frozenset(str(ext).lower() for ext in y["indexer"]["extensions"]),
        ),
        qdrant=QdrantConfig(
            host=q.get("host"),
            port=q.getint("port"),
            grpc_port=q.getint("grpc_port"),
            collection=q.get("collection"),
            storage=q.get("storage"),
        ),
    )


CONFIG = _load()
