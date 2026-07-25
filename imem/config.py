"""
config.py — single source of truth for iMem's tunable constants.

Application/model constants live in ``config.yml`` (nested, human-editable);
Qdrant connection & infra settings live in ``config.ini``. Both files ship
inside the package and are loaded via ``importlib.resources`` (so they resolve
correctly whether installed as a wheel or run from source). This module loads
them once at import time and exposes them through a frozen ``CONFIG`` object, so
the rest of the codebase never hardcodes these values.

    from imem.config import CONFIG
    CONFIG.encoder.model_id      # "google/siglip2-base-patch16-224"
    CONFIG.qdrant.port           # 6333
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

_PACKAGE = "imem"


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
    chunk_size: int
    extensions: frozenset


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    grpc_port: int
    collection: str
    storage: str


@dataclass(frozen=True)
class ApiConfig:
    # Base dir for resolving relative stored image paths; "" = use CWD.
    base_dir: str


@dataclass(frozen=True)
class Config:
    encoder: EncoderConfig
    vector_store: VectorStoreConfig
    dataset: DatasetConfig
    indexer: IndexerConfig
    qdrant: QdrantConfig
    api: ApiConfig


def _load() -> Config:
    resources = files(_PACKAGE)
    y = yaml.safe_load(resources.joinpath("config.yml").read_text(encoding="utf-8"))

    ini = configparser.ConfigParser()
    ini.read_string(resources.joinpath("config.ini").read_text(encoding="utf-8"))
    q = ini["qdrant"]
    # base_dir is machine-specific, so allow an env override (IMEM_BASE_DIR)
    # that wins over the tracked config.ini value.
    api_base_dir = os.environ.get("IMEM_BASE_DIR")
    if api_base_dir is None:
        api_base_dir = ini.get("api", "base_dir", fallback="") if ini.has_section("api") else ""
    api_base_dir = api_base_dir.strip()

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
            # Dev-only location for materialized test datasets (used by
            # tools/dataloader.py); kept relative, resolved by callers against CWD.
            data_dir=Path(str(y["dataset"]["data_dir"])),
        ),
        indexer=IndexerConfig(
            chunk_size=int(y["indexer"]["chunk_size"]),
            extensions=frozenset(str(ext).lower() for ext in y["indexer"]["extensions"]),
        ),
        qdrant=QdrantConfig(
            host=q.get("host"),
            port=q.getint("port"),
            grpc_port=q.getint("grpc_port"),
            collection=q.get("collection"),
            storage=q.get("storage"),
        ),
        api=ApiConfig(base_dir=api_base_dir),
    )


CONFIG = _load()
