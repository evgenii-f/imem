"""Pydantic request/response models for the iMem HTTP API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    model_loaded: bool
    qdrant_reachable: bool


class ConfigResponse(BaseModel):
    default_top_k: int
    default_collection: str
    extensions: List[str]
    model_id: str
    distance: str


class CollectionInfo(BaseModel):
    name: str
    count: int


class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]


class QueryRequest(BaseModel):
    collection: str
    query: str
    mode: Literal["auto", "text", "image"] = "auto"
    top_k: Optional[int] = Field(default=None, ge=1)


class ResultItem(BaseModel):
    path: str
    score: float


class QueryResponse(BaseModel):
    results: List[ResultItem]
    mode: Literal["text", "image"]
