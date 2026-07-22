"""
app.py — FastAPI application wrapping the imem library for the frontend.

A thin HTTP layer over imem's existing functions:
    - catalog.list_collections / collection_exists  -> /collections
    - query.query_images                            -> /query, /query/upload
    - encoder.ImageTextEncoder                      -> loaded once, shared
    - config.CONFIG                                 -> /config defaults

The SigLIP2 encoder is heavy to load, so it is created once in the lifespan and
shared across requests via app.state. A ``store_factory`` on app.state builds a
per-collection ImageVectorStore; it is an injection seam so tests can back the
routes with an in-memory Qdrant instead of a live server.

Run with: ``imem serve`` (see imem/cli.py) or ``uvicorn imem.api.app:app``.
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from qdrant_client import QdrantClient

from ..catalog import collection_exists, list_collections
from ..config import CONFIG
from ..encoder import ImageTextEncoder
from ..query import looks_like_image_path, query_images
from ..vector_store import ImageVectorStore
from .schemas import (
    CollectionsResponse,
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

# A store_factory takes (collection_name, embedding_dim) and returns a store
# bound to that collection. Only called after the collection is known to exist,
# so it never creates a collection as a side effect.
StoreFactory = Callable[[str, int], ImageVectorStore]


def _default_store_factory(collection: str, embedding_dim: int) -> ImageVectorStore:
    """Server-mode store, targeting the Qdrant host/port from config."""
    return ImageVectorStore(
        collection,
        embedding_dim,
        host=CONFIG.qdrant.host,
        port=CONFIG.qdrant.port,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Populate only what isn't already set, so tests can pre-seed app.state
    # (encoder / qdrant_client / store_factory) and skip the real model load.
    if getattr(app.state, "encoder", None) is None:
        app.state.encoder = ImageTextEncoder()
    if getattr(app.state, "qdrant_client", None) is None:
        app.state.qdrant_client = QdrantClient(
            host=CONFIG.qdrant.host, port=CONFIG.qdrant.port
        )
    if getattr(app.state, "store_factory", None) is None:
        app.state.store_factory = _default_store_factory
    try:
        yield
    finally:
        client = getattr(app.state, "qdrant_client", None)
        if client is not None:
            client.close()


# ---- dependencies (thin accessors over app.state; overridable in tests) ----

def get_encoder(request: Request) -> ImageTextEncoder:
    return request.app.state.encoder


def get_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client


def get_store_factory(request: Request) -> StoreFactory:
    return request.app.state.store_factory


def _require_collection(client: QdrantClient, collection: str) -> None:
    """404 if the collection doesn't exist (avoids the get-or-create side effect
    of constructing an ImageVectorStore against a missing collection)."""
    if not collection_exists(client, collection):
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection!r}")


def resolve_stored_path(path: str) -> Path:
    """Map a path stored in Qdrant to a filesystem path.

    Current indexing stores absolute paths (used as-is). Legacy collections
    stored paths relative to the indexing directory; those are joined with
    CONFIG.api.base_dir when set, else made absolute against the server CWD.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = CONFIG.api.base_dir
    return Path(base) / p if base else p.absolute()


def _validate_image_path(path: str) -> Path:
    """Constrain /image to real image files (an arbitrary-file-read guard)."""
    p = resolve_stored_path(path)
    if p.suffix.lower() not in CONFIG.indexer.extensions or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return p


def create_app() -> FastAPI:
    app = FastAPI(title="iMem API", version="0.1.0", lifespan=lifespan)

    # Localhost personal tool: allow any origin so the containerized frontend
    # (served on whatever port) can reach the host backend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model_loaded = getattr(request.app.state, "encoder", None) is not None
        client = getattr(request.app.state, "qdrant_client", None)
        qdrant_reachable = False
        if client is not None:
            try:
                client.get_collections()
                qdrant_reachable = True
            except Exception:
                qdrant_reachable = False
        status = "ok" if (model_loaded and qdrant_reachable) else "degraded"
        return HealthResponse(
            status=status, model_loaded=model_loaded, qdrant_reachable=qdrant_reachable
        )

    @app.get("/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        return ConfigResponse(
            default_top_k=CONFIG.vector_store.top_k,
            default_collection=CONFIG.qdrant.collection,
            extensions=sorted(CONFIG.indexer.extensions),
            model_id=CONFIG.encoder.model_id,
            distance=CONFIG.vector_store.distance,
        )

    @app.get("/collections", response_model=CollectionsResponse)
    def collections(client: QdrantClient = Depends(get_client)) -> CollectionsResponse:
        rows = list_collections(client)
        return CollectionsResponse(
            collections=[{"name": name, "count": count} for name, count in rows]
        )

    @app.post("/query", response_model=QueryResponse)
    def query(
        req: QueryRequest,
        client: QdrantClient = Depends(get_client),
        encoder: ImageTextEncoder = Depends(get_encoder),
        store_factory: StoreFactory = Depends(get_store_factory),
    ) -> QueryResponse:
        _require_collection(client, req.collection)
        store = store_factory(req.collection, encoder.embedding_dim)
        top_k = req.top_k or CONFIG.vector_store.top_k
        try:
            results = query_images(req.query, store, encoder, top_k=top_k, mode=req.mode)
        except ValueError as e:
            # e.g. an "image" query pointing at a file we can't read.
            raise HTTPException(status_code=400, detail=str(e))
        resolved = req.mode
        if resolved == "auto":
            resolved = "image" if looks_like_image_path(req.query) else "text"
        return QueryResponse(
            results=[{"path": p, "score": s} for p, s in results], mode=resolved
        )

    @app.post("/query/upload", response_model=QueryResponse)
    async def query_upload(
        file: UploadFile = File(...),
        collection: str = Form(...),
        top_k: Optional[int] = Form(default=None),
        client: QdrantClient = Depends(get_client),
        encoder: ImageTextEncoder = Depends(get_encoder),
        store_factory: StoreFactory = Depends(get_store_factory),
    ) -> QueryResponse:
        _require_collection(client, collection)

        # Persist the upload so the encoder (which reads from a path) can open it.
        suffix = Path(file.filename or "").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            embeddings, valid = encoder.encode_images([tmp_path], show_progress=False)
            if len(valid) == 0:
                raise HTTPException(status_code=400, detail="Could not read uploaded image")
            store = store_factory(collection, encoder.embedding_dim)
            result = store.search(embeddings[0], top_k=top_k or CONFIG.vector_store.top_k)
        finally:
            os.unlink(tmp_path)

        return QueryResponse(
            results=[{"path": p.payload["path"], "score": p.score} for p in result.points],
            mode="image",
        )

    @app.get("/image")
    def image(
        path: str = Query(...),
        thumb: Optional[int] = Query(default=None, ge=1, le=4096),
        download: bool = Query(default=False),
    ):
        p = _validate_image_path(path)

        if thumb is not None:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb, thumb))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
            return Response(content=buf.getvalue(), media_type="image/jpeg")

        return FileResponse(
            p,
            filename=p.name,
            content_disposition_type="attachment" if download else "inline",
        )

    return app


# Module-level app for `uvicorn imem.api.app:app`.
app = create_app()
