# iMem — Semantic Image Search

A semantic image retrieval system using dual-encoder models (CLIP-like architectures) and vector databases for fast text-to-image and image-to-image search.

## Overview

iMem uses multimodal embeddings to encode both images and text into a shared embedding space, enabling semantic search across image collections. Perfect for finding photos by natural language queries ("beach sunset with dogs") or by visual similarity.

### Architecture

```
   ┌──────────────┐     ┌──────────────────────┐
   │   Web UI     │     │         CLI          │
   │  (React,     │     │  imem add            │
   │   nginx)     │     │  imem query          │
   └──────┬───────┘     │  imem collection     │
          │ HTTP        └───────────┬──────────┘
          ▼                         │  
   ┌──────────────┐                 │
   │   FastAPI    │                 │
   │ `imem serve` │                 │
   └──────┬───────┘                 │
          └───────────┬─────────────┘
                      ▼
        ┌────────────────────────────┐
        │  imem package              │
        │  query · catalog · index   │
        └───────┬────────────┬───────┘
                │            │
         encode ▼            ▼ upsert / similarity search
      ┌──────────────┐   ┌──────────────────┐
      │   Encoder    │──▶│  Qdrant VectorDB │
      │  (SigLIP2)   │   │                  │
      └──────┬───────┘   └──────────────────┘
             ▲ index
      ┌──────┴───────┐
      │ Image files  │
      │ (JPEG/PNG)   │
      └──────────────┘

   Runtime:  Docker → Web UI · Qdrant      Host → FastAPI · CLI · Encoder
```

**Phase 1 (done):** Jupyter notebook POC for indexing and retrieval  
**Phase 2 (done):** Modular pipeline + command-line interface — indexing, text/image search, collection management  
**Phase 3 (done):** Local install — pip-installable `imem`, embedded Qdrant by default (no server required)  
**Phase 4 (done):** Dockerized React web frontend + FastAPI backend (`imem serve`) over the same library  
**Phase 5 (planned):** Advanced querying — image metadata indexed at encode time, exact/structured filters over it, and multi-modal (text + image) queries  
**Phase 6 (planned):** Manage collections & index from the web UI — create/delete collections, add folders, live indexing progress  
**Phase 7 (planned):** Saved queries — store a reference (e.g. a face) to re-run later, with match alerts down the line  

## Setup

### Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) — optimized for MPS (Metal Performance Shaders)
  - Can also run on Intel/Linux, just swap `device="mps"` for `device="cuda"` or `"cpu"`

### Installation

```bash
pip install -e .
```

This installs the `imem` command. Optional extras pull dev/dataset/notebook tooling:

```bash
pip install -e ".[dev,datasets,viz,notebook]"
```

### Quick Start (Notebook)
1. Start the Qdrant container:
   ```bash
   docker compose up -d
   ```

2. Open the notebook:
   ```bash
   jupyter notebook notebooks/01-poc.ipynb
   ```

3. Run the cells to:
   - Load the SigLIP2 model (auto-downloads from Hugging Face)
   - Load & encode the Olivetti Faces + Caltech-101 datasets, index into Qdrant (server mode)
   - Search by text query
   - Search by image query

## CLI

The pipeline is also driven from the command line. All commands talk to a running
Qdrant server (`docker compose up -d`).

**Index** a folder (recursively) into a collection:

```bash
imem add ~/Photos --collection personal
```

**Search** by text, or by a reference image (auto-detected from the argument):

```bash
imem query "red cat on sofa"
imem query ~/reference.jpg --collection personal -k 10
```

**Manage collections:**

```bash
imem collection ls
imem collection rm personal -f
```

**Serve** the HTTP API used by the web frontend (see *Web app* below):

```bash
imem serve            # http://127.0.0.1:8000
```

Add `--help` to any command for its full flag list. The `imem` command is created by
`pip install`; without installing, `python -m imem.cli <command>` works too.

## Web app

A React single-page UI to search a collection by **text** or **image** (drag,
paste, or file picker), with a results grid that links each hit back to the file
on disk (open, download, copy path).

The stack spans two runtimes: **Qdrant and the frontend run in Docker**, while
the **FastAPI backend runs on the host** — it needs the host GPU/MPS and read
access to your indexed image files, so it isn't containerized. One command
brings up everything:

```bash
make dev        # docker compose up -d  →  pip install  →  imem serve
```

Then open http://localhost:8080. Container ports are configurable via a root
`.env` (see `.env.example`).

### Make targets

| Target | What it does |
|--------|--------------|
| `make dev` | Whole stack: `up` → `install` → `serve` |
| `make up` | Start the Qdrant + frontend containers |
| `make serve` | Run the backend API on the host (foreground) |
| `make install` | Install the package (`.[api,dev]`) into the active env |
| `make stop` | Pause the containers (resume with `make up`) |
| `make down` | Stop and remove the containers |
| `make build` | Build the frontend Docker image |
| `make test` | Run the test suite |

Run `make` with no arguments to see this list.

Collections indexed with **relative** paths (older ones) resolve against
`IMEM_BASE_DIR` (defaults to your home dir); newer collections store absolute
paths. On macOS, granting the terminal running `imem serve` access to protected
folders (Downloads/Desktop/Documents) lets it serve those thumbnails.

## Model Details

- **Base Model:** `google/siglip2-base-patch16-224`
  - Embedding dimension: 768
  - Input image size: 224×224
  - ~40M parameters, runs comfortably on M4 CPU+GPU
  - Stronger zero-shot retrieval than CLIP (SigLIP sigmoid loss)

- **Upgrade Option:** `google/siglip2-so400m-patch14-384`
  - 400M parameters, 384×384 input, embedding dim 1152
  - Better retrieval quality, slower inference
  - Drop-in replacement (same API)

## Project Structure

- `imem/cli.py` — `imem` command-line entry point (`add`, `query`, `collection ls/rm`)
- `imem/encoder.py` — `ImageTextEncoder`: model loading, image/text embedding (SigLIP2, MPS/CUDA/CPU)
- `imem/vector_store.py` — `ImageVectorStore`: Qdrant wrapper (local & server mode), de-dup, search
- `imem/indexer.py` — recursive folder discovery + indexing (`index_folders`, `iter_image_paths`)
- `imem/query.py` — text/image query against a collection (`query_images`)
- `imem/catalog.py` — instance-level collection ops (list, count, delete)
- `imem/config.py` (+ `config.yml`, `config.ini`) — packaged config loaded into a frozen `CONFIG`
- `imem/api/` — FastAPI app (`imem serve`): `/query`, `/query/upload`, `/collections`, `/image`, … — serves the web frontend
- `tools/dataloader.py` — test dataset loaders (Olivetti Faces, Caltech-101) — dev-only, not packaged
- `tools/visualize.py` — notebook result-grid helper — dev-only, not packaged
- `notebooks/01-poc.ipynb` — local/embedded Qdrant POC (test datasets)
- `notebooks/00-poc-personal-imgs.ipynb` — server/container Qdrant POC using your own image collection (`docker compose up -d`)
- `frontend/` — React + TypeScript + Mantine SPA (built to static, served by nginx in Docker)
- `Makefile` — dev orchestration (`make dev` / `up` / `serve` / `stop` / `down` / `test`)
- `docker-compose.yml` — Qdrant + frontend containers
- `tests/` — pytest tests for the encoder, vector store, indexer, query, catalog, API & config modules

## Testing

Tests cover the encoder pipeline (shapes, normalization, error handling), the Qdrant vector
store wrapper, and the indexer / query / catalog logic — all against an in-memory Qdrant
instance, no server needed:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

`tests/test_encoder.py` downloads the real SigLIP2 model on first run (network required). The
indexer and query tests run fully offline, using random stand-in embeddings instead of the
real model; `tests/test_catalog.py` and `tests/test_vector_store.py` don't touch the model at
all. Model-dependent tests auto-skip if `torch`/`transformers` aren't installed.

## References

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [Hugging Face Model Hub](https://huggingface.co/google/siglip2-base-patch16-224)
- [Qdrant Docs](https://qdrant.tech/)
- [Transformers Library](https://huggingface.co/docs/transformers)

## License

MIT

