# iMem — Semantic Image Search

A semantic image retrieval system using dual-encoder models (CLIP-like architectures) and vector databases for fast text-to-image and image-to-image search.

## Overview

iMem uses multimodal embeddings to encode both images and text into a shared embedding space, enabling semantic search across image collections. Perfect for finding photos by natural language queries ("beach sunset with dogs") or by visual similarity.

### Architecture

```
┌─────────────────────┐
│  Image Collection   │
│  (JPEG/PNG files)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐         ┌──────────────────┐
│       Encoder       │────────▶│  Qdrant VectorDB │
└─────────────────────┘         └──────────────────┘
           △
           │
    ┌──────┴──────┐
    │             │
┌───────┐   ┌─────────┐
│ Text  │   │ Image   │
│Query  │   │Query    │
└───────┘   └─────────┘
```

**Phase 1 (done):** Jupyter notebook POC for indexing and retrieval  
**Phase 2 (done):** Modular pipeline + command-line interface — indexing, text/image search, collection management  
**Phase 3 (current):** Local install — pip-installable `imem`, embedded Qdrant by default (no server required)  
**Phase 4 (planned):** Dockerized web frontend + FastAPI backend over the same library (Qdrant as a server)  

## Setup

### Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) — optimized for MPS (Metal Performance Shaders)
  - Can also run on Intel/Linux, just swap `device="mps"` for `device="cuda"` or `"cpu"`

### Installation

```bash
pip install -r requirements.txt
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
python -m imem.cli add ~/Photos --collection personal
```

**Search** by text, or by a reference image (auto-detected from the argument):

```bash
python -m imem.cli query "red cat on sofa"
python -m imem.cli query ~/reference.jpg --collection personal -k 10
```

**Manage collections:**

```bash
python -m imem.cli collection ls
python -m imem.cli collection rm personal -f
```

Add `--help` to any command for its full flag list. All commands run via
`python -m imem.cli` today; this becomes a single `imem` entry point once the package
is installable.

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
- `tools/dataloader.py` — test dataset loaders (Olivetti Faces, Caltech-101) — dev-only, not packaged
- `tools/visualize.py` — notebook result-grid helper — dev-only, not packaged
- `notebooks/01-poc.ipynb` — local/embedded Qdrant POC (test datasets)
- `notebooks/00-poc-personal-imgs.ipynb` — server/container Qdrant POC using your own image collection (`docker compose up -d`)
- `tests/` — pytest tests for the encoder, vector store, indexer, query & catalog modules

## Testing

Tests cover the encoder pipeline (shapes, normalization, error handling), the Qdrant vector
store wrapper, and the indexer / query / catalog logic — all against an in-memory Qdrant
instance, no server needed:

```bash
pip install -r requirements.txt
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

