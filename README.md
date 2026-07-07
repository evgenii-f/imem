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

**Phase 1 (current):** Jupyter notebook POC for indexing and retrieval  
**Phase 2 (planned):** FastAPI server  
**Phase 3 (planned):** Telegram bot client  

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
   jupyter notebook notebooks/1-poc.ipynb
   ```

3. Run the cells to:
   - Load the SigLIP2 model (auto-downloads from Hugging Face)
   - Load & encode the Olivetti Faces + Caltech-101 datasets, index into Qdrant (server mode)
   - Search by text query
   - Search by image query

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

## Performance Notes (M4)

- **First-run:** ~30–60s (model download + warmup)
- **Indexing:** ~50–100 images/min (batch=32, single-threaded)
- **Query latency:** ~100–200ms per text/image query
- **Vector DB:** Qdrant local mode has no network overhead

## Next Steps

- [ ] Notebook POC (text/image search working)
- [ ] Batch indexing pipeline for large image sets
- [ ] FastAPI server with `/search/text`, `/search/image`, `/index` endpoints
- [ ] Telegram bot UI
- [ ] Optional: Deploy vector DB as standalone Qdrant server

## References

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [Hugging Face Model Hub](https://huggingface.co/google/siglip2-base-patch16-224)
- [Qdrant Docs](https://qdrant.tech/)
- [Transformers Library](https://huggingface.co/docs/transformers)

## License

MIT

