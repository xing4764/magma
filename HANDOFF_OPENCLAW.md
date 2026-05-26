# MAGMA Memory Optimization Handoff

## What Changed

- Switched the default embedding model to `BAAI/bge-small-zh-v1.5` for Chinese e-commerce memory retrieval.
- Added lifecycle metadata to memory nodes: `status`, `importance`, `ttl_days`, `valid_from`, `valid_until`, `last_accessed_at`, and `access_count`.
- Added a shared retrieval layer that combines semantic score, Chinese keyword score, and lifecycle weighting.
- Updated FastAPI and MCP query paths to use the same retrieval logic.
- Added automatic rule-based consolidation in the FastAPI service.
- Added `reembed` CLI command to rebuild embeddings and FAISS index in place without re-importing source data.
- Changed the default Hugging Face endpoint to `https://huggingface.co`; mirrors can still be selected with `HF_ENDPOINT`.

## Rebuild Existing Memory Embeddings

Run this after deploying the optimized MAGMA code:

```powershell
python scripts\magma_cli.py reembed --batch-size 32
```

This keeps existing nodes and edges, updates node embeddings, and rebuilds:

- `data/faiss.index`
- `data/id_map.json`
- `data/faiss_meta.json`

## MCP Integration

The MCP server entrypoint remains:

```powershell
python -m magma.api.mcp_server
```

The registered tools stay compatible:

- `magma_query`
- `magma_add_node`
- `magma_add_edge`
- `magma_list_nodes`
- `magma_get_node`

## Current Runtime Verification

- Runtime embedding model is `BAAI/bge-small-zh-v1.5`.
- Runtime embedding dimension is 512.
- `data/faiss_meta.json` records `model=BAAI/bge-small-zh-v1.5` and `dimension=512`.
- SQLite currently stores 512-dimensional embeddings for all embedded nodes.
- Keep LLM backend and embedding model separate: DeepSeek V3 via OpenRouter is for slow-path reasoning/extraction; bge-small-zh-v1.5 is for local vector retrieval.

## Notes For Reviewers

- Historical 384-dimensional MiniLM vectors must not be treated as current runtime state. They were superseded by regenerated `BAAI/bge-small-zh-v1.5` 512-dimensional vectors.
- If model loading fails, query falls back to keyword retrieval instead of crashing the MCP tool.
- This package intentionally excludes local runtime data such as `magma.db`, `faiss.index`, `id_map.json`, and `__pycache__`.
