# MAGMA

Multi-Graph Adaptive Memory Architecture for OpenClaw agents.

MAGMA provides cross-session and cross-agent memory for an OpenClaw agent system. It stores conversation events, entities, relations, embeddings, recall events, and feedback in a local SQLite + FAISS based memory layer, then injects relevant memories into agents through the `magma-recall` OpenClaw plugin.

## Current Runtime

- Embedding model: local `BAAI/bge-small-zh-v1.5`
- Embedding dimension: 512
- Slow-path LLM backend: DeepSeek V3 via OpenRouter
- Main API: `http://127.0.0.1:8902`
- MCP server: thin HTTP proxy to the 8902 API

The LLM backend and embedding model are separate. DeepSeek V3 is used for slow-path reasoning/extraction. `bge-small-zh-v1.5` is used for local semantic retrieval. Historical MiniLM-L6-v2 / 384-dimensional vectors are not the current runtime state.

## Core Capabilities

- Automatic recall before prompt build
- Automatic L0 capture after agent responses
- Cross-agent memory with `source_agent_id` and `department`
- Semantic + keyword + lifecycle-aware retrieval
- Recall feedback and importance updates
- Operational anchors for high-value system facts
- Red/yellow/green diagnostics
- Conservative memory governance with dry-run and soft apply
- OpenClaw MCP compatibility through the 8902 main chain

## Repository Layout

```text
magma/
  api/
    server.py              # FastAPI service
    mcp_server.py          # MCP thin proxy
  graph/
    sqlite_store.py        # SQLite graph/memory store
  vector/
    encoder.py             # sentence-transformers encoder
  backup.py
  entities.py
  search.py

openclaw-plugin-magma-recall/
  index.js                 # OpenClaw hook plugin
  openclaw.plugin.json
  package.json

scripts/
  magma_doctor.py          # Health diagnostics
  magma_ops.py             # Status and safe repair checks
  magma_governance.py      # Dry-run / soft memory governance
  magma_recall_eval.py     # Recall quality evaluation
  migrate_source_agent.py  # Backfill source attribution
  seed_operational_anchors.py
  magma_cli.py

RUNBOOK.md
HANDOFF_OPENCLAW.md
起源.md
```

## Install

```powershell
cd C:\openclaw-magma
pip install -r requirements.txt
```

If Hugging Face access is slow or blocked, set `HF_ENDPOINT` before the first model load.

## Run API

```powershell
cd C:\openclaw-magma
python -m magma.api.server
```

The production OpenClaw integration expects the API on port `8902`.

## OpenClaw Integration

The OpenClaw plugin is in:

```text
openclaw-plugin-magma-recall/
```

The plugin registers:

- `before_prompt_build` for automatic recall
- `agent_end` for automatic capture and weak positive feedback
- `before_message_write` for stripping injected memory from persisted history

The MCP entrypoint remains compatible:

```powershell
python -m magma.api.mcp_server
```

All MCP tools proxy to `http://127.0.0.1:8902/api/v1/...` instead of loading SQLite and embeddings inside the stdio process.

## Operations

```powershell
# Health check
python scripts\magma_doctor.py --json

# Short status
python scripts\magma_ops.py status

# Safe repair checks
python scripts\magma_ops.py repair

# Governance dry-run
python scripts\magma_governance.py --dry-run --json

# Soft governance apply
python scripts\magma_governance.py --apply --json

# Recall quality evaluation
python scripts\magma_recall_eval.py
```

## Data Safety

Runtime data is intentionally excluded from Git:

- `data/`
- `*.db`, `*.db-shm`, `*.db-wal`
- `*.index`
- `backups/`
- `.env`

Do not commit local memory databases, FAISS indexes, OpenClaw credentials, Feishu secrets, OpenRouter/OpenAI keys, or token files.

## License

MIT
