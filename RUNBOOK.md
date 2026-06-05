# MAGMA Operations Runbook

This runbook is the first stop for MAGMA health checks and safe self-service triage.

## Daily Health Check

Run these commands from any PowerShell session:

```powershell
python C:\openclaw-magma\scripts\magma_acceptance.py --skip-benchmark
python C:\openclaw-magma\scripts\magma_ops.py status
python C:\openclaw-magma\scripts\magma_doctor.py --json
python C:\openclaw-magma\scripts\magma_doctor.py --agent
```

Use the full acceptance gate after code changes or model/index work:

```powershell
python C:\openclaw-magma\scripts\magma_acceptance.py
```

The acceptance gate checks API health, doctor, Python syntax for core modules,
recall eval, product benchmark, and L1 dry-run reachability. A `GREEN` result is
the normal release/hand-off condition.

## Status Meaning

- `GREEN`: MAGMA is healthy for daily use.
- `YELLOW`: MAGMA is usable, but one or more signals should be watched.
- `RED`: MAGMA needs immediate repair or escalation.

Common `YELLOW` examples:

- `recent_capture` is older than the threshold, usually meaning no recent messages were captured.
- An agent has not recalled recently, which may be normal if that agent has been idle.

Common `RED` examples:

- The configured MAGMA API is down.
- MCP is not in `http_proxy` mode.
- Embedding coverage drops below the safety threshold.
- The database cannot be read.

## Safe Repair

Run:

```powershell
python C:\openclaw-magma\scripts\magma_ops.py repair
```

The repair command checks:

- The configured MAGMA API port
- MAGMA API health
- MCP proxy mode
- SQLite database readability and embedding coverage
- OpenClaw MAGMA MCP registration
- OpenClaw and codex package pins

Safety boundary:

- It does not upgrade or downgrade OpenClaw.
- It does not delete data.
- It does not clean or consolidate memories.
- It does not rewrite large OpenClaw configuration sections.

## Version Pin

The known-compatible OpenClaw line is:

```text
openclaw: 2026.5.20
@openclaw/codex: 2026.5.20
```

Do not upgrade to `2026.5.22` as a repair step. That line has caused codex runtime compatibility issues in this environment.

## Agent Attribution

MAGMA stores both direct hook execution and source attribution:

- `agentId`: the OpenClaw agent that executed the hook.
- `source_agent_id`: the agent that originated the memory, such as `yunying`, `jishu`, `zhuli`, or `main`.
- `department`: the human-facing department/source label.

Use `source_agent_id` and `department` for cross-agent filtering, QA, and memory provenance.

## Retrieval Quality

The realtime recall path must stay fast because it runs before prompt build:

- Keep `Qwen3-Embedding-0.6B` as the default first-stage embedding model. Production embeddings should be 1024-dimensional; if they are 512-dimensional, rerun `magma_cli.py reembed` with the configured Qwen model.
- Prefer high-density `ops_anchor`, `L1`, `decision`, `fact`, and `current_state` memories for operational questions.
- Keep L0 raw chat memories as evidence, but do not let generic chat fragments dominate high-signal anchors.
- Use `scripts/qwen_embedding_probe.py` and `scripts/qwen_reranker_probe.py` for offline A/B tests only.
- Do not enable Qwen3 reranking in the realtime `before_prompt_build` path on CPU-only 12GB hosts; top20 reranking can take tens of seconds.

## L1 Distillation

L1 is the stable memory layer for decisions, preferences, project state,
pending actions, lessons, and concrete facts. It is intentionally stricter than
L0 capture: transient debug output, acknowledgements, diagnostic reports, and
assistant self-corrections should stay in L0 only.

Run L1 in dry-run mode first:

```powershell
python C:\openclaw-magma\scripts\magma_l1_runner.py --hours 24 --limit 200 --json
```

Apply only when the preview looks clean:

```powershell
python C:\openclaw-magma\scripts\magma_l1_runner.py --hours 24 --limit 200 --apply
```

For scoped testing, use `--source-agent-id benchmark` or another source agent.
Overlapping windows are safe because L1 node IDs are stable upserts.

### L1 Recall Weighting

L1 nodes receive higher recall weight than L0 in `magma/search.py`:

| Layer / Kind | Quality Multiplier |
|---|---|
| L1 路 current_state | 1.30 |
| L1 路 decision | 1.25 |
| L1 路 fact | 1.18 |
| L1 路 (unclassified) | 1.20 |
| ops_anchor | 1.08 |
| L0 路 user (non-question) | 0.86 |
| L0 路 user (question) | 0.62 |
| L0 路 assistant | 0.98 |

Combined with operational authority multiplier (up to 1.34x for keyword-matching L1 nodes), L1 current_state nodes can achieve up to ~1.74x total boost over L0 baseline.

## Escalation Rule

Escalate to technical maintenance when:

- `magma_ops.py status` reports `RED`.
- `magma_ops.py repair` reports unresolved issues.
- `magma_doctor.py --json` shows `overall: red`.
- The same `YELLOW` warning persists for more than one day and affects recall quality.
