# MAGMA Operations Runbook

This runbook is the first stop for MAGMA health checks and safe self-service triage.

## Daily Health Check

Run these commands from any PowerShell session:

```powershell
python C:\openclaw-magma\scripts\magma_ops.py status
python C:\openclaw-magma\scripts\magma_doctor.py --json
python C:\openclaw-magma\scripts\magma_doctor.py --agent
```

## Status Meaning

- `GREEN`: MAGMA is healthy for daily use.
- `YELLOW`: MAGMA is usable, but one or more signals should be watched.
- `RED`: MAGMA needs immediate repair or escalation.

Common `YELLOW` examples:

- `recent_capture` is older than the threshold, usually meaning no recent messages were captured.
- An agent has not recalled recently, which may be normal if that agent has been idle.

Common `RED` examples:

- `api_8902` is down.
- MCP is not in `http_proxy` mode.
- Embedding coverage drops below the safety threshold.
- The database cannot be read.

## Safe Repair

Run:

```powershell
python C:\openclaw-magma\scripts\magma_ops.py repair
```

The repair command checks:

- MAGMA API port `8902`
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

## Escalation Rule

Escalate to technical maintenance when:

- `magma_ops.py status` reports `RED`.
- `magma_ops.py repair` reports unresolved issues.
- `magma_doctor.py --json` shows `overall: red`.
- The same `YELLOW` warning persists for more than one day and affects recall quality.
