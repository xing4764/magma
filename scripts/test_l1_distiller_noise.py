"""Smoke tests for L1 distiller noise filtering."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magma.l1_distiller import build_l1_candidate


def node(content, role="assistant"):
    return {
        "id": "evt:test",
        "label": "event",
        "source_agent_id": "zhuli",
        "department": "assistant",
        "properties": {
            "layer": "L0",
            "role": role,
            "content": content,
            "agent_id": "zhuli",
        },
    }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_blocks_dreaming_artifacts():
    samples = [
        "Write a dream diary entry from these memory fragments: - Assistant: MAGMA API port is 8904.",
        "Here is a dream diary entry woven from those fragments: A number hovered at the edge of everything.",
        "[message_id: om_x100b6db0668064a0c3309917ea7f717] ou_489d2217d717695d17ff382e7dfb0168: MAGMA API port is 8904 [System: The content may include mention tags in XML format.]",
        "[Inter-session message] sourceSession=agent:yunying:subagent sourceChannel=webchat MAGMA API port is 8904",
        '{"traceSchema":"openclaw-trajectory","sessionFile":"agent.jsonl","toolCallId":"call_1"}',
    ]
    for sample in samples:
        require(build_l1_candidate(node(sample)) is None, f"artifact should be blocked: {sample[:80]}")


def test_allows_real_operational_fact():
    candidate = build_l1_candidate(node("MAGMA API port 8904 is the current production endpoint for recall.", role="user"))
    require(candidate is not None, "real operational fact should pass")
    require(candidate.kind == "fact", f"expected fact, got {candidate.kind}")


if __name__ == "__main__":
    test_blocks_dreaming_artifacts()
    test_allows_real_operational_fact()
    print("l1_distiller_noise: 2/2 passed")
