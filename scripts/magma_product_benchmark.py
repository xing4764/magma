"""Product benchmark for MAGMA.

This benchmark exercises the product capabilities that make MAGMA more than a
plain vector-memory layer:

- API health and core memory
- short-command routing
- capture governance and suppression
- recall explain/correction tools
- semantic retrieval over realistic business/ops memories

It writes temporary benchmark nodes and soft-deletes them at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configured_api_base() -> str:
    env_base = os.environ.get("MAGMA_API_BASE")
    if env_base:
        return env_base.rstrip("/")
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'"apiBaseUrl"\s*:\s*"(http://127\.0\.0\.1:\d+)"', text)
        if match:
            return match.group(1).rstrip("/")
    except OSError:
        pass
    return "http://127.0.0.1:8902"


class MagmaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: int = 30) -> Any:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8")
        return json.loads(raw) if raw else None

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, body: dict[str, Any], timeout: int = 30) -> Any:
        return self.request("POST", path, body, timeout=timeout)

    def put(self, path: str, body: dict[str, Any], timeout: int = 30) -> Any:
        return self.request("PUT", path, body, timeout=timeout)

    def delete_node(self, node_id: str) -> None:
        encoded = urllib.parse.quote(node_id, safe="")
        try:
            self.request("DELETE", f"/api/v1/nodes/{encoded}", timeout=10)
        except Exception:
            pass


class Benchmark:
    def __init__(self, client: MagmaClient, run_id: str):
        self.client = client
        self.run_id = run_id
        self.cleanup_nodes: list[str] = []
        self.reports: list[dict[str, Any]] = []

    def check(self, case_id: str, category: str, fn: Callable[[], tuple[bool, dict[str, Any]] | bool]) -> None:
        started = time.time()
        try:
            result = fn()
            if isinstance(result, tuple):
                ok, details = result
            else:
                ok, details = bool(result), {}
            self.reports.append({
                "id": case_id,
                "category": category,
                "ok": bool(ok),
                "duration_ms": round((time.time() - started) * 1000),
                **(details or {}),
            })
        except Exception as exc:
            self.reports.append({
                "id": case_id,
                "category": category,
                "ok": False,
                "duration_ms": round((time.time() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            })

    def add_node(self, suffix: str, label: str, content: str, **properties: Any) -> str:
        node_id = f"bench:{self.run_id}:{suffix}"
        props = {
            "layer": properties.pop("layer", "benchmark"),
            "content": content,
            "source": "magma_product_benchmark",
            "source_agent_id": properties.pop("source_agent_id", "benchmark"),
            "department": properties.pop("department", "benchmark"),
            "importance": properties.pop("importance", 0.7),
            "ttl_days": 1,
            **properties,
        }
        self.client.post("/api/v1/nodes", {"id": node_id, "label": label, "properties": props}, timeout=45)
        self.cleanup_nodes.append(node_id)
        return node_id

    def query_contains(self, query: str, expected: str, top_k: int = 5) -> tuple[bool, dict[str, Any]]:
        body = self.client.post("/api/v1/query", {
            "query": query,
            "top_k": top_k,
            "filters": {"pool_size": 5000, "include_related": True, "include_versions": True},
        }, timeout=45)
        results = body.get("results") or []
        haystack = json.dumps(results, ensure_ascii=False).lower()
        top_ids = [item.get("id") for item in results[:3]]
        return expected.lower() in haystack, {"top_ids": top_ids}

    def cleanup(self) -> None:
        for node_id in reversed(self.cleanup_nodes):
            self.client.delete_node(node_id)
        self.hard_cleanup()

    def hard_cleanup(self) -> None:
        """Remove benchmark artifacts, including background fact-extraction nodes."""
        time.sleep(1.0)
        db_path = os.environ.get("MAGMA_DB_PATH") or str(PROJECT_ROOT / "data" / "magma.db")
        if not Path(db_path).exists():
            return
        conn = sqlite3.connect(db_path)
        try:
            node_ids = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT id
                      FROM nodes
                     WHERE id LIKE ?
                        OR id LIKE 'core_memory:benchmark:%'
                        OR id LIKE 'suppression_pattern:%' AND json_extract(properties, '$.agent_id') = 'benchmark'
                        OR source_agent_id = 'benchmark'
                        OR json_extract(properties, '$.source_agent_id') = 'benchmark'
                        OR json_extract(properties, '$.source') = 'magma_product_benchmark'
                    """,
                    (f"bench:{self.run_id}:%",),
                ).fetchall()
            ]
            if node_ids:
                placeholders = ", ".join("?" for _ in node_ids)
                conn.execute(f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", node_ids * 2)
                conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)
            event_prefix = f"bench:recall:{self.run_id}"
            conn.execute("DELETE FROM recall_feedback WHERE event_id LIKE ?", (event_prefix + "%",))
            conn.execute("DELETE FROM recall_events WHERE id LIKE ?", (event_prefix + "%",))
            conn.commit()
        finally:
            conn.close()


def run_benchmark(api_base: str) -> dict[str, Any]:
    from magma.capture_policy import classify_capture
    from magma.short_command import is_short_command, normalize_short_command_query, resolve_short_command

    client = MagmaClient(api_base)
    run_id = f"{int(time.time())}"
    bench = Benchmark(client, run_id)

    try:
        # Health and core memory: 5 checks
        bench.check("health_ok", "api", lambda: (client.get("/api/v1/health").get("status") == "ok", {}))
        core_block = f"benchmark_core_{run_id}"
        core_content = f"MAGMA benchmark {run_id}: core memory must be stable, explainable, and maintainable."
        bench.check("core_memory_put", "core_memory", lambda: (
            client.put("/api/v1/core_memory", {
                "agent_id": "benchmark",
                "block_name": core_block,
                "content": core_content,
                "importance": 0.95,
            }).get("status") == "ok",
            {},
        ))
        bench.cleanup_nodes.append(f"core_memory:benchmark:{core_block}")
        bench.check("core_memory_get_exact", "core_memory", lambda: (
            core_content in json.dumps(client.get(f"/api/v1/core_memory?agent_id=benchmark&block_name={core_block}"), ensure_ascii=False),
            {},
        ))
        bench.check("core_memory_get_list", "core_memory", lambda: (
            core_block in json.dumps(client.get("/api/v1/core_memory?agent_id=benchmark"), ensure_ascii=False),
            {},
        ))
        bench.check("core_memory_semantic_recall", "core_memory", lambda: bench.query_contains(
            f"benchmark {run_id} stable explainable maintainable core memory",
            core_block,
        ))

        # L1 distillation: stable high-value summaries from L0 memories.
        l1_source = bench.add_node(
            "l1_source",
            "event",
            f"老板偏好：MAGMA benchmark {run_id} L1 提炼必须沉淀长期产品化原则。",
            role="user",
            layer="L0",
            source_agent_id="benchmark",
        )
        l1_written: list[str] = []

        bench.check("l1_distill_dry_run", "l1_distill", lambda: (
            client.post("/api/v1/distill_l1", {
                "hours": 24,
                "limit": 50,
                "dry_run": True,
                "source_agent_id": "benchmark",
            }, timeout=60).get("candidate_count", 0) >= 1,
            {},
        ))

        def l1_distill_apply() -> tuple[bool, dict[str, Any]]:
            body = client.post("/api/v1/distill_l1", {
                "hours": 24,
                "limit": 50,
                "dry_run": False,
                "source_agent_id": "benchmark",
            }, timeout=90)
            l1_written.extend(body.get("written") or [])
            bench.cleanup_nodes.extend(l1_written)
            return body.get("status") == "ok" and body.get("written_count", 0) >= 1, {
                "written_count": body.get("written_count"),
                "by_kind": body.get("by_kind"),
                "written": body.get("written", [])[:5],
            }

        bench.check("l1_distill_apply", "l1_distill", l1_distill_apply)
        bench.check("l1_distill_queryable", "l1_distill", lambda: bench.query_contains(
            f"benchmark {run_id} L1 长期产品化原则",
            run_id,
        ))
        bench.check("l1_distill_evidence_edge", "l1_distill", lambda: (
            bool(l1_written) and l1_source in json.dumps(
                client.get(f"/api/v1/nodes/{urllib.parse.quote(l1_written[0], safe='')}"),
                ensure_ascii=False,
            ),
            {"l1_written": l1_written[:3], "source": l1_source},
        ))

        # Short-command local routing: 8 checks
        recent_nodes = [{
            "id": f"bench:{run_id}:question",
            "label": "event",
            "properties": {
                "role": "assistant",
                "layer": "L0",
                "content": "需要我帮你更新 README 加公开仓库说明吗？",
            },
        }]
        short_cases = [
            ("short_update", lambda: is_short_command("更新")),
            ("short_start", lambda: is_short_command("开始")),
            ("short_continue", lambda: is_short_command("继续")),
            ("short_cancel", lambda: is_short_command("取消")),
            ("short_timestamp", lambda: is_short_command("[Fri 2026-06-05 19:53 GMT+8] 更新")),
            ("short_multiline", lambda: normalize_short_command_query("更新\n不要总结") == "更新"),
            ("short_resolve_confirm", lambda: (resolve_short_command("更新", recent_nodes) or {}).get("is_confirmation") is True),
            ("short_resolve_reject", lambda: (resolve_short_command("不", recent_nodes) or {}).get("is_rejection") is True),
        ]
        for case_id, fn in short_cases:
            bench.check(case_id, "short_command", lambda fn=fn: (fn(), {}))

        # Capture policy local checks: 7 checks
        capture_cases = [
            ("capture_noise_ok", lambda: classify_capture("OK", "").should_capture is False),
            ("capture_noise_rate_limit", lambda: classify_capture("API rate limit reached", "").should_capture is False),
            ("capture_strong_preference", lambda: classify_capture("老板偏好：MAGMA 必须真实可用", "").strength == "strong"),
            ("capture_strong_sku", lambda: classify_capture("SKU 8821 尺码表：身高160体重120穿L", "").strength == "strong"),
            ("capture_normal_substantial", lambda: classify_capture("这是一段足够长的普通项目上下文", "").should_capture is True),
            ("capture_suppression", lambda: classify_capture("临时压制样例", "", ["临时压制样例"]).strength == "suppressed"),
            ("capture_empty_skip", lambda: classify_capture("", "").should_capture is False),
        ]
        for case_id, fn in capture_cases:
            bench.check(case_id, "capture_policy", lambda fn=fn: (fn(), {}))

        # Capture API and dynamic suppression: 6 checks
        bench.check("capture_api_noise_skip", "capture_api", lambda: (
            client.post("/api/v1/capture", {
                "user_text": "OK",
                "assistant_text": "",
                "agent_id": "benchmark",
                "session_key": f"agent:benchmark:{run_id}",
                "source": "magma_product_benchmark",
                "ttl_days": 1,
            }).get("status") == "skipped",
            {},
        ))
        bench.check("capture_api_rate_limit_skip", "capture_api", lambda: (
            client.post("/api/v1/capture", {
                "user_text": "API rate limit reached. Please try again later.",
                "assistant_text": "",
                "agent_id": "benchmark",
                "session_key": f"agent:benchmark:{run_id}",
                "source": "magma_product_benchmark",
                "ttl_days": 1,
            }).get("status") == "skipped",
            {},
        ))
        strong_capture_ids: list[str] = []

        def strong_capture() -> tuple[bool, dict[str, Any]]:
            body = client.post("/api/v1/capture", {
                "user_text": f"老板偏好：MAGMA benchmark {run_id} 写入前降噪必须保留真实决策。",
                "assistant_text": "已记录产品化决策。",
                "agent_id": "benchmark",
                "session_key": f"agent:benchmark:{run_id}",
                "source": "magma_product_benchmark",
                "ttl_days": 1,
            }, timeout=45)
            strong_capture_ids.extend(body.get("written") or [])
            bench.cleanup_nodes.extend(strong_capture_ids)
            return body.get("status") == "ok" and body.get("count", 0) >= 1, {"written": body.get("written")}

        bench.check("capture_api_strong_write", "capture_api", strong_capture)
        suppress_pattern = f"bench-suppress-{run_id}"
        suppress_node = ""

        def add_suppression() -> tuple[bool, dict[str, Any]]:
            nonlocal suppress_node
            body = client.post("/api/v1/memory/suppress_pattern", {
                "pattern": suppress_pattern,
                "reason": "benchmark dynamic suppression",
                "agent_id": "benchmark",
                "ttl_days": 1,
            })
            suppress_node = body.get("node_id")
            if suppress_node:
                bench.cleanup_nodes.append(suppress_node)
            return body.get("status") == "ok", {"node_id": suppress_node}

        bench.check("suppress_pattern_add", "capture_api", add_suppression)
        bench.check("capture_api_dynamic_suppress", "capture_api", lambda: (
            client.post("/api/v1/capture", {
                "user_text": f"{suppress_pattern} should skip capture",
                "assistant_text": "",
                "agent_id": "benchmark",
                "session_key": f"agent:benchmark:{run_id}",
                "source": "magma_product_benchmark",
                "ttl_days": 1,
            }).get("capture_decision", {}).get("strength") == "suppressed",
            {},
        ))
        def written_nodes_gettable() -> tuple[bool, dict[str, Any]]:
            nodes = []
            for node_id in strong_capture_ids:
                body = client.get(f"/api/v1/nodes/{urllib.parse.quote(node_id, safe='')}")
                node = body.get("node") or {}
                nodes.append({
                    "id": node_id,
                    "strength": (node.get("properties") or {}).get("capture_strength"),
                    "status": node.get("status"),
                })
            return bool(nodes) and all(item["strength"] == "strong" for item in nodes), {"nodes": nodes}

        bench.check("capture_api_written_nodes_gettable", "capture_api", written_nodes_gettable)
        # Explain and correction tools: 6 checks
        explain_node = bench.add_node(
            "explain",
            "event",
            f"MAGMA benchmark {run_id}: recall explanation should show score and provenance.",
            role="user",
            layer="L0",
            source_agent_id="benchmark",
        )
        event_id = f"bench:recall:{run_id}"
        bench.check("feedback_record", "correction", lambda: (
            client.post("/api/v1/feedback", {
                "event_id": event_id,
                "query": f"benchmark {run_id} explain recall",
                "agent_id": "benchmark",
                "session_key": f"agent:benchmark:{run_id}",
                "recalled": [{"id": explain_node, "score": 1.0, "semantic_score": 0.7, "keyword_score": 0.3}],
                "used": [{"id": explain_node}],
                "positive_delta": 0.01,
                "unused_delta": -0.001,
            }).get("status") == "ok",
            {},
        ))
        bench.check("explain_recall_event", "correction", lambda: (
            client.post("/api/v1/recall/explain", {"event_id": event_id, "node_id": explain_node}).get("explanation", {}).get("node_id") == explain_node,
            {},
        ))
        bench.check("mark_important", "correction", lambda: (
            client.post("/api/v1/memory/mark_important", {
                "node_id": explain_node,
                "reason": "benchmark promote",
                "agent_id": "benchmark",
            }).get("new_importance", 0) >= 0.8,
            {},
        ))
        bench.check("important_survives_get_node", "correction", lambda: (
            (client.get(f"/api/v1/nodes/{urllib.parse.quote(explain_node, safe='')}").get("node") or {}).get("importance", 0) >= 0.8,
            {},
        ))
        bench.check("mark_wrong", "correction", lambda: (
            client.post("/api/v1/memory/mark_wrong", {
                "node_id": explain_node,
                "reason": "benchmark suppress",
                "agent_id": "benchmark",
            }).get("status") == "ok",
            {},
        ))
        bench.check("wrong_sets_suppressed", "correction", lambda: (
            (client.get(f"/api/v1/nodes/{urllib.parse.quote(explain_node, safe='')}").get("node") or {}).get("status") == "suppressed",
            {},
        ))

        # Synthetic realistic retrieval: 18 checks
        retrieval_memories = [
            ("sku_size", "SKU BMARK-001 冰丝凉感女裤：身高155-160体重95-115建议M码。", "BMARK-001 M码"),
            ("sku_plus", "SKU BMARK-002 中老年妈妈装：身高160-165体重130-145建议2XL。", "BMARK-002 2XL"),
            ("douyin_title", "抖音小店上架规则：标题必须保留冰丝、凉感、妈妈装三个卖点。", "冰丝 凉感 妈妈装"),
            ("douyin_image", "尺码助手必须从上传的尺码详情图识别尺码表，不允许臆造5XL。", "尺码详情图 5XL"),
            ("ops_gateway", "OpenClaw Gateway 升级铁律：先停 Gateway，再 npm install，最后启动验证。", "先停 Gateway npm install"),
            ("ops_rate_limit", "小米 coding provider 出现 429 时不要擅自切 provider，必须先征得老板同意。", "429 provider 老板同意"),
            ("magma_origin", "MAGMA 起源设计：语义图、时间图、情景图三层分离，查询时按需组合。", "三层分离 语义图 时间图"),
            ("magma_fast_slow", "MAGMA 摄入设计：快路径正则和向量立即可查，慢路径 LLM 巩固关系。", "快路径 慢路径 LLM"),
            ("magma_core", "Core Memory 用于长期身份偏好、固定事实、项目原则和老板长期要求。", "Core Memory 长期偏好"),
            ("magma_explain", "MAGMA 召回解释要回答为什么召回、来源是谁、错了怎么纠正。", "为什么召回 怎么纠正"),
            ("yunying_scope", "跨 Agent 共享不是混池，必须带 source_agent_id、department 和来源降权。", "source_agent_id department"),
            ("backup", "MAGMA 运维要求：doctor 红黄绿、ops repair、定期备份和可恢复演练。", "doctor repair 备份"),
            ("qwen_embedding", "Qwen3-Embedding-0.6B 作为中文 embedding 旁路测试，重嵌全量约199秒。", "Qwen3-Embedding 199秒"),
            ("reranker", "Qwen3-Reranker-0.6B 应做成可选二阶段召回，默认关闭避免 CPU 延迟。", "Reranker 二阶段 默认关闭"),
            ("l1", "L1 提炼层应沉淀 decision、preference、project_state、pending_action、lesson。", "L1 decision preference"),
            ("short_cmd", "短指令更新必须绑定最近会话 pending action，不能跑偏到版本更新。", "短指令 pending action"),
            ("public_repo", "MAGMA 公开仓库要求：README、安装指南、贡献指南必须真实可靠。", "公开仓库 README 贡献指南"),
            ("product_principle", "老板长期原则：MAGMA 优化要产品化、真实可用、长期可维护。", "产品化 真实可用 长期可维护"),
        ]
        for suffix, content, query in retrieval_memories:
            bench.add_node(suffix, "event", f"{content} benchmark_marker={run_id}", role="fact", layer="benchmark")
        for suffix, _content, query in retrieval_memories:
            bench.check(f"retrieval_{suffix}", "retrieval", lambda query=query: bench.query_contains(
                f"{query} benchmark_marker={run_id}",
                run_id,
            ))

    finally:
        bench.cleanup()

    passed = sum(1 for item in bench.reports if item.get("ok"))
    total = len(bench.reports)
    return {
        "api_base": api_base,
        "run_id": run_id,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pct": round(passed / total * 100, 1) if total else 0.0,
        "pass": total >= 50 and passed == total,
        "cases": bench.reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAGMA 50-check product benchmark.")
    parser.add_argument("--api-base", default=configured_api_base())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = run_benchmark(args.api_base)
    except urllib.error.URLError as exc:
        report = {
            "api_base": args.api_base,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "pct": 0.0,
            "pass": False,
            "error": str(exc),
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"MAGMA product benchmark: {report['passed']}/{report['total']} ({report['pct']}%)")
        for item in report.get("cases", []):
            marker = "OK" if item.get("ok") else "FAIL"
            print(f"- [{marker}] {item['id']} [{item['category']}] {item['duration_ms']}ms")
            if item.get("error"):
                print(f"  error: {item['error']}")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
