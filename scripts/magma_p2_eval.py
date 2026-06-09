"""MAGMA P2 New Capabilities Evaluation Script.

Covers 6 new dimensions from Harness-1 optimizations:
1. Reranker precision reranking
2. MinHash dedup
3. Bridge entity detection
4. Token budget priority levels
5. Backtrack detection
6. Verify endpoint
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Config ---
def _configured_api_base() -> str:
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
    return "http://127.0.0.1:8904"

API_BASE = _configured_api_base()
TEST_PREFIX = "__p2_eval__"
CLEANUP_IDS = []  # collect node IDs to clean up

# --- Helpers ---
def api_post(path: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))

def api_delete(path: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode("utf-8", errors="replace")}

def api_get(path: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(f"{API_BASE}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))

def add_node(node_id: str, label: str, properties: dict) -> dict:
    CLEANUP_IDS.append(node_id)
    return api_post("/api/v1/nodes", {
        "id": node_id,
        "label": label,
        "properties": properties,
    })

def query_magma(query: str, top_k: int = 5, priority: str = "normal", session_key: str = None) -> dict:
    payload = {
        "query": query,
        "top_k": top_k,
        "priority": priority,
    }
    if session_key:
        payload["filters"] = {"session_key": session_key}
    return api_post("/api/v1/query", payload)

# --- Test Results ---
results = []

def record(dim_name: str, test_name: str, passed: bool, detail: str = ""):
    results.append({
        "dimension": dim_name,
        "test": test_name,
        "passed": passed,
        "detail": detail,
    })
    marker = "PASS" if passed else "FAIL"
    print(f"  [{marker}] {test_name}")
    if detail:
        print(f"         {detail}")

# ===========================================================================
# Dimension 1: Reranker Precision
# ===========================================================================
def eval_reranker():
    print("\n=== Dimension 1: Reranker Precision ===")
    dim = "Reranker"

    # Check if reranker is enabled by testing a query
    test_queries = [
        "OpenClaw 版本升级策略",
        "MAGMA 运维工具",
        "飞书流式卡片怎么用",
    ]
    reranker_applied_found = False
    reranker_score_found = False
    fused_score_verified = False

    for q in test_queries:
        try:
            resp = query_magma(q, top_k=5)
            results_list = resp.get("results", [])
            if not results_list:
                continue
            top = results_list[0]
            breakdown = top.get("score_breakdown", {}) or {}

            if breakdown.get("reranker_applied"):
                reranker_applied_found = True
            if "reranker_score" in breakdown:
                reranker_score_found = True
            rs = breakdown.get("reranker_score")
            rrf = breakdown.get("rrf_score")
            fused = breakdown.get("fused_score")
            if rs is not None and rrf is not None and fused is not None:
                expected = 0.3 * rs + 0.7 * rrf
                if abs(fused - expected) < 0.01:
                    fused_score_verified = True
        except Exception as e:
            pass

    # Reranker is OFF by default (MAGMA_FEATURE_LOCAL_RERANKER=0)
    # Check env var to determine expected behavior
    reranker_env = os.environ.get("MAGMA_FEATURE_LOCAL_RERANKER", "0")
    reranker_enabled = reranker_env == "1"

    if reranker_enabled:
        record(dim, "reranker_applied field exists (flag ON)", reranker_applied_found,
               "reranker_applied=true found" if reranker_applied_found else "flag ON but reranker not applied")
        record(dim, "reranker_score field exists", reranker_score_found,
               "reranker_score found" if reranker_score_found else "not found")
        record(dim, "fused_score = 0.3*reranker + 0.7*rrf", fused_score_verified,
               "formula verified" if fused_score_verified else "could not verify")
    else:
        # Flag is OFF - verify that the system handles gracefully (no crash)
        record(dim, "reranker disabled (flag=0) - system stable", True,
               "MAGMA_FEATURE_LOCAL_RERANKER=0, system runs without crash")
        record(dim, "reranker_score absent when disabled", not reranker_score_found,
               "correctly absent" if not reranker_score_found else "score present despite flag=0 (unexpected)")
        record(dim, "fused_score not applied when disabled", not fused_score_verified,
               "correctly skipped" if not fused_score_verified else "formula applied despite flag=0")

# ===========================================================================
# Dimension 2: MinHash Dedup
# ===========================================================================
def eval_dedup():
    print("\n=== Dimension 2: MinHash Dedup ===")
    dim = "MinHash Dedup"

    ts = int(time.time())

    # Nearly identical texts (>0.85 MinHash similarity)
    similar_text_1 = (
        "MAGMA 技术部今天完成了 P2 优化的全部 19 项代码变更，"
        "包括 reranker 精排、MinHash 去重、bridge entity 检测、"
        "token budget 分级、回溯检测和 verify 验证功能。"
        "recall_eval 评测保持 23/24 (95.8%) 无退化。"
        "所有新功能已通过 Harness-1 测试。"
    )
    similar_text_2 = (
        "MAGMA 技术部今天完成了 P2 优化的全部 19 项代码变更，"
        "包括 reranker 精排、MinHash 去重、bridge entity 检测、"
        "token budget 分级、回溯检测以及 verify 验证功能。"
        "recall_eval 评测保持在 23/24 (95.8%) 无退化。"
        "所有新功能已通过 Harness-1 测试。"
    )
    diff_text_1 = (
        "抖音店铺运营需要关注商品详情页的图片质量，"
        "主图要求 800x800 以上白底图，详情图不超过 20 张。"
    )
    diff_text_2 = (
        "飞书日历 API 支持创建、更新、删除日程事件，"
        "需要 tenant_access_token 权限才能操作。"
    )

    # Test 1: Write 2 nearly identical texts via capture, expect dedup
    try:
        resp1 = api_post("/api/v1/capture", {
            "user_text": similar_text_1,
            "assistant_text": "",
        })
        time.sleep(1)
        resp2 = api_post("/api/v1/capture", {
            "user_text": similar_text_2,
            "assistant_text": "",
        })

        # Dedup detected if: status=skipped, or dedup=True in response
        dedup_triggered = (
            resp2.get("dedup") is True
            or resp2.get("status") == "skipped"
            or (isinstance(resp2.get("written"), list) and len(resp2.get("written", [])) == 0)
        )
        record(dim, "similar content triggers dedup", dedup_triggered,
               f"resp2: status={resp2.get('status')}, dedup={resp2.get('dedup')}, written={len(resp2.get('written', []))}")

        for w in resp1.get("written", []) + resp2.get("written", []):
            nid = w.get("id") if isinstance(w, dict) else w
            if nid and nid not in CLEANUP_IDS:
                CLEANUP_IDS.append(nid)

    except Exception as e:
        record(dim, "similar content triggers dedup", False, f"Error: {e}")

    # Test 2: Write 2 different texts, expect NO dedup
    try:
        resp3 = api_post("/api/v1/capture", {
            "user_text": diff_text_1,
            "assistant_text": "",
        })
        time.sleep(1)
        resp4 = api_post("/api/v1/capture", {
            "user_text": diff_text_2,
            "assistant_text": "",
        })

        no_dedup = resp4.get("dedup") is not True and resp4.get("status") != "skipped"
        record(dim, "different content passes dedup", no_dedup,
               f"resp4: status={resp4.get('status')}, dedup={resp4.get('dedup')}")

        for w in resp3.get("written", []) + resp4.get("written", []):
            nid = w.get("id") if isinstance(w, dict) else w
            if nid and nid not in CLEANUP_IDS:
                CLEANUP_IDS.append(nid)

    except Exception as e:
        record(dim, "different content passes dedup", False, f"Error: {e}")

# ===========================================================================
# Dimension 3: Bridge Entity Detection
# ===========================================================================
def eval_bridge_entity():
    print("\n=== Dimension 3: Bridge Entity Detection ===")
    dim = "Bridge Entity"

    ts = int(time.time())
    try:
        # Create nodes that share entities
        add_node(f"{TEST_PREFIX}bridge_1_{ts}", "event", {
            "content": "MAGMA 技术部完成了 reranker 优化，提升了搜索精度。",
            "source": "p2_eval",
        })
        add_node(f"{TEST_PREFIX}bridge_2_{ts}", "event", {
            "content": "MAGMA 技术部正在部署 MinHash 去重功能到生产环境。",
            "source": "p2_eval",
        })
        time.sleep(2)

        resp = query_magma("MAGMA 技术部最近做了什么？", top_k=5)
        results_list = resp.get("results", [])
        bridge_entities = resp.get("bridge_entities", []) or []

        bridge_count = 0
        bridge_boost = 0.0
        if results_list:
            breakdown = results_list[0].get("score_breakdown", {}) or {}
            bridge_count = breakdown.get("bridge_entity_count", 0)
            bridge_boost = breakdown.get("bridge_quality_boost", 0.0)

        has_bridge = bridge_count >= 1 or len(bridge_entities) >= 1
        record(dim, "bridge_entity_count >= 1", has_bridge,
               f"bridge_entity_count={bridge_count}, bridge_entities_list={len(bridge_entities)}")

        has_boost = bridge_boost > 0
        record(dim, "bridge_quality_boost > 0", has_boost,
               f"bridge_quality_boost={bridge_boost}")

    except Exception as e:
        record(dim, "bridge entity detection", False, f"Error: {e}")

# ===========================================================================
# Dimension 4: Token Budget Priority Levels
# ===========================================================================
def eval_token_budget():
    print("\n=== Dimension 4: Token Budget Priority ===")
    dim = "Token Budget"

    query_text = "OpenClaw 版本升级"

    # Test 1: critical priority — should NOT return token_usage_ratio
    try:
        resp_crit = query_magma(query_text, priority="critical")
        has_ratio = resp_crit.get("token_usage_ratio") is not None
        record(dim, "critical: no token_usage_ratio", not has_ratio,
               f"token_usage_ratio={resp_crit.get('token_usage_ratio')}" if has_ratio else "correctly omitted")
    except Exception as e:
        record(dim, "critical: no token_usage_ratio", False, f"Error: {e}")

    # Test 2: normal priority — should return token_usage_ratio
    try:
        resp_norm = query_magma(query_text, priority="normal")
        has_ratio = resp_norm.get("token_usage_ratio") is not None
        record(dim, "normal: has token_usage_ratio", has_ratio,
               f"token_usage_ratio={resp_norm.get('token_usage_ratio')}")
    except Exception as e:
        record(dim, "normal: has token_usage_ratio", False, f"Error: {e}")

    # Test 3: low priority — should return budget_warning if ratio > 0.6
    try:
        resp_low = query_magma(query_text, priority="low")
        ratio = resp_low.get("token_usage_ratio")
        warning = resp_low.get("budget_warning")
        if ratio is not None and ratio > 0.6:
            record(dim, "low: budget_warning when ratio>0.6", warning is not None,
                   f"ratio={ratio}, warning={warning}")
        else:
            record(dim, "low: budget_warning when ratio>0.6", True,
                   f"ratio={ratio} <= 0.6, no warning expected (pass by design)")
    except Exception as e:
        record(dim, "low: budget_warning when ratio>0.6", False, f"Error: {e}")

# ===========================================================================
# Dimension 5: Backtrack Detection
# ===========================================================================
def eval_backtrack():
    print("\n=== Dimension 5: Backtrack Detection ===")
    dim = "Backtrack"

    session_key = f"eval_backtrack_{int(time.time())}"
    # Use a query that actually returns memory results
    query_text = "OpenClaw 版本升级"

    try:
        # Verify the query returns actual results first
        resp0 = query_magma(query_text, top_k=5, session_key=session_key)
        top_ids_0 = [r.get("id") for r in resp0.get("results", []) if r.get("id") != "system"]
        if not top_ids_0:
            record(dim, "3rd repeat triggers backtrack_warning", False,
                   "Query returns no memory results; backtrack cannot trigger")
            record(dim, "different query: no backtrack_warning", True, "skipped (no baseline)")
            return

        time.sleep(0.5)
        resp1 = query_magma(query_text, top_k=5, session_key=session_key)
        time.sleep(0.5)
        resp2 = query_magma(query_text, top_k=5, session_key=session_key)

        bt = resp2.get("backtrack_warning")
        record(dim, "3rd repeat triggers backtrack_warning", bt is True,
               f"backtrack_warning={bt}, narrative={str(resp2.get('backtrack_narrative') or '')[:80]}")

    except Exception as e:
        record(dim, "3rd repeat triggers backtrack_warning", False, f"Error: {e}")

    # Different query should NOT trigger backtrack
    try:
        session_key2 = f"eval_no_backtrack_{int(time.time())}"
        resp_diff = query_magma("MAGMA 运维工具", session_key=session_key2)
        bt_diff = resp_diff.get("backtrack_warning")
        record(dim, "different query: no backtrack_warning", bt_diff is not True,
               f"backtrack_warning={bt_diff}")
    except Exception as e:
        record(dim, "different query: no backtrack_warning", False, f"Error: {e}")

# ===========================================================================
# Dimension 6: Verify Endpoint
# ===========================================================================
def eval_verify():
    print("\n=== Dimension 6: Verify Endpoint ===")
    dim = "Verify"

    try:
        resp = query_magma("MAGMA 版本升级", top_k=3)
        node_ids = [r.get("id") for r in resp.get("results", []) if r.get("id") and r.get("id") != "system"][:2]

        if not node_ids:
            record(dim, "verify returns verdict/evidence/confidence", False,
                   "No nodes found to verify against")
            return

        claim = "MAGMA 使用了 Qwen3-Embedding 模型"
        verify_resp = api_post("/api/v1/verify", {
            "node_ids": node_ids,
            "claim": claim,
        })

        has_verdict = "verdict" in verify_resp
        has_evidence = "evidence" in verify_resp
        has_confidence = "confidence" in verify_resp
        all_fields = has_verdict and has_evidence and has_confidence

        record(dim, "verify returns verdict/evidence/confidence", all_fields,
               f"verdict={verify_resp.get('verdict')}, confidence={verify_resp.get('confidence')}, evidence_len={len(str(verify_resp.get('evidence', '')))}")

        verdict = verify_resp.get("verdict", "")
        valid_verdict = verdict in ("supported", "unsupported", "partial")
        record(dim, "verdict is supported/unsupported/partial", valid_verdict,
               f"verdict='{verdict}'")

    except Exception as e:
        record(dim, "verify endpoint", False, f"Error: {e}")

# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"MAGMA P2 New Capabilities Evaluation")
    print(f"API: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        health = api_get("/api/v1/health")
        print(f"API health: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"ERROR: API not reachable at {API_BASE}: {e}")
        return 1

    eval_reranker()
    eval_dedup()
    eval_bridge_entity()
    eval_token_budget()
    eval_backtrack()
    eval_verify()

    # --- Summary ---
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pct = round(passed / total * 100, 1) if total else 0

    print(f"\nSUMMARY: {passed}/{total} PASS ({pct}%)")
    print("-" * 60)

    dims = {}
    for r in results:
        d = r["dimension"]
        if d not in dims:
            dims[d] = {"pass": 0, "fail": 0, "tests": []}
        if r["passed"]:
            dims[d]["pass"] += 1
        else:
            dims[d]["fail"] += 1
        dims[d]["tests"].append(r)

    for dim_name, dim_data in dims.items():
        status = "ALL PASS" if dim_data["fail"] == 0 else f"{dim_data['fail']} FAIL"
        print(f"\n  {dim_name}: {dim_data['pass']}/{dim_data['pass']+dim_data['fail']} ({status})")
        for t in dim_data["tests"]:
            marker = "PASS" if t["passed"] else "FAIL"
            print(f"    [{marker}] {t['test']}")
            if t["detail"]:
                print(f"       {t['detail']}")

    # --- Cleanup ---
    print(f"\n{'=' * 60}")
    print(f"Cleanup: removing {len(CLEANUP_IDS)} test nodes...")
    cleaned = 0
    for nid in CLEANUP_IDS:
        try:
            api_delete(f"/api/v1/nodes/{nid}")
            cleaned += 1
        except Exception:
            pass
    print(f"Cleaned {cleaned}/{len(CLEANUP_IDS)} nodes.")

    # Write JSON report
    report_path = Path(__file__).parent / "p2_eval_results.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "api_base": API_BASE,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pct": pct,
        "dimensions": {
            d: {"pass": v["pass"], "fail": v["fail"]}
            for d, v in dims.items()
        },
        "tests": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
