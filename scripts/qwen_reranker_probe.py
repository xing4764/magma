"""Probe Qwen3-Reranker-0.6B as a second-stage MAGMA reranker.

The script calls the live MAGMA API for first-stage candidates, then reranks
them locally with Qwen. It is read-only and does not change production MAGMA.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).parent.parent


def _configured_api_base() -> str:
    env_base = os.environ.get("MAGMA_API_BASE")
    if env_base:
        return env_base.rstrip("/")
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
        import re

        match = re.search(r'"apiBaseUrl"\s*:\s*"(http://127\.0\.0\.1:\d+)"', text)
        if match:
            return match.group(1).rstrip("/")
    except OSError:
        pass
    return "http://127.0.0.1:8902"


API_BASE = _configured_api_base()
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3-Reranker-0___6B")

CASES = [
    "MAGMA MCP 为什么要改成 8902 主链路薄代理？",
    "recent_capture 变成 yellow 是什么意思，系统坏了吗？",
    "OpenClaw 为什么固定在 2026.5.20，不升级 5.22？",
    "source_agent_id 对跨 agent 记忆有什么作用？",
    "magma_doctor.py 和 magma_ops.py 分别用来做什么？",
    "yunying 运营部的 MAGMA 注入为什么之前是 0？",
    "MAGMA 网关卡顿和 embedding 反复加载有什么关系？",
    "Qwen3 Embedding 是否适合替换 bge-small-zh？",
]


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return None


def memory_text(item: dict) -> str:
    props = item.get("properties") or {}
    parts = []
    for key in ("title", "name", "content", "summary", "message", "source", "role", "agent_id"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if item.get("related_context"):
        parts.append("Related: " + json.dumps(item["related_context"], ensure_ascii=False)[:900])
    if item.get("version_context"):
        parts.append("Versions: " + json.dumps(item["version_context"], ensure_ascii=False)[:900])
    if not parts:
        parts.append(json.dumps({"label": item.get("label"), "properties": props}, ensure_ascii=False))
    return "\n".join(parts)[:2400]


def call_magma(query: str, candidate_k: int) -> tuple[list[dict], float]:
    payload = {
        "query": query,
        "top_k": candidate_k,
        "filters": {
            "include_related": True,
            "related_limit": 2,
            "include_versions": True,
            "version_limit": 2,
            "pool_size": 5000,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/api/v1/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("results", []), (time.perf_counter() - start) * 1000


def format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int, max_length: int) -> list[float]:
    """Score query/document pairs with Qwen's yes/no causal-LM reranker."""
    import torch

    instruction = (
        "Given a MAGMA memory search query, retrieve the memories that directly answer it. "
        "Prefer current state, operational facts, root causes, decisions, and exact configuration details. "
        "Penalize outdated, merely similar, or vague chat fragments."
    )
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            texts = [format_instruction(instruction, query, doc) for query, doc in batch]
            inputs = tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            )
            for i, ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
            encoded = tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            for key in encoded:
                encoded[key] = encoded[key].to(model.device)
            outputs = model(**encoded)
            batch_scores = outputs.logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            yes_no = torch.stack([false_vector, true_vector], dim=1)
            yes_no = torch.nn.functional.log_softmax(yes_no, dim=1)
            scores.extend(float(x) for x in yes_no[:, 1].exp().detach().cpu().tolist())
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen reranking over live MAGMA candidates.")
    parser.add_argument("--model", default=os.environ.get("MAGMA_RERANKER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    model_path = Path(args.model)
    print(f"api={API_BASE}")
    print(f"model={model_path}")
    before = rss_mb()
    if before is not None:
        print(f"rss_before_mb={before:.1f}")

    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True).eval()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"load_ms={load_ms:.1f}")
    after_load = rss_mb()
    if after_load is not None:
        print(f"rss_after_load_mb={after_load:.1f}")
        if before is not None:
            print(f"rss_load_delta_mb={after_load - before:.1f}")

    all_rerank_ms = []
    for query in CASES:
        candidates, magma_ms = call_magma(query, args.candidate_k)
        pairs = [(query, memory_text(item)) for item in candidates]
        start = time.perf_counter()
        scores = score_pairs(model, tokenizer, pairs, args.batch_size, args.max_length)
        rerank_ms = (time.perf_counter() - start) * 1000
        all_rerank_ms.append(rerank_ms)
        reranked = []
        for item, score in zip(candidates, scores):
            merged = dict(item)
            merged["rerank_score"] = score
            reranked.append(merged)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        print("\nQUERY", query)
        print(f"magma_ms={magma_ms:.1f} candidates={len(candidates)} rerank_ms={rerank_ms:.1f}")
        print("BGE_TOP")
        for rank, item in enumerate(candidates[:args.top_k], start=1):
            title = (item.get("properties") or {}).get("title") or (item.get("properties") or {}).get("name") or (item.get("properties") or {}).get("content") or item.get("id")
            print(f"{rank}. score={item.get('score')} id={item.get('id')} title={str(title).replace(chr(10), ' ')[:100]}")
        print("RERANK_TOP")
        for rank, item in enumerate(reranked[:args.top_k], start=1):
            title = (item.get("properties") or {}).get("title") or (item.get("properties") or {}).get("name") or (item.get("properties") or {}).get("content") or item.get("id")
            print(f"{rank}. rscore={item.get('rerank_score'):.4f} bge={item.get('score')} id={item.get('id')} title={str(title).replace(chr(10), ' ')[:100]}")

    if all_rerank_ms:
        arr = sorted(all_rerank_ms)
        print("\nSUMMARY")
        print(f"rerank_ms_min={arr[0]:.1f}")
        print(f"rerank_ms_p50={arr[len(arr)//2]:.1f}")
        print(f"rerank_ms_max={arr[-1]:.1f}")
    final = rss_mb()
    if final is not None:
        print(f"rss_final_mb={final:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
