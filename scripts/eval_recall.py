#!/usr/bin/env python3
"""MAGMA recall evaluation - standalone script"""
import sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\openclaw-magma')

from magma.graph.sqlite_store import SQLiteStore
from magma.vector.encoder import get_encoder
from magma.vector.faiss_index import get_faiss_index
from magma.search import MemorySearcher

test_queries = [
    'MAGMA FAISS 优化',
    '飞书群消息修复',
    '抖音详情图上传',
    'OpenClaw 版本升级',
    'Scrapling 爬虫工具',
    'ChatGPT 图片生成',
    'MAGMA 记忆整合',
    '技术部员工管理',
    '飞书权限配置',
    '飞书流式卡片',
    'Bonjour mDNS 崩溃',
    'usage 解析崩溃',
]

store = SQLiteStore()
store.initialize()
encoder = get_encoder()
faiss_index = get_faiss_index()
searcher = MemorySearcher(store, encoder, faiss_index)

total = 0
hits = 0
latencies = []

print("=" * 70)
print("MAGMA Recall Evaluation")
print("=" * 70)

for q in test_queries:
    t0 = time.time()
    results = searcher.query(q, top_k=5)
    latency = (time.time() - t0) * 1000
    latencies.append(latency)
    total += 1
    if results:
        hits += 1
        top = results[0]
        node_id = top.get('id', 'unknown')[:50]
        score = top.get('score', 0)
        print(f'OK [{latency:.0f}ms] "{q}"')
        print(f'   -> {node_id} (score={score:.3f})')
    else:
        print(f'XX [{latency:.0f}ms] "{q}" -> no results')

avg_latency = sum(latencies) / len(latencies)
print("=" * 70)
print(f'Result: {hits}/{total} hit rate {hits/total*100:.0f}%, avg latency {avg_latency:.0f}ms')
print(f'   fastest: {min(latencies):.0f}ms, slowest: {max(latencies):.0f}ms')
