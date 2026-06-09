"""P2-4: Sentence Compression Module.

Extracts top-K key sentences from long content nodes using BM25 scoring.
Reduces context token consumption while preserving the most informative sentences.

Feature flag: MAGMA_FEATURE_SENTENCE_COMPRESS (default OFF)
Performance target: single node compression < 20ms
"""

import logging
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("magma.sentence_compress")

# Feature flag — default OFF (conservative, opt-in)
MAGMA_FEATURE_SENTENCE_COMPRESS = os.environ.get("MAGMA_FEATURE_SENTENCE_COMPRESS", "0") == "1"

# Default number of key sentences to extract
DEFAULT_TOP_K = int(os.environ.get("MAGMA_SENTENCE_COMPRESS_K", "4"))

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (Chinese + English aware)."""
    if not text:
        return []
    # Split on Chinese/English sentence boundaries
    sentences = re.split(r'(?<=[。！？!?\.\n])\s*', text.strip())
    # Filter empty and very short sentences
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 8:  # Minimum meaningful sentence length
            result.append(s)
    return result


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 (Chinese char + English word)."""
    tokens = []
    # English words
    tokens.extend(re.findall(r'[A-Za-z][A-Za-z0-9_.\-]{1,}', text.lower()))
    # Chinese characters (each char as token for BM25)
    tokens.extend(re.findall(r'[\u4e00-\u9fff]', text))
    # Technical terms
    tokens.extend(re.findall(r'[0-9]+(?:\.[0-9]+)?', text))
    return tokens


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_dl: float,
    total_docs: int,
    doc_freqs: Dict[str, int],
) -> float:
    """Compute BM25 score for a single document against query terms."""
    if not doc_tokens or not query_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0

    for qt in query_tokens:
        if qt not in tf:
            continue
        term_freq = tf[qt]
        df = doc_freqs.get(qt, 1)
        # IDF component
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        # TF component with length normalization
        tf_norm = (term_freq * (BM25_K1 + 1)) / (
            term_freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_dl, 1))
        )
        score += idf * tf_norm

    return score


def compress_sentences(
    content: str,
    query: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[str], bool]:
    """Extract top-K key sentences from content using BM25 scoring.

    Args:
        content: full text content to compress
        query: optional query for relevance scoring (if empty, uses term frequency)
        top_k: number of sentences to keep

    Returns:
        (list of selected sentences, True if compression was applied)
    """
    if not content:
        return [], False

    sentences = _split_sentences(content)
    if len(sentences) <= top_k:
        # Content is already short enough, no compression needed
        return sentences, False

    t0 = time.time()

    # Tokenize all sentences
    tokenized = [_tokenize(s) for s in sentences]

    # Build document frequency table
    doc_freqs: Dict[str, int] = Counter()
    for tokens in tokenized:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freqs[t] += 1

    total_docs = len(sentences)
    avg_dl = sum(len(t) for t in tokenized) / max(total_docs, 1)

    # Query tokens: either from explicit query or top-frequency terms
    if query:
        query_tokens = _tokenize(query)
    else:
        # Auto-extract key terms from content
        all_tokens = [t for tokens in tokenized for t in tokens]
        term_freq = Counter(all_tokens)
        query_tokens = [t for t, _ in term_freq.most_common(20)]

    # Score each sentence
    scored: List[Tuple[int, float, str]] = []
    for i, (sentence, tokens) in enumerate(zip(sentences, tokenized)):
        score = _bm25_score(query_tokens, tokens, avg_dl, total_docs, doc_freqs)
        scored.append((i, score, sentence))

    # Select top-K, preserving original order
    scored.sort(key=lambda x: x[1], reverse=True)
    selected_indices = sorted([item[0] for item in scored[:top_k]])
    selected = [sentences[i] for i in selected_indices]

    elapsed_ms = (time.time() - t0) * 1000
    if elapsed_ms > 20:
        logger.warning(f"Sentence compress took {elapsed_ms:.1f}ms (>20ms target)")

    return selected, True


def apply_compression_to_node(
    node: Dict[str, Any],
    query: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Apply sentence compression to a result node in-place.

    Args:
        node: result dict with properties
        query: original search query
        top_k: sentences to keep

    Returns:
        node with compressed content
    """
    props = node.get("properties", {}) or {}
    content = props.get("content", "") or props.get("summary", "") or ""

    if not content:
        return node

    compressed, was_compressed = compress_sentences(content, query=query, top_k=top_k)

    if was_compressed:
        node["compressed"] = True
        node["compressed_content"] = "\n".join(compressed)
        node["original_length"] = len(content)
        node["compressed_length"] = len(node["compressed_content"])
        node["compression_ratio"] = round(
            node["compressed_length"] / max(node["original_length"], 1), 3
        )
        logger.debug(
            f"Compressed node {node.get('id')}: "
            f"{node['original_length']} -> {node['compressed_length']} chars "
            f"({node['compression_ratio']:.1%})"
        )
    else:
        node["compressed"] = False

    return node


def is_compress_enabled() -> bool:
    """Check if sentence compression is enabled."""
    return MAGMA_FEATURE_SENTENCE_COMPRESS


def get_compress_status() -> Dict[str, Any]:
    """Return compression module status for diagnostics."""
    return {
        "feature_flag": MAGMA_FEATURE_SENTENCE_COMPRESS,
        "default_top_k": DEFAULT_TOP_K,
        "bm25_params": {"k1": BM25_K1, "b": BM25_B},
    }
