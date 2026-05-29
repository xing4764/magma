"""FAISS-based vector index for fast approximate nearest-neighbor search.

Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
making inner product equivalent to cosine similarity.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("magma.vector.faiss_index")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed; FAISS index unavailable")


class FAISSIndex:
    """Thread-safe FAISS index wrapper for MAGMA node embeddings.

    SQLite remains the source of truth; FAISS only accelerates vector lookup.
    """

    def __init__(self, dimension: int = 0):
        self._dimension = dimension
        self._index = None          # faiss.IndexFlatIP
        self._id_map: List[str] = []  # position -> node_id
        self._id_to_pos: Dict[str, int] = {}
        self._embeddings: Dict[str, np.ndarray] = {}  # node_id -> embedding (for rebuild)
        self._lock = threading.Lock()
        self._built = False

    @property
    def is_available(self) -> bool:
        return _FAISS_AVAILABLE and self._built and self._dimension > 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        if not self._built or self._index is None:
            return 0
        return self._index.ntotal

    def build_from_embeddings(self, entries: List[Tuple[str, np.ndarray]]):
        """Build a fresh index from (node_id, embedding) pairs.

        Args:
            entries: list of (node_id, np.ndarray float32) where ndarray is 1-D
                     and already L2-normalized.
        """
        if not _FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping build")
            return

        if not entries:
            with self._lock:
                self._dimension = self._dimension or 0
                self._index = None
                self._id_map = []
                self._id_to_pos = {}
                self._embeddings = {}
                self._built = True
            logger.info("FAISS index built with 0 vectors")
            return

        ids = []
        vectors = []
        emb_cache = {}
        for node_id, vec in entries:
            if vec is None or vec.ndim != 1:
                continue
            ids.append(node_id)
            vectors.append(vec.astype(np.float32))
            emb_cache[node_id] = vec.astype(np.float32)

        if not vectors:
            with self._lock:
                self._index = None
                self._id_map = []
                self._id_to_pos = {}
                self._embeddings = {}
                self._built = True
            logger.info("FAISS index built with 0 valid vectors")
            return

        mat = np.array(vectors, dtype=np.float32)
        dim = mat.shape[1]

        # Normalize (safety; caller should already normalize)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-10)

        index = faiss.IndexFlatIP(dim)

        with self._lock:
            index.add(mat)
            self._index = index
            self._dimension = dim
            self._id_map = list(ids)
            self._id_to_pos = {nid: i for i, nid in enumerate(ids)}
            self._embeddings = emb_cache
            self._built = True

        logger.info(f"FAISS index built: {len(ids)} vectors, dim={dim}")

    def add(self, node_id: str, embedding: np.ndarray):
        """Add a single vector to the index. If node_id already exists, replaces it (upsert)."""
        if not _FAISS_AVAILABLE or not self._built:
            return
        if embedding is None or embedding.ndim != 1:
            return

        vec = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._lock:
            # Upsert: remove old entry if exists
            if node_id in self._id_to_pos:
                self._remove_locked(node_id)

            if self._index is None:
                self._index = faiss.IndexFlatIP(vec.shape[1])
                self._dimension = vec.shape[1]
            if vec.shape[1] != self._dimension:
                logger.warning(f"Dimension mismatch: got {vec.shape[1]}, expected {self._dimension}")
                return
            self._index.add(vec)
            self._id_map.append(node_id)
            self._id_to_pos[node_id] = len(self._id_map) - 1
            self._embeddings[node_id] = embedding.astype(np.float32)

    def _remove_locked(self, node_id: str):
        """Internal remove (must hold self._lock). Uses rebuild approach."""
        if node_id not in self._id_to_pos:
            return
        # Rebuild index without this node
        keep_entries = [(nid, emb) for nid, emb in self._embeddings.items() if nid != node_id]
        del self._embeddings[node_id]
        if keep_entries:
            ids, vecs = zip(*keep_entries)
            mat = np.array(vecs, dtype=np.float32)
            self._index = faiss.IndexFlatIP(mat.shape[1])
            self._index.add(mat)
            self._id_map = list(ids)
            self._id_to_pos = {nid: i for i, nid in enumerate(self._id_map)}
        else:
            self._index = None
            self._id_map = []
            self._id_to_pos = {}

    def remove(self, node_id: str):
        """Remove a vector by node_id.

        Actually removes from the FAISS index by rebuilding without the
        deleted entry. Uses cached embeddings to avoid re-fetching.
        """
        if not _FAISS_AVAILABLE or not self._built:
            return
        with self._lock:
            if node_id not in self._id_to_pos:
                return
            # Remove from embeddings cache
            self._embeddings.pop(node_id, None)
            # Rebuild index without the deleted node
            self._rebuild_without_deleted_locked()
            logger.debug(f"FAISS removed {node_id}, index now has {self.count} vectors")

    def _rebuild_without_deleted_locked(self):
        """Rebuild the FAISS index using cached embeddings, excluding removed nodes.
        Caller must hold self._lock."""
        if not self._embeddings:
            self._index = None
            self._id_map = []
            self._id_to_pos = {}
            return

        ids = list(self._embeddings.keys())
        vectors = [self._embeddings[nid] for nid in ids]
        mat = np.array(vectors, dtype=np.float32)
        dim = mat.shape[1]

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-10)

        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        self._index = index
        self._dimension = dim
        self._id_map = ids
        self._id_to_pos = {nid: i for i, nid in enumerate(ids)}

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: 1-D float32 array, should be L2-normalized
            top_k: number of results

        Returns:
            List of (node_id, score) sorted by score descending.
            Score is inner product (= cosine for normalized vectors).
        """
        if not self.is_available or self._index is None:
            return []

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._lock:
            if self._index.ntotal == 0:
                return []
            # Request more candidates to account for tombstones
            k = min(top_k * 2, self._index.ntotal)
            scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            node_id = self._id_map[idx]
            if node_id is None:  # tombstone
                continue
            results.append((node_id, float(score)))
            if len(results) >= top_k:
                break
        return results

    def rebuild(self, entries: List[Tuple[str, np.ndarray]]):
        """Full rebuild (alias for build_from_embeddings)."""
        # Clear embeddings cache before rebuild
        self._embeddings = {}
        self.build_from_embeddings(entries)

    def clear(self):
        """Clear the index."""
        with self._lock:
            self._index = None
            self._id_map = []
            self._id_to_pos = {}
            self._embeddings = {}
            self._built = False


def get_faiss_index(dimension: int = 0) -> FAISSIndex:
    """Get or create a module-level FAISS index singleton."""
    if not hasattr(get_faiss_index, "_instance"):
        get_faiss_index._instance = FAISSIndex(dimension)
    elif dimension > 0 and get_faiss_index._instance.dimension == 0:
        get_faiss_index._instance._dimension = dimension
    return get_faiss_index._instance
