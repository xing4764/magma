"""Embedding encoder using sentence-transformers with idle timeout for memory optimization."""

import gc
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Union

logger = logging.getLogger("magma.vector.encoder")

# Use the official Hub by default. Set HF_ENDPOINT explicitly when a mirror is needed.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

_models = {}
_LOCAL_QWEN = str(Path(__file__).parent.parent.parent / "models" / "Qwen" / "Qwen3-Embedding-4B")
_MODEL_NAME = os.environ.get("MAGMA_EMBEDDING_MODEL", _LOCAL_QWEN)
_IDLE_TIMEOUT = int(os.environ.get("MAGMA_ENCODER_IDLE_TIMEOUT", "1800"))


def _get_model(model_name: str = _MODEL_NAME):
    model_name = model_name or _MODEL_NAME
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


class Encoder:
    """Sentence embedding encoder wrapping sentence-transformers.
    
    Supports idle timeout: model is automatically unloaded after MAGMA_ENCODER_IDLE_TIMEOUT
    seconds of inactivity to free memory. It will be reloaded on next use.
    """

    def __init__(self, model_name: str = _MODEL_NAME, idle_timeout: int = _IDLE_TIMEOUT):
        self.model_name = model_name
        self._model = None
        self._last_used = 0.0
        self._idle_timeout = idle_timeout

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
        self._last_used = time.time()
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode text(s) to embeddings.
        
        Returns:
            np.ndarray of shape (n, dim) or (dim,) for single text.
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        self._last_used = time.time()
        
        if single:
            return embeddings[0]
        return embeddings

    @property
    def dimension(self) -> int:
        # Compatible with both old (get_sentence_embedding_dimension) and new (get_embedding_dimension)
        if hasattr(self.model, 'get_embedding_dimension'):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def maybe_unload(self) -> bool:
        """Unload model if idle timeout exceeded. Returns True if model was unloaded."""
        if self._model is None:
            return False
        if self._idle_timeout <= 0:
            return False
        if time.time() - self._last_used > self._idle_timeout:
            logger.info(f"Unloading embedding model (idle {self._idle_timeout}s)")
            del self._model
            self._model = None
            gc.collect()
            return True
        return False

    def force_unload(self):
        """Force unload model immediately."""
        if self._model is not None:
            logger.info("Force unloading embedding model")
            del self._model
            self._model = None
            gc.collect()


class CloudEncoder:
    """Cloud embedding encoder using Alibaba Cloud DashScope API."""

    def __init__(self):
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self._model = os.environ.get("MAGMA_CLOUD_EMBEDDING_MODEL", "text-embedding-v4")
        self._dimension = int(os.environ.get("MAGMA_CLOUD_EMBEDDING_DIM", "2048"))
        self._session = None
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"cloud/{self._model}"

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.trust_env = False  # Ignore proxy env vars
        return self._session

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode text(s) to embeddings via cloud API."""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        session = self._get_session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimension,
            "encoding_format": "float"
        }

        try:
            # Ensure UTF-8 encoding for JSON payload
            import json as _json
            payload = _json.dumps(data, ensure_ascii=False).encode('utf-8')
            resp = session.post(
                self._api_url,
                headers=headers,
                data=payload,
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
            embeddings = [item["embedding"] for item in result["data"]]
            embeddings = np.array(embeddings, dtype=np.float32)

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

            if single:
                return embeddings[0]
            return embeddings
        except Exception as e:
            logger.error(f"Cloud embedding API error: {e}")
            raise

    def maybe_unload(self) -> bool:
        """No-op for cloud encoder."""
        return False

    def force_unload(self):
        """No-op for cloud encoder."""
        pass


def get_encoder(model_name: str = None) -> Union[Encoder, CloudEncoder]:
    """Get encoder based on MAGMA_EMBEDDING_BACKEND env var."""
    backend = os.environ.get("MAGMA_EMBEDDING_BACKEND", "local").lower()
    if backend == "cloud":
        logger.info("Using cloud embedding backend")
        return CloudEncoder()
    return Encoder(model_name or _MODEL_NAME)
