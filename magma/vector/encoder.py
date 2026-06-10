"""Embedding encoder using sentence-transformers."""

import os
import numpy as np
from pathlib import Path
from typing import List, Union

# Use the official Hub by default. Set HF_ENDPOINT explicitly when a mirror is needed.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

_models = {}
_LOCAL_QWEN = str(Path(__file__).parent.parent.parent / "models" / "Qwen" / "Qwen3-Embedding-4B")
_MODEL_NAME = os.environ.get("MAGMA_EMBEDDING_MODEL", _LOCAL_QWEN)


def _get_model(model_name: str = _MODEL_NAME):
    model_name = model_name or _MODEL_NAME
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


class Encoder:
    """Sentence embedding encoder wrapping sentence-transformers."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
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
        
        if single:
            return embeddings[0]
        return embeddings

    @property
    def dimension(self) -> int:
        # Compatible with both old (get_sentence_embedding_dimension) and new (get_embedding_dimension)
        if hasattr(self.model, 'get_embedding_dimension'):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()


def get_encoder(model_name: str = None) -> Encoder:
    return Encoder(model_name or _MODEL_NAME)
