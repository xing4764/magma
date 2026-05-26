"""Embedding encoder using sentence-transformers."""

import os
import numpy as np
from typing import List, Union

# Use the official Hub by default. Set HF_ENDPOINT explicitly when a mirror is needed.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

_model = None
_MODEL_NAME = os.environ.get("MAGMA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class Encoder:
    """Sentence embedding encoder wrapping sentence-transformers."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
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
        return self.model.get_sentence_embedding_dimension()


def get_encoder(model_name: str = None) -> Encoder:
    return Encoder(model_name or _MODEL_NAME)
