"""
Embedding utilities using sentence-transformers.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer with a simple interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Infer dimension from a dummy embedding
        dummy = self.model.encode(["test"], convert_to_numpy=True)
        self.dimension = int(dummy.shape[1])

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into normalized embeddings.
        """
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string.
        """
        return self.encode_texts([text])[0]
