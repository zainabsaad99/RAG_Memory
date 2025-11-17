"""
FAISS-based vector store for dense retrieval.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Tuple, Optional

import numpy as np
import faiss


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


class FaissVectorStore:
    """
    Minimal FAISS vector store using inner product on normalized embeddings.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_store: Dict[int, DocumentChunk] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._next_idx = 0

    def build_index(
        self,
        docs: List[DocumentChunk],
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> None:
        self.index.reset()
        self.doc_store.clear()
        self._id_to_idx.clear()
        self._next_idx = 0

        texts = [d.text for d in docs]
        embs = embed_fn(texts)

        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        self.index.add(embs.astype("float32"))
        for i, d in enumerate(docs):
            self.doc_store[i] = d
            self._id_to_idx[d.id] = i
            self._next_idx += 1

    def add_docs(
        self,
        docs: List[DocumentChunk],
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> None:
        texts = [d.text for d in docs]
        embs = embed_fn(texts)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        self.index.add(embs.astype("float32"))
        for i, d in enumerate(docs):
            idx = self._next_idx + i
            self.doc_store[idx] = d
            self._id_to_idx[d.id] = idx
        self._next_idx += len(docs)

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        scores, indices = self.index.search(
            query_emb.astype("float32"), top_k * 5
        )  # oversample

        scores = scores[0]
        indices = indices[0]

        results: List[Tuple[DocumentChunk, float]] = []

        for idx, score in zip(indices, scores):
            if idx == -1:
                continue
            chunk = self.doc_store.get(idx)
            if chunk is None:
                continue

            if metadata_filter:
                ok = True
                for k, v in metadata_filter.items():
                    if chunk.metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
