from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..data.models import Collection


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class SimilarityRanker:
    def __init__(self, embedding_backend, similarity: str = "cosine", dims: int | None = None) -> None:
        self.backend = embedding_backend
        self.similarity = similarity
        self.dims = dims

    def _sim(self, topics_emb: np.ndarray, docs_emb: np.ndarray) -> np.ndarray:
        # Prefer backend-provided similarity when available
        if hasattr(self.backend, "similarity"):
            return getattr(self.backend, "similarity")(topics_emb, docs_emb)
        if self.similarity == "cosine":
            return _cosine_similarity(topics_emb, docs_emb)
        raise ValueError(f"Unknown similarity: {self.similarity}")

    def score_docs_for_topics(self, collection: Collection) -> np.ndarray:
        # Prefer precomputed embeddings stored in collection
        if collection.topics_embeddings is not None and collection.docs_embeddings is not None:
            t = collection.topics_embeddings
            d = collection.docs_embeddings
            if self.dims is not None:
                t = t[:, : self.dims]
                d = d[:, : self.dims]
            return self._sim(t, d)

        topics_map = collection.topics
        docs = list(collection.documents.values())
        topic_texts = list(topics_map.values())
        doc_texts = [d.text for d in docs]
        topics_emb = self.backend.embed_texts(topic_texts, max_dim=self.dims)
        docs_emb = self.backend.embed_texts(doc_texts, max_dim=self.dims)
        return self._sim(topics_emb, docs_emb)


