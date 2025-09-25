from .data.io import load_collection, load_collection_with_embeddings
from .evaluation.pipeline import CollectionQualityEvaluator

# Lazy import for optional backend to avoid heavy deps at import time
def SentenceTransformerBackend(*args, **kwargs):  # type: ignore
    from .embeddings.sentence_transformer import SentenceTransformerBackend as _Backend
    return _Backend(*args, **kwargs)

__all__ = [
    "load_collection",
    "load_collection_with_embeddings",
    "CollectionQualityEvaluator",
    "SentenceTransformerBackend",
]


