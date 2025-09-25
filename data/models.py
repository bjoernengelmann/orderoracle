from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import numpy as np


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Qrel:
    topic_id: str
    doc_id: str
    relevance: int

@dataclass
class Collection:
    documents: Dict[str, Document]
    topics: Dict[str, str]
    qrels: List[Qrel]
    topics_embeddings: Optional[np.ndarray] = None  # shape [T, D]
    docs_embeddings: Optional[np.ndarray] = None    # shape [D, D]


