from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed_texts(self, texts: Iterable[str], max_dim: Optional[int] = None) -> np.ndarray:
        """
        Return an array of shape (num_items, dim). If max_dim is set, return only
        the first max_dim dimensions (Matryoshka-style truncation).
        """


