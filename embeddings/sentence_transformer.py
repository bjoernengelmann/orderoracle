from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception as e:  # pragma: no cover - optional heavy deps
    torch = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

from .base import EmbeddingBackend


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. Install: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = self._normalize_device(device)
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(model_name, device=self.device or None)

    def _normalize_device(self, device: Optional[str]) -> Optional[str]:
        if device is None:
            return None
        dev = device.strip().lower()
        if dev in ("gpu", "cuda"):
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if dev == "mps":
            # mps only on macOS ARM with torch compiled with mps
            if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
            return "cpu"
        if dev == "cpu":
            return "cpu"
        # Pass-through custom strings like 'cuda:1'
        return device

    def embed_texts(self, texts: Iterable[str], max_dim: Optional[int] = None, show_progress_bar: bool = True) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            texts_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress_bar,
        )
        if max_dim is not None:
            k = int(max_dim)
            if embeddings.shape[1] < k:
                raise ValueError(f"Requested dims {k} > model dim {embeddings.shape[1]}")
            embeddings = embeddings[:, :k]
        return embeddings.astype(np.float32, copy=False)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity using sentence-transformers util (torch-based)."""
        if util is None or torch is None:
            # Fallback to numpy cosine
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
            return a_norm @ b_norm.T
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        if self.device:
            a_t = a_t.to(self.device)
            b_t = b_t.to(self.device)
        with torch.no_grad():
            # Prefer the model's configured similarity if available (ST v3+)
            if hasattr(self.model, "similarity") and callable(getattr(self.model, "similarity")):
                sims = self.model.similarity(a_t, b_t)  # type: ignore[attr-defined]
            else:
                sims = util.cos_sim(a_t, b_t)  # [A, B]
        return sims.detach().cpu().numpy()


