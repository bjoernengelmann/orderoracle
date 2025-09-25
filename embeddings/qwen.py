from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional heavy deps
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from .base import EmbeddingBackend


class QwenMatryoshka(EmbeddingBackend):
    def __init__(
        self,
        model_name: str = "Qwen2-7B-Embedding",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
        dtype: str = "float16",
    ) -> None:
        if AutoModel is None:
            raise RuntimeError("transformers not installed. Install with extras: pip install orderoracle[hf]")

        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = int(batch_size)
        self.torch_dtype = getattr(torch, dtype) if torch is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        self.model.to(self.device)
        self.model.eval()

    def _to_numpy(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    def embed_texts(self, texts: Iterable[str], max_dim: Optional[int] = None) -> np.ndarray:
        texts_list = list(texts)
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i : i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state mean pooling as a placeholder; adjust for real Qwen embed head
                hidden = outputs.last_hidden_state  # [B, T, H]
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                emb = summed / denom
            all_vecs.append(self._to_numpy(emb))

        full = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 0), dtype=np.float32)
        if max_dim is not None:
            k = int(max_dim)
            if full.shape[1] < k:
                raise ValueError(f"Requested dims {k} > model dim {full.shape[1]}")
            return full[:, :k]
        return full


