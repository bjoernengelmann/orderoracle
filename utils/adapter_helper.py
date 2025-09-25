from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

def ranking_matrix_to_topk_df(
    ranking_matrix: Union["torch.Tensor", np.ndarray],
    k: int = 100,
    topic_ids: Optional[Sequence[Union[int, str]]] = None,
    doc_ids: Optional[Sequence[Union[int, str]]] = None,
    qid_start_at_one: bool = False,
) -> pd.DataFrame:
    """
    Convert a topic-by-document score matrix into a long DataFrame containing top-k results per topic.

    Parameters
    ----------
    ranking_matrix : torch.Tensor | np.ndarray
        A 2D matrix of shape (num_topics, num_docs) with similarity scores.
        Each row corresponds to a topic (qid), each column to a document.
    k : int, default 100
        Number of top documents to return per topic.
    topic_ids : Sequence[int | str] | None, optional
        Optional identifiers to use for qid. If not provided, uses row indices 0..num_topics-1.
    doc_ids : Sequence[int | str] | None, optional
        Optional identifiers to use for docno. If not provided, uses 1..num_docs.
    qid_start_at_one : bool, default False
        If True and topic_ids are integers or defaulted, qid will start at 1 instead of 0.

    Returns
    -------
    pd.DataFrame
        Columns: ["qid", "docno", "score", "rank"]. For each topic, the top-k documents
        are listed in descending score order with rank starting at 0.
    """
    if ranking_matrix is None:
        raise ValueError("ranking_matrix is required")

    # Torch path if available or tensor provided
    if _HAS_TORCH and isinstance(ranking_matrix, torch.Tensor):
        scores_t = ranking_matrix
    else:
        arr = np.asarray(ranking_matrix)
        if arr.ndim != 2:
            raise ValueError("ranking_matrix must be a 2D array/tensor")
        if not _HAS_TORCH:
            # Pure NumPy implementation
            n_topics, n_docs = arr.shape
            k_eff = int(min(int(k), n_docs))

            # Topic IDs
            if topic_ids is None:
                topic_ids_seq = list(range(n_topics))
            else:
                if len(topic_ids) != n_topics:
                    raise ValueError("topic_ids length must match number of rows in ranking_matrix")
                topic_ids_seq = list(topic_ids)
            if qid_start_at_one:
                topic_ids_seq = [tid + 1 if isinstance(tid, int) else tid for tid in topic_ids_seq]

            # Doc IDs
            if doc_ids is None:
                doc_ids_arr = np.arange(1, n_docs + 1, dtype=object)
            else:
                if len(doc_ids) != n_docs:
                    raise ValueError("doc_ids length must match number of columns in ranking_matrix")
                doc_ids_arr = np.asarray(doc_ids, dtype=object)

            # Argpartition for top-k per row, then stable sort by score desc, tie-break by doc index asc
            top_idx_rows = []
            top_val_rows = []
            for row in arr:
                idx = np.argpartition(row, -k_eff)[-k_eff:]
                # Sort: primary by score desc, secondary by doc index asc
                order = np.lexsort((idx, -row[idx]))
                idx_sorted = idx[order][::-1]
                top_idx_rows.append(idx_sorted)
                top_val_rows.append(row[idx_sorted])

            idx_matrix = np.stack(top_idx_rows, axis=0)
            val_matrix = np.stack(top_val_rows, axis=0)

            qid_col = np.repeat(np.asarray(topic_ids_seq, dtype=object), k_eff)
            docno_col = doc_ids_arr[idx_matrix.reshape(-1)]
            score_col = val_matrix.reshape(-1)
            # 0-based ranks
            rank_col = np.tile(np.arange(0, k_eff), n_topics)

            return pd.DataFrame({
                "qid": qid_col,
                "docno": docno_col,
                "score": score_col,
                "rank": rank_col,
            })
        else:
            scores_t = torch.as_tensor(arr)

    if scores_t.ndim != 2:
        raise ValueError("ranking_matrix must be a 2D array/tensor")

    n_topics, n_docs = int(scores_t.shape[0]), int(scores_t.shape[1])
    k_eff = int(min(int(k), n_docs))

    # Topic IDs
    if topic_ids is None:
        topic_ids_seq = list(range(n_topics))
    else:
        if len(topic_ids) != n_topics:
            raise ValueError("topic_ids length must match number of rows in ranking_matrix")
        topic_ids_seq = list(topic_ids)
    if qid_start_at_one:
        topic_ids_seq = [tid + 1 if isinstance(tid, int) else tid for tid in topic_ids_seq]

    # Doc IDs
    if doc_ids is None:
        doc_ids_arr = np.arange(1, n_docs + 1, dtype=object)
    else:
        if len(doc_ids) != n_docs:
            raise ValueError("doc_ids length must match number of columns in ranking_matrix")
        doc_ids_arr = np.asarray(doc_ids, dtype=object)

    # Efficient top-k using torch
    values, indices = torch.topk(scores_t, k=k_eff, dim=1, largest=True, sorted=True)
    values = values.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    qid_col = np.repeat(np.asarray(topic_ids_seq, dtype=object), k_eff)
    docno_col = doc_ids_arr[indices.reshape(-1)]
    score_col = values.reshape(-1)
    # 0-based ranks
    rank_col = np.tile(np.arange(0, k_eff), n_topics)

    res = pd.DataFrame({
        "qid": qid_col,
        "docno": docno_col,
        "score": score_col,
        "rank": rank_col,
    })

    res["qid"] = res["qid"].astype(str)
    res["docno"] = res["docno"].astype(str)
    res["score"] = res["score"].astype(float)
    return res

def truncate_embeddings_matryoshka(
    embeddings: Union["torch.Tensor", np.ndarray],
    truncate_to_dims: int,
) -> Union["torch.Tensor", np.ndarray]:
    from ...helper import truncate_embeddings_matryoshka as _orig  # type: ignore
    return _orig(embeddings, truncate_to_dims)


