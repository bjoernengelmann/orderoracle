from __future__ import annotations

from typing import Dict, Iterable, Tuple, Sequence, List, Tuple as _Tuple

import numpy as np
from scipy.stats import kendalltau


def kendalls_tau_with_p(rank_a: Dict[str, float], rank_b: Dict[str, float]) -> _Tuple[float, float]:
    """
    Tie-aware Kendall's tau and p-value between two system score maps. Systems missing
    in either map are dropped. NaN scores are ignored.
    """
    keys = [k for k in rank_a.keys() if k in rank_b]
    if not keys:
        return float("nan"), float("nan")
    a = np.array([rank_a[k] for k in keys], dtype=float)
    b = np.array([rank_b[k] for k in keys], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan"), float("nan")
    tau, p = kendalltau(a[mask], b[mask])
    return float(tau), float(p)


def kendalls_tau(rank_a: Dict[str, float], rank_b: Dict[str, float]) -> float:
    """
    Backwards-compatible wrapper returning only the tau value.
    """
    tau, _p = kendalls_tau_with_p(rank_a, rank_b)
    return float(tau)



def rbo_overlap(
    rank_list_a: Sequence[str],
    rank_list_b: Sequence[str],
    p: float = 0.9,
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists using the
    optional `rbo` package. Returns NaN if lists are empty or disjoint.

    Parameters
    ----------
    rank_list_a : Sequence[str]
        Ranked list A (from top to bottom).
    rank_list_b : Sequence[str]
        Ranked list B (from top to bottom).
    p : float, default 0.9
        RBO persistence parameter (higher emphasizes top ranks more).
    """
    try:
        import rbo as rbo_lib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'rbo' package is required for RBO correlation. Install with: pip install rbo"
        ) from e

    if not rank_list_a or not rank_list_b:
        return float("nan")
    try:
        rs = rbo_lib.RankingSimilarity(list(rank_list_a), list(rank_list_b))
        val = rs.rbo(p=p)
        return float(val)
    except Exception:
        return float("nan")


