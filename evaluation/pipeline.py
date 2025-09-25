from __future__ import annotations

from typing import Dict, Sequence, List, Mapping, Tuple

from ..data.models import Collection
from ..metrics.correlation import kendalls_tau, kendalls_tau_with_p, rbo_overlap
import pandas as pd
import numpy as np
import logging


class CollectionQualityEvaluator:
    def __init__(self, backend, dims_schedule: Sequence[int]) -> None:
        self.backend = backend
        self.dims_schedule = list(dims_schedule)
        self._logger = logging.getLogger(__name__)

    # --- PyTerrier-backed variant (preferred when PyTerrier is available) ---
    def empirical_scores_from_embeddings_pt(self, collection: Collection, metric: str = "ndcg", k: int = 10) -> Dict[int, float]:
        try:
            import pyterrier as pt  # type: ignore
            from pyterrier import measures as MS  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyterrier is required for PT-based evaluation. pip install python-terrier") from e

        from ..utils.adapter_helper import ranking_matrix_to_topk_df
        topics_map = collection.topics
        docs = list(collection.documents.values())
        topic_texts = list(topics_map.values())
        doc_texts = [d.text for d in docs]
        topic_ids = [str(qid) for qid in topics_map.keys()]
        doc_ids = [str(d.doc_id) for d in docs]
        topics_df = pd.DataFrame({"qid": topic_ids, "query": topic_texts})
        qrels_df = pd.DataFrame({
            "qid": [str(q.topic_id) for q in collection.qrels],
            "docno": [str(q.doc_id) for q in collection.qrels],
            "label": [int(q.relevance) for q in collection.qrels],
        })
        self._logger.info("PT evaluation setup: topics=%d docs=%d dims_schedule=%s metric=%s k=%d", len(topics_map), len(docs), self.dims_schedule, metric, k)
        # Select official PT measure objects (back-compat for single metric)
        if metric.lower() == "ndcg":
            measure = pt.measures.nDCG @ k  # e.g., pt.measures.nDCG@10
        elif metric.lower() == "map":
            measure = pt.measures.AP
        else:
            # Try to parse via ir_measures (e.g., "P@10", "nDCG@10", "R@10")
            try:
                import ir_measures
                from ir_measures.util import parse_measure
                m = parse_measure(metric)
                measure = m
            except Exception as e:
                raise ValueError(f"Unsupported metric for PT: {metric}") from e
        measure_label = str(measure)
        results: Dict[int, float] = {}
        transformers_or_dfs = []
        names = []
        for dims in self.dims_schedule:
            if collection.topics_embeddings is not None and collection.docs_embeddings is not None:
                topics_emb = collection.topics_embeddings[:, :dims]
                docs_emb = collection.docs_embeddings[:, :dims]
            else:
                topics_emb = self.backend.embed_texts(topic_texts, max_dim=dims)
                docs_emb = self.backend.embed_texts(doc_texts, max_dim=dims)
            self._logger.debug("dims=%d topics_emb=%s docs_emb=%s", dims, tuple(topics_emb.shape), tuple(docs_emb.shape))
            if hasattr(self.backend, "similarity"):
                sim = getattr(self.backend, "similarity")(topics_emb, docs_emb)
            else:
                a = topics_emb / (np.linalg.norm(topics_emb, axis=1, keepdims=True) + 1e-12)
                b = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)
                sim = a @ b.T
            self._logger.debug("dims=%d similarity_matrix=%s", dims, tuple(sim.shape))
            df = ranking_matrix_to_topk_df(sim, k=k, topic_ids=topic_ids, doc_ids=doc_ids, qid_start_at_one=False)
            transformers_or_dfs.append(df)
            names.append(f"dims_{dims}")
        # Run a PT experiment
        exp = pt.Experiment(
            transformers_or_dfs,
            topics_df,
            qrels_df,
            [measure],
            names,
            
        )
        # Extract measure results
        for i, dims in enumerate(self.dims_schedule):
            score_val = float(exp.iloc[i][measure_label])  # type: ignore[index]
            results[dims] = score_val
            self._logger.info("dims=%d %s=%.6f", dims, measure_label, score_val)
        return results

        
    def quality_from_embeddings_pt(self, collection: Collection, metric: str = "ndcg", k: int = 10) -> float:
        self._logger.info("Computing Kendall tau from PT scores (metric=%s, k=%d)", metric, k)
        tau, _ = self.quality_with_p_from_embeddings_pt(collection, metric=metric, k=k)
        return tau

    def quality_with_p_from_embeddings_pt(self, collection: Collection, metric: str = "ndcg", k: int = 10) -> Tuple[float, float]:
        self._logger.info("Computing Kendall tau (with p) from PT scores (metric=%s, k=%d)", metric, k)
        scores = self.empirical_scores_from_embeddings_pt(collection, metric=metric, k=k)
        apriori = {str(d): i for i, d in enumerate(self.dims_schedule)}
        empirical = {str(d): s for d, s in scores.items()}
        tau, p = kendalls_tau_with_p(apriori, empirical)
        self._logger.info("Kendall tau=%.6f p=%.6g", tau, p)
        return float(tau), float(p)

    # ---- Multi-measure and multi-correlation support ----
    def empirical_scores_multi_pt(
        self,
        collection: Collection,
        measures: List[str],
        default_k: int = 10,
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute empirical IR scores per dims for each requested measure string.

        measures accepts strings like: "AP", "P@10", "nDCG@10", "R@10".
        """
        try:
            import pyterrier as pt  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyterrier is required for PT-based evaluation. pip install python-terrier") from e

        from ..utils.adapter_helper import ranking_matrix_to_topk_df
        try:
            from ir_measures.util import parse_measure
        except Exception as e:  # pragma: no cover
            raise RuntimeError("ir_measures is required for measure parsing. pip install ir-measures") from e

        topics_map = collection.topics
        docs = list(collection.documents.values())
        topic_texts = list(topics_map.values())
        doc_texts = [d.text for d in docs]
        topic_ids = [str(qid) for qid in topics_map.keys()]
        doc_ids = [str(d.doc_id) for d in docs]
        topics_df = pd.DataFrame({"qid": topic_ids, "query": topic_texts})
        qrels_df = pd.DataFrame({
            "qid": [str(q.topic_id) for q in collection.qrels],
            "docno": [str(q.doc_id) for q in collection.qrels],
            "label": [int(q.relevance) for q in collection.qrels],
        })
        self._logger.info(
            "PT multi-measure setup: topics=%d docs=%d dims_schedule=%s measures=%s",
            len(topics_map), len(docs), self.dims_schedule, measures,
        )

        # Parse measures; if a measure lacks cutoff (e.g., "nDCG"), use as-is.
        parsed_measures = []
        for m in measures:
            try:
                parsed_measures.append(parse_measure(m))
            except Exception:
                # Fallback friendly names
                low = m.lower()
                if low == "ndcg":
                    parsed_measures.append(pt.measures.nDCG @ default_k)  # type: ignore[attr-defined]
                elif low == "map":
                    parsed_measures.append(pt.measures.AP)  # type: ignore[attr-defined]
                else:
                    raise
        measure_labels = [str(m) for m in parsed_measures]

        transformers_or_dfs = []
        names = []
        for dims in self.dims_schedule:
            if collection.topics_embeddings is not None and collection.docs_embeddings is not None:
                topics_emb = collection.topics_embeddings[:, :dims]
                docs_emb = collection.docs_embeddings[:, :dims]
            else:
                topics_emb = self.backend.embed_texts(topic_texts, max_dim=dims)
                docs_emb = self.backend.embed_texts(doc_texts, max_dim=dims)
            self._logger.debug("dims=%d topics_emb=%s docs_emb=%s", dims, tuple(topics_emb.shape), tuple(docs_emb.shape))
            if hasattr(self.backend, "similarity"):
                sim = getattr(self.backend, "similarity")(topics_emb, docs_emb)
            else:
                a = topics_emb / (np.linalg.norm(topics_emb, axis=1, keepdims=True) + 1e-12)
                b = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)
                sim = a @ b.T
            self._logger.debug("dims=%d similarity_matrix=%s", dims, tuple(sim.shape))
            df = ranking_matrix_to_topk_df(sim, k=default_k, topic_ids=topic_ids, doc_ids=doc_ids, qid_start_at_one=False)
            transformers_or_dfs.append(df)
            names.append(f"dims_{dims}")

        exp = pt.Experiment(
            transformers_or_dfs,
            topics_df,
            qrels_df,
            parsed_measures,
            names,
        )

        results: Dict[str, Dict[int, float]] = {lbl: {} for lbl in measure_labels}
        for i, dims in enumerate(self.dims_schedule):
            row = exp.iloc[i]
            for lbl in measure_labels:
                try:
                    results[lbl][dims] = float(row[lbl])  # type: ignore[index]
                except Exception:
                    results[lbl][dims] = float("nan")
            self._logger.info("dims=%d scores=%s", dims, {lbl: results[lbl][dims] for lbl in measure_labels})
        return results

    def correlations_from_embeddings_pt(
        self,
        collection: Collection,
        measures: List[str],
        default_k: int = 10,
        correlations: List[str] | None = None,
        rbo_p: float = 0.9,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-measure correlation(s) between dims ordering and measure-based ordering.

        Returns a dict with keys:
          - "per_measure": mapping of measure label -> {corr_name: value}
          - "average": mapping of corr_name -> average over measures (ignoring NaNs)
        """
        corr_names = [c.lower() for c in (correlations or ["kendall"])]
        scores = self.empirical_scores_multi_pt(collection, measures, default_k)
        apriori_rank = [str(d) for d in self.dims_schedule]

        per_measure: Dict[str, Dict[str, float]] = {}
        # Compute per-measure correlations
        for mlabel, dims_scores in scores.items():
            # Kendall tau between apriori index and score values
            kendall_val = float("nan")
            kendall_p = float("nan")
            if "kendall" in corr_names or "kendall_tau" in corr_names:
                apriori_map = {str(d): idx for idx, d in enumerate(self.dims_schedule)}
                empirical_map = {str(d): s for d, s in dims_scores.items()}
                kendall_val, kendall_p = kendalls_tau_with_p(apriori_map, empirical_map)

            # RBO between apriori list and dims sorted by score desc
            rbo_val = float("nan")
            if "rbo" in corr_names:
                sorted_dims = sorted(dims_scores.items(), key=lambda x: ((x[1] if np.isfinite(x[1]) else -np.inf), x[0]))
                empirical_rank = [str(d) for d, _ in sorted_dims]
                
                rbo_val = rbo_overlap(apriori_rank, empirical_rank, p=rbo_p)

            vals: Dict[str, float] = {}
            if "kendall" in corr_names or "kendall_tau" in corr_names:
                vals["kendall_tau"] = kendall_val
                vals["kendall_tau_p"] = kendall_p
            if "rbo" in corr_names:
                vals["rbo"] = rbo_val
            per_measure[mlabel] = vals

        # Averages across measures
        avg: Dict[str, float] = {}
        if per_measure:
            for cname in ("kendall_tau", "rbo"):
                if cname in (per_measure[next(iter(per_measure))].keys()):
                    arr = np.array([v.get(cname, np.nan) for v in per_measure.values()], dtype=float)
                    if np.isfinite(arr).any():
                        avg[cname] = float(np.nanmean(arr))
                    else:
                        avg[cname] = float("nan")

        return {"per_measure": per_measure, "average": avg}


