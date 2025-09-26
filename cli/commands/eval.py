from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

import typer
from rich import print

from ... import SentenceTransformerBackend, load_collection_with_embeddings
from ...evaluation.pipeline import CollectionQualityEvaluator
from ..helpers import setup_logging, parse_dims_pow2, parse_list_option


logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("eval")
    def eval_cmd(
        docs_path: Path = typer.Option(...),
        topics_path: Path = typer.Option(...),
        qrels_path: Path = typer.Option(...),
        emb_path: Path = typer.Option(..., help="Path to precomputed embeddings (.npz or directory)"),
        dims: Optional[List[int]] = typer.Option(None, help="Explicit dims schedule", show_default=False),
        dims_pow2: Optional[str] = typer.Option(None, help="Generate dims as powers-of-two range, e.g., '16-512'"),
        model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B"),
        metric: str = typer.Option("ndcg", help="Single metric (legacy) if --measures not set"),
        measures: Optional[List[str]] = typer.Option(None, help="Repeatable: --measures AP --measures P@10 ..."),
        measures_csv: Optional[str] = typer.Option(None, help="Comma/space separated measures: e.g., 'AP,P@10,nDCG@10,R@10'"),
        correlations: List[str] = typer.Option(["kendall"], help="Repeatable: --correlations kendall --correlations rbo"),
        correlations_csv: Optional[str] = typer.Option(None, help="Comma/space separated correlations: e.g., 'kendall,rbo'"),
        rbo_p: float = typer.Option(0.9, help="RBO persistence parameter p"),
        k: int = typer.Option(10),
        device: Optional[str] = typer.Option(None, help="Device: cpu | mps | gpu | cuda[:N]"),
    ) -> None:
        setup_logging()
        dims_schedule = dims if dims is not None else parse_dims_pow2(dims_pow2)
        if not dims_schedule:
            raise typer.BadParameter("Provide either --dims or --dims-pow2 (e.g., 16-512)")
        backend = SentenceTransformerBackend(model_name=model_name, device=device)
        coll = load_collection_with_embeddings(docs_path, topics_path, qrels_path, emb_path)
        evaluator = CollectionQualityEvaluator(backend, dims_schedule=dims_schedule)

        measures_list = parse_list_option(measures, measures_csv)
        correlations_list = parse_list_option(correlations, correlations_csv) or ["kendall"]

        if measures_list is None or len(measures_list) == 0:
            logger.info(
                "Evaluating (legacy) dims=%s metric=%s k=%d model=%s device=%s",
                dims_schedule, metric, k, model_name, device,
            )
            tau, p = evaluator.quality_with_p_from_embeddings_pt(coll, metric=metric, k=k)
            logger.info("Computed Kendall tau: %.6f p=%.6g", tau, p)
            print(f"[bold]kendall_tau={tau:.6f}[/] [dim]kendall_tau_p={p:.6g}[/]")
            return

        logger.info(
            "Evaluating multi-measure: dims=%s measures=%s correlations=%s k=%d model=%s device=%s rbo_p=%.3f",
            dims_schedule, measures_list, correlations_list, k, model_name, device, rbo_p,
        )
        corr = evaluator.correlations_from_embeddings_pt(
            coll,
            measures=measures_list,
            default_k=k,
            correlations=correlations_list,
            rbo_p=rbo_p,
        )
        per = corr.get("per_measure", {})
        avg = corr.get("average", {})
        for m, vals in per.items():
            parts = [f"{k}={v:.6f}" for k, v in vals.items() if isinstance(v, float)]
            print(f"[cyan]measure={m}[/] " + " ".join(parts))
        if avg:
            parts = [f"{k}={v:.6f}" for k, v in avg.items() if isinstance(v, float)]
            print("[magenta]average[/] " + " ".join(parts))


