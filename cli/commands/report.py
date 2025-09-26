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
    @app.command("report")
    def report_cmd(
        docs_path: Path = typer.Option(...),
        topics_path: Path = typer.Option(...),
        qrels_path: Path = typer.Option(...),
        emb_path: Path = typer.Option(..., help="Path to precomputed embeddings (.npz or directory)"),
        dims: Optional[List[int]] = typer.Option(None, help="Explicit dims schedule", show_default=False),
        dims_pow2: Optional[str] = typer.Option(None, help="Generate dims as powers-of-two range, e.g., '16-512'"),
        model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B"),
        output_csv: Optional[Path] = typer.Option(None),
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
            tau, p = evaluator.quality_with_p_from_embeddings_pt(coll, metric=metric, k=k)
            if output_csv:
                with output_csv.open("w", newline="", encoding="utf-8") as f:
                    import csv
                    w = csv.writer(f)
                    w.writerow(["measure", "kendall_tau", "kendall_tau_p"])
                    w.writerow([metric, tau, p])
                logger.info("Saved report CSV to %s", str(output_csv))
                print(f"[green]Saved report to {output_csv}[/]")
            else:
                print(f"[bold]kendall_tau={tau:.6f}[/] [dim]kendall_tau_p={p:.6g}[/]")
            return

        corr = evaluator.correlations_from_embeddings_pt(
            coll,
            measures=measures_list,
            default_k=k,
            correlations=correlations_list,
            rbo_p=rbo_p,
        )
        per = corr.get("per_measure", {})
        avg = corr.get("average", {})
        if output_csv:
            with output_csv.open("w", newline="", encoding="utf-8") as f:
                import csv
                header = ["measure"]
                corr_keys = set()
                for vals in per.values():
                    corr_keys.update(vals.keys())
                corr_keys = list(sorted(corr_keys))
                header.extend(corr_keys)
                w = csv.writer(f)
                w.writerow(header)
                for m, vals in per.items():
                    row = [m] + [vals.get(k, float("nan")) for k in corr_keys]
                    w.writerow(row)
                if avg:
                    row = ["AVERAGE"] + [avg.get(k, float("nan")) for k in corr_keys]
                    w.writerow(row)
            logger.info("Saved report CSV to %s", str(output_csv))
            print(f"[green]Saved report to {output_csv}[/]")
        else:
            for m, vals in per.items():
                parts = [f"{k}={float(v):.6f}" for k, v in vals.items()]
                print(f"[cyan]measure={m}[/] " + " ".join(parts))
            if avg:
                parts = [f"{k}={float(v):.6f}" for k, v in avg.items()]
                print("[magenta]average[/] " + " ".join(parts))


