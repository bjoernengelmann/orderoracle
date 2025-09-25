from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging
import os

import typer
from rich import print

from .. import load_collection, load_collection_with_embeddings, SentenceTransformerBackend
from ..data.io import load_documents, load_topics
from ..evaluation.pipeline import CollectionQualityEvaluator
from ..ranking.rankers import SimilarityRanker


app = typer.Typer(no_args_is_help=True, add_completion=False)


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure basic logging for CLI if not already configured.

    Honors ORDERORACLE_LOG_LEVEL env var (default INFO).
    """
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("ORDERORACLE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _parse_dims_pow2(spec: Optional[str]) -> Optional[List[int]]:
    """Parse a simple powers-of-two spec into a dims schedule.

    Accepted forms:
    - "start-end" or "start:end" → generate [start, start*2, ..., end]
    - "N" (single int) → [N]

    Constraints: start and end must be positive integers, start <= end, and
    each generated value is a power of two progression via doubling.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None
    if "-" in s or ":" in s:
        sep = "-" if "-" in s else ":"
        left, right = s.split(sep, 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            raise typer.BadParameter("--dims-pow2 must be in the form START-END, e.g., 16-512")
        if start <= 0 or end <= 0:
            raise typer.BadParameter("--dims-pow2 values must be positive")
        if start > end:
            raise typer.BadParameter("--dims-pow2 start must be <= end")
        dims: List[int] = []
        val = start
        while val <= end:
            dims.append(val)
            val *= 2
        return dims
    else:
        # Single integer
        try:
            single = int(s)
        except ValueError:
            raise typer.BadParameter("--dims-pow2 must be an int or range like 16-512")
        if single <= 0:
            raise typer.BadParameter("--dims-pow2 must be positive")
        return [single]


def _parse_list_option(values: Optional[List[str]], csv: Optional[str]) -> Optional[List[str]]:
    """Merge a repeatable option list with an optional comma/space-separated string.

    Returns None if no values present.
    """
    merged: List[str] = []
    if values:
        merged.extend(values)
    if csv:
        # Split on commas and whitespace
        for token in csv.replace(",", " ").split():
            merged.append(token)
    return merged or None

@app.command("pt-export-docs")
def pt_export_docs_cmd(
    dataset: str = typer.Option(..., help="PyTerrier dataset name, e.g., 'irds:vaswani'"),
    output_path: Path = typer.Option(..., help="Output JSONL path with fields {id, text}"),
    text_field: Optional[str] = typer.Option(None, help="Field name to use for text; auto-detect if omitted"),
    limit: Optional[int] = typer.Option(None, help="Optional max number of docs to export"),
):
    """Export documents from a PyTerrier dataset into package-compatible JSONL (id, text)."""
    _setup_logging()
    try:
        import pyterrier as pt  # type: ignore
    except Exception as e:
        raise typer.BadParameter(
            "pyterrier is required for this command. Install with: pip install python-terrier"
        ) from e

    # Initialize PyTerrier without noisy logging
    try:
        pt.init(boot_packages=[], initialise_logging=False, logging=False)  # type: ignore
    except Exception:
        pt.init()

    logger.info("Initializing PyTerrier dataset: %s", dataset)
    ds = pt.get_dataset(dataset)

    def _iter_records():
        if hasattr(ds, "get_corpus_iter"):
            yield from ds.get_corpus_iter()  # type: ignore[attr-defined]
            return
        if hasattr(ds, "get_corpus"):
            corpus = ds.get_corpus()  # DataFrame for many datasets
            # If DataFrame-like
            if hasattr(corpus, "iterrows"):
                for _, row in corpus.iterrows():
                    yield row.to_dict()
                return
            # If list-like of dicts
            if isinstance(corpus, list):
                for rec in corpus:
                    yield rec
                return
        raise RuntimeError("Dataset does not expose a corpus iterator or corpus table")

    # Auto-detect text field if not specified
    preferred_fields = ("text", "body", "contents", "raw")
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for rec in _iter_records():
            # Normalize dict-like rows
            if not isinstance(rec, dict):
                try:
                    rec = dict(rec)
                except Exception:
                    continue

            doc_id = rec.get("docno") or rec.get("id") or rec.get("doc_id")
            if doc_id is None:
                continue
            if text_field is not None:
                text_val = rec.get(text_field)
            else:
                text_val = None
                for f in preferred_fields:
                    if f in rec and rec[f] not in (None, ""):
                        text_val = rec[f]
                        break
            if text_val is None:
                # Last resort: join all string-like fields
                text_val = " ".join(str(v) for k, v in rec.items() if isinstance(v, str))
            obj = {"id": str(doc_id), "text": str(text_val)}
            out.write(__import__("json").dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= int(limit):
                break
    logger.info("Wrote %d documents to %s", written, str(output_path))
    print(f"Wrote {written} documents to {output_path}")

@app.command("embed")
def embed_cmd(
    model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B", help="Sentence Transformers model name"),
    docs_path: Path = typer.Option(..., help="Path to documents JSONL with 'id' and 'text'"),
    topics_path: Path = typer.Option(..., help="Path to topics JSON (vaswani-style or records)"),
    output_path: Path = typer.Option(..., help="Output directory, base path, or .npz file for embeddings"),
    device: Optional[str] = typer.Option(None, help="Device: cpu | mps | gpu | cuda[:N]"),
):
    """Embed documents and topics and save in a format load_collection_with_embeddings expects.

    If output_path ends with .npz, saves a single NPZ with keys 'topics' and 'docs'.
    If output_path is a directory, saves 'topics.npy' and 'docs.npy' inside it.
    Otherwise, treats output_path as a base stem and saves '<stem>_topics.npy' and '<stem>_docs.npy'.
    """
    _setup_logging()
    logger.info("Starting embedding with model=%s device=%s", model_name, device)
    import numpy as np

    # Determine targets and skip if already present
    if output_path.suffix.lower() == ".npz":
        existing_topics = None
        existing_docs = None
        have_topics = False
        have_docs = False
        if output_path.exists():
            try:
                data = np.load(output_path)
                if "topics" in data:
                    existing_topics = data["topics"]
                    have_topics = True
                if "docs" in data:
                    existing_docs = data["docs"]
                    have_docs = True
            except Exception as e:
                logger.warning("Failed to read existing NPZ at %s: %s; will recompute both", str(output_path), e)
                have_topics = False
                have_docs = False

        if have_topics and have_docs:
            logger.info("Embeddings NPZ already has both arrays at %s; skipping", str(output_path))
            print(f"Embeddings already exist at {output_path}; skipping")
            return

        backend = SentenceTransformerBackend(model_name=model_name, device=device)

        # Compute only what's missing, then (re)write NPZ with both arrays
        if not have_docs:
            docs = load_documents(docs_path)
            doc_texts = [d.text for d in docs.values()]
            logger.info("Loaded %d docs for embedding (NPZ)", len(doc_texts))
            docs_emb = backend.embed_texts(doc_texts, max_dim=None)
            logger.info("Computed docs embeddings: shape=%s", tuple(docs_emb.shape))
        else:
            docs_emb = existing_docs  # type: ignore[assignment]

        if not have_topics:
            topics = load_topics(topics_path)
            topic_texts = list(topics.values())
            logger.info("Loaded %d topics for embedding (NPZ)", len(topic_texts))
            topics_emb = backend.embed_texts(topic_texts, max_dim=None)
            logger.info("Computed topics embeddings: shape=%s", tuple(topics_emb.shape))
        else:
            topics_emb = existing_topics  # type: ignore[assignment]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, topics=topics_emb, docs=docs_emb)
        logger.info("Saved NPZ embeddings to %s (keys: topics, docs)", str(output_path))
        print(f"Saved NPZ embeddings to {output_path}")
        return
    else:
        # Directory or base stem; support partial creation and skipping
        if output_path.is_dir():
            out_dir = output_path
            topics_file = out_dir / "topics.npy"
            docs_file = out_dir / "docs.npy"
        else:
            out_dir = output_path.parent
            topics_file = out_dir / f"{output_path.name}_topics.npy"
            docs_file = out_dir / f"{output_path.name}_docs.npy"

        topics_exists = topics_file.exists()
        docs_exists = docs_file.exists()

        if topics_exists and docs_exists:
            logger.info("Embeddings already exist: topics=%s docs=%s; skipping encoding", str(topics_file), str(docs_file))
            print(f"Embeddings already exist at {topics_file} and {docs_file}; skipping")
            return

        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        backend = SentenceTransformerBackend(model_name=model_name, device=device)

        if not docs_exists:
            docs = load_documents(docs_path)
            doc_texts = [d.text for d in docs.values()]
            logger.info("Loaded %d docs for embedding", len(doc_texts))
            docs_emb = backend.embed_texts(doc_texts, max_dim=None)
            logger.info("Computed docs embeddings: shape=%s", tuple(docs_emb.shape))
            np.save(docs_file, docs_emb)
            logger.info("Saved docs embeddings to %s", str(docs_file))
            print(f"Saved docs embeddings to {docs_file}")

        if not topics_exists:
            topics = load_topics(topics_path)
            topic_texts = list(topics.values())
            logger.info("Loaded %d topics for embedding", len(topic_texts))
            topics_emb = backend.embed_texts(topic_texts, max_dim=None)
            logger.info("Computed topics embeddings: shape=%s", tuple(topics_emb.shape))
            np.save(topics_file, topics_emb)
            logger.info("Saved topics embeddings to %s", str(topics_file))
            print(f"Saved topics embeddings to {topics_file}")


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
):
    _setup_logging()
    dims_schedule = dims if dims is not None else _parse_dims_pow2(dims_pow2)
    if not dims_schedule:
        raise typer.BadParameter("Provide either --dims or --dims-pow2 (e.g., 16-512)")
    backend = SentenceTransformerBackend(model_name=model_name, device=device)
    coll = load_collection_with_embeddings(docs_path, topics_path, qrels_path, emb_path)
    evaluator = CollectionQualityEvaluator(backend, dims_schedule=dims_schedule)

    measures = _parse_list_option(measures, measures_csv)
    correlations = _parse_list_option(correlations, correlations_csv) or ["kendall"]

    if measures is None or len(measures) == 0:
        # Legacy single-metric Kendall tau
        logger.info(
            "Evaluating (legacy) dims=%s metric=%s k=%d model=%s device=%s",
            dims_schedule, metric, k, model_name, device,
        )
        tau, p = evaluator.quality_with_p_from_embeddings_pt(coll, metric=metric, k=k)
        logger.info("Computed Kendall tau: %.6f p=%.6g", tau, p)
        print(f"kendall_tau={tau:.6f} kendall_tau_p={p:.6g}")
        return

    logger.info(
        "Evaluating multi-measure: dims=%s measures=%s correlations=%s k=%d model=%s device=%s rbo_p=%.3f",
        dims_schedule, measures, correlations, k, model_name, device, rbo_p,
    )
    corr = evaluator.correlations_from_embeddings_pt(
        coll,
        measures=measures,
        default_k=k,
        correlations=correlations,
        rbo_p=rbo_p,
    )
    per = corr.get("per_measure", {})
    avg = corr.get("average", {})
    # Human-readable output
    for m, vals in per.items():
        parts = [f"{k}={v:.6f}" for k, v in vals.items() if isinstance(v, float)]
        print(f"measure={m} " + " ".join(parts))
    if avg:
        parts = [f"{k}={v:.6f}" for k, v in avg.items() if isinstance(v, float)]
        print("average " + " ".join(parts))


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
):
    _setup_logging()
    dims_schedule = dims if dims is not None else _parse_dims_pow2(dims_pow2)
    if not dims_schedule:
        raise typer.BadParameter("Provide either --dims or --dims-pow2 (e.g., 16-512)")
    backend = SentenceTransformerBackend(model_name=model_name, device=device)
    coll = load_collection_with_embeddings(docs_path, topics_path, qrels_path, emb_path)
    evaluator = CollectionQualityEvaluator(backend, dims_schedule=dims_schedule)

    measures = _parse_list_option(measures, measures_csv)
    correlations = _parse_list_option(correlations, correlations_csv) or ["kendall"]

    if measures is None or len(measures) == 0:
        tau, p = evaluator.quality_with_p_from_embeddings_pt(coll, metric=metric, k=k)
        if output_csv:
            with output_csv.open("w", newline="", encoding="utf-8") as f:
                import csv
                w = csv.writer(f)
                w.writerow(["measure", "kendall_tau", "kendall_tau_p"])
                w.writerow([metric, tau, p])
            logger.info("Saved report CSV to %s", str(output_csv))
            print(f"Saved report to {output_csv}")
        else:
            print(f"kendall_tau={tau:.6f} kendall_tau_p={p:.6g}")
        return

    corr = evaluator.correlations_from_embeddings_pt(
        coll,
        measures=measures,
        default_k=k,
        correlations=correlations,
        rbo_p=rbo_p,
    )
    per = corr.get("per_measure", {})
    avg = corr.get("average", {})
    if output_csv:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            import csv
            # Determine header dynamically from correlations chosen
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
        print(f"Saved report to {output_csv}")
    else:
        for m, vals in per.items():
            parts = [f"{k}={float(v):.6f}" for k, v in vals.items()]
            print(f"measure={m} " + " ".join(parts))
        if avg:
            parts = [f"{k}={float(v):.6f}" for k, v in avg.items()]
            print("average " + " ".join(parts))

def main():  # pragma: no cover
    app()


