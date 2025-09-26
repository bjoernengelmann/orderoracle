from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import typer
from rich import print

from ..helpers import setup_logging, parse_list_option


logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("pt-export-docs")
    def pt_export_docs_cmd(
        dataset: str = typer.Option(..., help="PyTerrier dataset name, e.g., 'irds:vaswani'"),
        output_path: Path = typer.Option(..., help="Output JSONL path with fields {id, text}"),
        text_field: Optional[str] = typer.Option(None, help="Field name to use for text; auto-detect if omitted"),
        text_fields: Optional[str] = typer.Option(None, help="Comma-separated field names to concatenate in order (all non-empty); overrides --text-field"),
        limit: Optional[int] = typer.Option(None, help="Optional max number of docs to export"),
    ) -> None:
        """Export documents from a PyTerrier dataset into package-compatible JSONL (id, text)."""
        setup_logging()
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
                if hasattr(corpus, "iterrows"):
                    for _, row in corpus.iterrows():
                        yield row.to_dict()
                    return
                if isinstance(corpus, list):
                    for rec in corpus:
                        yield rec
                    return
            raise RuntimeError("Dataset does not expose a corpus iterator or corpus table")

        preferred_fields = ("text", "body", "contents", "raw")
        written = 0
        with output_path.open("w", encoding="utf-8") as out:
            for rec in _iter_records():
                if not isinstance(rec, dict):
                    try:
                        rec = dict(rec)
                    except Exception:
                        continue

                doc_id = rec.get("docno") or rec.get("id") or rec.get("doc_id")
                if doc_id is None:
                    continue
                # Determine candidate fields to extract text from
                candidates = None
                if text_fields is not None:
                    candidates = parse_list_option(None, text_fields) or []
                elif text_field is not None:
                    candidates = [text_field]
                else:
                    candidates = list(preferred_fields)

                text_val = None
                if text_fields is not None:
                    parts = []
                    for f in candidates:
                        val = rec.get(f)
                        if isinstance(val, str) and val.strip():
                            parts.append(val.strip())
                    if parts:
                        text_val = " ".join(parts)
                else:
                    for f in candidates:
                        val = rec.get(f)
                        if isinstance(val, str) and val.strip():
                            text_val = val
                            break
                if text_val is None:
                    text_val = " ".join(str(v) for k, v in rec.items() if isinstance(v, str))
                obj = {"id": str(doc_id), "text": str(text_val)}
                out.write(__import__("json").dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                if limit is not None and written >= int(limit):
                    break
        logger.info("Wrote %d documents to %s", written, str(output_path))
        print(f"[green]Wrote {written} documents to {output_path}[/]")


