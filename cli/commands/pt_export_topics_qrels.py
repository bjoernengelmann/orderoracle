from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import json

import typer
from rich import print

from ..helpers import setup_logging


logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("pt-export-topics-qrels")
    def pt_export_topics_qrels_cmd(
        dataset: str = typer.Option(..., help="PyTerrier dataset name, e.g., 'irds:vaswani'"),
        topics_output: Path = typer.Option(..., help="Output topics JSON (list of {qid,text})"),
        qrels_output: Path = typer.Option(..., help="Output qrels file (.txt for TREC lines or .json for wide JSON)"),
        topics_text_field: Optional[str] = typer.Option(
            None,
            help="Preferred text field in topics (e.g., 'query' or 'title'); auto-detect if omitted",
        ),
    ) -> None:
        """Export topics and qrels from a PyTerrier dataset into package-compatible formats.

        - Topics JSON: a JSON array of objects with keys {"qid", "text"}
        - Qrels: TREC format if qrels_output ends with .txt; otherwise wide JSON with keys {"qid","docno","label"}
        """
        setup_logging()
        try:
            import pyterrier as pt  # type: ignore
        except Exception as e:
            raise typer.BadParameter(
                "pyterrier is required for this command. Install with: pip install python-terrier"
            ) from e

        # Initialize PyTerrier quietly
        try:
            pt.init(boot_packages=[], initialise_logging=False, logging=False)  # type: ignore
        except Exception:
            pt.init()

        logger.info("Initializing PyTerrier dataset: %s", dataset)
        ds = pt.get_dataset(dataset)

        # --- Export topics ---
        logger.info("Loading topics from dataset")
        topics_df = None
        if hasattr(ds, "get_topics"):
            topics_df = ds.get_topics()  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Dataset does not provide topics via get_topics()")

        # Normalize columns to ['qid','text']
        cols = {c.lower(): c for c in topics_df.columns}
        qid_col = cols.get("qid") or cols.get("topic") or cols.get("id")
        if qid_col is None:
            raise RuntimeError("Could not find a qid-like column in topics (expected 'qid'/'id')")

        text_col: Optional[str] = None
        if topics_text_field is not None:
            # honor explicit preference if present
            cand = cols.get(topics_text_field.lower())
            if cand is None:
                raise typer.BadParameter(f"Requested topics text field '{topics_text_field}' not found in topics table")
            text_col = cand
        else:
            for name in ("text", "query", "title", "question"):
                cand = cols.get(name)
                if cand is not None:
                    text_col = cand
                    break
        if text_col is None:
            # Fallback: concatenate any string-like columns except qid
            text_col = None

        topics_records = []
        if text_col is not None:
            for _, row in topics_df.iterrows():
                qid = str(row[qid_col])
                text = str(row[text_col]) if row[text_col] is not None else ""
                topics_records.append({"qid": qid, "text": text})
        else:
            string_cols = [c for c in topics_df.columns if c != qid_col]
            for _, row in topics_df.iterrows():
                qid = str(row[qid_col])
                parts = []
                for c in string_cols:
                    val = row[c]
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                topics_records.append({"qid": qid, "text": " ".join(parts)})

        topics_output.parent.mkdir(parents=True, exist_ok=True)
        with topics_output.open("w", encoding="utf-8") as f:
            json.dump(topics_records, f, ensure_ascii=False)
        logger.info("Wrote %d topics to %s", len(topics_records), str(topics_output))
        print(f"[green]Wrote {len(topics_records)} topics to {topics_output}[/]")

        # --- Export qrels ---
        logger.info("Loading qrels from dataset")
        if not hasattr(ds, "get_qrels"):
            raise RuntimeError("Dataset does not provide qrels via get_qrels()")
        qrels_df = ds.get_qrels()  # type: ignore[attr-defined]

        # Normalize qrels column names
        qcols = {c.lower(): c for c in qrels_df.columns}
        q_qid = qcols.get("qid") or qcols.get("topic") or qcols.get("id")
        q_docno = qcols.get("docno") or qcols.get("docid") or qcols.get("doc") or qcols.get("id_document")
        q_label = qcols.get("label") or qcols.get("relevance") or qcols.get("rel")
        if q_qid is None or q_docno is None or q_label is None:
            raise RuntimeError("Qrels table must have qid/docno/label (or recognizable aliases)")

        qrels_output.parent.mkdir(parents=True, exist_ok=True)
        if qrels_output.suffix.lower() == ".json":
            qids = []
            docnos = []
            labels = []
            for _, row in qrels_df.iterrows():
                qids.append(str(row[q_qid]))
                docnos.append(str(row[q_docno]))
                try:
                    labels.append(int(row[q_label]))
                except Exception:
                    # Best-effort integer cast
                    try:
                        labels.append(int(float(row[q_label])))
                    except Exception:
                        labels.append(0)
            with qrels_output.open("w", encoding="utf-8") as f:
                json.dump({"qid": qids, "docno": docnos, "label": labels}, f, ensure_ascii=False)
            logger.info("Wrote %d qrels (JSON) to %s", len(qids), str(qrels_output))
            print(f"[green]Wrote {len(qids)} qrels (JSON) to {qrels_output}[/]")
        else:
            # Default to TREC qrels lines
            written = 0
            with qrels_output.open("w", encoding="utf-8") as f:
                for _, row in qrels_df.iterrows():
                    qid = str(row[q_qid])
                    docno = str(row[q_docno])
                    try:
                        rel = int(row[q_label])
                    except Exception:
                        try:
                            rel = int(float(row[q_label]))
                        except Exception:
                            rel = 0
                    f.write(f"{qid} 0 {docno} {rel}\n")
                    written += 1
            logger.info("Wrote %d qrels (TREC) to %s", written, str(qrels_output))
            print(f"[green]Wrote {written} qrels (TREC) to {qrels_output}[/]")


