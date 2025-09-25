from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np

import pandas as pd

from .models import Collection, Document, Qrel


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_documents(docs_path: str | Path) -> Dict[str, Document]:
    path = Path(docs_path)
    docs: Dict[str, Document] = {}
    for rec in _load_jsonl(path):
        doc_id = str(rec["id"]) if "id" in rec else str(rec["doc_id"])
        text = str(rec["text"]) if "text" in rec else ""
        docs[doc_id] = Document(doc_id=doc_id, text=text)
    return docs


def load_topics(topics_path: str | Path) -> Dict[str, str]:
    """Load topics from JSON using pandas.read_json.

    Expected formats:
    - Wide JSON object with columns like the Vaswani artifacts, e.g., keys 'qid' and 'query'
    - Records/list-like where each row has 'qid'/'topic_id' and 'text'/'query'
    """
    path = Path(topics_path)
    df = pd.read_json(path)
    # Normalize to have columns 'qid' and 'text'
    if "qid" not in df.columns:
        if "topic_id" in df.columns:
            df = df.rename(columns={"topic_id": "qid"})
        elif "id" in df.columns:
            df = df.rename(columns={"id": "qid"})
        else:
            # If 'qid' provided as index, reset
            if df.index.name == "qid" or df.index.name in (None,):
                try:
                    df = df.reset_index().rename(columns={"index": "qid"})
                except Exception:
                    pass
    if "text" not in df.columns:
        if "query" in df.columns:
            df = df.rename(columns={"query": "text"})
        elif "title" in df.columns:
            df = df.rename(columns={"title": "text"})

    # Handle Vaswani wide JSON with dict-like columns
    # Convert dict-like columns to Series
    for col in ("qid", "text"):
        if col in df.columns and isinstance(df[col].iloc[0], dict):
            s = pd.Series(df[col])
            # Ensure row order by numeric index if possible
            try:
                s = s.apply(lambda d: [d[k] for k in sorted(d.keys(), key=lambda x: int(x))])
            except Exception:
                s = s.apply(lambda d: [d[k] for k in sorted(d.keys())])
            # Explode into rows and take first column since we only need one vector
            # Instead build from the dict directly
            df[col] = df[col].map(lambda d: list(d.values()))

    # If columns became lists, align their lengths and construct mapping
    if "qid" in df.columns and isinstance(df["qid"].iloc[0], (list, tuple)):
        qids = [str(x) for x in df["qid"].iloc[0]]
        texts_series = df.get("text")
        if texts_series is not None and isinstance(texts_series.iloc[0], (list, tuple)):
            texts = [str(x) for x in texts_series.iloc[0]]
        else:
            # If 'query' column remained
            queries_series = df.get("query")
            if queries_series is not None and isinstance(queries_series.iloc[0], (list, tuple)):
                texts = [str(x) for x in queries_series.iloc[0]]
            else:
                texts = [""] * len(qids)
        return {qid: text for qid, text in zip(qids, texts)}

    # Otherwise, treat as row-wise table
    # Ensure required columns
    if "qid" not in df.columns:
        raise ValueError(f"Could not determine 'qid' column in topics JSON: {path}")
    if "text" not in df.columns:
        raise ValueError(f"Could not determine 'text' column in topics JSON: {path}")
    return {str(row["qid"]): str(row["text"]) for _, row in df.iterrows()}


def load_qrels_trec(qrels_path: str | Path) -> List[Qrel]:
    path = Path(qrels_path)
    qrels: List[Qrel] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # TREC format: qid 0 docno rel
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docno, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.append(Qrel(topic_id=str(qid), doc_id=str(docno), relevance=int(rel)))
    return qrels


def load_qrels_json(qrels_path: str | Path) -> List[Qrel]:
    """Load qrels from a JSON file in the 'wide columns' format like vaswaniqrels.json.

    Expected top-level keys: 'qid', 'docno', 'label' (and optionally 'iteration').
    Each of these maps either to a dict of row_index -> value (as strings),
    or to a list/array indexed by row_index.
    """
    path = Path(qrels_path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    required_cols = ("qid", "docno", "label")
    for col in required_cols:
        if col not in obj:
            raise ValueError(f"Missing required key '{col}' in qrels JSON: {path}")

    # Determine the iterable of row keys/indices from the 'qid' column
    qid_col = obj["qid"]
    if isinstance(qid_col, dict):
        row_keys = list(qid_col.keys())
        # Sort by integer value if keys look like integers
        try:
            row_keys.sort(key=lambda k: int(k))
        except Exception:
            row_keys.sort()
    elif isinstance(qid_col, list):
        row_keys = [str(i) for i in range(len(qid_col))]
    else:
        raise ValueError("Unsupported qrels JSON structure: 'qid' must be dict or list")

    def get_val(column_obj, row_key: str):
        if isinstance(column_obj, dict):
            return column_obj[row_key]
        if isinstance(column_obj, list):
            return column_obj[int(row_key)]
        raise ValueError("Unsupported column structure in qrels JSON")

    qrels: List[Qrel] = []
    for rk in row_keys:
        qid = get_val(obj["qid"], rk)
        docno = get_val(obj["docno"], rk)
        label = get_val(obj["label"], rk)
        qrels.append(Qrel(topic_id=str(qid), doc_id=str(docno), relevance=int(label)))
    return qrels


def load_qrels(qrels_path: str | Path) -> List[Qrel]:
    """Auto-detect qrels format by file extension and load accordingly.

    - '.json' -> load_qrels_json (wide columns JSON, e.g., vaswaniqrels.json)
    - otherwise -> load_qrels_trec (plain TREC qrels lines: qid 0 docno rel)
    """
    path = Path(qrels_path)
    if path.suffix.lower() == ".json":
        return load_qrels_json(path)
    return load_qrels_trec(path)

def load_collection(
    docs_path: str | Path,
    topics_path: str | Path,
    qrels_path: str | Path,
) -> Collection:
    docs = load_documents(docs_path)
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    return Collection(documents=docs, topics=topics, qrels=qrels)


def load_collection_with_embeddings(
    docs_path: str | Path,
    topics_path: str | Path,
    qrels_path: str | Path,
    emb_path: str | Path,
) -> Collection:
    coll = load_collection(docs_path, topics_path, qrels_path)
    emb_path = Path(emb_path)
    # Expect two .npy files alongside emb_path or a single .npz with keys
    if emb_path.suffix.lower() == ".npz":
        data = np.load(emb_path)
        topics_emb = data.get("topics")
        docs_emb = data.get("docs")
        if topics_emb is None or docs_emb is None:
            raise ValueError("NPZ must contain 'topics' and 'docs' arrays")
        coll.topics_embeddings = topics_emb.astype(np.float32, copy=False)
        coll.docs_embeddings = docs_emb.astype(np.float32, copy=False)
        return coll
    # Otherwise, treat as directory or base stem
    if emb_path.is_dir():
        topics_file = emb_path / "topics.npy"
        docs_file = emb_path / "docs.npy"
    else:
        # If a base stem like path/to/emb, expect emb_topics.npy and emb_docs.npy
        topics_file = emb_path.parent / f"{emb_path.name}_topics.npy"
        docs_file = emb_path.parent / f"{emb_path.name}_docs.npy"
    if not topics_file.exists() or not docs_file.exists():
        raise FileNotFoundError(f"Expected embeddings files: {topics_file} and {docs_file}")
    coll.topics_embeddings = np.load(topics_file).astype(np.float32, copy=False)
    coll.docs_embeddings = np.load(docs_file).astype(np.float32, copy=False)
    return coll
