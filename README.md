### OrderOracle

OrderOracle helps you evaluate matryoshka-style embedding dimensionality schedules for retrieval. It measures how well decreasing embedding dimensions preserve ranking quality using IR metrics and correlations (Kendall's tau, RBO) on your datasets.

It includes a simple CLI for:
- pt-export-docs: Export documents from a PyTerrier dataset to JSONL
- embed: Create topic/document embeddings with Sentence-Transformers
- eval: Compute correlation between a dimension schedule and empirical IR scores
- report: Generate a CSV report over multiple measures/correlations

### Why this exists

Matryoshka embeddings let you truncate vectors to smaller prefixes. This project quantifies how rankings change as you truncate from high to low dimensions, so you can choose the right trade-off between speed and effectiveness.

### Installation

Basic install (core functionality without heavy extras):
```bash
pip install orderoracle
```

With Sentence-Transformers backend:
```bash
pip install "orderoracle[st]"
```

With HuggingFace/Transformers Qwen prototype backend:
```bash
pip install "orderoracle[hf]"
```

With PyTerrier + ir_measures for evaluation:
```bash
pip install "orderoracle[pt]"
```

Everything (including optional RBO correlation support):
```bash
pip install "orderoracle[st,hf,pt,extras]"
```

Note for macOS on Apple Silicon: installing torch may require following PyTorch's official instructions. You can also run the CPU-only pipelines.

### Quickstart

After installing, the `orderoracle` CLI is available.

1) Export docs from a PyTerrier dataset (optional helper):
```bash
orderoracle pt-export-docs \
  --dataset irds:vaswani \
  --output-path ./data/docs.jsonl
```

2) Prepare topics and qrels
- topics: JSON with columns `qid` and `text`, or wide Vaswani-style JSON
- qrels: TREC format (`qid 0 docno rel`) or wide JSON with keys `qid`,`docno`,`label`

3) Build embeddings (Sentence-Transformers by default):
```bash
orderoracle embed \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --docs-path ./data/docs.jsonl \
  --topics-path ./data/topics.json \
  --output-path ./data/embeddings
```
This creates `topics.npy` and `docs.npy` in the output directory, or an `.npz` if `--output-path` ends with `.npz`.

4) Evaluate correlations over a dimension schedule:
```bash
orderoracle eval \
  --docs-path ./data/docs.jsonl \
  --topics-path ./data/topics.json \
  --qrels-path ./data/qrels.txt \
  --emb-path ./data/embeddings \
  --dims-pow2 16-512 \
  --measures ndcg,map,P@10 \
  --correlations kendall,rbo \
  --k 10
```

5) Produce a CSV report instead of console output:
```bash
orderoracle report \
  --docs-path ./data/docs.jsonl \
  --topics-path ./data/topics.json \
  --qrels-path ./data/qrels.txt \
  --emb-path ./data/embeddings \
  --dims-pow2 16-512 \
  --measures ndcg,P@10 \
  --correlations kendall,rbo \
  --output-csv ./results/report.csv
```

### Data formats

- Documents (`docs.jsonl`): JSONL with objects `{ "id": str, "text": str }`.
- Topics (`topics.json`):
  - Table-like JSON with columns `qid` and `text`, or
  - Wide Vaswani-style JSON; columns are normalized automatically.
- Qrels:
  - TREC format: `qid 0 docno rel`, or
  - Wide JSON with keys `qid`, `docno`, `label`.
- Embeddings:
  - Directory with `topics.npy` and `docs.npy`, or
  - Single `.npz` with arrays `topics` and `docs`.

### CLI reference

```bash
orderoracle pt-export-docs --dataset irds:vaswani --output-path ./docs.jsonl [--text-field text] [--limit 10000]
```
Exports corpus rows to `{id, text}` JSONL using PyTerrier.

```bash
orderoracle embed --model-name Qwen/Qwen3-Embedding-0.6B --docs-path docs.jsonl --topics-path topics.json --output-path ./embeddings [--device cpu|mps|cuda]
```
Builds embeddings and saves `.npy` or `.npz` files. Skips recomputation for already-present arrays.

```bash
orderoracle eval --docs-path docs.jsonl --topics-path topics.json --qrels-path qrels.txt --emb-path ./embeddings --dims-pow2 16-512 [--measures ndcg,P@10] [--correlations kendall,rbo] [--k 10]
```
Computes correlations between the dimension schedule and effectiveness-based orderings.

```bash
orderoracle report ... --output-csv report.csv
```
Writes a CSV with per-measure correlation values and averages.

### Python API

Basic programmatic usage:
```python
from orderoracle import load_collection_with_embeddings, CollectionQualityEvaluator, SentenceTransformerBackend

backend = SentenceTransformerBackend(model_name="Qwen/Qwen3-Embedding-0.6B")
coll = load_collection_with_embeddings("./data/docs.jsonl", "./data/topics.json", "./data/qrels.txt", "./data/embeddings")
evaluator = CollectionQualityEvaluator(backend, dims_schedule=[16, 32, 64, 128, 256, 512])
result = evaluator.correlations_from_embeddings_pt(coll, measures=["nDCG@10", "P@10"], correlations=["kendall", "rbo"], default_k=10)
print(result)
```

### Troubleshooting

- If RBO is requested, ensure `rbo` is installed: `pip install rbo`.
- For PyTerrier commands, install `python-terrier` and initialize Java if needed.
- On macOS MPS or CUDA devices, ensure your PyTorch install supports the device.

### License

MIT. See `LICENSE`.
