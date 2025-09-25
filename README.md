### OrderOracle

OrderOracle evaluates the quality of IR test collections (documents, topics, qrels) by checking how well the empirical ranking of retrieval systems induced by the collection correlates with an a priori target ranking. The core assumption is: the stronger this correlation, the better the test collection.

It includes a simple CLI for:
- pt-export-docs: Export documents from a PyTerrier dataset to JSONL
- embed: Create topic/document embeddings with Sentence-Transformers
- eval: Compute correlation between the empirical systems ranking and an a priori ranking
- report: Generate a CSV report over multiple measures/correlations

### Why this exists

Evaluating a test collection should not only look at absolute effectiveness numbers; it should also reflect whether the collection preserves the expected relative ordering of systems. If the system ordering you obtain from a collection aligns with a reasonable a priori ordering (e.g., based on theory, cost/complexity, or an external leaderboard), then the collection is likely informative and discriminative. OrderOracle operationalizes this idea by computing correlations (e.g., Kendall's tau, RBO) between the empirical system ranking induced by effectiveness measurements and a provided a priori ranking.

### Current specialization: matryoshka-style embedding schedules

The current implementation specializes the general idea by treating different embedding truncation dimensions ("matryoshka" schedules) as the set of "systems". The a priori ranking is simply the natural order of dimensions (e.g., 16 < 32 < 64 < ...), and the empirical ranking is derived from IR effectiveness measured over your topics/qrels. High correlation indicates that smaller dimensional variants degrade gracefully in the expected order.

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

The Quickstart below demonstrates the matryoshka specialization where "systems" are embedding truncation dimensions and the a priori ranking is the dimension order.

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
Computes correlations between the empirical systems ranking (from effectiveness) and the a priori ranking. In the default specialization, systems are embedding dimensions ordered by size.

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
The example above illustrates the matryoshka specialization where `dims_schedule` enumerates the systems and their a priori order.

### Troubleshooting

- To enable debug logging, set `ORDERORACLE_LOG_LEVEL=DEBUG`:
```bash
# macOS/Linux (current shell)
export ORDERORACLE_LOG_LEVEL=DEBUG

# One-shot for a single command
ORDERORACLE_LOG_LEVEL=DEBUG orderoracle eval ...

# Windows PowerShell (current session)
$env:ORDERORACLE_LOG_LEVEL = 'DEBUG'
```

### License

MIT. See `LICENSE`.
