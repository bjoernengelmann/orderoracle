from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Tuple, List, Callable
import logging

import numpy as np
import requests
import typer
from rich import print
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from threading import Lock

try:  # Optional async HTTP client for best performance
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore

from ... import SentenceTransformerBackend
from ...data.io import load_documents, load_topics
from ..helpers import setup_logging


logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("embed")
    def embed_cmd(
        model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B", help="Sentence Transformers model name"),
        docs_path: Path = typer.Option(..., help="Path to documents JSONL with 'id' and 'text'"),
        topics_path: Path = typer.Option(..., help="Path to topics JSON (vaswani-style or records)"),
        output_path: Path = typer.Option(..., help="Output directory, base path, or .npz file for embeddings"),
        device: Optional[str] = typer.Option(None, help="Device: cpu | mps | gpu | cuda[:N]"),
        batch_size: int = typer.Option(32, help="Batch size for the embedding model"),
        truncate_chars: Optional[int] = typer.Option(
            None,
            help="If set, truncate each input to the first N characters",
        ),
        truncate_tokens: Optional[int] = typer.Option(
            None,
            help="If set, truncate each input to the first N tokens (uses model tokenizer if available)",
        ),
    ) -> None:
        """Embed documents and topics and save in a format load_collection_with_embeddings expects."""
        setup_logging()
        logger.info(
            "Starting embedding with model=%s device=%s batch_size=%d trunc_chars=%s trunc_tokens=%s",
            model_name,
            device,
            int(batch_size),
            str(truncate_chars),
            str(truncate_tokens),
        )

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
                print(f"[yellow]Embeddings already exist at {output_path}; skipping[/]")
                return

            backend = SentenceTransformerBackend(model_name=model_name, device=device, batch_size=batch_size)

            def _embed_topics_with_prompt(texts: list[str]) -> np.ndarray:
                model = getattr(backend, "model", None)
                encode_fn = getattr(model, "encode", None)
                if encode_fn is not None:
                    try:
                        return model.encode(  # type: ignore[no-any-return]
                            texts,
                            batch_size=backend.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=True,
                            prompt_name="query",
                        )
                    except TypeError:
                        # Model/encode does not support prompt_name; fall back
                        pass
                return backend.embed_texts(texts, max_dim=None)

            def _truncate_texts(texts: list[str]) -> list[str]:
                if truncate_chars is not None and truncate_tokens is not None:
                    raise typer.BadParameter("Use only one of --truncate-chars or --truncate-tokens")
                if truncate_chars is not None:
                    n = int(truncate_chars)
                    return [t[:n] for t in texts]
                if truncate_tokens is not None:
                    # Prefer tokenizer from the backend model, fallback to HF AutoTokenizer
                    tok = getattr(getattr(backend, "model", None), "tokenizer", None)
                    if tok is None:
                        try:
                            from transformers import AutoTokenizer  # type: ignore

                            tok = AutoTokenizer.from_pretrained(model_name)
                        except Exception as e:
                            raise typer.BadParameter(
                                "Token-based truncation requires a tokenizer (install 'transformers') or use --truncate-chars"
                            ) from e
                    enc = tok(
                        texts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=int(truncate_tokens),
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    input_ids_list = enc["input_ids"]
                    return tok.batch_decode(input_ids_list, skip_special_tokens=True)
                return texts

            if not have_topics:
                topics = load_topics(topics_path)
                topic_texts = list(topics.values())
                topic_texts = _truncate_texts(topic_texts)
                logger.info("Loaded %d topics for embedding (NPZ)", len(topic_texts))
                topics_emb = _embed_topics_with_prompt(topic_texts)
                logger.info("Computed topics embeddings: shape=%s", tuple(topics_emb.shape))
            else:
                topics_emb = existing_topics  # type: ignore[assignment]

            if not have_docs:
                docs = load_documents(docs_path)
                doc_texts = [d.text for d in docs.values()]
                doc_texts = _truncate_texts(doc_texts)
                logger.info("Loaded %d docs for embedding (NPZ)", len(doc_texts))
                docs_emb = backend.embed_texts(doc_texts, max_dim=None)
                logger.info("Computed docs embeddings: shape=%s", tuple(docs_emb.shape))
            else:
                docs_emb = existing_docs  # type: ignore[assignment]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, topics=topics_emb, docs=docs_emb)
            logger.info("Saved NPZ embeddings to %s (keys: topics, docs)", str(output_path))
            print(f"[green]Saved NPZ embeddings to {output_path}[/]")
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
                print(f"[yellow]Embeddings already exist at {topics_file} and {docs_file}; skipping[/]")
                return

            out_dir.mkdir(parents=True, exist_ok=True)

            backend = SentenceTransformerBackend(model_name=model_name, device=device, batch_size=batch_size)

            def _embed_topics_with_prompt(texts: list[str]) -> np.ndarray:
                model = getattr(backend, "model", None)
                encode_fn = getattr(model, "encode", None)
                if encode_fn is not None:
                    try:
                        return model.encode(  # type: ignore[no-any-return]
                            texts,
                            batch_size=backend.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=True,
                            prompt_name="query",
                        )
                    except TypeError:
                        # Model/encode does not support prompt_name; fall back
                        pass
                return backend.embed_texts(texts, max_dim=None)

            def _truncate_texts(texts: list[str]) -> list[str]:
                if truncate_chars is not None and truncate_tokens is not None:
                    raise typer.BadParameter("Use only one of --truncate-chars or --truncate-tokens")
                if truncate_chars is not None:
                    n = int(truncate_chars)
                    return [t[:n] for t in texts]
                if truncate_tokens is not None:
                    tok = getattr(getattr(backend, "model", None), "tokenizer", None)
                    if tok is None:
                        try:
                            from transformers import AutoTokenizer  # type: ignore

                            tok = AutoTokenizer.from_pretrained(model_name)
                        except Exception as e:
                            raise typer.BadParameter(
                                "Token-based truncation requires a tokenizer (install 'transformers') or use --truncate-chars"
                            ) from e
                    enc = tok(
                        texts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=int(truncate_tokens),
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    input_ids_list = enc["input_ids"]
                    return tok.batch_decode(input_ids_list, skip_special_tokens=True)
                return texts

            if not topics_exists:
                topics = load_topics(topics_path)
                topic_texts = list(topics.values())
                topic_texts = _truncate_texts(topic_texts)
                logger.info("Loaded %d topics for embedding", len(topic_texts))
                topics_emb = _embed_topics_with_prompt(topic_texts)
                logger.info("Computed topics embeddings: shape=%s", tuple(topics_emb.shape))
                np.save(topics_file, topics_emb)
                logger.info("Saved topics embeddings to %s", str(topics_file))
                print(f"[green]Saved topics embeddings to {topics_file}[/]")

            if not docs_exists:
                docs = load_documents(docs_path)
                doc_texts = [d.text for d in docs.values()]
                doc_texts = _truncate_texts(doc_texts)
                logger.info("Loaded %d docs for embedding", len(doc_texts))
                docs_emb = backend.embed_texts(doc_texts, max_dim=None)
                logger.info("Computed docs embeddings: shape=%s", tuple(docs_emb.shape))
                np.save(docs_file, docs_emb)
                logger.info("Saved docs embeddings to %s", str(docs_file))
                print(f"[green]Saved docs embeddings to {docs_file}[/]")




    @app.command("embed_via_api")
    def embed_via_api_cmd(
        model_name: str = typer.Option("Qwen/Qwen3-Embedding-0.6B", help="Embeddings model name on the SGLang server"),
        docs_path: Path = typer.Option(..., help="Path to documents JSONL with 'id' and 'text'"),
        topics_path: Path = typer.Option(..., help="Path to topics JSON (vaswani-style or records)"),
        output_path: Path = typer.Option(..., help="Output directory, base path, or .npz file for embeddings"),
        api_url: str = typer.Option("http://localhost:30000", help="Base URL of the SGLang server (no trailing /v1)"),
        batch_size: int = typer.Option(64, help="Batch size per API request"),
        concurrency: int = typer.Option(8, help="Max concurrent requests to the API"),
        timeout: int = typer.Option(300, help="HTTP timeout (seconds) for embedding requests"),
        truncate_chars: Optional[int] = typer.Option(
            None,
            help="If set, truncate each input to the first N characters",
        ),
        truncate_tokens: Optional[int] = typer.Option(
            None,
            help="If set, truncate each input to the first N tokens (uses model tokenizer if available)",
        ),
    ) -> None:
        """Embed documents and topics via SGLang-style /v1/embeddings API and save alongside local pipeline outputs."""
        setup_logging()
        logger.info(
            "Starting API embedding with model=%s api_url=%s batch_size=%d trunc_chars=%s trunc_tokens=%s timeout=%d",
            model_name,
            api_url,
            int(batch_size),
            str(truncate_chars),
            str(truncate_tokens),
            int(timeout),
        )

        # Suppress noisy httpx/httpcore INFO logs (e.g., per-request lines)
        try:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)
            httpx_logger.propagate = False
            httpcore_logger = logging.getLogger("httpcore")
            httpcore_logger.setLevel(logging.WARNING)
            httpcore_logger.propagate = False
        except Exception:
            pass

        embeddings_endpoint = api_url.rstrip("/") + "/v1/embeddings"

        def _resolve_paths_for_label(label: str) -> Tuple[Optional[Path], Path, Path]:
            # Returns (final_target_path_if_applicable, partial_npy_path, meta_json_path)
            if output_path.suffix.lower() == ".npz":
                base_dir = output_path.parent
                stem = output_path.stem
                partial = base_dir / f"{stem}.{label}.partial.npy"
                meta = base_dir / f"{stem}.{label}.partial.json"
                return (None, partial, meta)
            else:
                if output_path.is_dir():
                    out_dir = output_path
                    final = out_dir / f"{label}.npy"
                    partial = out_dir / f"{label}.partial.npy"
                    meta = out_dir / f"{label}.partial.json"
                else:
                    out_dir = output_path.parent
                    final = out_dir / f"{output_path.name}_{label}.npy"
                    partial = out_dir / f"{output_path.name}_{label}.partial.npy"
                    meta = out_dir / f"{output_path.name}_{label}.partial.json"
                return (final, partial, meta)

        def _load_meta(meta_path: Path) -> dict:
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return {}
            return {}

        def _save_meta(meta_path: Path, data: dict) -> None:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(meta_path)

        def _truncate_texts(texts: list[str]) -> list[str]:
            if truncate_chars is not None and truncate_tokens is not None:
                raise typer.BadParameter("Use only one of --truncate-chars or --truncate-tokens")
            if truncate_chars is not None:
                n = int(truncate_chars)
                return [t[:n] for t in texts]
            if truncate_tokens is not None:
                # Prefer tokenizer from HF to match local behavior
                try:
                    from transformers import AutoTokenizer  # type: ignore

                    tok = AutoTokenizer.from_pretrained(model_name)
                except Exception as e:
                    raise typer.BadParameter(
                        "Token-based truncation requires a tokenizer (install 'transformers') or use --truncate-chars"
                    ) from e
                enc = tok(
                    texts,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=int(truncate_tokens),
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids_list = enc["input_ids"]
                return tok.batch_decode(input_ids_list, skip_special_tokens=True)
            return texts

        def _post_embeddings(payload: dict) -> dict:
            r = requests.post(embeddings_endpoint, json=payload, timeout=float(timeout))
            if not r.ok:
                raise RuntimeError(f"Embeddings API error {r.status_code}: {r.text[:300]}")
            return r.json()

        async def _post_embeddings_async(client: Any, payload: dict) -> dict:
            assert httpx is not None  # for type checkers
            resp = await client.post(embeddings_endpoint, json=payload)
            if resp.status_code < 200 or resp.status_code >= 300:
                text = resp.text[:300]
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {text}")
            return resp.json()

        async def _embed_texts_api_async(
            texts: list[str],
            *,
            label: str,
            writer: Optional[Callable[[int, np.ndarray], None]] = None,
            start_offset: int = 0,
            total_for_progress: Optional[int] = None,
        ) -> np.ndarray:
            assert httpx is not None  # for type checkers
            total = len(texts)
            if total == 0:
                return np.zeros((0, 0), dtype=np.float32)
            results: List[Optional[np.ndarray]] = [None] * total  # type: ignore[name-defined]
            sem = asyncio.Semaphore(max(1, int(concurrency)))

            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                with Progress(
                    TextColumn("[bold blue]" + label + "[/]"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                ) as progress:
                    task_total = total if total_for_progress is None else int(total_for_progress)
                    task_id = progress.add_task("embedding", total=task_total)

                    async def process_batch(start_index: int, batch_texts: list[str]) -> None:
                        async with sem:
                            payload = {"model": model_name, "input": batch_texts, "encoding_format": "float"}
                            try:
                                data = await _post_embeddings_async(client, payload)
                                parsed = [np.asarray(d["embedding"], dtype=np.float32) for d in data.get("data", [])]
                                if len(parsed) != len(batch_texts):
                                    raise RuntimeError(
                                        f"Embeddings API returned {len(parsed)} vectors for {len(batch_texts)} inputs"
                                    )
                                for i, vec in enumerate(parsed):
                                    abs_index = start_index + i
                                    results[abs_index] = vec
                                    if writer is not None:
                                        writer(start_offset + abs_index, vec)
                                progress.update(task_id, advance=len(batch_texts))
                            except Exception as e:
                                logger.warning(
                                    "Batch %d-%d failed: %s; falling back to per-item",
                                    start_index,
                                    start_index + len(batch_texts),
                                    e,
                                )
                                for i, text in enumerate(batch_texts):
                                    single_payload = {"model": model_name, "input": text, "encoding_format": "float"}
                                    try:
                                        data_single = await _post_embeddings_async(client, single_payload)
                                    except Exception:
                                        if truncate_chars is not None and isinstance(text, str):
                                            truncated = text[: int(truncate_chars)]
                                            data_single = await _post_embeddings_async(
                                                client, {**single_payload, "input": truncated}
                                            )
                                        else:
                                            raise
                                    vec = np.asarray(data_single["data"][0]["embedding"], dtype=np.float32)
                                    abs_index = start_index + i
                                    results[abs_index] = vec
                                    if writer is not None:
                                        writer(start_offset + abs_index, vec)
                                    progress.update(task_id, advance=1)

                    tasks = []
                    for start in range(0, total, int(batch_size)):
                        batch = texts[start : start + int(batch_size)]
                        tasks.append(asyncio.create_task(process_batch(start, batch)))
                    await asyncio.gather(*tasks)

            # Ensure all results filled
            filled: List[np.ndarray] = []
            for idx, v in enumerate(results):
                if v is None:
                    raise RuntimeError(f"Missing embedding at position {idx}")
                filled.append(v)
            return np.vstack(filled)

        def _embed_texts_api_threaded(
            texts: list[str],
            *,
            label: str,
            writer: Optional[Callable[[int, np.ndarray], None]] = None,
            start_offset: int = 0,
            total_for_progress: Optional[int] = None,
        ) -> np.ndarray:
            # Concurrent requests using threads and requests library
            total = len(texts)
            if total == 0:
                return np.zeros((0, 0), dtype=np.float32)
            results: List[Optional[np.ndarray]] = [None] * total  # type: ignore[name-defined]

            def post_batch(start_index: int, batch_texts: list[str]) -> Tuple[int, Optional[List[np.ndarray]], Optional[Exception]]:
                payload = {"model": model_name, "input": batch_texts, "encoding_format": "float"}
                try:
                    data = _post_embeddings(payload)
                    parsed = [np.asarray(d["embedding"], dtype=np.float32) for d in data.get("data", [])]
                    if len(parsed) != len(batch_texts):
                        raise RuntimeError(
                            f"Embeddings API returned {len(parsed)} vectors for {len(batch_texts)} inputs"
                        )
                    return (start_index, parsed, None)
                except Exception as e:
                    return (start_index, None, e)

            with Progress(
                TextColumn("[bold blue]" + label + "[/]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task_total = total if total_for_progress is None else int(total_for_progress)
                task_id = progress.add_task("embedding", total=task_total)
                with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
                    futures = []
                    for start in range(0, total, int(batch_size)):
                        batch = texts[start : start + int(batch_size)]
                        futures.append(executor.submit(post_batch, start, batch))
                    for fut in as_completed(futures):
                        start_index, parsed, err = fut.result()
                        if parsed is not None:
                            for i, vec in enumerate(parsed):
                                abs_index = start_index + i
                                results[abs_index] = vec
                                if writer is not None:
                                    writer(start_offset + abs_index, vec)
                            progress.update(task_id, advance=len(parsed))
                        else:
                            logger.warning(
                                "Batch %d-%d failed: %s; falling back to per-item",
                                start_index,
                                start_index + int(batch_size),
                                err,
                            )
                            batch_texts = texts[start_index : start_index + int(batch_size)]
                            for i, text in enumerate(batch_texts):
                                single_payload = {"model": model_name, "input": text, "encoding_format": "float"}
                                try:
                                    data_single = _post_embeddings(single_payload)
                                except Exception:
                                    if truncate_chars is not None and isinstance(text, str):
                                        truncated = text[: int(truncate_chars)]
                                        data_single = _post_embeddings({**single_payload, "input": truncated})
                                    else:
                                        raise
                                vec = np.asarray(data_single["data"][0]["embedding"], dtype=np.float32)
                                abs_index = start_index + i
                                results[abs_index] = vec
                                if writer is not None:
                                    writer(start_offset + abs_index, vec)
                                progress.update(task_id, advance=1)

            filled: List[np.ndarray] = []
            for idx, v in enumerate(results):
                if v is None:
                    raise RuntimeError(f"Missing embedding at position {idx}")
                filled.append(v)
            return np.vstack(filled)

        def _embed_texts_api(
            texts: list[str], *, label: str, writer: Optional[Callable[[int, np.ndarray], None]] = None, start_offset: int = 0, total_for_progress: Optional[int] = None
        ) -> np.ndarray:
            if httpx is not None:
                try:
                    return asyncio.run(
                        _embed_texts_api_async(
                            texts, label=label, writer=writer, start_offset=start_offset, total_for_progress=total_for_progress
                        )
                    )
                except RuntimeError as e:
                    # If already in an event loop (e.g., Jupyter), use nest_asyncio-like pattern
                    try:
                        import nest_asyncio  # type: ignore

                        nest_asyncio.apply()  # type: ignore[attr-defined]
                        loop = asyncio.get_event_loop()
                        return loop.run_until_complete(
                            _embed_texts_api_async(
                                texts, label=label, writer=writer, start_offset=start_offset, total_for_progress=total_for_progress
                            )
                        )
                    except Exception:
                        raise e
            logger.warning("httpx not available; falling back to threaded requests for concurrency")
            return _embed_texts_api_threaded(
                texts, label=label, writer=writer, start_offset=start_offset, total_for_progress=total_for_progress
            )

        # NPZ path: allow partial reuse and skipping when both present
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
                print(f"[yellow]Embeddings already exist at {output_path}; skipping[/]")
                return

            topics_emb: Optional[np.ndarray]
            docs_emb: Optional[np.ndarray]

            if not have_topics:
                topics = load_topics(topics_path)
                topic_texts = list(topics.values())
                topic_texts = _truncate_texts(topic_texts)
                logger.info("Loaded %d topics for API embedding (NPZ)", len(topic_texts))
                # Checkpointing setup
                _, partial_npy, meta_json = _resolve_paths_for_label("topics")
                meta = _load_meta(meta_json)
                completed = int(meta.get("completed", 0))
                dim = meta.get("dim")
                mmap: Optional[np.memmap] = None
                lock = Lock()

                def writer(idx: int, vec: np.ndarray) -> None:
                    nonlocal mmap, dim, completed
                    with lock:
                        if mmap is None:
                            dim = int(vec.shape[0])
                            shape = (len(topic_texts), dim)
                            mmap = np.lib.format.open_memmap(partial_npy, mode="w+", dtype=np.float32, shape=shape)
                        mmap[idx] = vec
                        completed = max(completed, idx + 1)
                        _save_meta(meta_json, {"completed": completed, "dim": dim})

                # If resuming, pre-create memmap and update progress start
                start_offset = 0
                if partial_npy.exists() and meta_json.exists() and completed > 0 and dim is not None:
                    shape = (len(topic_texts), int(dim))
                    mmap = np.lib.format.open_memmap(partial_npy, mode="r+", dtype=np.float32, shape=shape)
                    start_offset = completed
                remaining_texts = topic_texts[start_offset:]
                topics_emb = None
                if len(remaining_texts) > 0:
                    _ = _embed_texts_api(
                        remaining_texts,
                        label="topics",
                        writer=lambda rel_idx, v: writer(rel_idx, v),
                        start_offset=start_offset,
                        total_for_progress=len(topic_texts),
                    )
                # Load full array (from memmap or built if no checkpointing occurred)
                if partial_npy.exists() and meta_json.exists():
                    shape = (len(topic_texts), int(dim if dim is not None else 0))
                    mmap = np.lib.format.open_memmap(partial_npy, mode="r", dtype=np.float32, shape=shape)
                    topics_emb = np.array(mmap)
                else:
                    # In the unlikely case writer wasn't used (empty texts), create empty
                    topics_emb = np.zeros((len(topic_texts), 0), dtype=np.float32)
                logger.info("Computed topics embeddings via API: shape=%s", tuple(topics_emb.shape))
            else:
                topics_emb = existing_topics  # type: ignore[assignment]

            if not have_docs:
                docs = load_documents(docs_path)
                doc_texts = [d.text for d in docs.values()]
                doc_texts = _truncate_texts(doc_texts)
                logger.info("Loaded %d docs for API embedding (NPZ)", len(doc_texts))
                # Checkpointing for docs
                _, partial_npy_d, meta_json_d = _resolve_paths_for_label("docs")
                meta_d = _load_meta(meta_json_d)
                completed_d = int(meta_d.get("completed", 0))
                dim_d = meta_d.get("dim")
                mmap_d: Optional[np.memmap] = None
                lock_d = Lock()

                def writer_d(idx: int, vec: np.ndarray) -> None:
                    nonlocal mmap_d, dim_d, completed_d
                    with lock_d:
                        if mmap_d is None:
                            dim_d = int(vec.shape[0])
                            shape_d = (len(doc_texts), dim_d)
                            mmap_d = np.lib.format.open_memmap(partial_npy_d, mode="w+", dtype=np.float32, shape=shape_d)
                        mmap_d[idx] = vec
                        completed_d = max(completed_d, idx + 1)
                        _save_meta(meta_json_d, {"completed": completed_d, "dim": dim_d})

                start_offset_d = 0
                if partial_npy_d.exists() and meta_json_d.exists() and completed_d > 0 and dim_d is not None:
                    shape_d = (len(doc_texts), int(dim_d))
                    mmap_d = np.lib.format.open_memmap(partial_npy_d, mode="r+", dtype=np.float32, shape=shape_d)
                    start_offset_d = completed_d
                remaining_docs = doc_texts[start_offset_d:]
                docs_emb = None
                if len(remaining_docs) > 0:
                    _ = _embed_texts_api(
                        remaining_docs,
                        label="docs",
                        writer=lambda rel_idx, v: writer_d(rel_idx, v),
                        start_offset=start_offset_d,
                        total_for_progress=len(doc_texts),
                    )
                if partial_npy_d.exists() and meta_json_d.exists():
                    shape_d = (len(doc_texts), int(dim_d if dim_d is not None else 0))
                    mmap_d = np.lib.format.open_memmap(partial_npy_d, mode="r", dtype=np.float32, shape=shape_d)
                    docs_emb = np.array(mmap_d)
                else:
                    docs_emb = np.zeros((len(doc_texts), 0), dtype=np.float32)
                logger.info("Computed docs embeddings via API: shape=%s", tuple(docs_emb.shape))
            else:
                docs_emb = existing_docs  # type: ignore[assignment]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, topics=topics_emb, docs=docs_emb)
            # Cleanup partials on success
            for lbl in ("topics", "docs"):
                _, p_npy, p_meta = _resolve_paths_for_label(lbl)
                try:
                    if p_npy.exists():
                        p_npy.unlink()
                    if p_meta.exists():
                        p_meta.unlink()
                except Exception:
                    pass
            logger.info("Saved NPZ embeddings to %s (keys: topics, docs)", str(output_path))
            print(f"[green]Saved NPZ embeddings to {output_path}[/]")
            return

        # Directory or base stem outputs
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
            print(f"[yellow]Embeddings already exist at {topics_file} and {docs_file}; skipping[/]")
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        if not topics_exists:
            topics = load_topics(topics_path)
            topic_texts = list(topics.values())
            topic_texts = _truncate_texts(topic_texts)
            logger.info("Loaded %d topics for API embedding", len(topic_texts))
            final_path_t, partial_t, meta_t = _resolve_paths_for_label("topics")
            meta = _load_meta(meta_t)
            completed = int(meta.get("completed", 0))
            dim = meta.get("dim")
            mmap: Optional[np.memmap] = None
            lock = Lock()

            def writer(idx: int, vec: np.ndarray) -> None:
                nonlocal mmap, dim, completed
                with lock:
                    if mmap is None:
                        dim = int(vec.shape[0])
                        shape = (len(topic_texts), dim)
                        mmap = np.lib.format.open_memmap(partial_t, mode="w+", dtype=np.float32, shape=shape)
                    mmap[idx] = vec
                    completed = max(completed, idx + 1)
                    _save_meta(meta_t, {"completed": completed, "dim": dim})

            start_offset = 0
            if partial_t.exists() and meta_t.exists() and completed > 0 and dim is not None:
                shape = (len(topic_texts), int(dim))
                mmap = np.lib.format.open_memmap(partial_t, mode="r+", dtype=np.float32, shape=shape)
                start_offset = completed
            remaining_texts = topic_texts[start_offset:]
            if len(remaining_texts) > 0:
                _ = _embed_texts_api(
                    remaining_texts,
                    label="topics",
                    writer=lambda rel_idx, v: writer(rel_idx, v),
                    start_offset=start_offset,
                    total_for_progress=len(topic_texts),
                )
            # finalize to final file
            if partial_t.exists() and meta_t.exists():
                shape = (len(topic_texts), int(dim if dim is not None else 0))
                mmap = np.lib.format.open_memmap(partial_t, mode="r", dtype=np.float32, shape=shape)
                np.save(topics_file, np.array(mmap))
                try:
                    partial_t.unlink()
                    meta_t.unlink()
                except Exception:
                    pass
            logger.info("Saved topics embeddings to %s", str(topics_file))
            print(f"[green]Saved topics embeddings to {topics_file}[/]")

        if not docs_exists:
            docs = load_documents(docs_path)
            doc_texts = [d.text for d in docs.values()]
            doc_texts = _truncate_texts(doc_texts)
            logger.info("Loaded %d docs for API embedding", len(doc_texts))
            final_path_d, partial_d, meta_d = _resolve_paths_for_label("docs")
            meta = _load_meta(meta_d)
            completed = int(meta.get("completed", 0))
            dim = meta.get("dim")
            mmap: Optional[np.memmap] = None
            lock = Lock()

            def writer_d(idx: int, vec: np.ndarray) -> None:
                nonlocal mmap, dim, completed
                with lock:
                    if mmap is None:
                        dim = int(vec.shape[0])
                        shape = (len(doc_texts), dim)
                        mmap = np.lib.format.open_memmap(partial_d, mode="w+", dtype=np.float32, shape=shape)
                    mmap[idx] = vec
                    completed = max(completed, idx + 1)
                    _save_meta(meta_d, {"completed": completed, "dim": dim})

            start_offset = 0
            if partial_d.exists() and meta_d.exists() and completed > 0 and dim is not None:
                shape = (len(doc_texts), int(dim))
                mmap = np.lib.format.open_memmap(partial_d, mode="r+", dtype=np.float32, shape=shape)
                start_offset = completed
            remaining_docs = doc_texts[start_offset:]
            if len(remaining_docs) > 0:
                _ = _embed_texts_api(
                    remaining_docs,
                    label="docs",
                    writer=lambda rel_idx, v: writer_d(rel_idx, v),
                    start_offset=start_offset,
                    total_for_progress=len(doc_texts),
                )
            if partial_d.exists() and meta_d.exists():
                shape = (len(doc_texts), int(dim if dim is not None else 0))
                mmap = np.lib.format.open_memmap(partial_d, mode="r", dtype=np.float32, shape=shape)
                np.save(docs_file, np.array(mmap))
                try:
                    partial_d.unlink()
                    meta_d.unlink()
                except Exception:
                    pass
            logger.info("Saved docs embeddings to %s", str(docs_file))
            print(f"[green]Saved docs embeddings to {docs_file}[/]")
