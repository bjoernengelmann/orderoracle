from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import typer
from rich import print

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


