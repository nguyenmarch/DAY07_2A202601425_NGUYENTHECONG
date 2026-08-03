#!/usr/bin/env python3
"""Run the five HUST benchmark queries and print their top-k retrieval results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest import build_knowledge_base
from src import (
    FastEmbedder,
    FixedSizeChunker,
    HeadingSectionChunker,
    LocalEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)


DATA_DIR = ROOT / "data" / "hust_academic"
BENCHMARK_PATH = DATA_DIR / "benchmarks.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("fastembed", "local", "mock"), default="fastembed")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=("fixed", "sentence", "recursive", "heading"),
        default="fixed",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")

    if args.provider == "fastembed":
        try:
            embedder = FastEmbedder()
        except Exception as error:
            raise SystemExit(
                "FastEmbed is unavailable. Run: "
                ".venv/bin/python -m pip install -r requirements-fastembed.txt\n"
                f"Cause: {error}"
            ) from error
    elif args.provider == "local":
        try:
            embedder = LocalEmbedder()
        except Exception as error:
            raise SystemExit(
                "Local embedder is unavailable. Run: "
                ".venv/bin/python -m pip install -r requirements-local.txt\n"
                f"Cause: {error}"
            ) from error
    else:
        embedder = _mock_embed
        print("WARNING: mock scores are only a pipeline smoke test, not semantic evaluation.\n")

    if args.strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.strategy == "sentence":
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    elif args.strategy == "recursive":
        chunker = RecursiveChunker(chunk_size=args.chunk_size)
    else:
        chunker = HeadingSectionChunker(chunk_size=args.chunk_size)

    store = build_knowledge_base(
        DATA_DIR,
        embedding_fn=embedder,
        chunker=chunker,
        collection_name="hust_academic_benchmark",
    )
    benchmarks = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    hits = 0

    for item in benchmarks:
        results = store.search_with_filter(
            item["query"],
            top_k=args.top_k,
            metadata_filter=item["metadata_filter"],
        )
        expected_text = item["expected_text"].casefold()
        relevant = any(
            result["metadata"].get("doc_id") == item["expected_doc_id"]
            and expected_text in result["content"].casefold()
            for result in results
        )
        hits += int(relevant)
        print(f"{item['id']}: {item['query']}")
        print(f"  expected={item['expected_doc_id']} relevant-chunk-in-top-{args.top_k}={relevant}")
        for rank, result in enumerate(results, start=1):
            preview = " ".join(result["content"].split())[:160]
            print(
                f"  {rank}. score={result['score']:.4f} "
                f"doc={result['metadata'].get('doc_id')} "
                f"chunk={result['metadata'].get('chunk_index')} | {preview}"
            )
        print()

    print(f"Retrieval hit rate: {hits}/{len(benchmarks)} queries have the expected document in top-{args.top_k}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
