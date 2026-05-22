"""
src/evaluation.py

Evaluation pipeline — STAGE 1: Generation.

Runs each RAG variant (Naive, Hybrid, Agentic) against every query in
data/eval/eval_queries.json and persists the raw outputs to
data/eval/generations.json. Stage 2 (metrics) reads from this file
without needing to re-run any LLM calls.

Outputs per record include:
  - The query and its metadata (id, type, category, expected_source_ids)
  - The generated answer
  - The retrieved Q-A pairs (with source_id for citation grounding metric)
  - Agentic extras: sub-queries, verification verdict, revision flag

Usage:
    python -m src.evaluation generate                  # all variants, all queries
    python -m src.evaluation generate --variant naive  # one variant only
    python -m src.evaluation generate --top-k 3
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import os
from typing import Optional

import numpy as np

from src.eval_metrics import MetricsEvaluator, MetricsRecord
from tqdm import tqdm

from src.rag_pipeline import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_VARIANTS = ["naive", "hybrid", "agentic"]
EVAL_QUERIES_PATH = Path("data/eval/eval_queries.json")
GENERATIONS_PATH = Path("data/eval/generations.json")
RESULTS_PATH = Path("data/eval/results.json")

def load_eval_queries() -> list[dict]:
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def serialize_retrieved(retrieved: list[dict]) -> list[dict]:
    """
    Strip non-serializable fields and normalise per-result shape across
    retrieval variants (Naive doesn't have rerank_score, etc.).
    """
    out: list[dict] = []
    for r in retrieved:
        out.append({
            "id": r.get("id", ""),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "source_id": r.get("source_id", ""),
            "source_type": r.get("source_type", ""),
            "source_url": r.get("source_url", ""),
            "qa_type": r.get("qa_type", ""),
            "distance": float(r.get("distance", 0.0)),
            "rerank_score": (
                float(r["rerank_score"]) if "rerank_score" in r else None
            ),
            "fused_score": (
                float(r["fused_score"]) if "fused_score" in r else None
            ),
            "from_subquery": r.get("from_subquery"),
        })
    return out


def run_single_query(pipeline, variant: str, q: dict, top_k: int) -> dict:
    """Run one query through a variant; return a result record."""
    record: dict = {
        "variant_id": variant,
        "query_id": q["id"],
        "query": q["query"],
        "category": q["category"],
        "type": q["type"],
        "expected_source_ids": q["expected_source_ids"],
        "expected_behavior": q.get("expected_behavior", ""),
        "top_k": top_k,
    }
    try:
        output = pipeline.answer(q["query"], top_k=top_k)
        record["variant_name"] = output.get("variant", variant)
        record["answer"] = output["answer"]
        record["retrieved"] = serialize_retrieved(output["retrieved"])
        if "subqueries" in output:
            record["subqueries"] = output["subqueries"]
        if "verification" in output:
            record["verification"] = output["verification"]
            record["revised"] = output.get("revised", False)
            if "original_answer" in output:
                record["original_answer"] = output["original_answer"]
        record["error"] = None
    except Exception as e:
        logger.error(f"  [{variant}] Query {q['id']} failed: {e}")
        record["answer"] = ""
        record["retrieved"] = []
        record["error"] = str(e)
    return record


def save_generations(all_records: list[dict], n_queries: int, top_k: int):
    """Write the current set of records to disk (called incrementally)."""
    payload = {
        "generated_at": datetime.now().isoformat(),
        "top_k": top_k,
        "n_variants": len({r["variant_id"] for r in all_records}),
        "n_queries": n_queries,
        "results": all_records,
    }
    GENERATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GENERATIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cmd_generate(args):
    variants_to_run = [args.variant] if args.variant else DEFAULT_VARIANTS
    queries = load_eval_queries()

    # Load whatever already exists.
    existing: list[dict] = []
    if GENERATIONS_PATH.exists():
        with GENERATIONS_PATH.open(encoding="utf-8") as f:
            existing = json.load(f).get("results", [])

    # Index already-successful records so --resume can skip them.
    ok_done = {
        (r["variant_id"], r["query_id"]): r
        for r in existing
        if not r.get("error")
    }

    # Start with records for variants we are NOT regenerating.
    all_records: list[dict] = [
        r for r in existing if r["variant_id"] not in variants_to_run
    ]

    logger.info(
        f"Generating for {variants_to_run} × {len(queries)} queries "
        f"(top_k={args.top_k}, resume={args.resume})"
    )

    for variant in variants_to_run:
        logger.info(f"\n=== Variant: {variant} ===")
        pipeline = None  # lazy init so we don't load models if everything's resumed
        for q in tqdm(queries, desc=f"  {variant}"):
            key = (variant, q["id"])
            if args.resume and key in ok_done:
                all_records.append(ok_done[key])  # keep prior success
                continue
            if pipeline is None:
                pipeline = build_pipeline(variant=variant)
            record = run_single_query(pipeline, variant, q, args.top_k)
            all_records.append(record)
            save_generations(all_records, len(queries), args.top_k)  # checkpoint

    save_generations(all_records, len(queries), args.top_k)  # final write

    # Summary
    by_variant: dict[str, dict] = {}
    for r in all_records:
        if r["variant_id"] not in variants_to_run:
            continue
        v = r["variant_id"]
        by_variant.setdefault(v, {"ok": 0, "fail": 0})
        if r.get("error"):
            by_variant[v]["fail"] += 1
        else:
            by_variant[v]["ok"] += 1

    logger.info(f"\nSaved {len(all_records)} total records to {GENERATIONS_PATH}")
    for v, counts in by_variant.items():
        logger.info(f"  {v}: {counts['ok']} OK, {counts['fail']} failed")

def compute_summary(records: list[MetricsRecord]) -> dict:
    """Per-variant means across all metrics."""
    by_variant: dict[str, list[MetricsRecord]] = defaultdict(list)
    for r in records:
        by_variant[r.variant_id].append(r)

    summary = {}
    for variant, recs in by_variant.items():
        def _mean(field: str) -> Optional[float]:
            xs = [getattr(r, field) for r in recs if getattr(r, field) is not None]
            return float(np.mean(xs)) if xs else None

        summary[variant] = {
            "context_relevance": _mean("context_relevance"),
            "answer_relevance": _mean("answer_relevance"),
            "faithfulness": _mean("faithfulness"),
            "citation_grounding": _mean("citation_grounding"),
            "hallucination_boundary": _mean("hallucination_boundary"),
            "n_records": len(recs),
            "n_errors": sum(1 for r in recs if r.error),
            "n_in_scope": sum(1 for r in recs if r.category == "in_scope"),
            "n_out_of_scope": sum(1 for r in recs if r.category == "out_of_scope"),
        }
    return summary


def print_summary_table(summary: dict):
    print()
    print("=" * 96)
    print(
        f"{'Variant':<14}{'CtxRel':>9}{'AnsRel':>9}{'Faith':>9}"
        f"{'CiteGrd':>10}{'HalluBnd':>11}{'N':>5}{'Err':>5}"
    )
    print("-" * 96)
    for v, m in sorted(summary.items()):
        def f(x): return f"{x:.3f}" if x is not None else "  N/A"
        print(
            f"{v:<14}"
            f"{f(m['context_relevance']):>9}"
            f"{f(m['answer_relevance']):>9}"
            f"{f(m['faithfulness']):>9}"
            f"{f(m['citation_grounding']):>10}"
            f"{f(m['hallucination_boundary']):>11}"
            f"{m['n_records']:>5}{m['n_errors']:>5}"
        )
    print("=" * 96)


def cmd_metrics(args):
    """Stage 2: compute metrics on saved generations."""
    if not GENERATIONS_PATH.exists():
        logger.error(f"No generations file at {GENERATIONS_PATH}. Run 'generate' first.")
        return

    with GENERATIONS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("results", [])

    if args.variant:
        records = [r for r in records if r["variant_id"] == args.variant]
    if args.limit:
        records = records[: args.limit]

    logger.info(
        f"Computing metrics for {len(records)} record(s) "
        f"using judge model: {os.getenv('GROQ_EVAL_MODEL', 'llama-3.1-8b-instant')}"
    )

    evaluator = MetricsEvaluator()
    metric_records: list[MetricsRecord] = []

    for r in tqdm(records, desc="Scoring"):
        metric_records.append(evaluator.evaluate_record(r))

    summary = compute_summary(metric_records)

    payload = {
        "computed_at": datetime.now().isoformat(),
        "judge_model": os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant"),
        "n_records": len(metric_records),
        "summary": summary,
        "per_record": [asdict(m) for m in metric_records],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metrics to {RESULTS_PATH}")
    print_summary_table(summary)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Run variants on all eval queries")
    p_gen.add_argument("--variant", choices=DEFAULT_VARIANTS, default=None,
                       help="Only run one variant (default: all three)")
    p_gen.add_argument("--top-k", type=int, default=3)
    p_gen.add_argument("--resume", action="store_true",
                       help="Skip queries that already succeeded; only run failed/missing ones")
    p_gen.set_defaults(func=cmd_generate)
    
    p_metrics = sub.add_parser("metrics", help="Compute RAGAS + custom metrics on generations")
    p_metrics.add_argument("--variant", choices=DEFAULT_VARIANTS, default=None,
                           help="Only score one variant (default: all)")
    p_metrics.add_argument("--limit", type=int, default=None,
                           help="Only score the first N records (for quick testing)")
    p_metrics.set_defaults(func=cmd_metrics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()