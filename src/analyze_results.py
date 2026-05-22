"""
src/analyze_results.py

Step 15 — Error analysis and visualisation.

Reads:
  data/eval/results.json      (per-record metrics from Stage 2)
  data/eval/generations.json  (raw answers, for qualitative failure examples)

Produces:
  reports/figures/metrics_by_variant.png      grouped bar chart
  reports/figures/citation_by_type.png         per-query-type citation grounding
  reports/figures/faithfulness_tradeoff.png     faithfulness vs context relevance
  data/eval/error_analysis.md                   structured written analysis

Usage:
    python -m src.analyze_results
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("data/eval/results.json")
GENERATIONS_PATH = Path("data/eval/generations.json")
FIGURES_DIR = Path("reports/figures")
ANALYSIS_PATH = Path("data/eval/error_analysis.md")

METRIC_LABELS = {
    "context_relevance": "Context Relevance",
    "answer_relevance": "Answer Relevance",
    "faithfulness": "Faithfulness",
    "citation_grounding": "Citation Grounding",
    "hallucination_boundary": "Hallucination Boundary",
}
VARIANT_ORDER = ["naive", "hybrid", "agentic"]
VARIANT_COLORS = {"naive": "#4C72B0", "hybrid": "#55A868", "agentic": "#C44E52"}


def load_data():
    with RESULTS_PATH.open(encoding="utf-8") as f:
        results = json.load(f)
    with GENERATIONS_PATH.open(encoding="utf-8") as f:
        generations = json.load(f)
    return results, generations


# ---------- Aggregation ----------

def per_type_breakdown(per_record: list[dict]) -> dict:
    """Mean of each metric, grouped by (variant, query_type)."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in per_record:
        v = r["variant_id"]
        t = r["type"]
        for metric in METRIC_LABELS:
            val = r.get(metric)
            if val is not None:
                groups[(v, t)][metric].append(val)

    out = {}
    for (v, t), metrics in groups.items():
        out[(v, t)] = {m: float(np.mean(vals)) for m, vals in metrics.items()}
    return out


def worst_records(per_record: list[dict], generations: dict, metric: str, n: int = 5):
    """Return the n lowest-scoring records on a metric, with their answers."""
    gen_lookup = {
        (g["variant_id"], g["query_id"]): g
        for g in generations["results"]
    }
    scored = [
        r for r in per_record
        if r.get(metric) is not None
    ]
    scored.sort(key=lambda r: r[metric])
    out = []
    for r in scored[:n]:
        gen = gen_lookup.get((r["variant_id"], r["query_id"]), {})
        out.append({
            "variant": r["variant_id"],
            "query_id": r["query_id"],
            "type": r["type"],
            "score": r[metric],
            "query": gen.get("query", ""),
            "answer": gen.get("answer", "")[:400],
        })
    return out


# ---------- Figures ----------

def fig_metrics_by_variant(summary: dict, path: Path):
    """Grouped bar chart: metrics on x-axis, one bar per variant."""
    metrics = list(METRIC_LABELS.keys())
    variants = [v for v in VARIANT_ORDER if v in summary]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, v in enumerate(variants):
        vals = [summary[v].get(m) or 0.0 for m in metrics]
        ax.bar(x + i * width, vals, width, label=v, color=VARIANT_COLORS.get(v))
        for xi, val in zip(x + i * width, vals):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAG Variant Comparison Across Evaluation Metrics")
    ax.legend(title="Variant")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_citation_by_type(breakdown: dict, path: Path):
    """Heatmap-style grouped bars: citation grounding per query type per variant."""
    # in-scope types only (citation grounding is N/A for OOS)
    types = sorted({t for (v, t) in breakdown if "citation_grounding" in breakdown[(v, t)]})
    variants = [v for v in VARIANT_ORDER if any((v, t) in breakdown for t in types)]
    x = np.arange(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(variants):
        vals = [breakdown.get((v, t), {}).get("citation_grounding", 0.0) for t in types]
        ax.bar(x + i * width, vals, width, label=v, color=VARIANT_COLORS.get(v))

    ax.set_xticks(x + width)
    ax.set_xticklabels(types, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Citation Grounding (0–1)")
    ax.set_title("Citation Grounding by Query Type and Variant")
    ax.legend(title="Variant")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_faithfulness_tradeoff(summary: dict, path: Path):
    """Paired bars: Context Relevance vs Faithfulness, illustrating the tradeoff."""
    variants = [v for v in VARIANT_ORDER if v in summary]
    x = np.arange(len(variants))
    width = 0.35

    ctx = [summary[v].get("context_relevance") or 0.0 for v in variants]
    faith = [summary[v].get("faithfulness") or 0.0 for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ctx, width, label="Context Relevance", color="#4C72B0")
    ax.bar(x + width / 2, faith, width, label="Faithfulness", color="#C44E52")
    for xi, val in zip(x - width / 2, ctx):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    for xi, val in zip(x + width / 2, faith):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Context Relevance vs Faithfulness Tradeoff")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ---------- Markdown analysis ----------

def write_analysis_md(summary, breakdown, generations, per_record, path: Path):
    lines = []
    lines.append("# Error Analysis and Results Discussion\n")
    lines.append("Auto-generated from `results.json` and `generations.json`. "
                 "Figures are in `reports/figures/`.\n")

    # Overall table
    lines.append("## Overall metric comparison\n")
    lines.append("| Variant | Context Rel. | Answer Rel. | Faithfulness | Citation Grd. | Hallu. Boundary |")
    lines.append("|---|---|---|---|---|---|")
    for v in VARIANT_ORDER:
        if v not in summary:
            continue
        s = summary[v]
        def f(m): 
            x = s.get(m)
            return f"{x:.3f}" if x is not None else "N/A"
        lines.append(
            f"| {v} | {f('context_relevance')} | {f('answer_relevance')} | "
            f"{f('faithfulness')} | {f('citation_grounding')} | {f('hallucination_boundary')} |"
        )
    lines.append("")

    # Agentic revision stats
    agentic_gens = [g for g in generations["results"] if g["variant_id"] == "agentic"]
    revised = sum(1 for g in agentic_gens if g.get("revised"))
    decomposed = sum(
        1 for g in agentic_gens
        if g.get("subqueries") and len(g["subqueries"]) > 1
    )
    lines.append("## Agentic behaviour\n")
    lines.append(f"- Queries decomposed into multiple sub-queries: **{decomposed}/{len(agentic_gens)}**")
    lines.append(f"- Answers revised after failed self-verification: **{revised}/{len(agentic_gens)}**\n")

    # Worst records per key metric
    for metric in ["faithfulness", "context_relevance", "citation_grounding"]:
        lines.append(f"## Lowest-scoring records: {METRIC_LABELS[metric]}\n")
        worst = worst_records(per_record, generations, metric, n=5)
        for w in worst:
            lines.append(f"- **{w['variant']}** / `{w['query_id']}` ({w['type']}) — "
                         f"score {w['score']:.3f}")
            lines.append(f"  - Query: {w['query']}")
            lines.append(f"  - Answer (truncated): {w['answer']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {path}")


# ---------- Main ----------

def main():
    results, generations = load_data()
    summary = results["summary"]
    per_record = results["per_record"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)

    breakdown = per_type_breakdown(per_record)

    fig_metrics_by_variant(summary, FIGURES_DIR / "metrics_by_variant.png")
    fig_citation_by_type(breakdown, FIGURES_DIR / "citation_by_type.png")
    fig_faithfulness_tradeoff(summary, FIGURES_DIR / "faithfulness_tradeoff.png")

    write_analysis_md(summary, breakdown, generations, per_record, ANALYSIS_PATH)

    logger.info("Error analysis complete.")


if __name__ == "__main__":
    main()