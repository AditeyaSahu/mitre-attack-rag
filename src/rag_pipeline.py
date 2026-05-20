"""
src/rag_pipeline.py

Orchestrates the RAG pipeline: retrieve → format context → generate → return.

The pipeline is retriever-agnostic — pass any BaseRetriever subclass
(Naive, Hybrid, Agentic) and it just works.

The system prompt is engineered to:
  1. Force grounding in retrieved context (refuse if insufficient)
  2. Require MITRE ID citation (transparency — Criterion 4)
  3. Produce actionable practitioner guidance (assignment objective)

Usage:
    # Single query
    python -m src.rag_pipeline ask "How do I defend against phishing?"

    # With variant override (defaults to naive)
    python -m src.rag_pipeline ask "What is T1059?" --variant naive --top-k 5
"""

import argparse
import json
import logging
from typing import Optional

from src.generator import GroqGenerator
from src.retrievers.base import BaseRetriever
from src.retrievers.naive import NaiveRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.agentic import AgenticRetriever, AnswerVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Prompts ----------

DEFAULT_SYSTEM_PROMPT = """You are a cybersecurity expert assistant specialising in the MITRE ATT&CK Enterprise framework. Your task is to generate clear, accurate, actionable guidelines for cybersecurity practitioners by grounding every claim in the retrieved knowledge provided in context.

Strict rules:
1. Ground every factual claim in the retrieved context. Do not introduce facts that are not in the context.
2. Always cite the relevant MITRE identifier(s) (T-IDs for techniques, M-IDs for mitigations, TA-IDs for tactics, DET-IDs for detection strategies) when referring to a specific MITRE entry.
3. If the retrieved context does not contain enough information to answer the question, explicitly say so — do not guess and do not fall back on general knowledge.
4. Structure answers for practitioners: lead with the direct answer, follow with actionable detail, end with citations.
5. Be concise. Definitions: 2-4 sentences. How-to and mitigation answers: bullet list of concrete steps, each with its citation.
6. Never invent MITRE IDs. Only cite IDs that appear in the retrieved context."""

DEFAULT_USER_TEMPLATE = """[Retrieved MITRE ATT&CK context]
{context}

[Practitioner question]
{query}

Provide a clear, actionable answer grounded strictly in the context above. Cite MITRE IDs explicitly. If context is insufficient, say so."""


def format_context(retrieved: list[dict]) -> str:
    """Render retrieved Q-A pairs as numbered context blocks for the prompt."""
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(
            f"--- Source {i} ---\n"
            f"MITRE ID: {r['source_id']} ({r['source_type']})\n"
            f"URL: {r['source_url']}\n"
            f"Indexed question: {r['question']}\n"
            f"Indexed answer: {r['answer']}"
        )
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


# ---------- Pipeline ----------

class RAGPipeline:
    """Generic retrieve-then-generate pipeline."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: GroqGenerator,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or DEFAULT_USER_TEMPLATE

    def answer(self, query: str, top_k: int = 3) -> dict:
        # 1. Retrieve
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # 2. Build context
        context = format_context(retrieved)

        # 3. Format prompts
        user_prompt = self.user_template.format(context=context, query=query)

        # 4. Generate
        answer_text = self.generator.generate(self.system_prompt, user_prompt)

        return {
            "query": query,
            "variant": self.retriever.name,
            "answer": answer_text,
            "retrieved": retrieved,
            "top_k": top_k,
        }

    def answer_batch(self, queries: list[str], top_k: int = 3) -> list[dict]:
        return [self.answer(q, top_k) for q in queries]


# ---------- Agentic pipeline with verify + revise loop ----------

REVISION_SYSTEM_PROMPT = """You are revising a cybersecurity answer after a faithfulness review flagged unsupported claims. Produce a revised answer that:
- Removes or explicitly hedges every flagged claim.
- Keeps only statements grounded in the retrieved MITRE ATT&CK context.
- States plainly when the retrieved context is insufficient to answer a sub-part of the question.
- Preserves all correctly grounded MITRE ID citations from the original answer.
- Never invents new facts or new MITRE identifiers."""

REVISION_USER_TEMPLATE = """[ORIGINAL PRACTITIONER QUESTION]
{query}

[RETRIEVED MITRE ATT&CK CONTEXT]
{context}

[ORIGINAL ANSWER]
{answer}

[VERIFIER REASON]
{reason}

[UNSUPPORTED CLAIMS FLAGGED]
{flagged}

Produce the revised, faithfully grounded answer below."""


class AgenticRAGPipeline(RAGPipeline):
    """
    Pipeline extension for Variant 3. Adds a self-verification step after
    generation, and a single revision pass if verification fails.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: GroqGenerator,
        verifier: Optional[AnswerVerifier] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        super().__init__(retriever, generator, system_prompt, user_template)
        self.verifier = verifier or AnswerVerifier()

    def answer(self, query: str, top_k: int = 3) -> dict:
        # Run the standard retrieve-then-generate pipeline first.
        result = super().answer(query, top_k=top_k)
        context = format_context(result["retrieved"])

        # Surface the decomposition (if the retriever supports it).
        if hasattr(self.retriever, "decompose"):
            try:
                result["subqueries"] = self.retriever.decompose(query)
            except Exception:
                result["subqueries"] = [query]

        # Self-verify
        verdict = self.verifier.verify(query, result["answer"], context)
        result["verification"] = {
            "status": verdict.status,
            "reason": verdict.reason,
            "unsupported_claims": verdict.unsupported_claims,
        }

        # Conditional revision
        if verdict.status == "FAIL":
            flagged = "\n".join(f"- {c}" for c in verdict.unsupported_claims) or "(none listed)"
            revision_prompt = REVISION_USER_TEMPLATE.format(
                query=query,
                context=context,
                answer=result["answer"],
                reason=verdict.reason,
                flagged=flagged,
            )
            revised = self.generator.generate(REVISION_SYSTEM_PROMPT, revision_prompt)
            result["original_answer"] = result["answer"]
            result["answer"] = revised
            result["revised"] = True
        else:
            result["revised"] = False

        return result

# ---------- Variant factory ----------

def build_pipeline(variant: str = "naive") -> RAGPipeline:
    """Construct a RAGPipeline (or AgenticRAGPipeline) for the requested variant."""
    generator = GroqGenerator()
    if variant == "naive":
        return RAGPipeline(retriever=NaiveRetriever(), generator=generator)
    elif variant == "hybrid":
        return RAGPipeline(retriever=HybridRetriever(), generator=generator)
    elif variant == "agentic":
        return AgenticRAGPipeline(
            retriever=AgenticRetriever(),
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")


# ---------- CLI ----------

def cmd_ask(args):
    pipeline = build_pipeline(variant=args.variant)
    result = pipeline.answer(args.query, top_k=args.top_k)

    print("=" * 80)
    print(f"Variant: {result['variant']}    top_k: {result['top_k']}")
    print(f"Query:   {result['query']}")
    print("-" * 80)

    # Agentic-only: show decomposition
    if "subqueries" in result and result["subqueries"] != [result["query"]]:
        print("Sub-queries (from decomposition):")
        for i, sq in enumerate(result["subqueries"], start=1):
            print(f"  [{i}] {sq}")
        print("-" * 80)

    print("Retrieved sources:")
    for i, r in enumerate(result["retrieved"], start=1):
        print(f"  [{i}] {r['source_id']} ({r['qa_type']}) — distance={r['distance']:.4f}")
        print(f"      {r['question']}")
    print("-" * 80)

    # Agentic-only: show verification verdict
    if "verification" in result:
        v = result["verification"]
        print(f"Verification: {v['status']} — {v['reason']}")
        if v["unsupported_claims"]:
            print("Flagged claims:")
            for c in v["unsupported_claims"]:
                print(f"  - {c}")
        print(f"Revised: {result.get('revised', False)}")
        print("-" * 80)

    print("Generated answer:")
    print(result["answer"])
    print("=" * 80)

    if args.json:
        print("\nFull result as JSON:")
        printable = {**result, "retrieved": [dict(r) for r in result["retrieved"]]}
        print(json.dumps(printable, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Ask a single question via the RAG pipeline")
    p_ask.add_argument("query", help="The practitioner question to answer")
    p_ask.add_argument("--variant", default="naive", choices=["naive", "hybrid", "agentic"],
                       help="Which retrieval variant to use (default: naive)")
    p_ask.add_argument("--top-k", type=int, default=3)
    p_ask.add_argument("--json", action="store_true",
                       help="Also print the full result as JSON")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()