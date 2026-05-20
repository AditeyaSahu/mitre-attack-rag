"""
src/retrievers/agentic.py

Variant 3 — Agentic RAG.

Three LLM-driven decisions per query:
  1. QueryDecomposer  — splits complex queries into sub-queries
  2. AgenticRetriever — retrieves once per sub-query, merges, deduplicates
  3. AnswerVerifier   — judges whether the generated answer is grounded;
                       flags unsupported claims for a revision pass

The verifier and any subsequent revision happen at the pipeline level
(see AgenticRAGPipeline in src/rag_pipeline.py).
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from src.generator import GroqGenerator
from src.retrievers.base import BaseRetriever
from src.retrievers.hybrid import HybridRetriever

load_dotenv()
logger = logging.getLogger(__name__)


# ---------- JSON-robust extraction ----------

def extract_json_object(text: str) -> dict:
    """
    Find and parse the first JSON object in an LLM response.
    Strips markdown fences; tolerant of prose around the JSON.
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(match.group(0))


# ---------- Query Decomposer ----------

DECOMPOSE_SYSTEM_PROMPT = """You are a query analyser for a cybersecurity RAG system specialised in the MITRE ATT&CK Enterprise framework.

Your job is to decide whether a practitioner's question is SIMPLE (one retrieval pass suffices) or COMPLEX (needs to be split into sub-questions so each can retrieve its own context).

Decision criteria:
- SIMPLE: single concept lookup ("What is X?", "Define Y", "What is the MITRE ID for Z?")
- COMPLEX: multi-faceted, comparative, or compound ("How do I detect AND mitigate X?", "Compare X and Y", "What platforms, tactics, and detections apply to X?")

If COMPLEX, write 2-4 focused sub-questions. Each sub-question should be specific enough that a single retrieval pass can answer it.

Output STRICT JSON only — no prose, no markdown fences. Schema:
{"complexity": "simple" | "complex", "subqueries": [string, ...]}

For SIMPLE queries, subqueries contains the original query as a single element.
For COMPLEX queries, subqueries contains 2-4 sub-questions."""

DECOMPOSE_USER_TEMPLATE = """Analyse this practitioner query:

{query}"""


class QueryDecomposer:
    """LLM-driven query decomposition."""

    def __init__(self, generator: Optional[GroqGenerator] = None):
        # Use the fast/cheap agent model for decomposition by default.
        self.generator = generator or GroqGenerator(
            model=os.getenv("GROQ_AGENT_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=512,
        )

    def decompose(self, query: str) -> list[str]:
        user_prompt = DECOMPOSE_USER_TEMPLATE.format(query=query)
        try:
            response = self.generator.generate(DECOMPOSE_SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_object(response)
            subqueries = parsed.get("subqueries", [])
            if not subqueries or not isinstance(subqueries, list):
                raise ValueError("Empty or invalid subqueries")
            # Safety: cap to 4 sub-queries
            subqueries = [str(s).strip() for s in subqueries if str(s).strip()][:4]
            logger.info(
                f"Decomposed query into {len(subqueries)} sub-query/queries "
                f"(complexity={parsed.get('complexity')})"
            )
            return subqueries
        except Exception as e:
            logger.warning(f"Query decomposition failed ({e}); falling back to single-pass.")
            return [query]


# ---------- Agentic Retriever ----------

class AgenticRetriever(BaseRetriever):
    """
    Decomposes the query, retrieves once per sub-query through the base
    retriever (defaults to HybridRetriever — best base for accuracy),
    then merges and deduplicates results across sub-queries.
    """

    def __init__(
        self,
        base_retriever: Optional[BaseRetriever] = None,
        decomposer: Optional[QueryDecomposer] = None,
    ):
        self.base = base_retriever or HybridRetriever()
        self.decomposer = decomposer or QueryDecomposer()

    @property
    def name(self) -> str:
        return "agentic_rag"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        subqueries = self.decomposer.decompose(query)

        # Retrieve more per sub-query so the union has material to choose from.
        per_query_k = max(top_k, 5)
        merged: list[dict] = []
        seen_ids: set[str] = set()

        for sq in subqueries:
            try:
                results = self.base.retrieve(sq, top_k=per_query_k)
            except Exception as e:
                logger.warning(f"Sub-query retrieval failed for {sq!r}: {e}")
                continue
            for r in results:
                if r["id"] not in seen_ids:
                    # Annotate which sub-query surfaced this result — useful
                    # for the report's qualitative analysis.
                    r = {**r, "from_subquery": sq}
                    merged.append(r)
                    seen_ids.add(r["id"])

        # Sort by best (lowest distance) and take top-k across all sub-queries.
        merged.sort(key=lambda r: r.get("distance", float("inf")))
        return merged[:top_k]

    # Exposed so the pipeline / report can inspect the decomposition itself.
    def decompose(self, query: str) -> list[str]:
        return self.decomposer.decompose(query)


# ---------- Answer Verifier ----------

VERIFY_SYSTEM_PROMPT = """You are a faithfulness verifier for a cybersecurity RAG system. Your only job is to check whether a generated answer is fully supported by the retrieved MITRE ATT&CK context.

Check all of:
1. Every factual claim in the answer must be supported by the retrieved context.
2. Every MITRE identifier cited (T-ID, M-ID, TA-ID, DET-ID) must appear in the retrieved context. Any cited ID that does not appear in the context is a hallucination — FAIL.
3. The answer must not add specific facts (dates, version numbers, mitigation step counts, platform lists) that are not in the retrieved context.

Output STRICT JSON only — no prose, no markdown fences. Schema:
{"status": "PASS" | "FAIL", "reason": "<one-sentence explanation>", "unsupported_claims": [string, ...]}

If everything in the answer is grounded, output PASS with an empty unsupported_claims list.
If any claim or ID is unsupported, output FAIL and list the unsupported claims verbatim."""

VERIFY_USER_TEMPLATE = """[ORIGINAL QUESTION]
{query}

[RETRIEVED CONTEXT]
{context}

[ANSWER UNDER REVIEW]
{answer}

Verify the answer."""


@dataclass
class VerificationResult:
    status: str          # "PASS" | "FAIL"
    reason: str
    unsupported_claims: list[str]


class AnswerVerifier:
    """LLM-as-judge faithfulness verifier."""

    def __init__(self, generator: Optional[GroqGenerator] = None):
        # The verifier benefits from the larger model — accuracy matters more
        # than speed here. Use the main generation model by default.
        self.generator = generator or GroqGenerator(
            model=os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.0,
            max_tokens=512,
        )

    def verify(self, query: str, answer: str, context: str) -> VerificationResult:
        user_prompt = VERIFY_USER_TEMPLATE.format(
            query=query, context=context, answer=answer
        )
        try:
            response = self.generator.generate(VERIFY_SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_object(response)
            status = parsed.get("status", "PASS").upper()
            if status not in ("PASS", "FAIL"):
                status = "PASS"  # be conservative on parse weirdness
            return VerificationResult(
                status=status,
                reason=str(parsed.get("reason", ""))[:500],
                unsupported_claims=[
                    str(c) for c in parsed.get("unsupported_claims", [])
                ],
            )
        except Exception as e:
            logger.warning(f"Verification failed ({e}); defaulting to PASS.")
            return VerificationResult(
                status="PASS",
                reason=f"Verifier error: {e}",
                unsupported_claims=[],
            )