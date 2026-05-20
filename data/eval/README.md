# Evaluation Query Set — Design Rationale

This file documents the design of `eval_queries.json`, the fixed evaluation
set used to measure all three RAG variants (Naive, Hybrid, Agentic).

## Composition

| Category      | Count | Purpose |
|---------------|-------|---------|
| In-scope      | 15    | Measure standard RAG metrics (Context Relevance, Answer Relevance, Faithfulness) and the custom Citation Grounding metric. |
| Out-of-scope  | 5     | Measure the custom Hallucination Boundary metric — does the system refuse questions whose answers are not in the knowledge base, instead of generating confident but ungrounded text? |
| **Total**     | **20**| 2× the assignment's "e.g., 10 queries" suggestion, for statistical signal across three variants. |

## In-scope query type coverage

Every major Q-A pattern in the knowledge base is exercised by at least one query:

- `definition` (Q1) — basic concept lookup
- `mitigation` (Q2) — "how do I defend" practitioner queries
- `detection` (Q3) — "how do I detect" practitioner queries
- `platforms` (Q4) — target system information
- `id_lookup` (Q5) — exact MITRE ID retrieval (BM25's strength)
- `sub_technique_listing` (Q6) — parent → children navigation
- `mitigation_lookup` (Q7) — M-ID resolution
- `tactic_mapping` (Q8) — technique → tactic relationships
- `how` (Q9) — mechanism / process explanation
- `procedures` (Q10) — real-world adversary examples
- `tactic_goal` (Q11) — tactic-level definition
- `compound` (Q12, Q14) — multi-faceted ("detect AND mitigate") queries that should trigger Agentic decomposition
- `comparative` (Q13) — side-by-side comparison
- `detection_tactic_level` (Q15) — broader tactic-level monitoring queries

## Out-of-scope query selection

The five OOS queries span unrelated domains (geography, cooking, demographics,
sports, opinion) to ensure the Hallucination Boundary metric is not gameable
by topic-level heuristics. A system that recognises one but answers the others
fails the metric.

## Expected fields

Each query record includes:

- `id`: stable identifier for reproducibility
- `query`: the practitioner question text
- `category`: `in_scope` or `out_of_scope`
- `type`: fine-grained query category (for per-type breakdown in results)
- `expected_source_ids`: MITRE IDs that a correct retrieval should surface.
  Used by the **Citation Grounding** custom metric (did the generated answer
  cite at least one of these IDs?).
- `expected_behavior`: human-readable description of the correct response,
  used in qualitative error analysis.
