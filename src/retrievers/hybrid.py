"""
src/retrievers/hybrid.py

Variant 2 — Hybrid RAG.

Pipeline:
    1. Dense retrieval (top-N from ChromaDB)
    2. Sparse retrieval (top-N from BM25)
    3. Reciprocal Rank Fusion to merge the two rankings
    4. Cross-encoder re-ranking on the fused top-M candidates
    5. Return top-k

Each stage is justified in the report; the variant exists to demonstrate
that hybrid retrieval + re-ranking beats dense-only on cybersecurity
queries that mix natural language with literal MITRE identifiers.
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src.retrievers.base import BaseRetriever
from src.vector_store import MITREVectorStore

load_dotenv()
logger = logging.getLogger(__name__)

QA_PAIRS_PATH = Path("data/processed/qa_pairs.json")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# RRF smoothing constant — 60 is the value from the original Cormack et al
# RRF paper and is the conventional default.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token extraction for BM25 indexing."""
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009).

    Given multiple ranked lists of document IDs, produce a combined score
    per document where higher is better. Robust to different score scales
    across retrievers — uses only rank position.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: BM25 + dense fusion + cross-encoder re-ranking.
    Implements the BaseRetriever contract.
    """

    def __init__(
        self,
        qa_pairs_path: Path = QA_PAIRS_PATH,
        vector_store: Optional[MITREVectorStore] = None,
        reranker_model_name: str = RERANKER_MODEL,
    ):
        self.vector_store = vector_store or MITREVectorStore()
        self.reranker_model_name = reranker_model_name
        self._reranker = None

        # Load Q-A pairs and build BM25 index in memory
        with open(qa_pairs_path, encoding="utf-8") as f:
            self.qa_pairs: list[dict] = json.load(f)
        self.qa_by_id: dict[str, dict] = {p["id"]: p for p in self.qa_pairs}
        self.id_list: list[str] = [p["id"] for p in self.qa_pairs]

        # Index question + answer combined — answers contain MITRE IDs and
        # technical terms that BM25 specifically helps catch.
        logger.info(f"Building BM25 index over {len(self.qa_pairs)} Q-A pairs")
        tokenized_corpus = [
            tokenize(p["question"] + " " + p["answer"]) for p in self.qa_pairs
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @property
    def name(self) -> str:
        return "hybrid_rag"

    @property
    def reranker(self):
        """Lazy-load the cross-encoder on first use."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder re-ranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        fusion_n: int = 20,
        rerank_n: int = 10,
    ) -> list[dict]:
        """
        End-to-end retrieval:
          1. Dense top-N
          2. Sparse (BM25) top-N
          3. RRF fusion -> top-rerank_n candidates
          4. Cross-encoder re-rank -> top-k
        """
        # 1. Dense
        dense_results = self.vector_store.query(query, top_k=fusion_n)
        dense_ids = [r["id"] for r in dense_results]

        # 2. Sparse (BM25)
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25 = np.argsort(bm25_scores)[::-1][:fusion_n]
        sparse_ids = [self.id_list[i] for i in top_bm25]

        # 3. RRF fusion
        fused_scores = reciprocal_rank_fusion([dense_ids, sparse_ids])
        candidate_ids = sorted(
            fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True
        )[:rerank_n]
        candidates = [self.qa_by_id[i] for i in candidate_ids if i in self.qa_by_id]

        if not candidates:
            return []

        # 4. Cross-encoder rerank — score (query, question+answer) pairs.
        # Cap answer to 500 chars so the cross-encoder isn't overwhelmed.
        rerank_pairs = [
            (query, c["question"] + " " + c["answer"][:500]) for c in candidates
        ]
        rerank_scores = self.reranker.predict(rerank_pairs)

        # 5. Final ranking — sort by rerank score, take top-k
        ranked = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[dict] = []
        for rank, (c, score) in enumerate(ranked, start=1):
            results.append({
                "rank": rank,
                "id": c["id"],
                "question": c["question"],
                "answer": c["answer"],
                "source_id": c["source_id"],
                "source_type": c["source_type"],
                "source_url": c["source_url"],
                "qa_type": c["qa_type"],
                # Use 1 - rerank_score as a pseudo-"distance" so the
                # downstream pipeline can keep using one field name.
                "distance": float(1.0 - score),
                "rerank_score": float(score),
                "fused_score": float(fused_scores.get(c["id"], 0.0)),
            })
        return results