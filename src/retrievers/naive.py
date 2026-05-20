"""
src/retrievers/naive.py

Variant 1 — Naive RAG.

Pure dense retrieval against the ChromaDB vector store, no re-ranking, no
sparse fusion, no agent loop. This is the baseline against which the
Hybrid and Agentic variants are evaluated.
"""

from src.retrievers.base import BaseRetriever
from src.vector_store import MITREVectorStore


class NaiveRetriever(BaseRetriever):
    """Dense-only retrieval using MiniLM embeddings + ChromaDB."""

    def __init__(self, vector_store: MITREVectorStore = None):
        self.vector_store = vector_store or MITREVectorStore()

    @property
    def name(self) -> str:
        return "naive_rag"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        return self.vector_store.query(query, top_k=top_k)