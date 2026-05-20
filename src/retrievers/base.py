"""
src/retrievers/base.py

Abstract retriever interface. All three retrieval variants (Naive, Hybrid,
Agentic) implement this contract so the rest of the RAG pipeline is
retriever-agnostic.
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Contract every retriever must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier (used in logs and evaluation)."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return a list of retrieved Q-A pair dicts, ranked best-first.

        Each dict must contain at minimum:
            id, question, answer, source_id, source_type, source_url, qa_type
        Plus a numeric relevance signal under some key (distance or score).
        """
        ...