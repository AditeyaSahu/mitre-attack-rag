"""
src/vector_store.py

Vector store for the MITRE ATT&CK Q-A knowledge base.

Uses sentence-transformers (MiniLM) for embeddings and ChromaDB as the
persistent vector store. Embeddings are computed explicitly (not via
ChromaDB's embedding-function abstraction) so behaviour is stable across
ChromaDB versions.

We embed the QUESTION text of each Q-A pair. At query time we embed the
user's query and find the nearest questions, returning the corresponding
answer and source metadata.

Usage:
    # One-time: build the persistent index
    python -m src.vector_store build

    # Ad-hoc retrieval test
    python -m src.vector_store query "What is phishing?"
    python -m src.vector_store query "How do attackers gain initial access?" --top-k 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "mitre_attack_qa"
QA_PAIRS_PATH = Path("data/processed/qa_pairs.json")


# ---------- VectorStore ----------

class MITREVectorStore:
    """ChromaDB-backed vector store for MITRE ATT&CK Q-A pairs."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self._model: Optional[SentenceTransformer] = None

        # Persistent client — vectors survive between runs
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model so import is cheap."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "MITRE ATT&CK Enterprise Q-A pairs",
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model_name,
            },
        )

    def reset(self):
        """Delete and recreate the collection (clean rebuild)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass
        return self._get_or_create_collection()

    def build(self, qa_pairs: list[dict], batch_size: int = 64, reset: bool = True):
        """
        Embed Q-A questions and load them into ChromaDB.

        Each entry is keyed by its qa-pair id; the question is embedded
        and the answer + source metadata is attached for retrieval.
        """
        collection = self.reset() if reset else self._get_or_create_collection()

        questions = [p["question"] for p in qa_pairs]
        logger.info(f"Encoding {len(questions)} questions (batch_size={batch_size})...")
        embeddings = self.model.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        ids = [p["id"] for p in qa_pairs]
        metadatas = [
            {
                "answer": p["answer"],
                "source_id": p["source_id"],
                "source_type": p["source_type"],
                "source_url": p["source_url"],
                "qa_type": p["qa_type"],
                "question": p["question"],
            }
            for p in qa_pairs
        ]

        # Insert in chunks to stay under ChromaDB's per-batch limit
        chunk = 1000
        for i in tqdm(range(0, len(ids), chunk), desc="Indexing"):
            collection.add(
                ids=ids[i:i + chunk],
                documents=questions[i:i + chunk],
                embeddings=embeddings[i:i + chunk].tolist(),
                metadatas=metadatas[i:i + chunk],
            )

        logger.info(f"Indexed {collection.count()} entries in '{self.collection_name}'")

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        """Retrieve the top-k most similar Q-A pairs for a query string."""
        collection = self._get_or_create_collection()
        query_embedding = self.model.encode([text], convert_to_numpy=True)[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        out: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return out

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            out.append({
                "rank": i + 1,
                "id": results["ids"][0][i],
                "question": results["documents"][0][i],
                "answer": meta["answer"],
                "source_id": meta["source_id"],
                "source_type": meta["source_type"],
                "source_url": meta["source_url"],
                "qa_type": meta["qa_type"],
                "distance": results["distances"][0][i],
            })
        return out


# ---------- CLI ----------

def cmd_build(args):
    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        logger.error(f"Q-A pairs file not found: {qa_path}")
        logger.error("Run `python -m src.preprocessing` first.")
        sys.exit(1)

    with qa_path.open(encoding="utf-8") as f:
        pairs = json.load(f)

    store = MITREVectorStore()
    store.build(pairs, reset=True)


def cmd_query(args):
    store = MITREVectorStore()
    results = store.query(args.text, top_k=args.top_k)

    print(f"\nQuery: {args.text!r}")
    print(f"Top {len(results)} results:")
    print("=" * 80)
    for r in results:
        print(f"\n[Rank {r['rank']}] distance={r['distance']:.4f}  qa_type={r['qa_type']}")
        print(f"  Question:    {r['question']}")
        ans_preview = r["answer"][:280] + ("..." if len(r["answer"]) > 280 else "")
        print(f"  Answer:      {ans_preview}")
        print(f"  Source:      {r['source_id']} ({r['source_type']}) — {r['source_url']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the vector index from qa_pairs.json")
    p_build.add_argument("--qa-path", default=str(QA_PAIRS_PATH))
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Test a query against the index")
    p_query.add_argument("text", help="Query string")
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()