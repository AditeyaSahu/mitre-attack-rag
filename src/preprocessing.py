"""
src/preprocessing.py

Generate question-answer pairs from the raw MITRE ATT&CK scraper output.

For each tactic, technique, and sub-technique, produce multiple Q-A pairs
covering different practitioner-style query patterns (definition, ID
lookup, "how does it work", parent/child relationships, sub-technique
listings). Each pair retains source metadata so retrieval results can be
cited back to the original MITRE entry.

Usage:
    python -m src.preprocessing
"""

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass
class QAPair:
    """One Q-A pair derived from the MITRE ATT&CK knowledge base."""
    id: str               # unique pair id, e.g. "qa_T1548_def"
    question: str
    answer: str
    source_id: str        # MITRE entity id this pair was derived from
    source_type: str      # "tactic" | "technique" | "sub-technique"
    source_url: str
    qa_type: str          # "definition" | "id_lookup" | "how" | "parent" | "sub_techniques" | "goal"


# ---------- Helpers ----------

def first_sentence(text: str, max_words: int = 30) -> str:
    """Extract the first sentence (capped at max_words) from a description."""
    if not text:
        return ""
    m = re.search(r"^(.+?[.!?])(\s|$)", text.strip())
    sentence = m.group(1) if m else text.strip()
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]) + "..."
    return sentence


def safe_id(raw_id: str) -> str:
    """Convert a MITRE id to a filename-safe id token (replace dots)."""
    return raw_id.replace(".", "_")


# ---------- Q-A generators ----------

def generate_tactic_qa(tactic: dict) -> list[QAPair]:
    """Three Q-A pairs per tactic: definition, ID lookup, goal."""
    tid = tactic["id"]
    name = tactic["name"]
    desc = tactic["description"]
    url = tactic["url"]

    return [
        QAPair(
            id=f"qa_{tid}_def",
            question=f"What is the {name} tactic in MITRE ATT&CK?",
            answer=f"{name} ({tid}) is a MITRE ATT&CK Enterprise tactic. {desc}",
            source_id=tid,
            source_type="tactic",
            source_url=url,
            qa_type="definition",
        ),
        QAPair(
            id=f"qa_{tid}_id",
            question=f"What does the MITRE ATT&CK tactic {tid} represent?",
            answer=f"Tactic {tid} is {name}: {desc}",
            source_id=tid,
            source_type="tactic",
            source_url=url,
            qa_type="id_lookup",
        ),
        QAPair(
            id=f"qa_{tid}_goal",
            question=f"What is the adversary's goal in the {name} tactic?",
            answer=f"The adversary's goal in the {name} tactic ({tid}) is: {desc}",
            source_id=tid,
            source_type="tactic",
            source_url=url,
            qa_type="goal",
        ),
    ]


def generate_technique_qa(
    tech: dict,
    parent_tech: Optional[dict] = None,
    all_techniques: Optional[list[dict]] = None,
) -> list[QAPair]:
    """Multiple Q-A pairs per technique/sub-technique."""
    tid = tech["id"]
    name = tech["name"]
    desc = tech["description"]
    url = tech["url"]
    is_sub = tech.get("is_sub_technique", False)
    source_type = "sub-technique" if is_sub else "technique"
    sid = safe_id(tid)

    pairs: list[QAPair] = [
        # Definition — most natural practitioner query
        QAPair(
            id=f"qa_{sid}_def",
            question=f"What is {name}?",
            answer=f"{name} ({tid}) is a MITRE ATT&CK Enterprise {source_type}. {desc}",
            source_id=tid,
            source_type=source_type,
            source_url=url,
            qa_type="definition",
        ),
        # ID lookup — catches queries phrased around the MITRE ID directly
        QAPair(
            id=f"qa_{sid}_id",
            question=f"What is MITRE ATT&CK {source_type} {tid}?",
            answer=f"{tid} is {name}. {desc}",
            source_id=tid,
            source_type=source_type,
            source_url=url,
            qa_type="id_lookup",
        ),
        # "How does X work" — practitioner-style behavioural query
        QAPair(
            id=f"qa_{sid}_how",
            question=f"How does the {name} technique work?",
            answer=f"In MITRE ATT&CK, {name} ({tid}) is described as follows: {desc}",
            source_id=tid,
            source_type=source_type,
            source_url=url,
            qa_type="how",
        ),
    ]

    # Sub-technique only: link back to its parent
    if is_sub and parent_tech:
        pairs.append(QAPair(
            id=f"qa_{sid}_parent",
            question=f"What is the parent technique of {name} ({tid})?",
            answer=(
                f"The parent technique of {name} ({tid}) is "
                f"{parent_tech['name']} ({parent_tech['id']}). "
                f"{parent_tech['description']}"
            ),
            source_id=tid,
            source_type=source_type,
            source_url=url,
            qa_type="parent",
        ))

    # Parent technique only: enumerate its sub-techniques (if any exist)
    if not is_sub and all_techniques:
        subs = [
            t for t in all_techniques
            if t.get("is_sub_technique") and t.get("parent_id") == tid
        ]
        if subs:
            sub_list = "; ".join(f"{s['name']} ({s['id']})" for s in subs)
            pairs.append(QAPair(
                id=f"qa_{sid}_subs",
                question=f"What are the sub-techniques of {name} ({tid})?",
                answer=(
                    f"{name} ({tid}) has {len(subs)} sub-technique(s) "
                    f"in MITRE ATT&CK Enterprise: {sub_list}."
                ),
                source_id=tid,
                source_type=source_type,
                source_url=url,
                qa_type="sub_techniques",
            ))

    return pairs


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw scraper output
    with open(raw_dir / "tactics.json", encoding="utf-8") as f:
        tactics = json.load(f)
    with open(raw_dir / "techniques.json", encoding="utf-8") as f:
        techniques = json.load(f)

    logger.info(f"Loaded {len(tactics)} tactics, {len(techniques)} techniques")

    # Build parent lookup so sub-techniques can reference their parent
    parent_by_id = {
        t["id"]: t for t in techniques if not t.get("is_sub_technique")
    }

    # Generate
    all_pairs: list[QAPair] = []

    for tactic in tactics:
        all_pairs.extend(generate_tactic_qa(tactic))

    for tech in techniques:
        parent_tech = (
            parent_by_id.get(tech.get("parent_id"))
            if tech.get("is_sub_technique") else None
        )
        all_pairs.extend(generate_technique_qa(tech, parent_tech, techniques))

    logger.info(f"Generated {len(all_pairs)} Q-A pairs")

    # Persist
    out_path = out_dir / "qa_pairs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in all_pairs], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_pairs)} pairs to {out_path}")

    # Diagnostics — useful for the report
    type_counts = Counter(p.qa_type for p in all_pairs)
    source_counts = Counter(p.source_type for p in all_pairs)
    logger.info(f"Q-A type breakdown: {dict(type_counts)}")
    logger.info(f"Source-type breakdown: {dict(source_counts)}")


if __name__ == "__main__":
    main()