"""
src/preprocessing.py

Generate question-answer pairs from raw MITRE ATT&CK scraper output.

Two passes of source data are used:
  1. tactics.json + techniques.json (from src/scraper.py)
     - Names, IDs, descriptions, parent/child relationships
  2. technique_details.json (from src/enrichment.py)
     - Mitigations, detection analytics, platforms, tactics, procedures

Each Q-A pair retains source metadata (source_id, source_url, source_type)
so retrieved answers can be cited back to the original MITRE entry —
required for transparency and Criterion 4 (real-world deployment).

Usage:
    python -m src.preprocessing
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
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
MITIGATION_URL_BASE = "https://attack.mitre.org/mitigations"


@dataclass
class QAPair:
    id: str
    question: str
    answer: str
    source_id: str
    source_type: str          # tactic | technique | sub-technique | mitigation
    source_url: str
    qa_type: str


# ---------- Helpers ----------

def first_sentence(text: str, max_words: int = 30) -> str:
    if not text:
        return ""
    m = re.search(r"^(.+?[.!?])(\s|$)", text.strip())
    sentence = m.group(1) if m else text.strip()
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]) + "..."
    return sentence


def safe_id(raw_id: str) -> str:
    return raw_id.replace(".", "_")


def truncate(text: str, max_chars: int = 1500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ---------- Tactic Q-A ----------

def generate_tactic_qa(tactic: dict) -> list[QAPair]:
    tid = tactic["id"]
    name = tactic["name"]
    desc = tactic["description"]
    url = tactic["url"]

    return [
        QAPair(
            id=f"qa_{tid}_def",
            question=f"What is the {name} tactic in MITRE ATT&CK?",
            answer=f"{name} ({tid}) is a MITRE ATT&CK Enterprise tactic. {desc}",
            source_id=tid, source_type="tactic", source_url=url, qa_type="definition",
        ),
        QAPair(
            id=f"qa_{tid}_id",
            question=f"What does the MITRE ATT&CK tactic {tid} represent?",
            answer=f"Tactic {tid} is {name}: {desc}",
            source_id=tid, source_type="tactic", source_url=url, qa_type="id_lookup",
        ),
        QAPair(
            id=f"qa_{tid}_goal",
            question=f"What is the adversary's goal in the {name} tactic?",
            answer=f"The adversary's goal in the {name} tactic ({tid}) is: {desc}",
            source_id=tid, source_type="tactic", source_url=url, qa_type="goal",
        ),
    ]


# ---------- Base technique Q-A (descriptions only) ----------

def generate_technique_qa(
    tech: dict,
    parent_tech: Optional[dict] = None,
    all_techniques: Optional[list[dict]] = None,
) -> list[QAPair]:
    tid = tech["id"]
    name = tech["name"]
    desc = tech["description"]
    url = tech["url"]
    is_sub = tech.get("is_sub_technique", False)
    source_type = "sub-technique" if is_sub else "technique"
    sid = safe_id(tid)

    pairs: list[QAPair] = [
        QAPair(
            id=f"qa_{sid}_def",
            question=f"What is {name}?",
            answer=f"{name} ({tid}) is a MITRE ATT&CK Enterprise {source_type}. {desc}",
            source_id=tid, source_type=source_type, source_url=url, qa_type="definition",
        ),
        QAPair(
            id=f"qa_{sid}_id",
            question=f"What is MITRE ATT&CK {source_type} {tid}?",
            answer=f"{tid} is {name}. {desc}",
            source_id=tid, source_type=source_type, source_url=url, qa_type="id_lookup",
        ),
        QAPair(
            id=f"qa_{sid}_how",
            question=f"How does the {name} technique work?",
            answer=f"In MITRE ATT&CK, {name} ({tid}) is described as: {desc}",
            source_id=tid, source_type=source_type, source_url=url, qa_type="how",
        ),
    ]

    if is_sub and parent_tech:
        pairs.append(QAPair(
            id=f"qa_{sid}_parent",
            question=f"What is the parent technique of {name} ({tid})?",
            answer=(
                f"The parent technique of {name} ({tid}) is "
                f"{parent_tech['name']} ({parent_tech['id']}). "
                f"{parent_tech['description']}"
            ),
            source_id=tid, source_type=source_type, source_url=url, qa_type="parent",
        ))

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
                    f"{name} ({tid}) has {len(subs)} sub-technique(s) in "
                    f"MITRE ATT&CK Enterprise: {sub_list}."
                ),
                source_id=tid, source_type=source_type, source_url=url, qa_type="sub_techniques",
            ))

    return pairs


# ---------- Enriched technique Q-A (mitigations, detection, platforms, etc.) ----------

def generate_mitigation_qa(tech: dict, details: dict) -> list[QAPair]:
    """Two practitioner-facing Q-A pairs about mitigating this technique."""
    if not details or not details.get("mitigations"):
        return []

    tid = tech["id"]
    name = tech["name"]
    url = tech["url"]
    sid = safe_id(tid)
    mits = details["mitigations"]

    mit_summary = "\n".join(
        f"- {m['name']} ({m['id']}): {m['description']}" for m in mits
    )

    return [
        QAPair(
            id=f"qa_{sid}_mitigation_overview",
            question=f"How do I mitigate the {name} technique?",
            answer=truncate(
                f"To mitigate {name} ({tid}), MITRE ATT&CK recommends "
                f"{len(mits)} mitigation(s):\n{mit_summary}"
            ),
            source_id=tid, source_type="technique", source_url=url,
            qa_type="mitigation_overview",
        ),
        QAPair(
            id=f"qa_{sid}_mitigation_defend",
            question=f"How can I defend against {name}?",
            answer=truncate(
                f"Defending against {name} ({tid}) involves these MITRE ATT&CK mitigations:\n{mit_summary}"
            ),
            source_id=tid, source_type="technique", source_url=url,
            qa_type="mitigation_defend",
        ),
    ]


def generate_detection_qa(tech: dict, details: dict) -> list[QAPair]:
    """Practitioner Q-A pairs on detection strategies and analytics."""
    if not details or not details.get("detections"):
        return []

    tid = tech["id"]
    name = tech["name"]
    url = tech["url"]
    sid = safe_id(tid)
    dets = details["detections"]

    # Group analytics under their detection strategy
    strategies = defaultdict(lambda: {"name": "", "analytics": []})
    for d in dets:
        det_id = d["detection_id"] or "unknown"
        strategies[det_id]["name"] = d["detection_name"]
        strategies[det_id]["analytics"].append(
            f"  - {d['analytic_id']}: {d['analytic_description']}"
        )

    strategy_text = ""
    for det_id, sg in strategies.items():
        strategy_text += f"\n{sg['name']} ({det_id}):\n" + "\n".join(sg["analytics"])

    return [
        QAPair(
            id=f"qa_{sid}_detection_overview",
            question=f"How is the {name} technique detected?",
            answer=truncate(
                f"MITRE ATT&CK provides {len(strategies)} detection strategy/strategies "
                f"with {len(dets)} analytic(s) for detecting {name} ({tid}):{strategy_text}"
            ),
            source_id=tid, source_type="technique", source_url=url,
            qa_type="detection_overview",
        ),
        QAPair(
            id=f"qa_{sid}_detection_monitor",
            question=f"What should I monitor to detect {name}?",
            answer=truncate(
                f"To detect {name} ({tid}), MITRE ATT&CK recommends the following "
                f"monitoring approaches:{strategy_text}"
            ),
            source_id=tid, source_type="technique", source_url=url,
            qa_type="detection_monitor",
        ),
    ]


def generate_platform_qa(tech: dict, details: dict) -> list[QAPair]:
    if not details or not details.get("platforms"):
        return []

    tid = tech["id"]
    name = tech["name"]
    url = tech["url"]
    sid = safe_id(tid)
    platforms = details["platforms"]

    return [QAPair(
        id=f"qa_{sid}_platforms",
        question=f"What platforms does the {name} technique target?",
        answer=(
            f"{name} ({tid}) targets the following platform(s) in MITRE ATT&CK "
            f"Enterprise: {', '.join(platforms)}."
        ),
        source_id=tid, source_type="technique", source_url=url, qa_type="platforms",
    )]


def generate_tactic_mapping_qa(tech: dict, details: dict) -> list[QAPair]:
    if not details or not details.get("tactics"):
        return []

    tid = tech["id"]
    name = tech["name"]
    url = tech["url"]
    sid = safe_id(tid)
    tactics = details["tactics"]

    tactic_text = ", ".join(f"{t['name']} ({t['id']})" for t in tactics)

    return [QAPair(
        id=f"qa_{sid}_tactic_mapping",
        question=f"Which MITRE ATT&CK tactics does {name} support?",
        answer=(
            f"{name} ({tid}) is associated with the following MITRE ATT&CK Enterprise "
            f"tactic(s): {tactic_text}."
        ),
        source_id=tid, source_type="technique", source_url=url, qa_type="tactic_mapping",
    )]


def generate_procedure_qa(
    tech: dict, details: dict, max_examples: int = 6,
) -> list[QAPair]:
    """Real-world adversary usage examples — one Q-A per technique."""
    if not details or not details.get("procedure_examples"):
        return []

    tid = tech["id"]
    name = tech["name"]
    url = tech["url"]
    sid = safe_id(tid)
    procs = details["procedure_examples"]

    sample = procs[:max_examples]
    sample_text = "\n".join(
        f"- {p['actor_name']} ({p['actor_id']}): {p['description']}"
        for p in sample
    )
    suffix = ""
    if len(procs) > max_examples:
        suffix = f" Plus {len(procs) - max_examples} additional documented example(s)."

    return [QAPair(
        id=f"qa_{sid}_procedures",
        question=f"What are real-world examples of {name}?",
        answer=truncate(
            f"MITRE ATT&CK documents {len(procs)} real-world example(s) of "
            f"{name} ({tid}) being used by threat actors and malware.{suffix} "
            f"Selected examples:\n{sample_text}"
        ),
        source_id=tid, source_type="technique", source_url=url, qa_type="procedures",
    )]


def generate_mitigation_lookup_qa(details_list: list[dict]) -> list[QAPair]:
    """One Q-A per unique mitigation (M-ID) across all techniques."""
    mit_index: dict[str, dict] = {}
    for details in details_list:
        for mit in details.get("mitigations", []):
            mid = mit["id"]
            if mid not in mit_index:
                mit_index[mid] = {
                    "name": mit["name"],
                    "first_description": mit["description"],
                    "used_for": [],
                }
            mit_index[mid]["used_for"].append(details["id"])

    pairs = []
    for mid, info in sorted(mit_index.items()):
        usage = ", ".join(info["used_for"][:8])
        if len(info["used_for"]) > 8:
            usage += f", and {len(info['used_for']) - 8} more"
        pairs.append(QAPair(
            id=f"qa_mit_{mid}",
            question=f"What is the MITRE ATT&CK mitigation {mid}?",
            answer=(
                f"{info['name']} ({mid}) is a MITRE ATT&CK Enterprise mitigation. "
                f"Example usage: {info['first_description']} "
                f"It applies to techniques including: {usage}."
            ),
            source_id=mid, source_type="mitigation",
            source_url=f"{MITIGATION_URL_BASE}/{mid}",
            qa_type="mitigation_lookup",
        ))
    logger.info(f"Generated {len(pairs)} unique mitigation lookup Q-As")
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

    # Load all three source files
    with open(raw_dir / "tactics.json", encoding="utf-8") as f:
        tactics = json.load(f)
    with open(raw_dir / "techniques.json", encoding="utf-8") as f:
        techniques = json.load(f)

    details_path = raw_dir / "technique_details.json"
    if details_path.exists():
        with details_path.open(encoding="utf-8") as f:
            details_list = json.load(f)
    else:
        logger.warning(
            "technique_details.json not found — enriched Q-A generation will be skipped. "
            "Run `python -m src.enrichment` first."
        )
        details_list = []

    logger.info(
        f"Loaded {len(tactics)} tactics, {len(techniques)} techniques, "
        f"{len(details_list)} enriched technique details"
    )

    # Lookups
    parent_by_id = {t["id"]: t for t in techniques if not t.get("is_sub_technique")}
    details_by_id = {d["id"]: d for d in details_list}

    all_pairs: list[QAPair] = []

    # Tactics
    for tactic in tactics:
        all_pairs.extend(generate_tactic_qa(tactic))

    # Techniques (base + enriched)
    for tech in techniques:
        is_sub = tech.get("is_sub_technique", False)
        parent_tech = parent_by_id.get(tech.get("parent_id")) if is_sub else None
        all_pairs.extend(generate_technique_qa(tech, parent_tech, techniques))

        # Enrichment only applies to parent techniques (we only scraped parents)
        if not is_sub:
            details = details_by_id.get(tech["id"])
            if details:
                all_pairs.extend(generate_mitigation_qa(tech, details))
                all_pairs.extend(generate_detection_qa(tech, details))
                all_pairs.extend(generate_platform_qa(tech, details))
                all_pairs.extend(generate_tactic_mapping_qa(tech, details))
                all_pairs.extend(generate_procedure_qa(tech, details))

    # Unique mitigation lookup Q-As
    if details_list:
        all_pairs.extend(generate_mitigation_lookup_qa(details_list))

    logger.info(f"Generated {len(all_pairs)} Q-A pairs total")

    # Persist
    out_path = out_dir / "qa_pairs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in all_pairs], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")

    # Diagnostics
    type_counts = Counter(p.qa_type for p in all_pairs)
    source_counts = Counter(p.source_type for p in all_pairs)
    logger.info(f"Q-A type breakdown: {dict(sorted(type_counts.items()))}")
    logger.info(f"Source-type breakdown: {dict(source_counts)}")


if __name__ == "__main__":
    main()