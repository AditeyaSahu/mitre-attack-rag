"""
src/enrichment.py

Second-pass scraper that fetches each parent technique's detail page from
MITRE ATT&CK to enrich the knowledge base with mitigations, detection
strategies, platforms, tactics, and procedure examples.

This complements src/scraper.py — which collects index-page data (names,
descriptions) — by adding the practitioner-actionable content (how to
mitigate, how to detect, real-world adversary procedures) needed for
generating high-quality cybersecurity guidelines.

Usage:
    python -m src.enrichment                # full scrape (~4 minutes for ~222 techniques)
    python -m src.enrichment --limit 5      # quick test on first 5 techniques
"""

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Reuse config from the index scraper
from src.scraper import (
    BASE_URL,
    OUTPUT_DIR,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Data structures ----------

@dataclass
class Mitigation:
    id: str          # e.g. "M1047"
    name: str
    description: str


@dataclass
class DetectionAnalytic:
    detection_id: str        # e.g. "DET0345"
    detection_name: str
    analytic_id: str         # e.g. "AN0975"
    analytic_description: str


@dataclass
class ProcedureExample:
    actor_id: str            # "S####" (software) or "G####" (threat group)
    actor_name: str
    description: str


@dataclass
class TechniqueDetails:
    id: str
    name: str
    url: str
    platforms: list[str] = field(default_factory=list)
    tactics: list[dict] = field(default_factory=list)              # [{id, name}, ...]
    mitigations: list[Mitigation] = field(default_factory=list)
    detections: list[DetectionAnalytic] = field(default_factory=list)
    procedure_examples: list[ProcedureExample] = field(default_factory=list)


# ---------- Enricher ----------

class TechniqueEnricher:
    """Fetch and parse per-technique detail pages from MITRE ATT&CK."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        delay: float = REQUEST_DELAY_SECONDS,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """GET a URL; returns None on failure rather than raising."""
        try:
            r = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
        time.sleep(self.delay)
        return BeautifulSoup(r.text, "lxml")

    # ---- Table parsers ----

    def _parse_mitigations_table(self, table) -> list[Mitigation]:
        """3-col table: [M-ID | Name | Description]."""
        items: list[Mitigation] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            mid = cells[0].get_text(strip=True)
            if not re.match(r"^M\d{4}$", mid):
                continue
            items.append(Mitigation(
                id=mid,
                name=cells[1].get_text(strip=True),
                description=cells[2].get_text(strip=True, separator=" "),
            ))
        return items

    def _parse_detection_table(self, table) -> list[DetectionAnalytic]:
        """
        Detection Strategy table:
          Full row : [DET-ID | Name | AN-ID | Description]
          Sub-row  : [AN-ID  | Description]   (analytic under the previous DET)
        Rowspan on the DET column means sub-rows have 2 cells, not 4.
        """
        items: list[DetectionAnalytic] = []
        current_det_id: Optional[str] = None
        current_det_name: Optional[str] = None

        for row in table.find_all("tr"):
            cells = row.find_all("td")

            if len(cells) >= 4:
                first = cells[0].get_text(strip=True)
                if re.match(r"^DET\d+$", first):
                    current_det_id = first
                    current_det_name = cells[1].get_text(strip=True)
                    an_id = cells[2].get_text(strip=True)
                    an_desc = cells[3].get_text(strip=True, separator=" ")
                elif re.match(r"^AN\d+$", first):
                    an_id = first
                    an_desc = cells[1].get_text(strip=True, separator=" ")
                else:
                    continue
                items.append(DetectionAnalytic(
                    detection_id=current_det_id or "",
                    detection_name=current_det_name or "",
                    analytic_id=an_id,
                    analytic_description=an_desc,
                ))

            elif len(cells) == 2:
                first = cells[0].get_text(strip=True)
                if not re.match(r"^AN\d+$", first):
                    continue
                items.append(DetectionAnalytic(
                    detection_id=current_det_id or "",
                    detection_name=current_det_name or "",
                    analytic_id=first,
                    analytic_description=cells[1].get_text(strip=True, separator=" "),
                ))
        return items

    def _parse_procedure_table(self, table) -> list[ProcedureExample]:
        """3-col table: [S####/G#### | Actor Name | Description]."""
        items: list[ProcedureExample] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            actor_id = cells[0].get_text(strip=True)
            if not re.match(r"^[SG]\d{4}$", actor_id):
                continue
            items.append(ProcedureExample(
                actor_id=actor_id,
                actor_name=cells[1].get_text(strip=True),
                description=cells[2].get_text(strip=True, separator=" "),
            ))
        return items

    # ---- Page-level parse ----

    def scrape_technique_details(
        self, technique_id: str, technique_name: str
    ) -> Optional[TechniqueDetails]:
        url = f"{BASE_URL}/techniques/{technique_id}/"
        soup = self._fetch(url)
        if soup is None:
            return None

        details = TechniqueDetails(id=technique_id, name=technique_name, url=url)

        # Section tables — match by h2 heading text
        for h2 in soup.find_all("h2"):
            heading = h2.get_text(strip=True).lower()
            table = h2.find_next("table")
            if table is None:
                continue
            if "mitigation" in heading:
                details.mitigations = self._parse_mitigations_table(table)
            elif "detection" in heading:
                details.detections = self._parse_detection_table(table)
            elif "procedure" in heading:
                details.procedure_examples = self._parse_procedure_table(table)

        # Platforms — inline text after the "Platforms:" label
        body_text = soup.get_text("\n", strip=True)
        m = re.search(r"Platforms:\s*([^\n]+)", body_text)
        if m:
            details.platforms = [p.strip() for p in m.group(1).split(",") if p.strip()]

        # Tactics — find anchor links to TA-prefixed tactic pages
        seen_tactic_ids: set[str] = set()
        for link in soup.find_all("a", href=True):
            tm = re.match(r"^/tactics/(TA\d{4})", link["href"])
            if not tm:
                continue
            tid = tm.group(1)
            tname = link.get_text(strip=True)
            if tname and tname != tid and tid not in seen_tactic_ids:
                details.tactics.append({"id": tid, "name": tname})
                seen_tactic_ids.add(tid)

        return details

    # ---- Persistence ----

    def save_json(self, records: list[TechniqueDetails], filename: str) -> Path:
        path = self.output_dir / filename
        payload = [asdict(r) for r in records]
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records to {path}")
        return path


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only enrich the first N parent techniques (use for testing).",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SECONDS,
        help="Seconds between HTTP requests (default 1.0).",
    )
    parser.add_argument(
        "--techniques-path", default="data/raw/techniques.json",
        help="Path to the techniques.json from the index scraper.",
    )
    parser.add_argument(
        "--out", default="technique_details.json",
        help="Output filename, written under data/raw/.",
    )
    args = parser.parse_args()

    techniques_path = Path(args.techniques_path)
    if not techniques_path.exists():
        logger.error(
            f"Cannot find {techniques_path}. "
            "Run `python -m src.scraper` first."
        )
        return

    with techniques_path.open(encoding="utf-8") as f:
        all_techniques = json.load(f)

    parents = [t for t in all_techniques if not t.get("is_sub_technique")]
    if args.limit:
        parents = parents[: args.limit]
    logger.info(f"Enriching {len(parents)} parent techniques...")

    enricher = TechniqueEnricher(delay=args.delay)
    results: list[TechniqueDetails] = []

    for tech in tqdm(parents, desc="Enriching"):
        details = enricher.scrape_technique_details(tech["id"], tech["name"])
        if details:
            results.append(details)

    enricher.save_json(results, args.out)

    # Summary stats — useful for the report
    total_mits = sum(len(d.mitigations) for d in results)
    total_dets = sum(len(d.detections) for d in results)
    total_procs = sum(len(d.procedure_examples) for d in results)
    techniques_with_mits = sum(1 for d in results if d.mitigations)
    techniques_with_dets = sum(1 for d in results if d.detections)
    logger.info(
        f"Done. Enriched {len(results)} techniques: "
        f"{total_mits} mitigations across {techniques_with_mits} techniques, "
        f"{total_dets} detection analytics across {techniques_with_dets} techniques, "
        f"{total_procs} procedure examples."
    )


if __name__ == "__main__":
    main()