"""
src/scraper.py

MITRE ATT&CK Enterprise web scraper.

Scrapes the public MITRE ATT&CK website (https://attack.mitre.org) to build
a structured knowledge base of tactics, techniques, and sub-techniques for
the Enterprise matrix (v19).

Design notes:
- Polite scraping: 1 second delay between requests, descriptive User-Agent.
- Two HTTP requests get us all 15 tactics + 222 techniques + 475 sub-techniques,
  because the techniques index page renders the full table inline.
- A second enrichment pass (future work) can fetch per-technique pages for
  detection strategies, mitigations, and procedure examples.
- Output is written as JSON to data/raw/.

Usage:
    python -m src.scraper              # full scrape
    python -m src.scraper --test       # only tactics (quick sanity check)
"""

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------- Configuration ----------

BASE_URL = "https://attack.mitre.org"
TACTICS_URL = f"{BASE_URL}/tactics/enterprise/"
TECHNIQUES_URL = f"{BASE_URL}/techniques/enterprise/"

USER_AGENT = (
    "MITRE-ATTCK-RAG-Research/1.0 "
    "(University of Adelaide assignment; academic use only; "
    "contact: aditeya2003@gmail.com)"
)

REQUEST_DELAY_SECONDS = 1.0      # polite scraping
REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_DIR = Path("data/raw")

# ID patterns
TACTIC_ID_PATTERN = re.compile(r"^TA\d{4}$")
TECHNIQUE_ID_PATTERN = re.compile(r"^T\d{4}$")
SUB_TECHNIQUE_SUFFIX_PATTERN = re.compile(r"^\.\d{3}$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Data classes ----------

@dataclass
class Tactic:
    """One MITRE ATT&CK Enterprise tactic (e.g. Initial Access)."""
    id: str
    name: str
    description: str
    url: str


@dataclass
class Technique:
    """One MITRE ATT&CK Enterprise technique or sub-technique."""
    id: str                          # e.g. "T1548" or "T1548.001"
    name: str
    description: str
    url: str
    is_sub_technique: bool = False
    parent_id: Optional[str] = None  # populated for sub-techniques only


# ---------- Scraper ----------

class MITREAttackScraper:
    """
    Polite HTML scraper for the MITRE ATT&CK Enterprise framework.

    Each scrape_* method returns parsed objects; persistence is handled by
    save_json so concerns are separated and easier to unit-test.
    """

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

    # ---- HTTP ----

    def _fetch(self, url: str) -> BeautifulSoup:
        """GET a URL and return parsed HTML. Polite delay applied after the request."""
        logger.info(f"GET {url}")
        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        time.sleep(self.delay)
        return soup

    # ---- Tactics ----

    def scrape_tactics(self) -> list[Tactic]:
        """
        Scrape the 15 Enterprise tactics from the tactics index page.
        Table columns: [TA-ID link | Name link | Description].
        """
        soup = self._fetch(TACTICS_URL)
        tactics: list[Tactic] = []

        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                id_link = cells[0].find("a")
                name_link = cells[1].find("a")
                if not (id_link and name_link):
                    continue

                tid = id_link.get_text(strip=True)
                if not TACTIC_ID_PATTERN.match(tid):
                    continue

                tactics.append(Tactic(
                    id=tid,
                    name=name_link.get_text(strip=True),
                    description=cells[2].get_text(strip=True, separator=" "),
                    url=BASE_URL + id_link["href"],
                ))

        logger.info(f"Parsed {len(tactics)} tactics")
        return tactics

    # ---- Techniques ----

    def scrape_techniques(self) -> list[Technique]:
        """
        Scrape all techniques + sub-techniques from the Enterprise techniques
        index page.

        Strategy: find every anchor link whose href matches a technique URL
        pattern (/techniques/T1234 or /techniques/T1234/001), then walk up
        to the containing <tr> and pull the description from the last <td>.
        This is robust against rowspan/colspan tricks and column reordering.
        """
        soup = self._fetch(TECHNIQUES_URL)
        techniques: list[Technique] = []
        seen_ids: set[str] = set()

        parent_href_pattern = re.compile(r"^/techniques/(T\d{4})/?$")
        sub_href_pattern = re.compile(r"^/techniques/(T\d{4})/(\d{3})/?$")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            parent_match = parent_href_pattern.match(href)
            sub_match = sub_href_pattern.match(href)

            if parent_match:
                tid = parent_match.group(1)
                if tid in seen_ids:
                    continue
                row = link.find_parent("tr")
                if not row:
                    continue
                cells = row.find_all("td")
                description = (
                    cells[-1].get_text(strip=True, separator=" ") if cells else ""
                )
                # Use the human-readable name if the link text is the bare ID.
                link_text = link.get_text(strip=True)
                if link_text == tid:
                    name_links = [
                        a for a in row.find_all("a")
                        if a.get_text(strip=True) and a.get_text(strip=True) != tid
                    ]
                    name = name_links[0].get_text(strip=True) if name_links else tid
                else:
                    name = link_text
                if not description:
                    # Skip nav/breadcrumb links that aren't in a data row.
                    continue
                techniques.append(Technique(
                    id=tid,
                    name=name,
                    description=description,
                    url=BASE_URL + href,
                    is_sub_technique=False,
                    parent_id=None,
                ))
                seen_ids.add(tid)

            elif sub_match:
                parent_tid = sub_match.group(1)
                sub_num = sub_match.group(2)
                full_id = f"{parent_tid}.{sub_num}"
                if full_id in seen_ids:
                    continue
                row = link.find_parent("tr")
                if not row:
                    continue
                cells = row.find_all("td")
                description = (
                    cells[-1].get_text(strip=True, separator=" ") if cells else ""
                )
                link_text = link.get_text(strip=True)
                if link_text == f".{sub_num}":
                    name_links = [
                        a for a in row.find_all("a")
                        if a.get_text(strip=True)
                        and a.get_text(strip=True) not in (f".{sub_num}", parent_tid)
                    ]
                    name = name_links[0].get_text(strip=True) if name_links else full_id
                else:
                    name = link_text
                if not description:
                    continue
                techniques.append(Technique(
                    id=full_id,
                    name=name,
                    description=description,
                    url=BASE_URL + href,
                    is_sub_technique=True,
                    parent_id=parent_tid,
                ))
                seen_ids.add(full_id)

        parents = sum(1 for t in techniques if not t.is_sub_technique)
        subs = sum(1 for t in techniques if t.is_sub_technique)
        logger.info(f"Parsed {parents} techniques and {subs} sub-techniques")
        return techniques

    # ---- Persistence ----

    def save_json(self, records: list, filename: str) -> Path:
        """Save a list of dataclass instances as a JSON array."""
        path = self.output_dir / filename
        payload = [asdict(r) for r in records]
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records to {path}")
        return path


# ---------- CLI entry point ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only scrape the tactics page (quick sanity check).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help="Seconds between HTTP requests (default: 1.0).",
    )
    args = parser.parse_args()

    scraper = MITREAttackScraper(delay=args.delay)

    # 1. Tactics
    tactics = scraper.scrape_tactics()
    scraper.save_json(tactics, "tactics.json")

    if args.test:
        logger.info("Test mode complete (skipped techniques scrape).")
        return

    # 2. Techniques + sub-techniques
    techniques = scraper.scrape_techniques()
    scraper.save_json(techniques, "techniques.json")

    # 3. Summary
    parents = sum(1 for t in techniques if not t.is_sub_technique)
    subs = sum(1 for t in techniques if t.is_sub_technique)
    logger.info(
        f"Done. {len(tactics)} tactics, {parents} techniques, {subs} sub-techniques."
    )


if __name__ == "__main__":
    main()