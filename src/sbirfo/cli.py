"""Command-line entry point: run any subset of the pipeline in order.

    python -m sbirfo --data-dir /path/to/data --sec-user-agent "Name email@domain"
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__, acquire, federal, financials, link, outcomes, panel, patents, places, release, sam, sbir, validate
from .config import Paths

OPTIONAL = {"federal", "sam", "places", "financials", "patents", "panel"}   # the outcomes step runs without these
STEPS = ["acquire", "sbir", "link", "federal", "sam", "places", "financials", "patents", "outcomes", "panel", "validate",
         "release"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sbirfo", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path, help="folder holding raw/, interim/, processed/ ...")
    ap.add_argument("--sec-user-agent", default="", help="name and email sent to the SEC (required for acquire)")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--refresh", action="store_true", help="re-parse raw files even if parsed tables exist")
    ap.add_argument("--include-medium-links", action="store_true",
                    help="build outcomes from medium- as well as high-confidence links (sensitivity version)")
    ap.add_argument("--no-geocode", action="store_true", help="skip the Census geocoder (Opportunity Zones)")
    ap.add_argument("--sam-dir", nargs="*", type=Path, default=[],
                    help="folders holding SAM extracts sam_public_YYYY_MM.parquet, copied into raw/sam")
    ap.add_argument("--patents-dir", nargs="*", type=Path, default=[],
                    help="folders holding PatentsView tables and the USPTO Patent Assignment Dataset")
    ap.add_argument("--no-patent-download", action="store_true", help="use only patent files already on disk")
    ap.add_argument("--research-codes", action="store_true",
                    help="also download FPDS research codes for Phase III candidate contracts (slow: the "
                         "USAspending download service allows only a few jobs per client)")
    ap.add_argument("--no-research-codes", "--no-contract-details", dest="no_research_codes", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--usaspending-workers", type=int, default=1,
                    help="parallel USAspending requests (default 1; the API drops connections at 4)")
    ap.add_argument("--census-key", default=None, help="Census API key (optional; CENSUS_API_KEY also works)")
    ap.add_argument("--version", default=__version__, help="release version label")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    log = logging.getLogger("sbirfo")
    paths = Paths(args.data_dir).make()
    conf = ("high", "medium") if args.include_medium_links else ("high",)

    actions = {
        "acquire": lambda: acquire.run(paths, args.sec_user_agent, args.refresh),
        "sbir": lambda: sbir.run(paths),
        "link": lambda: link.run(paths),
        "federal": lambda: federal.run(paths, contract_details=args.research_codes and not args.no_research_codes, workers=args.usaspending_workers),
        "sam": lambda: sam.run(paths, args.sam_dir),
        "places": lambda: places.run(paths, args.census_key),
        "financials": lambda: financials.run(paths, args.sec_user_agent),
        "panel": lambda: panel.run(paths),
        "patents": lambda: patents.run(paths, args.patents_dir, download=not args.no_patent_download, link_confidence=conf),
        "outcomes": lambda: outcomes.run(paths, conf, geocode=not args.no_geocode),
        "validate": lambda: validate.run(paths),
        "release": lambda: release.run(paths, args.version),
    }
    for step in [s for s in STEPS if s in args.steps]:
        t0 = time.time()
        log.info("== %s", step)
        try:
            actions[step]()
        except Exception:
            if step not in OPTIONAL:
                raise
            log.exception("%s failed; continuing without it. Rerun with --steps %s once fixed.", step, step)
            continue
        log.info("== %s done in %.1f min", step, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
