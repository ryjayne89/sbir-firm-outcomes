"""Paths and constants shared by every step of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Coverage of each SEC source. Outcomes for firms whose post-award window falls before
# these dates are unobserved rather than negative.
FORMD_COVERAGE_START = pd.Timestamp("2009-01-01")
FORMC_COVERAGE_START = pd.Timestamp("2016-05-16")
EDGAR_COVERAGE_START = pd.Timestamp("1996-05-01")
OZ_DESIGNATION_YEAR = 2018
SBIR_PROGRAM_START_YEAR = 1982
HUBZONE_START_YEAR = 1999
USASPENDING_COVERAGE_START = pd.Timestamp("2007-10-01")   # fiscal year 2008
PGPUB_COVERAGE_START = pd.Timestamp("2001-03-15")         # first pre-grant publications
FWD_CITE_YEARS = 5                                        # forward-citation window after grant

# Linkage thresholds (see docs/methods.md).
MIN_LOOSE_KEY = 4          # shortest distinctive name core allowed to link
MIN_LOOSE_KEY_MATCH = 6    # shortest name core for a generic-word-insensitive match
FUZZY_MIN = 90             # token-sort ratio kept as a fuzzy candidate
FUZZY_STATE_ONLY = 95      # bar for a fuzzy match supported by state alone
RADIUS_MILES = 15          # local capital density radius
GEOCODE_BATCH = 5000

PUBLIC_DATA_LICENSE = "CC-BY-4.0"

# USPTO bulk products used as inputs, as listed in the USPTO Open Data Portal Bulk Dataset
# Directory (data.uspto.gov) when they were downloaded on 2026-10-16. PatentsView revises its
# disambiguation between releases, so these identify the exact inputs.
USPTO_RELEASES = {
    "g_": ("PVGPATDIS", "PatentsView Granted Patent Disambiguated Data", "grants 1976-01-01 to 2025-12-31", "2026-04-10"),
    "pg_": ("PVPGPUBDIS", "PatentsView Pre-Grant Publication Disambiguated Data", "publications 2001-03-15 to 2025-12-31", "2026-10-06"),
    "assignment": ("ECORSEXC", "Patent Assignment Data for Academia and Researchers", "recordations from 1980", "2026-09-25"),
}


def uspto_release(table: str) -> str:
    for prefix, (pid, name, cover, updated) in USPTO_RELEASES.items():
        if table.startswith(prefix) or (prefix == "assignment" and table in
                                        ("assignment", "assignment_conveyance", "assignor", "assignee", "documentid")):
            return f"{name} ({pid}), {cover}, last updated {updated}"
    return ""


@dataclass(frozen=True)
class Paths:
    data: Path

    def __post_init__(self):
        object.__setattr__(self, "data", Path(self.data))

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def audit(self) -> Path:
        return self.data / "audit"

    @property
    def validation(self) -> Path:
        return self.data / "validation"

    @property
    def release(self) -> Path:
        return self.data / "release"

    @property
    def sec(self) -> Path:
        return self.interim / "sec"

    @property
    def manifest(self) -> Path:
        return self.raw / "MANIFEST.csv"

    def make(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.audit, self.validation, self.release,
                  self.sec, self.interim / "sba", self.interim / "geo", self.interim / "macro",
                  self.interim / "usaspending", self.interim / "patents"):
            p.mkdir(parents=True, exist_ok=True)
        return self
