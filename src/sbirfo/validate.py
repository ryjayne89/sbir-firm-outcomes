"""Hand-labeled validation of the SBIR-to-SEC linkage.

Two samples are drawn with a fixed seed:

* a simple random sample of firms (default 300), labeled whether or not they were linked.
  Because it is drawn from all firms, it estimates both precision and recall;
* an oversample of linked pairs within each tier (default 40), for tier-level precision.

The labeler fills in `true_cik` for every row: the CIK(s) that belong to the firm, separated
by semicolons, or `none` when EDGAR has no filer for it. `make_sheet` writes the sheet;
`score` reads the completed sheet and writes precision, recall and F1 with Wilson intervals.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from .config import Paths

log = logging.getLogger(__name__)

SHEET = "label_sheet.csv"
LABELED = "label_sheet_labeled.csv"
# EDGAR's company search matches the start of a filer's conformed name, which abbreviates legal
# forms (CORP, INC, CO), so the search term is the name without its legal form and punctuation.
EDGAR_NAME = "https://www.sec.gov/cgi-bin/browse-edgar?company={q}&owner=exclude&action=getcompany&count=100"
EDGAR_CIK = "https://www.sec.gov/edgar/browse/?CIK={cik}"


def search_term(name) -> str:
    """Firm name without trailing legal-form words, a leading 'the' or punctuation."""
    from .names import LEGAL_SUFFIXES
    words = re.sub(r"[^A-Za-z0-9& ]+", " ", str(name)).split()
    while len(words) > 1 and words[-1].lower() in LEGAL_SUFFIXES:
        words.pop()
    if len(words) > 1 and words[0].lower() == "the":
        words.pop(0)
    return " ".join(words)


def links(names: pd.Series, ciks: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "edgar_name_search": names.map(lambda n: EDGAR_NAME.format(q=quote_plus(search_term(n)))),
        "edgar_predicted": ciks.map(lambda c: EDGAR_CIK.format(cik=str(c).lstrip("0")) if isinstance(c, str) and c else ""),
    }, index=names.index)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def make_sheet(paths: Paths, n_random: int = 300, n_per_tier: int = 40, seed: int = 20261015) -> pd.DataFrame:
    out = paths.validation / SHEET
    if out.exists():
        log.info("label sheet exists, not overwritten: %s", out)
        return pd.read_csv(out, dtype=str)
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    links = pd.read_parquet(paths.processed / "link_sbir_sec.parquet")
    rng = np.random.default_rng(seed)

    rand = firms.sample(n=min(n_random, len(firms)), random_state=int(rng.integers(1 << 31)))
    rand = rand.assign(sample="random").merge(links[links.is_primary], on="firm_id_sbir", how="left")

    tiers = []
    pool = links[~links.firm_id_sbir.isin(rand.firm_id_sbir)]
    for tier, d in pool.groupby("tier"):
        s = d.sample(n=min(n_per_tier, len(d)), random_state=int(rng.integers(1 << 31)))
        tiers.append(s.assign(sample=f"tier_{tier}"))
    tiers = pd.concat(tiers, ignore_index=True).merge(firms, on="firm_id_sbir", how="left") if tiers else pd.DataFrame()

    sheet = pd.concat([rand, tiers], ignore_index=True)
    sheet = pd.DataFrame({
        "sample": sheet["sample"], "firm_id_sbir": sheet.firm_id_sbir,
        "sbir_names": sheet.names_all, "sbir_city": sheet.city, "sbir_state": sheet.state,
        "first_award": sheet.first_award_date.dt.year, "last_award": sheet.last_award_date.dt.year,
        "predicted_cik": sheet.cik, "predicted_tier": sheet.tier.astype("Int64"),
        "sec_name": sheet.name_raw_sec, "sec_city": sheet.city_sec, "sec_state": sheet.state_sec,
        "edgar_name_search": links(sheet.firm_name, sheet.cik).edgar_name_search,
        "edgar_predicted": links(sheet.firm_name, sheet.cik).edgar_predicted,
        "true_cik": "", "notes": "",
    })
    sheet.to_csv(out, index=False)
    log.info("label sheet: %d rows (%d random firms) -> %s", len(sheet), (sheet["sample"] == "random").sum(), out)
    return sheet


def _cik_set(s) -> set[str]:
    if not isinstance(s, str) or s.strip().lower() in ("", "none", "nan"):
        return set()
    return {x.strip().lstrip("0") for x in s.split(";") if x.strip()}


def score(paths: Paths) -> pd.DataFrame | None:
    f = paths.validation / LABELED
    if not f.exists():
        log.info("no labeled sheet at %s; skipping scoring", f)
        return None
    lab = pd.read_csv(f, dtype=str)
    lab = lab[lab.true_cik.fillna("").str.strip() != ""].copy()
    lab["truth"] = lab.true_cik.map(_cik_set)
    lab["pred"] = lab.predicted_cik.fillna("").str.lstrip("0")
    lab["has_pred"] = lab.pred != ""
    lab["has_truth"] = lab.truth.map(bool)
    lab["correct"] = [p in t for p, t in zip(lab.pred, lab.truth)]

    rows = []
    r = lab[lab["sample"] == "random"]
    tp = int((r.has_pred & r.correct).sum())
    for name, k, n in [("precision", tp, int(r.has_pred.sum())), ("recall", tp, int(r.has_truth.sum()))]:
        p, lo, hi = wilson(k, n)
        rows.append({"sample": "random", "metric": name, "k": k, "n": n, "estimate": p, "ci_low": lo, "ci_high": hi})
    pr, rc = rows[0]["estimate"], rows[1]["estimate"]
    rows.append({"sample": "random", "metric": "f1", "k": np.nan, "n": len(r),
                 "estimate": 2 * pr * rc / (pr + rc) if pr and rc else np.nan, "ci_low": np.nan, "ci_high": np.nan})
    lab["predicted_tier"] = pd.to_numeric(lab.predicted_tier, errors="coerce").astype("Int64")
    for tier, d in lab[lab.has_pred].groupby("predicted_tier"):
        p, lo, hi = wilson(int(d.correct.sum()), len(d))
        rows.append({"sample": f"tier_{tier}", "metric": "precision", "k": int(d.correct.sum()), "n": len(d),
                     "estimate": p, "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(paths.validation / "validation_results.csv", index=False)
    log.info("validation: precision %.3f, recall %.3f (random sample of %d labeled firms)", pr, rc, len(r))
    return res


def run(paths: Paths) -> None:
    make_sheet(paths)
    score(paths)
