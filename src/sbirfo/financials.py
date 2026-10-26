"""Annual financial statements of SBIR firms that file with the SEC, from XBRL company facts.

The SEC's companyfacts.zip holds every XBRL value each filer has reported, from about 2009 on.
For each firm's primary high-confidence CIK, annual values are taken from the calendar-year
frames the SEC assigns (CY2020 for a full-year flow, CY2020Q4I for a year-end balance), which
already pick one value per company, concept and year.
"""
from __future__ import annotations

import json
import logging
import zipfile

import numpy as np
import pandas as pd

from . import io as sio
from .config import Paths

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet"],
    "rnd_expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "stockholders_equity": ["StockholdersEquity"],
    "ipo_proceeds": ["ProceedsFromIssuanceInitialPublicOffering"],
    "stock_issuance_proceeds": ["ProceedsFromIssuanceOfCommonStock"],
}
INSTANT = {"assets", "cash", "stockholders_equity"}


def _annual(facts: dict, concepts: list[str], instant: bool) -> pd.Series:
    """Values by calendar year from SEC frames. Companies switch concepts over time (revenue moved
    to RevenueFromContractWithCustomer... in 2018), so each year takes the first concept in
    priority order that reports it."""
    pat = r"^CY(\d{4})Q4I$" if instant else r"^CY(\d{4})$"
    parts = []
    for rank, c in enumerate(concepts):
        for ns in ("us-gaap", "dei", "ifrs-full"):
            node = facts.get(ns, {}).get(c)
            if not node:
                continue
            rows = [v for vals in node.get("units", {}).values() for v in vals if v.get("frame")]
            if rows:
                d = pd.DataFrame(rows)
                d["year"] = d.frame.str.extract(pat, expand=False)
                parts.append(d.dropna(subset=["year"]).assign(rank=rank))
    if not parts:
        return pd.Series(dtype="float")
    d = pd.concat(parts).sort_values("rank").drop_duplicates("year")
    return d.assign(year=d.year.astype(int)).set_index("year").val.sort_index()


def run(paths: Paths, sec_user_agent: str = "") -> None:
    lk = paths.processed / "link_sbir_sec.parquet"
    links = pd.read_parquet(lk)
    links = links[links.is_primary & (links.confidence == "high") & ~links.is_fund_cik]
    ciks = dict(zip(links.cik.str.zfill(10), links.firm_id_sbir))
    dest = paths.raw / "sec_edgar" / "companyfacts.zip"
    if not dest.exists():
        if not sec_user_agent:
            log.warning("companyfacts.zip not on disk and no --sec-user-agent given; financial fields left empty")
            return
        sio.download(sio.sec_session(sec_user_agent), COMPANYFACTS_URL, dest, sio.Manifest(paths.manifest),
                     "sec_xbrl_companyfacts", timeout=3600)
    else:
        sio.Manifest(paths.manifest).add("sec_xbrl_companyfacts", dest, COMPANYFACTS_URL, note="present on disk")

    rows = []
    with zipfile.ZipFile(dest) as zf:
        names = {n.split("CIK")[-1][:10]: n for n in zf.namelist() if n.endswith(".json")}
        for cik, firm in ciks.items():
            n = names.get(cik)
            if n is None:
                continue
            facts = json.loads(zf.read(n)).get("facts", {})
            cols = {k: _annual(facts, v, k in INSTANT) for k, v in CONCEPTS.items()}
            d = pd.DataFrame(cols)
            if len(d):
                rows.append(d.rename_axis("year").reset_index().assign(firm_id_sbir=firm, cik=cik))
    fin = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["firm_id_sbir", "cik", "year"])
    fin.to_parquet(paths.processed / "firm_financials_year.parquet", index=False)

    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet").set_index("firm_id_sbir")
    g = fin.sort_values("year").groupby("firm_id_sbir")
    out = pd.DataFrame(index=firms.index)
    out["xbrl_first_year"], out["xbrl_last_year"] = g.year.min(), g.year.max()
    last = lambda c: fin.dropna(subset=[c]).sort_values("year").groupby("firm_id_sbir")[c].last()
    for c in ("revenue", "rnd_expense", "net_income", "assets"):
        if c in fin:
            out[f"{c}_latest"] = last(c)
    out["ipo_proceeds_xbrl"] = (fin[fin.ipo_proceeds > 0].groupby("firm_id_sbir").ipo_proceeds.max()
                                if "ipo_proceeds" in fin else np.nan)
    out["revenue_max"] = g.revenue.max() if "revenue" in fin else np.nan
    out.reset_index().to_parquet(paths.processed / "firm_financials.parquet", index=False)
    stats = {"ciks_requested": len(ciks), "firms_with_xbrl": int(fin.firm_id_sbir.nunique()), "firm_years": int(len(fin)),
             "firms_with_revenue": int(out.revenue_latest.notna().sum()) if "revenue_latest" in out else 0,
             "firms_with_ipo_proceeds": int(out.ipo_proceeds_xbrl.notna().sum())}
    (paths.audit / "financials.json").write_text(json.dumps(stats, indent=2))
    log.info("financials: %s", stats)
