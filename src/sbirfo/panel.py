"""Firm-year panel over each firm's whole observed life, before and after its first award.

One row per firm and calendar year, from the firm's start (its founding year, or its first
observed record if earlier) to END_YEAR. Every source is counted by year whether the year falls
before or after the first award, so the panel supports models of the award's effect relative to
each firm's baseline.

Each source has a coverage flag (cov_*). A count is zero only in a year the source covers;
outside coverage it is missing, so absence of a record is never confused with absence of the
event. Place measures follow the firm's address in each year.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import places
from .config import Paths

log = logging.getLogger(__name__)

END_YEAR = 2026
EARLIEST = 1976                  # PatentsView grants begin in 1976; nothing earlier is observed
COVERAGE = {                     # first year each source is complete, and the last where relevant
    "formd": (2009, END_YEAR), "formc": (2016, END_YEAR), "edgar": (1996, END_YEAR), "federal": (2008, END_YEAR),
    "patents": (1976, END_YEAR), "sba": (1991, END_YEAR), "xbrl": (2009, END_YEAR), "sbir": (1983, END_YEAR),
}
PATENT_GRANT_LAG = 3             # applications filed in the last few years are mostly not yet granted


def _read(paths: Paths, name: str, sub: str = "processed") -> pd.DataFrame | None:
    f = getattr(paths, sub) / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def _count(df: pd.DataFrame | None, date_col: str, prefix: str, amount: str | None = None,
           by: str | None = None) -> pd.DataFrame | None:
    """Counts (and sums) per firm and year, optionally split into columns by a category."""
    if df is None or df.empty:
        return None
    d = df.assign(year=pd.to_datetime(df[date_col], errors="coerce").dt.year).dropna(subset=["year"])
    d["year"] = d.year.astype(int)
    keys = ["firm_id_sbir", "year"] + ([by] if by else [])
    agg = {f"{prefix}_n": (date_col, "size")}
    if amount:
        agg[f"{prefix}_amount"] = (amount, "sum")
    g = d.groupby(keys).agg(**agg)
    if by:
        g = g.unstack(by)
        g.columns = [f"{a.replace(prefix, prefix + '_' + str(b), 1)}" for a, b in g.columns]
    return g


def run(paths: Paths) -> None:
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    fo = _read(paths, "firm_outcomes")
    founded = fo.set_index("firm_id_sbir").founded_year if fo is not None and "founded_year" in fo else pd.Series(dtype=float)
    first_award = firms.set_index("firm_id_sbir").first_award_date.dt.year

    parts: list[pd.DataFrame] = []

    # SBIR/STTR awards and the employee counts firms report on them
    aw = _read(paths, "sbir_awards")
    aw["date"] = aw.award_date.fillna(pd.to_datetime(aw.award_year.astype("string") + "-07-01", errors="coerce"))
    aw["phase"] = "phase" + aw.phase_num.astype("string")
    parts += [_count(aw, "date", "sbir", "amount"), _count(aw, "date", "sbir", "amount", by="phase")]
    parts.append(aw.assign(year=aw.date.dt.year).groupby(["firm_id_sbir", "year"]).employees.max()
                 .rename("employees_reported").to_frame())

    # Regulation D offerings, by the year of the first filing, with the size the issuer reports
    fd = _read(paths, "formd_offerings_firm")
    if fd is not None:
        fd = fd[~fd.is_bizcomb.fillna(False)]
        parts.append(_count(fd, "first_filing", "formd", "amount_sold"))
        parts.append(_count(fd[fd.is_equity.fillna(False)], "first_filing", "formd_equity", "amount_sold"))
        y = fd.assign(year=fd.first_filing.dt.year)
        parts.append(y.groupby(["firm_id_sbir", "year"]).agg(formd_investors_max=("investors", "max"),
                                                              formd_revenue_range=("revenue_range", "last")))

    # SEC filings: years as a public filer, current reports, ownership stakes, listing events
    ed = _read(paths, "firm_edgar_filings", "sec")
    if ed is not None:
        e = ed.assign(year=ed.date.dt.year)
        flags = {"public_filer": ["periodic_report", "quarterly_report"], "current_reports": ["current_report"],
                 "stake_13d": ["ownership_13d"], "stake_13g": ["ownership_13g"],
                 "ipo_prospectus": ["ipo_prospectus"], "registration_filed": ["registration"],
                 "delisting": ["deregistration", "delisting"]}
        cols = {k: e[e.event.isin(v)].groupby(["firm_id_sbir", "year"]).size() for k, v in flags.items()}
        x = pd.DataFrame(cols)
        x["public_filer"] = x.public_filer.fillna(0) > 0
        parts.append(x)

    # Regulation Crowdfunding and SBA loans, from the event table
    ev = _read(paths, "firm_events")
    if ev is not None:
        parts.append(_count(ev[(ev.source == "formc") & (ev.event == "regcf_c")], "date", "formc", "amount"))
        parts.append(_count(ev[(ev.source == "sba") & ev.event.str.endswith("_high")], "date", "sba", "amount"))

    # Federal contracts and grants (name-verified), split into SBIR, Phase III and other work
    fa = _read(paths, "federal_awards")
    if fa is not None:
        if "recipient_name_matches" in fa:
            fa = fa[fa.recipient_name_matches]
        fa = fa.assign(kind=np.where(fa.award_class == "non_sbir", "other_" + fa.award_group, fa.award_class))
        parts.append(_count(fa, "date", "fed", "amount", by="kind"))

    # Patents by filing and grant year, five-year forward citations by filing year, applications
    fp = _read(paths, "firm_patents")
    if fp is not None:
        parts.append(_count(fp, "filing_date", "patents_filed"))
        parts.append(_count(fp, "grant_date", "patents_granted"))
        c = fp.assign(year=fp.filing_date.dt.year).groupby(["firm_id_sbir", "year"])
        parts.append(c.fwd_cites_5y.sum(min_count=1).rename("patents_filed_fwd_cites_5y").to_frame()
                     if "fwd_cites_5y" in fp else None)
        if "gov_interest_cites_sbir" in fp:
            parts.append(_count(fp[fp.gov_interest_cites_sbir.fillna(False).astype(bool)], "filing_date", "patents_sbir_funded"))
    pgf = paths.interim / "patents" / "firm_pgpubs.parquet"
    if pgf.exists():
        parts.append(_count(pd.read_parquet(pgf), "filing_date", "patent_apps_filed"))
    tr = _read(paths, "patent_transfers")
    if tr is not None:
        tr = tr.drop_duplicates(["firm_id_sbir", "rf_id"])
        parts.append(_count(tr, "date", "patent", by="transfer_type"))

    # Financial statements of public filers
    fin = _read(paths, "firm_financials_year")
    if fin is not None and len(fin):
        keep = [c for c in ("revenue", "rnd_expense", "net_income", "operating_income", "assets", "cash",
                            "stockholders_equity") if c in fin]
        parts.append(fin.groupby(["firm_id_sbir", "year"])[keep].first())

    data = pd.concat([p for p in parts if p is not None], axis=1)
    data.index = data.index.set_names(["firm_id_sbir", "year"])

    # the years each firm is observed: from its start to END_YEAR
    seen = data.reset_index().groupby("firm_id_sbir").year.min()
    start = pd.concat([seen, founded.reindex(seen.index.union(founded.index))], axis=1).min(axis=1)
    start = start.reindex(firms.firm_id_sbir).fillna(first_award.reindex(firms.firm_id_sbir)).clip(lower=EARLIEST)
    lengths = (END_YEAR - start + 1).clip(lower=1).astype(int)
    panel = pd.DataFrame({"firm_id_sbir": np.repeat(start.index.values, lengths.values),
                          "year": np.concatenate([np.arange(s, END_YEAR + 1) for s in start.astype(int).values])})
    panel = panel.merge(data.reset_index(), on=["firm_id_sbir", "year"], how="left")

    # timing relative to the program and to the firm
    panel["first_award_year"] = panel.firm_id_sbir.map(first_award)
    panel["event_time"] = panel.year - panel.first_award_year
    panel["post_first_award"] = panel.event_time > 0
    panel["founded_year"] = panel.firm_id_sbir.map(founded)
    panel["firm_age"] = panel.year - panel.founded_year
    panel["before_founding"] = panel.firm_age < 0

    # coverage: counts are zero inside a source's window and missing outside it
    for src, (lo, hi) in COVERAGE.items():
        panel[f"cov_{src}"] = panel.year.between(lo, hi)
    groups = {"sbir": "sbir", "formd": "formd", "formc": "formc", "sba": "sba", "fed": "federal",
              "patents_filed": "patents", "patents_granted": "patents", "patents_sbir_funded": "patents",
              "patent_apps_filed": "patents", "patent_": "patents", "public_filer": "edgar", "current_reports": "edgar",
              "stake_": "edgar", "ipo_prospectus": "edgar", "registration_filed": "edgar", "delisting": "edgar"}
    for col in panel.columns:
        src = next((v for k, v in groups.items() if col.startswith(k)), None)
        if src is None or col in ("formd_revenue_range", "formd_investors_max"):
            continue
        cov = panel[f"cov_{src}"]
        if col == "public_filer":
            panel[col] = panel[col].astype("boolean").fillna(False).where(cov)
        else:
            panel[col] = panel[col].fillna(0).where(cov)
    panel["cov_patent_grants_complete"] = panel.year <= END_YEAR - PATENT_GRANT_LAG
    panel["cov_xbrl"] = panel.year.between(*COVERAGE["xbrl"]) & panel.public_filer.fillna(False).astype(bool)

    # SAM registration in the years a vintage exists
    sam = _read(paths, "firm_sam")
    if sam is not None:
        vint = {int(c.split("_")[2]): c for c in sam.columns if c.startswith("sam_registered_2")}
        reg = sam.set_index("firm_id_sbir")
        panel["sam_registered"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
        for y, c in vint.items():
            m = panel.year == y
            panel.loc[m, "sam_registered"] = panel.loc[m, "firm_id_sbir"].map(reg[c]).astype("boolean").values

    # location in each year: the latest SBIR address in use by then, and the first one before that
    ad = _read(paths, "sbir_firm_addresses")
    ad = ad.assign(year=ad.first_used.dt.year).dropna(subset=["year", "zip5"])
    ad = ad.sort_values(["first_used", "firm_id_sbir"], kind="stable").drop_duplicates(["firm_id_sbir", "year"], keep="last")
    ad["year"] = ad.year.astype(int)
    loc = pd.merge_asof(panel[["firm_id_sbir", "year"]].reset_index().sort_values("year"),
                        ad[["firm_id_sbir", "year", "zip5", "state", "city"]].sort_values("year"),
                        on="year", by="firm_id_sbir", direction="backward").set_index("index").sort_index()
    firstloc = ad.drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
    for c in ("zip5", "state", "city"):
        panel[c] = loc[c].fillna(panel.firm_id_sbir.map(firstloc[c]))

    # place measures for that location and year
    acs_z = _read(paths, "zcta_characteristics")
    c2000 = _read(paths, "zcta_characteristics_2000")
    if acs_z is not None or c2000 is not None:
        acs_z = acs_z if acs_z is not None else pd.DataFrame()
        c2000 = c2000 if c2000 is not None else pd.DataFrame()
        pv = places.place_by_year(panel[["zip5", "year"]].dropna(), acs_z, c2000)
        pv = pv.rename(columns={m: f"zip_{m}" for m in places.PANEL_MEASURES})
        panel = panel.merge(pv, on=["zip5", "year"], how="left")

    # price index to convert any dollar column to 2025 dollars
    cpi_f = paths.raw / "macro" / "CPIAUCSL.csv"
    if cpi_f.exists():
        cpi = pd.read_csv(cpi_f)
        cpi.columns = ["date", "cpi"]
        by_year = cpi.groupby(pd.to_datetime(cpi.date).dt.year).cpi.mean()
        panel["deflator_to_2025"] = by_year.get(2025, by_year.iloc[-1]) / panel.year.map(by_year)

    panel = panel.sort_values(["firm_id_sbir", "year"]).reset_index(drop=True)
    panel.to_parquet(paths.processed / "firm_year_panel.parquet", index=False)

    pre = panel[panel.event_time < 0]
    stats = {
        "firm_years": int(len(panel)), "firms": int(panel.firm_id_sbir.nunique()),
        "years": [int(panel.year.min()), int(panel.year.max())],
        "median_years_before_first_award": float(pre.groupby("firm_id_sbir").size().median()) if len(pre) else 0,
        "firms_with_pre_award_years": int(pre.firm_id_sbir.nunique()),
        "columns": len(panel.columns),
        "place_source": panel.place_source.value_counts().to_dict() if "place_source" in panel else {},
    }
    (paths.audit / "panel.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("panel: %s firm-years for %s firms, %s to %s", f"{len(panel):,}", f"{stats['firms']:,}", *stats["years"])
