"""Outcome and covariate fields, the firm and event tables, and the codebook.

Post-award means strictly after the firm's first SBIR/STTR award date. Each SEC outcome has a
coverage flag because the SEC sources begin at different dates (see config).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import geo
from . import io as sio
from . import names as nm
from .config import (EDGAR_COVERAGE_START, FORMC_COVERAGE_START, FORMD_COVERAGE_START, OZ_DESIGNATION_YEAR,
                     RADIUS_MILES, Paths)

log = logging.getLogger(__name__)

EDGAR_EVENTS = {
    "424B1": "ipo_prospectus", "424B4": "ipo_prospectus", "424B3": "prospectus_other",
    "8-A12B": "exchange_listing", "10-12B": "exchange_listing", "10-12B/A": "exchange_listing",
    "8-A12G": "exchange_registration", "10-12G": "form10_registration", "10-12G/A": "form10_registration",
    "8-K12B": "successor_registration", "8-K12G3": "successor_registration",
    "S-1": "registration", "S-1/A": "registration", "S-1MEF": "registration", "F-1": "registration",
    "F-1/A": "registration", "F-1MEF": "registration", "SB-2": "registration", "SB-2/A": "registration",
    "RW": "registration_withdrawn", "EFFECT": "registration_effective",
    "10-K": "periodic_report", "10-K405": "periodic_report", "10-KSB": "periodic_report", "20-F": "periodic_report",
    "40-F": "periodic_report", "10-Q": "quarterly_report", "10-QSB": "quarterly_report", "8-K": "current_report",
    "SC 13D": "ownership_13d", "SC 13D/A": "ownership_13d", "SCHEDULE 13D": "ownership_13d",
    "SCHEDULE 13D/A": "ownership_13d", "SC 13G": "ownership_13g", "SC 13G/A": "ownership_13g",
    "SCHEDULE 13G": "ownership_13g", "SCHEDULE 13G/A": "ownership_13g",
    "SC TO-T": "tender_offer_target", "SC TO-T/A": "tender_offer_target", "SC 14D9": "tender_offer_target",
    "SC 14D9/A": "tender_offer_target", "DEFM14A": "merger_proxy", "DEFM14C": "merger_proxy",
    "S-4": "merger_registration", "S-4/A": "merger_registration", "F-4": "merger_registration",
    "F-4/A": "merger_registration", "425": "merger_communication",
    "15-12B": "deregistration", "15-12G": "deregistration", "15-15D": "deregistration", "25-NSE": "delisting",
    "25": "delisting",
    "1-A": "reg_a_offering", "1-A/A": "reg_a_offering", "253G2": "reg_a_offering", "1-K": "reg_a_annual",
}
SPAC_NAME = r"\bACQUISITIONS?\b|\bSPAC\b|BLANK CHECK|\bMERGER (?:CORP|SUB)"


def despac_dates(paths: Paths, ciks: pd.Series) -> pd.Series:
    """For CIKs that began as a SPAC (blank-check company) and took an operating company's name,
    the date the SPAC name was dropped, which is when the merger closed. Filings before it belong
    to the SPAC, not the operating company."""
    f = paths.sec / "edgar_former_names.parquet"
    if not f.exists():
        return pd.Series(dtype="datetime64[ns]")
    fn = pd.read_parquet(f)
    fn["cik_s"] = fn.cik.astype("string").str.lstrip("0")
    fn = fn[fn.cik_s.isin(set(ciks))]
    spac = fn[fn.name.astype("string").str.upper().str.contains(SPAC_NAME, regex=True, na=False)]
    ent = pd.read_parquet(paths.sec / "edgar_entities.parquet", columns=["cik", "name", "sic"])
    ent["cik_s"] = ent.cik.astype("string").str.lstrip("0")
    still_spac = set(ent[(ent.sic.astype("string") == "6770")
                         | ent.name.astype("string").str.upper().str.contains(SPAC_NAME, regex=True, na=False)].cik_s)
    spac = spac[~spac.cik_s.isin(still_spac)]
    return sio.to_date(spac["to"]).groupby(spac.cik_s).max().dropna()


def offering_root(df: pd.DataFrame) -> pd.Series:
    """Offering ID: each amendment's PREVIOUSACCESSIONNUMBER chain followed back to the original notice."""
    prev = sio.col(df, "PREVIOUSACCESSIONNUMBER").astype("string")
    parent = {a: p for a, p in zip(df.ACCESSIONNUMBER, prev) if isinstance(p, str) and p and p != a}

    def root(a):
        for _ in range(100):
            if a not in parent:
                break
            a = parent[a]
        return a
    return df.ACCESSIONNUMBER.map(root)


def _formd_filings(paths: Paths) -> pd.DataFrame:
    sub = pd.read_parquet(paths.sec / "formd_formdsubmission.parquet")
    iss = pd.read_parquet(paths.sec / "formd_issuers.parquet", columns=["ACCESSIONNUMBER", "CIK", "ZIPCODE"])
    off = pd.read_parquet(paths.sec / "formd_offering.parquet")
    for d in (sub, off):
        d.drop(columns=[c for c in d.columns if c == "QUARTER"], inplace=True)
    x = iss.merge(sub, on="ACCESSIONNUMBER", how="left").merge(off, on="ACCESSIONNUMBER", how="left")
    x["cik_s"] = x.CIK.astype("string").str.lstrip("0")
    x["filing_date"] = sio.to_date(sio.col(x, "FILING_DATE"))
    x["sale_date"] = sio.to_date(sio.col(x, "SALE_DATE"))
    x["amount_sold"] = sio.to_num(sio.col(x, "TOTALAMOUNTSOLD"))
    truth = lambda c: sio.col(x, c).astype("string").str.lower().eq("true").fillna(False)
    x["is_equity"], x["is_bizcomb"] = truth("ISEQUITYTYPE"), truth("ISBUSINESSCOMBINATIONTRANS")
    x["is_fund"] = sio.col(x, "INDUSTRYGROUPTYPE").eq("Pooled Investment Fund").fillna(False)
    x["amount_offered"] = sio.to_num(sio.col(x, "TOTALOFFERINGAMOUNT"))
    x["investors"] = sio.to_num(sio.col(x, "TOTALNUMBERALREADYINVESTED"))
    x["revenue_range"] = sio.col(x, "REVENUERANGE").astype("string")
    x["industry_group"] = sio.col(x, "INDUSTRYGROUPTYPE").astype("string")
    x["offering_id"] = offering_root(x)
    return x


def formd(paths: Paths, fc: pd.DataFrame, anchor: pd.Series, events: list) -> tuple[pd.DataFrame, dict]:
    fd = _formd_filings(paths)
    fd = fd[~fd.is_fund].merge(fc[["firm_id_sbir", "cik_s"]], on="cik_s")
    last_text = lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NA
    offr = (fd.sort_values("filing_date").groupby(["firm_id_sbir", "offering_id"])
              .agg(first_filing=("filing_date", "min"), first_sale=("sale_date", "min"),
                   amount_sold=("amount_sold", "max"), amount_offered=("amount_offered", "max"),
                   investors=("investors", "max"), revenue_range=("revenue_range", last_text),
                   industry_group=("industry_group", last_text), is_equity=("is_equity", "max"),
                   is_bizcomb=("is_bizcomb", "max"))
              .reset_index())
    offr.to_parquet(paths.processed / "formd_offerings_firm.parquet", index=False)
    offr["award0"] = offr.firm_id_sbir.map(anchor)
    offr["start"] = offr.first_sale.fillna(offr.first_filing)
    offr["start_src"] = np.where(offr.first_sale.notna(), "date_of_first_sale", "filing_date")
    offr["post"] = offr.first_filing > offr.award0
    offr["amount_missing"] = offr.amount_sold.isna() | (offr.amount_sold <= 0)

    post = offr[offr.post]
    eq = offr[offr.is_equity & ~offr.is_bizcomb & (offr.start >= offr.award0)].sort_values("start")
    first_eq = eq.drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
    out = pd.DataFrame({
        "has_form_d": post.groupby("firm_id_sbir").size() > 0,
        "has_form_d_pre": offr[~offr.post].groupby("firm_id_sbir").size() > 0,
        "formd_amount_sold_pre": offr[~offr.post].groupby("firm_id_sbir").amount_sold.sum(min_count=1),
        "formd_revenue_range_pre": offr[~offr.post].sort_values("first_filing").groupby("firm_id_sbir").revenue_range.last(),
        "formd_first_filing_post_date": post.groupby("firm_id_sbir").first_filing.min(),
        "formd_n_offerings_post": post.groupby("firm_id_sbir").size(),
        "formd_amount_sold_post": post.groupby("firm_id_sbir").amount_sold.sum(min_count=1),
        "formd_n_offerings_amount_missing": post.groupby("firm_id_sbir").amount_missing.sum(),
        "formd_bizcomb_post": post.groupby("firm_id_sbir").is_bizcomb.max(),
        "first_private_equity_date": first_eq.start,
        "years_to_private_cap": (first_eq.start - first_eq.award0).dt.days / 365.25,
        "years_to_private_cap_source": first_eq.start_src,
    })
    kind = np.where(offr.is_bizcomb, "regd_business_combination", np.where(offr.is_equity, "regd_equity", "regd_other"))
    events.append(offr.assign(source="formd", event=kind)[["firm_id_sbir", "start", "source", "event", "amount_sold"]]
                  .rename(columns={"start": "date", "amount_sold": "amount"}))
    stats = {"offerings_post": int(len(post)),
             "offerings_post_amount_missing_share": round(float(post.amount_missing.mean()), 4) if len(post) else None,
             "first_sale_date_missing_share": round(float(offr.first_sale.isna().mean()), 4) if len(offr) else None}
    return out, stats


def formc(paths: Paths, fc: pd.DataFrame, anchor: pd.Series, events: list) -> pd.DataFrame:
    cs = pd.read_parquet(paths.sec / "formc_form_c_submission.parquet")
    cd = pd.read_parquet(paths.sec / "formc_form_c_disclosure.parquet")
    x = cs.assign(cik_s=sio.col(cs, "CIK").astype("string").str.lstrip("0")).merge(fc[["firm_id_sbir", "cik_s"]], on="cik_s")
    x = x.merge(pd.DataFrame({"ACCESSION_NUMBER": cd.ACCESSION_NUMBER, "target": sio.to_num(sio.col(cd, "OFFERINGAMOUNT"))}),
                on="ACCESSION_NUMBER", how="left")
    x["filing_date"] = sio.to_date(sio.col(x, "FILING_DATE"))
    x["type"] = sio.col(x, "SUBMISSION_TYPE").astype("string").str.upper()
    x["post"] = x.filing_date > x.firm_id_sbir.map(anchor)
    orig = x[x.type.eq("C")]
    events.append(x.assign(source="formc", event="regcf_" + x.type.str.lower().str.replace("/", "_"))
                  [["firm_id_sbir", "filing_date", "source", "event", "target"]]
                  .rename(columns={"filing_date": "date", "target": "amount"}))
    return pd.DataFrame({
        "has_form_c": orig[orig.post].groupby("firm_id_sbir").size() > 0,
        "has_form_c_pre": orig[~orig.post].groupby("firm_id_sbir").size() > 0,
        "formc_n_offerings_post": orig[orig.post].groupby("firm_id_sbir").size(),
        "formc_target_amount_post": orig[orig.post].groupby("firm_id_sbir").target.sum(min_count=1),
        "formc_progress_update": x[x.type.eq("C-U")].groupby("firm_id_sbir").size() > 0,
    })


def edgar(paths: Paths, fc: pd.DataFrame, anchor: pd.Series, events: list) -> pd.DataFrame:
    fil = pd.read_parquet(paths.sec / "edgar_filings.parquet")
    fil = fil.assign(cik_s=fil.cik.astype("string").str.lstrip("0")).merge(fc[["firm_id_sbir", "cik_s"]], on="cik_s")
    fil["date"], fil["event"] = sio.to_date(fil.filingDate), fil.form.map(EDGAR_EVENTS)
    fil = fil.dropna(subset=["event", "date"])
    # a CIK that started as a SPAC: its filings before the merger are the SPAC's own
    ds = despac_dates(paths, fc.cik_s)
    fil["despac_date"] = fil.cik_s.map(ds)
    spac_period = fil.despac_date.notna() & (fil.date < fil.despac_date)
    fil = fil[~spac_period]
    fil["post"] = fil.date > fil.firm_id_sbir.map(anchor)
    first = lambda kinds, d=fil: d[d.event.isin(kinds) & d.post].groupby("firm_id_sbir").date.min()

    spac = (fc.assign(d=fc.cik_s.map(ds)).dropna(subset=["d"]).groupby("firm_id_sbir").d.min())
    spac = spac[spac > spac.index.map(anchor)]
    ipo = first(["ipo_prospectus"])
    listing = first(["exchange_listing"])
    out = pd.DataFrame({
        "ipo_prospectus_date": ipo, "exchange_listing_date": listing, "spac_merger_date": spac,
        "acquired_public_date": first(["tender_offer_target", "merger_proxy"]),
        "regulation_a_date": first(["reg_a_offering"]),
        "deregistered_date": first(["deregistration", "delisting"]),
        "ipo_registration_date": first(["registration"]),
        "registration_withdrawn_date": first(["registration_withdrawn"]),
        "first_periodic_report_date": fil[fil.event.isin(["periodic_report", "quarterly_report"])].groupby("firm_id_sbir").date.min(),
        "last_periodic_report_date": fil[fil.event.isin(["periodic_report", "quarterly_report"])].groupby("firm_id_sbir").date.max(),
        "years_filing_periodic_reports": fil[fil.event.isin(["periodic_report", "quarterly_report"])]
                                         .assign(y=lambda d: d.date.dt.year).groupby("firm_id_sbir").y.nunique(),
        "public_before_award": fil[fil.event.isin(["periodic_report", "quarterly_report", "exchange_listing"]) & ~fil.post]
                               .groupby("firm_id_sbir").size() > 0,
    })
    routes = out[["ipo_prospectus_date", "exchange_listing_date", "spac_merger_date"]]
    out["went_public_date"] = routes.min(axis=1)
    # An IPO's exchange registration (8-A12B) is usually filed a few days before its final
    # prospectus, so a prospectus within 120 days of the first public filing marks an IPO.
    near_ipo = (out.ipo_prospectus_date - out.went_public_date).dt.days.abs() <= 120
    out["went_public_route"] = np.select(
        [out.went_public_date.notna() & (out.went_public_date == out.spac_merger_date),
         out.went_public_date.notna() & near_ipo.fillna(False),
         out.went_public_date.notna()], ["spac_merger", "ipo", "listing"], None)
    out["went_public_route"] = out.went_public_route.where(out.went_public_date.notna())
    # an S-1 or F-1 after the award that was withdrawn, with no listing within three years
    out["ipo_withdrawn"] = (out.registration_withdrawn_date.notna() & out.ipo_registration_date.notna()
                            & ~(out.went_public_date <= out.ipo_registration_date + pd.DateOffset(years=3)).fillna(False))
    for c in ("went_public", "acquired_public", "regulation_a", "deregistered"):
        out[c if c != "acquired_public" else "acquired_public_signal"] = out[f"{c}_date"].notna()
    exit_date = out[["went_public_date", "acquired_public_date"]].min(axis=1)
    out["public_market_exit"] = exit_date.notna()
    out["public_market_exit_date"] = exit_date
    out["public_market_exit_type"] = np.select(
        [out.went_public_date.notna() & (out.went_public_date <= out.acquired_public_date.fillna(pd.Timestamp.max)),
         out.acquired_public_date.notna()], ["went_public", "acquired_while_public"], None)
    ev = fil[~fil.event.isin(["current_report", "quarterly_report"])]       # the panel counts these by year
    events.append(ev.assign(source="edgar", amount=np.nan)[["firm_id_sbir", "date", "source", "event", "amount"]])
    if len(spac):
        events.append(pd.DataFrame({"firm_id_sbir": spac.index, "date": spac.values, "source": "edgar",
                                    "event": "spac_merger", "amount": np.nan}))
    fil[["firm_id_sbir", "cik_s", "date", "form", "event"]].to_parquet(paths.interim / "sec" / "firm_edgar_filings.parquet",
                                                                     index=False)
    return out


def sba(paths: Paths, anchor: pd.Series, events: list) -> pd.DataFrame | None:
    f = paths.interim / "sba" / "sba_loans.parquet"
    if not f.exists():
        log.warning("SBA loans not available; SBA fields left empty")
        return None
    loans = pd.read_parquet(f)
    loans["state"], loans["zip5"], loans["city"] = (nm.clean_state(loans.borr_state), nm.clean_zip(loans.borr_zip),
                                                    nm.clean_city(loans.borr_city))
    loans = nm.add_keys(loans.assign(name_raw=loans.borr_name))
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "company_name", "city", "state", "zip5"])
    sbn = nm.add_keys(aw.rename(columns={"company_name": "name_raw"}).assign(city=lambda d: nm.clean_city(d.city)))
    sbn = sbn[sbn.linkable][["firm_id_sbir", "name_key", "state", "city", "zip5"]].drop_duplicates()
    keys = ["name_key", "state", "city", "zip5"]
    m = loans[loans.linkable][keys].drop_duplicates().merge(sbn, on=["name_key", "state"], suffixes=("", "_sb"))
    m["local"] = sio.same(m.zip5, m.zip5_sb) | sio.same(m.city, m.city_sb)
    m = m.sort_values("local", ascending=False).drop_duplicates(keys + ["firm_id_sbir"])
    m["sba_link_confidence"] = np.where(m.local, "high", "medium")
    m = m[m.groupby(keys).firm_id_sbir.transform("nunique") == 1]
    loans = loans.merge(m[keys + ["firm_id_sbir", "sba_link_confidence"]], on=keys)
    loans["date"], loans["amount"] = sio.to_date(loans.approval_date), sio.to_num(loans.gross_approval)
    loans["post"] = loans.date > loans.firm_id_sbir.map(anchor)
    loans["charged_off"] = sio.to_date(sio.col(loans, "chargeoff_date")).notna()
    hi = loans[loans.sba_link_confidence == "high"]
    events.append(loans.assign(source="sba", event="sba_" + loans.program.str.lower() + "_" + loans.sba_link_confidence)
                  [["firm_id_sbir", "date", "source", "event", "amount"]])
    return pd.DataFrame({
        "has_sba_loan": hi[hi.post].groupby("firm_id_sbir").size() > 0,
        "has_sba_loan_pre": hi[~hi.post].groupby("firm_id_sbir").size() > 0,
        "sba_n_loans_post": hi[hi.post].groupby("firm_id_sbir").size(),
        "sba_amount_post": hi[hi.post].groupby("firm_id_sbir").amount.sum(min_count=1),
        "sba_chargeoff_ever": hi.groupby("firm_id_sbir").charged_off.max(),
        "has_sba_loan_medium": loans[loans.post].groupby("firm_id_sbir").size() > 0,
    })


def linked_tables(paths: Paths, events: list) -> list[pd.DataFrame]:
    """Firm fields and dated events from the federal, SAM and patent steps, when they have run."""
    parts = []
    for name in ("firm_federal", "firm_sam", "firm_places", "firm_financials", "firm_patents_summary"):
        f = paths.processed / f"{name}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f).set_index("firm_id_sbir"))
        else:
            log.warning("%s not found; run its step to add those fields", f.name)
    fa = paths.processed / "federal_awards.parquet"
    if fa.exists():
        a = pd.read_parquet(fa)
        if "recipient_name_matches" in a:
            a = a[a.recipient_name_matches]
        a = a[a.award_class != "sbir_program"]
        events.append(a.assign(source="usaspending", event=np.where(a.award_class == "sbir_phase3", "sbir_phase3_contract",
                                                                     "federal_" + a.award_group))
                      [["firm_id_sbir", "date", "source", "event", "amount"]])
    fp = paths.processed / "firm_patents.parquet"
    if fp.exists():
        p = pd.read_parquet(fp, columns=["firm_id_sbir", "filing_date", "grant_date"])
        for date, kind in (("filing_date", "patent_filed"), ("grant_date", "patent_granted")):
            events.append(pd.DataFrame({"firm_id_sbir": p.firm_id_sbir, "date": p[date], "source": "uspto",
                                        "event": kind, "amount": np.nan}))
    tr = paths.processed / "patent_transfers.parquet"
    if tr.exists():
        t = pd.read_parquet(tr).drop_duplicates(["firm_id_sbir", "rf_id"])
        events.append(t.assign(source="uspto_assignment", event="patent_" + t.transfer_type, amount=np.nan)
                      [["firm_id_sbir", "date", "source", "event", "amount"]])
    return parts


def capital_density(paths: Paths, firms: pd.DataFrame) -> pd.DataFrame:
    """Reg D dollars sold by non-fund issuers within RADIUS_MILES of the first-award ZIP, first-award year."""
    from sklearn.neighbors import BallTree

    zc = pd.read_parquet(paths.interim / "geo" / "zcta_centroids.parquet").dropna().drop_duplicates("zip5").reset_index(drop=True)
    x = _formd_filings(paths)
    x = x[~x.is_fund].assign(zip5=lambda d: nm.clean_zip(d.ZIPCODE))
    o = x.groupby("offering_id").agg(zip5=("zip5", "first"), date=("filing_date", "min"), amt=("amount_sold", "max"))
    o["year"] = o.date.dt.year
    zy = o.dropna(subset=["zip5", "year"]).groupby(["zip5", "year"]).agg(regd_amt=("amt", "sum"), regd_n=("amt", "size")).reset_index()

    tree = BallTree(np.radians(zc[["lat", "lon"]].values), metric="haversine")
    f = firms[["firm_id_sbir", "zip5", "first_award_date"]].merge(zc, on="zip5", how="left")
    f["year"] = f.first_award_date.dt.year
    f = f[f.lat.notna()]
    nbrs = tree.query_radius(np.radians(f[["lat", "lon"]].values), r=RADIUS_MILES / 3958.8)
    sizes = [len(n) for n in nbrs]
    pairs = pd.DataFrame({"firm_id_sbir": np.repeat(f.firm_id_sbir.values, sizes),
                          "year": np.repeat(f.year.values, sizes),
                          "zip5": zc.zip5.values[np.concatenate(nbrs)] if sizes else []})
    dens = (pairs.merge(zy, on=["zip5", "year"]).groupby("firm_id_sbir")
            .agg(local_capital_density=("regd_amt", "sum"), local_regd_offerings=("regd_n", "sum")))
    dens = dens.reindex(f.firm_id_sbir).fillna(0)
    dens.loc[f.set_index("firm_id_sbir").year.reindex(dens.index) < FORMD_COVERAGE_START.year] = np.nan
    return dens


def macro(paths: Paths, firms: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=firms.firm_id_sbir)
    ff_path = paths.interim / "macro" / "fedfunds_monthly.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path)
        rate = ff.set_index(ff.date.dt.to_period("M")).fedfunds
        out["macro_fed_funds_rate"] = firms.first_award_date.dt.to_period("M").map(rate).values
    cpi_path = paths.raw / "macro" / "CPIAUCSL.csv"
    if cpi_path.exists():
        cpi = pd.read_csv(cpi_path)
        cpi.columns = ["date", "cpi"]
        by_year = cpi.groupby(pd.to_datetime(cpi.date).dt.year).cpi.mean()
        base = by_year.get(2025, by_year.iloc[-1])
        aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "award_year", "amount"])
        aw["real"] = aw.amount * base / aw.award_year.astype("float").map(by_year)
        out["total_grant_funding_real"] = aw.groupby("firm_id_sbir").real.sum(min_count=1)
    else:
        log.warning("CPIAUCSL.csv not found; total_grant_funding_real left empty")
    return out


CODEBOOK = [
    ("firm_id_sbir", "string", "SBIR.gov", "Firm identifier built from awards sharing a UEI, DUNS, website, or strict name and location.", "static"),
    ("firm_cik_sec", "string(10)", "SEC EDGAR", "Primary SEC Central Index Key, zero-padded. Best link tier, non-fund preferred.", "static"),
    ("link_tier", "int", "derived", "Linkage tier of the primary CIK: 1 to 4 high confidence, 5 to 6 medium (docs/methods.md).", "static"),
    ("sbir_phase_flag", "int", "SBIR.gov", "1 Phase I only; 2 Phase II without an observed Phase I; 3 Phase I followed by Phase II.", "full history"),
    ("total_grant_funding", "float", "SBIR.gov", "Sum of SBIR/STTR award amounts, nominal dollars.", "full history"),
    ("total_grant_funding_real", "float", "SBIR.gov, BLS CPI-U", "Sum of award amounts in 2025 dollars.", "full history"),
    ("has_form_d", "bool", "SEC Form D", "A new Regulation D notice from a non-fund issuer after the first award. A lower bound on private fundraising: not every round files a Form D.", "after first award"),
    ("formd_amount_sold_post", "float", "SEC Form D", "Sum over post-award offerings of the largest amount sold reported on any filing in the offering. Lower bound: amounts are often omitted.", "after first award"),
    ("years_to_private_cap", "float", "SEC Form D", "Years from first award to the date of first sale of the first post-award equity offering, excluding business combinations.", "after first award"),
    ("has_form_c", "bool", "SEC Form C", "An original Regulation Crowdfunding offering (Form C) after the first award. Form C reports no amount sold.", "after first award"),
    ("public_market_exit", "bool", "SEC EDGAR", "After the first award, the firm went public (IPO prospectus 424B1/424B4 or exchange listing 8-A12B) or, while public, was the target of a tender offer or merger vote (SC TO-T, SC 14D9, DEFM14A). Acquisitions of private firms are not observable.", "after first award"),
    ("went_public_route", "string", "SEC EDGAR", "How the firm went public: ipo (first 424B1 or 424B4 prospectus), spac_merger (its CIK began as a blank-check company and took the firm's name), or listing (an 8-A12B or Form 10-12B exchange registration with no IPO prospectus within 120 days, such as a direct listing or a move up from over-the-counter trading).", "after first award"),
    ("spac_merger_date", "date", "SEC EDGAR", "Date the linked CIK dropped its SPAC name, when the merger closed. The SPAC's own earlier filings are excluded from every other field.", "after first award"),
    ("ipo_registration_date", "date", "SEC EDGAR", "First S-1, F-1 or SB-2 registration after the first award.", "after first award"),
    ("ipo_withdrawn", "bool", "SEC EDGAR", "A registration after the first award was withdrawn (form RW) and the firm did not list within three years.", "after first award"),
    ("years_filing_periodic_reports", "int", "SEC EDGAR", "Calendar years with a 10-K, 10-Q, 20-F or 40-F, a measure of time spent as a public company.", "full history"),
    ("public_market_exit_type", "string", "SEC EDGAR", "went_public or acquired_while_public, whichever came first.", "after first award"),
    ("public_before_award", "bool", "SEC EDGAR", "The firm filed a 10-K, 20-F or exchange listing before its first award.", "before first award"),
    ("regulation_a", "bool", "SEC EDGAR", "A Regulation A offering (1-A, 253G2) after the first award.", "after first award"),
    ("has_sba_loan", "bool", "SBA 7(a), 504 FOIA", "An SBA 7(a) or 504 loan matched at high confidence and approved after the first award.", "after first award"),
    ("hubzone_status", "bool", "SBIR.gov", "HUBZone-owned flag on the firm's first award. Missing for awards before 1999, when HUBZone certification began.", "at first award"),
    ("opportunity_zone_flag", "bool", "Census geocoder, CDFI Fund", "The firm's geocoded address lies in a 2018-designated Qualified Opportunity Zone (2010 tract). Missing when no address could be placed.", "at first award"),
    ("tract_geoid_2010", "string(11)", "Census geocoder", "2010 census tract of the firm's address.", "at first award"),
    ("geo_source", "string", "derived", "Which address placed the firm: first_award_address, first_award_address_cleaned or other_award_address.", "static"),
    ("geo_fail_reason", "string", "derived", "Why no address could be placed: po_box, no_street_number, blank_address, zip_not_a_zcta, non_us_or_missing_state, tie or no_match.", "static"),
    ("oz_zip_share", "float", "Census ZCTA-tract relationship file, CDFI Fund", "Share of the 2010 population of the firm's ZIP (ZCTA) living in Opportunity Zone tracts. Available for unplaced firms too.", "at first award"),
    ("local_capital_density", "float", "SEC Form D, Census ZCTA", "Reg D dollars sold by non-fund issuers within 15 miles of the first-award ZIP in the first-award year; 2009 onward.", "first award year"),
    ("macro_fed_funds_rate", "float", "FRED FEDFUNDS", "Effective federal funds rate in the month of the first award.", "first award month"),
    ("fed_contracts_post_n", "int", "USAspending", "Federal prime contracts (types A to D) starting after the first award, excluding SBIR/STTR awards themselves.", "after first award"),
    ("fed_contracts_post_amount", "float", "USAspending", "Total obligated amount of those contracts, nominal dollars.", "after first award"),
    ("fed_grants_post_n", "int", "USAspending", "Federal grants and cooperative agreements (types 02 to 05) after the first award, excluding SBIR/STTR awards.", "after first award"),
    ("fed_grants_post_amount", "float", "USAspending", "Total obligated amount of those grants, nominal dollars.", "after first award"),
    ("has_federal_followon", "bool", "USAspending", "Any non-SBIR federal contract or grant, Phase III included, after the first award. Observed from fiscal year 2008.", "after first award"),
    ("fed_n_agencies_post", "int", "USAspending", "Distinct awarding agencies of those follow-on awards.", "after first award"),
    ("sbir_phase3", "bool", "USAspending (FPDS)", "A contract coded SBIR or STTR Phase III (FPDS research code SR3 or ST3), or described as SBIR/STTR Phase III when no code is reported.", "full history, FY2008 onward"),
    ("phase3_n", "int", "USAspending (FPDS)", "Number of Phase III contracts.", "FY2008 onward"),
    ("phase3_amount", "float", "USAspending (FPDS)", "Total obligated amount of Phase III contracts, nominal dollars.", "FY2008 onward"),
    ("phase3_first_date", "date", "USAspending (FPDS)", "Base obligation date of the first Phase III contract.", "FY2008 onward"),
    ("years_to_phase3", "float", "derived", "Years from first SBIR award to first Phase III contract.", "after first award"),
    ("phase3_source", "string", "derived", "fpds_research_code when any Phase III contract carries the FPDS code, else description.", "static"),
    ("fed_any_award_found", "bool", "USAspending", "The firm's UEI or DUNS appears as recipient on any USAspending award, SBIR awards included.", "FY2008 onward"),
    ("fed_window_covered", "bool", "derived", "First award on or after 2007-10-01, so the whole post-award window is in USAspending.", "static"),
    ("founded_year", "int", "SAM.gov, SEC Form D", "Year the firm was founded: the SAM entity start date, or the Form D year of incorporation when SAM has none.", "static"),
    ("founded_source", "string", "derived", "sam_uei, sam_name_location or form_d_year_of_inc.", "static"),
    ("age_at_first_award", "int", "derived", "Year of first award minus founded_year. Negative values (flagged by founded_after_first_award) usually mark a re-registered entity.", "at first award"),
    ("founded_after_first_award", "bool", "derived", "founded_year is later than the first award year.", "static"),
    ("sam_match", "string", "SAM.gov", "How the firm was matched to SAM: uei or name_location. Missing when the firm is in no SAM vintage on disk.", "static"),
    ("sam_entity_start_date", "date", "SAM.gov", "Earliest entity start date reported across SAM vintages.", "static"),
    ("sam_registered_latest", "bool", "SAM.gov", "Registered in the most recent SAM vintage on disk. A weak survival proxy: firms without federal work let registrations lapse.", "latest vintage"),
    ("formd_year_of_inc", "int", "SEC Form D", "Earliest year of incorporation reported on the firm's Form D filings.", "static"),
    ("patents_n", "int", "PatentsView", "Granted U.S. patents owned by a high-confidence linked assignee, any filing date.", "full history"),
    ("patents_pre_n", "int", "PatentsView", "Those patents filed on or before the first award date.", "before first award"),
    ("patents_post_n", "int", "PatentsView", "Those patents filed after the first award date.", "after first award"),
    ("patents_post_5y_n", "int", "PatentsView", "Those patents filed within five years after the first award.", "five years after first award"),
    ("has_patent_post", "bool", "PatentsView", "patents_post_n is at least one.", "after first award"),
    ("years_to_first_patent", "float", "PatentsView", "Years from first award to the filing date of the first post-award patent.", "after first award"),
    ("patents_sbir_funded_post_n", "int", "PatentsView", "Post-award patents whose government-interest statement cites one of the firm's SBIR award numbers or names SBIR or STTR.", "after first award"),
    ("patents_citing_firm_sbir_award_n", "int", "PatentsView", "Granted patents, whoever owns them, whose government-interest statement cites one of the firm's SBIR award numbers.", "full history"),
    ("patents_gov_interest_post_n", "int", "PatentsView", "Post-award patents with any federal government-interest statement.", "after first award"),
    ("fwd_cites_5y_post", "float", "PatentsView", "U.S. patent citations received within five years of grant by post-award patents, summed over patents granted at least five years before the data end.", "after first award"),
    ("fwd_cites_5y_post_mean", "float", "PatentsView", "The same, per patent.", "after first award"),
    ("patent_cpc_section_modal", "string", "PatentsView", "Most common first-listed CPC section of the firm's patents.", "full history"),
    ("patent_wipo_sector_modal", "string", "PatentsView", "Most common first-listed WIPO technology sector.", "full history"),
    ("patent_n_cpc_subclasses", "int", "PatentsView", "Distinct first-listed CPC subclasses, a breadth measure.", "full history"),
    ("patent_apps_post_n", "int", "PatentsView pre-grant", "Published patent applications filed after the first award, granted or not. Publication began in 2001.", "after first award"),
    ("patent_apps_window_covered", "bool", "derived", "First award on or after 2001-03-15, when pre-grant publication began.", "static"),
    ("patents_transferred_out_n", "int", "USPTO Patent Assignment Dataset", "Owned patents assigned by the firm to another party after the first award.", "after first award"),
    ("patent_transfer_main_buyer", "string", "USPTO Patent Assignment Dataset", "Assignee that received the most of the firm's patents.", "after first award"),
    ("patent_transfer_main_buyer_share", "float", "USPTO Patent Assignment Dataset", "Share of the firm's patents transferred to that assignee.", "after first award"),
    ("patent_merger_date", "date", "USPTO Patent Assignment Dataset", "First recorded merger conveyance with the firm as assignor.", "after first award"),
    ("patent_security_interest", "bool", "USPTO Patent Assignment Dataset", "The firm pledged patents as collateral after the first award, a sign of debt financing.", "after first award"),
    ("private_acquisition_signal", "bool", "USPTO Patent Assignment Dataset", "A recorded merger, or at least half of the firm's patents (two or more) assigned to one buyer, after the first award. Captures acquisitions of private firms that SEC data miss.", "after first award"),
    ("private_acquisition_signal_date", "date", "USPTO Patent Assignment Dataset", "Date of that merger or first transfer.", "after first award"),
    ("exit_signal_any", "bool", "derived", "public_market_exit or private_acquisition_signal.", "after first award"),
    ("zip_ba_plus_share", "float", "Census ACS 5-year", "Share of adults 25 and over in the firm's ZIP (ZCTA) with a bachelor's degree or higher, from the ACS release nearest the first award year (2012 to 2024 releases).", "first award year"),
    ("zip_grad_share", "float", "Census ACS 5-year", "Share of adults 25 and over in the ZIP with a graduate or professional degree.", "first award year"),
    ("zip_median_hh_income", "float", "Census ACS 5-year", "Median household income of the ZIP, nominal dollars of the ACS release used.", "first award year"),
    ("zip_unemployment_rate", "float", "Census ACS 5-year", "Unemployed share of the civilian labor force in the ZIP.", "first award year"),
    ("zip_pop", "float", "Census ACS 5-year", "Population of the ZIP.", "first award year"),
    ("area_ba_plus_share", "float", "Census ACS 5-year", "Bachelor's-or-higher share of adults 25 and over, pooled over all ZIPs within 15 miles of the firm's ZIP.", "first award year"),
    ("area_grad_share", "float", "Census ACS 5-year", "Graduate-degree share pooled over all ZIPs within 15 miles.", "first award year"),
    ("area_pop", "float", "Census ACS 5-year", "Population of all ZIPs within 15 miles.", "first award year"),
    ("acs_year_used", "int", "derived", "Final year of the ACS 5-year release used: the first award year, clamped to the 2012 to 2024 releases. Values for firms that entered before 2012 describe their ZIP later, not at entry.", "static"),
    ("univ_n_15mi", "int", "NCES IPEDS", "Degree-granting colleges and universities within 15 miles of the firm's ZIP.", "latest directory"),
    ("univ_r1_n_15mi", "int", "NCES IPEDS, Carnegie", "R1 (very high research activity) universities within 15 miles.", "latest directory"),
    ("univ_r1r2_n_15mi", "int", "NCES IPEDS, Carnegie", "R1 or R2 doctoral research universities within 15 miles.", "latest directory"),
    ("miles_to_nearest_r1", "float", "NCES IPEDS, Carnegie", "Distance from the firm's ZIP centroid to the nearest R1 university.", "latest directory"),
    ("formd_amount_sold_pre", "float", "SEC Form D", "Amount sold in Regulation D offerings filed on or before the first award. Lower bound.", "before first award"),
    ("formd_revenue_range_pre", "string", "SEC Form D", "Revenue range the firm reported on its last Form D before the first award (often 'Decline to Disclose').", "before first award"),
    ("fed_contracts_pre_n", "int", "USAspending", "Non-SBIR federal contracts before the first award, FY2008 onward.", "before first award"),
    ("fed_contracts_pre_amount", "float", "USAspending", "Their total obligated amount.", "before first award"),
    ("revenue_latest", "float", "SEC XBRL", "Most recent annual revenue reported in XBRL by the firm's primary CIK, 2009 onward.", "latest"),
    ("rnd_expense_latest", "float", "SEC XBRL", "Most recent annual R&D expense reported in XBRL.", "latest"),
    ("ipo_proceeds_xbrl", "float", "SEC XBRL", "Proceeds from the initial public offering as reported in the cash flow statement, when tagged.", "at IPO"),
    ("xbrl_first_year", "int", "SEC XBRL", "First calendar year with XBRL financial data.", "static"),
    ("established_firm_flag", "bool", "derived", "Public before the first award, at least 50 patents filed by then, or 500 or more employees on the first award. These firms dominate totals; report medians or set them aside.", "at first award"),
    ("institution_name_flag", "bool", "derived", "The awardee name looks like a university, hospital or research institute rather than a small business.", "static"),
    ("age_implausible_flag", "bool", "derived", "age_at_first_award above 50 or below -5 years, usually a re-registered entity or a data error.", "static"),
    ("first_award_date_imputed", "bool", "SBIR.gov", "The award file gives only the year, so the date is set to July 1. True for nearly all awards before 2000 and about two-thirds in the 2000s, so timing for older firms is accurate to the year.", "static"),
    ("formd_window_covered", "bool", "derived", "First award on or after 2009-01-01, so absence of a Form D is informative.", "static"),
    ("formc_window_covered", "bool", "derived", "First award on or after 2016-05-16.", "static"),
    ("edgar_window_covered", "bool", "derived", "First award on or after 1996-05-01.", "static"),
]
BOOLS = ["has_form_d", "has_form_d_pre", "has_form_c", "has_form_c_pre", "formc_progress_update", "went_public",
         "acquired_public_signal", "public_market_exit", "regulation_a", "deregistered", "public_before_award",
         "has_sba_loan", "has_sba_loan_pre", "has_sba_loan_medium", "sba_chargeoff_ever", "formd_bizcomb_post",
         "has_federal_followon", "sbir_phase3", "has_patent_post", "patent_security_interest",
         "private_acquisition_signal", "ipo_withdrawn"]


RATE_FIELDS = ["has_form_d", "has_form_c", "public_market_exit", "has_sba_loan", "has_federal_followon",
               "sbir_phase3", "has_patent_post", "private_acquisition_signal"]


def run(paths: Paths, link_confidence: tuple[str, ...] = ("high",), geocode: bool = True) -> None:
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    anchor = firms.set_index("firm_id_sbir").first_award_date
    links = pd.read_parquet(paths.processed / "link_sbir_sec.parquet")
    fc = links[links.confidence.isin(link_confidence) & ~links.is_fund_cik][["firm_id_sbir", "cik"]].drop_duplicates()
    fc["cik_s"] = fc.cik.str.lstrip("0")
    web = sio.session()
    events: list[pd.DataFrame] = []

    fd, fd_stats = formd(paths, fc, anchor, events)
    parts = [fd, formc(paths, fc, anchor, events), edgar(paths, fc, anchor, events),
             sba(paths, anchor, events), capital_density(paths, firms), macro(paths, firms)]
    if geocode:
        parts.append(geo.run(paths, web))
    parts += linked_tables(paths, events)

    out = firms.set_index("firm_id_sbir")
    for p in parts:
        if p is not None:
            out = out.join(p.drop(columns=[c for c in p.columns if c in out.columns]), how="left")
    for c in BOOLS:
        if c in out:
            out[c] = out[c].astype("boolean").fillna(False)
    # A firm already public before its first award cannot "go public" afterwards; a later listing
    # filing (a new exchange or a relisting) is kept in firm_events but not counted as an exit.
    relist = out.went_public & out.public_before_award
    out.loc[relist, ["went_public", "went_public_date"]] = [False, pd.NaT]
    if "went_public_route" in out:
        out.loc[relist, "went_public_route"] = None
    out["public_market_exit"] = out.went_public | out.acquired_public_signal
    out["public_market_exit_date"] = out[["went_public_date", "acquired_public_date"]].min(axis=1).where(out.public_market_exit)
    out["public_market_exit_type"] = np.select(
        [out.went_public & (out.went_public_date <= out.acquired_public_date.fillna(pd.Timestamp.max)),
         out.acquired_public_signal], ["went_public", "acquired_while_public"], None)
    out["public_market_exit_type"] = out.public_market_exit_type.where(out.public_market_exit)

    # Flags for analysis samples. Established firms (already public, holding many patents, or
    # large) dominate totals of patents and federal dollars, so most analyses should report
    # medians or set them aside.
    pre_patents = out.get("patents_pre_n", pd.Series(0, index=out.index)).fillna(0)
    out["established_firm_flag"] = (out.public_before_award | (pre_patents >= 50)
                                    | (out.employees_first_award.astype("float") >= 500)).fillna(False)
    inst = r"\bUNIVERSITY\b|\bUNIV\b|\bCOLLEGE\b|\bHOSPITAL\b|MEDICAL CENTER|CANCER (?:CENTER|INSTITUTE)|\bREGENTS\b|BOARD OF TRUSTEES"
    out["institution_name_flag"] = (out.firm_name.fillna("").str.upper().str.contains(inst, regex=True)
                                    & ~out.firm_name.fillna("").str.upper().str.contains(r"\bINC\b|\bLLC\b|\bCORP", regex=True))
    out["age_implausible_flag"] = (out.age_at_first_award > 50) | (out.age_at_first_award < -5)
    out["hubzone_status"] = out.hubzone_first_award
    if "private_acquisition_signal" in out:
        out["exit_signal_any"] = out.public_market_exit | out.private_acquisition_signal
    out["oz_in_effect_at_first_award"] = out.first_award_date.dt.year >= OZ_DESIGNATION_YEAR
    out["formd_window_covered"] = out.first_award_date >= FORMD_COVERAGE_START
    out["formc_window_covered"] = out.first_award_date >= FORMC_COVERAGE_START
    out["edgar_window_covered"] = out.first_award_date >= EDGAR_COVERAGE_START
    out["link_confidence_used"] = "+".join(link_confidence)
    firm_outcomes = out.reset_index()

    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet",
                         columns=["firm_id_sbir", "award_date", "award_year", "agency", "phase_num", "amount"])
    aw["date"] = aw.award_date.fillna(pd.to_datetime(aw.award_year.astype("string") + "-07-01", errors="coerce"))
    aw = aw.assign(source="sbir", event="sbir_phase" + aw.phase_num.astype("string") + "_" + aw.agency)
    firm_events = (pd.concat([aw[["firm_id_sbir", "date", "source", "event", "amount"]]] + events, ignore_index=True)
                   .dropna(subset=["date"]).sort_values(["firm_id_sbir", "date"]).reset_index(drop=True))

    codebook = pd.DataFrame(CODEBOOK, columns=["field", "type", "source", "definition", "timing"])
    codebook["share_non_missing"] = codebook.field.map(
        lambda c: round(float(firm_outcomes[c].notna().mean()), 3) if c in firm_outcomes else np.nan)

    firm_outcomes.to_parquet(paths.processed / "firm_outcomes.parquet", index=False)
    firm_events.to_parquet(paths.processed / "firm_events.parquet", index=False)
    codebook.to_csv(paths.processed / "codebook.csv", index=False)

    f = firm_outcomes
    dera = set(f[f.has_form_d | f.has_form_d_pre].firm_id_sbir)
    fil = pd.read_parquet(paths.sec / "edgar_filings.parquet", columns=["cik", "form"])
    d_edgar = set(fil[fil.form.isin(["D", "D/A"])].assign(cik_s=lambda d: d.cik.astype("string").str.lstrip("0"))
                  .merge(fc, on="cik_s").firm_id_sbir)
    yr = f.years_to_private_cap.dropna()
    stats = {
        "firms": len(f), "events": len(firm_events), "formd": fd_stats,
        "formd_dera_vs_edgar_agreement": round(len(dera & d_edgar) / max(len(dera), 1), 4),
        "years_to_private_cap": {"n": int(len(yr)), "median": round(float(yr.median()), 2) if len(yr) else None},
        "rates_by_phase": f.groupby("sbir_phase_flag")[[c for c in RATE_FIELDS if c in f]].mean().round(4).to_dict(),
    }
    (paths.audit / "outcomes.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("outcomes: %s firms, %s events; Form D DERA/EDGAR agreement %.1f%%",
             f"{len(f):,}", f"{len(firm_events):,}", 100 * stats["formd_dera_vs_edgar_agreement"])
