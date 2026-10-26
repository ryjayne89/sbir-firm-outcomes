"""SBIR/STTR backbone: clean awards and assemble them into firms.

Awards join into one firm when they share a UEI, a DUNS, a website domain used by few
distinct names, the same strict name in the same state, or (for names too short to be
distinctive) the same strict name at the same ZIP code. Connected awards form a firm.
"""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd

from . import io as sio
from . import names as nm
from .config import HUBZONE_START_YEAR, SBIR_PROGRAM_START_YEAR, Paths

log = logging.getLogger(__name__)

FIELDS = {
    "company": ["Company"], "uei": ["UEI"], "duns": ["Duns", "DUNS"], "website": ["Company Website"],
    "address1": ["Address1"], "address2": ["Address2"], "city": ["City"], "state_raw": ["State"],
    "zip_raw": ["Zip"], "agency": ["Agency"], "branch": ["Branch"], "phase": ["Phase"],
    "program": ["Program"], "title": ["Award Title"], "tracking_number": ["Agency Tracking Number"],
    "contract": ["Contract"], "award_date_raw": ["Proposal Award Date"], "end_date_raw": ["Contract End Date"],
    "solicitation_number": ["Solicitation Number"], "solicitation_year": ["Solicitation Year"],
    "topic_code": ["Topic Code"], "award_year_raw": ["Award Year"], "amount_raw": ["Award Amount"],
    "hubzone_raw": ["HUBZone Owned"], "sed_raw": ["Socially and Economically Disadvantaged"],
    "woman_raw": ["Woman Owned"], "employees_raw": ["Number Employees"], "ri_name": ["RI Name"],
}
GENERIC_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "aol.com", "msn.com", "outlook.com",
                   "comcast.net", "verizon.net", "att.net", "earthlink.net", "icloud.com", "me.com",
                   "sites.google.com", "google.com", "linkedin.com", "facebook.com", "wix.com",
                   "wordpress.com", "godaddy.com", "squarespace.com", "none.com", "na.com", "sbir.gov"}
YES_NO = {"Y": True, "YES": True, "TRUE": True, "T": True, "1": True,
          "N": False, "NO": False, "FALSE": False, "F": False, "0": False}


def _yn(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().map(YES_NO).astype("boolean")


def load(paths: Paths) -> tuple[pd.DataFrame, str]:
    """Pick the most complete SBIR file on disk: one with a UEI column, then the most rows."""
    import pyarrow.parquet as pq

    d = paths.raw / "sbir"
    candidates = []
    for f in sorted(d.glob("sbir_awards.parquet")) + sorted(d.glob("award_data*.csv")):
        if f.suffix == ".parquet":
            meta = pq.ParquetFile(f)
            cols, n = meta.schema_arrow.names, meta.metadata.num_rows
        else:
            cols = pd.read_csv(f, nrows=0).columns
            with open(f, "rb") as fh:
                n = sum(1 for _ in fh) - 1
        candidates.append((sio.pick(cols, "UEI") is not None, n, f))
    if not candidates:
        raise FileNotFoundError(f"no SBIR award file in {d}")
    has_uei, n, f = max(candidates, key=lambda c: (c[0], c[1]))
    df = (pd.read_parquet(f).astype("string") if f.suffix == ".parquet"
          else pd.read_csv(f, dtype=str, low_memory=False, encoding_errors="replace"))
    log.info("SBIR source %s: %s awards (UEI column: %s)", f.name, f"{len(df):,}", has_uei)
    return df, f.name


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {k: sio.pick(raw.columns, *v) for k, v in FIELDS.items()}
    a = pd.DataFrame({k: raw[v] if v else pd.Series(pd.NA, index=raw.index, dtype="string")
                      for k, v in cols.items()})
    a["state"] = nm.clean_state(a.state_raw)
    a["zip5"] = nm.clean_zip(a.zip_raw)
    a["award_date"] = sio.to_date(a.award_date_raw)
    a["end_date"] = sio.to_date(a.end_date_raw)
    a["award_year"] = pd.to_numeric(a.award_year_raw, errors="coerce").astype("Int64")
    a.loc[a.award_date.dt.year < SBIR_PROGRAM_START_YEAR, "award_date"] = pd.NaT
    a.loc[a.award_year < SBIR_PROGRAM_START_YEAR, "award_year"] = pd.NA
    a["amount"] = sio.to_num(a.amount_raw)
    a["employees"] = pd.to_numeric(a.employees_raw, errors="coerce")
    a["hubzone"], a["sed"], a["woman_owned"] = _yn(a.hubzone_raw), _yn(a.sed_raw), _yn(a.woman_raw)
    # HUBZone certification began in 1999, so earlier awards carry no meaningful flag.
    a.loc[a.award_year.astype("float") < HUBZONE_START_YEAR, "hubzone"] = pd.NA
    a["phase_num"] = a.phase.astype("string").str.extract(r"Phase\s+(I+)\b", expand=False).str.len()
    a["is_sttr"] = a.program.astype("string").str.upper().eq("STTR")
    a["fasttrack"] = a.title.astype("string").str.contains(r"fast.?track", case=False, na=False)

    a["uei"] = a.uei.astype("string").str.strip().str.upper()
    a.loc[~a.uei.fillna("").str.fullmatch(r"[A-Z0-9]{12}"), "uei"] = pd.NA
    d = a.duns.astype("string").str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    a["duns"] = d.str.zfill(9).where(d.str.len().between(7, 9) & ~d.fillna("").str.fullmatch(r"0*"))
    dom = (a.website.astype("string").str.lower().str.strip()
           .str.replace(r"^[a-z]+://", "", regex=True).str.replace(r"^www\d?\.", "", regex=True)
           .str.split("/").str[0].str.split("?").str[0])
    a["domain"] = dom.where(dom.str.contains(r"^[a-z0-9.-]+\.[a-z]{2,}$", na=False))

    n0 = len(a)
    a = a.drop_duplicates().reset_index(drop=True)
    ident = ["agency", "tracking_number", "contract", "phase", "company", "amount_raw"]
    a["possible_duplicate"] = a.duplicated(subset=ident, keep=False) & a[ident].notna().all(axis=1)
    key = a[["agency", "tracking_number", "contract", "phase", "award_date_raw", "company", "amount_raw"]]
    a["award_id"] = "A" + key.astype("string").fillna("").agg("|".join, axis=1).map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:12])
    dup = a.award_id.duplicated(keep=False)
    a.loc[dup, "award_id"] += "-" + a[dup].groupby("award_id").cumcount().astype(str)
    log.info("awards: %s (exact duplicates dropped %d, flagged possible duplicates %d)",
             f"{len(a):,}", n0 - len(a), int(a.possible_duplicate.sum()))
    return a


def resolve_firms(a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = nm.add_keys(a.rename(columns={"company": "name_raw"}))
    a["node_id"] = a.award_id
    per_domain = a.dropna(subset=["domain"]).groupby("domain").loose_key.nunique()
    domains = set(per_domain[per_domain <= 3].index) - GENERIC_DOMAINS
    edges = pd.concat([
        nm.star_edges(a, ["uei"], "uei"),
        nm.star_edges(a, ["duns"], "duns"),
        nm.star_edges(a[a.linkable], ["name_key", "state"], "name_state"),
        nm.star_edges(a[a.domain.isin(domains)], ["domain"], "website"),
        nm.star_edges(a[~a.linkable & a.name_key.notna()], ["name_key", "state", "zip5"], "name_zip_short"),
    ], ignore_index=True)
    a["_comp"] = nm.components(a.node_id, edges)

    # Stable ID: hash of the smallest UEI, else DUNS, else name|state. Firms that share a
    # name|state key (names too generic to link) get a suffix ordered by earliest award_id.
    a["_idkey"] = np.where(a.uei.notna(), "1U" + a.uei.fillna(""),
                  np.where(a.duns.notna(), "2D" + a.duns.fillna(""),
                           "3N" + a.name_key.fillna(a.name_raw.fillna("")) + "|" + a.state.fillna("")))
    comp = (a.groupby("_comp").agg(key=("_idkey", "min"), first_award=("award_id", "min"))
              .sort_values(["key", "first_award"]))
    comp["seq"] = comp.groupby("key").cumcount()
    comp["firm_id_sbir"] = ("S" + comp.key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:10])
                            + np.where(comp.seq > 0, "-" + comp.seq.astype(str), ""))
    a["firm_id_sbir"] = a._comp.map(comp.firm_id_sbir)
    assert comp.firm_id_sbir.is_unique
    log.info("firms: %s from %s awards; edges %s", f"{a.firm_id_sbir.nunique():,}", f"{len(a):,}",
             edges.rule.value_counts().to_dict())
    return a, edges


def firm_table(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["date_imputed"] = a.award_date.isna()
    a["_t"] = a.award_date.fillna(pd.to_datetime(a.award_year.astype("string") + "-07-01", errors="coerce"))
    a = a.sort_values(["firm_id_sbir", "_t"])
    g = a.groupby("firm_id_sbir")
    first = a.drop_duplicates("firm_id_sbir", keep="first").set_index("firm_id_sbir")
    last = a.drop_duplicates("firm_id_sbir", keep="last").set_index("firm_id_sbir")
    p1 = a[a.phase_num == 1].groupby("firm_id_sbir")._t.min()
    p2 = a[a.phase_num == 2].groupby("firm_id_sbir")._t.min()
    primary_agency = (a.groupby(["firm_id_sbir", "agency"]).amount.sum().reset_index()
                      .sort_values("amount", ascending=False).drop_duplicates("firm_id_sbir")
                      .set_index("firm_id_sbir").agency)
    joined = lambda c: g[c].agg(lambda s: "|".join(sorted(set(s.dropna().astype(str)))))

    f = pd.DataFrame({
        "firm_name": last.name_raw, "names_all": joined("name_raw"), "n_names": g.name_key.nunique(),
        "uei_all": joined("uei"), "duns_all": joined("duns"), "domain": nm.modal(a, "firm_id_sbir", "domain"),
        "address1": last.address1, "city": last.city, "state": last.state, "zip5": last.zip5,
        "n_states": g.state.nunique(),
        "first_award_date": g._t.min(), "first_award_date_imputed": first.date_imputed,
        "last_award_date": g._t.max(), "first_award_agency": first.agency, "first_award_phase": first.phase_num,
        "first_phase1_date": p1, "first_phase2_date": p2,
        "n_awards": g.size(), "n_phase1": (a.phase_num == 1).groupby(a.firm_id_sbir).sum(),
        "n_phase2": (a.phase_num == 2).groupby(a.firm_id_sbir).sum(),
        "total_grant_funding": g.amount.sum(min_count=1),
        "agencies": joined("agency"), "n_agencies": g.agency.nunique(), "primary_agency": primary_agency,
        "ever_sttr": g.is_sttr.any(), "ever_fasttrack": g.fasttrack.any(),
        "hubzone_ever": g.hubzone.max(), "hubzone_first_award": first.hubzone,
        "sed_ever": g.sed.max(), "woman_owned_ever": g.woman_owned.max(),
        "employees_first_award": first.employees, "employees_max": g.employees.max(),
        "name_flag_nonfirm": g.name_key.agg(lambda s: s.fillna("").str.contains(nm.NONFIRM).any()),
    })
    has1, has2 = f.first_phase1_date.notna(), f.first_phase2_date.notna()
    f["sbir_phase_flag"] = np.select([has1 & ~has2, ~has1 & has2, has1 & has2], [1, 2, 3], 0)
    f.index.name = "firm_id_sbir"
    return f.reset_index()


def run(paths: Paths) -> None:
    raw, source = load(paths)
    awards, edges = resolve_firms(clean(raw))
    firms = firm_table(awards)
    awards = awards.rename(columns={"name_raw": "company_name"})

    keep = ["award_id", "firm_id_sbir", "agency", "branch", "program", "phase", "phase_num", "is_sttr",
            "fasttrack", "award_date", "award_year", "end_date", "amount", "solicitation_number",
            "solicitation_year", "topic_code", "tracking_number", "contract", "title", "company_name",
            "uei", "duns", "domain", "address1", "address2", "city", "state", "zip5", "hubzone", "sed",
            "woman_owned", "employees", "ri_name", "possible_duplicate"]
    awards[keep].to_parquet(paths.processed / "sbir_awards.parquet", index=False)
    firms.to_parquet(paths.processed / "sbir_firms.parquet", index=False)

    t = awards.award_date.fillna(pd.to_datetime(awards.award_year.astype("string") + "-07-01", errors="coerce"))
    (awards.assign(year=t.dt.year).groupby(["firm_id_sbir", "year"])
     .agg(n_awards=("award_id", "size"), amount=("amount", "sum"),
          n_phase1=("phase_num", lambda s: (s == 1).sum()), n_phase2=("phase_num", lambda s: (s == 2).sum()))
     .reset_index().to_parquet(paths.processed / "sbir_firm_year.parquet", index=False))
    (awards.assign(_t=t).dropna(subset=["zip5"])
     .groupby(["firm_id_sbir", "address1", "city", "state", "zip5"], dropna=False)
     .agg(first_used=("_t", "min"), last_used=("_t", "max"), n_awards=("award_id", "size"))
     .reset_index().to_parquet(paths.processed / "sbir_firm_addresses.parquet", index=False))

    (paths.audit / "sbir.json").write_text(json.dumps({
        "source_file": source, "awards": len(awards), "firms": len(firms),
        "edges": edges.rule.value_counts().to_dict(),
        "phase_flag": {int(k): int(v) for k, v in firms.sbir_phase_flag.value_counts().items()},
        "awards_by_agency": awards.agency.value_counts().to_dict(),
    }, indent=2, default=str))
