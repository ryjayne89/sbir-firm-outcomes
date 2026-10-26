"""Firm founding year and registration status from SAM.gov entity extracts.

SAM.gov public entity extracts (one file per vintage, named sam_public_YYYY_MM.parquet or .csv)
go in data/raw/sam. Each SBIR firm is matched to SAM entities by UEI. Firms with no UEI match
are matched by strict name and state with city or ZIP agreement, when the name is held by one
SBIR firm and one SAM entity.

The founding year comes from the SAM entity start date, preferring a UEI match. The Form D year
of incorporation, for SEC-linked firms, is reported beside it as a check and used when SAM has
no date. Presence in each vintage is reported, since a registration that lapses is a weak
signal that a firm stopped seeking federal work.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as sio
from . import names as nm
from .config import Paths

log = logging.getLogger(__name__)

SAM_COLUMNS = {
    "uei": ["uei", "uei_sam", "ueisam", "unique_entity_id", "sam_uei"],
    "name": ["sam_name", "legal_business_name", "legal_name", "name"],
    "dba": ["dba_name", "doing_business_as"],
    "city": ["city", "physical_address_city", "phys_city"],
    "state": ["state", "physical_address_province_or_state", "phys_state"],
    "zip": ["zip", "zip5", "physical_address_zip_postal_code", "phys_zip"],
    "entity_start": ["entity_start", "entity_start_date", "business_start_date"],
    "registration_date": ["registration_date", "initial_registration_date", "reg_date"],
    "expiration_date": ["expiration_date", "registration_expiration_date", "exp_date"],
    "status": ["status", "registration_status", "sam_extract_code"],
    "structure": ["structure", "entity_structure", "entity_structure_code"],
    "state_inc": ["state_inc", "state_of_incorporation", "entity_state_of_incorporation"],
    "naics_primary": ["naics_primary", "primary_naics"],
}


def _vintage(path: Path) -> str | None:
    m = re.search(r"(\d{4})[_-](\d{2})", path.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def acquire(paths: Paths, source_dirs: list[Path] | None = None) -> list[Path]:
    """Copy SAM vintages found in source_dirs into data/raw/sam and log them in the manifest."""
    dest = paths.raw / "sam"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = sio.Manifest(paths.manifest)
    for d in source_dirs or []:
        for f in sorted(Path(d).glob("sam_public_*_*.*")):
            if f.suffix in (".parquet", ".csv") and _vintage(f) and not (dest / f.name).exists():
                shutil.copyfile(f, dest / f.name)
                log.info("copied %s", f.name)
    files = sorted(f for f in dest.glob("sam_public_*") if f.suffix in (".parquet", ".csv") and _vintage(f))
    for f in files:
        manifest.add("sam_entity_extract", f, "https://sam.gov/data-services (Entity Registration, public extract)",
                     note=f"vintage {_vintage(f)}")
    return files


def read_vintage(f: Path) -> pd.DataFrame:
    raw = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, dtype=str, low_memory=False)
    df = pd.DataFrame({k: sio.col(raw, *v).astype("string") for k, v in SAM_COLUMNS.items()})
    df["uei"] = df.uei.str.strip().str.upper()
    start = df.entity_start.str.replace(r"\D", "", regex=True)
    df["entity_start"] = pd.to_datetime(start.str[:8], format="%Y%m%d", errors="coerce")
    df.loc[df.entity_start.dt.year < 1800, "entity_start"] = pd.NaT
    df["vintage"] = _vintage(f)
    return df


def run(paths: Paths, source_dirs: list[Path] | None = None) -> None:
    files = acquire(paths, source_dirs)
    if not files:
        log.warning("No SAM extracts in %s; firm age falls back to Form D year of incorporation", paths.raw / "sam")
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    sam = pd.concat([read_vintage(f) for f in files], ignore_index=True) if files else pd.DataFrame(
        columns=list(SAM_COLUMNS) + ["vintage"])
    vintages = sorted(sam.vintage.dropna().unique())

    # UEI match
    fu = (firms[["firm_id_sbir", "uei_all"]].assign(uei=firms.uei_all.fillna("").str.split("|"))
          .explode("uei").query("uei != ''")[["firm_id_sbir", "uei"]])
    by_uei = sam.merge(fu, on="uei").assign(sam_match="uei")

    # name, state and city or ZIP for firms without a UEI match
    rest = firms[~firms.firm_id_sbir.isin(by_uei.firm_id_sbir)]
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "company_name", "city", "state", "zip5"])
    sbn = nm.add_keys(aw[aw.firm_id_sbir.isin(rest.firm_id_sbir)].rename(columns={"company_name": "name_raw"})
                      .assign(city=lambda d: nm.clean_city(d.city)))
    sbn = sbn[sbn.linkable][["firm_id_sbir", "name_key", "state", "city", "zip5"]].drop_duplicates()
    sn = sam[sam.uei.notna()].drop_duplicates(["uei", "name", "city", "state", "zip"]).copy()
    sn["state"], sn["city"], sn["zip5"] = nm.clean_state(sn.state), nm.clean_city(sn.city), nm.clean_zip(sn.zip)
    sn = nm.add_keys(sn.assign(name_raw=sn.name))
    sn = sn[sn.linkable]
    m = sbn.merge(sn[["uei", "name_key", "state", "city", "zip5"]], on=["name_key", "state"], suffixes=("", "_sam"))
    m = m[sio.same(m.city, m.city_sam) | sio.same(m.zip5, m.zip5_sam)]
    m = m[(m.groupby("name_key").uei.transform("nunique") == 1) & (m.groupby("uei").firm_id_sbir.transform("nunique") == 1)]
    by_name = sam.merge(m[["firm_id_sbir", "uei"]].drop_duplicates(), on="uei").assign(sam_match="name_location")
    hits = pd.concat([by_uei, by_name], ignore_index=True)

    g = hits.groupby("firm_id_sbir")
    latest = hits.sort_values("vintage").drop_duplicates("firm_id_sbir", keep="last").set_index("firm_id_sbir")
    out = pd.DataFrame(index=firms.firm_id_sbir)
    out["sam_match"] = g.sam_match.first()
    out["sam_uei"] = latest.uei
    out["sam_entity_start_date"] = g.entity_start.min()
    out["sam_start_dates_disagree"] = g.entity_start.nunique() > 1
    out["sam_structure"] = latest.structure
    out["sam_state_of_incorporation"] = latest.state_inc
    out["sam_naics_primary"] = latest.naics_primary
    out["sam_first_vintage"], out["sam_last_vintage"] = g.vintage.min(), g.vintage.max()
    for v in vintages:
        out[f"sam_registered_{v.replace('-', '_')}"] = hits[hits.vintage == v].groupby("firm_id_sbir").size().reindex(out.index).gt(0)
    out["sam_registered_latest"] = out.sam_last_vintage.eq(vintages[-1]) if vintages else pd.NA

    yoi = formd_year_of_inc(paths)
    if yoi is not None:
        out = out.join(yoi)
    else:
        out["formd_year_of_inc"] = np.nan
    first = firms.set_index("firm_id_sbir").first_award_date
    sam_year = out.sam_entity_start_date.dt.year
    use_sam = sam_year.notna() & ~(out.sam_match.eq("name_location") & out.formd_year_of_inc.notna())
    out["founded_year"] = np.where(use_sam, sam_year, out.formd_year_of_inc)
    out["founded_source"] = np.where(use_sam, "sam_" + out.sam_match.fillna(""),
                                     np.where(out.formd_year_of_inc.notna(), "form_d_year_of_inc", None))
    out["age_at_first_award"] = first.dt.year - out.founded_year
    # SAM start dates after the first award usually mark a re-registered or reorganized entity.
    out["founded_after_first_award"] = out.age_at_first_award < 0
    for c in [c for c in out.columns if c.startswith("sam_registered_")] + ["sam_start_dates_disagree"]:
        out[c] = out[c].astype("boolean").fillna(False)

    out.reset_index().to_parquet(paths.processed / "firm_sam.parquet", index=False)
    both = out.dropna(subset=["formd_year_of_inc"])
    both = both[both.sam_entity_start_date.notna()]
    diff = (both.sam_entity_start_date.dt.year - both.formd_year_of_inc).abs()
    stats = {
        "vintages": vintages, "sam_rows": int(len(sam)),
        "firms_matched": out.sam_match.value_counts().to_dict(),
        "founded_source": out.founded_source.value_counts().to_dict(),
        "founded_after_first_award": int(out.founded_after_first_award.sum()),
        "age_at_first_award_median": float(out.age_at_first_award[out.age_at_first_award >= 0].median())
        if out.age_at_first_award.notna().any() else None,
        "sam_vs_formd_year": {"n": int(len(diff)), "exact_share": round(float((diff == 0).mean()), 4) if len(diff) else None,
                              "within_1y_share": round(float((diff <= 1).mean()), 4) if len(diff) else None},
    }
    (paths.audit / "sam.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("SAM: %s firms matched, founded year for %s", f"{out.sam_match.notna().sum():,}", f"{out.founded_year.notna().sum():,}")


def formd_year_of_inc(paths: Paths) -> pd.DataFrame | None:
    """Earliest Form D year of incorporation reported by a firm's high-confidence CIKs."""
    f = paths.sec / "formd_issuers.parquet"
    lk = paths.processed / "link_sbir_sec.parquet"
    if not (f.exists() and lk.exists()):
        return None
    iss = pd.read_parquet(f)
    year = sio.to_num(sio.col(iss, "YEAROFINC_VALUE_ENTERED"))
    iss = pd.DataFrame({"cik": iss.CIK.astype("string").str.lstrip("0"), "year": year})
    iss = iss[iss.year.between(1800, 2100)]
    links = pd.read_parquet(lk)
    links = links[(links.confidence == "high") & ~links.is_fund_cik].assign(cik=lambda d: d.cik.str.lstrip("0"))
    return links.merge(iss, on="cik").groupby("firm_id_sbir").year.min().rename("formd_year_of_inc").to_frame()
