"""Download the public sources and flatten them into parquet tables.

Raw files land in data/raw and are logged in data/raw/MANIFEST.csv. Parsed tables land in
data/interim. Every step skips work whose output already exists, so reruns are cheap.
"""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io as sio
from .config import Paths

log = logging.getLogger(__name__)

FORMD_PAGES = ["https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets",
               "https://www.sec.gov/dera/data/form-d"]
FORMC_PAGES = ["https://www.sec.gov/data-research/sec-markets-data/crowdfunding-offerings-data-sets",
               "https://www.sec.gov/dera/data/crowdfunding-offerings-data-sets"]
SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
SBIR_PAGE = "https://www.sbir.gov/data-resources"
SBA_PAGE = "https://data.sba.gov/dataset/7-a-504-foia"
ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip"
QOZ_URL = "https://www.cdfifund.gov/system/files/documents/designated-qozs.12.14.18.xlsx"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# EDGAR forms kept from each CIK's filing history.
EDGAR_FORMS = {
    # registration and going public
    "S-1", "S-1/A", "S-1MEF", "F-1", "F-1/A", "F-1MEF", "SB-2", "SB-2/A", "RW", "EFFECT",
    "424B1", "424B3", "424B4", "8-A12B", "8-A12G", "10-12B", "10-12B/A", "10-12G", "10-12G/A",
    "8-K12B", "8-K12G3",
    # periodic and current reports (years as a public company)
    "10-K", "10-K405", "10-KSB", "10-Q", "10-QSB", "20-F", "40-F", "8-K",
    # ownership stakes
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A",
    # acquisitions and mergers
    "SC TO-T", "SC TO-T/A", "SC 14D9", "SC 14D9/A", "DEFM14A", "DEFM14C", "S-4", "S-4/A", "F-4", "F-4/A", "425",
    # leaving the public market
    "15-12B", "15-12G", "15-15D", "25-NSE", "25",
    # Regulation A, Form D, Form C
    "1-A", "1-A/A", "1-K", "253G2", "D", "D/A", "C", "C/A", "C-U", "C-AR", "C-TR",
}
EDGAR_FORMS_VERSION = "2"     # bump when EDGAR_FORMS changes, so submissions.zip is parsed again

SBA_COLUMNS = {
    "borr_name": ["BorrName", "BorrowerName"], "borr_street": ["BorrStreet", "BorrowerStreet"],
    "borr_city": ["BorrCity", "BorrowerCity"], "borr_state": ["BorrState", "BorrowerState"],
    "borr_zip": ["BorrZip", "BorrowerZip"], "approval_date": ["ApprovalDate"],
    "fy": ["ApprovalFiscalYear", "ApprovalFY"], "gross_approval": ["GrossApproval"],
    "sba_guaranteed": ["SBAGuaranteedApproval"], "naics": ["NaicsCode", "NAICSCode"],
    "jobs": ["JobsSupported"], "business_type": ["BusinessType"], "business_age": ["BusinessAge"],
    "loan_status": ["LoanStatus"], "first_disbursement_date": ["FirstDisbursementDate"],
    "paid_in_full_date": ["PaidInFullDate"], "chargeoff_date": ["ChargeOffDate"],
}


def sbir(paths: Paths, manifest: sio.Manifest, web) -> None:
    d = paths.raw / "sbir"
    d.mkdir(parents=True, exist_ok=True)
    have = list(d.glob("sbir_awards.parquet")) + list(d.glob("award_data*.csv"))
    if have:
        for f in have:
            manifest.add("sbir_awards", f, SBIR_PAGE, note="present on disk")
        return
    for url in sio.list_links(web, SBIR_PAGE, r"award_data[^\"]*\.csv")[:1]:
        sio.download(web, url, d / url.split("/")[-1], manifest, "sbir_awards")
    if not any(d.iterdir()):
        raise FileNotFoundError(f"No SBIR award file. Download award_data.csv from {SBIR_PAGE} into {d}.")


def _quarterly(sec, pages: list[str], pattern: str, dest: Path, dataset: str, manifest) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    for page in pages:
        links = sio.list_links(sec, page, pattern)
        if links:
            break
    by_quarter = {sio.quarter_of(u.split("/")[-1]): u for u in links}
    by_quarter.pop(None, None)
    for q, url in sorted(by_quarter.items()):
        try:
            sio.download(sec, url, dest / f"{q}.zip", manifest, dataset)
        except FileNotFoundError:
            log.warning("%s %s not found", dataset, q)
    files = sorted(dest.glob("*.zip"))
    for f in files:
        manifest.add(dataset, f, note="present on disk")
    log.info("%s: %d quarters listed online, %d on disk", dataset, len(by_quarter), len(files))
    return files


def _stack(files: list[Path], tables: list[str], prefix: str, out_dir: Path) -> None:
    for t in tables:
        out = out_dir / f"{prefix}_{t.lower()}.parquet"
        frames = []
        for f in files:
            d = sio.zip_table(f, t)
            if d is not None:
                d.columns = [c.upper() for c in d.columns]
                frames.append(d.assign(QUARTER=f.stem))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_parquet(out, index=False)
            log.info("%s: %s rows", out.name, f"{len(df):,}")


def formd(paths: Paths, manifest, sec, refresh: bool = False) -> None:
    files = _quarterly(sec, FORMD_PAGES, r"form-d-data-sets/[^\"]+\.zip$", paths.raw / "sec_formd", "sec_formd", manifest)
    if refresh or not (paths.sec / "formd_offering.parquet").exists():
        _stack(files, ["FORMDSUBMISSION", "ISSUERS", "OFFERING"], "formd", paths.sec)


def formc(paths: Paths, manifest, sec, refresh: bool = False) -> None:
    files = _quarterly(sec, FORMC_PAGES, r"crowdfunding-offerings-data-sets/[^\"]+\.zip$",
                       paths.raw / "sec_formc", "sec_formc", manifest)
    if refresh or not (paths.sec / "formc_form_c_disclosure.parquet").exists():
        _stack(files, ["FORM_C_SUBMISSION", "FORM_C_ISSUER_INFORMATION", "FORM_C_DISCLOSURE"], "formc", paths.sec)


def _filings(cik: str, cols: dict | None) -> pd.DataFrame | None:
    if not cols or "form" not in cols:
        return None
    n = len(cols["form"])
    f = pd.DataFrame({k: (cols.get(k) or [None] * n) for k in ("accessionNumber", "filingDate", "reportDate", "form")})
    f = f[f.form.isin(EDGAR_FORMS)]
    return f.assign(cik=cik) if len(f) else None


def edgar(paths: Paths, manifest, sec, refresh: bool = False) -> None:
    """Parse submissions.zip into entities, former names and the kept filing history."""
    zpath = sio.download(sec, SUBMISSIONS_URL, paths.raw / "sec_edgar" / "submissions.zip",
                         manifest, "sec_edgar_submissions", timeout=1800)
    marker = paths.sec / "edgar_forms_version.txt"
    current = marker.exists() and marker.read_text().strip() == EDGAR_FORMS_VERSION
    if not refresh and current and (paths.sec / "edgar_entities.parquet").exists():
        return
    entities, former, filings = [], [], []
    writer = None

    def flush():
        nonlocal writer, filings
        if filings:
            table = pa.Table.from_pandas(pd.concat(filings, ignore_index=True).astype("string"), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(paths.sec / "edgar_filings.parquet", table.schema)
            writer.write_table(table.cast(writer.schema))
            filings = []

    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        for i, name in enumerate(names, 1):
            m = re.match(r"CIK(\d{10})(-submissions-\d+)?\.json$", Path(name).name)
            if not m:
                continue
            cik, j = m.group(1), json.loads(zf.read(name))
            if m.group(2):
                f = _filings(cik, j)
            else:
                addr = j.get("addresses") or {}
                biz, mail = addr.get("business") or {}, addr.get("mailing") or {}
                entities.append({
                    "cik": cik, "name": j.get("name"), "entity_type": j.get("entityType"),
                    "sic": j.get("sic"), "sic_desc": j.get("sicDescription"),
                    "state_inc": j.get("stateOfIncorporation"),
                    "tickers": "|".join(j.get("tickers") or []),
                    "exchanges": "|".join(x for x in (j.get("exchanges") or []) if x),
                    "biz_street1": biz.get("street1"), "biz_city": biz.get("city"),
                    "biz_state": biz.get("stateOrCountry"), "biz_zip": biz.get("zipCode"),
                    "mail_city": mail.get("city"), "mail_state": mail.get("stateOrCountry"),
                    "mail_zip": mail.get("zipCode"),
                })
                former += [{"cik": cik, "name": x.get("name"), "from": x.get("from"), "to": x.get("to")}
                           for x in j.get("formerNames") or []]
                f = _filings(cik, (j.get("filings") or {}).get("recent"))
            if f is not None:
                filings.append(f)
            if i % 100_000 == 0:
                flush()
                log.info("submissions: %s of %s files", f"{i:,}", f"{len(names):,}")
    flush()
    if writer is not None:
        writer.close()
    pd.DataFrame(entities).to_parquet(paths.sec / "edgar_entities.parquet", index=False)
    pd.DataFrame(former).to_parquet(paths.sec / "edgar_former_names.parquet", index=False)
    marker.write_text(EDGAR_FORMS_VERSION)
    log.info("EDGAR: %s entities, %s former names", f"{len(entities):,}", f"{len(former):,}")


def sba(paths: Paths, manifest, web, refresh: bool = False) -> bool:
    """SBA 7(a) and 504 FOIA files. data.sba.gov refuses many scripted requests, so files placed
    in data/raw/sba_foia by hand are used as-is. Returns False when no files are available."""
    d = paths.raw / "sba_foia"
    d.mkdir(parents=True, exist_ok=True)
    out = paths.interim / "sba" / "sba_loans.parquet"
    if not list(d.glob("*.csv")):
        urls = sio.list_links(web, SBA_PAGE, r"\.csv$")
        for url in urls:
            sio.download(web, url, d / re.sub(r"[^A-Za-z0-9._()-]+", "_", url.split("/")[-1]), manifest, "sba_foia")
    files = sorted(d.glob("*.csv"))
    if not files:
        log.warning("No SBA FOIA files. Download the 7(a) and 504 CSVs from %s into %s and rerun.", SBA_PAGE, d)
        return False
    for f in sorted(d.glob("*")):
        if f.suffix.lower() in (".csv", ".xlsx"):
            manifest.add("sba_foia", f, "https://data.sba.gov/sites/default/files/uploaded_resources/" + f.name,
                         note="present on disk")
    if out.exists() and not refresh:
        return True
    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0, encoding_errors="replace").columns
        keep = {sio.pick(header, *v): k for k, v in SBA_COLUMNS.items() if sio.pick(header, *v)}
        frames.append(pd.read_csv(f, usecols=list(keep), dtype=str, low_memory=False,
                                  encoding_errors="replace", on_bad_lines="skip")
                      .rename(columns=keep).assign(program="504" if "504" in f.name else "7A", source_file=f.name))
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("SBA loans: %s rows", f"{sum(len(x) for x in frames):,}")
    return True


def geography(paths: Paths, manifest, web) -> None:
    geo = paths.raw / "geo"
    zc_out, qoz_out = paths.interim / "geo" / "zcta_centroids.parquet", paths.interim / "geo" / "qoz_tracts_2010.parquet"
    if not zc_out.exists():
        z = sio.download(web, ZCTA_URL, geo / "2020_Gaz_zcta_national.zip", manifest, "census_zcta_gazetteer")
        with zipfile.ZipFile(z) as zf:
            g = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), sep="\t", dtype={"GEOID": str})
        g.columns = [c.strip() for c in g.columns]
        g[["GEOID", "INTPTLAT", "INTPTLONG"]].set_axis(["zip5", "lat", "lon"], axis=1).to_parquet(zc_out, index=False)
    if not qoz_out.exists():
        x = sio.download(web, QOZ_URL, geo / "designated-qozs.12.14.18.xlsx", manifest, "cdfi_qoz")
        raw = pd.read_excel(x, header=None, dtype=str)
        header = raw.index[raw.apply(lambda r: r.astype(str).str.contains("Census Tract", case=False).any(), axis=1)][0]
        q = pd.read_excel(x, header=header, dtype=str)
        tract_col = [c for c in q.columns if "tract" in str(c).lower() and "type" not in str(c).lower()][0]
        tracts = q[tract_col].str.replace(r"\D", "", regex=True).str.zfill(11)
        pd.DataFrame({"tract_geoid_2010": tracts[tracts.str.len() == 11].drop_duplicates()}).to_parquet(qoz_out, index=False)


def fred(paths: Paths, manifest, web, series: str) -> Path | None:
    """FRED series as CSV. FRED often times out from cloud hosts; a manually saved
    data/raw/macro/<SERIES>.csv is used when present."""
    dest = paths.raw / "macro" / f"{series}.csv"
    try:
        return sio.download(web, FRED_URL.format(series=series), dest, manifest, f"fred_{series.lower()}",
                            retries=2, timeout=60)
    except Exception:
        log.warning("Could not download %s. Save it from https://fred.stlouisfed.org/series/%s as %s.",
                    series, series, dest)
        return None


def run(paths: Paths, sec_user_agent: str, refresh: bool = False) -> None:
    manifest = sio.Manifest(paths.manifest)
    sec, web = sio.sec_session(sec_user_agent), sio.session()
    sbir(paths, manifest, web)
    formd(paths, manifest, sec, refresh)
    formc(paths, manifest, sec, refresh)
    edgar(paths, manifest, sec, refresh)
    sba(paths, manifest, web, refresh)
    geography(paths, manifest, web)
    ff = fred(paths, manifest, web, "FEDFUNDS")
    if ff is not None:
        d = pd.read_csv(ff)
        d.columns = ["date", "fedfunds"]
        d["date"] = pd.to_datetime(d.date)
        d.to_parquet(paths.interim / "macro" / "fedfunds_monthly.parquet", index=False)
    fred(paths, manifest, web, "CPIAUCSL")
