"""Characteristics of the places where SBIR firms are located.

Two public sources, joined to firms by ZIP code (ZCTA).

* American Community Survey 5-year estimates, by ZCTA and year (Census Bureau API): population,
  educational attainment, household income, unemployment and home values. Each firm gets the
  estimates for its own ZCTA and for all ZCTAs within RADIUS_MILES, from the ACS release
  closest to its first award year.
* The IPEDS institutional directory (NCES): colleges and universities with their locations and
  Carnegie classification, counted within RADIUS_MILES of each firm.

A Census API key is optional. Without one the API allows a few hundred requests a day, which is
enough for this step; set the CENSUS_API_KEY environment variable or pass --census-key to use one.
"""
from __future__ import annotations

import io
import json
import re
import logging
import os
import time
import zipfile

import numpy as np
import pandas as pd

from . import io as sio
from .config import RADIUS_MILES, Paths

log = logging.getLogger(__name__)

ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
ACS_YEARS = range(2012, 2025)          # 5-year releases with ZCTA detail tables B15003
ACS_VARS = {
    "B01003_001E": "pop",
    "B01002_001E": "median_age",
    "B15003_001E": "pop25",
    "B15003_022E": "ba", "B15003_023E": "ma", "B15003_024E": "prof", "B15003_025E": "phd",
    "B19013_001E": "median_hh_income",
    "B23025_003E": "civ_labor_force", "B23025_005E": "unemployed",
    "B25077_001E": "median_home_value",
}
CENSUS2000_URL = "https://api.census.gov/data/2000/dec/sf3"
CENSUS2000_VARS = {
    "P001001": "pop", "P037001": "pop25",
    "P037015": "m_ba", "P037016": "m_ma", "P037017": "m_prof", "P037018": "m_phd",
    "P037032": "f_ba", "P037033": "f_ma", "P037034": "f_prof", "P037035": "f_phd",
    "P053001": "median_hh_income", "H085001": "median_home_value",
}
PANEL_MEASURES = ["pop", "ba_plus_share", "grad_share", "median_hh_income", "median_home_value"]
IPEDS_URL = "https://nces.ed.gov/ipeds/datacenter/data/HD{year}.zip"
IPEDS_YEARS = (2023, 2022, 2021)
STATE_FIPS = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19", "20",
              "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37",
              "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56", "72"]


class BadKey(RuntimeError):
    """The Census API rejected the key."""


def _get(web, url: str, params: dict, tries: int = 3, timeout: int = 120):
    """One Census API request. Returns the rows, or None when the request has no data. The API
    answers a bad key or an unsupported query with a page that is not JSON, so those are not retried."""
    for k in range(tries):
        try:
            r = web.get(url, params=params, timeout=timeout)
        except Exception:
            time.sleep(3 * (k + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                if "key" in r.text.lower() and "invalid" in r.text.lower():
                    raise BadKey(r.text[:200])
                return None
        if r.status_code in (204, 400, 404):
            return None
        time.sleep(3 * (k + 1))
    return None


def _check_api(web, key: str | None) -> str | None:
    """A one-row test request. Returns the key to use (None to go without), and raises when the
    API cannot be reached at all, so the step does not spend hours on doomed requests."""
    test = {"get": "B01003_001E", "for": "zip code tabulation area:78701"}
    url = ACS_URL.format(year=2022)
    if key and not isinstance(key, str):
        log.warning("The Census key is a %s, not text; continuing without one", type(key).__name__)
        key = None
    if key:
        try:
            if _get(web, url, {**test, "key": key.strip()}, tries=2, timeout=30):
                return key.strip()
        except BadKey:
            log.warning("The Census API rejected the key; continuing without one (fewer requests allowed per day)")
    if _get(web, url, test, tries=2, timeout=30):
        return None
    raise RuntimeError("The Census API could not be reached.")


def acs_year(web, year: int, key: str | None) -> pd.DataFrame | None:
    """One ACS 5-year release for every ZCTA. Releases before 2020 nest ZCTAs within states."""
    params = {"get": ",".join(ACS_VARS), "for": "zip code tabulation area:*"}
    if key:
        params["key"] = key
    rows = _get(web, ACS_URL.format(year=year), params)
    if rows is None and year >= 2020:
        return None                               # not released, or the API is unreachable
    if rows is None:
        rows = []
        for st in STATE_FIPS:
            part = _get(web, ACS_URL.format(year=year), {**params, "in": f"state:{st}"})
            if part:
                rows += part[1:] if rows else part
        if not rows:
            return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    zc = [c for c in df.columns if c.startswith("zip code tabulation area")][0]
    out = df[list(ACS_VARS)].apply(pd.to_numeric, errors="coerce").rename(columns=ACS_VARS)
    out[out < -100000] = np.nan                  # the API codes suppressed values as large negatives
    out["zip5"] = df[zc].str.zfill(5)
    out["acs_year"] = year
    return out


def acs(paths: Paths, web, key: str | None, fetch: bool = True) -> pd.DataFrame:
    """Every ACS release, read from the cache on disk and fetched from the API only when missing
    (and only when fetch is true), so a down API never discards releases already saved."""
    cache = paths.interim / "places"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = sio.Manifest(paths.manifest)
    frames = []
    for y in ACS_YEARS:
        f = cache / f"acs5_zcta_{y}.parquet"
        if not f.exists() and not fetch:
            log.warning("ACS %s not on disk and the API is unavailable; skipped", y)
            continue
        if not f.exists():
            d = acs_year(web, y, key)
            if d is None:
                log.info("ACS %s 5-year ZCTA estimates not available; skipped", y)
                continue
            d.to_parquet(f, index=False)
            manifest.add("census_acs5_zcta", f, ACS_URL.format(year=y), note=f"ACS 5-year {y - 4}-{y}, ZCTA, "
                         + ", ".join(ACS_VARS))
            log.info("ACS %s: %s ZCTAs", y, f"{len(d):,}")
        frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    z["ba_plus"] = z[["ba", "ma", "prof", "phd"]].sum(axis=1, min_count=1)
    z["grad"] = z[["ma", "prof", "phd"]].sum(axis=1, min_count=1)
    z["ba_plus_share"] = z.ba_plus / z.pop25.where(z.pop25 > 0)
    z["grad_share"] = z.grad / z.pop25.where(z.pop25 > 0)
    z["unemployment_rate"] = z.unemployed / z.civ_labor_force.where(z.civ_labor_force > 0)
    return z


def census2000(paths: Paths, web, key: str | None) -> pd.DataFrame:
    """2000 Census (Summary File 3) by ZCTA: the baseline for firms that entered before the ACS.
    Income is for 1999, as the census asked."""
    f = paths.interim / "places" / "census2000_zcta.parquet"
    if not f.exists():
        rows = []
        for geo in ("zip code tabulation area (or part)", "zip code tabulation area"):
            params = {"get": ",".join(CENSUS2000_VARS), "for": f"{geo}:*"}
            if key:
                params["key"] = key
            if not _get(web, CENSUS2000_URL, {**params, "in": "state:11"}):     # quick probe on DC
                continue
            for st in STATE_FIPS:
                part = _get(web, CENSUS2000_URL, {**params, "in": f"state:{st}"})
                if part:
                    rows += part[1:] if rows else part
            break
        if not rows:
            log.warning("2000 Census ZCTA estimates not available; the panel starts place measures in 2012")
            return pd.DataFrame()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        zc = [c for c in df.columns if c.startswith("zip code tabulation area")][0]
        d = df[list(CENSUS2000_VARS)].apply(pd.to_numeric, errors="coerce").rename(columns=CENSUS2000_VARS)
        d["zip5"] = df[zc].str[:5].str.zfill(5)
        if d.zip5.duplicated().any():
            # a ZCTA crossing a state line comes in parts: add the counts, and take medians from
            # the part with the most people
            counts = [c for c in d.columns if c not in ("zip5", "median_hh_income", "median_home_value")]
            big = d.sort_values("pop", ascending=False).drop_duplicates("zip5").set_index("zip5")
            d = d.groupby("zip5")[counts].sum(min_count=1).join(big[["median_hh_income", "median_home_value"]]).reset_index()
        d.to_parquet(f, index=False)
        sio.Manifest(paths.manifest).add("census_2000_sf3_zcta", f, CENSUS2000_URL,
                                         note="2000 Census SF3, ZCTA, " + ", ".join(CENSUS2000_VARS))
    d = pd.read_parquet(f)
    ba = d[["m_ba", "m_ma", "m_prof", "m_phd", "f_ba", "f_ma", "f_prof", "f_phd"]].sum(axis=1, min_count=1)
    grad = d[["m_ma", "m_prof", "m_phd", "f_ma", "f_prof", "f_phd"]].sum(axis=1, min_count=1)
    d["ba_plus_share"] = ba / d.pop25.where(d.pop25 > 0)
    d["grad_share"] = grad / d.pop25.where(d.pop25 > 0)
    bad = ~d.ba_plus_share.between(0, 1) & d.ba_plus_share.notna()
    if bad.mean() > 0.01:
        log.warning("2000 Census education shares look wrong (%.0f%% outside 0 to 1); not used", 100 * bad.mean())
        return pd.DataFrame()
    return d.assign(year=2000)[["zip5", "year", "pop25"] + PANEL_MEASURES]


def place_by_year(zip_year: pd.DataFrame, acs_z: pd.DataFrame, c2000: pd.DataFrame) -> pd.DataFrame:
    """Place measures for (zip5, year) rows: the 2000 Census through 2000, the ACS release ending in
    that year from 2012 (the latest release after it), and a straight line between the 2000 and
    2012 values in 2001 to 2011. place_source records which applies."""
    out = zip_year[["zip5", "year"]].drop_duplicates().copy()
    if len(acs_z):
        a = acs_z.rename(columns={"acs_year": "year"})[["zip5", "year"] + PANEL_MEASURES]
        lo, hi = int(a.year.min()), int(a.year.max())
    else:
        a, lo, hi = pd.DataFrame(columns=["zip5", "year"] + PANEL_MEASURES), 9999, 9999
    out["src_year"] = out.year.clip(upper=hi)
    res = out.merge(a.rename(columns={"year": "src_year"}), on=["zip5", "src_year"], how="left")
    res["place_source"] = np.where(res.year >= lo, "acs", None)
    if len(c2000):
        base = c2000.set_index("zip5")[PANEL_MEASURES]
        first = a[a.year == lo].set_index("zip5")[PANEL_MEASURES] if len(a) else base.iloc[0:0]
        early = res.year <= 2000
        for m in PANEL_MEASURES:
            res.loc[early, m] = res.loc[early, "zip5"].map(base[m]).values
        res.loc[early & res[PANEL_MEASURES[1]].notna(), "place_source"] = "census2000"
        mid = (res.year > 2000) & (res.year < lo)
        w = ((res.loc[mid, "year"] - 2000) / (lo - 2000)).values
        for m in PANEL_MEASURES:
            v0 = res.loc[mid, "zip5"].map(base[m]).values
            v1 = res.loc[mid, "zip5"].map(first[m]).values if len(first) else np.full(mid.sum(), np.nan)
            res.loc[mid, m] = v0 + (v1 - v0) * w
        res.loc[mid & res[PANEL_MEASURES[1]].notna(), "place_source"] = "interpolated"
    return res.drop(columns="src_year")


def ipeds(paths: Paths, web) -> pd.DataFrame | None:
    """Degree-granting institutions with location and Carnegie basic classification."""
    dest = paths.raw / "ipeds"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = sio.Manifest(paths.manifest)
    for y in IPEDS_YEARS:
        f = dest / f"HD{y}.zip"
        try:
            sio.download(web, IPEDS_URL.format(year=y), f, manifest, "nces_ipeds_hd", retries=2, timeout=300)
        except Exception:
            continue
        with zipfile.ZipFile(f) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".csv") and "_rv" not in n.lower()]
            name = name or [n for n in zf.namelist() if n.lower().endswith(".csv")]
            raw = zf.read(name[0])
        try:
            d = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            d = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="latin-1")
        # strip a byte-order mark however it was decoded, and any stray spaces or quotes
        d.columns = [re.sub(r"^[^A-Za-z]+", "", c).strip().strip('"').upper() for c in d.columns]
        if "UNITID" not in d.columns:
            log.warning("IPEDS %s has no UNITID column (%s ...); skipped", y, list(d.columns)[:5])
            continue
        carnegie = next((c for c in ("C21BASIC", "C18BASIC", "C15BASIC") if c in d.columns), None)
        out = pd.DataFrame({
            "unitid": d.UNITID, "name": d.INSTNM, "zip5": d.ZIP.str[:5],
            "lat": pd.to_numeric(d.get("LATITUDE"), errors="coerce"), "lon": pd.to_numeric(d.get("LONGITUD"), errors="coerce"),
            "level": pd.to_numeric(d.get("ICLEVEL"), errors="coerce"),          # 1 = four-year or above
            "degree_granting": pd.to_numeric(d.get("DEGGRANT"), errors="coerce").eq(1),
            "carnegie": pd.to_numeric(d[carnegie], errors="coerce") if carnegie else np.nan,
            "active": d.get("CYACTIVE", pd.Series("1", index=d.index)).eq("1"),
        })
        out = out[out.degree_granting & out.active & out.lat.notna()]
        out["r1"] = out.carnegie.eq(15)
        out["r1r2"] = out.carnegie.isin([15, 16])
        out["four_year"] = out.level.eq(1)
        log.info("IPEDS %s: %s degree-granting institutions (%s R1)", y, f"{len(out):,}", int(out.r1.sum()))
        return out.assign(ipeds_year=y)
    log.warning("IPEDS directory not downloaded; university fields left empty")
    return None


def _tree(lat, lon):
    from sklearn.neighbors import BallTree
    return BallTree(np.radians(np.c_[lat, lon]), metric="haversine")


def run(paths: Paths, census_key: str | None = None) -> None:
    web = sio.session()
    key = census_key or os.environ.get("CENSUS_API_KEY")
    if key and key.strip().upper() in ("YOUR_KEY", "YOUR-KEY", ""):
        key = None
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    zc = pd.read_parquet(paths.interim / "geo" / "zcta_centroids.parquet").dropna().drop_duplicates("zip5")
    first = (pd.read_parquet(paths.processed / "sbir_firm_addresses.parquet")
             .sort_values(["first_used", "firm_id_sbir"], kind="stable").drop_duplicates("firm_id_sbir")
             [["firm_id_sbir", "zip5"]])
    f = firms[["firm_id_sbir", "first_award_date"]].merge(first, on="firm_id_sbir", how="left").merge(zc, on="zip5", how="left")
    f["year"] = f.first_award_date.dt.year
    out = f[["firm_id_sbir"]].set_index(f.firm_id_sbir).drop(columns="firm_id_sbir")
    stats = {"firms": len(f), "firms_with_zcta_centroid": int(f.lat.notna().sum())}

    api_ok = True
    missing = [y for y in ACS_YEARS if not (paths.interim / "places" / f"acs5_zcta_{y}.parquet").exists()]
    if missing:
        try:
            key = _check_api(web, key)
        except RuntimeError as exc:
            log.warning("%s Using the %d ACS releases already on disk.", exc, len(ACS_YEARS) - len(missing))
            api_ok = False
    else:
        log.info("all ACS releases %d to %d already on disk; the Census API is not needed", ACS_YEARS[0], ACS_YEARS[-1])
    z = acs(paths, web, key if api_ok else None, fetch=api_ok)
    if len(z):
        years = sorted(z.acs_year.unique())
        f["acs_year"] = f.year.clip(min(years), max(years))
        z.to_parquet(paths.processed / "zcta_characteristics.parquet", index=False)
        own = f.merge(z, on=["zip5", "acs_year"], how="left").set_index("firm_id_sbir")
        for c in ("pop", "median_age", "ba_plus_share", "grad_share", "median_hh_income", "unemployment_rate",
                  "median_home_value"):
            out[f"zip_{c}"] = own[c]
        out["acs_year_used"] = own.acs_year
        # the same measures pooled over every ZCTA within RADIUS_MILES, a closer stand-in for the local labor market
        zz = z.merge(zc, on="zip5")
        loc = f[f.lat.notna()].reset_index(drop=True)
        tree = _tree(zc.lat.values, zc.lon.values)
        nb = tree.query_radius(np.radians(loc[["lat", "lon"]].values), r=RADIUS_MILES / 3958.8)
        pairs = pd.DataFrame({"firm_id_sbir": np.repeat(loc.firm_id_sbir.values, [len(n) for n in nb]),
                              "acs_year": np.repeat(loc.acs_year.values, [len(n) for n in nb]),
                              "zip5": zc.zip5.values[np.concatenate(nb)] if len(nb) else []})
        agg = (pairs.merge(zz[["zip5", "acs_year", "pop", "pop25", "ba_plus", "grad", "civ_labor_force", "unemployed"]],
                           on=["zip5", "acs_year"])
               .groupby("firm_id_sbir")[["pop", "pop25", "ba_plus", "grad", "civ_labor_force", "unemployed"]].sum())
        out["area_pop"] = agg["pop"]
        out["area_ba_plus_share"] = agg.ba_plus / agg.pop25.where(agg.pop25 > 0)
        out["area_grad_share"] = agg.grad / agg.pop25.where(agg.pop25 > 0)
        out["area_unemployment_rate"] = agg.unemployed / agg.civ_labor_force.where(agg.civ_labor_force > 0)
        stats["acs_years"] = [int(y) for y in years]
        stats["firms_with_zip_acs"] = int(out.zip_ba_plus_share.notna().sum())

    have_2000 = (paths.interim / "places" / "census2000_zcta.parquet").exists()
    c2000 = census2000(paths, web, key) if (api_ok or have_2000) else pd.DataFrame()
    if len(c2000):
        c2000.to_parquet(paths.processed / "zcta_characteristics_2000.parquet", index=False)
        stats["census2000_zctas"] = int(len(c2000))

    u = ipeds(paths, web)
    if u is not None:
        u.to_parquet(paths.processed / "institutions.parquet", index=False)
        loc = f[f.lat.notna()].reset_index(drop=True)
        pts = np.radians(loc[["lat", "lon"]].values)
        r = RADIUS_MILES / 3958.8
        for label, sub in (("univ", u), ("univ_4yr", u[u.four_year]), ("univ_r1r2", u[u.r1r2]), ("univ_r1", u[u.r1])):
            if len(sub):
                cnt = _tree(sub.lat.values, sub.lon.values).query_radius(pts, r=r, count_only=True)
                out.loc[loc.firm_id_sbir, f"{label}_n_{RADIUS_MILES}mi"] = cnt
        for label, sub in (("r1", u[u.r1]), ("r1r2", u[u.r1r2])):
            if len(sub):
                d, _ = _tree(sub.lat.values, sub.lon.values).query(pts, k=1)
                out.loc[loc.firm_id_sbir, f"miles_to_nearest_{label}"] = (d[:, 0] * 3958.8).round(1)
        stats["institutions"] = int(len(u))
        stats["ipeds_year"] = int(u.ipeds_year.iloc[0])

    out.reset_index().to_parquet(paths.processed / "firm_places.parquet", index=False)
    (paths.audit / "places.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("places: %s", stats)
