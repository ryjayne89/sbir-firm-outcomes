"""Geocode firm addresses to 2010 census tracts and assign Opportunity Zone status.

Addresses go through three passes with the Census batch geocoder (2010 tract vintage, the
vintage of the 2018 Opportunity Zone designations):

    1. the address on the firm's first award, as reported
    2. the same address cleaned: unit and suite designators removed, Address2 used when
       Address1 has no street number
    3. the firm's other reported addresses, cleaned, earliest first

Firms that still cannot be placed get a ZIP-level measure instead: the share of the ZIP's
2010 population living in designated tracts, from the Census ZCTA-to-tract relationship file.
Every firm carries the source of its tract and, when unmatched, the reason.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time

import numpy as np
import pandas as pd

from . import io as sio
from .config import GEOCODE_BATCH, Paths

log = logging.getLogger(__name__)

GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
ZCTA_TRACT_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_tract_rel_10.txt"
COLS = ["id", "input", "match", "match_type", "matched", "coords", "tiger_id", "side", "st", "co", "tract", "block"]
MAX_MILES_FROM_ZIP = 30

_UNIT = re.compile(r"\b(STE|SUITE|UNIT|APT|BLDG|BUILDING|FL|FLR|FLOOR|RM|ROOM|DEPT|MS|MAIL STOP|SPC|LOT|BOX)\b\.?\s*[A-Z0-9-]*"
                   r"|#\s*[A-Z0-9-]+")
_PO_BOX = re.compile(r"\bP\s*O\s*BOX\b|\bPOST OFFICE BOX\b|\bPOB\b")


def clean_street(address1: pd.Series, address2: pd.Series) -> pd.Series:
    def norm(s):
        s = s.astype("string").str.upper().str.replace(r"[.,]", " ", regex=True)
        return s.str.replace(r"\s+", " ", regex=True).str.strip()

    a1, a2 = norm(address1), norm(address2)
    has_num = lambda s: s.fillna("").str.contains(r"^\s*\d") | s.fillna("").str.contains(r"\b\d+\s+[A-Z]")
    street = a1.where(has_num(a1) & ~a1.fillna("").str.contains(_PO_BOX), a2.where(has_num(a2)))
    street = street.str.replace(_UNIT, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return street.where(street.fillna("").str.len() > 3)


def failure_reason(df: pd.DataFrame, zctas: set[str]) -> pd.Series:
    a1 = df.address1.astype("string").fillna("").str.upper()
    has_digit = a1.str.contains(r"\d") | df.address2.astype("string").fillna("").str.contains(r"\d")
    return pd.Series(np.select(
        [a1.str.strip().eq(""),
         a1.str.replace(r"[.,]", " ", regex=True).str.contains(_PO_BOX),
         ~has_digit,
         df.state.isna(),
         df.zip5.isna() | ~df.zip5.isin(zctas),
         df.match.eq("Tie")],
        ["blank_address", "po_box", "no_street_number", "non_us_or_missing_state", "zip_not_a_zcta", "tie"],
        "no_match"), index=df.index)


def _post(chunk: pd.DataFrame, web, tries: int = 3) -> str | None:
    buf = io.StringIO()
    chunk[["id", "street", "city", "state", "zip5"]].to_csv(buf, index=False, header=False)
    for t in range(tries):
        try:
            r = web.post(GEOCODER, files={"addressFile": ("addr.csv", buf.getvalue())},
                         data={"benchmark": "Public_AR_Current", "vintage": "Census2010_Current"}, timeout=900)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            log.warning("geocoder attempt %d failed (%s)", t + 1, type(exc).__name__)
            time.sleep(30 * (t + 1))
    return None


def _geocode(df: pd.DataFrame, cache, web) -> pd.DataFrame:
    """Geocode rows with columns id, street, city, state, zip5, caching each batch on disk.

    Batches are named by a hash of their contents, so a changed input never reuses a stale result."""
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for k in range(0, len(df), GEOCODE_BATCH):
        chunk = df.iloc[k:k + GEOCODE_BATCH]
        digest = hashlib.sha1(chunk[["id", "street", "city", "state", "zip5"]].to_csv(index=False).encode()).hexdigest()[:10]
        out = cache / f"batch_{digest}.csv"
        if not out.exists():
            text = _post(chunk, web)
            if text is None:
                pieces = [_post(chunk.iloc[j:j + 1000], web) for j in range(0, len(chunk), 1000)]
                text = None if any(p is None for p in pieces) else "".join(p if p.endswith("\n") else p + "\n" for p in pieces)
            if text is None:
                log.warning("geocoder batch %s failed; rerun to retry", out.name)
                continue
            out.write_text(text)
        parts.append(pd.read_csv(out, header=None, names=COLS, dtype=str, on_bad_lines="skip"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def _verified(g: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    """Keep geocoder rows whose echoed input street is the street that was sent under that ID."""
    if g.empty:
        return g
    alnum = lambda s: s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    echoed = alnum(g.set_index("id").input)
    expect = alnum(sent.set_index("id").street.reindex(echoed.index))
    ok = pd.Series([isinstance(e, str) and isinstance(x, str) and e.startswith(x) for e, x in zip(echoed, expect)],
                   index=echoed.index)
    if (~ok).any():
        log.warning("geocoder: %s results did not echo the address sent under their ID and were dropped", int((~ok).sum()))
    return g[g.id.map(ok).fillna(False).astype(bool)]


def _tract(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g.match.eq("Match")].copy()
    g["tract_geoid_2010"] = g.st.str.zfill(2) + g.co.str.zfill(3) + g.tract.str.zfill(6)
    xy = g.coords.str.split(",", expand=True)
    g["lon"], g["lat"] = pd.to_numeric(xy[0], errors="coerce"), pd.to_numeric(xy[1], errors="coerce")
    return g[["id", "tract_geoid_2010", "match_type", "lat", "lon"]]


def zcta_oz_share(paths: Paths, web, qoz: set[str]) -> pd.Series | None:
    raw = paths.raw / "geo" / "zcta_tract_rel_10.txt"
    try:
        sio.download(web, ZCTA_TRACT_URL, raw, sio.Manifest(paths.manifest), "census_zcta_tract_rel_2010", timeout=300)
    except Exception:
        log.warning("Could not download %s. Save it to %s to add the ZIP-level Opportunity Zone share.",
                    ZCTA_TRACT_URL, raw)
        return None
    rel = pd.read_csv(raw, dtype=str, encoding_errors="replace")
    rel.columns = [c.strip().lstrip("\ufeff").upper() for c in rel.columns]
    zcta, geoid = sio.col(rel, "ZCTA5", "ZCTA5CE10"), sio.col(rel, "GEOID", "GEOID10")
    popn = sio.to_num(sio.col(rel, "POPPT")).fillna(0)
    land = sio.to_num(sio.col(rel, "AREALANDPT", "AREALAND_PT", "ALANDPT", "AREAPT")).fillna(0)
    d = pd.DataFrame({"zip5": zcta.str.zfill(5), "oz": geoid.str.zfill(11).isin(qoz).astype(float),
                      "pop": popn, "land": land}).dropna(subset=["zip5"])
    d["pop_oz"], d["land_oz"] = d["pop"] * d.oz, d.land * d.oz
    g = d.groupby("zip5")[["pop", "land", "pop_oz", "land_oz"]].sum()
    # population share where the ZIP is populated, land-area share otherwise
    pop = (g.pop_oz / g["pop"]).where(g["pop"] > 0, g.land_oz / g.land.where(g.land > 0))
    return pop.rename("oz_zip_share")


def run(paths: Paths, web) -> pd.DataFrame:
    addr = pd.read_parquet(paths.processed / "sbir_firm_addresses.parquet")
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet",
                         columns=["firm_id_sbir", "address1", "address2", "city", "state", "zip5"])
    a2 = aw.dropna(subset=["address1"]).drop_duplicates(["firm_id_sbir", "address1"]).set_index(["firm_id_sbir", "address1"]).address2
    addr["address2"] = [a2.get((f, a), pd.NA) for f, a in zip(addr.firm_id_sbir, addr.address1)]
    zctas = set(pd.read_parquet(paths.interim / "geo" / "zcta_centroids.parquet").zip5)
    qoz = set(pd.read_parquet(paths.interim / "geo" / "qoz_tracts_2010.parquet").tract_geoid_2010)
    cache = paths.interim / "geo"

    # pass 1: first-award address as reported. Rows are keyed by firm ID, never by position, and
    # batches are cached under a hash of their contents, so a cached result always belongs to the
    # address that produced it.
    first = (addr.sort_values(["first_used", "firm_id_sbir"], kind="stable").drop_duplicates("firm_id_sbir")
             [["firm_id_sbir", "address1", "address2", "city", "state", "zip5"]].reset_index(drop=True))
    p1_in = first.assign(id="f|" + first.firm_id_sbir, street=first.address1)
    g1 = _verified(_geocode(p1_in, cache / "geocode_2010_v2", web), p1_in)
    status = p1_in[["id", "firm_id_sbir"]].merge(g1[["id", "match"]], on="id", how="left")
    res = _tract(g1).merge(p1_in[["id", "firm_id_sbir"]], on="id").assign(geo_source="first_award_address")

    # pass 2: cleaned first-award address for firms not yet placed
    todo = first[~first.firm_id_sbir.isin(res.firm_id_sbir)].copy()
    todo["street"] = clean_street(todo.address1, todo.address2)
    todo = todo.dropna(subset=["street"])
    todo = todo[todo.street != todo.address1.astype("string").str.upper().str.strip()]
    if len(todo):
        p2 = todo.assign(id="c|" + todo.firm_id_sbir)
        res2 = _tract(_verified(_geocode(p2, cache / "geocode_2010_cleaned", web), p2)).merge(p2[["id", "firm_id_sbir"]], on="id")
        res = pd.concat([res, res2.assign(geo_source="first_award_address_cleaned")], ignore_index=True)

    # pass 3: the firm's other addresses, cleaned, earliest first
    other = addr[~addr.firm_id_sbir.isin(res.firm_id_sbir)].copy()
    other = other.merge(first[["firm_id_sbir", "address1"]].rename(columns={"address1": "_first"}), on="firm_id_sbir")
    other = other[other.address1 != other._first].sort_values(["first_used", "firm_id_sbir"], kind="stable")
    other["street"] = clean_street(other.address1, other.address2)
    other = other.dropna(subset=["street"]).reset_index(drop=True)
    if len(other):
        p3 = other.assign(id="o|" + other.firm_id_sbir + "|" + other.index.astype(str))
        res3 = (_tract(_verified(_geocode(p3, cache / "geocode_2010_other", web), p3))
                .merge(p3[["id", "firm_id_sbir", "first_used"]], on="id")
                .sort_values("first_used", kind="stable").drop_duplicates("firm_id_sbir").drop(columns="first_used"))
        res = pd.concat([res, res3.assign(geo_source="other_award_address")], ignore_index=True)

    res = res.drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
    out = first.set_index("firm_id_sbir")[["state", "zip5"]].join(
        res[["tract_geoid_2010", "geo_source", "match_type", "lat", "lon"]])
    out["opportunity_zone_flag"] = out.tract_geoid_2010.isin(qoz).where(out.tract_geoid_2010.notna())

    # checks on placed firms: tract state agrees with the reported state, and the point is near the ZIP
    fips_to_state = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
                     "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
                     "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
                     "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
                     "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
                     "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
                     "54": "WV", "55": "WI", "56": "WY", "72": "PR"}
    out["geo_state_agrees"] = (out.tract_geoid_2010.str[:2].map(fips_to_state) == out.state).where(out.tract_geoid_2010.notna())
    zc = pd.read_parquet(paths.interim / "geo" / "zcta_centroids.parquet").drop_duplicates("zip5").set_index("zip5")
    zlat, zlon = out.zip5.map(zc.lat), out.zip5.map(zc.lon)
    r = np.radians
    d = 2 * 3958.8 * np.arcsin(np.sqrt(np.sin((r(out.lat) - r(zlat)) / 2) ** 2
                                       + np.cos(r(zlat)) * np.cos(r(out.lat)) * np.sin((r(out.lon) - r(zlon)) / 2) ** 2))
    out["geo_miles_from_zip"] = d.round(1)

    # reasons for firms never placed, judged on the first-award address
    miss = first.set_index("firm_id_sbir").loc[out.index[out.tract_geoid_2010.isna()]].copy()
    miss["match"] = status.set_index("firm_id_sbir").match.reindex(miss.index)
    out.loc[miss.index, "geo_fail_reason"] = failure_reason(miss, zctas)

    share = zcta_oz_share(paths, web, qoz)
    if share is not None:
        out["oz_zip_share"] = out.zip5.map(share)
    out = out.drop(columns=["state", "zip5", "lat", "lon"])

    placed = out.tract_geoid_2010.notna()
    stats = {
        "firms": int(len(out)), "placed": int(placed.sum()), "placed_share": round(float(placed.mean()), 4),
        "by_source": out.geo_source.value_counts().to_dict(),
        "fail_reasons": out.geo_fail_reason.value_counts().to_dict(),
        "state_disagrees": int((out.geo_state_agrees == False).sum()),
        f"over_{MAX_MILES_FROM_ZIP}_miles_from_zip": int((out.geo_miles_from_zip > MAX_MILES_FROM_ZIP).sum()),
        "zip_level_oz_share_available": int(out.get("oz_zip_share", pd.Series(dtype=float)).notna().sum()),
    }
    (paths.audit / "geocode.json").write_text(json.dumps(stats, indent=2, default=str))
    unplaced = first.set_index("firm_id_sbir").join(out[["geo_fail_reason"]], how="inner")
    unplaced = unplaced[unplaced.geo_fail_reason.notna()]
    (unplaced.sample(frac=1, random_state=1).groupby("geo_fail_reason").head(25)
     .sort_values("geo_fail_reason").to_csv(paths.audit / "geocode_unplaced_sample.csv"))
    log.info("geocoding: %s of %s firms placed (%.1f%%); sources %s; unplaced reasons %s; state disagrees for %s",
             f"{stats['placed']:,}", f"{stats['firms']:,}", 100 * stats["placed_share"], stats["by_source"],
             stats["fail_reasons"], f"{stats['state_disagrees']:,}")
    if stats["placed"] and stats["state_disagrees"] > 0.05 * stats["placed"]:
        log.warning("more than 5%% of placed firms fall in a tract outside their reported state; "
                    "check the geocoder cache in %s before using the tract fields", paths.interim / "geo")
    return out
