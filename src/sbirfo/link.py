"""Link SBIR firms to SEC Central Index Keys.

Candidates come from three name comparisons (exact strict key, equal generic-word-free key,
and rapidfuzz token-sort similarity within state and first-letter blocks). Each candidate is
then scored on state, city and ZIP agreement and assigned a tier:

    1  exact name, same state, same city or ZIP          high
    2  exact name, same state                            high
    3  same name core, same state, same city or ZIP      high
    4  fuzzy name, same state, same city or ZIP          high
    5  fuzzy name >= FUZZY_STATE_ONLY, same state         medium
    6  exact name, other state, name unique on both sides medium
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from . import io as sio
from . import names as nm
from .config import FUZZY_MIN, FUZZY_STATE_ONLY, MIN_LOOSE_KEY_MATCH, Paths

log = logging.getLogger(__name__)

SB_COLS = ["firm_id_sbir", "name_key", "loose_key", "state", "city", "zip5", "name_raw"]
SEC_COLS = ["cik", "name_key", "loose_key", "state", "city", "zip5", "name_raw", "sec_source"]


def _cik(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lstrip("0")


def sbir_names(paths: Paths) -> pd.DataFrame:
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet",
                         columns=["firm_id_sbir", "company_name", "city", "state", "zip5"])
    sb = aw.rename(columns={"company_name": "name_raw"}).assign(city=lambda d: nm.clean_city(d.city)).drop_duplicates()
    sb = nm.add_keys(sb)
    return sb[sb.name_key.notna()]


def sec_names(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every (CIK, name, location) from EDGAR current and former names, Form D and Form C."""
    ent = pd.read_parquet(paths.sec / "edgar_entities.parquet")
    ent["cik"] = _cik(ent.cik)
    addr = pd.concat([
        pd.DataFrame({"cik": ent.cik, "name_raw": ent.name, "city": ent[c], "state": ent[s], "zip": ent[z],
                      "sec_source": kind})
        for kind, c, s, z in [("edgar_business", "biz_city", "biz_state", "biz_zip"),
                              ("edgar_mailing", "mail_city", "mail_state", "mail_zip")]], ignore_index=True)

    fn = pd.read_parquet(paths.sec / "edgar_former_names.parquet")
    former = (fn.assign(cik=_cik(fn.cik))[["cik", "name"]].rename(columns={"name": "name_raw"})
              .merge(addr[["cik", "city", "state", "zip"]].drop_duplicates(), on="cik", how="left")
              .assign(sec_source="edgar_former"))

    iss = pd.read_parquet(paths.sec / "formd_issuers.parquet")
    loc = ["CITY", "STATEORCOUNTRY", "ZIPCODE"]
    fd = [iss[["CIK", "ENTITYNAME"] + loc].set_axis(["cik", "name_raw", "city", "state", "zip"], axis=1)
          .assign(sec_source="formd")]
    for c in [c for c in iss.columns if re.match(r"(ISSUER|EDGAR)_PREVIOUSNAME_\d", c)]:
        fd.append(iss[["CIK", c] + loc].set_axis(["cik", "name_raw", "city", "state", "zip"], axis=1)
                  .assign(sec_source="formd_previous"))
    fd = pd.concat(fd, ignore_index=True).assign(cik=lambda d: _cik(d.cik))

    cs = pd.read_parquet(paths.sec / "formc_form_c_submission.parquet")
    ci = pd.read_parquet(paths.sec / "formc_form_c_issuer_information.parquet")
    fc = (cs[["ACCESSION_NUMBER", "CIK"]].merge(ci, on="ACCESSION_NUMBER")
          [["CIK", "NAMEOFISSUER", "CITY", "STATEORCOUNTRY", "ZIPCODE"]]
          .set_axis(["cik", "name_raw", "city", "state", "zip"], axis=1).assign(sec_source="formc"))
    fc["cik"] = _cik(fc.cik)

    sec = pd.concat([addr, former, fd, fc], ignore_index=True)
    sec = sec[sec.name_raw.notna() & sec.cik.fillna("").ne("")]
    sec["state"], sec["zip5"], sec["city"] = nm.clean_state(sec.state), nm.clean_zip(sec.zip), nm.clean_city(sec.city)
    sec = sec.drop(columns="zip").drop_duplicates(["cik", "name_raw", "city", "state", "zip5"])
    sec = nm.add_keys(sec)
    sec = sec[sec.name_key.notna()]

    off = pd.read_parquet(paths.sec / "formd_offering.parquet", columns=["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE"])
    fund = (iss[["ACCESSIONNUMBER", "CIK"]].merge(off, on="ACCESSIONNUMBER")
            .assign(cik=lambda d: _cik(d.CIK), is_fund=lambda d: d.INDUSTRYGROUPTYPE.eq("Pooled Investment Fund"))
            .groupby("cik").is_fund.mean().rename("fund_share"))
    n_fil = _cik(pd.read_parquet(paths.sec / "edgar_filings.parquet", columns=["cik"]).cik).value_counts()
    info = ent.set_index("cik")[["name", "entity_type", "sic", "tickers", "exchanges"]].join(fund, how="outer")
    info["n_sec_filings"] = n_fil.reindex(info.index).fillna(0).astype(int)
    log.info("SEC names: %s rows for %s CIKs", f"{len(sec):,}", f"{sec.cik.nunique():,}")
    return sec, info


def candidates(sb: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    from rapidfuzz import fuzz, process

    sbl, secl = sb[sb.linkable][SB_COLS], sec[sec.linkable][SEC_COLS]

    exact = sbl.merge(secl, on="name_key", suffixes=("_sb", "_sec")).assign(method="exact", score=100.0)
    exact = exact.rename(columns={"name_key": "name_key_sb"}).assign(name_key_sec=lambda d: d.name_key_sb)

    ls = sbl[sbl.loose_key.str.len() >= MIN_LOOSE_KEY_MATCH]
    lc = secl[secl.loose_key.str.len() >= MIN_LOOSE_KEY_MATCH]
    loose = ls.merge(lc, on="loose_key", suffixes=("_sb", "_sec"))
    loose = loose[loose.name_key_sb != loose.name_key_sec].assign(method="loose", score=99.0)

    digits = lambda s: re.sub(r"\D", "", s)
    q = (sbl[~sbl.name_key.isin(set(exact.name_key_sb))].drop_duplicates(["name_key", "state"])
         .dropna(subset=["state"]))
    c = secl.drop_duplicates(["name_key", "state"]).dropna(subset=["state"])
    q, c = q.assign(blk=q.state + q.loose_key.str[0]), c.assign(blk=c.state + c.loose_key.str[0])
    blocks = dict(tuple(c.groupby("blk")))
    pairs = []
    for blk, qq in q.groupby("blk"):
        cc = blocks.get(blk)
        if cc is None:
            continue
        m = process.cdist(qq.name_key.tolist(), cc.name_key.tolist(), scorer=fuzz.token_sort_ratio,
                          score_cutoff=FUZZY_MIN, dtype=np.uint8, workers=-1)
        i, j = np.nonzero(m)
        if len(i):
            pairs.append(pd.DataFrame({"name_key_sb": qq.name_key.values[i], "name_key_sec": cc.name_key.values[j],
                                       "score": m[i, j].astype(float)}))
    fz = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["name_key_sb", "name_key_sec", "score"])
    fz = fz[fz.name_key_sb.map(digits) == fz.name_key_sec.map(digits)]
    fuzzy = (sbl.rename(columns={"name_key": "name_key_sb"}).merge(fz, on="name_key_sb")
             .merge(secl.rename(columns={"name_key": "name_key_sec"}), on="name_key_sec", suffixes=("_sb", "_sec"))
             .assign(method="fuzzy"))
    log.info("candidates: exact %s, loose %s, fuzzy %s name pairs", f"{len(exact):,}", f"{len(loose):,}", f"{len(fz):,}")
    return pd.concat([exact, loose, fuzzy], ignore_index=True)


def assign_tiers(cand: pd.DataFrame, sb: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()
    cand["same_state"] = sio.same(cand.state_sb, cand.state_sec)
    cand["same_city"] = sio.same(cand.city_sb, cand.city_sec) & cand.same_state
    cand["same_zip"] = sio.same(cand.zip5_sb, cand.zip5_sec)
    local = cand.same_city | cand.same_zip
    firms_per_key = sb[sb.linkable].groupby("name_key").firm_id_sbir.nunique()
    ciks_per_key = sec[sec.linkable].groupby("name_key").cik.nunique()
    unique = ((cand.name_key_sb.map(firms_per_key) == 1) & (cand.name_key_sb.map(ciks_per_key) == 1)).fillna(False).astype(bool)
    m, score = cand.method.astype(str), cand.score.astype(float)
    cand["tier"] = np.select(
        [(m == "exact") & cand.same_state & local,
         (m == "exact") & cand.same_state,
         (m == "loose") & cand.same_state & local,
         (m == "fuzzy") & cand.same_state & local,
         (m == "fuzzy") & cand.same_state & (score >= FUZZY_STATE_ONLY),
         (m == "exact") & ~cand.same_state & unique],
        [1, 2, 3, 4, 5, 6], 0)
    return cand[cand.tier > 0]


def choose(cand: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    best = (cand.sort_values(["tier", "score"], ascending=[True, False])
            .drop_duplicates(["firm_id_sbir", "cik"]).copy())
    best["confidence"] = np.where(best.tier <= 4, "high", "medium")
    best = best.join(info[["fund_share", "n_sec_filings", "entity_type", "tickers"]], on="cik")
    best["n_sec_filings"] = best.n_sec_filings.fillna(0).astype(int)
    best["is_fund_cik"] = best.fund_share.fillna(0) > 0.5
    best["n_firms_for_cik"] = best.groupby("cik").firm_id_sbir.transform("nunique")
    best["cik10"] = best.cik.str.zfill(10)
    best["_shared"] = best.n_firms_for_cik > 1
    best["_cik_int"] = best.cik.astype("int64")
    primary = (best.sort_values(["firm_id_sbir", "tier", "is_fund_cik", "_shared", "n_sec_filings", "_cik_int"],
                                ascending=[True, True, True, True, False, True])
               .drop_duplicates("firm_id_sbir"))
    best["is_primary"] = best.index.isin(primary.index)
    return best


def run(paths: Paths) -> None:
    sb = sbir_names(paths)
    sec, info = sec_names(paths)
    cand = assign_tiers(candidates(sb, sec), sb, sec)
    best = choose(cand, info)

    links = best[["firm_id_sbir", "cik10", "tier", "confidence", "method", "score", "same_state", "same_city",
                  "same_zip", "name_raw_sb", "name_raw_sec", "city_sb", "state_sb", "zip5_sb", "city_sec",
                  "state_sec", "zip5_sec", "sec_source", "is_fund_cik", "n_firms_for_cik", "n_sec_filings",
                  "tickers", "is_primary"]].rename(columns={"cik10": "cik"})
    links.to_parquet(paths.processed / "link_sbir_sec.parquet", index=False)

    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    p = links[links.is_primary].set_index("firm_id_sbir")
    summary = links.groupby("firm_id_sbir").agg(n_ciks_high=("confidence", lambda s: (s == "high").sum()),
                                                n_ciks_any=("cik", "size"))
    add = (p[["cik", "tier", "confidence", "is_fund_cik", "n_firms_for_cik", "tickers"]]
           .rename(columns={"cik": "firm_cik_sec", "tier": "link_tier", "confidence": "link_confidence",
                            "is_fund_cik": "primary_cik_is_fund", "n_firms_for_cik": "primary_cik_n_firms",
                            "tickers": "sec_tickers"})
           .join(summary))
    firms = firms.drop(columns=[c for c in add.columns if c in firms.columns]).merge(
        add, left_on="firm_id_sbir", right_index=True, how="left")
    firms[["n_ciks_high", "n_ciks_any"]] = firms[["n_ciks_high", "n_ciks_any"]].fillna(0).astype(int)
    firms.to_parquet(paths.processed / "sbir_firms.parquet", index=False)

    hi = firms.link_confidence.eq("high")
    stats = {
        "pairs_by_tier": {int(k): int(v) for k, v in links.tier.value_counts().sort_index().items()},
        "firms": len(firms), "firms_linked_high": int(hi.sum()),
        "firms_linked_any": int(firms.firm_cik_sec.notna().sum()),
        "ciks_shared_by_firms": int((links.drop_duplicates("cik").n_firms_for_cik > 1).sum()),
    }
    (paths.audit / "link.json").write_text(json.dumps(stats, indent=2))
    log.info("linked firms: %s high, %s any, of %s", f"{stats['firms_linked_high']:,}",
             f"{stats['firms_linked_any']:,}", f"{stats['firms']:,}")
