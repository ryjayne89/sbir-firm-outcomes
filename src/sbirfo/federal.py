"""Federal contracts and grants from USAspending.gov, and SBIR Phase III identification.

Every UEI and DUNS number in a firm's SBIR record is searched in USAspending's award search,
separately for contracts (types A to D) and grants (02 to 05). A returned award is kept only
when its recipient UEI or DUNS equals one of the firm's identifiers, so a name search never
creates a link. USAspending is complete from fiscal year 2008.

Each award is then classed as

* an SBIR/STTR award itself, when its award ID equals a contract or grant number on one of the
  firm's SBIR.gov awards, its FPDS research code is SR1, SR2, ST1 or ST2, its NIH activity code
  is R41 to R44 or U43/U44, or its description names SBIR or STTR without Phase III;
* an SBIR/STTR Phase III contract, when its FPDS research code is SR3 or ST3, or its
  description names SBIR or STTR Phase III;
* otherwise a non-SBIR federal award.

Award numbers are compared as letters and digits, and NIH grants also by institute code and
serial number, since Phase I and II of one NIH project share them. When an SBIR.gov award number
is not found under the firm, it is looked up directly; the recipient UEI or DUNS on that award
is added to the firm and searched as well, which recovers DUNS numbers SBIR.gov never recorded.

The FPDS research code is not returned by the search or award detail endpoints, only in the
award download files, so it is read through the download service for contracts that could be
Phase III. All requests are cached in shards under data/interim/usaspending, so an interrupted
run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as sio
from . import names as nm
from .config import USASPENDING_COVERAGE_START, Paths

log = logging.getLogger(__name__)

API = "https://api.usaspending.gov/api/v2"
USER_AGENT = "sbir-firm-outcomes (research data pipeline; python-requests)"
GROUPS = {"contract": ["A", "B", "C", "D"], "grant": ["02", "03", "04", "05"]}
BASE_FIELDS = ["Award ID", "Recipient Name", "Recipient UEI", "Recipient DUNS Number", "Start Date", "End Date",
               "Base Obligation Date", "Award Amount", "Awarding Agency", "Awarding Sub Agency", "Funding Agency",
               "Description", "generated_internal_id"]
EXTRA_FIELDS = {"contract": ["Contract Award Type", "NAICS", "PSC"], "grant": ["Award Type", "CFDA Number"]}
BATCH = 10            # identifiers per search request
SHARD = 50            # requests per cache shard
WORKERS = 1           # parallel requests; USAspending drops connections at 4

SBIR_TEXT = re.compile(r"\bSBIR\b|\bSTTR\b|SMALL BUSINESS INNOVATI|SMALL BUSINESS TECHNOLOGY TRANSFER", re.I)
PHASE3_TEXT = re.compile(r"PHASE[\s-]*(?:III|3)\b", re.I)
NIH_SBIR = re.compile(r"^\d?(?:R4[1-4]|U4[34]|SB1)[A-Z]{2}\d{6}")
RESEARCH_SBIR = {"SR1", "SR2", "ST1", "ST2"}
RESEARCH_PHASE3 = {"SR3", "ST3"}


def norm_award_id(s: pd.Series) -> pd.Series:
    """Award numbers compared as upper-case letters and digits, with NIH year and support suffixes removed."""
    x = s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    nih = x.str.extract(r"^\d?([RUFSK]\d{2}[A-Z]{2}\d{6})", expand=False)
    x = nih.fillna(x)
    return x.where(x.str.len() >= 6)


class Shards:
    """Results of many API requests, saved a few hundred requests at a time."""

    def __init__(self, folder: Path, prefix: str):
        self.folder, self.prefix = Path(folder), prefix
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rows: list[pd.DataFrame] = []
        self.keys: list[str] = []

    def load(self) -> tuple[pd.DataFrame, set[str]]:
        rows = [pd.read_parquet(f) for f in sorted(self.folder.glob(f"{self.prefix}_rows_*.parquet"))]
        keys = [pd.read_parquet(f) for f in sorted(self.folder.glob(f"{self.prefix}_keys_*.parquet"))]
        done = set(pd.concat(keys).key) if keys else set()
        return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), done

    def add(self, key: str, rows: pd.DataFrame | None) -> None:
        self.keys.append(key)
        if rows is not None and len(rows):
            self.rows.append(rows.astype("string"))
        if len(self.keys) >= SHARD:
            self.flush()

    def flush(self) -> None:
        if not self.keys:
            return
        tag = f"{time.strftime('%Y%m%d%H%M%S')}_{hashlib.sha1(''.join(self.keys).encode()).hexdigest()[:8]}"
        if self.rows:
            pd.concat(self.rows, ignore_index=True).to_parquet(self.folder / f"{self.prefix}_rows_{tag}.parquet", index=False)
        pd.DataFrame({"key": self.keys}).to_parquet(self.folder / f"{self.prefix}_keys_{tag}.parquet", index=False)
        self.rows, self.keys = [], []


class JobFailed(RuntimeError):
    """A download job was accepted but failed on the server, usually because it was too large."""


class Refused(RuntimeError):
    """USAspending kept dropping connections; progress so far is cached."""


_failures = {"in_a_row": 0}
MAX_FAILURES_IN_A_ROW = 10
MIN_GAP = 0.25        # seconds between requests from one worker


def _request(web, method: str, url: str, body: dict | None = None, tries: int = 8) -> dict | None:
    """One API call with retries. USAspending needs no key. From cloud hosts it refuses a share of
    new connections at random and at once, while requests that connect finish in under a second,
    so a refused request is retried quickly (1 s, then 2, 4, 8, 15, 30 and 60 s)."""
    for attempt in range(tries):
        time.sleep(MIN_GAP)
        try:
            r = web.post(url, json=body, timeout=180) if method == "POST" else web.get(url, timeout=180)
            if r.status_code == 404:
                return None
            if r.status_code in (400, 422):
                log.warning("USAspending rejected a request (%s): %s", r.status_code, r.text[:300])
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            _failures["in_a_row"] = 0
            return r.json()
        except Exception as exc:
            if attempt == tries - 1:
                _failures["in_a_row"] += 1
                log.warning("USAspending request failed after %d tries (%s)", tries, type(exc).__name__)
                if _failures["in_a_row"] >= MAX_FAILURES_IN_A_ROW:
                    raise Refused(f"{MAX_FAILURES_IN_A_ROW} USAspending requests in a row failed. Progress is "
                                  "saved; rerun the federal step later and it resumes where it stopped.")
                return None
            web.close()     # drop pooled connections the server may have closed
            time.sleep([1, 2, 4, 8, 15, 30, 60][min(attempt, 6)])
    return None


def _flat(v):
    if isinstance(v, dict):
        return v.get("code") or v.get("name") or json.dumps(v)
    return v


def search(web, ids: list[str], group: str, end_date: str) -> pd.DataFrame | None:
    """Every award of one type group whose recipient matches any of ids, all pages. None on failure."""
    body = {"filters": {"recipient_search_text": ids, "award_type_codes": GROUPS[group],
                        "time_period": [{"start_date": str(USASPENDING_COVERAGE_START.date()), "end_date": end_date}]},
            "fields": BASE_FIELDS + EXTRA_FIELDS[group], "limit": 100, "sort": "Award Amount", "order": "desc"}
    out, page = [], 1
    while True:
        js = _request(web, "POST", f"{API}/search/spending_by_award/", {**body, "page": page})
        if js is None:
            return None
        out += js.get("results", [])
        if not (js.get("page_metadata") or {}).get("hasNext") or page >= 100:
            break
        page += 1
    df = pd.DataFrame([{k: _flat(v) for k, v in r.items()} for r in out])
    return df.assign(award_group=group) if len(df) else df


NAME_FUZZY_MIN = 85


def firm_name_keys(paths: Paths) -> pd.DataFrame:
    """Every strict and loose name key a firm used on its SBIR awards."""
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "company_name"])
    k = nm.add_keys(aw.drop_duplicates().rename(columns={"company_name": "name_raw"}))
    return k.dropna(subset=["name_key"])[["firm_id_sbir", "name_key", "loose_key"]].drop_duplicates()


def recipient_matches_firm(firm_ids: pd.Series, recipients: pd.Series, keys: pd.DataFrame) -> pd.Series:
    """True when the award's recipient name is one of the firm's SBIR names: the same strict key,
    the same distinctive loose key, or a token-sort ratio of at least NAME_FUZZY_MIN. An identifier
    can outlive the firm (an acquired firm's DUNS used by its new parent, or an identifier recovered
    from a prime contractor), and this check keeps such awards out of the firm's outcomes.
    Awards with no recipient name are kept."""
    from rapidfuzz import fuzz

    pairs = pd.DataFrame({"firm_id_sbir": firm_ids.values, "name_raw": recipients.values}).drop_duplicates()
    pairs = nm.add_keys(pairs)
    strict = keys.groupby("firm_id_sbir").name_key.agg(set)
    loose = keys[keys.loose_key.fillna("").str.len() >= 4].groupby("firm_id_sbir").loose_key.agg(set)

    def ok(f, raw, k, lk):
        if not isinstance(raw, str) or not raw.strip():
            return True
        own = strict.get(f, set())
        if k in own or (isinstance(lk, str) and lk in loose.get(f, set())):
            return True
        return isinstance(k, str) and any(fuzz.token_sort_ratio(k, o) >= NAME_FUZZY_MIN for o in own)

    pairs["ok"] = [ok(f, r, k, lk) for f, r, k, lk in zip(pairs.firm_id_sbir, pairs.name_raw, pairs.name_key, pairs.loose_key)]
    look = pairs.set_index(["firm_id_sbir", "name_raw"]).ok
    return pd.Series([look.get((f, r), True) for f, r in zip(firm_ids, recipients)], index=firm_ids.index)


def _identifiers(paths: Paths) -> pd.DataFrame:
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "uei", "duns"])
    ids = pd.concat([aw[["firm_id_sbir", "uei"]].rename(columns={"uei": "id"}).assign(kind="uei"),
                     aw[["firm_id_sbir", "duns"]].rename(columns={"duns": "id"}).assign(kind="duns")])
    ids = ids.dropna(subset=["id"]).drop_duplicates()
    # An identifier held by two firms cannot place an award, so it is dropped.
    return ids[ids.groupby("id").firm_id_sbir.transform("nunique") == 1]


def _parallel(fn, items, workers=WORKERS):
    if workers <= 1:
        for item in items:
            yield item, fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from zip(items, pool.map(fn, items))


def fetch(paths: Paths, web, values: list[str], prefix: str = "search", workers: int = WORKERS) -> pd.DataFrame:
    """Search results for every identifier in values, contracts and grants, cached under prefix."""
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    folder = paths.interim / "usaspending"
    values = sorted(set(values))
    batch_size = _check_batching(web, values, end_date, folder)
    batches = [values[i:i + batch_size] for i in range(0, len(values), batch_size)]
    frames = []
    for group in GROUPS:
        cache = Shards(folder, f"{prefix}_{group}")
        have, done = cache.load()
        todo = [b for b in batches if "|".join(b) not in done]
        log.info("USAspending %s %s: %s of %s requests cached", group, prefix, f"{len(batches) - len(todo):,}", f"{len(batches):,}")
        t0 = time.time()
        try:
            for i, (b, df) in enumerate(_parallel(lambda b: search(web, b, group, end_date), todo, workers), 1):
                if df is not None:
                    cache.add("|".join(b), df)
                if i == 10 or i % 100 == 0:
                    left = (time.time() - t0) / i * (len(todo) - i) / 60
                    log.info("USAspending %s %s: %s of %s, about %.0f min left", group, prefix, f"{i:,}", f"{len(todo):,}", left)
        finally:
            cache.flush()
        frames.append(cache.load()[0])
    return pd.concat(frames, ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()


def search_award_ids(web, numbers: list[str], group: str, end_date: str) -> pd.DataFrame | None:
    """Awards whose award ID (PIID, FAIN or URI) is one of numbers."""
    body = {"filters": {"award_ids": numbers, "award_type_codes": GROUPS[group],
                        "time_period": [{"start_date": str(USASPENDING_COVERAGE_START.date()), "end_date": end_date}]},
            "fields": BASE_FIELDS + EXTRA_FIELDS[group], "limit": 100, "page": 1}
    js = _request(web, "POST", f"{API}/search/spending_by_award/", body)
    if js is None:
        return None
    df = pd.DataFrame([{k: _flat(v) for k, v in r.items()} for r in js.get("results", [])])
    return df.assign(award_group=group) if len(df) else df


def expand_identifiers(paths: Paths, web, a: pd.DataFrame, sb: pd.DataFrame, ids: pd.DataFrame,
                       workers: int = WORKERS) -> pd.DataFrame:
    """Identifiers found by looking up SBIR.gov award numbers that the identifier search missed.

    SBIR.gov often lacks the DUNS a firm used on older awards. Looking the award number up in
    USAspending returns the recipient's UEI or DUNS, which is added to the firm when the award
    number belongs to that firm alone and the identifier belongs to no other firm."""
    own = sb[sb.award_date >= USASPENDING_COVERAGE_START].assign(key=norm_award_id(sb.contract)).dropna(subset=["key"])
    own = own[own.groupby("key").firm_id_sbir.transform("nunique") == 1]
    found = set(zip(a.firm_id_sbir, norm_award_id(a.award_id)))
    miss = own[[(f, k) not in found for f, k in zip(own.firm_id_sbir, own.key)]].drop_duplicates("key")
    if miss.empty:
        return ids.iloc[0:0]
    raw_numbers = sorted(set(miss.contract.dropna().str.strip()) | set(miss.key))
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    batches = [raw_numbers[i:i + 50] for i in range(0, len(raw_numbers), 50)]
    rows = []
    for group in GROUPS:
        cache = Shards(paths.interim / "usaspending", f"awardid_{group}")
        have, done = cache.load()
        todo = [b for b in batches if "|".join(b) not in done]
        log.info("USAspending %s lookup of %s missed SBIR award numbers: %s of %s requests cached",
                 group, f"{len(miss):,}", f"{len(batches) - len(todo):,}", f"{len(batches):,}")
        try:
            for b, df in _parallel(lambda b: search_award_ids(web, b, group, end_date), todo, workers):
                if df is not None:
                    cache.add("|".join(b), df)
        finally:
            cache.flush()
        rows.append(cache.load()[0])
    got = pd.concat(rows, ignore_index=True) if any(len(r) for r in rows) else pd.DataFrame()
    if got.empty:
        return ids.iloc[0:0]
    got["key"] = norm_award_id(sio.col(got, "Award ID"))
    got = got.merge(miss[["key", "firm_id_sbir"]], on="key")
    new = pd.concat([
        pd.DataFrame({"firm_id_sbir": got.firm_id_sbir, "id": sio.col(got, "Recipient UEI").str.upper(), "kind": "uei"}),
        pd.DataFrame({"firm_id_sbir": got.firm_id_sbir, "kind": "duns",
                      "id": sio.col(got, "Recipient DUNS Number").str.replace(r"\D", "", regex=True).str.zfill(9)})])
    new = new.dropna(subset=["id"]).query("id != '000000000' and id != ''").drop_duplicates()
    new = new[new.groupby("id").firm_id_sbir.transform("nunique") == 1]
    new = new[~new.id.isin(ids.id)]
    log.info("award-number lookup added %s identifiers for %s firms", f"{new.id.nunique():,}", f"{new.firm_id_sbir.nunique():,}")
    return new


def _check_batching(web, values: list[str], end_date: str, folder: Path) -> int:
    """Confirm, once, that several identifiers in one search return the union of their single searches."""
    saved = folder / "batch_size.json"
    if saved.exists():
        return json.loads(saved.read_text())["batch_size"]
    size = _probe(web, values, end_date)
    saved.write_text(json.dumps({"batch_size": size}))
    return size


def _probe(web, values: list[str], end_date: str) -> int:
    probe = values[:BATCH]
    if len(probe) < 2:
        return BATCH
    joint = search(web, probe, "contract", end_date)
    single = [search(web, [v], "contract", end_date) for v in probe]
    if joint is None:
        raise RuntimeError("USAspending award search failed; see the warning above. Rerun later, or skip "
                           "this step with --steps that leave out federal.")
    if any(s is None for s in single):
        return BATCH
    n_single = len(pd.concat([s for s in single if len(s)]).drop_duplicates("generated_internal_id")) if any(len(s) for s in single) else 0
    if len(joint) < n_single:
        log.warning("USAspending search does not combine identifiers; searching one at a time")
        return 1
    return BATCH


RESEARCH_COLUMNS = ["contract_award_unique_key", "award_id_piid", "research_code", "research"]


def _download(web, award_ids: list[str], end_date: str, columns: list[str] | None) -> pd.DataFrame | None:
    """Contract prime award summaries for award_ids from the USAspending download service.

    The FPDS research code is in the award download files but not in the search or award
    detail endpoints, so this is the only way to read it."""
    body = {"filters": {"award_ids": award_ids, "award_type_codes": GROUPS["contract"],
                        "time_period": [{"start_date": str(USASPENDING_COVERAGE_START.date()), "end_date": end_date}]},
            "file_format": "csv"}
    if columns:
        body["columns"] = columns
    job = _request(web, "POST", f"{API}/download/awards/", body)
    if not job or not job.get("status_url"):
        return None
    for _ in range(180):                      # up to 30 minutes per job
        time.sleep(10)
        st = _request(web, "GET", job["status_url"]) or {}
        if st.get("status") == "finished":
            break
        if st.get("status") == "failed":
            raise JobFailed(str(st.get("message")))
    else:
        return None
    for attempt in range(4):
        try:
            r = web.get(st.get("file_url") or job["file_url"], timeout=600)
            r.raise_for_status()
            break
        except Exception:
            time.sleep(10 * (attempt + 1))
    else:
        return None
    frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and "subaward" not in name.lower():
                d = pd.read_csv(zf.open(name), dtype=str, low_memory=False)
                if "research_code" in d.columns or "research" in d.columns:
                    frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESEARCH_COLUMNS)
    for c in RESEARCH_COLUMNS:
        if c not in out:
            out[c] = pd.NA
    return out[RESEARCH_COLUMNS]


JOB_SIZE = 500           # award IDs per download job; halved when the server fails a job
MIN_JOB_SIZE = 50
COOLDOWN = 300           # first pause after a refusal, seconds; doubles while refusals continue
COOLDOWN_MAX = 1800
GAP_MAX = 120            # longest spacing between jobs, seconds
MAX_REFUSALS = 12        # about five hours of refusals in a row before the step gives up


def research_codes(paths: Paths, web, contracts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FPDS research codes for contracts, keyed by generated_internal_id (contract_award_unique_key).

    Contracts are requested by award ID (PIID) in download jobs of JOB_SIZE. PIIDs shorter than
    eight characters, mostly old delivery order numbers such as 0001, match thousands of unrelated
    awards and are left out; their Phase III status rests on the award description.

    The download service accepts a burst of jobs and then refuses connections for a while. A
    refusal pauses the step (5 minutes, doubling up to 30) and spaces later jobs further apart
    (up to two minutes). Only after about five hours of refusals in a row does the step finish
    with the codes it has; a later run then requests only the PIIDs still missing."""
    piids = contracts.assign(piid=contracts.award_id.astype("string").str.strip())
    piids = piids[piids.piid.str.len() >= 8]
    numbers = sorted(set(piids.piid))
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = Shards(paths.interim / "usaspending", "research")
    have, done = cache.load()
    done_piids = {x for k in done for x in k.split("|")}
    remaining = [x for x in numbers if x not in done_piids]
    log.info("USAspending research codes: %s of %s PIIDs already downloaded (%s contracts with short PIIDs skipped)",
             f"{len(numbers) - len(remaining):,}", f"{len(numbers):,}", f"{len(contracts) - len(piids):,}")
    t0, refused_in_a_row, size, i, gap, streak = time.time(), 0, JOB_SIZE, 0, 3.0, 0
    try:
        while i < len(remaining):
            b = remaining[i:i + size]
            try:
                df = _download(web, b, end_date, RESEARCH_COLUMNS)
            except JobFailed as exc:
                if size > MIN_JOB_SIZE:
                    size = max(MIN_JOB_SIZE, size // 2)
                    log.info("download job failed on the server (%s); trying %s award IDs per job", exc, size)
                    continue
                log.warning("download job of %s award IDs failed on the server (%s); skipping it", len(b), exc)
                i += len(b)
                continue
            except Refused:
                _failures["in_a_row"] = 0
                df = None
            if df is None:
                # The service accepts a burst of jobs and then refuses connections for a while.
                # Wait it out with growing pauses and space later jobs further apart.
                refused_in_a_row += 1
                if refused_in_a_row > MAX_REFUSALS:
                    log.warning("USAspending kept refusing download jobs for hours; stopping with %s of %s PIIDs done. "
                                "Rerun the federal step to continue.", f"{i:,}", f"{len(remaining):,}")
                    break
                pause = min(COOLDOWN_MAX, COOLDOWN * 2 ** (refused_in_a_row - 1))
                gap, streak = min(GAP_MAX, gap * 2), 0
                log.info("download service refusing; pausing %d min, then one job every %d s", pause // 60, gap)
                time.sleep(pause)
                continue
            refused_in_a_row = 0
            streak += 1
            if streak >= 20 and gap > 3:          # ease back after a long run of accepted jobs
                gap, streak = max(3.0, gap / 2), 0
            cache.add("|".join(b), df)
            cache.flush()
            i += len(b)
            time.sleep(gap)
            if (i // size) % 10 == 0 or i >= len(remaining):
                left = (time.time() - t0) / i * (len(remaining) - i) / 60
                log.info("USAspending research codes: %s of %s PIIDs, about %.0f min left", f"{i:,}", f"{len(remaining):,}", left)
    finally:
        cache.flush()
    have, done = cache.load()
    done_piids = {x for k in done for x in k.split("|")}
    progress = {"piids_requested": len(numbers), "piids_downloaded": len(done_piids & set(numbers)),
                "complete": not (set(numbers) - done_piids)}
    if have.empty:
        return pd.DataFrame(columns=["generated_internal_id", "research_code", "research_description"]), progress
    got = have.rename(columns={"contract_award_unique_key": "generated_internal_id", "research": "research_description"})
    return got.dropna(subset=["generated_internal_id"]).drop_duplicates("generated_internal_id")[
        ["generated_internal_id", "research_code", "research_description"]], progress


def nih_core(keys: pd.Series) -> pd.Series:
    """Institute code and serial number of an NIH grant (GM112196 in R44GM112196). Phase I and
    Phase II of one NIH project share it, and USAspending sometimes carries only one of them."""
    return keys.astype("string").str.extract(r"^[RUFSK]\d{2}([A-Z]{2}\d{6})$", expand=False)


def match_sbir_numbers(a: pd.DataFrame, sbir_awards: pd.DataFrame) -> pd.Series:
    """For each USAspending award, the firm's SBIR.gov award with the same number, if any."""
    own = sbir_awards.assign(award_key=norm_award_id(sbir_awards.contract)).dropna(subset=["award_key"])
    own = own.assign(core=nih_core(own.award_key))
    exact = (a[["firm_id_sbir", "award_key"]].reset_index()
             .merge(own.drop_duplicates(["firm_id_sbir", "award_key"])[["firm_id_sbir", "award_key", "award_id"]],
                    on=["firm_id_sbir", "award_key"]).set_index("index").award_id)
    core = (a.assign(core=nih_core(a.award_key))[["firm_id_sbir", "core"]].dropna().reset_index()
            .merge(own.dropna(subset=["core"]).drop_duplicates(["firm_id_sbir", "core"])[["firm_id_sbir", "core", "award_id"]],
                   on=["firm_id_sbir", "core"]).set_index("index").award_id)
    return exact.combine_first(core).reindex(a.index)


def classify(awards: pd.DataFrame, sbir_awards: pd.DataFrame, firms: pd.DataFrame) -> pd.DataFrame:
    a = awards.reset_index(drop=True).copy()
    a["award_key"] = norm_award_id(a.award_id)
    a["sbir_award_id"] = match_sbir_numbers(a, sbir_awards)
    text = a.description.fillna("")
    research = a.research_code.astype("string").str.upper().str.strip()
    a["desc_sbir"] = text.str.contains(SBIR_TEXT)
    a["desc_phase3"] = a.desc_sbir & text.str.contains(PHASE3_TEXT)
    a["phase3"] = research.isin(RESEARCH_PHASE3).fillna(False) | (a.desc_phase3 & research.isna())
    a["phase3_source"] = np.where(research.isin(RESEARCH_PHASE3).fillna(False), "fpds_research_code",
                                  np.where(a.phase3, "description", None))
    a["sbir_program_award"] = (~a.phase3 & (a.sbir_award_id.notna() | research.isin(RESEARCH_SBIR).fillna(False)
                               | a.award_id.astype("string").str.upper().str.contains(NIH_SBIR, na=False)
                               | (a.desc_sbir & ~a.desc_phase3)))
    a["award_class"] = np.select([a.phase3, a.sbir_program_award], ["sbir_phase3", "sbir_program"], "non_sbir")
    first = firms.set_index("firm_id_sbir").first_award_date
    a["post"] = a.date > a.firm_id_sbir.map(first)
    return a


def sbir_numbers_found(a: pd.DataFrame, sb: pd.DataFrame, firm_ids) -> float | None:
    """Share of SBIR.gov awards since FY2008 (of firms in firm_ids) found in USAspending under the same firm."""
    own = sb[(sb.award_date >= USASPENDING_COVERAGE_START) & sb.firm_id_sbir.isin(firm_ids)].copy()
    own["key"] = norm_award_id(own.contract)
    own = own.dropna(subset=["key"])
    if own.empty:
        return None
    keys = set(zip(a.firm_id_sbir, a.award_key)) | set(zip(a.firm_id_sbir, nih_core(a.award_key)))
    hit = [(f, k) in keys or (f, c) in keys for f, k, c in zip(own.firm_id_sbir, own.key, nih_core(own.key))]
    return round(float(np.mean(hit)), 4)


def firm_fields(a: pd.DataFrame, firms: pd.DataFrame) -> pd.DataFrame:
    f = firms.set_index("firm_id_sbir")
    post = a[a.post]
    ns = post[post.award_class == "non_sbir"]
    p3 = a[a.award_class == "sbir_phase3"].sort_values("date")
    g = lambda d: d.groupby("firm_id_sbir")
    first_p3 = p3.drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
    out = pd.DataFrame(index=f.index)
    out["fed_has_identifier"] = f.uei_all.fillna("").ne("") | f.duns_all.fillna("").ne("")
    out["fed_any_award_found"] = g(a).size().reindex(out.index).fillna(0) > 0
    out["fed_contracts_post_n"] = g(ns[ns.award_group == "contract"]).size()
    out["fed_contracts_post_amount"] = g(ns[ns.award_group == "contract"]).amount.sum(min_count=1)
    out["fed_grants_post_n"] = g(ns[ns.award_group == "grant"]).size()
    out["fed_grants_post_amount"] = g(ns[ns.award_group == "grant"]).amount.sum(min_count=1)
    pre = a[~a.post & (a.award_class == "non_sbir")]
    out["fed_contracts_pre_n"] = g(pre[pre.award_group == "contract"]).size()
    out["fed_contracts_pre_amount"] = g(pre[pre.award_group == "contract"]).amount.sum(min_count=1)
    out["fed_grants_pre_n"] = g(pre[pre.award_group == "grant"]).size()
    out["has_federal_followon"] = g(post[post.award_class != "sbir_program"]).size() > 0
    out["fed_n_agencies_post"] = g(post[post.award_class != "sbir_program"]).awarding_agency.nunique()
    out["sbir_phase3"] = g(p3).size() > 0
    out["phase3_n"] = g(p3).size()
    out["phase3_amount"] = g(p3).amount.sum(min_count=1)
    out["phase3_first_date"] = first_p3.date
    out["phase3_first_agency"] = first_p3.awarding_agency
    out["phase3_source"] = g(p3).phase3_source.agg(lambda s: "fpds_research_code" if (s == "fpds_research_code").any() else "description")
    out["years_to_phase3"] = (out.phase3_first_date - f.first_award_date).dt.days / 365.25
    out["fed_window_covered"] = f.first_award_date >= USASPENDING_COVERAGE_START
    for c in ("fed_contracts_post_n", "fed_grants_post_n", "phase3_n", "fed_contracts_pre_n", "fed_grants_pre_n"):
        out[c] = out[c].fillna(0).astype(int)
    for c in ("has_federal_followon", "sbir_phase3"):
        out[c] = out[c].astype("boolean").fillna(False)
    return out


def tidy(raw: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    c = lambda *k: sio.col(raw, *k)
    a = pd.DataFrame({
        "award_id": c("Award ID"), "generated_internal_id": c("generated_internal_id"), "award_group": c("award_group"),
        "recipient_name": c("Recipient Name"), "recipient_uei": c("Recipient UEI").str.upper(),
        "recipient_duns": c("Recipient DUNS Number").str.replace(r"\D", "", regex=True).str.zfill(9),
        "start_date": sio.to_date(c("Start Date")), "end_date": sio.to_date(c("End Date")),
        "base_obligation_date": sio.to_date(c("Base Obligation Date")),
        "amount": sio.to_num(c("Award Amount")), "awarding_agency": c("Awarding Agency"),
        "awarding_sub_agency": c("Awarding Sub Agency"), "funding_agency": c("Funding Agency"),
        "award_type": c("Contract Award Type").fillna(c("Award Type")), "naics": c("NAICS"), "psc": c("PSC"),
        "cfda": c("CFDA Number"), "description": c("Description"),
    }).drop_duplicates(["generated_internal_id"])
    a["date"] = a.base_obligation_date.fillna(a.start_date)
    by_uei = a.merge(ids[ids.kind == "uei"][["id", "firm_id_sbir"]], left_on="recipient_uei", right_on="id")
    by_duns = a.merge(ids[ids.kind == "duns"][["id", "firm_id_sbir"]], left_on="recipient_duns", right_on="id")
    m = pd.concat([by_uei.assign(matched_on="uei"), by_duns.assign(matched_on="duns")])
    m = m.sort_values("matched_on", ascending=False).drop_duplicates(["generated_internal_id", "firm_id_sbir"])
    # an award matched to two firms (UEI of one, DUNS of another) is dropped
    m = m[m.groupby("generated_internal_id").firm_id_sbir.transform("nunique") == 1]
    return m.drop(columns="id")


def run(paths: Paths, contract_details: bool = False, workers: int = WORKERS) -> None:
    web = sio.session(USER_AGENT)
    web.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    sb = pd.read_parquet(paths.processed / "sbir_awards.parquet",
                         columns=["award_id", "firm_id_sbir", "contract", "award_date", "award_year", "uei", "duns"])
    ids = _identifiers(paths)
    raw = fetch(paths, web, ids.id.tolist(), "search", workers)
    if raw.empty:
        log.warning("no USAspending results; federal fields left empty")
        return
    a = classify(tidy(raw, ids).assign(research_code=pd.NA), sb, firms)
    found_before = sbir_numbers_found(a, sb, ids.firm_id_sbir)

    # Second pass: identifiers recovered from SBIR award numbers the first pass missed.
    extra = expand_identifiers(paths, web, a, sb, ids, workers)
    if len(extra):
        ids = pd.concat([ids, extra], ignore_index=True)
        raw = pd.concat([raw, fetch(paths, web, extra.id.tolist(), "search_expanded", workers)], ignore_index=True)
    a = tidy(raw, ids)

    # Phase III work follows an earlier SBIR award, so the FPDS research code is needed only for
    # contracts that start after the firm's first award, carry a positive amount and are not
    # SBIR awards by number.
    pre = classify(a.assign(research_code=pd.NA), sb, firms)
    need = pre[(pre.award_group == "contract") & pre.post & pre.sbir_award_id.isna() & (pre.amount > 0)]
    codes, code_progress = research_codes(paths, web, need) if contract_details else (pd.DataFrame(), {})
    a = a.merge(codes, on="generated_internal_id", how="left") if len(codes) else a.assign(
        research_code=pd.NA, research_description=pd.NA)
    a = classify(a, sb, firms)
    # Phase III work follows an SBIR award, so a Phase III contract dated before the firm's first
    # award points to a misread description and is not counted as Phase III.
    early = (a.award_class == "sbir_phase3") & ~a.post
    a.loc[early, "award_class"] = "non_sbir"
    a.loc[early, "phase3_source"] = None
    a["recipient_name_matches"] = recipient_matches_firm(a.firm_id_sbir, a.recipient_name, firm_name_keys(paths))
    # A firm's award counts toward its outcomes only when the recipient is the firm itself.
    fields = firm_fields(a[a.recipient_name_matches], firms)

    keep = ["firm_id_sbir", "award_id", "generated_internal_id", "award_group", "award_class", "phase3_source",
            "sbir_award_id", "matched_on", "recipient_name", "recipient_uei", "recipient_duns", "date", "start_date",
            "end_date", "amount", "awarding_agency", "awarding_sub_agency", "funding_agency", "award_type", "naics",
            "psc", "cfda", "research_code", "research_description", "description", "post", "recipient_name_matches"]
    a[keep].to_parquet(paths.processed / "federal_awards.parquet", index=False)
    fields.reset_index().to_parquet(paths.processed / "firm_federal.parquet", index=False)

    contracts_needing = int(len(need))
    stats = {
        "identifiers_searched": int(ids.id.nunique()), "identifiers_added_from_award_numbers": int(extra.id.nunique()),
        "firms_with_identifier": int(ids.firm_id_sbir.nunique()),
        "awards_matched": int(len(a)), "firms_with_any_award": int(fields.fed_any_award_found.sum()),
        "awards_recipient_name_differs": int((~a.recipient_name_matches).sum()),
        "amount_recipient_name_differs_billion": round(float(a.loc[~a.recipient_name_matches, "amount"].sum()) / 1e9, 2),
        "amount_kept_billion": round(float(a.loc[a.recipient_name_matches, "amount"].sum()) / 1e9, 2),
        "award_class": a.award_class.value_counts().to_dict(),
        "phase3_source": a.phase3_source.value_counts().to_dict(),
        "phase3_candidate_contracts": contracts_needing,
        "research_code_downloads": code_progress,
        "phase3_candidates_with_research_code": int(a[a.generated_internal_id.isin(need.generated_internal_id)]
                                                    .research_code.notna().sum()),
        "firms_sbir_phase3": int(fields.sbir_phase3.sum()),
        "firms_federal_followon": int(fields.has_federal_followon.sum()),
        "sbir_gov_awards_since_fy2008_found_first_pass": found_before,
        "sbir_gov_awards_since_fy2008_found": sbir_numbers_found(a, sb, ids.firm_id_sbir),
    }
    (paths.audit / "federal.json").write_text(json.dumps(stats, indent=2, default=str))
    if code_progress and not code_progress.get("complete"):
        log.warning("research codes are incomplete (%s of %s PIIDs); Phase III falls back to descriptions for the rest. "
                    "Rerun the federal step to finish.", f"{code_progress['piids_downloaded']:,}", f"{code_progress['piids_requested']:,}")
    log.info("federal: %s awards for %s firms; SBIR.gov numbers found %s; Phase III firms %s", f"{len(a):,}",
             f"{stats['firms_with_any_award']:,}", stats["sbir_gov_awards_since_fy2008_found"], f"{stats['firms_sbir_phase3']:,}")
