"""Patents, patent applications and patent transfers from USPTO data.

Sources (data/raw/patents, or folders passed with --patents-dir)

* PatentsView granted-patent tables (g_*): patents, application filing dates, disambiguated
  assignees and their locations, CPC and WIPO technology classes, inventors, U.S. patent
  citations, and government-interest statements with the contract numbers they cite.
* PatentsView pre-grant publication tables (pg_*): published applications and their assignees.
* USPTO Patent Assignment Dataset (assignment, assignor, assignee, documentid and
  assignment_conveyance CSVs): recorded transfers, mergers and security interests.

Two links are made. Disambiguated assignees are linked to SBIR firms with the same name and
location tiers as the SEC linkage (sbirfo.link), so a patent is the firm's when a
high-confidence assignee owns it. Separately, a government-interest statement that cites the
contract or grant number of an SBIR.gov award ties the patent to that award, whoever owns it.
"""
from __future__ import annotations

import hashlib
import json
import os
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

from . import io as sio
from . import link
from . import names as nm
from .config import FWD_CITE_YEARS, PGPUB_COVERAGE_START, Paths, uspto_release
from .federal import SBIR_TEXT, norm_award_id

log = logging.getLogger(__name__)

PV_GRANT = "https://s3.amazonaws.com/data.patentsview.org/download/{t}.tsv.zip"
PV_PGPUB = "https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/{t}.tsv.zip"
PV_HOME = "https://data.uspto.gov/bulkdata (Bulk Dataset Directory)"
ASSIGNMENT_HOME = "https://data.uspto.gov/bulkdata (Bulk Dataset Directory, ECORSEXC)"

TABLES = {
    "g_patent": ["patent_id", "patent_type", "patent_date", "num_claims", "withdrawn"],
    "g_application": ["patent_id", "application_id", "filing_date"],
    "g_assignee_disambiguated": ["patent_id", "assignee_id", "disambig_assignee_organization", "assignee_type", "location_id"],
    "g_location_disambiguated": ["location_id", "disambig_city", "disambig_state", "disambig_country"],
    "g_cpc_current": ["patent_id", "cpc_sequence", "cpc_section", "cpc_subclass", "cpc_type"],
    "g_wipo_technology": ["patent_id", "wipo_field_sequence", "wipo_sector_title", "wipo_field_title"],
    "g_inventor_disambiguated": ["patent_id", "inventor_id"],
    "g_us_patent_citation": ["patent_id", "citation_patent_id"],
    "g_gov_interest": ["patent_id", "gi_statement"],
    "g_gov_interest_contracts": ["patent_id", "contract_award_number"],
    "g_gov_interest_org": ["patent_id", "fedagency_name", "level_one"],
    "pg_published_application": ["pgpub_id", "application_id", "filing_date", "published_date"],
    "pg_assignee_disambiguated": ["pgpub_id", "assignee_id", "disambig_assignee_organization", "location_id"],
    "pg_location_disambiguated": ["location_id", "disambig_city", "disambig_state", "disambig_country"],
}
REQUIRED = ["g_patent", "g_application", "g_assignee_disambiguated", "g_location_disambiguated"]
LOCATION_SHARE = 0.25     # share of an assignee's documents a location needs to be used for matching
MIN_LOOSE_PATENT = 8      # shortest name core for a loose match to a patent assignee

ASSIGNMENT_FILES = {
    "assignment": ["rf_id", "record_dt", "convey_text"],
    "assignment_conveyance": ["rf_id", "convey_ty", "employer_assign"],
    "assignor": ["rf_id", "or_name", "exec_dt"],
    "assignee": ["rf_id", "ee_name", "ee_city", "ee_state", "ee_country"],
    "documentid": ["rf_id", "grant_doc_num", "appno_doc_num"],
}


class Source:
    """Finds a table on disk in any of the usual PatentsView or USPTO layouts."""

    def __init__(self, paths: Paths, dirs: list[Path] | None, download: bool = True):
        self.paths = paths
        self.dirs = [paths.raw / "patents"] + [Path(d) for d in dirs or []]
        self.cache = paths.interim / "patents"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.download = download
        self.manifest = sio.Manifest(paths.manifest)
        self.web = sio.session()

    SUFFIXES = (".parquet", ".tsv", ".tsv.zip", ".csv", ".csv.zip", ".dta", ".txt", ".zip")

    def _index(self) -> dict[str, list[Path]]:
        """Every file under the search folders, by lower-case name, so tables are found at any depth
        (Drive often unzips g_patent.tsv.zip into a folder g_patent.tsv holding g_patent.tsv)."""
        if getattr(self, "_files", None) is None:
            self._files = {}
            for d in self.dirs:
                if d.exists():
                    for root, dirs, files in os.walk(d):
                        dirs[:] = [x for x in dirs if not x.startswith(".")]
                        for name in files:
                            self._files.setdefault(name.lower(), []).append(Path(root) / name)
        return self._files

    def listing(self, n: int = 40) -> list[str]:
        return sorted(str(p.relative_to(p.parents[min(2, len(p.parents) - 1)])) for v in self._index().values() for p in v)[:n]

    def find(self, t: str) -> Path | None:
        idx = self._index()
        for suffix in self.SUFFIXES:
            hits = sorted(idx.get(f"{t}{suffix}".lower(), []), key=lambda p: len(p.parts))
            if hits:
                p = hits[0]
                if not self.manifest.has(p):
                    self.manifest.add("uspto_" + t, p, PV_HOME if t in TABLES else ASSIGNMENT_HOME,
                                      note="manual download; " + uspto_release(t))
                return p
        if self.download and t in TABLES:
            url = (PV_PGPUB if t.startswith("pg_") else PV_GRANT).format(t=t)
            try:
                return sio.download(self.web, url, self.dirs[0] / f"{t}.tsv.zip", self.manifest, "patentsview_" + t,
                                    retries=2, timeout=3600)
            except Exception as exc:
                log.warning("could not download %s (%s)", t, type(exc).__name__)
        return None

    def read(self, t: str, cols: list[str], keep: tuple | None = None) -> pd.DataFrame | None:
        """Selected columns of a table, optionally only rows whose keep[0] is in keep[1], after
        passing the column through keep[2] when given (a normalizing function). Cached."""
        f = self.find(t)
        if f is None:
            return None
        sig = hashlib.sha1(("|".join(cols) + (keep[0] + "|".join(sorted(keep[1])) if keep else "")
                            + (keep[2].__name__ if keep and len(keep) > 2 else "")).encode()).hexdigest()[:10]
        norm = keep[2] if keep and len(keep) > 2 else (lambda s: s)
        out = self.cache / f"{t}_{sig}.parquet"
        if out.exists() and out.stat().st_mtime > f.stat().st_mtime:
            return pd.read_parquet(out)
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            names = pq.ParquetFile(f).schema_arrow.names
            df = pd.read_parquet(f, columns=[c for c in cols if c in names]).astype("string")
            chunks = [df]
        elif f.suffix.lower() == ".dta":
            # The Patent Assignment Dataset ships as Stata files; numeric IDs are read as whole
            # numbers so they compare equal to the same IDs read as text elsewhere.
            with pd.read_stata(f, iterator=True) as r:
                header = list(r.variable_labels())
            use = [c for c in cols if c in header]
            chunks = []
            with _TolerantStataReader(f, columns=use, chunksize=1_000_000, convert_categoricals=False,
                                      preserve_dtypes=False) as reader:
                for chunk in reader:
                    chunk = chunk.apply(_as_text)
                    if keep:
                        chunk = chunk[norm(chunk[keep[0]]).isin(keep[1])]
                    chunks.append(chunk)
        else:
            sep = "," if ".csv" in f.name else "\t"
            header = pd.read_csv(f, sep=sep, nrows=0).columns
            use = [c for c in cols if c in header]
            reader = pd.read_csv(f, sep=sep, usecols=use, dtype=str, chunksize=2_000_000, on_bad_lines="skip",
                                 encoding_errors="replace", low_memory=False)
            chunks = []
            for chunk in reader:
                if keep:
                    chunk = chunk[norm(chunk[keep[0]]).isin(keep[1])]
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
        if keep and f.suffix == ".parquet":
            df = df[norm(df[keep[0]]).isin(keep[1])]
        for c in cols:
            if c not in df:
                df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        df = df[cols].astype("string")
        df.to_parquet(out, index=False)
        log.info("%s: %s rows kept", t, f"{len(df):,}")
        return df


class _TolerantStataReader(StataReader):
    """Stata reader that survives long strings (strL) holding invalid UTF-8.

    Some Patent Assignment Dataset files contain long text fields cut in the middle of a
    multi-byte character, which pandas refuses to decode. When that happens the long strings
    of that file are read as Latin-1 instead. The pipeline uses only short ID and name columns
    from these files, so the affected text is not used."""

    def _read_strls(self) -> None:
        enc = self._encoding
        try:
            super()._read_strls()
        except UnicodeDecodeError:
            log.info("%s: long text fields hold invalid UTF-8; reading them as Latin-1", Path(str(self._path_or_buf.name)).name
                     if hasattr(self._path_or_buf, "name") else "Stata file")
            self._encoding = "latin-1"
            try:
                super()._read_strls()
            finally:
                self._encoding = enc


def _as_text(s: pd.Series) -> pd.Series:
    """A column as text, with whole-number floats written without a decimal part."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d").astype("string")
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if (x.dropna() % 1 == 0).all():
            return x.astype("Int64").astype("string")
        return x.astype("string")
    return s.astype("string").str.strip().replace("", pd.NA)


def patent_number(s: pd.Series) -> pd.Series:
    """USPTO document numbers in PatentsView form: utility numbers without leading zeros, D/RE/PP kept."""
    x = s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    m = x.str.extract(r"^(D|RE|PP|H|T)?0*(\d+)$")
    return (m[0].fillna("") + m[1]).where(m[1].notna())


def assignee_names(src: Source, prefix: str, table: str, loc_table: str, id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = src.read(table, TABLES[table])
    if a is None:
        return pd.DataFrame(columns=link.SEC_COLS), pd.DataFrame(columns=["doc_id", "entity"])
    loc = src.read(loc_table, TABLES[loc_table])
    if loc is None and loc_table.startswith("pg_"):
        loc = src.read("g_location_disambiguated", TABLES["g_location_disambiguated"])
    a = a[a.disambig_assignee_organization.notna() & a.assignee_id.notna()]
    a = a.merge(loc, on="location_id", how="left") if loc is not None else a.assign(
        disambig_city=pd.NA, disambig_state=pd.NA, disambig_country=pd.NA)
    a["entity"] = prefix + ":" + a.assignee_id
    docs = a[[id_col, "entity"]].rename(columns={id_col: "doc_id"}).drop_duplicates()
    # A disambiguated assignee pools every patent filed under one name, so a large company can
    # carry a stray location (Broadcom Corporation once listed in Mahwah, NJ). Only locations that
    # hold at least LOCATION_SHARE of the entity's documents, or its most common one, are used
    # for matching, so a small firm is not linked to a large namesake through a stray address.
    cnt = a.groupby(["entity", "location_id"], dropna=False)[id_col].nunique()
    share = cnt / cnt.groupby(level=0).transform("sum")
    main = share[(share >= LOCATION_SHARE) | (share == share.groupby(level=0).transform("max"))].index
    a = a.set_index(["entity", "location_id"]).loc[lambda d: d.index.isin(main)].reset_index()
    us = a.disambig_country.fillna("US").eq("US")
    n = pd.DataFrame({"cik": a.entity, "name_raw": a.disambig_assignee_organization,
                      "city": nm.clean_city(a.disambig_city), "state": nm.clean_state(a.disambig_state.where(us)),
                      "zip5": pd.Series(pd.NA, index=a.index, dtype="string"), "sec_source": prefix})
    n = nm.add_keys(n.drop_duplicates(["cik", "name_raw", "city", "state"]))
    return n[n.name_key.notna()], docs


def link_assignees(paths: Paths, src: Source) -> tuple[pd.DataFrame, pd.DataFrame]:
    sb = link.sbir_names(paths)
    g_names, g_docs = assignee_names(src, "g", "g_assignee_disambiguated", "g_location_disambiguated", "patent_id")
    p_names, p_docs = assignee_names(src, "pg", "pg_assignee_disambiguated", "pg_location_disambiguated", "pgpub_id")
    names = pd.concat([g_names, p_names], ignore_index=True)
    cand = link.assign_tiers(link.candidates(sb, names), sb, names)
    # A shared name core as short as "sandia" ties unrelated organizations (Sandia Systems and
    # Sandia Corporation), so generic-word-insensitive matches need a longer core here.
    core = cand.get("loose_key", pd.Series(pd.NA, index=cand.index)).fillna(cand.loose_key_sb)
    short = (cand.method == "loose") & (core.fillna("").str.len() < MIN_LOOSE_PATENT)
    cand = cand[~short]
    best = (cand.sort_values(["tier", "score"], ascending=[True, False]).drop_duplicates(["firm_id_sbir", "cik"])
            .rename(columns={"cik": "entity"}))
    best["confidence"] = np.where(best.tier <= 4, "high", "medium")
    # an assignee entity claimed by several SBIR firms at its best tier stays with none of them
    top = best.groupby("entity").tier.transform("min")
    best = best[best.tier == top]
    best = best[best.groupby("entity").firm_id_sbir.transform("nunique") == 1]
    log.info("assignee links: %s entities, %s firms (high %s)", f"{best.entity.nunique():,}",
             f"{best.firm_id_sbir.nunique():,}", f"{best[best.confidence == 'high'].firm_id_sbir.nunique():,}")
    return best, pd.concat([g_docs.assign(doc_type="grant"), p_docs.assign(doc_type="pgpub")], ignore_index=True)


def gov_interest_links(src: Source, sb_awards: pd.DataFrame) -> pd.DataFrame:
    gc = src.read("g_gov_interest_contracts", TABLES["g_gov_interest_contracts"])
    if gc is None:
        return pd.DataFrame(columns=["patent_id", "firm_id_sbir", "sbir_award_id"])
    own = sb_awards.assign(key=norm_award_id(sb_awards.contract)).dropna(subset=["key"])
    own = own[own.groupby("key").firm_id_sbir.transform("nunique") == 1]
    # several numbers are often written in one field, separated by commas, "and" or semicolons
    parts = gc.assign(num=gc.contract_award_number.str.split(r"[,;/]|\band\b", regex=True)).explode("num")
    parts["key"] = norm_award_id(parts.num.str.strip())
    m = parts.merge(own[["key", "firm_id_sbir", "award_id"]], on="key")
    # NIH grants are often cited by institute code and serial number alone (GM123456)
    own["core"] = own.key.str.extract(r"^[RUFSK]\d{2}([A-Z]{2}\d{6})$", expand=False)
    own_core = own.dropna(subset=["core"])
    own_core = own_core[own_core.groupby("core").firm_id_sbir.transform("nunique") == 1]
    parts["core"] = parts.key.str.extract(r"([A-Z]{2}\d{6})$", expand=False)
    m2 = parts.dropna(subset=["core"]).merge(own_core[["core", "firm_id_sbir", "award_id"]], on="core")
    m = pd.concat([m, m2]).rename(columns={"award_id": "sbir_award_id"})
    return m[["patent_id", "firm_id_sbir", "sbir_award_id"]].drop_duplicates(["patent_id", "firm_id_sbir"])


def patent_attributes(src: Source, ids: set[str]) -> pd.DataFrame:
    keep = ("patent_id", ids)
    pat = src.read("g_patent", TABLES["g_patent"], keep)
    app = src.read("g_application", TABLES["g_application"], keep).drop_duplicates("patent_id")
    p = pat.merge(app[["patent_id", "filing_date"]], on="patent_id", how="left")
    p["grant_date"], p["filing_date"] = sio.to_date(p.patent_date), sio.to_date(p.filing_date)
    p["num_claims"] = sio.to_num(p.num_claims)
    p = p[p.withdrawn.fillna("0").isin(["0", "False", "false", ""])]

    cpc = src.read("g_cpc_current", TABLES["g_cpc_current"], keep)
    if cpc is not None and len(cpc):
        cpc = cpc.assign(seq=sio.to_num(cpc.cpc_sequence)).sort_values(["patent_id", "seq"])
        p = p.merge(cpc.drop_duplicates("patent_id")[["patent_id", "cpc_section", "cpc_subclass"]], on="patent_id", how="left")
        p = p.merge(cpc.groupby("patent_id").cpc_subclass.agg(lambda s: "|".join(sorted(set(s.dropna()))))
                    .rename("cpc_subclasses_all"), on="patent_id", how="left")
    wipo = src.read("g_wipo_technology", TABLES["g_wipo_technology"], keep)
    if wipo is not None and len(wipo):
        wipo = wipo.assign(seq=sio.to_num(wipo.wipo_field_sequence)).sort_values(["patent_id", "seq"])
        p = p.merge(wipo.drop_duplicates("patent_id")[["patent_id", "wipo_sector_title", "wipo_field_title"]],
                    on="patent_id", how="left")
    inv = src.read("g_inventor_disambiguated", TABLES["g_inventor_disambiguated"], keep)
    if inv is not None:
        p = p.merge(inv.groupby("patent_id").inventor_id.nunique().rename("n_inventors"), on="patent_id", how="left")
    gi = src.read("g_gov_interest", TABLES["g_gov_interest"], keep)
    if gi is not None:
        gi = gi.groupby("patent_id").gi_statement.agg(lambda s: " ".join(s.dropna()))
        p["gov_interest"] = p.patent_id.isin(gi.index)
        p["gov_interest_cites_sbir"] = p.patent_id.map(gi).fillna("").str.contains(SBIR_TEXT)
    go = src.read("g_gov_interest_org", TABLES["g_gov_interest_org"], keep)
    if go is not None and len(go):
        p = p.merge(go.groupby("patent_id").fedagency_name.agg(lambda s: "|".join(sorted(set(s.dropna()))))
                    .rename("gov_interest_agencies"), on="patent_id", how="left")

    cit = src.read("g_us_patent_citation", TABLES["g_us_patent_citation"], ("citation_patent_id", ids))
    if cit is not None:
        citing = src.read("g_patent", ["patent_id", "patent_date"], ("patent_id", set(cit.patent_id.dropna())))
        cit = cit.merge(citing.rename(columns={"patent_id": "citing_id", "patent_date": "citing_date"}),
                        left_on="patent_id", right_on="citing_id", how="left")
        cit["citing_date"] = sio.to_date(cit.citing_date)
        cit = cit.merge(p[["patent_id", "grant_date"]].rename(columns={"patent_id": "citation_patent_id"}),
                        on="citation_patent_id")
        within = cit.citing_date <= cit.grant_date + pd.DateOffset(years=FWD_CITE_YEARS)
        p["fwd_cites_total"] = p.patent_id.map(cit.groupby("citation_patent_id").size()).fillna(0).astype(int)
        p["fwd_cites_5y"] = p.patent_id.map(cit[within].groupby("citation_patent_id").size()).fillna(0)
        everything = src.read("g_patent", ["patent_date"])
        data_end = sio.to_date(everything.patent_date).max()
        p.loc[p.grant_date > data_end - pd.DateOffset(years=FWD_CITE_YEARS), "fwd_cites_5y"] = np.nan
    return p.drop(columns=["patent_date", "withdrawn"])


def assignments(src: Source, firm_patents: pd.DataFrame, paths: Paths) -> pd.DataFrame | None:
    """Recorded transfers out of SBIR firms of patents they own (USPTO Patent Assignment Dataset)."""
    doc = src.find("documentid")
    if doc is None:
        log.warning("Patent Assignment Dataset not found; patent transfer fields left empty (%s)", ASSIGNMENT_HOME)
        return None
    nums = set(firm_patents.patent_id)
    # documentid has a row for every recorded document, so only rows for the firms' patents are kept
    d = src.read("documentid", ASSIGNMENT_FILES["documentid"], ("grant_doc_num", nums, patent_number))
    d["patent_id"] = patent_number(d.grant_doc_num)
    d = d[d.patent_id.isin(nums)]
    rf = set(d.rf_id)
    tabs = {t: src.read(t, ASSIGNMENT_FILES[t], ("rf_id", rf)) for t in ("assignment", "assignment_conveyance", "assignor", "assignee")}
    if any(v is None for v in tabs.values()):
        log.warning("Patent Assignment Dataset incomplete; transfer fields left empty")
        return None
    aw = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["firm_id_sbir", "company_name"])
    keys = nm.add_keys(aw.rename(columns={"company_name": "name_raw"}))
    own_keys = keys.groupby("firm_id_sbir").loose_key.agg(set)
    own_strict = keys.groupby("firm_id_sbir").name_key.agg(set)

    x = d[["rf_id", "patent_id"]].merge(firm_patents[["patent_id", "firm_id_sbir"]], on="patent_id")
    x = x.merge(tabs["assignment_conveyance"][["rf_id", "convey_ty"]], on="rf_id", how="left")
    x = x.merge(tabs["assignment"][["rf_id", "record_dt"]], on="rf_id", how="left")
    ors = nm.add_keys(tabs["assignor"].rename(columns={"or_name": "name_raw"}))
    ees = nm.add_keys(tabs["assignee"].rename(columns={"ee_name": "name_raw"}))
    is_firm = lambda df: [(k in own_strict.get(f, set())) or (lk in own_keys.get(f, set()) and isinstance(lk, str) and len(lk) >= 4)
                          for f, k, lk in zip(df.firm_id_sbir, df.name_key, df.loose_key)]
    xo = x.merge(ors[["rf_id", "name_key", "loose_key", "exec_dt"]], on="rf_id")
    xo["firm_is_assignor"] = is_firm(xo)
    xo = (xo.sort_values("firm_is_assignor", ascending=False).drop_duplicates(["rf_id", "patent_id", "firm_id_sbir"]))
    xe = x[["rf_id", "firm_id_sbir"]].drop_duplicates().merge(ees[["rf_id", "name_raw", "name_key", "loose_key", "ee_state", "ee_country"]], on="rf_id")
    xe["firm_is_assignee"] = is_firm(xe)
    buyer = xe[~xe.firm_is_assignee].drop_duplicates(["rf_id", "firm_id_sbir"]).set_index(["rf_id", "firm_id_sbir"]).name_raw
    out = xo[xo.firm_is_assignor].copy()
    out["date"] = sio.to_date(out.exec_dt).fillna(sio.to_date(out.record_dt))
    out["counterparty"] = [buyer.get((r, f)) for r, f in zip(out.rf_id, out.firm_id_sbir)]
    t = out.convey_ty.astype("string").str.lower().fillna("")
    out["transfer_type"] = np.select(
        [t.str.contains("merger"), t.str.contains("security"), t.str.contains("release"),
         t.str.contains("namechg|name change"), t.str.contains("assign") & out.counterparty.notna(), t.str.contains("license")],
        ["merger", "security_interest", "security_release", "name_change", "transfer_out", "license"], "other")
    return out[["firm_id_sbir", "patent_id", "rf_id", "date", "transfer_type", "counterparty", "convey_ty"]]


def _flag(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c].astype("boolean").fillna(False).astype(bool) if c in df else pd.Series(False, index=df.index)


def firm_fields(fp: pd.DataFrame, apps: pd.DataFrame, gi: pd.DataFrame, tr: pd.DataFrame | None,
                firms: pd.DataFrame) -> pd.DataFrame:
    f = firms.set_index("firm_id_sbir")
    t0 = fp.firm_id_sbir.map(f.first_award_date)
    fp = fp.assign(post=fp.filing_date > t0, pre=fp.filing_date <= t0,
                   within5=(fp.filing_date > t0) & (fp.filing_date <= t0 + pd.DateOffset(years=5)))
    g = lambda d: d.groupby("firm_id_sbir")
    post = fp[fp.post]
    out = pd.DataFrame(index=f.index)
    out["patents_n"] = g(fp).size()
    out["patents_pre_n"] = g(fp[fp.pre]).size()
    out["patents_post_n"] = g(post).size()
    out["patents_post_5y_n"] = g(fp[fp.within5]).size()
    out["patent_first_filing_post_date"] = g(post).filing_date.min()
    out["years_to_first_patent"] = (out.patent_first_filing_post_date - f.first_award_date).dt.days / 365.25
    out["patents_gov_interest_post_n"] = g(post[_flag(post, "gov_interest")]).size()
    sbir_cited = post.patent_id.isin(gi.patent_id) | _flag(post, "gov_interest_cites_sbir")
    out["patents_sbir_funded_post_n"] = g(post[sbir_cited]).size()
    out["patents_citing_firm_sbir_award_n"] = g(gi).patent_id.nunique()
    if "fwd_cites_5y" in fp:
        pc = post.dropna(subset=["fwd_cites_5y"])
        out["fwd_cites_5y_post"] = g(pc).fwd_cites_5y.sum()
        out["fwd_cites_5y_post_mean"] = g(pc).fwd_cites_5y.mean()
    out["patent_claims_post_mean"] = g(post).num_claims.mean()
    if "n_inventors" in fp:
        out["patent_inventors_post_mean"] = g(post).n_inventors.mean()
    for c, name in (("cpc_section", "patent_cpc_section_modal"), ("wipo_sector_title", "patent_wipo_sector_modal")):
        if c in fp:
            out[name] = g(fp.dropna(subset=[c])).agg(m=(c, lambda s: s.value_counts().index[0])).m
    if "cpc_subclass" in fp:
        out["patent_n_cpc_subclasses"] = g(fp).cpc_subclass.nunique()
    if len(apps):
        a0 = apps.firm_id_sbir.map(f.first_award_date)
        out["patent_apps_pre_n"] = g(apps[apps.filing_date <= a0]).size()
        out["patent_apps_post_n"] = g(apps[apps.filing_date > a0]).size()
    out["has_patent_post"] = out.patents_post_n.fillna(0) > 0
    out["patent_apps_window_covered"] = f.first_award_date >= PGPUB_COVERAGE_START

    if tr is not None and len(tr):
        tr = tr.assign(post=tr.date > tr.firm_id_sbir.map(f.first_award_date))
        to = tr[(tr.transfer_type == "transfer_out") & tr.post]
        out["patents_transferred_out_n"] = g(to).patent_id.nunique()
        out["patent_transfer_first_date"] = g(to).date.min()
        if len(to):
            top = to.groupby(["firm_id_sbir", "counterparty"]).patent_id.nunique().reset_index()
            top = top.sort_values("patent_id", ascending=False).drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
            out["patent_transfer_main_buyer"] = top.counterparty
            out["patent_transfer_main_buyer_share"] = top.patent_id / out.patents_n.reindex(top.index)
        else:
            out["patent_transfer_main_buyer"], out["patent_transfer_main_buyer_share"] = pd.NA, np.nan
        mg = tr[(tr.transfer_type == "merger") & tr.post].sort_values("date").drop_duplicates("firm_id_sbir").set_index("firm_id_sbir")
        out["patent_merger_date"], out["patent_merger_counterparty"] = mg.date, mg.counterparty
        si = tr[(tr.transfer_type == "security_interest") & tr.post]
        out["patent_security_interest"] = g(si).size() > 0
        out["patent_security_interest_first_date"] = g(si).date.min()
        share = out.patent_transfer_main_buyer_share.fillna(0)
        bulk = (share >= 0.5) & (g(to).patent_id.nunique().reindex(out.index).fillna(0) >= 2)
        out["private_acquisition_signal"] = out.patent_merger_date.notna() | bulk
        out["private_acquisition_signal_date"] = out.patent_merger_date.where(out.patent_merger_date.notna(),
                                                                               out.patent_transfer_first_date.where(bulk))
    for c in [c for c in out.columns if c.endswith("_n")]:
        out[c] = out[c].fillna(0).astype(int)
    for c in ("patent_security_interest", "private_acquisition_signal"):
        if c in out:
            out[c] = out[c].astype("boolean").fillna(False)
    return out


def run(paths: Paths, source_dirs: list[Path] | None = None, download: bool = True,
        link_confidence: tuple[str, ...] = ("high",)) -> None:
    src = Source(paths, source_dirs, download)
    missing = [t for t in REQUIRED if src.find(t) is None]
    if missing:
        log.warning("PatentsView tables missing (%s). Looked in %s, at any depth, for files named like "
                    "g_patent.tsv or g_patent.tsv.zip. Files found there: %s. Patent fields left empty.",
                    ", ".join(missing), ", ".join(str(d) for d in src.dirs), src.listing() or "none")
        return
    firms = pd.read_parquet(paths.processed / "sbir_firms.parquet")
    sb_awards = pd.read_parquet(paths.processed / "sbir_awards.parquet", columns=["award_id", "firm_id_sbir", "contract"])

    links, docs = link_assignees(paths, src)
    use = links[links.confidence.isin(link_confidence)]
    owned = docs.merge(use[["entity", "firm_id_sbir", "tier"]], on="entity")
    owned_g = (owned[owned.doc_type == "grant"].sort_values("tier").drop_duplicates(["doc_id", "firm_id_sbir"])
               .rename(columns={"doc_id": "patent_id", "tier": "assignee_tier"}))
    gi = gov_interest_links(src, sb_awards)

    ids = set(owned_g.patent_id) | set(gi.patent_id)
    attrs = patent_attributes(src, ids)
    fp = owned_g[["patent_id", "firm_id_sbir", "assignee_tier"]].merge(attrs, on="patent_id")
    fp = fp.merge(gi.rename(columns={"sbir_award_id": "cites_sbir_award_id"}), on=["patent_id", "firm_id_sbir"], how="left")

    pg = owned[owned.doc_type == "pgpub"].drop_duplicates(["doc_id", "firm_id_sbir"])
    pub = src.read("pg_published_application", TABLES["pg_published_application"], ("pgpub_id", set(pg.doc_id)))
    apps = pd.DataFrame(columns=["firm_id_sbir", "pgpub_id", "filing_date", "published_date"])
    if pub is not None and len(pg):
        apps = pg.rename(columns={"doc_id": "pgpub_id"})[["pgpub_id", "firm_id_sbir"]].merge(pub, on="pgpub_id")
        apps["filing_date"], apps["published_date"] = sio.to_date(apps.filing_date), sio.to_date(apps.published_date)

    tr = assignments(src, fp, paths)
    fields = firm_fields(fp, apps, gi, tr, firms)

    gi_all = gi.merge(attrs, on="patent_id", how="left").assign(owned_by_firm=lambda d: d.patent_id.isin(fp.patent_id))
    fp.to_parquet(paths.processed / "firm_patents.parquet", index=False)
    gi_all.to_parquet(paths.processed / "patent_sbir_award_citations.parquet", index=False)
    links.to_parquet(paths.processed / "link_sbir_assignee.parquet", index=False)
    apps.to_parquet(paths.interim / "patents" / "firm_pgpubs.parquet", index=False)
    if tr is not None:
        tr.to_parquet(paths.processed / "patent_transfers.parquet", index=False)
    fields.reset_index().to_parquet(paths.processed / "firm_patents_summary.parquet", index=False)

    stats = {
        "assignee_links_by_tier": {int(k): int(v) for k, v in links.tier.value_counts().sort_index().items()},
        "firms_with_patents": int((fields.patents_n > 0).sum()),
        "patents_owned": int(fp.patent_id.nunique()),
        "patents_citing_an_sbir_award": int(gi.patent_id.nunique()),
        "patents_citing_award_owned_by_that_firm": int(gi_all.owned_by_firm.sum()),
        "published_applications": int(len(apps)),
        "transfers": tr.transfer_type.value_counts().to_dict() if tr is not None else None,
        "firms_private_acquisition_signal": int(fields.get("private_acquisition_signal", pd.Series(dtype=bool)).sum()),
    }
    (paths.audit / "patents.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("patents: %s firms with %s patents; %s patents cite an SBIR award", f"{stats['firms_with_patents']:,}",
             f"{stats['patents_owned']:,}", f"{stats['patents_citing_an_sbir_award']:,}")
