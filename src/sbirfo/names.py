"""Firm-name, place-name and ZIP normalization used for every match in the pipeline."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .config import MIN_LOOSE_KEY

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "ltd", "limited",
    "llc", "lc", "llp", "lllp", "lp", "pllc", "pc", "plc", "pa", "gmbh", "ag", "sa", "nv",
    "bv", "ab", "as", "oy", "srl", "spa", "pty", "pte", "kk", "sas", "sarl", "ulc",
}
GENERIC_WORDS = {
    "technologies", "technology", "tech", "systems", "system", "solutions", "labs", "lab",
    "laboratories", "laboratory", "sciences", "science", "international", "intl", "group",
    "holdings", "holding", "enterprises", "enterprise", "industries", "innovations",
    "innovation", "research", "associates", "services", "global", "usa", "us", "america",
    "and", "the", "of", "company", "corporation",
}
_COLLAPSE = [(re.compile(r"\bp l l c\b"), "pllc"), (re.compile(r"\bl l c\b"), "llc"),
             (re.compile(r"\bl l p\b"), "llp"), (re.compile(r"\bl p\b"), "lp"),
             (re.compile(r"\bp c\b"), "pc")]
_DBA = re.compile(r"^(.+?)\s+(d b a|dba|doing business as|f k a|fka|formerly known as|a k a|aka)\b.*$")
_STATE_CORP = re.compile(r"^(.+?)\s+an?\s+[a-z ]{2,30}?\s+"
                         r"(corporation|company|limited liability company|limited partnership|llc|corp)$")
_NULLS = {"", "none", "null", "n a", "na", "nan", "unknown"}

NONFIRM = re.compile(r"\b(?:university|universit|college|regents|institute of technology|"
                     r"school of|hospital|medical center|foundation|government|department of|"
                     r"secretary of|national laborator|trustees)\b")

US_STATES = set("""AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO
MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR""".split())
STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR",
}


def name_key(name) -> str | None:
    """Strict key: lowercase ASCII, no punctuation, trailing legal forms and leading 'the' removed."""
    if not isinstance(name, str):
        return None
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _NULLS:
        return None
    s = _DBA.sub(r"\1", s)
    s = _STATE_CORP.sub(r"\1", s)
    for pattern, repl in _COLLAPSE:
        s = pattern.sub(repl, s)
    tokens = s.split()
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    return " ".join(tokens) or None


def loose_key(key: str | None) -> str | None:
    """Strict key without generic words or spaces; used for blocking and distinctiveness only."""
    if not key:
        return None
    return "".join(t for t in key.split() if t not in GENERIC_WORDS) or None


def add_keys(df: pd.DataFrame, col: str = "name_raw") -> pd.DataFrame:
    uniq = pd.Series(df[col].dropna().unique())
    strict = dict(zip(uniq, uniq.map(name_key)))
    keys = df[col].map(strict)
    loose = {k: loose_key(k) for k in keys.dropna().unique()}
    lk = keys.map(loose)
    return df.assign(name_key=keys, loose_key=lk, linkable=lk.fillna("").str.len() >= MIN_LOOSE_KEY)


def clean_state(s: pd.Series) -> pd.Series:
    """Two-letter US code from a code or full state name; anything else becomes missing."""
    s = s.astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    s = s.replace(STATE_NAMES)
    return s.where(s.isin(US_STATES))


def clean_zip(s: pd.Series) -> pd.Series:
    d = s.astype("string").str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    d = d.where(~d.str.len().isin([4, 8]), "0" + d)
    return d.str[:5].where(d.str.len() >= 5)


def clean_city(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.upper().str.replace(r"[^A-Z ]", " ", regex=True)
    s = s.str.replace(r"\bSAINT\b", "ST", regex=True).str.replace(r"\bFORT\b", "FT", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def modal(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    d = df.dropna(subset=[col])
    return (d.groupby([key, col], observed=True).size().reset_index(name="_n")
             .sort_values([key, "_n"], ascending=[True, False])
             .drop_duplicates(key).set_index(key)[col])


def star_edges(df: pd.DataFrame, by: list[str], rule: str, node: str = "node_id") -> pd.DataFrame:
    """Join each group's members to its first member: n-1 edges per group instead of n^2."""
    df = df.dropna(subset=by).drop_duplicates([node] + by)
    first = df.groupby(by, observed=True)[node].transform("first")
    keep = df[node] != first
    return pd.DataFrame({"node_a": first[keep].values, "node_b": df.loc[keep, node].values, "rule": rule})


def components(nodes: pd.Series, edges: pd.DataFrame) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    index = pd.Series(np.arange(len(nodes)), index=nodes.values)
    a, b = index[edges.node_a].values, index[edges.node_b].values
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    return connected_components(graph, directed=False)[1]
