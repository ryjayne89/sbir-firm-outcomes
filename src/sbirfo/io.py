"""Downloads, provenance records and table readers."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import logging
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

log = logging.getLogger(__name__)

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_MIN_GAP = {"www.sec.gov": 0.15}       # SEC fair-access limit is 10 requests per second
_last_hit: dict[str, float] = {}


class Manifest:
    """Append-only provenance log: source URL, retrieval time, size and SHA-256 of every raw file."""

    COLS = ["dataset", "file", "url", "retrieved_utc", "bytes", "sha256", "note"]

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.COLS)

    def has(self, file: Path) -> bool:
        return str(Path(file)) in set(pd.read_csv(self.path, usecols=["file"])["file"])

    def add(self, dataset: str, file: Path, url: str = "", note: str = "") -> None:
        file = Path(file)
        if self.has(file):
            return
        with open(self.path, "a", newline="") as fh:
            csv.writer(fh).writerow([dataset, str(file), url,
                                     dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                     file.stat().st_size, sha256(file), note])

    def table(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def session(user_agent: str = BROWSER_UA) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def sec_session(user_agent: str) -> requests.Session:
    if not user_agent or "@" not in user_agent:
        raise ValueError("The SEC requires a User-Agent with a name and contact email, "
                         "for example --sec-user-agent 'Jane Doe jane@university.edu'.")
    return session(user_agent)


def _throttle(url: str) -> None:
    host = re.sub(r"^https?://([^/]+).*$", r"\1", url)
    wait = _MIN_GAP.get(host, 0.05) - (time.time() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.time()


def download(sess: requests.Session, url: str, dest: Path, manifest: Manifest | None = None,
             dataset: str = "", retries: int = 3, timeout: int = 300) -> Path:
    """Stream a file to disk. Existing files are kept, so interrupted runs resume."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        if manifest is not None:
            manifest.add(dataset, dest, url, note="present on disk")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            _throttle(url)
            with sess.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.rename(dest)
            if manifest is not None:
                manifest.add(dataset, dest, url)
            return dest
        except FileNotFoundError:
            raise
        except Exception as exc:
            wait = 10 * (attempt + 1)
            log.warning("download %s failed (%s), retry in %ss", dest.name, type(exc).__name__, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}")


def list_links(sess: requests.Session, page: str, pattern: str) -> list[str]:
    """Absolute URLs on a page whose href matches pattern; empty if the page is unreachable."""
    try:
        _throttle(page)
        r = sess.get(page, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        log.warning("could not read %s (%s)", page, type(exc).__name__)
        return []
    return sorted({urljoin(page, h) for h in re.findall(r'href="([^"]+)"', r.text) if re.search(pattern, h)})


def zip_table(path: Path, table: str) -> pd.DataFrame | None:
    """Read the zip member whose stem equals table, as strings."""
    with zipfile.ZipFile(path) as zf:
        hits = [n for n in zf.namelist()
                if Path(n).stem.upper() == table.upper() and Path(n).suffix.lower() in (".tsv", ".txt", ".csv")]
        if not hits:
            return None
        raw = zf.read(hits[0])
    tab = not hits[0].lower().endswith(".csv")
    return pd.read_csv(io.BytesIO(raw), sep="\t" if tab else ",", dtype=str, on_bad_lines="skip",
                       encoding_errors="replace", low_memory=False,
                       quoting=csv.QUOTE_NONE if tab else csv.QUOTE_MINIMAL)


def quarter_of(name: str) -> str | None:
    m = re.search(r"(\d{4})q([1-4])", name.lower())
    return f"{m.group(1)}q{m.group(2)}" if m else None


def pick(cols, *candidates) -> str | None:
    """First candidate present in cols, ignoring case, spaces and punctuation."""
    norm = lambda c: re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
    lookup = {norm(c): c for c in cols}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    return None


def col(df: pd.DataFrame, *candidates) -> pd.Series:
    """Column by any of its spellings, or an empty string column when absent."""
    k = pick(df.columns, *candidates)
    return df[k] if k else pd.Series(pd.NA, index=df.index, dtype="string")


def to_date(s: pd.Series) -> pd.Series:
    """Dates as timezone-free timestamps. Some EDGAR fields carry a UTC time zone
    (2021-06-01T00:00:00.000Z) and others none, and the two cannot be compared."""
    return pd.to_datetime(s, errors="coerce", format="mixed", utc=True).dt.tz_convert(None)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce")


def same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise equality that treats missing values as unequal."""
    return (a.notna() & (a == b)).fillna(False).astype(bool)
