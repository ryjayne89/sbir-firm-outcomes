"""Assemble the public release: CSV tables, codebook, provenance, checksums and Zenodo metadata."""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from . import __version__
from . import io as sio
from .config import PUBLIC_DATA_LICENSE, USPTO_RELEASES, Paths

log = logging.getLogger(__name__)

TABLES = {
    "firm_outcomes": "One row per SBIR/STTR firm with its award history, SEC and SBA outcomes and covariates.",
    "firm_events": "One row per dated event (SBIR awards, Form D offerings, Form C filings, EDGAR filings, SBA loans, federal contracts and grants, patent filings and grants, and patent transfers).",
    "link_sbir_sec": "Every SBIR firm to SEC CIK link with its tier, confidence and evidence.",
    "sbir_awards": "Cleaned SBIR/STTR awards with firm_id_sbir. Personal contact fields are excluded.",
    "federal_awards": "Every USAspending contract and grant matched to an SBIR firm by UEI or DUNS, classed as SBIR, Phase III or other.",
    "firm_patents": "Every granted patent owned by a linked assignee, with filing and grant dates, technology classes and citations.",
    "patent_sbir_award_citations": "Every granted patent whose government-interest statement cites an SBIR.gov award number.",
    "patent_transfers": "Recorded assignments, mergers and security interests with an SBIR firm as assignor.",
    "link_sbir_assignee": "Every SBIR firm to PatentsView assignee link with its tier and evidence.",
    "firm_year_panel": "One row per firm and year over the firm's observed life, before and after its first award, with every source by year and coverage flags.",
    "firm_financials_year": "Annual XBRL financials (revenue, R&D, income, assets, cash) of firms that file with the SEC.",
    "formd_offerings_firm": "Every Regulation D offering of a linked firm, before and after its first award, with amounts, investors and revenue range.",
    "zcta_characteristics_2000": "2000 Census population, education, income and home values by ZIP (ZCTA), the baseline before the ACS.",
    "zcta_characteristics": "ACS 5-year population, education, income, unemployment and home values by ZIP (ZCTA) and release year.",
}


def _validation_section(paths: Paths) -> list[str]:
    """README text on the hand-labeled check of the SEC linkage, with numbers read from the
    scored results, so the text always matches validation_results.csv."""
    f = paths.validation / "validation_results.csv"
    if not f.exists():
        return []
    r = pd.read_csv(f).set_index(["sample", "metric"])
    if ("random", "precision") not in r.index or ("random", "recall") not in r.index:
        return []
    pr, rc = r.loc[("random", "precision")], r.loc[("random", "recall")]
    pct = lambda x: f"{100 * x:.0f}%"
    n = int(r.loc[("random", "f1"), "n"]) if ("random", "f1") in r.index else None
    return [
        "## Validation of the SEC linkage", "",
        f"{n:,} firms drawn at random from all SBIR/STTR firms were checked by hand against EDGAR. For each, "
        "the labeler searched EDGAR by name and recorded the CIK of every filer that is the same company, or "
        "none. Because the firms were drawn from all firms, linked or not, the sample measures both kinds of error.", "",
        f"* Precision is {pct(pr.estimate)} ({int(pr.k)} of {int(pr.n)} predicted links correct; 95% interval "
        f"{pct(pr.ci_low)} to {pct(pr.ci_high)}).",
        f"* Recall is {pct(rc.estimate)} ({int(rc.k)} of {int(rc.n)} firms with an SEC filer linked to it; 95% "
        f"interval {pct(rc.ci_low)} to {pct(rc.ci_high)}).", "",
        "Notes recorded during labeling show what a correct link looks like when the names differ.", "",
        "* EDGAR stores a standardized name, so capitalization and legal forms differ from SBIR.gov "
        "(Nanoflex Power Corporation and NanoFlex Power Corp). These are not errors.",
        "* EDGAR keeps a company's CIK when it changes its name, so links hold across renamings "
        "(SynQuest to Viewlocity, Pearl Street Technologies LLC to Inc., Calysta BioSystems to Calysta, "
        "Biometrix to BioMetrix Co.).",
        "* Some firms file under a holding company's name (Equinox Corp as Equinox Holdings Inc.).",
        "* A revoked Exchange Act registration ends a company's reporting but not its identity, so the CIK "
        "still identifies the firm (Mammatech Corporation).",
        "* Both missed filers were early SEC registrants (Computer Technology Associates; Science & "
        "Engineering Associates, registered in 1984), so recall may be lower for firms active mainly in the "
        "1980s and 1990s.", "",
        "The tier rows in `validation_results.csv` use the labeled random firms only. The per-tier oversample "
        "and the rest of the random sample are to be labeled in a later version.", "",
    ]


def _data_readme(paths: Paths, version: str) -> str:
    stats = {}
    for name in ("sbir", "link", "outcomes", "federal", "sam", "patents", "places", "financials", "panel"):
        p = paths.audit / f"{name}.json"
        stats[name] = json.loads(p.read_text()) if p.exists() else {}
    o, l = stats.get("outcomes", {}), stats.get("link", {})
    fd = o.get("formd", {})
    miss = fd.get("offerings_post_amount_missing_share")
    miss = f"{100 * miss:.0f}%" if isinstance(miss, (int, float)) else "n/a"
    lines = [
        f"# SBIR/STTR Firm Outcomes Database, version {version}", "",
        "Firm-level panel linking every SBIR/STTR awardee (all agencies, 1983 onward) to SEC "
        "Regulation D, Regulation Crowdfunding and EDGAR filings, SBA 7(a) and 504 loans, federal "
        "contracts and grants including SBIR Phase III (USAspending), founding year (SAM.gov), USPTO "
        "patents, applications and patent transfers, and spatial and macroeconomic covariates. Built entirely from public sources with the code at "
        "the repository cited below.", "",
        "## Files", "",
    ]
    lines += [f"* `{t}.csv` holds {d[0].lower() + d[1:]}" for t, d in TABLES.items()]
    lines += ["* `codebook.csv` gives the definition, source and timing rule of each field.",
              "* `MANIFEST.csv` records the URL, retrieval time and SHA-256 hash of every raw input.",
              "* `validation_results.csv` reports precision and recall of the SEC linkage from hand-labeled samples, when present.",
              "* `SHA256SUMS.txt` lists checksums of the files in this release.", "",
              "## Coverage", "",
              f"* {stats['sbir']['firms']:,} SBIR/STTR firms and {stats['sbir']['awards']:,} awards" if stats.get("sbir") else "",
              f"* {l['firms_linked_high']:,} firms with a high-confidence SEC link" if l else "",
              f"* {stats['federal']['firms_with_any_award']:,} firms found in USAspending, {stats['federal']['firms_sbir_phase3']:,} with a Phase III contract" if stats.get("federal") else "",
              f"* {stats['patents']['firms_with_patents']:,} firms owning {stats['patents']['patents_owned']:,} granted patents; {stats['patents']['patents_citing_an_sbir_award']:,} patents cite an SBIR award number" if stats.get("patents") else "",
              "", "## Usage notes", "",
              "* `has_form_d` and `formd_amount_sold_post` are lower bounds on private fundraising. Many "
              "equity rounds never file a Form D, and filings often omit the amount sold "
              f"(missing or zero for {miss} of post-award offerings of linked firms).",
              "* Form D data are electronic from 2009, Form C from May 2016 and EDGAR from about 1996. "
              "A False outcome for a firm whose post-award window falls outside a source's coverage means the "
              "event was unobserved, and the `*_window_covered` flags mark those firms.",
              "* `public_market_exit` captures firms that went public, or were acquired while public. "
              "Acquisitions of private firms are not observable in public SEC data. Schedule 13D and 13G "
              "filings appear only in `firm_events`, since they exist only for firms with registered shares.",
              "* Federal awards come from USAspending, which is complete from fiscal year 2008. They are matched "
              "by UEI or DUNS only, so firms without either identifier on any SBIR award have no federal fields.",
              "* `sbir_phase3` rests on the FPDS research code (SR3, ST3) where USAspending reports it and on the "
              "award description otherwise. Agencies are known to under-code Phase III, so the flag is a lower bound.",
              "* `founded_year` comes from SAM.gov entity start dates. SAM vintages on disk begin in 2022, so firms "
              "that stopped registering earlier rely on the Form D year of incorporation or have no founding year.",
              "* Patents are the firm's when a high-confidence linked assignee owns them. Patents assigned to "
              "individual inventors or never assigned are not counted. `patents_citing_firm_sbir_award_n` counts "
              "patents tied to the firm's awards by their government-interest statements, whoever owns them.",
              "* `private_acquisition_signal` comes from recorded patent assignments and so only covers firms with patents.",
              "* `firm_year_panel` covers each firm from its founding year (or first record) to 2026. Counts are zero "
              "only inside a source's coverage window, marked by the `cov_*` columns, and missing outside it. "
              "Patent grants for the last three years are incomplete (`cov_patent_grants_complete`).",
              "* Public-company fields exclude a SPAC's own filings before it merged with the firm, and a firm "
              "public before its first award is not counted as going public later.",
              "* Links are probabilistic. Restrict to `link_tier <= 2` for the most conservative analyses, "
              "and see `validation_results.csv` for measured precision and recall.", "",
              *_validation_section(paths),
              "## USPTO input releases", "",
              *[f"* {name} ({pid}), {cover}, last updated {upd}" for pid, name, cover, upd in USPTO_RELEASES.values()],
              "", "## License", "", f"The data are released under {PUBLIC_DATA_LICENSE}. All inputs are U.S. government public records.", ""]
    return "\n".join(x for x in lines if x is not None)


def run(paths: Paths, version: str = __version__, creators: list[dict] | None = None) -> Path:
    out = paths.release / f"v{version}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in TABLES:
        src = paths.processed / f"{name}.parquet"
        if src.exists():
            pd.read_parquet(src).to_csv(out / f"{name}.csv", index=False)
    for src in (paths.processed / "codebook.csv", paths.manifest, paths.validation / "validation_results.csv"):
        if src.exists():
            shutil.copyfile(src, out / src.name)
    (out / "README.md").write_text(_data_readme(paths, version))
    with open(out / "SHA256SUMS.txt", "w") as fh:
        for f in sorted(out.iterdir()):
            if f.name != "SHA256SUMS.txt":
                fh.write(f"{sio.sha256(f)}  {f.name}\n")

    zenodo = {
        "title": "SBIR/STTR Firm Outcomes Database: linking federal R&D awards to SEC, SBA, procurement and patent records",
        "upload_type": "dataset", "version": version, "license": PUBLIC_DATA_LICENSE.lower(),
        "creators": creators or [{"name": "Jayne, Ryan", "orcid": "0009-0001-4662-9686"}],
        "description": "Firm-level database linking all SBIR/STTR awardees to SEC Regulation D, Regulation "
                       "Crowdfunding and EDGAR filings, SBA loans, USAspending federal awards (including SBIR "
                       "Phase III), SAM.gov founding dates and USPTO patents and patent transfers, with spatial "
                       "and macroeconomic covariates. "
                       "Built from public sources with fully reproducible code.",
        "keywords": ["SBIR", "STTR", "innovation policy", "venture capital", "Regulation D", "patents", "federal procurement", "record linkage"],
    }
    (out / ".zenodo.json").write_text(json.dumps(zenodo, indent=2))

    archive = paths.release / f"sbir_firm_outcomes_v{version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.iterdir()):
            zf.write(f, f"sbir_firm_outcomes_v{version}/{f.name}")
    log.info("release written: %s", archive)
    return archive
