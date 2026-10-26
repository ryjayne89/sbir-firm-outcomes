# SBIR/STTR Firm Outcomes Database

Code that builds a firm-level database linking every Small Business Innovation Research (SBIR) and Small Business Technology Transfer (STTR) awardee, across all federal agencies since 1983, to its later private financing and public-market events. All inputs are public U.S. government records, and every step from download to release runs from one command.

The database links SBIR/STTR firms to

* SEC Regulation D notices (Form D), 2009 onward
* SEC Regulation Crowdfunding filings (Form C), 2016 onward
* the EDGAR filing history of each linked company (IPOs, direct listings, SPAC mergers, withdrawn IPOs, years as a public company, tender offers, merger votes, Regulation A offerings, deregistrations) and its annual XBRL financials
* SBA 7(a) and 504 loans, 1991 onward
* federal prime contracts and grants from USAspending, fiscal year 2008 onward, with SBIR/STTR Phase III contracts identified by their FPDS research code
* founding year from SAM.gov entity registrations, checked against the Form D year of incorporation
* USPTO patents and published applications from PatentsView, with filing and grant dates, CPC and WIPO technology classes, inventor counts, forward citations, and government-interest statements that cite specific SBIR awards
* recorded patent transfers, mergers and security interests from the USPTO Patent Assignment Dataset, which reveal acquisitions of private firms that SEC data miss
* the education, income and population of each firm's ZIP code and surrounding area (Census ACS), and nearby colleges and research universities (IPEDS)
* spatial and macroeconomic covariates (HUBZone status, Opportunity Zone tracts, nearby Regulation D capital, the federal funds rate)

## Quick start

```bash
pip install -e .
python -m sbirfo --data-dir ./data --sec-user-agent "Your Name you@institution.edu"
```

The SEC asks automated users to identify themselves, so `--sec-user-agent` must contain a name and an email address. On Google Colab, open `notebooks/run_colab.ipynb`, set the variables in its first cell and run all cells.

A full run downloads about 3 GB of SEC and Census files plus the PatentsView tables (about 10 GB unzipped), and takes several hours, most of it spent parsing the EDGAR bulk file, geocoding addresses, querying USAspending and scanning the patent citation table. Every step skips work whose output already exists, so an interrupted run resumes where it stopped. Individual steps can be run with `--steps`, for example `--steps link outcomes`.

### Files that may need a manual download

Two hosts often refuse scripted requests from cloud machines. When that happens the pipeline logs a warning, continues without them, and uses any copy you place in the folder below on the next run.

| source | where to get it | where to put it |
|---|---|---|
| SBA 7(a) and 504 FOIA loan files | https://data.sba.gov/dataset/7-a-504-foia | `data/raw/sba_foia/` |
| CPI-U (series CPIAUCSL) | https://fred.stlouisfed.org/series/CPIAUCSL | `data/raw/macro/CPIAUCSL.csv` |
| SAM.gov entity extracts, one per vintage, named `sam_public_YYYY_MM.parquet` or `.csv` | https://sam.gov/data-services (Entity Registration, public extract) | `data/raw/sam/`, or any folder passed with `--sam-dir` |
| PatentsView Granted Patent Disambiguated Data (PVGPATDIS) and Pre-Grant Publication Disambiguated Data (PVPGPUBDIS), the `g_*` and `pg_*` tables | https://data.uspto.gov/bulkdata | `data/raw/patents/` |
| Patent Assignment Data for Academia and Researchers (ECORSEXC), Stata or TSV files `assignment`, `assignment_conveyance`, `assignor`, `assignee`, `documentid` | https://data.uspto.gov/bulkdata | anywhere under `data/raw/patents/` |

The releases used for version 1.1.0 are listed in `docs/methods.md`. The patent step uses files on disk first and tries the old PatentsView download links only for tables it cannot find. Tables may be zipped, unzipped, or unzipped into a folder of the same name, as Google Drive often leaves them. The patent step needs `g_patent`, `g_application`, `g_assignee_disambiguated` and `g_location_disambiguated`; the remaining tables (`g_cpc_current`, `g_wipo_technology`, `g_inventor_disambiguated`, `g_us_patent_citation`, `g_gov_interest`, `g_gov_interest_contracts`, `g_gov_interest_org`, `pg_published_application`, `pg_assignee_disambiguated`) each add fields and are skipped when absent.

Every raw file, whether downloaded by the pipeline or by hand, is recorded in `data/raw/MANIFEST.csv` with its source URL, retrieval time, size and SHA-256 hash.

## Pipeline

| step | module | output |
|---|---|---|
| acquire | `acquire.py` | raw files and parsed tables in `data/raw` and `data/interim` |
| sbir | `sbir.py` | cleaned awards and the firm table, keyed by `firm_id_sbir` |
| link | `link.py` | firm to SEC CIK links with confidence tiers |
| federal | `federal.py` | USAspending contracts and grants per firm, classed as SBIR, Phase III or other |
| sam | `sam.py` | founding year and SAM registration by vintage |
| financials | `financials.py` | annual XBRL financials of firms that file with the SEC |
| places | `places.py` | ZIP-level education, income and population (ACS) and nearby universities (IPEDS) |
| patents | `patents.py` | assignee links, owned patents and applications, award-cited patents, patent transfers |
| outcomes | `outcomes.py` | `firm_outcomes`, `firm_events` and the codebook |
| panel | `panel.py` | `firm_year_panel`, every source by firm and year over each firm's life |
| validate | `validate.py` | a labeling sheet, and precision and recall once it is labeled |
| release | `release.py` | CSV tables, data README, checksums and Zenodo metadata in `data/release` |

`docs/methods.md` describes the firm assembly and linkage rules in detail, including the confidence tiers.

## The firm-year panel

`firm_year_panel` follows each firm from its founding year, or its first record in any source, to 2026, so outcomes can be compared with each firm's own baseline before its first award. Its `cov_*` columns mark the years each source covers. Within coverage a missing event is a zero; outside it the value is missing.

## Validating the linkage

`validate` writes `data/validation/label_sheet.csv`. It contains a simple random sample of 300 firms, which yields precision and recall, and 40 linked pairs per tier, which yields precision by tier. Each row has a link to an EDGAR company search. For every row, enter in `true_cik` the CIK or CIKs that belong to the firm, separated by semicolons, or `none` if EDGAR has no filer for it. Save the result as `label_sheet_labeled.csv` in the same folder and rerun `--steps validate release`. The scores, with Wilson 95% intervals, go to `validation_results.csv` and into the release.

## Interpreting the outcomes

* `has_form_d` and the Form D amounts are lower bounds on private fundraising. Not every equity round files a Form D, and filings often omit the amount sold.
* Form D data are electronic from 2009 and Form C data begin in May 2016. A False outcome for a firm whose post-award window falls before those dates means the event was not observable, which the `*_window_covered` flags record.
* `public_market_exit` covers firms that went public, or that were acquired while public. The acquisition of a private firm leaves no public SEC record tied to the target, so it cannot be observed here. Schedule 13D and 13G filings are kept in `firm_events` but are not used as an outcome, because they exist only for companies with registered shares.
* `private_acquisition_signal` fills part of that gap. It is set when the firm, as assignor, records a merger with the USPTO, or assigns at least half of its patents (two or more) to one buyer. It is observable only for firms with patents.
* `sbir_phase3` is a lower bound. Agencies under-report Phase III in FPDS, and USAspending begins in fiscal year 2008.
* `founded_year` depends on SAM registrations from 2022 onward, so it is most complete for firms still doing federal business. Firms that stopped registering earlier fall back to the Form D year of incorporation.

## Related work

Earlier studies linked SBIR awards to later private financing using program-specific samples or commercial venture databases. Lerner (1999) followed SBIR awardees and matched non-awardees to study long-run growth and venture financing. Howell (2017) used Department of Energy applicant rankings to estimate the effect of Phase I grants on later venture capital. The National Academies' program assessments, including the 2009 study of NIH SBIR and venture funding and the 2022 assessment of the NSF programs, drew on surveys and proprietary venture data. This database differs in using only public sources, covering every agency and the full program history, publishing the linkage code with measured accuracy, and attaching federal procurement, patent and spatial covariates.

* Lerner, J. (1999). The government as venture capitalist: the long-run impact of the SBIR program. *Journal of Business*, 72(3), 285 to 318.
* Howell, S. T. (2017). Financing innovation: evidence from R&D grants. *American Economic Review*, 107(4), 1136 to 1164.
* National Research Council (2009). *Venture Funding and the NIH SBIR Program*. National Academies Press.
* National Academies of Sciences, Engineering, and Medicine (2022). *Assessment of the SBIR and STTR Programs at the National Science Foundation*. National Academies Press.

## Testing

```bash
pip install -e ".[test]"
pytest
```

## License and citation

Code is released under the MIT License. The released data are CC-BY-4.0; all inputs are public records of the U.S. government. Citation metadata are in `CITATION.cff`.

## Figures

`python figures/make_figures.py --data-dir <DATA_DIR>/processed --out figures/out` writes seven single-panel descriptive figures (PNG and PDF) from the release tables. Event-time figures use a balanced cohort, counting every firm in every year of the window.
