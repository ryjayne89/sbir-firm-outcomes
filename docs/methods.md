# Methods

## Sources

| source | coverage | key | used for |
|---|---|---|---|
| SBIR.gov award file | 1983 onward, all agencies | none (firm built from awards) | awards, firm backbone |
| SEC Form D data sets (DERA) | 2008 Q1 onward | CIK, accession number | Regulation D offerings |
| SEC Crowdfunding data sets | 2016 Q2 onward | CIK, accession number | Regulation Crowdfunding offerings |
| SEC EDGAR `submissions.zip` | all filers | CIK | names, former names, addresses, filing history |
| SBA 7(a) and 504 FOIA files | FY1991 onward | none | SBA loans |
| Census 2020 ZCTA gazetteer | national | ZIP | ZIP centroids |
| CDFI Fund Opportunity Zone list | 2018 designations | 2010 census tract | Opportunity Zone flag |
| Census batch geocoder | Census2010 vintage | address | firm tract |
| FRED FEDFUNDS, CPIAUCSL | monthly | date | macro covariate, real dollars |
| USAspending award search and award downloads | fiscal year 2008 onward | UEI, DUNS | federal contracts and grants, Phase III |
| SAM.gov entity extracts | vintages on disk (2022 onward) | UEI | founding year, registration status |
| PatentsView granted and pre-grant tables | grants 1976 onward, applications 2001 onward | disambiguated assignee ID | patents, applications, classes, citations, government interest |
| USPTO Patent Assignment Dataset | recorded assignments 1980 onward | patent number | transfers, mergers, security interests |
| Census ACS 5-year estimates (API) | releases 2012 to 2024 | ZCTA | population, education, income, unemployment, home values |
| NCES IPEDS institutional directory | latest directory | location | colleges and universities, Carnegie class |
| 2000 Census Summary File 3 (API) | 2000 | ZCTA | baseline population, education, income, home values |
| SEC XBRL company facts (`companyfacts.zip`) | about 2009 onward | CIK | annual revenue, R&D, income, assets, IPO proceeds |

USPTO inputs were downloaded by hand on 16 October 2026 from the USPTO Open Data Portal Bulk Dataset Directory (data.uspto.gov). PatentsView revises its entity disambiguation between releases, so the releases are listed here and their file hashes are in `MANIFEST.csv`.

| product | identifier | coverage | last updated |
|---|---|---|---|
| PatentsView Granted Patent Disambiguated Data | PVGPATDIS | grants 1 January 1976 to 31 December 2025 | 10 April 2026 |
| PatentsView Pre-Grant Publication Disambiguated Data | PVPGPUBDIS | publications 15 March 2001 to 31 December 2025 | 6 October 2026 |
| Patent Assignment Data for Academia and Researchers | ECORSEXC | recordations from August 1980 | 25 September 2026 |

## Name normalization

All matching uses one function, `sbirfo.names.name_key`. It converts a name to lowercase ASCII, replaces `&` with `and`, removes punctuation, cuts "doing business as" and "a Delaware corporation" tails, and removes trailing legal-form tokens (Inc, LLC, Corp and similar) and a leading "the". Only trailing tokens are removed, so a "co" inside a name is kept.

A second key, `loose_key`, also removes generic words (Technologies, Systems, Labs and similar) and spaces. It never creates a link by itself. It decides whether a name is distinctive enough to link on (at least four characters must remain) and, with city or ZIP agreement, matches names that differ only in generic words.

## Firm assembly

The award file has no firm identifier. Each award is a node, and two awards are joined when they share

1. a UEI,
2. a DUNS number,
3. a website domain used by no more than three distinct names and not a generic host,
4. the same strict name in the same state, for distinctive names, or
5. the same strict name at the same ZIP code, for names too short to be distinctive.

Connected awards form a firm. `firm_id_sbir` is a hash of the firm's smallest UEI, or smallest DUNS, or smallest name and state, so the same data always produce the same identifiers. Two firms that share a name and state key receive numbered suffixes in the order of their earliest award.

## Linkage to SEC CIKs

SEC names come from EDGAR current names (business and mailing addresses), EDGAR former names, Form D issuer records with their previous names, and Form C issuers. Candidates are proposed three ways.

* exact, the same strict key
* loose, the same generic-word-free key of at least six characters
* fuzzy, a rapidfuzz token-sort ratio of at least 90 between names in the same state whose loose keys share a first letter, with identical digits

Each candidate is scored on state, city and ZIP agreement and assigned a tier.

| tier | rule | confidence |
|---|---|---|
| 1 | exact name, same state, same city or ZIP | high |
| 2 | exact name, same state | high |
| 3 | same loose key, same state, same city or ZIP | high |
| 4 | fuzzy name, same state, same city or ZIP | high |
| 5 | fuzzy name at a ratio of 95 or more, same state | medium |
| 6 | exact name in another state, the name held by one SBIR firm and one CIK | medium |

A firm may link to several CIKs, for example a predecessor entity or an affiliate. All links are kept. The primary CIK is chosen by best tier, then non-fund, then a CIK not shared with another SBIR firm, then most SEC filings, then lowest CIK.

## Outcomes

Post-award means strictly after the firm's first award date. Award dates missing in the source are set to July 1 of the award year and flagged.

Form D amendments carry cumulative amounts, so each amendment is traced to its original notice through `PREVIOUSACCESSIONNUMBER`, and an offering's amount is the largest amount reported on any of its filings. `years_to_private_cap` uses the date of first sale reported on Form D and falls back to the filing date when that is missing.

## Geocoding and Opportunity Zones

Opportunity Zones were designated in 2018 as 2010 census tracts, so firm addresses are geocoded with the Census Bureau's batch geocoder using the `Census2010_Current` vintage. Each firm goes through three passes, stopping at the first match.

1. The address on the firm's first award, as reported.
2. The same address cleaned. Unit, suite, floor and building designators are removed, and Address2 is used when Address1 has no street number or is a PO box.
3. The firm's other reported addresses, cleaned, taken in order of first use.

`geo_source` records which pass placed the firm. Two checks run on every placed firm. The state in the tract code must agree with the reported state, and the matched point's distance from the ZIP centroid is reported in `geo_miles_from_zip`. Firms never placed carry `geo_fail_reason`, judged on the first-award address in this order: blank address, PO box, no street number, missing or non-US state, ZIP that is not a ZCTA (PO-box-only and single-organization ZIPs), several equally good candidates (`tie`), and otherwise `no_match`. A stratified sample of unplaced firms is written to `data/audit/geocode_unplaced_sample.csv` for inspection.

For every firm with a ZIP, `oz_zip_share` gives the share of the ZIP's 2010 population living in designated tracts, from the Census ZCTA-to-tract relationship file (land area is used for unpopulated ZIPs). It is a ZIP-level measure, available for firms the geocoder could not place, and should not be read as the firm's own status.

## Federal contracts and grants

Each UEI and DUNS number on a firm's SBIR awards is searched in the USAspending award search, contracts (award types A to D) and grants (02 to 05) separately, for awards with activity from 1 October 2007. Identifiers held by two SBIR firms are not searched. The search also matches recipient names, so an award is kept only when its recipient UEI or DUNS equals one of the firm's identifiers. The pipeline checks once that a search on ten identifiers returns the union of the ten single searches and falls back to single searches if not.

The FPDS research code (data element 10Q) marks SBIR Phase I, II and III contracts as SR1, SR2 and SR3, and STTR contracts as ST1, ST2 and ST3. USAspending returns it only in its award download files, and its download service accepts only a few jobs per client before refusing, so the code is not read by default. Phase III is then identified from award descriptions and is a lower bound. Passing `--research-codes` reads the code for contracts that could be Phase III (those starting after the firm's first award, with a positive obligation, that are not SBIR awards by number), through download jobs of 500 award IDs that resume across runs. Each award is classed in this order.

1. SBIR/STTR Phase III: research code SR3 or ST3, or, when no code is reported, a description that names SBIR or STTR and Phase III.
2. SBIR/STTR award: an award number equal to a contract or grant number on one of the firm's SBIR.gov awards (compared as letters and digits, with NIH year and support suffixes removed, and for NIH grants also by institute code and serial number), research code SR1, SR2, ST1 or ST2, an NIH activity code R41 to R44 or U43/U44, or a description naming SBIR or STTR.
3. Non-SBIR federal award.

When an SBIR.gov award number since fiscal year 2008 is not found under its firm, the number is looked up in USAspending directly. If it belongs to one SBIR firm, the recipient UEI or DUNS on the returned award is added to that firm, unless another firm holds it, and searched in a second pass. This recovers DUNS numbers that SBIR.gov never recorded. In a test sample of 25 firms, 92% of SBIR.gov awards were found before this step.

Post-award fields count awards whose base obligation date (or start date) is after the first SBIR award. `data/audit/federal.json` reports the share of SBIR.gov awards from fiscal year 2008 onward that are found in USAspending under the same firm, a direct check on the identifier match.

## Founding year

SAM entity extracts are matched by UEI. A firm with no UEI match is matched by strict name and state with city or ZIP agreement, when the name is held by one SBIR firm and one SAM entity. `founded_year` is the year of the earliest SAM entity start date. When SAM has only a name match and the firm's Form D filings report a year of incorporation, the Form D year is used. Negative ages, where the SAM start date follows the first award, are kept and flagged, since they usually mark an entity that was re-registered after a reorganization. The audit reports agreement between SAM and Form D years for firms with both.

## Patents

PatentsView disambiguated assignees, granted and pre-grant, are linked to SBIR firms with the candidate generation and tiers of the SEC linkage, using the assignee's disambiguated city and state. Assignees outside the United States carry no state and so link only through tier 6. An assignee that two SBIR firms claim at the same best tier is assigned to neither. A granted patent belongs to a firm when a high-confidence linked assignee owns it.

Separately, the contract numbers in government-interest statements (`g_gov_interest_contracts`) are compared with SBIR.gov award numbers held by one firm, by full number and, for NIH grants, by institute code and serial number. A match ties the patent to a specific SBIR award whether the firm, a university partner or an inventor owns it.

Patent dates are the application filing date (`g_application`) and grant date. Technology fields are the first-listed current CPC subclass and WIPO field. Forward citations count U.S. patents citing the patent within five years of its grant, and are left missing for patents granted less than five years before the end of the data.

## Patent transfers

The Patent Assignment Dataset records are restricted to documents naming a firm's owned patents. A record counts as the firm's when an assignor name has the firm's strict name key or loose key. Conveyance types are read as transfer out (an assignment to another party), merger, security interest, release and name change. `private_acquisition_signal` is set by a merger conveyance, or by assignments that move at least half of the firm's patents, and at least two, to a single buyer after the first award.

## Going public and public-company data

EDGAR filing histories are kept for registration, listing, periodic, current-report, ownership, merger and deregistration forms. A firm goes public by one of three routes, taking the earliest after its first award.

* `ipo`, the first 424B1 or 424B4 prospectus.
* `listing`, an 8-A12B or Form 10-12B exchange registration with no IPO prospectus within 120 days, which covers direct listings and moves up from over-the-counter trading. An IPO prospectus within 120 days of the first such filing marks the listing as an `ipo`, because the exchange registration is usually filed a few days before the final prospectus.
* `spac_merger`. Many recent firms went public by merging into a special purpose acquisition company, whose CIK then takes the firm's name and so links to it. A linked CIK whose former names include a blank-check name (for example "... Acquisition Corp") and whose current name does not is treated as a completed SPAC merger, dated when the blank-check name was dropped. The SPAC's own filings before that date, including its own IPO, are excluded from every field.

A registration (S-1, F-1, SB-2) followed by a withdrawal (RW) with no listing within three years is recorded as `ipo_withdrawn`. For the firm's primary high-confidence CIK, annual financials come from the SEC's XBRL company facts, using the calendar-year frames the SEC assigns so that each company, concept and year has one value. Because companies change tags (revenue moved to `RevenueFromContractWithCustomer...` in 2018), each year takes the first tag in a fixed priority order that reports it.

## Firm-year panel

`firm_year_panel` has one row per firm and calendar year from the firm's start to 2026. The start is the earlier of its founding year and its first record in any source, and not before 1976, when PatentsView begins. Every source is counted by year before and after the first award: SBIR awards and reported employees, Regulation D offerings with the investors and revenue range the issuer reports, Regulation Crowdfunding offerings, SBA loans, federal contracts and grants (name-verified), patents filed and granted with their forward citations, published applications, patent transfers and security interests, years as a public filer, XBRL financials, and SAM registration in the years a SAM vintage exists. Each source has a coverage flag (`cov_formd` from 2009, `cov_federal` from 2008, `cov_sba` from 1991 and so on); a count is zero only inside its source's coverage and missing outside it. `event_time` is the year minus the first award year, and `firm_age` is the year minus the founding year. Because the start depends on the first record when the founding year is unknown, the panel is unbalanced before the award: a firm with no founding year enters in the year of its first raise, patent or award, so averages over existing pre-award rows over-represent active firms. Event-study comparisons should fix a window and count every firm in every year of it, with zero where a covered source recorded nothing, as `figures/make_figures.py` does.

The firm's location in a year is its latest SBIR address in use by then, and its first address before that. Place measures for that ZIP come from the 2000 Census through 2000, from the ACS 5-year release ending in that year from 2012 (the latest release after 2024), and from a straight line between the 2000 and 2012 values in 2001 to 2011; `place_source` records which. `deflator_to_2025` converts any dollar column to 2025 dollars with the CPI-U.

## Place characteristics

Each firm's ZIP is taken from its first-award address. American Community Survey 5-year estimates for that ZCTA come from the release whose final year is closest to the first award year, within the 2012 to 2024 releases; `acs_year_used` records it, so firms that entered before 2012 carry later values. The same counts are also pooled over every ZCTA whose centroid lies within 15 miles, a closer stand-in for the local labor market than a single ZIP. Colleges and universities are degree-granting, active institutions in the IPEDS directory, with R1 and R2 taken from the Carnegie basic classification, counted within 15 miles of the ZIP centroid.

## Checks that restrict what counts as an outcome

* A federal award found through a UEI or DUNS counts toward a firm only when the recipient name is one of the firm's SBIR names (the same strict key, the same distinctive loose key, or a token-sort ratio of at least 85). Identifiers outlive firms. An acquired firm's DUNS can be used by its new parent, and an identifier recovered through an award number can belong to a prime contractor, so without this check a few firms were credited with tens of billions of dollars of their parents' contracts. All matched awards stay in `federal_awards` with `recipient_name_matches`.
* A Phase III contract dated before the firm's first SBIR award is not counted as Phase III.
* A firm public before its first award is not counted as going public later.
* A patent assignee is matched only on locations that hold at least a quarter of its documents, or its most common one, and generic-word-insensitive matches to assignees need a name core of at least eight characters. This keeps small firms from being linked to large namesakes through a stray address or a short shared word.
* `established_firm_flag`, `institution_name_flag` and `age_implausible_flag` mark firms to set aside or examine in analyses.

## Validation

`sbirfo.validate` draws a simple random sample of firms and an oversample of linked pairs per tier with a fixed seed. Precision is the share of predicted primary CIKs that the labeler confirms. Recall is the share of firms with a true CIK whose predicted primary CIK is correct. Intervals are Wilson 95% intervals. In version 1.2.0 the first 200 random rows were labeled; the release README reports the results and the patterns noted while labeling (renamed filers, holding-company names and early registrants).

A consistency check compares firms with a Form D in the DERA data sets to firms with a `D` filing in their EDGAR history, since both describe the same filings.
