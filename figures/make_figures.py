"""Descriptive figures of SBIR/STTR firms over time, built from the release tables.

    python figures/make_figures.py --data-dir /path/to/data --out figures/out

Each figure is one panel. All text is one size, with titles larger, so the figures read the
same at any print size. Colors are three fixed categorical slots that stay distinguishable
under common color-vision deficiencies; every line also carries a direct label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEXT, TITLE = 13, 15

plt.rcParams.update({
    "font.size": TEXT, "axes.titlesize": TITLE, "axes.labelsize": TEXT, "xtick.labelsize": TEXT,
    "ytick.labelsize": TEXT, "legend.fontsize": TEXT, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.grid.axis": "y", "grid.linewidth": 0.8, "lines.linewidth": 2, "figure.dpi": 150,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def _save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def _label_end(ax, x, y, text, color):
    """Direct label at the end of a line, in ink, with a short colored key."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=TEXT)


def entrants(fo, out):
    y = fo.first_award_date.dt.year.value_counts().sort_index().loc[1983:2025]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(y.index, y.values, color=BLUE)
    ax.set_title("First-time SBIR/STTR awardees per year")
    ax.set_ylabel("Firms")
    ax.set_ylim(0, None)
    _save(fig, out, "fig1_first_time_awardees")


def age_at_entry(fo, out):
    d = fo[(fo.founded_source.astype("string").str.startswith("sam_")) & ~fo.age_implausible_flag.fillna(False)].copy()
    d["cohort"] = d.first_award_date.dt.year
    d = d[d.cohort.between(2010, 2025) & (d.age_at_first_award >= 0)]
    g = d.groupby("cohort").age_at_first_award
    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(med.index, q1.values, q3.values, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(med.index, med.values, color=BLUE)
    _label_end(ax, med.index[-1], med.values[-1], "Median", BLUE)
    _label_end(ax, q3.index[-1], q3.values[-1], "75th percentile", BLUE)
    ax.set_title("Firm age at first award, by year of first award")
    ax.set_ylabel("Years since founding")
    ax.set_ylim(0, None)
    _save(fig, out, "fig2_age_at_first_award")


def event_time(panel, fo, out, col, cohorts, window, title, ylabel, name, per=100, share=False):
    """Balanced cohort: every firm in the cohorts counts in every event year, with zero where the
    source recorded nothing. The panel itself starts at a firm's founding year or, when that is
    unknown, at its first recorded activity, so averaging only over existing panel rows would
    over-represent active firms before the award."""
    phase = fo.set_index("firm_id_sbir").sbir_phase_flag.map({1: "Phase I only", 3: "Phase I then II"})
    firms = fo.loc[fo.first_award_date.dt.year.between(*cohorts) & ~fo.established_firm_flag.fillna(False)
                   & ~fo.institution_name_flag.fillna(False), "firm_id_sbir"]
    grid = pd.MultiIndex.from_product([firms, range(window[0], window[1] + 1)], names=["firm_id_sbir", "event_time"])
    vals = panel.set_index(["firm_id_sbir", "event_time"])[col]
    p = vals.reindex(grid).fillna(0).rename("raw").reset_index()
    p["group"] = p.firm_id_sbir.map(phase)
    p = p.dropna(subset=["group"])
    p["v"] = (p.raw > 0).astype(float) if share else p.raw
    g = p.groupby(["group", "event_time"]).v.mean().unstack(0) * per
    n = p.groupby("group").firm_id_sbir.nunique()
    print(f"{name}: firms per group {n.to_dict()}")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for grp, color in (("Phase I then II", BLUE), ("Phase I only", ORANGE)):
        if grp in g:
            ax.plot(g.index, g[grp].values, color=color, marker="o", markersize=4)
            _label_end(ax, g.index[-1], g[grp].values[-1], grp, color)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title(title)
    ax.set_xlabel("Years from first award")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    _save(fig, out, name)


def outcomes_by_cohort(fo, panel, out):
    """Share of firms with each outcome within five years after the first award, for cohorts
    whose five-year window ends by 2023, since patent applications publish about 18 months after filing."""
    keep = fo.loc[~fo.established_firm_flag.fillna(False) & ~fo.institution_name_flag.fillna(False), "firm_id_sbir"]
    p = panel[panel.event_time.between(1, 5) & panel.first_award_year.between(2009, 2018) & panel.firm_id_sbir.isin(keep)]
    agg = p.groupby("firm_id_sbir").agg(formd=("formd_n", "sum"), fed=("fed_other_contract_n", "sum"),
                                        fedg=("fed_other_grant_n", "sum"), pat=("patents_filed_n", "sum"),
                                        cohort=("first_award_year", "first"))
    rates = pd.DataFrame({
        "Regulation D raise": agg.formd.gt(0),
        "Non-SBIR federal award": (agg.fed + agg.fedg).gt(0),
        "Patent filed": agg.pat.gt(0),
    }).groupby(agg.cohort).mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for (name, s), color in zip(rates.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(s.index, s.values, color=color, marker="o", markersize=4)
        _label_end(ax, s.index[-1], s.values[-1], name, color)
    ax.set_title("Outcomes within five years of the first award")
    ax.set_xlabel("Year of first award")
    ax.set_ylabel("Percent of firms")
    ax.set_ylim(0, None)
    _save(fig, out, "fig5_outcomes_by_cohort")


def going_public(fo, out):
    r = fo[["ipo_prospectus_date", "exchange_listing_date", "spac_merger_date"]]
    when = r.min(axis=1)
    near = (fo.ipo_prospectus_date - when).dt.days.abs() <= 120
    route = np.select([when.notna() & (when == fo.spac_merger_date), when.notna() & near.fillna(False), when.notna()],
                      ["SPAC merger", "IPO", "Exchange listing"], None)
    d = pd.DataFrame({"year": when.dt.year, "route": route})
    d = d[d.route.notna() & ~fo.public_before_award.fillna(False).values & d.year.between(1995, 2025)]
    t = d.groupby(["year", "route"]).size().unstack(fill_value=0).reindex(range(1995, 2026), fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bottom = np.zeros(len(t))
    for name, color in (("IPO", BLUE), ("SPAC merger", ORANGE), ("Exchange listing", AQUA)):
        if name in t:
            ax.bar(t.index, t[name].values, bottom=bottom, color=color, width=0.8, edgecolor="#fcfcfb",
                   linewidth=1, label=name)
            bottom += t[name].values
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("SBIR/STTR firms going public, by route")
    ax.set_ylabel("Firms")
    _save(fig, out, "fig6_going_public")


def national_ba_share(zc, z2000):
    """National share of adults 25+ with a bachelor's degree, built the same way as the firm
    series: the 2000 Census, the ACS from 2012, and a straight line between them."""
    if not len(zc):
        return None
    us = zc.groupby("acs_year").apply(lambda d: d.ba_plus.sum() / d.pop25.sum())
    if len(z2000):
        z = z2000.dropna(subset=["ba_plus_share"])
        w = z["pop25"] if "pop25" in z else z["pop"]       # tables built before pop25 was kept
        us.loc[2000] = (z.ba_plus_share * w).sum() / w.sum()
        us = us.reindex(range(2000, int(us.index.max()) + 1)).interpolate()
    return us * 100


def education(panel, zc, z2000, out):
    first = panel[panel.event_time == 0].dropna(subset=["zip_ba_plus_share"])
    firm = first.groupby("first_award_year").zip_ba_plus_share.median() * 100
    us = national_ba_share(zc, z2000)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    s = firm.loc[2000:2024]
    ax.plot(s.index, s.values, color=BLUE)
    _label_end(ax, s.index[-1], s.values[-1], "Firms' ZIP codes (median)", BLUE)
    if us is not None:
        u = us.loc[2000:2024]
        ax.plot(u.index, u.values, color=ORANGE)
        _label_end(ax, u.index[-1], u.values[-1], "United States", ORANGE)
    ax.set_title("Adults with a bachelor's degree, where SBIR firms start")
    ax.set_xlabel("Year of first award")
    ax.set_ylabel("Percent of adults 25 and over")
    ax.set_ylim(0, None)
    _save(fig, out, "fig7_education_of_location")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding the processed parquet tables")
    ap.add_argument("--out", type=Path, default=Path("figures/out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = a.data_dir
    fo = pd.read_parquet(d / "firm_outcomes.parquet")
    panel = pd.read_parquet(d / "firm_year_panel.parquet")
    zc = pd.read_parquet(d / "zcta_characteristics.parquet") if (d / "zcta_characteristics.parquet").exists() else pd.DataFrame()

    entrants(fo, a.out)
    age_at_entry(fo, a.out)
    event_time(panel, fo, a.out, "formd_n", (2014, 2016), (-5, 9), "Firms raising private capital around the first award",
               "Percent raising under Regulation D", "fig3_regd_event_time", share=True)
    event_time(panel, fo, a.out, "patents_filed_n", (2003, 2010), (-5, 10), "Patents filed around the first award",
               "Patents per 100 firms", "fig4_patents_event_time")
    outcomes_by_cohort(fo, panel, a.out)
    going_public(fo, a.out)
    z2000 = pd.read_parquet(d / "zcta_characteristics_2000.parquet") if (d / "zcta_characteristics_2000.parquet").exists() else pd.DataFrame()
    education(panel, zc, z2000, a.out)


if __name__ == "__main__":
    main()
