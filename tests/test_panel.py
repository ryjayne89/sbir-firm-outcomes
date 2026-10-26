import pandas as pd

from sbirfo.financials import _annual
from sbirfo.places import place_by_year


def test_xbrl_years_combine_concepts_in_priority_order():
    facts = {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [{"val": 7, "frame": "CY2016"}, {"val": 99, "frame": "CY2020"}]}},
        "Revenues": {"units": {"USD": [{"val": 10, "frame": "CY2020"}, {"val": 5, "frame": "CY2020Q1"}]}}}}
    assert _annual(facts, ["Revenues", "SalesRevenueNet"], instant=False).to_dict() == {2016: 7, 2020: 10}


def test_place_measures_bridge_the_2000_census_and_the_acs():
    acs = pd.DataFrame({"zip5": ["00001"], "acs_year": [2012], "pop": [100], "ba_plus_share": [0.4],
                        "grad_share": [0.1], "median_hh_income": [50], "median_home_value": [1]})
    c2000 = pd.DataFrame({"zip5": ["00001"], "year": [2000], "pop": [80], "ba_plus_share": [0.2],
                          "grad_share": [0.05], "median_hh_income": [30], "median_home_value": [0.5]})
    out = place_by_year(pd.DataFrame({"zip5": "00001", "year": [1990, 2006, 2020]}), acs, c2000).set_index("year")
    assert out.loc[1990, "place_source"] == "census2000" and out.loc[1990, "ba_plus_share"] == 0.2
    assert out.loc[2006, "place_source"] == "interpolated" and abs(out.loc[2006, "ba_plus_share"] - 0.3) < 1e-9
    assert out.loc[2020, "place_source"] == "acs" and out.loc[2020, "ba_plus_share"] == 0.4
