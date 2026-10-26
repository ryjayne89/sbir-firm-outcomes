import pandas as pd

from sbirfo.geo import clean_street, failure_reason


def test_clean_street_drops_units_and_uses_address2():
    a1 = pd.Series(["5 Oak St, Suite 210", "PO Box 77", "Acme Labs Inc", "12 Main St #4B", None])
    a2 = pd.Series([None, "7 Bay Rd", "100 First Ave", None, None])
    assert clean_street(a1, a2).tolist()[:4] == ["5 OAK ST", "7 BAY RD", "100 FIRST AVE", "12 MAIN ST"]
    assert pd.isna(clean_street(a1, a2).iloc[4])


def test_failure_reason_order():
    df = pd.DataFrame({"address1": ["", "PO Box 1", "Acme Inc", "9 Elm", "9 Elm", "9 Elm"],
                       "address2": [None] * 6,
                       "state": ["TX", "TX", "TX", None, "TX", "TX"],
                       "zip5": ["78701", "78701", "99999", "78701", "99999", "78701"],
                       "match": ["No_Match", "No_Match", "No_Match", "No_Match", "No_Match", "Tie"]})
    assert failure_reason(df, {"78701"}).tolist() == [
        "blank_address", "po_box", "no_street_number", "non_us_or_missing_state", "zip_not_a_zcta", "tie"]
