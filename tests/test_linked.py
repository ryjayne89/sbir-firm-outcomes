import pandas as pd

from sbirfo.federal import classify, norm_award_id
from sbirfo.patents import patent_number


def test_award_numbers_compare_without_punctuation_or_nih_suffixes():
    s = pd.Series(["1R43CA123456-01A1", "R43CA123456", "W911NF-15-C-0123", "w911nf15c0123", "C1", None])
    out = norm_award_id(s).tolist()
    assert out[0] == out[1] == "R43CA123456"
    assert out[2] == out[3] == "W911NF15C0123"
    assert pd.isna(out[4]) and pd.isna(out[5])


def test_patent_numbers_drop_leading_zeros_and_keep_prefixes():
    assert patent_number(pd.Series(["09000001", "D0612345", "RE045678", "10,500,001"])).tolist() == \
        ["9000001", "D612345", "RE45678", "10500001"]


def _awards(**over):
    base = dict(firm_id_sbir="F1", award_id="X1", description="", research_code=pd.NA, date=pd.Timestamp("2015-01-01"))
    base.update(over)
    return base


def test_phase3_prefers_the_fpds_code_and_falls_back_to_the_description():
    a = pd.DataFrame([
        _awards(award_id="A", research_code="SR3"),
        _awards(award_id="B", description="SBIR Phase III follow-on"),
        _awards(award_id="C", description="SBIR Phase III follow-on", research_code="SR2"),
        _awards(award_id="R44GM123456", description="device"),
        _awards(award_id="W911NF15C0123", description="services"),
        _awards(award_id="D", description="logistics"),
    ])
    sb = pd.DataFrame({"firm_id_sbir": ["F1"], "contract": ["W911NF-15-C-0123"], "award_id": ["SB1"]})
    firms = pd.DataFrame({"firm_id_sbir": ["F1"], "first_award_date": [pd.Timestamp("2010-01-01")]})
    out = classify(a, sb, firms).set_index("award_id")
    assert out.loc["A", "award_class"] == "sbir_phase3" and out.loc["A", "phase3_source"] == "fpds_research_code"
    assert out.loc["B", "award_class"] == "sbir_phase3" and out.loc["B", "phase3_source"] == "description"
    assert out.loc["C", "award_class"] == "sbir_program"
    assert out.loc["R44GM123456", "award_class"] == "sbir_program"
    assert out.loc["W911NF15C0123", "award_class"] == "sbir_program"
    assert out.loc["W911NF15C0123", "sbir_award_id"] == "SB1"
    assert out.loc["D", "award_class"] == "non_sbir"


def test_gov_interest_statements_tie_patents_to_unique_award_numbers():
    from sbirfo.patents import gov_interest_links

    class Src:
        def read(self, t, cols, keep=None):
            return pd.DataFrame({"patent_id": ["1", "2", "3"],
                                 "contract_award_number": ["W911NF-15-C-0123", "CA123456; GM999999", "N00014-1"]})
    sb = pd.DataFrame({"award_id": ["a", "b", "c"], "firm_id_sbir": ["F1", "F2", "F3"],
                       "contract": ["W911NF15C0123", "1R43GM999999-01", "N00014-1"]})
    out = gov_interest_links(Src(), sb).sort_values("patent_id")
    assert out.patent_id.tolist() == ["1", "2", "3"]
    assert out.firm_id_sbir.tolist() == ["F1", "F2", "F3"]


def test_nih_phase2_number_matches_phase1_record_by_institute_and_serial():
    from sbirfo.federal import match_sbir_numbers
    a = pd.DataFrame({"firm_id_sbir": ["F1", "F1"], "award_key": norm_award_id(pd.Series(["R43GM112196", "R01CA000001"]))})
    sb = pd.DataFrame({"firm_id_sbir": ["F1"], "contract": ["2R44GM112196-02"], "award_id": ["S2"]})
    assert match_sbir_numbers(a, sb).tolist()[0] == "S2"
    assert pd.isna(match_sbir_numbers(a, sb).tolist()[1])


def test_missed_award_numbers_add_the_recipient_identifier(tmp_path, monkeypatch):
    from sbirfo import federal
    from sbirfo.config import Paths
    paths = Paths(tmp_path).make()
    monkeypatch.setattr(federal, "search_award_ids", lambda web, nums, group, end: pd.DataFrame(
        {"Award ID": ["FA955011C0018"], "Recipient UEI": [None], "Recipient DUNS Number": ["12345678"]})
        if group == "contract" else pd.DataFrame())
    sb = pd.DataFrame({"firm_id_sbir": ["F1"], "contract": ["FA9550-11-C-0018"], "award_date": [pd.Timestamp("2011-05-01")]})
    a = pd.DataFrame({"firm_id_sbir": pd.Series([], dtype="string"), "award_id": pd.Series([], dtype="string")})
    ids = pd.DataFrame({"firm_id_sbir": ["F1"], "id": ["AAAABBBBCCCC"], "kind": ["uei"]})
    new = federal.expand_identifiers(paths, None, a, sb, ids)
    assert new.to_dict("records") == [{"firm_id_sbir": "F1", "id": "012345678", "kind": "duns"}]


def test_recipient_name_check_keeps_the_firm_and_drops_an_acquirer():
    from sbirfo.federal import recipient_matches_firm
    from sbirfo import names as nm
    keys = nm.add_keys(pd.DataFrame({"firm_id_sbir": ["F1", "F1"], "name_raw": ["Impact Technologies LLC", "Impact Tech, LLC"]}))
    firms = pd.Series(["F1", "F1", "F1", "F1"])
    recips = pd.Series(["IMPACT TECHNOLOGIES, L.L.C.", "SIKORSKY AIRCRAFT CORPORATION", None, "IMPACT TECHNOLOGY LLC"])
    assert recipient_matches_firm(firms, recips, keys).tolist() == [True, False, True, True]
