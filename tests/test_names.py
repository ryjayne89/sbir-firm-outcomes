import pandas as pd
import pytest

from sbirfo import names as nm
from sbirfo.validate import wilson


@pytest.mark.parametrize("raw, key", [
    ("Acme Technologies, Inc.", "acme technologies"),
    ("ACME TECHNOLOGIES INC", "acme technologies"),
    ("Foo Bar, L.L.C.", "foo bar"),
    ("The Boeing Company", "boeing"),
    ("Widget Co. Packaging Corp", "widget co packaging"),
    ("Nanoco Inc., a Delaware corporation", "nanoco"),
    ("Blue Sky Labs dba SkyBio", "blue sky labs"),
    ("DBA Systems Inc", "dba systems"),
    ("Smith & Jones Holdings, LLC", "smith and jones holdings"),
    ("", None),
    (None, None),
])
def test_name_key(raw, key):
    assert nm.name_key(raw) == key


def test_loose_key_drops_generic_words():
    assert nm.loose_key("acme technologies") == "acme"
    assert nm.loose_key("technologies") is None


def test_generic_names_are_not_linkable():
    df = nm.add_keys(pd.DataFrame({"name_raw": ["Technologies Inc", "Acme Labs", "Luminex Optics"]}))
    assert df.linkable.tolist() == [False, True, True]


def test_clean_state_accepts_names_and_codes():
    s = nm.clean_state(pd.Series(["California", "tx", "New  York", "A6", None]))
    assert s.tolist()[:3] == ["CA", "TX", "NY"]
    assert s.isna().tolist()[3:] == [True, True]


def test_clean_zip():
    z = nm.clean_zip(pd.Series(["02110-1234", "2110", "78701.0", "abc"]))
    assert z.tolist()[:3] == ["02110", "02110", "78701"]
    assert pd.isna(z.iloc[3])


def test_star_edges_and_components():
    df = pd.DataFrame({"node_id": ["a", "b", "c", "d"], "k": ["x", "x", "x", "y"]})
    edges = nm.star_edges(df, ["k"], "rule")
    assert len(edges) == 2
    labels = nm.components(df.node_id, edges)
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_wilson_interval_brackets_estimate():
    p, lo, hi = wilson(45, 50)
    assert lo < p < hi and 0 <= lo and hi <= 1
