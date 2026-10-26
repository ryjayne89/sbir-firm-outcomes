import pandas as pd

from sbirfo import places
from sbirfo.config import Paths


class Down:
    def get(self, *a, **k):
        raise ConnectionError("down")


def test_cached_acs_used_when_api_down(tmp_path):
    paths = Paths(tmp_path)
    paths.make()
    cache = paths.interim / "places"
    cache.mkdir(parents=True, exist_ok=True)
    row = {"zip5": "78701", "acs_year": 2012, "pop": 100, "pop25": 50, "ba": 10, "ma": 5, "prof": 1, "phd": 1,
           "civ_labor_force": 60, "unemployed": 3, "median_hh_income": 1, "median_home_value": 1, "median_age": 30}
    pd.DataFrame([row]).to_parquet(cache / "acs5_zcta_2012.parquet")
    z = places.acs(paths, Down(), None, fetch=False)
    assert len(z) == 1 and abs(z.ba_plus_share.iloc[0] - 17 / 50) < 1e-9


def test_non_text_key_ignored(monkeypatch):
    calls = []
    monkeypatch.setattr(places, "_get", lambda web, url, params, **k: calls.append(params) or [[1]])
    assert places._check_api(None, {"oops": 1}) is None
    assert all("key" not in p for p in calls)
