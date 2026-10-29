import pytest

from researchreadernerd import catalogs
from researchreadernerd.catalogs import _UcrDataset, search_tpc, search_ucr


@pytest.fixture(autouse=True)
def fake_ucr(monkeypatch):
    rows = (
        _UcrDataset("Adiac", "Image", "Image time series", 781),
        _UcrDataset("ArrowHead", "Image", "Image time series", 211),
        _UcrDataset("ECG200", "ECG", "ECG time series", 200),
        _UcrDataset("ECG5000", "ECG", "ECG time series", 5000),
        _UcrDataset("TwoLeadECG", "ECG", "ECG time series", 1162),
        _UcrDataset("Wafer", "Sensor", "Sensor time series", 7164),
        _UcrDataset("Car", "Sensor", "Sensor time series", 120),
        _UcrDataset("UWaveGestureLibraryX", "Motion", "Motion time series", 4478),
    )
    monkeypatch.setattr(catalogs, "_ucr_rows", lambda: rows)


def names(hits):
    return [h.name for h in hits]


def test_generic_time_series_words_match_nothing():
    # Previously this matched every dataset and returned the first ones alphabetically.
    assert search_ucr("time series classification") == []
    assert search_ucr("univariate time series archive benchmark") == []


def test_type_matches_ranked_by_size():
    assert names(search_ucr("sensor time series", 5)) == ["Wafer", "Car"]


def test_name_matches_beat_type_matches():
    # All three ECG datasets match on type; the ones with "ecg" in the name also match on name.
    assert names(search_ucr("ECG heartbeat", 5)) == ["ECG5000", "TwoLeadECG", "ECG200"]


def test_synonyms_point_to_types():
    assert names(search_ucr("accelerometer recognition")) == ["UWaveGestureLibraryX"]
    assert names(search_ucr("cardiac monitoring")) == ["ECG5000", "TwoLeadECG", "ECG200"]


def test_links_point_to_dataset_pages():
    hit = search_ucr("ECG")[0]
    assert hit.url == "https://www.timeseriesclassification.com/description.php?Dataset=ECG5000"


def test_tpc_matches_on_description():
    assert names(search_tpc("decision support queries")) == ["TPC-H", "TPC-DS"]
    assert search_tpc("image classification") == []
