"""Yield against the relief: the comparison, its refusals and its route.

The analytical properties are checked against fields whose answer is known
by construction — a yield made an exact function of the ground comes back
with a near-perfect correlation and bands that fall in order; a yield made
of noise comes back with nothing and has to say so — because a correlation
routine that always answers "0.6" passes every test written against real
data. The route tests drive it the way the interface does, through
:class:`fastapi.testclient.TestClient`, and check the shapes the tab relies
on and that every number survives JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset, DatasetMeta
from agrosuite.formats import raster
from agrosuite.terrain import analysis as analysis_mod
from agrosuite.terrain import yieldrelief
from agrosuite.terrain.synthetic import synthetic_dem, synthetic_terrain

TOP_KEYS = {
    "terrain", "values", "points", "overall", "elevation_bands", "slope_classes",
    "landforms", "wetness", "relations", "findings",
}
BAND_KEYS = {
    "index", "from_m", "to_m", "mean_elev_m", "points", "area_ha",
    "mean", "median", "p25", "p75", "sd", "delta_pct",
}


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(scope="module")
def quiet_field():
    """A noiseless synthetic field and its analysis, so a correlation the
    test builds on purpose is not blurred by the GPS noise the generator
    normally plants."""
    ds, truth = synthetic_terrain(noise_m=0.0, pass_offset_m=0.0)
    return ds, truth, analysis_mod.analyze(ds)


def _with_value(source: Dataset, value: np.ndarray, name: str = "Made-up yield") -> Dataset:
    """A copy of ``source`` carrying ``value`` in the value column."""
    df = source.df.copy()
    df[sch.VALUE] = np.asarray(value, dtype="float64")
    meta = DatasetMeta(
        name=name, source_path="<test>", source_format="test", operation="harvest",
        crop="canola", value_label="Yield", value_unit="kg/ha",
    )
    return Dataset(df, meta)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from agrosuite.app import server as server_mod

    return TestClient(server_mod.app)


@pytest.fixture(scope="module")
def demo(client):
    """The terrain demo loaded and analysed: one dataset carrying both the
    altitude and the yield, which is the case the tab opens on."""
    loaded = client.post("/api/terrain/demo")
    assert loaded.status_code == 200, loaded.text
    dataset_id = loaded.json()["id"]
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dataset_id})
    assert analysed.status_code == 200, analysed.text
    return dataset_id


# ==========================================================================
# The comparison itself
# ==========================================================================

def test_the_same_dataset_carries_both_layers(quiet_field):
    """The common case: a yield map whose own altitude was analysed."""
    ds, _truth, result = quiet_field
    summary = yieldrelief.analyse(
        result, ds, terrain_id="t", terrain_label="Terrain demo",
        values_id="t", values_label="Terrain demo",
    )
    assert set(summary) == TOP_KEYS
    assert summary["terrain"] == {"dataset_id": "t", "label": "Terrain demo"}
    assert summary["values"]["column"] == sch.VALUE
    assert summary["values"]["value_unit"] == "kg/ha"
    assert summary["values"]["cleaned"] is False
    assert summary["points"]["matched"] == summary["points"]["total"]
    assert summary["points"]["off_grid"] == 0
    assert summary["overall"]["points"] == summary["points"]["matched"]
    assert summary["overall"]["mean"] > 0 and summary["overall"]["cv_pct"] > 0
    # Nothing numpy, nothing NaN: the interface reads this straight.
    assert json.loads(json.dumps(summary)) == summary


def test_a_yield_that_follows_the_ground_comes_back_as_one(quiet_field):
    """Built as an exact function of the height, it must read as one: a
    correlation near -1, bands falling in order, and a finding saying the
    relief is most of the story."""
    ds, truth, result = quiet_field
    height = truth["surface"](ds.df[sch.X].to_numpy(), ds.df[sch.Y].to_numpy())
    # Against the field's own mean height, so the yields are plausible
    # numbers: 200 kg/ha a metre around 5 t/ha, not 5 000 minus 700 metres.
    exact = _with_value(ds, 5_000.0 - 200.0 * (height - height.mean()))

    summary = yieldrelief.analyse(result, exact, bands=6)
    assert summary["relations"]["elevation_r"] < -0.98
    assert summary["relations"]["elevation_r2"] > 0.96
    assert summary["relations"]["strongest"]["key"] == "elevation"

    means = [band["mean"] for band in summary["elevation_bands"]]
    assert len(means) == 6
    assert all(b > a for a, b in zip(means[1:], means[:-1])), means
    # Highest band lowest, lowest band highest, and both said so in percent.
    assert summary["elevation_bands"][0]["delta_pct"] > 5
    assert summary["elevation_bands"][-1]["delta_pct"] < -5
    text = " ".join(f["text"] for f in summary["findings"])
    assert "accounts for" in text and "the relief is most of the story" in text


def test_a_yield_of_pure_noise_says_the_relief_explains_little(quiet_field):
    """The other half of the same claim: made independent of the ground, it
    must come back near zero and the finding must read as weak."""
    ds, _truth, result = quiet_field
    rng = np.random.default_rng(11)
    noise = _with_value(ds, rng.normal(3_000.0, 300.0, len(ds.df)))

    summary = yieldrelief.analyse(result, noise)
    assert abs(summary["relations"]["elevation_r"]) < 0.05
    assert summary["relations"]["strongest"]["r2"] < 0.02
    text = " ".join(f["text"] for f in summary["findings"])
    assert "The relief explains little of this season's variation" in text
    # Every band is within a couple of percent of the field: no invented shape.
    assert max(abs(b["delta_pct"]) for b in summary["elevation_bands"]) < 3.0


def test_bands_are_equal_count_and_their_areas_add_up_to_the_field(quiet_field):
    ds, _truth, result = quiet_field
    summary = yieldrelief.analyse(result, ds, bands=5)
    bands = summary["elevation_bands"]
    assert len(bands) == 5
    assert all(set(band) == BAND_KEYS for band in bands)
    assert [band["index"] for band in bands] == [0, 1, 2, 3, 4]

    counts = [band["points"] for band in bands]
    assert sum(counts) == summary["points"]["matched"]
    # Equal-count bands: no band carries more than one point over a fifth.
    assert max(counts) - min(counts) <= 1

    # The edges run on without a gap, and the ranges are what the bands hold.
    for lower, upper in zip(bands, bands[1:]):
        assert lower["to_m"] == upper["from_m"]
        assert lower["from_m"] <= lower["mean_elev_m"] <= lower["to_m"]
    # The areas are the field's own ground, band by band, and add to it.
    assert sum(band["area_ha"] for band in bands) == pytest.approx(
        result.grid.area_ha(), rel=1e-9)


def test_the_groups_cover_the_matched_points(quiet_field):
    """Slope classes, landforms and the wet split each account for every
    point that fell on the field — a class silently dropping points would
    make its mean a different field's."""
    ds, _truth, result = quiet_field
    summary = yieldrelief.analyse(result, ds)
    matched = summary["points"]["matched"]

    assert sum(cls["points"] for cls in summary["slope_classes"]) == matched
    assert sum(cls["points"] for cls in summary["landforms"]) == matched
    assert sum(row["points"] for row in summary["wetness"]) == matched

    assert [cls["code"] for cls in summary["landforms"]] == [1, 2, 3, 4, 5, 6]
    assert all(cls["color"].startswith("#") for cls in summary["landforms"])
    assert [row["key"] for row in summary["wetness"]] == ["wet", "dry"]
    # An empty class is listed as empty rather than left out.
    assert all(
        cls["mean"] is None if cls["points"] == 0 else cls["mean"] > 0
        for cls in summary["slope_classes"]
    )


def test_a_layer_cleaned_elsewhere_is_not_warned_about(quiet_field):
    ds, _truth, result = quiet_field
    summary = yieldrelief.analyse(result, ds, cleaned=True)
    assert summary["values"]["cleaned"] is True
    assert not any("has not been cleaned" in f["text"] for f in summary["findings"])
    assert any("one season on one field" in f["text"] for f in summary["findings"])


# ==========================================================================
# Two files of one field
# ==========================================================================

def test_points_in_another_crs_are_reprojected_before_sampling(quiet_field):
    """Two files of one field can sit in different UTM zones. Sampling the
    grid with the other zone's eastings would put every point tens of
    kilometres away; reprojecting first must give exactly the same answer
    as a file already in the grid's CRS."""
    ds, _truth, result = quiet_field
    same = yieldrelief.analyse(result, ds, bands=4)

    shifted = Dataset(ds.df.copy(), ds.meta)
    shifted.project("EPSG:32613")
    assert shifted.metric_crs != result.grid.crs
    other = yieldrelief.analyse(result, shifted, bands=4)

    assert other["points"] == same["points"]
    np.testing.assert_allclose(
        [band["mean"] for band in other["elevation_bands"]],
        [band["mean"] for band in same["elevation_bands"]],
        rtol=1e-9,
    )
    assert other["relations"]["elevation_r"] == pytest.approx(
        same["relations"]["elevation_r"], rel=1e-9)


# ==========================================================================
# Refusals
# ==========================================================================

def test_a_layer_with_no_value_is_refused_by_name(quiet_field):
    ds, _truth, result = quiet_field
    bare = Dataset(ds.df.drop(columns=[sch.VALUE]), DatasetMeta(name="Boundary only"))
    with pytest.raises(ValueError, match="carries no value column"):
        yieldrelief.analyse(result, bare, values_label="Boundary only")


def test_an_unknown_column_lists_the_ones_there_are(quiet_field):
    ds, _truth, result = quiet_field
    with pytest.raises(ValueError, match="no column called 'protein'"):
        yieldrelief.analyse(result, ds, value_column="protein")


def test_a_field_somewhere_else_says_how_many_points_matched(quiet_field):
    """The usual reason for a comparison with nothing in it is two files of
    two different fields, and the message has to name that."""
    ds, _truth, result = quiet_field
    far = ds.df.copy()
    far[sch.LON] = far[sch.LON] + 1.0
    elsewhere = Dataset(far, ds.meta)
    with pytest.raises(ValueError, match="fall on the analysed field"):
        yieldrelief.analyse(result, elsewhere)
    try:
        yieldrelief.analyse(result, elsewhere)
    except ValueError as exc:
        assert f"of {len(far)} points" in str(exc)


def test_the_zone_layer_of_this_relief_is_refused(quiet_field):
    ds, _truth, result = quiet_field
    zones = analysis_mod.zones_points(result, "landform")
    with pytest.raises(ValueError, match="zone codes, not a measurement"):
        yieldrelief.analyse(result, zones, values_label=zones.meta.name)


def test_the_band_count_is_bounded(quiet_field):
    ds, _truth, result = quiet_field
    for bands in (1, 50):
        with pytest.raises(ValueError, match="elevation bands"):
            yieldrelief.analyse(result, ds, bands=bands)


# ==========================================================================
# The profile corridor
# ==========================================================================

def test_the_profile_values_line_up_and_break_off_the_corridor(quiet_field):
    """A value layer covering half the field leaves the other half's
    stations empty, so the chart breaks rather than drawing a line over
    ground that was never harvested."""
    from agrosuite.terrain.contours import profile as profile_along

    ds, truth, result = quiet_field
    ox, oy = truth["origin_m"]
    half = ds.df[ds.df[sch.Y] < oy + 0.5 * truth["size_m"]].copy()
    southern = Dataset(half, ds.meta)

    lon0, lat0 = truth["to_lonlat"](ox + 0.5 * truth["size_m"], oy + 0.08 * truth["size_m"])
    lon1, lat1 = truth["to_lonlat"](ox + 0.5 * truth["size_m"], oy + 0.92 * truth["size_m"])
    line = profile_along(result.grid, [[float(lon0), float(lat0)], [float(lon1), float(lat1)]], n=40)

    series, meta = yieldrelief.profile_values(result.grid, line, southern)
    assert len(series) == len(line["distance_m"])
    assert meta["column"] == sch.VALUE and meta["value_unit"] == "kg/ha"
    assert meta["corridor_m"] == pytest.approx(max(2 * result.grid.cell, 4.5))
    assert meta["points_used"] > 0

    measured = [i for i, v in enumerate(series) if v is not None]
    assert measured, series
    # The line runs south to north and the layer stops half way: the first
    # stations carry a value and the last ones do not.
    assert series[0] is not None and series[-1] is None
    assert measured == list(range(measured[0], measured[-1] + 1))
    assert all(v is None for v in series[measured[-1] + 1:])
    assert all(1_000.0 < v < 6_000.0 for v in series if v is not None)
    assert json.loads(json.dumps(series)) == series


def test_the_whole_field_fills_every_station(quiet_field):
    from agrosuite.terrain.contours import profile as profile_along

    ds, truth, result = quiet_field
    ox, oy = truth["origin_m"]
    lon0, lat0 = truth["to_lonlat"](ox + 0.4 * truth["size_m"], oy + 0.1 * truth["size_m"])
    lon1, lat1 = truth["to_lonlat"](ox + 0.6 * truth["size_m"], oy + 0.9 * truth["size_m"])
    line = profile_along(result.grid, [[float(lon0), float(lat0)], [float(lon1), float(lat1)]], n=25)
    series, _meta = yieldrelief.profile_values(result.grid, line, ds)
    assert all(v is not None for v in series)


# ==========================================================================
# The route
# ==========================================================================

def test_the_route_answers_the_same_dataset_case(client, demo):
    response = client.post(f"/api/terrain/{demo}/yield", json={"yield_dataset_id": demo})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == TOP_KEYS
    assert body["terrain"]["dataset_id"] == demo
    assert body["values"]["dataset_id"] == demo
    assert body["values"]["label"] == "Terrain demo"
    assert body["points"]["matched"] > 1_000
    assert len(body["elevation_bands"]) == yieldrelief.DEFAULT_BANDS
    assert body["findings"] and all(
        f["level"] in {"ok", "info", "warning"} for f in body["findings"])
    # The demo's yield follows its relief, so the tab has something to draw.
    assert body["relations"]["elevation_r"] < -0.3
    json.dumps(body)

    # It is kept with the analysed dataset, which is what the printed page reads.
    stored = client.get(f"/api/datasets/{demo}/report/terrain_yield")
    assert stored.status_code == 200, stored.text
    assert stored.json()["values"]["dataset_id"] == demo


def test_the_route_takes_a_band_count_and_a_column(client, demo):
    response = client.post(
        f"/api/terrain/{demo}/yield",
        json={"yield_dataset_id": demo, "bands": 4, "value_column": "value"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["elevation_bands"]) == 4


def test_the_route_refuses_a_typo_rather_than_defaulting(client, demo):
    response = client.post(
        f"/api/terrain/{demo}/yield", json={"yield_dataset_id": demo, "band": 4})
    assert response.status_code == 422


def test_the_route_asks_for_the_analysis_first(client):
    loaded = client.post("/api/terrain/demo")
    dataset_id = loaded.json()["id"]
    response = client.post(
        f"/api/terrain/{dataset_id}/yield", json={"yield_dataset_id": dataset_id})
    assert response.status_code == 404
    assert "Run the terrain analysis first" in response.json()["detail"]


def test_the_route_names_a_dataset_that_is_not_loaded(client, demo):
    response = client.post(f"/api/terrain/{demo}/yield", json={"yield_dataset_id": "nope"})
    assert response.status_code == 404
    assert "not loaded" in response.json()["detail"]


def test_the_route_refuses_two_different_fields(client, demo):
    """The demo harvest map sits in Brazil and the terrain demo in Alberta:
    nothing of one falls on the other, and the message says so."""
    loaded = client.post("/api/import/demo", json={"kind": "harvest"})
    assert loaded.status_code == 200, loaded.text
    elsewhere = loaded.json()["id"]
    assert elsewhere != demo
    response = client.post(
        f"/api/terrain/{demo}/yield", json={"yield_dataset_id": elsewhere})
    assert response.status_code == 400
    assert "fall on the analysed field" in response.json()["detail"]


def test_a_dem_analysed_with_a_yield_map_beside_it(client, tmp_path):
    """The two-file case end to end: a DEM GeoTIFF is analysed, the yield
    map of the same field is loaded separately, and the comparison samples
    the raster's grid under the monitor's points."""
    dem, _truth = synthetic_dem(cell_m=5.0)
    path = raster.write_geotiff(dem, tmp_path / "field_dem.tif", dtype="float64")
    loaded = client.post("/api/import/path", json={"path": str(path)})
    assert loaded.status_code == 200, loaded.text
    dem_id = loaded.json()["id"]
    analysed = client.post("/api/terrain/analyze", json={"dataset_id": dem_id})
    assert analysed.status_code == 200, analysed.text

    yield_id = client.post("/api/terrain/demo").json()["id"]
    response = client.post(
        f"/api/terrain/{dem_id}/yield", json={"yield_dataset_id": yield_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["terrain"]["dataset_id"] == dem_id
    assert body["values"]["dataset_id"] == yield_id
    assert body["values"]["label"] == "Terrain demo"
    # The yield map covers the DEM's field, so nearly every point lands.
    assert body["points"]["matched"] > 0.95 * body["points"]["total"]
    assert body["relations"]["elevation_r"] < -0.3
    assert len(body["elevation_bands"]) == yieldrelief.DEFAULT_BANDS
    json.dumps(body)


def test_the_route_refuses_the_zone_layer(client, demo):
    zones = client.post(f"/api/terrain/{demo}/zones", json={"by": "landform"})
    assert zones.status_code == 200, zones.text
    zone_id = zones.json()["dataset"]["id"]
    response = client.post(f"/api/terrain/{demo}/yield", json={"yield_dataset_id": zone_id})
    assert response.status_code == 400
    assert "zone codes, not a measurement" in response.json()["detail"]


# ==========================================================================
# The profile route
# ==========================================================================

PROFILE_LINE = [[-113.5535987, 51.7477414], [-113.546363, 51.7522586]]


def test_the_profile_keeps_its_old_shape_without_a_value_layer(client, demo):
    response = client.post(
        f"/api/terrain/{demo}/profile", json={"points": PROFILE_LINE, "n": 20})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"distance_m", "elev_m", "slope_pct", "points", "length_m"}


def test_the_profile_carries_the_value_when_one_is_asked_for(client, demo):
    response = client.post(f"/api/terrain/{demo}/profile", json={
        "points": PROFILE_LINE, "n": 20, "values_dataset_id": demo,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["values"]) == len(body["distance_m"])
    meta = body["values_meta"]
    assert meta["value_label"] == "Yield" and meta["value_unit"] == "kg/ha"
    assert meta["operation"] == "harvest" and meta["crop"] == "canola"
    assert meta["corridor_m"] > 0 and meta["points_used"] > 0
    assert any(v is not None for v in body["values"])
    json.dumps(body)


def test_the_profile_refuses_a_value_layer_with_no_value(client, demo, tmp_path):
    dem, _truth = synthetic_dem(cell_m=20.0)
    path = raster.write_geotiff(dem, tmp_path / "beside.tif", dtype="float64")
    dem_id = client.post("/api/import/path", json={"path": str(path)}).json()["id"]
    response = client.post(f"/api/terrain/{demo}/profile", json={
        "points": PROFILE_LINE, "n": 10,
        "values_dataset_id": dem_id, "value_column": "protein",
    })
    assert response.status_code == 400
    assert "no column called 'protein'" in response.json()["detail"]


# ==========================================================================
# The sentences, in the reader's own units
# ==========================================================================

CANADA = {
    "yield_unit": "bu/ac", "input_rate_unit": "lb/ac", "area_unit": "ac",
    "length_unit": "ft", "speed_unit": "mph", "currency": "CAD", "crop": "canola",
}


def test_the_findings_are_written_in_the_unit_set_they_are_given(quiet_field):
    """The sentence is the finding: it cannot be restated in another unit at
    display time without re-deciding what it says, so the unit set travels
    into the writing of it."""
    ds, _truth, result = quiet_field
    metric = yieldrelief.analyse(result, ds, values_label="Terrain demo")
    canadian = yieldrelief.analyse(result, ds, values_label="Terrain demo", units=CANADA)

    metric_text = " ".join(f["text"] for f in metric["findings"])
    canadian_text = " ".join(f["text"] for f in canadian["findings"])
    assert "kg/ha" in metric_text and " m " in metric_text and " ha " in metric_text
    assert "bu/ac" in canadian_text and " ft" in canadian_text and " ac " in canadian_text
    assert "kg/ha" not in canadian_text and " ha " not in canadian_text

    # Only the prose changes: every stored number stays metric, or a saved
    # report would depend on a screen preference.
    assert metric["overall"] == canadian["overall"]
    assert metric["elevation_bands"] == canadian["elevation_bands"]


def test_an_application_layer_is_phrased_as_a_rate_not_a_yield(quiet_field):
    """lb/ac of fertilizer written as bu/ac of grain is a gross error, so the
    operation decides the unit, not the caller."""
    ds, _truth, result = quiet_field
    applied = _with_value(ds, ds.df[sch.VALUE].to_numpy() / 20.0, name="As-applied")
    applied.meta.operation = "application"
    applied.meta.value_label = "Rate"

    summary = yieldrelief.analyse(result, applied, units=CANADA)
    text = " ".join(f["text"] for f in summary["findings"])
    assert "lb/ac" in text and "bu/ac" not in text


def test_a_unit_that_does_not_exist_is_refused_by_name(quiet_field):
    ds, _truth, result = quiet_field
    with pytest.raises(ValueError, match="not a valid length unit"):
        yieldrelief.analyse(result, ds, units={"length_unit": "furlong"})


def test_the_route_takes_the_unit_set_and_the_profile_note_follows_it(client, demo):
    response = client.post(f"/api/terrain/{demo}/yield", json={
        "yield_dataset_id": demo, "units": CANADA,
    })
    assert response.status_code == 200, response.text
    text = " ".join(f["text"] for f in response.json()["findings"])
    assert "bu/ac" in text and "kg/ha" not in text
    # The numbers themselves are untouched: the interface converts those.
    assert response.json()["overall"]["mean"] > 1_000

    line = client.post(f"/api/terrain/{demo}/profile", json={
        "points": PROFILE_LINE, "n": 12,
        "values_dataset_id": demo, "units": CANADA,
    })
    assert line.status_code == 200, line.text
    note = line.json()["values_meta"]["note"]
    assert "ft of the line" in note and " m of the line" not in note


def test_the_printed_page_writes_the_findings_in_its_own_units(client, demo, tmp_path):
    """A report asked for in acres must not carry sentences in hectares,
    whatever units were on screen when Compare was pressed."""
    from agrosuite import report as report_mod
    from agrosuite.app import server as server_mod

    client.post(f"/api/terrain/{demo}/yield", json={"yield_dataset_id": demo})
    entry = server_mod.state.get(demo)
    assert "kg/ha" in entry.reports["terrain_yield"]["findings"][0]["text"]

    out = report_mod.build_pdf(
        entry, tmp_path / "page.pdf", CANADA, "2026-09-17", {"name": "Demo"})
    assert "terrain_yield" in out["sections"]

    written = yieldrelief.findings(entry.reports["terrain_yield"], CANADA)
    assert "bu/ac" in written[0]["text"] and "kg/ha" not in written[0]["text"]
