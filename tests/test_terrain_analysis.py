"""Tests for the terrain analysis layer: landform classes, discrete features,
the summary, the findings and the zone datasets.

The synthetic field has a known surface, so the tests ask whether the
analysis finds what was put there — two hills at their summits, the trough,
the closed bowl with its volume, the general fall — within tolerances that
allow for the GPS noise, the smoothing and the grid. The truth's own numbers
come from the surface function, not from the analysis, so the comparison is
independent wherever the definition allows it.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset
from agrosuite.formats import raster, registry
from agrosuite.terrain import ElevationGrid, synthetic_dem, synthetic_terrain
from agrosuite.terrain import landforms as lf
from agrosuite.terrain.analysis import (
    LAYER_SPECS,
    TerrainOptions,
    TerrainResult,
    analyze,
    findings,
    zones_points,
    zones_polygons,
)

SUMMARY_KEYS = {
    "source", "grid", "elevation", "slope", "aspect", "trend", "character",
    "landforms", "features", "wetness", "contours", "layers", "findings", "options",
}
LAYER_KEYS = [spec[0] for spec in LAYER_SPECS]


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture(scope="module")
def default_case():
    ds, truth = synthetic_terrain()
    return analyze(ds), truth, ds


@pytest.fixture(scope="module")
def noiseless_case():
    ds, truth = synthetic_terrain(noise_m=0.0, pass_offset_m=0.0)
    return analyze(ds), truth, ds


def _truth_prominence(result: TerrainResult, truth: dict, hill: dict) -> float:
    """Height of a detected hill on the TRUE relief, with the module's own
    definition: top of the patch minus the level of its foot, the foot
    being the ring walk's level or the saddle to the other listed
    features, whichever is higher."""
    grid = result.grid
    X, Y = grid.xy()
    relief = truth["relief"](X, Y)
    relief[~grid.mask] = np.nan
    comp = result.labels["hills"] == hill["id"]
    others = ((result.labels["hills"] > 0) & ~comp) | (result.labels["lows"] > 0)
    foot = lf.foot_level(relief, comp, 1.0)
    saddle = lf.saddle_level(relief, comp, others, 1.0)
    if np.isfinite(saddle):
        foot = max(foot, saddle)
    return float(np.nanmax(relief[comp]) - foot)


def _walk_for_numpy(value, path="summary"):
    """Fail on the first numpy scalar or NaN hiding in a JSON summary."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"{path}: key {k!r} is not a str"
            _walk_for_numpy(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _walk_for_numpy(v, f"{path}[{i}]")
    else:
        assert not isinstance(value, np.generic), f"{path}: numpy type {type(value)}"
        assert value is None or isinstance(value, (str, int, float, bool)), f"{path}: {type(value)}"
        if isinstance(value, float):
            assert math.isfinite(value), f"{path}: non-finite float"


def _truth_character(result: TerrainResult, truth: dict) -> str:
    """What the character should be, from the true surface's own slope."""
    grid = result.grid
    X, Y = grid.xy()
    z = truth["surface"](X, Y)
    inside = truth["inside"](X, Y)
    gy, gx = np.gradient(z, grid.cell)
    slope_pct = 100.0 * np.hypot(gx, -gy)[inside]
    relief = float(z[inside].max() - z[inside].min())
    return lf.character(relief, float(slope_pct.mean()), float(np.percentile(slope_pct, 95)))["key"]


# ==========================================================================
# Features against the truth
# ==========================================================================

@pytest.mark.parametrize("case", ["default_case", "noiseless_case"])
def test_two_hills_at_their_summits(case, request):
    result, truth, _ = request.getfixturevalue(case)
    hills = result.features["hills"]
    assert len(hills) == 2, [h["label"] + " " + h["position"] for h in hills]
    for hill in hills:
        dist = min(math.hypot(hill["x"] - t["summit_m"][0], hill["y"] - t["summit_m"][1])
                   for t in truth["hills"])
        assert dist <= 25.0
        prominence = _truth_prominence(result, truth, hill)
        assert abs(hill["height_m"] - prominence) <= 0.30 * prominence
        assert hill["area_ha"] > 0.5
        assert hill["mean_slope_pct"] > 0
    # The bigger hill is Hill 1 and lies where it was put: the south-west.
    assert hills[0]["height_m"] > hills[1]["height_m"]
    assert hills[0]["position"] == "south-west part"
    assert hills[1]["position"] == "eastern part"


def test_closed_depression_matches_the_bowl(default_case):
    """The demo's bowl ponds 0.19 m on its tilted floor, under the 0.59 m
    feature floor the noisy demo file earns, so by default it is counted
    among the unlisted hollows; asked to list features down to 0.1 m, the
    analysis finds it where it was put, at the depth and volume of the
    truth."""
    default, truth, ds = default_case
    bottom = truth["depression"]["bottom_m"]
    assert default.summary()["features"]["depressions_unlisted"] >= 1
    assert not [d for d in default.features["depressions"]
                if math.hypot(d["x"] - bottom[0], d["y"] - bottom[1]) <= 25.0]
    result = analyze(ds, TerrainOptions(min_feature_height_m=0.1))
    assert result.summary()["features"]["depression_floor_m"] == pytest.approx(0.1)
    near = [d for d in result.features["depressions"]
            if math.hypot(d["x"] - bottom[0], d["y"] - bottom[1]) <= 25.0]
    assert len(near) == 1
    dep = near[0]
    assert abs(dep["max_depth_m"] - truth["depression"]["closed_depth_m"]) <= 0.3
    truth_volume = truth["depression"]["volume_m3"]
    assert abs(dep["volume_m3"] - truth_volume) <= 0.35 * truth_volume
    assert dep["label"].startswith("Depression ")
    assert dep["position"] == "southern part"
    assert any("m³" in f["text"] for f in result.summary()["findings"])


def test_lows_include_the_valley_trough(default_case):
    result, truth, _ = default_case
    lows = result.features["lows"]
    assert lows, "no low found at all"
    axis = truth["valley"]["axis_m"]
    row, col = result.grid.rowcol(axis[:, 0], axis[:, 1])
    r = np.rint(row).astype(int)
    c = np.rint(col).astype(int)
    ok = (r >= 0) & (r < result.grid.rows) & (c >= 0) & (c < result.grid.cols)
    in_low = result.labels["lows"][r[ok], c[ok]] > 0
    # The trough runs through hills that dam it, so not every axis point is
    # below its surroundings; most of the axis is.
    assert in_low.mean() >= 0.5, in_low.mean()
    valleys = [l for l in lows if l["elongation"] >= 2.5]
    assert valleys and valleys[0]["closed"] is True  # dammed by the hills


def test_trend_matches_the_tilt(default_case):
    result, truth, _ = default_case
    trend = result.summary()["trend"]
    delta = abs((trend["direction_deg"] - truth["tilt"]["direction_deg"] + 180.0) % 360.0 - 180.0)
    assert delta <= 15.0
    true_gradient = 100.0 * truth["tilt"]["gradient"]
    assert abs(trend["gradient_pct"] - true_gradient) <= 0.20 * true_gradient
    assert trend["direction_label"] == "south-west"
    assert 0.3 < trend["r2"] <= 1.0
    assert trend["drop_m"] > 0.5 * truth["tilt"]["fall_m"]


def test_character_agrees_with_the_truth(default_case):
    result, truth, _ = default_case
    expected = _truth_character(result, truth)
    assert expected in ("gently_undulating", "rolling")
    char = result.summary()["character"]
    assert char["key"] == expected
    assert char["label"] in char["why"] or "%" in char["why"]


# ==========================================================================
# Shares, layers, contours
# ==========================================================================

def test_class_and_sector_shares_add_up(default_case):
    result, _, _ = default_case
    s = result.summary()
    area = s["grid"]["area_ha"]
    landform_area = sum(c["area_ha"] for c in s["landforms"]["classes"])
    assert abs(landform_area - area) <= 0.005 * area
    assert [c["code"] for c in s["landforms"]["classes"]] == [1, 2, 3, 4, 5, 6]
    assert all(c["color"].startswith("#") for c in s["landforms"]["classes"])
    assert abs(sum(c["pct"] for c in s["slope"]["classes"]) - 100.0) <= 0.5
    assert [c["key"] for c in s["slope"]["classes"]] == ["flat", "gentle", "moderate", "strong", "steep"]
    assert abs(sum(c["pct"] for c in s["aspect"]["sectors"]) - 100.0) <= 0.5
    assert [c["key"] for c in s["aspect"]["sectors"]] == ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "flat"]
    # The field falls to the south-west, so most of it faces that way.
    by_key = {c["key"]: c["pct"] for c in s["aspect"]["sectors"]}
    assert max(by_key, key=by_key.get) in ("SW", "S", "W")


def test_layers_and_contours(default_case):
    result, _, _ = default_case
    s = result.summary()
    assert [l["key"] for l in s["layers"]] == LAYER_KEYS
    assert set(result.layers) == set(LAYER_KEYS)
    for key, arr in result.layers.items():
        assert arr.shape == result.grid.shape, key
        if key != "aspect_deg":  # aspect is NaN on flat cells by design
            assert np.isfinite(arr[result.grid.mask]).all(), key
        assert np.isnan(arr[~result.grid.mask]).all(), key
    meta = {l["key"]: l for l in s["layers"]}
    assert meta["landform"]["kind"] == "categorical"
    assert meta["slope_pct"]["unit"] == "%" and meta["elevation"]["unit"] == "m"
    assert meta["landform"]["min"] >= 1 and meta["landform"]["max"] <= 6
    assert s["contours"]["count"] == len(result.contours["features"]) > 0
    assert s["contours"]["interval_m"] == result.contour_interval_m
    assert result.drainage["type"] == "FeatureCollection"
    assert all(f["properties"]["length_m"] > 0 for f in result.drainage["features"])
    assert s["wetness"]["drainage_length_m"] > 0


# ==========================================================================
# Summary and findings
# ==========================================================================

def test_summary_is_plain_json(default_case):
    result, _, _ = default_case
    s = result.summary()
    assert set(s) == SUMMARY_KEYS
    _walk_for_numpy(s)
    json.dumps(s)
    assert s["source"]["kind"] == "points"
    assert s["source"]["points_total"] == 33400
    assert s["source"]["detrended"] is True
    assert s["source"]["dem_path"] is None
    assert s["elevation"]["relief_m"] > 8
    assert len(s["elevation"]["histogram"]["edges"]) == len(s["elevation"]["histogram"]["counts"]) + 1
    assert s["features"]["hills"][0]["label"] == "Hill 1"
    assert result.summary() is s  # cached


def test_findings_read_like_a_report(default_case):
    result, _, _ = default_case
    found = result.summary()["findings"]
    assert found == findings(result)
    assert all(f["level"] in ("ok", "info", "warning") for f in found)
    texts = [f["text"] for f in found]
    joined = "\n".join(texts)
    assert "Hill 1" in joined
    assert "depression" in joined
    assert "south-west" in joined
    assert texts[0].startswith("Total relief is ")
    assert "gently undulating" in texts[0]
    assert "falls about" in texts[1] and "north-east to the south-west" in texts[1]
    # The demo's basins pond under the feature floor: counted, not listed.
    assert not any("m³" in t for t in texts)
    assert any("shallow hollows within the elevation noise" in t and "not listed" in t
               for t in texts)
    assert any("GPS elevation noise" in t for t in texts)
    # The synthetic receiver drifts 0.4 m between passes: no RTK.
    assert any(f["level"] == "warning" and "RTK" in f["text"] for f in found)
    # Nothing in the text is a bare formatted float with many decimals.
    assert not any(".0000" in t or "e-" in t for t in texts)


def test_feature_collection_polygons(default_case):
    result, _, _ = default_case
    fc = result.feature_collection()
    json.dumps(fc)
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    assert kinds.count("hill") == 2
    assert kinds.count("depression") == len(result.features["depressions"])
    assert kinds.count("low") == len(result.features["lows"])
    from shapely.geometry import Point, shape

    for feature in fc["features"]:
        geom = shape(feature["geometry"])
        assert geom.is_valid and geom.area > 0
        props = feature["properties"]
        if props["kind"] == "hill":
            assert geom.buffer(1e-6).contains(Point(props["lon"], props["lat"]))
            assert props["label"] == f"Hill {props['id']}"


# ==========================================================================
# Zones
# ==========================================================================

@pytest.mark.parametrize("by", ["landform", "slope_class", "elevation_bands", "wetness"])
def test_zone_datasets(default_case, by):
    result, _, ds = default_case
    area = result.grid.area_ha()
    pts = zones_points(result, by)
    assert isinstance(pts, Dataset)
    assert len(pts.df) == result.grid.valid_count
    for col in (sch.LON, sch.LAT, sch.VALUE, "zone", "zone_label", sch.ELEVATION,
                "slope_pct", "twi", "landform", "aspect_deg", sch.X, sch.Y):
        assert col in pts.df.columns, col
    assert pts.metric_crs == result.grid.crs
    assert pts.meta.operation == "elevation"
    assert pts.meta.value_label == "Terrain zone"
    assert pts.meta.extra["zones_by"] == by
    assert pts.meta.extra["terrain_source"] == ds.meta.name
    labels = pts.meta.extra["zone_labels"]
    assert set(pts.df["zone"].unique()) <= set(labels)
    assert (pts.df[sch.VALUE] == pts.df["zone"]).all()
    json.dumps(pts.summary())

    polys = zones_polygons(result, by)
    assert polys.meta.geometry_type == "polygon"
    assert polys.geometry is not None and len(polys.geometry) == len(polys.df)
    assert abs(polys.df["area_ha"].sum() - area) <= 0.02 * area
    for col in ("zone", "zone_label", sch.VALUE, "area_ha", "mean_elev_m", "mean_slope_pct"):
        assert col in polys.df.columns, col
    gdf = polys.to_geodataframe(metric=True)
    assert gdf.geometry.is_valid.all()
    # The dissolved outlines cover the field: their area is the grid area.
    assert abs(gdf.geometry.area.sum() / 10_000.0 - area) <= 0.02 * area
    if by == "elevation_bands":
        assert list(polys.df["zone_label"])[0].startswith("Low (")
        assert list(polys.df["zone_label"])[-1].startswith("High (")
    if by == "wetness":
        assert set(polys.df["zone_label"]) == {"Well drained", "Likely wet"}


def test_zones_reject_an_unknown_kind(default_case):
    result, _, _ = default_case
    with pytest.raises(ValueError, match="landform"):
        zones_points(result, "soil_type")


# ==========================================================================
# Options, DEM source, timing
# ==========================================================================

def test_options_are_honoured_and_echoed():
    ds, _ = synthetic_terrain()
    options = TerrainOptions(
        cell_m=8.0, tpi_small_m=40.0, tpi_large_m=120.0, contour_interval_m=0.5,
        min_feature_height_m=1.0, hillshade_azimuth=135.0, min_upstream_ha=2.0,
    )
    result = analyze(ds, options)
    assert result.grid.cell == 8.0
    echoed = result.summary()["options"]
    assert echoed["cell_m"] == 8.0
    assert echoed["tpi_small_m"] == 40.0 and echoed["tpi_large_m"] == 120.0
    assert echoed["contour_interval_m"] == 0.5
    assert echoed["min_feature_height_m"] == 1.0
    assert echoed["hillshade_azimuth"] == 135.0
    assert echoed["min_upstream_ha"] == 2.0
    assert result.summary()["contours"]["interval_m"] == 0.5
    assert all(f["properties"]["level_m"] % 0.5 == 0 for f in result.contours["features"])
    # The 1 m threshold still keeps both hills (2.6 m and 1.5 m).
    assert len(result.features["hills"]) == 2

    # Defaults are resolved from the grid and echoed too.
    default = analyze(ds).summary()["options"]
    assert default["cell_m"] == 5.0
    assert default["tpi_small_m"] == 30.0 and default["tpi_large_m"] == 150.0
    assert default["min_feature_height_m"] == pytest.approx(
        max(0.5, 2.0 * analyze(ds).source_report["vertical_noise_m"])
    )
    assert TerrainOptions.from_dict({"cell_m": 6}).cell_m == 6
    with pytest.raises(ValueError, match="Unknown terrain option"):
        TerrainOptions.from_dict({"cellsize": 6})
    with pytest.raises(ValueError, match="cell size"):
        analyze(ds, TerrainOptions(cell_m=-1))


def test_dem_backed_dataset_is_analysed_from_the_raster(tmp_path):
    grid, truth = synthetic_dem()
    path = raster.write_geotiff(grid, tmp_path / "field_dem.tif", dtype="float64")
    ds = registry.read_any(path)
    assert ds.meta.extra["dem_path"] == str(path)
    assert len(ds.df) <= grid.valid_count  # the points are (a sample of) the cells
    result = analyze(ds)
    s = result.summary()
    assert s["source"]["kind"] == "dem"
    assert s["source"]["dem_path"] == str(path)
    assert s["source"]["points_used"] == grid.valid_count
    assert result.grid.shape == grid.shape and result.grid.cell == grid.cell
    assert result.grid.crs == grid.crs
    assert len(result.features["hills"]) == 2
    for hill in result.features["hills"]:
        dist = min(math.hypot(hill["x"] - t["summit_m"][0], hill["y"] - t["summit_m"][1])
                   for t in truth["hills"])
        assert dist <= 25.0
    assert any("elevation raster" in f["text"] for f in s["findings"])
    assert not any("GPS elevation noise" in f["text"] for f in s["findings"])
    _walk_for_numpy(s)
    # A cell request resamples the raster rather than gridding the points.
    coarse = analyze(ds, TerrainOptions(cell_m=10.0))
    assert coarse.grid.cell == 10.0
    assert coarse.summary()["source"]["kind"] == "dem"


def test_flat_plane_reports_nothing_alarming():
    grid, _ = synthetic_dem(hills=(), valley=None, depression=None, notch=None)
    from agrosuite.formats.raster import dataset_from_dem

    ds = dataset_from_dem(grid, {"path": "", "notes": []})
    result = analyze(ds)  # no dem_path on disk: gridded from the sampled points
    s = result.summary()
    assert s["source"]["kind"] == "points"
    assert result.features["hills"] == [] and result.features["depressions"] == []
    assert s["trend"]["direction_label"] == "south-west"
    assert abs(s["trend"]["gradient_pct"] - 0.8) < 0.1
    levels = [f["level"] for f in s["findings"]]
    assert levels[-1] == "ok"
    assert "No closed depressions or steep ground" in s["findings"][-1]["text"]
    assert any("No distinct hills or hollows" in f["text"] for f in s["findings"])


def test_flat_dem_reports_no_depression_and_no_drainage(tmp_path):
    """A dead-flat raster (a filled pond, a hydro-flattened lake, a plateau
    quantised to whole metres) must not be read as a basin that ponds
    water, nor drain to one corner."""
    n = 300
    grid = ElevationGrid(np.full((n, n), 700.0), 500_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    path = raster.write_geotiff(grid, tmp_path / "flat.tif", dtype="float64")
    s = analyze(registry.read_any(path)).summary()
    assert s["source"]["kind"] == "dem"
    assert s["features"]["depressions"] == [] and s["features"]["lows"] == []
    assert s["wetness"]["drainage_length_m"] == 0.0
    assert not any(f["level"] == "warning" for f in s["findings"])
    assert not any("depression" in f["text"].lower() and f["level"] != "ok" for f in s["findings"])
    assert s["findings"][-1]["level"] == "ok"
    depth = next(layer for layer in s["layers"] if layer["key"] == "depression_depth")
    assert depth["max"] < 1e-5


def test_timing_default_field(default_case):
    _, _, ds = default_case
    t0 = time.perf_counter()
    result = analyze(ds)
    elapsed = time.perf_counter() - t0
    assert elapsed < 6.0, result.timing_s
    t0 = time.perf_counter()
    result.summary()
    result.feature_collection()
    assert time.perf_counter() - t0 < 2.0


def test_timing_400k_cells(tmp_path):
    grid, _ = synthetic_dem(
        size_m=3160, cell_m=5.0,
        hills=((0.25, 0.30, 12.0, 300.0), (0.70, 0.65, 9.0, 250.0)),
        valley=(0.20, 0.0, 0.85, 1.0, 6.0, 200.0), depression=(0.55, 0.20, 3.0, 150.0),
    )
    assert grid.rows * grid.cols >= 399_000
    path = raster.write_geotiff(grid, tmp_path / "big.tif")
    ds = registry.read_any(path)
    t0 = time.perf_counter()
    result = analyze(ds)
    s = result.summary()
    elapsed = time.perf_counter() - t0
    assert elapsed < 25.0, result.timing_s
    assert s["source"]["kind"] == "dem" and s["grid"]["rows"] == grid.rows
    assert len(result.features["hills"]) >= 2
    t0 = time.perf_counter()
    zones_polygons(result, "landform")
    zones_points(result, "landform")
    assert time.perf_counter() - t0 < 10.0


# ==========================================================================
# landforms.py on its own
# ==========================================================================

def test_classify_rule_table():
    small = np.array([[2.0, 0.0, -2.0, 2.0, 0.0, 0.0, -2.0, 2.0, 0.0, -2.0, np.nan]])
    large = np.array([[2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0]])
    slope = np.array([[5.0, 5.0, 5.0, 5.0, 1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 1.0]])
    codes = lf.classify(small, large, slope)
    assert codes.dtype == np.int8
    assert codes.tolist() == [[1, 2, 2, 2, 4, 3, 5, 5, 5, 6, 0]]
    with pytest.raises(ValueError):
        lf.classify(small, large[:, :5], slope)


def test_position_labels_on_a_toy_grid():
    grid = ElevationGrid(np.zeros((30, 30)), 1000.0, 2000.0, 10.0, "EPSG:32612")
    assert lf.position_label(grid, 1005.0, 1995.0) == "north-west part"
    assert lf.position_label(grid, 1150.0, 1995.0) == "northern part"
    assert lf.position_label(grid, 1295.0, 1995.0) == "north-east part"
    assert lf.position_label(grid, 1005.0, 1850.0) == "western part"
    assert lf.position_label(grid, 1150.0, 1850.0) == "centre"
    assert lf.position_label(grid, 1295.0, 1850.0) == "eastern part"
    assert lf.position_label(grid, 1005.0, 1705.0) == "south-west part"
    assert lf.position_label(grid, 1150.0, 1705.0) == "southern part"
    assert lf.position_label(grid, 1295.0, 1705.0) == "south-east part"
    # The thirds follow the VALID extent, not the array.
    z = np.full((30, 30), np.nan)
    z[:, :10] = 1.0
    narrow = ElevationGrid(z, 1000.0, 2000.0, 10.0, "EPSG:32612")
    assert lf.position_label(narrow, 1050.0, 1850.0) == "centre"


def test_character_thresholds():
    assert lf.character(1.0, 0.5, 1.0)["key"] == "flat"
    assert lf.character(5.0, 0.5, 1.0)["key"] == "gently_undulating"
    assert lf.character(10.0, 2.9, 6.0)["key"] == "gently_undulating"
    assert lf.character(10.0, 3.0, 6.0)["key"] == "rolling"
    assert lf.character(30.0, 8.0, 15.0)["key"] == "hilly"
    assert lf.character(float("nan"), float("nan"), float("nan"))["key"] == "flat"
    why = lf.character(10.0, 3.0, 6.0)["why"]
    assert "3.0 %" in why and "10.0 m" in why


def test_slope_classes_and_aspect_distribution():
    slope = np.array([[0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, np.nan]])
    classes = lf.slope_classes(slope, 10.0)
    assert [c["key"] for c in classes] == ["flat", "gentle", "moderate", "strong", "steep"]
    assert [round(c["area_ha"], 3) for c in classes] == [0.02, 0.01, 0.01, 0.01, 0.02]
    assert sum(c["pct"] for c in classes) == pytest.approx(100.0)
    aspect = np.array([[np.nan, 0.0, 90.0, 180.0, 270.0, 359.0, 45.0, np.nan]])
    sectors = lf.aspect_distribution(aspect, slope, 10.0, flat_pct=0.5)
    by_key = {s["key"]: s["pct"] for s in sectors}
    assert by_key["flat"] == pytest.approx(100.0 / 7)
    assert by_key["N"] == pytest.approx(200.0 / 7)  # 0 and 359 degrees
    assert by_key["NE"] == pytest.approx(100.0 / 7)
    assert sum(by_key.values()) == pytest.approx(100.0)


def test_hills_and_lows_on_a_known_surface():
    grid, truth = synthetic_dem(notch=None)
    from agrosuite.terrain.derivatives import plane_trend, slope, tpi, tpi_standardized

    trend = plane_trend(grid)
    X, Y = grid.xy()
    detrended = grid.with_values(grid.z - trend["a"] * X - trend["b"] * Y)
    tpi_std = tpi_standardized(tpi(detrended, 150.0))
    slope_pct = slope(grid)[0]
    hills, labels = lf.hills(grid, tpi_std, slope_pct, min_height_m=0.5, measure=detrended.z)
    assert len(hills) == 2
    assert labels.dtype == np.int32 and set(np.unique(labels)) == {0, 1, 2}
    assert hills[0]["label"] == "Hill 1" and hills[0]["height_m"] > hills[1]["height_m"]
    for h in hills:
        assert (labels == h["id"]).sum() * grid.cell ** 2 / 10_000.0 == pytest.approx(h["area_ha"])
        assert labels[h["row"], h["col"]] == h["id"]
        assert grid.z[h["row"], h["col"]] == pytest.approx(h["summit_m"])
    ponded = np.zeros(grid.shape, bool)
    lows, low_labels = lf.lows(grid, tpi_std, ponded, slope_pct, min_depth_m=0.5, measure=detrended.z)
    assert lows and all(l["closed"] is False for l in lows)
    assert lows[0]["label"] == "Low 1" and lows[0]["depth_m"] > 0.5
    ponded[lows[0]["row"], lows[0]["col"]] = True
    lows2, _ = lf.lows(grid, tpi_std, ponded, slope_pct, min_depth_m=0.5, measure=detrended.z)
    assert lows2[0]["closed"] is True
    # A high threshold on the height leaves nothing.
    assert lf.hills(grid, tpi_std, slope_pct, min_height_m=50.0)[0] == []


def test_feature_height_is_the_physical_prominence():
    """A hill's height must be the height of the hill, not the part of it
    above wherever the TPI threshold happened to cut the flank: the same
    6 m hill must read 6 m whatever radius found it, and a 1 m bowl 1 m."""
    from agrosuite.terrain.derivatives import slope, tpi, tpi_standardized
    from agrosuite.terrain import hydrology as hydro

    n, cell = 160, 5.0
    grid = ElevationGrid(np.zeros((n, n)), 500_000.0, 5_730_000.0, cell, "EPSG:32612")
    X, Y = grid.xy()
    r2 = (X - 500_400.0) ** 2 + (Y - 5_729_600.0) ** 2
    grid.z = 700.0 + 6.0 * np.exp(-r2 / (2.0 * 60.0 ** 2))
    slope_pct = slope(grid)[0]
    heights = {}
    for radius in (60.0, 150.0, 400.0):
        hills, _ = lf.hills(grid, tpi_standardized(tpi(grid, radius)), slope_pct)
        assert len(hills) == 1
        heights[radius] = hills[0]["height_m"]
    for radius, height in heights.items():
        assert abs(height - 6.0) <= 0.1, heights
    # A paraboloid bowl 1 m deep on flat ground.
    bowl = ElevationGrid(np.full((n, n), 700.0), 500_000.0, 5_730_000.0, cell, "EPSG:32612")
    inside = r2 < 100.0 ** 2
    z = bowl.z
    z[inside] = 700.0 - 1.0 * (1.0 - r2[inside] / 100.0 ** 2)
    bowl.z = z
    slope_b = slope(bowl)[0]
    ponded = hydro.analyse_hydrology(bowl, slope_b).depth > hydro.WET_DEPTH_M
    lows, _ = lf.lows(bowl, tpi_standardized(tpi(bowl, 150.0)), ponded, slope_b)
    assert len(lows) == 1 and abs(lows[0]["depth_m"] - 1.0) <= 0.05
    assert lows[0]["closed"] is True
    # foot_level on its own: NaN when the patch has no ring at all.
    assert math.isnan(lf.foot_level(grid.z, np.ones(grid.shape, bool)))


# ==========================================================================
# Second-round findings: noise-gated depressions, interior shares, wet
# ground with real slope, saddle-limited hill heights
# ==========================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402  (tests/fixtures.py)
from agrosuite.core.dataset import apply_source_units  # noqa: E402
from agrosuite.terrain import analysis as an  # noqa: E402
from agrosuite.terrain import hydrology as hydro  # noqa: E402


def _john_deere_in_metres(tmp_path):
    """The Operations Center fixture with its feet declared: 585 m of
    flat ground under 0.4 m of GPS noise."""
    ds = registry.read_any(fixtures.john_deere_shapefile(tmp_path))
    apply_source_units(ds, {"swath_m": "ft", "distance_m": "ft", "elev_m": "ft"}, "canola")
    return ds


def _dem_dataset(grid, path):
    return registry.read_any(raster.write_geotiff(grid, path, dtype="float64"))


def _uniform_grid(n: int = 160, cell: float = 5.0) -> tuple[ElevationGrid, np.ndarray, np.ndarray]:
    grid = ElevationGrid(np.zeros((n, n)), 500_000.0, 5_730_000.0, cell, "EPSG:32612")
    X, Y = grid.xy()
    return grid, X, Y


def test_depressions_within_the_noise_are_counted_not_listed(tmp_path, default_case):
    """A flat field under 0.4 m of GPS noise ponds nowhere; read unsmoothed
    it is pocked with noise dips deeper than 5 cm, which were listed as
    dozens of potholes with volumes. Now a basin is listed on the terms of
    a hill — deeper than the feature floor and at least the feature area —
    and the rest are counted in one sentence."""
    ds = _john_deere_in_metres(tmp_path)
    summary = analyze(ds).summary()
    assert summary["features"]["depressions"] == []
    assert not any("holds roughly" in f["text"] for f in summary["findings"])
    assert not any(f["level"] == "warning" for f in summary["findings"])

    result = analyze(ds, TerrainOptions(smooth_m=0.0))  # the raw, noisy surface
    summary = result.summary()
    floor = summary["features"]["depression_floor_m"]
    assert floor == pytest.approx(max(0.05, summary["options"]["min_feature_height_m"]))
    listed = summary["features"]["depressions"]
    unlisted = summary["features"]["depressions_unlisted"]
    found = hydro.depressions(result.grid, result.filled)  # every basin over 5 cm
    assert len(found) > 100 and unlisted >= 50
    assert len(listed) + unlisted == len(found)
    assert all(d["max_depth_m"] >= floor and d["area_ha"] >= an.MIN_DEPRESSION_AREA_HA
               for d in listed)
    assert [d["id"] for d in listed] == list(range(1, len(listed) + 1))
    assert set(np.unique(result.labels["depressions"])) == set(range(len(listed) + 1))
    sentences = [f["text"] for f in summary["findings"]
                 if "shallow hollows within the elevation noise" in f["text"]]
    assert len(sentences) == 1
    assert sentences[0].startswith(f"{unlisted} shallow hollows") and "not listed" in sentences[0]

    # The demo field: its bowl and its dammed valley pond under the floor.
    result, _, _ = default_case
    summary = result.summary()
    assert summary["features"]["depressions"] == []
    assert summary["features"]["depressions_unlisted"] >= 2
    assert summary["features"]["depression_floor_m"] == pytest.approx(
        summary["options"]["min_feature_height_m"]
    )


def test_shares_are_read_inside_the_edge_ring(tmp_path):
    """A uniform 3 % plane is 100 % gentle: the outer ring of cells, whose
    slope is halved by the replicated border, stays on the layers but
    out of the shares."""
    grid, X, Y = _uniform_grid(200)
    grid.z = 700.0 + 0.03 * (X - 500_000.0)
    result = analyze(_dem_dataset(grid, tmp_path / "plane.tif"))
    summary = result.summary()
    slope_by_key = {c["key"]: c["pct"] for c in summary["slope"]["classes"]}
    assert slope_by_key["flat"] == 0.0 and slope_by_key["gentle"] == 100.0
    landform_by_key = {c["key"]: c["pct"] for c in summary["landforms"]["classes"]}
    assert landform_by_key["flat"] == 0.0 and landform_by_key["mid_slope"] == 100.0
    aspect_by_key = {c["key"]: c["pct"] for c in summary["aspect"]["sectors"]}
    assert aspect_by_key["W"] == 100.0 and aspect_by_key["flat"] == 0.0
    # The layer keeps the ring: its west and east columns read 1.5 %.
    slope = result.layers["slope_pct"]
    assert (slope[1:-1, 0] < 2.0).all() and (slope[1:-1, 1] == pytest.approx(3.0, abs=1e-6))
    # The grid section says so, and the areas still add up to the field.
    g = summary["grid"]
    assert g["edge_ring_ha"] == pytest.approx((4 * 200 - 4) * 25 / 10_000.0)
    assert g["interior_ha"] + g["edge_ring_ha"] == pytest.approx(g["area_ha"])
    assert "ring" in g["shares_note"] and "shares" in g["shares_note"]
    for table in (summary["slope"]["classes"], summary["landforms"]["classes"], summary["aspect"]["sectors"]):
        assert sum(c["area_ha"] for c in table) == pytest.approx(g["area_ha"])


def test_wet_ground_needs_a_slope_the_surface_can_vouch_for(tmp_path):
    """One hill on a dead-flat plain: the plain's tan(beta) is floored, so
    its wetness index is a constant and its top decile was whichever
    floored cells the accumulation favoured — a tenth of the field
    'likely wet'. Only cells with a slope above the noise slope are
    ranked now, and the wet ground is the foot of the hill."""
    grid, X, Y = _uniform_grid()
    r2 = (X - 500_400.0) ** 2 + (Y - 5_729_600.0) ** 2
    grid.z = 700.0 + 6.0 * np.exp(-r2 / (2.0 * 60.0 ** 2))
    result = analyze(_dem_dataset(grid, tmp_path / "hill.tif"))
    summary = result.summary()
    wet = summary["wetness"]
    assert wet["noise_slope_pct"] == an.MIN_NOISE_SLOPE_PCT  # a raster has no noise figure
    assert 0.0 < wet["wet_pct"] < 3.0, wet
    assert 10.0 < wet["ranked_share_pct"] < 40.0
    slope = result.layers["slope_pct"]
    with np.errstate(invalid="ignore"):
        floored = result.grid.mask & (slope < wet["noise_slope_pct"])
    assert floored.sum() > 0.5 * result.grid.valid_count
    assert not (result.wet & floored).any()
    assert wet["twi_threshold"] == pytest.approx(result.twi_threshold)

    # Too little of the field with a slope: no ranking at all, only the
    # listed depressions, and the finding says why.
    twi = result.layers["twi"]
    sparse = np.zeros(result.grid.shape)
    sparse[:2, :] = 5.0  # 1.25 % of the cells have a slope
    listed = np.zeros(result.grid.shape, dtype=bool)
    listed[80, 80] = True
    cells, threshold, share = an._wet_cells(result.grid, twi, sparse, listed, 0.1, True)
    assert share < an.MIN_WET_CANDIDATE_SHARE and math.isnan(threshold)
    assert cells.sum() == 1 and cells[80, 80]


def test_hill_heights_stop_at_the_saddle_on_a_rolling_field(tmp_path):
    """On a 2-D sinusoid every hill is 1 m above its saddles; the foot walk
    used to descend past them into the neighbouring pits and read 8 %
    more. The saddle to the listed hills and lows now bounds the foot."""
    grid, X, Y = _uniform_grid()
    L = 200.0
    grid.z = 700.0 + np.sin(2.0 * np.pi * (X - 500_000.0) / L) * np.sin(2.0 * np.pi * (Y - 5_729_200.0) / L)
    result = analyze(_dem_dataset(grid, tmp_path / "rolling.tif"))
    hills = result.features["hills"]
    lows = result.features["lows"]
    assert len(hills) >= 30 and len(lows) >= 30
    for h in hills:
        assert abs(h["height_m"] - 1.0) <= 0.03, (h["label"], h["height_m"])
    for l in lows:
        assert abs(l["depth_m"] - 1.0) <= 0.03, (l["label"], l["depth_m"])
    # A lone hill has no saddle: its foot is still where the ground goes flat.
    r2 = (X - 500_400.0) ** 2 + (Y - 5_729_600.0) ** 2
    grid.z = 700.0 + 6.0 * np.exp(-r2 / (2.0 * 60.0 ** 2))
    lone = analyze(_dem_dataset(grid, tmp_path / "lone.tif")).features["hills"]
    assert len(lone) == 1 and abs(lone[0]["height_m"] - 6.0) <= 0.1
    # saddle_level on its own: exact on a two-hill ridge with a col at 2 m.
    z = np.zeros((40, 80))
    xx = np.arange(80)
    z[:, :] = np.where(xx < 40, 5.0 * np.exp(-((xx - 15) / 6.0) ** 2), 4.0 * np.exp(-((xx - 65) / 6.0) ** 2))[None, :]
    z += 2.0 * np.exp(-((xx - 40) / 30.0) ** 2)[None, :]  # a col rising to about 2 m
    left, right = np.zeros_like(z, bool), np.zeros_like(z, bool)
    left[:, 10:21] = True
    right[:, 60:71] = True
    saddle = lf.saddle_level(z, left, right, 1.0)
    assert saddle == pytest.approx(float(z[0, 21:60].min()), abs=2e-3)
    assert math.isnan(lf.saddle_level(z, left, np.zeros_like(z, bool), 1.0))
