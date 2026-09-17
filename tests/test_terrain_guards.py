"""The terrain analyser on the inputs a user can actually hand it: nonsense
options, a constant or noise-only altitude column, a raster that has gone.

Each case here once produced a wrong answer with a straight face (a flat
field a tenth hilltop and a tenth wet, a sentence about the fall of a level
field) or a crash with numpy's words in it (a terabyte kernel, "too many
bins"). The analyser must refuse in a sentence or answer honestly.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from agrosuite.core import schema as sch
from agrosuite.core.units import UNIT_PRESETS
from agrosuite.formats import raster, registry
from agrosuite.terrain import analysis as an
from agrosuite.terrain.analysis import TerrainOptions, analyze, zones_polygons
from agrosuite.terrain.grid import ElevationGrid
from agrosuite.terrain.synthetic import synthetic_dem, synthetic_terrain


@pytest.fixture(scope="module")
def field():
    ds, _ = synthetic_terrain()
    return ds


@pytest.fixture(scope="module")
def constant_dem(tmp_path_factory):
    grid = ElevationGrid(
        np.full((200, 200), 700.0), 500_000.0, 5_730_000.0, 5.0, "EPSG:32612"
    )
    path = raster.write_geotiff(grid, tmp_path_factory.mktemp("dem") / "constant.tif", dtype="float64")
    return analyze(registry.read_any(path))


# ==========================================================================
# Options that cannot mean anything
# ==========================================================================

@pytest.mark.parametrize("options, words", [
    ({"smooth_m": 1e6}, "smoothing scale of 1e+06 m is wider than the field"),
    ({"tpi_small_m": 1e6}, "small TPI radius of 1e+06 m is wider than the field"),
    ({"tpi_large_m": 1e6}, "large TPI radius of 1e+06 m is wider than the field"),
    ({"min_upstream_ha": 0}, "upstream area for a drainage line must be a positive"),
    ({"min_upstream_ha": -2}, "upstream area for a drainage line must be a positive"),
    ({"hillshade_altitude": 0}, "sun altitude must be above 0"),
    ({"hillshade_altitude": 200}, "sun altitude must be above 0"),
    ({"hillshade_altitude": -30}, "sun altitude must be above 0"),
    ({"tpi_small_m": 500, "tpi_large_m": 30}, "small TPI radius (500 m) must be smaller than the large one (30 m)"),
    ({"tpi_large_m": 5}, "small TPI radius (5 m) must be smaller than the large one (5 m)"),
])
def test_nonsense_options_are_refused_in_a_sentence(field, options, words):
    with pytest.raises(ValueError) as err:
        analyze(field, options)
    assert words in str(err.value)
    assert "allocate" not in str(err.value)  # never numpy's MemoryError text


@pytest.mark.parametrize("options, words", [
    ({"smooth_m": 1e6}, "smoothing scale of 3.28084e+06 ft is wider than the field"),
    ({"tpi_large_m": 1e6}, "large TPI radius of 3.28084e+06 ft is wider than the field"),
    ({"tpi_small_m": 500, "tpi_large_m": 30},
     "small TPI radius (1640.42 ft) must be smaller than the large one (98.4252 ft)"),
    ({"cell_m": -2}, "cell size must be a positive number of feet"),
    ({"min_upstream_ha": 0}, "must be a positive number of acres"),
])
def test_a_refusal_is_written_in_the_units_it_was_typed_in(field, options, words):
    """The box this number came out of is in feet when the picker is.

    Refused in metres, the user has to convert the app's own answer back
    before they can tell what to type instead — and the number they typed
    does not appear anywhere in the sentence about it. The absurd ones
    keep their exponent: 3.28084e+06 ft is plainly not a scale anyone
    meant, where "3 280 840 ft" reads like a figure with a typo.
    """
    with pytest.raises(ValueError) as err:
        analyze(field, options, UNIT_PRESETS["canada"])
    said = str(err.value)
    assert words in said
    # Not one quantity left in the store's own units.
    assert not re.search(r"\d\s?(?:m|ha|km|cm|m²)\b", said), said


def test_huge_smoothing_is_refused_on_a_dem_too(tmp_path):
    grid, _ = synthetic_dem()
    path = raster.write_geotiff(grid, tmp_path / "dem.tif", dtype="float64")
    ds = registry.read_any(path)
    with pytest.raises(ValueError, match="smoothing scale of 1e\\+06 m is wider than the field"):
        analyze(ds, {"smooth_m": 1e6})


def test_one_tpi_radius_keeps_the_other_on_its_side(field):
    small_only = analyze(field, {"tpi_small_m": 100.0}).options
    assert small_only.tpi_small_m == 100.0 and small_only.tpi_large_m >= 200.0
    large_only = analyze(field, {"tpi_large_m": 40.0}).options
    assert large_only.tpi_large_m == 40.0 and large_only.tpi_small_m < 40.0


def test_sun_azimuth_is_wrapped_and_echoed(field):
    result = analyze(field, {"hillshade_azimuth": -720.0})
    assert result.options.hillshade_azimuth == 0.0
    result = analyze(field, {"hillshade_azimuth": 405.0})
    assert result.options.hillshade_azimuth == 45.0
    shade = result.layers["hillshade"]
    assert np.nanmax(shade) > 0.5  # a real sun, not a black layer


# ==========================================================================
# Level ground answers honestly
# ==========================================================================

def _class_shares(summary):
    return {c["key"]: c["pct"] for c in summary["landforms"]["classes"]}


def _feature_lists(summary):
    """The three feature lists (the section also carries the count and the
    floor of the depressions left unlisted)."""
    return {k: summary["features"][k] for k in ("hills", "lows", "depressions")}


def test_constant_altitude_column_is_a_flat_field(field):
    ds = synthetic_terrain()[0]
    ds.df[sch.ELEVATION] = 700.0  # a monitor with no altitude output writes a placeholder
    summary = analyze(ds).summary()  # used to raise numpy's "Too many bins"
    assert summary["character"]["key"] == "flat"
    assert _class_shares(summary)["flat"] == pytest.approx(100.0)
    assert summary["wetness"]["wet_pct"] == 0.0
    assert _feature_lists(summary) == {"hills": [], "lows": [], "depressions": []}
    assert len(summary["elevation"]["histogram"]["counts"]) == 20
    assert not any("falls about" in f["text"] for f in summary["findings"])


def test_constant_dem_has_one_landform_and_no_wet_ground(constant_dem):
    summary = constant_dem.summary()
    shares = _class_shares(summary)
    assert shares["flat"] == pytest.approx(100.0)
    assert all(v == 0.0 for k, v in shares.items() if k != "flat")
    assert summary["wetness"]["wet_pct"] == 0.0
    assert not any("falls about" in f["text"] for f in summary["findings"])
    # Level within the noise: the one finding that says why nothing is drawn.
    assert summary["elevation"]["level"] is True
    assert any("level within the precision" in f["text"] for f in summary["findings"])
    zones = zones_polygons(constant_dem, "landform")
    assert list(zones.df["zone_label"]) == ["Flat"]
    assert set(zones_polygons(constant_dem, "wetness").df["zone_label"]) == {"Well drained"}


def test_noise_only_altitude_is_flat_and_dry():
    ds, _ = synthetic_terrain()
    rng = np.random.default_rng(0)
    ds.df[sch.ELEVATION] = 700.0 + rng.normal(0.0, 0.02, len(ds.df))
    result = analyze(ds)
    summary = result.summary()
    assert summary["character"]["key"] == "flat"
    assert _class_shares(summary)["flat"] >= 99.5  # a few edge cells carry more noise
    assert summary["features"]["hills"] == [] and summary["features"]["lows"] == []
    assert summary["wetness"]["wet_pct"] < 1.0
    assert result.wet_by_index is False
    assert not any("Likely wet ground" in f["text"] for f in summary["findings"])


def _level_contract(summary):
    """What a level field must come back as: nothing drawn, one sentence why."""
    assert summary["elevation"]["level"] is True
    assert summary["elevation"]["residual_relief_m"] < summary["elevation"]["relief_floor_m"]
    assert summary["contours"] == {"interval_m": None, "count": 0}
    assert summary["wetness"]["drainage_length_m"] == 0.0
    assert summary["wetness"]["wet_pct"] == 0.0 and summary["wetness"]["wet_area_ha"] == 0.0
    assert _feature_lists(summary) == {"hills": [], "lows": [], "depressions": []}
    level_findings = [f for f in summary["findings"] if "level within the precision" in f["text"]]
    assert len(level_findings) == 1
    text = level_findings[0]["text"]
    assert "no contour lines, drainage lines, likely-wet ground or hills and hollows" in text
    assert not any("cannot tell wet ground from dry" in f["text"] for f in summary["findings"])


def test_constant_altitude_column_draws_nothing_from_rounding_noise(field):
    """A constant column comes off the gridding with 1e-12 m of float noise,
    which used to draw 953 contour loops and 1.6 km of drainage lines."""
    ds = synthetic_terrain()[0]
    ds.df[sch.ELEVATION] = 700.0
    result = analyze(ds)
    assert result.level is True and result.contour_interval_m is None
    assert result.contours["features"] == [] and result.drainage["features"] == []
    assert not result.wet.any()
    _level_contract(result.summary())


def test_two_centimetre_ripple_is_level(tmp_path):
    """A 2 cm ripple on a raster is under the 5 cm the elevation can vouch
    for: level, with no contours at 0.1 m and no drainage lines."""
    n = 200
    ripple = 700.0 + 0.01 * np.sin(np.arange(n) / 7.0)[None, :] * np.ones((n, 1))
    grid = ElevationGrid(ripple, 500_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    path = raster.write_geotiff(grid, tmp_path / "ripple.tif", dtype="float64")
    result = analyze(registry.read_any(path))
    summary = result.summary()
    _level_contract(summary)
    assert summary["elevation"]["residual_relief_m"] == pytest.approx(0.02, abs=0.005)
    assert summary["elevation"]["relief_floor_m"] == pytest.approx(an.LEVEL_RELIEF_M)
    # An explicit interval changes nothing on a level field either.
    assert analyze(registry.read_any(path), {"contour_interval_m": 0.01}).summary()["contours"]["count"] == 0


def test_tilted_plane_is_not_level(tmp_path):
    """No relief beyond the tilt is not no relief: the fall of a plane has
    real, parallel contours, so the plane's drop counts toward the test."""
    from agrosuite.formats.raster import dataset_from_dem

    grid, _ = synthetic_dem(hills=(), valley=None, depression=None, notch=None)
    summary = analyze(dataset_from_dem(grid, {"path": "", "notes": []})).summary()
    assert summary["elevation"]["level"] is False
    assert summary["elevation"]["residual_relief_m"] < summary["elevation"]["relief_floor_m"]
    assert summary["trend"]["drop_m"] > summary["elevation"]["relief_floor_m"]
    assert summary["contours"]["count"] > 3 and summary["contours"]["interval_m"] == 1.0


@pytest.mark.parametrize("shape", [(1, 40), (40, 1), (2, 40), (40, 2)])
def test_a_raster_narrower_than_three_cells_is_too_small(tmp_path, shape):
    """A single row of cells is a line, not a surface; refused in a sentence
    rather than skimage's 'Input array must be at least 2x2'."""
    z = 700.0 + 0.01 * np.arange(shape[0] * shape[1]).reshape(shape)
    grid = ElevationGrid(z, 500_000.0, 5_730_000.0, 5.0, "EPSG:32612")
    path = raster.write_geotiff(grid, tmp_path / f"r{shape[0]}x{shape[1]}.tif", dtype="float64")
    with pytest.raises(ValueError) as err:
        analyze(registry.read_any(path))
    text = str(err.value)
    assert "too small to analyse" in text and "at least 3 by 3" in text
    assert f"{shape[0]} row" in text and f"{shape[1]} column" in text
    assert "2x2" not in text


def test_a_real_field_still_gets_its_classes_and_wet_ground(field):
    result = analyze(field)
    summary = result.summary()
    shares = _class_shares(summary)
    assert shares["hilltop"] > 0 and shares["valley"] > 0
    assert result.wet_by_index is True
    assert summary["wetness"]["wet_pct"] > 5.0
    assert any("falls about" in f["text"] for f in summary["findings"])


def test_elevation_bands_need_relief(constant_dem, tmp_path):
    with pytest.raises(ValueError, match="too little to split into 4 elevation bands"):
        zones_polygons(constant_dem, "elevation_bands")
    # A quantized raster repeats quantile edges; the empty bands are merged,
    # never labelled "700.0–700.0 m".
    grid, _ = synthetic_dem()
    grid.z = np.round(grid.z)
    ds = registry.read_any(raster.write_geotiff(grid, tmp_path / "quantized.tif", dtype="float64"))
    zones = zones_polygons(analyze(ds), "elevation_bands", bands=8)
    labels = list(zones.df["zone_label"])
    assert len(labels) == len(set(labels))
    for label in labels:
        lo, hi = label[label.index("(") + 1:label.index(" m)")].split("–")
        assert float(hi) - float(lo) >= an.BAND_MIN_M


# ==========================================================================
# A raster that has gone
# ==========================================================================

def test_missing_dem_file_is_refused_not_regridded(tmp_path):
    grid, _ = synthetic_dem()
    path = raster.write_geotiff(grid, tmp_path / "field.tif", dtype="float64")
    ds = registry.read_any(path)
    os.remove(path)
    with pytest.raises(ValueError) as err:
        analyze(ds)
    text = str(err.value)
    assert "can no longer be found" in text and str(path) in text
    # A dataset that never had a raster path is gridded from its points as before.
    from agrosuite.formats.raster import dataset_from_dem

    plain = dataset_from_dem(grid, {"path": "", "notes": []})
    assert analyze(plain).summary()["source"]["kind"] == "points"


def test_options_dataclass_defaults_are_still_valid(field):
    resolved = analyze(field, TerrainOptions()).options
    assert 0.0 < resolved.hillshade_altitude <= 90.0
    assert resolved.min_upstream_ha > 0
    assert resolved.tpi_small_m < resolved.tpi_large_m


# ==========================================================================
# Options and columns that are not numbers
# ==========================================================================

@pytest.mark.parametrize("options, words", [
    ({"cell_m": "abc"}, "terrain option cell_m must be a number, not 'abc'"),
    ({"max_cells": "abc"}, "terrain option max_cells must be a number, not 'abc'"),
    ({"max_cells": 2.5}, "terrain option max_cells must be a whole number, not 2.5"),
    ({"tpi_large_m": [150]}, "terrain option tpi_large_m must be a number, not [150]"),
    ({"detrend_passes": "maybe"}, "terrain option detrend_passes must be true or false, not 'maybe'"),
])
def test_non_numeric_options_are_refused_in_a_sentence(field, options, words):
    with pytest.raises(ValueError) as err:
        analyze(field, options)
    assert words in str(err.value)
    assert "could not convert" not in str(err.value) and "invalid literal" not in str(err.value)


def test_blank_and_string_options_are_read_as_numbers(field):
    """An interface sends '' for a box left empty and '6' for one typed in."""
    result = analyze(field, {"cell_m": "", "smooth_m": " ", "tpi_small_m": "40", "max_cells": ""})
    assert result.grid.cell == 5.0  # chosen from the data
    assert result.options.tpi_small_m == 40.0 and result.options.max_cells == 400_000
    assert analyze(field, {"detrend_passes": "false"}).source_report["detrended"] is False


def test_non_numeric_altitudes_are_counted_as_missing():
    ds, _ = synthetic_terrain()
    column = ds.df[sch.ELEVATION].astype(object)
    column.iloc[::50] = "n/a"
    ds.df[sch.ELEVATION] = column
    result = analyze(ds)  # used to raise "could not convert string to float: 'n/a'"
    source = result.summary()["source"]
    assert source["non_numeric_elevations"] == 668
    assert source["missing_elevations"] == 668
    assert source["points_used"] <= source["points_total"] - 668
    assert any("668 of the 33400 records carry an altitude that is not a number ('n/a')" in n
               for n in source["notes"])
    assert len(result.features["hills"]) == 2
