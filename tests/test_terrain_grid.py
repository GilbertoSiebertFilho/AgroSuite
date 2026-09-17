"""Tests for the terrain elevation grid.

Two kinds of test. The contract tests pin the geometry of
:class:`ElevationGrid` — row 0 north, cell centres at half-cell offsets,
NaN outside the field — because every later terrain module is written
against exactly that. The gridding tests run on a synthetic field whose
surface is known, so they can ask whether the grid recovers it, whether the
GPS spikes and pass offsets planted in the data were removed, and whether
the mask follows the field's real outline rather than its convex hull.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.core import schema as sch
from agrosuite.core.dataset import Dataset, DatasetMeta
from agrosuite.core.units import UNIT_PRESETS, Phrase
from agrosuite.terrain import (
    ElevationGrid,
    connected_components,
    disc_kernel,
    gaussian_smooth,
    grid_from_points,
    nan_convolve,
    synthetic_dem,
    synthetic_terrain,
)
from agrosuite.terrain import grid as grid_mod
from agrosuite.terrain import notes as notes_mod
from agrosuite.terrain.grid import fill_nearest

REPORT_KEYS = {
    "points_total", "points_used", "outliers_removed", "passes", "pass_offset_sd_m",
    "detrended", "cell_m", "rows", "cols", "area_ha", "max_gap_m", "edge_gap_m", "smooth_m",
    "point_noise_m", "vertical_noise_m", "spacing_m", "crs", "notes", "note_facts",
    "fill_values_removed", "passes_dropped", "value_step_m",
    "surface_noise_m", "missing_elevations", "non_numeric_elevations",
}


def _toy_grid(rows: int = 5, cols: int = 6, cell: float = 2.0) -> ElevationGrid:
    """A grid whose value encodes its own index: z[r, c] = 10 r + c."""
    r, c = np.indices((rows, cols))
    return ElevationGrid((10 * r + c).astype("float64"), 1000.0, 2000.0, cell, "EPSG:32612")


def _pass_bias_sd(grid: ElevationGrid, ds: Dataset, truth: dict) -> float:
    """Spread of the per-pass median error of the grid at the points.

    Pass offsets left in the grid show up as a bias that is constant along a
    pass and differs between passes; this is that bias' standard deviation.
    """
    x = ds.df[sch.X].to_numpy()
    y = ds.df[sch.Y].to_numpy()
    p = ds.df[sch.PASS].to_numpy()
    err = grid.sample(x, y) - truth["surface"](x, y)
    medians = np.array([np.nanmedian(err[p == k]) for k in np.unique(p)])
    return float(np.nanstd(medians))


@pytest.fixture(scope="module")
def field():
    """The default synthetic field and its grid, built once."""
    ds, truth = synthetic_terrain()
    grid, report = grid_from_points(ds)
    return ds, truth, grid, report


# ==========================================================================
# Contract
# ==========================================================================

def test_rowcol_and_xy_round_trip():
    grid = _toy_grid()
    X, Y = grid.xy()
    assert X.shape == grid.shape and Y.shape == grid.shape
    # Row 0 is the north edge: y decreases down the array.
    assert Y[0, 0] > Y[-1, 0]
    assert X[0, 0] == 1000.0 + 1.0 and Y[0, 0] == 2000.0 - 1.0
    row, col = grid.rowcol(X, Y)
    r, c = np.indices(grid.shape)
    np.testing.assert_allclose(row, r)
    np.testing.assert_allclose(col, c)


def test_bounds_transform_and_area():
    grid = _toy_grid(rows=5, cols=6, cell=2.0)
    assert grid.bounds_metric() == (1000.0, 1990.0, 1012.0, 2000.0)
    t = grid.transform()
    assert t @ (0, 0) == (1000.0, 2000.0)
    assert t @ (grid.cols, grid.rows) == (1012.0, 1990.0)
    assert grid.area_ha() == pytest.approx(30 * 4 / 10_000)
    grid.z[0, 0] = np.nan
    assert grid.valid_count == 29
    assert grid.mask.sum() == 29


def test_lonlat_bounds_and_round_trip():
    grid = ElevationGrid(np.zeros((40, 50)), 323_000.0, 5_736_000.0, 10.0, "EPSG:32612")
    west, south, east, north = grid.bounds_lonlat()
    X, Y = grid.xy()
    lon, lat = grid.to_lonlat(X, Y)
    assert west < lon.min() and lon.max() < east
    assert south < lat.min() and lat.max() < north
    x_back, y_back = grid.from_lonlat(lon, lat)
    np.testing.assert_allclose(x_back, X, atol=1e-4)
    np.testing.assert_allclose(y_back, Y, atol=1e-4)
    doc = grid.to_dict()
    assert set(doc) == {"rows", "cols", "cell_m", "crs", "bounds_lonlat", "area_ha"}
    json.dumps(doc)


def test_with_values_keeps_geometry_and_checks_shape():
    grid = _toy_grid()
    other = grid.with_values(np.ones(grid.shape))
    assert (other.x0, other.y0, other.cell, other.crs) == (grid.x0, grid.y0, grid.cell, grid.crs)
    assert other.z.sum() == grid.z.size
    with pytest.raises(ValueError):
        grid.with_values(np.ones((2, 2)))
    with pytest.raises(ValueError):
        ElevationGrid(np.ones(4), 0.0, 0.0, 1.0, "EPSG:32612")


def test_sample_known_values():
    grid = _toy_grid()
    X, Y = grid.xy()
    np.testing.assert_allclose(grid.sample(X, Y), grid.z)
    # Halfway between two neighbouring centres, linear interpolation gives the mean.
    assert grid.sample(X[0, 0] + 1.0, Y[0, 0]) == pytest.approx(0.5)
    assert grid.sample(X[0, 0], Y[0, 0] - 1.0) == pytest.approx(5.0)
    assert grid.sample(X[2, 3], Y[2, 3], order=0) == 23.0
    # The outer half-cell rim is still inside the raster; beyond it is not.
    assert grid.sample(grid.x0 + 0.1, grid.y0 - 0.1) == pytest.approx(0.0)
    assert np.isnan(grid.sample(grid.x0 - 0.1, Y[0, 0]))
    assert np.isnan(grid.sample(X[0, 0], grid.y0 - grid.rows * grid.cell - 0.1))
    # A masked cell samples NaN, and does not bleed into its neighbours.
    grid.z[2, 2] = np.nan
    assert np.isnan(grid.sample(X[2, 2], Y[2, 2]))
    assert np.isfinite(grid.sample(X[2, 3], Y[2, 3]))
    # Next to the hole the interpolation runs on the nearest-filled copy:
    # a neighbour's value stands in for the masked cell, never NaN.
    near_hole = grid.sample(X[2, 3] - 0.8, Y[2, 3])
    assert np.isfinite(near_hole) and abs(near_hole - 23.0) < 1.5
    out = grid.sample(np.array([X[1, 1], grid.x0 - 5]), np.array([Y[1, 1], Y[1, 1]]))
    assert out[0] == 11.0 and np.isnan(out[1])


def test_fill_nearest():
    grid = _toy_grid()
    grid.z[1:3, 1:3] = np.nan
    filled = grid.fill_nearest()
    assert np.isfinite(filled).all()
    assert filled[1, 1] in (0.0, 1.0, 10.0)
    np.testing.assert_array_equal(filled[0], grid.z[0])
    assert np.isnan(grid.z[1, 1])  # the grid itself is untouched
    np.testing.assert_array_equal(fill_nearest(np.full((2, 2), np.nan)), np.full((2, 2), np.nan))


def test_coarsen_block_means():
    grid = _toy_grid(rows=4, cols=6, cell=2.0)
    grid.z[0, 0] = np.nan
    coarse = grid.coarsen(2)
    assert coarse.shape == (2, 3) and coarse.cell == 4.0
    assert (coarse.x0, coarse.y0) == (grid.x0, grid.y0)
    assert coarse.z[0, 0] == pytest.approx(np.nanmean([1.0, 10.0, 11.0]))
    assert coarse.z[1, 2] == pytest.approx(np.mean([24, 25, 34, 35]))
    ragged = _toy_grid(rows=5, cols=5, cell=1.0).coarsen(2)
    assert ragged.shape == (3, 3)
    ragged.z[:] = np.nan
    assert np.isnan(ragged.coarsen(2).z).all()
    with pytest.raises(ValueError):
        grid.coarsen(0)


def test_nan_convolve_preserves_constant_and_does_not_bleed():
    values = np.full((20, 20), 3.5)
    values[5:9, 5:9] = np.nan
    values[0, :] = np.nan
    out = nan_convolve(values, disc_kernel(2.5))
    valid = np.isfinite(values)
    np.testing.assert_allclose(out[valid], 3.5)
    assert np.isfinite(out).all()  # the holes get a value; callers re-mask
    assert np.isnan(nan_convolve(np.full((4, 4), np.nan), disc_kernel(1))).all()


def test_disc_kernel_symmetry():
    k = disc_kernel(3)
    assert k.shape == (7, 7)
    assert k.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(k, k[::-1, :])
    np.testing.assert_array_equal(k, k[:, ::-1])
    np.testing.assert_array_equal(k, k.T)
    assert k[3, 3] > 0 and k[0, 0] == 0.0 and k[3, 0] > 0
    assert disc_kernel(0).shape == (1, 1)


def test_gaussian_smooth_keeps_a_ramp():
    r, c = np.indices((30, 30))
    ramp = 0.5 * r + 0.2 * c
    out = gaussian_smooth(ramp, 1.5)
    np.testing.assert_allclose(out[6:-6, 6:-6], ramp[6:-6, 6:-6], atol=1e-6)
    np.testing.assert_array_equal(gaussian_smooth(ramp, 0.0), ramp)


def test_connected_components_drop_small():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True      # 9 cells
    mask[6, 6] = True          # 1 cell, touches the next diagonally
    mask[7, 7] = True
    labels, count = connected_components(mask)
    assert count == 2 and labels.dtype == np.int32
    labels, count = connected_components(mask, min_cells=3)
    assert count == 1
    assert labels[2, 2] == 1 and labels[6, 6] == 0
    assert connected_components(np.zeros((3, 3), bool))[1] == 0


# ==========================================================================
# Synthetic truth
# ==========================================================================

def test_synthetic_terrain_is_consistent():
    ds, truth = synthetic_terrain()
    assert ds.meta.operation == "harvest" and ds.meta.crop == "canola"
    assert ds.meta.name == "Terrain demo"
    for col in (sch.LON, sch.LAT, sch.TIMESTAMP, sch.VALUE, sch.SWATH, sch.SPEED, sch.PASS, sch.ELEVATION):
        assert col in ds.df.columns
    assert ds.metric_crs == truth["crs"]
    x = ds.df[sch.X].to_numpy()
    y = ds.df[sch.Y].to_numpy()
    np.testing.assert_allclose(truth["surface"](x, y), truth["elev_true_m"], atol=1e-6)
    assert truth["inside"](x, y).all()
    # The notch was actually left out of the track.
    ox, oy = truth["origin_m"]
    assert not truth["inside"](ox + 0.1 * 800, oy + 0.9 * 800)
    assert truth["area_ha"] == pytest.approx(60.0)
    assert truth["tilt"]["fall_m"] == pytest.approx(0.008 * 800 * np.sqrt(2), rel=1e-6)
    for hill in truth["hills"]:
        assert hill["summit_z_m"] >= truth["surface"](*hill["centre_m"])
        assert np.hypot(*(np.subtract(hill["summit_m"], hill["centre_m"]))) < hill["radius_m"]
    dep = truth["depression"]
    assert 0.0 < dep["closed_depth_m"] < dep["depth_param_m"]
    assert dep["volume_m3"] > 0 and dep["area_m2"] > 0
    assert dep["spill_z_m"] == pytest.approx(dep["bottom_z_m"] + dep["closed_depth_m"])
    assert truth["valley"]["axis_m"].shape[1] == 2
    assert len(truth["pass_offsets_m"]) == truth["n_passes"]
    # Deterministic.
    ds2, _ = synthetic_terrain()
    np.testing.assert_array_equal(ds.df[sch.ELEVATION].to_numpy(), ds2.df[sch.ELEVATION].to_numpy())


def test_synthetic_dem_matches_surface():
    dem, truth = synthetic_dem(cell_m=10.0)
    assert dem.shape == (80, 80) and dem.cell == 10.0
    X, Y = dem.xy()
    inside = truth["inside"](X, Y)
    np.testing.assert_allclose(dem.z[inside], truth["surface"](X[inside], Y[inside]))
    assert np.isnan(dem.z[~inside]).all()
    assert dem.area_ha() == pytest.approx(60.0)
    full, _ = synthetic_dem(cell_m=20.0, notch=None)
    assert full.valid_count == full.z.size


# ==========================================================================
# grid_from_points
# ==========================================================================

def test_grid_recovers_the_truth(field):
    ds, truth, grid, report = field
    X, Y = grid.xy()
    inside = truth["inside"](X, Y) & grid.mask
    assert inside.sum() > 0.9 * truth["area_ha"] * 10_000 / grid.cell ** 2
    err = grid.z[inside] - truth["surface"](X[inside], Y[inside])
    assert np.sqrt(np.mean(err ** 2)) < 0.35
    assert abs(np.mean(err)) < 0.1
    assert grid.crs == ds.metric_crs
    assert report["vertical_noise_m"] == pytest.approx(0.3, abs=0.06)


def test_mask_follows_the_footprint(field):
    ds, truth, grid, report = field
    ox, oy = truth["origin_m"]
    size = truth["size_m"]
    # The corner cut out of the field is a hole in the grid, not filled from
    # the convex hull.
    assert np.isnan(grid.sample(ox + 0.1 * size, oy + 0.9 * size))
    X, Y = grid.xy()
    notch = (X < ox + 0.25 * size - 20) & (Y > oy + 0.75 * size + 20) & (X > ox) & (Y < oy + size)
    assert np.isnan(grid.z[notch]).mean() > 0.95
    assert report["area_ha"] == pytest.approx(truth["area_ha"], rel=0.15)
    assert grid.area_ha() == report["area_ha"]
    # Row 0 north: the highest ground (north-east) is at low row, high column.
    r, c = np.unravel_index(np.nanargmax(grid.z), grid.shape)
    assert r < grid.rows / 2 and c > grid.cols / 2


def test_outlier_spikes_are_removed():
    ds, truth = synthetic_terrain()
    rng = np.random.default_rng(5)
    idx = rng.choice(len(ds), 10, replace=False)
    ds.df.loc[idx, sch.ELEVATION] += 25.0
    grid, report = grid_from_points(ds)
    assert report["outliers_removed"] >= 10
    assert report["outliers_removed"] < 0.01 * len(ds)
    assert report["points_used"] == report["points_total"] - report["outliers_removed"]
    x = ds.df[sch.X].to_numpy()[idx]
    y = ds.df[sch.Y].to_numpy()[idx]
    err = grid.sample(x, y) - truth["surface"](x, y)
    assert np.nanmax(np.abs(err)) < 0.5
    raw, _ = grid_from_points(ds, remove_outliers=False)
    err_raw = raw.sample(x, y) - truth["surface"](x, y)
    assert np.nanmax(np.abs(err_raw)) > 1.0


def test_pass_offsets_are_reduced():
    ds, truth = synthetic_terrain(pass_offset_m=0.6, noise_m=0.2)
    # No smoothing, so the comparison isolates the detrending itself.
    with_detrend, report = grid_from_points(ds, cell_m=3.0, smooth_m=0.0)
    without, report_raw = grid_from_points(ds, cell_m=3.0, smooth_m=0.0, detrend_passes=False)
    before = _pass_bias_sd(without, ds, truth)
    after = _pass_bias_sd(with_detrend, ds, truth)
    assert after < 0.3 * before
    assert report["detrended"] is True and report["passes"] == truth["n_passes"]
    assert report_raw["detrended"] is False and report_raw["pass_offset_sd_m"] is None
    true_sd = float(np.std(truth["pass_offsets_m"]))
    assert report["pass_offset_sd_m"] == pytest.approx(true_sd, rel=0.3)


def test_detrend_is_skipped_with_one_pass():
    ds, _ = synthetic_terrain(size_m=400)
    ds.df[sch.PASS] = 0
    grid, report = grid_from_points(ds)
    assert report["passes"] == 1
    assert report["detrended"] is False and report["pass_offset_sd_m"] is None
    ds.df.drop(columns=[sch.PASS], inplace=True)
    _, report = grid_from_points(ds)
    assert report["passes"] == 0 and report["detrended"] is False


def test_auto_cell_size_and_cap(field):
    _, _, grid, report = field
    assert report["cell_m"] == 5.0 and grid.cell == 5.0
    assert report["max_gap_m"] == pytest.approx(15.0)
    assert report["edge_gap_m"] == pytest.approx(4.5)
    # 30 cm of noise on a 5 m grid needs about two cells of smoothing.
    assert report["smooth_m"] == pytest.approx(10.0, abs=1.0)
    wide, _ = synthetic_terrain(size_m=600, swath_m=30.0, spacing_m=3.0)
    _, wide_report = grid_from_points(wide)
    assert wide_report["cell_m"] == 15.0
    assert wide_report["max_gap_m"] == pytest.approx(45.0)
    capped, capped_report = grid_from_points(wide, cell_m=1.0, max_cells=50_000)
    assert capped_report["rows"] * capped_report["cols"] <= 50_000
    assert capped.cell > 1.0 and capped_report["cell_m"] == capped.cell
    # The note says which limit was hit, in the reader's own thousands.
    assert any(f"more than {Phrase().number(50_000)} cells" in note
               for note in capped_report["notes"])
    explicit, explicit_report = grid_from_points(wide, cell_m=8.0, smooth_m=0.0, max_gap_m=20.0)
    assert explicit.cell == 8.0 and explicit_report["max_gap_m"] == 20.0
    assert explicit_report["smooth_m"] == 0.0


def test_too_few_elevations_is_a_clear_error():
    ds, _ = synthetic_terrain(size_m=300)
    no_elev = ds.copy()
    no_elev.df = no_elev.df.drop(columns=[sch.ELEVATION])
    with pytest.raises(ValueError, match="elevation"):
        grid_from_points(no_elev)
    sparse = ds.copy()
    sparse.df.loc[20:, sch.ELEVATION] = np.nan
    with pytest.raises(ValueError, match="DEM"):
        grid_from_points(sparse)
    # Points exactly on a line are not a surface.
    line = ds.subset(ds.df[sch.PASS].to_numpy() == 0)
    line.df[sch.X] = float(line.df[sch.X].iloc[0])
    with pytest.raises(ValueError, match="line"):
        grid_from_points(line)


def test_report_keys_and_plain_types(field):
    _, _, _, report = field
    assert set(report) == REPORT_KEYS
    json.dumps(report)
    assert isinstance(report["rows"], int) and isinstance(report["cols"], int)
    assert isinstance(report["cell_m"], float) and isinstance(report["detrended"], bool)
    assert isinstance(report["notes"], list)
    assert report["crs"].startswith("EPSG:")
    assert report["points_total"] >= report["points_used"] > 0


def test_projects_when_metric_coordinates_are_missing():
    ds, truth = synthetic_terrain(size_m=400)
    raw = Dataset(ds.df[[sch.LON, sch.LAT, sch.ELEVATION, sch.PASS, sch.SWATH]].copy(), DatasetMeta())
    raw.df = raw.df.drop(columns=[sch.X, sch.Y])
    raw.metric_crs = None
    grid, report = grid_from_points(raw)
    assert report["crs"] == truth["crs"]
    assert sch.X in raw.df.columns


def test_gridding_120k_points_is_fast():
    ds, _ = synthetic_terrain(size_m=1550)
    assert len(ds) >= 120_000
    start = time.perf_counter()
    grid, report = grid_from_points(ds)
    elapsed = time.perf_counter() - start
    assert elapsed < 15.0, f"gridding {len(ds)} points took {elapsed:.1f} s"
    assert report["rows"] * report["cols"] <= 400_000
    assert grid.valid_count > 0


# ==========================================================================
# Fixes: smoothing, the field edge, memory, messages
# ==========================================================================

def _plane_field(**kw):
    """A noiseless 3 % plane falling east, logged without pass offsets."""
    return synthetic_terrain(
        noise_m=0.0, pass_offset_m=0.0, hills=(), valley=None, depression=None,
        tilt=(0.03, 90.0), notch=None, **kw,
    )


def test_noiseless_points_are_not_smoothed():
    # The default smoothing is set by the noise the file actually carries:
    # a clean file keeps its potholes at full depth instead of losing a
    # third of them to a fixed 10 m Gaussian.
    ds, truth = synthetic_terrain(noise_m=0.0, pass_offset_m=0.0, hills=(), valley=None, notch=None)
    grid, report = grid_from_points(ds, detrend_passes=False, remove_outliers=False)
    assert report["smooth_m"] == 0.0
    assert report["point_noise_m"] < 0.01
    assert not any("smoothed" in note for note in report["notes"])
    bottom = truth["depression"]["bottom_m"]
    assert grid.sample(*bottom) == pytest.approx(truth["depression"]["bottom_z_m"], abs=0.02)
    X, Y = grid.xy()
    err = grid.z[grid.mask] - truth["surface"](X[grid.mask], Y[grid.mask])
    assert np.sqrt(np.mean(err ** 2)) < 0.02


def test_default_smoothing_follows_the_noise():
    quiet, _ = synthetic_terrain(size_m=400, noise_m=0.1, pass_offset_m=0.0)
    noisy, _ = synthetic_terrain(size_m=400, noise_m=0.3, pass_offset_m=0.0)
    _, quiet_report = grid_from_points(quiet)
    _, noisy_report = grid_from_points(noisy)
    assert quiet_report["point_noise_m"] == pytest.approx(0.1, rel=0.3)
    assert noisy_report["point_noise_m"] == pytest.approx(0.3, rel=0.3)
    assert 0 < quiet_report["smooth_m"] < noisy_report["smooth_m"]
    # The user is told that the surface was smoothed and what it costs,
    # with the scale the report holds, written in the set this call asked
    # for (none, so the metric store).
    smoothing = next(n for n in noisy_report["notes"] if "potholes" in n)
    assert f"smoothed over {Phrase().length(noisy_report['smooth_m'])}" in smoothing
    # An explicit value is taken as given, and a coarse cell needs less
    # smoothing in cells because it already spans more relief.
    _, coarse = grid_from_points(noisy, cell_m=20.0)
    assert coarse["smooth_m"] <= noisy_report["smooth_m"] + 1.0


def test_grid_stops_at_the_field_edge():
    # Only the half swath the header covered lies beyond the outer track;
    # the old 1.5-swath rim of copied readings overstated the area by 7 %
    # and put a flat, zero-slope band around every field.
    ds, truth = _plane_field()
    grid, report = grid_from_points(ds, detrend_passes=False, remove_outliers=False)
    assert report["area_ha"] == pytest.approx(truth["area_ha"], rel=0.01)
    X, Y = grid.xy()
    outside = grid.mask & ~truth["inside"](X, Y)
    assert outside.sum() < 0.01 * grid.valid_count
    # The band beyond the outer track continues the slope, so the plane is
    # exact right up to the edge.
    err = grid.z[grid.mask] - truth["plane"](X[grid.mask], Y[grid.mask])
    assert np.abs(err).max() < 1e-6
    # The default field (with its notch) comes out at its true area too.
    ds, truth = synthetic_terrain()
    _, report = grid_from_points(ds)
    assert report["area_ha"] == pytest.approx(truth["area_ha"], rel=0.02)


def test_skipped_strip_is_bridged_but_a_notch_is_not(field):
    ds, truth, grid, report = field
    # A single skipped pass leaves no hole: the triangulation reaches
    # across it and the gap tolerance allows it.
    passes = ds.df[sch.PASS].to_numpy()
    skipped = ds.subset(passes != 40)
    thin, _ = grid_from_points(skipped)
    assert thin.valid_count == pytest.approx(grid.valid_count, rel=0.01)
    # The notch, spanned by long triangles, stays empty right up to its edge.
    ox, oy = truth["origin_m"]
    size = truth["size_m"]
    X, Y = grid.xy()
    notch = (X < ox + 0.25 * size - 10) & (Y > oy + 0.75 * size + 10) & (X > ox) & (Y < oy + size)
    assert np.isnan(grid.z[notch]).all()


def test_large_cells_and_dense_files_do_not_exhaust_memory(monkeypatch):
    # The pass search radius follows the swath, not the cell: a 250 m
    # 'overview' cell used to ask for every pair of points in the file.
    ds, truth = synthetic_terrain()
    start = time.perf_counter()
    grid, report = grid_from_points(ds, cell_m=250.0)
    assert time.perf_counter() - start < 5.0
    assert report["detrended"] is True and grid.cell == 250.0
    # A swath column in the wrong unit is capped and reported, not obeyed.
    wrong = ds.copy()
    wrong.df[sch.SWATH] = 5000.0
    _, wrong_report = grid_from_points(wrong)
    assert wrong_report["cell_m"] <= 25.0 and wrong_report["max_gap_m"] <= 90.0
    assert any(f"swath width reads {Phrase().length(5000.0, None)}" in note
               for note in wrong_report["notes"])
    # When the pairs would not fit, the offsets are estimated on a sample
    # and still remove the bulk of the pass-to-pass shift.
    monkeypatch.setattr(grid_mod, "_MAX_PAIRS", 50_000)
    ds, truth = synthetic_terrain(pass_offset_m=0.6, noise_m=0.2)
    sampled, sampled_report = grid_from_points(ds, cell_m=3.0, smooth_m=0.0)
    without, _ = grid_from_points(ds, cell_m=3.0, smooth_m=0.0, detrend_passes=False)
    assert sampled_report["detrended"] is True
    assert _pass_bias_sd(sampled, ds, truth) < 0.5 * _pass_bias_sd(without, ds, truth)


def test_coordinate_and_coverage_problems_are_explained():
    ds, _ = synthetic_terrain(size_m=400)
    # Latitude and longitude that parse to nothing.
    df = ds.df.drop(columns=[sch.X, sch.Y]).copy()
    df[sch.LON] = np.nan
    df[sch.LAT] = np.nan
    with pytest.raises(ValueError, match="empty in every row"):
        grid_from_points(Dataset(df, DatasetMeta(name="nan")))
    # Latitude and longitude swapped project to infinity.
    swapped = ds.copy()
    swapped.df[sch.LON] = ds.df[sch.LAT].to_numpy()
    swapped.df[sch.LAT] = ds.df[sch.LON].to_numpy()
    swapped.project()
    with pytest.raises(ValueError, match="swapped"):
        grid_from_points(swapped)
    # Too few readings says how many there are and how many are needed.
    rng = np.random.default_rng(3)
    few = ds.subset(np.isin(np.arange(len(ds)), rng.choice(len(ds), 29, replace=False)))
    with pytest.raises(ValueError, match="29 of the 29 records.*at least 30"):
        grid_from_points(few)
    # Readings scattered too far apart are not a survey of the ground.
    sparse = ds.subset(np.isin(np.arange(len(ds)), rng.choice(len(ds), 30, replace=False)))
    with pytest.raises(ValueError, match="too far"):
        grid_from_points(sparse)


def test_notes_on_a_strip_and_on_separate_blocks():
    ds, _ = synthetic_terrain()
    one = ds.subset(ds.df[sch.PASS].to_numpy() == 5)
    grid, report = grid_from_points(one)
    assert grid.valid_count > 0
    assert any("strip" in note for note in report["notes"])
    # Two fields 35 km apart in one file: the cell is widened to span them
    # and the user is told the numbers mix two blocks of ground.
    two = ds.copy()
    two.df.loc[::2, sch.LON] += 0.5
    two.metric_crs = None
    two.project()
    _, report = grid_from_points(two)
    assert report["cell_m"] > 5.0
    assert any(f"widened from {Phrase().length(5.0, None)}" in note for note in report["notes"])
    blocks = next(f for f in report["note_facts"] if f["key"] == "blocks")
    assert blocks["blocks"] == 2 and blocks["apart_m"] > 30_000.0
    said = next(n for n in report["notes"] if "separate blocks" in n)
    assert f"about {Phrase().length(blocks['apart_m'], 0)} apart" in said
    # A whole field has neither note.
    _, plain = grid_from_points(ds)
    assert not any("strip" in note or "blocks" in note for note in plain["notes"])


def test_a_note_is_written_again_in_the_reader_s_units():
    """The same gridding, two readers: the unit moves, the ground does not.

    A note is stored as the fact it states, so saying it in another unit
    set is writing it again from the same metric numbers — not measuring
    anything again, and not touching what was measured.
    """
    ds, _ = synthetic_terrain(size_m=400, noise_m=0.3, pass_offset_m=0.0)
    _, report = grid_from_points(ds)
    facts = report["note_facts"]
    smooth_m = next(f for f in facts if f["key"] == "smoothing")["smooth_m"]
    assert smooth_m > 0

    metric = notes_mod.render(facts, UNIT_PRESETS["metric"])
    canadian = notes_mod.render(facts, UNIT_PRESETS["canada"])
    assert len(metric) == len(canadian) == len(report["notes"])

    assert f"smoothed over {smooth_m:.1f} m" in next(n for n in metric if "potholes" in n)
    assert f"smoothed over {smooth_m / 0.3048:.1f} ft" in \
        next(n for n in canadian if "potholes" in n)
    # Nothing in the Canadian reading is left in metres, and the fact the
    # two sentences were written from is untouched by either.
    assert not any(" m " in n or n.endswith(" m") for n in canadian)
    assert next(f for f in report["note_facts"] if f["key"] == "smoothing")["smooth_m"] \
        == smooth_m


# ==========================================================================
# Readings that are no height at all, and altitudes in whole metres
# ==========================================================================

def _brief(ds) -> dict:
    """Hills, depressions and character of a full analysis, for comparison."""
    from agrosuite.terrain import analyze

    s = analyze(ds).summary()
    return {
        "hills": len(s["features"]["hills"]),
        "depressions": len(s["features"]["depressions"]),
        "character": s["character"]["key"],
        "p95": s["slope"]["p95_pct"],
        "source": s["source"],
    }


def test_a_pass_logged_at_zero_is_dropped_not_blended():
    """A receiver that lost its fix for one run logs 0 m; every neighbour on
    the pass agrees, so the spike filter keeps it, and blending a 700 m
    offset into the field turned a gentle field 'hilly' with 37 % steep."""
    clean, _ = synthetic_terrain()
    ds, _ = synthetic_terrain()
    ds.df.loc[ds.df[sch.PASS] == 40, sch.ELEVATION] = 0.0
    _, report = grid_from_points(ds)
    assert report["passes_dropped"] == 1 and report["passes"] == 88
    assert report["fill_values_removed"] == int((ds.df[sch.PASS] == 40).sum())
    assert any("exactly 0 m" in n for n in report["notes"])

    expected = _brief(clean)
    got = _brief(ds)
    assert got["hills"] == expected["hills"] == 2
    assert got["depressions"] == expected["depressions"]
    assert got["character"] == expected["character"] == "gently_undulating"
    assert got["p95"] == pytest.approx(expected["p95"], abs=0.3)
    assert got["source"]["passes_dropped"] == 1


def test_no_data_values_and_a_stray_pass_are_dropped():
    ds, truth = synthetic_terrain()
    ds.df.loc[ds.df[sch.PASS] == 40, sch.ELEVATION] = -9999.0
    ds.df.loc[ds.df[sch.PASS] == 41, sch.ELEVATION] += 120.0  # the last height it knew
    grid, report = grid_from_points(ds)
    assert report["passes_dropped"] == 2 and report["passes"] == truth["n_passes"] - 2
    assert report["points_used"] == report["points_total"] - 800 - report["outliers_removed"]
    assert any("pass 41 (8" in n and "was logged at an altitude far from" in n for n in report["notes"])
    x = ds.df[sch.X].to_numpy()
    y = ds.df[sch.Y].to_numpy()
    err = grid.sample(x, y) - truth["surface"](x, y)
    assert np.nanpercentile(np.abs(err), 99) < 1.0

    # A field that really lies at sea level keeps its zeros: they are heights.
    coastal, _ = synthetic_terrain(base_m=0.6, size_m=400)
    coastal.df.loc[coastal.df[sch.PASS] == 20, sch.ELEVATION] = 0.0
    _, report = grid_from_points(coastal)
    assert report["fill_values_removed"] == 0 and report["passes_dropped"] == 0

    # Two passes cannot say which one is wrong, so neither is dropped.
    pair, _ = synthetic_terrain(size_m=200, swath_m=100.0)
    pair.df.loc[pair.df[sch.PASS] == 1, sch.ELEVATION] += 200.0
    _, report = grid_from_points(pair)
    assert report["passes"] == 2 and report["passes_dropped"] == 0

    # Scattered lost-fix readings, not a whole pass, go the same way.
    ds, _ = synthetic_terrain(size_m=400)
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ds), 50, replace=False)
    ds.df.loc[idx, sch.ELEVATION] = 0.0
    _, report = grid_from_points(ds)
    assert report["fill_values_removed"] == 50 and report["passes_dropped"] == 0


def test_value_step_helper():
    rng = np.random.default_rng(0)
    continuous = 700.0 + rng.normal(0.0, 3.0, 500)
    assert grid_mod.value_step(continuous) == 0.0
    assert grid_mod.value_step(np.round(continuous)) == 1.0
    assert grid_mod.value_step(np.round(continuous * 2.0) / 2.0) == 0.5
    assert grid_mod.value_step(np.round(continuous, 1)) == 0.1
    assert grid_mod.value_step(np.round(continuous).astype("float32")) == 1.0
    # Few distinct values are no objection: a flat field in whole metres.
    assert grid_mod.value_step(np.full(200, 700.0) + rng.integers(0, 4, 200)) == 1.0
    # A handful of readings can land on whole metres by chance; NaN does not count.
    assert grid_mod.value_step(np.array([700.0, 701.0, 703.0])) == 0.0
    assert grid_mod.value_step(np.array([np.nan] * 60)) == 0.0
    assert grid_mod.value_step(np.array([700.0, 701.0, 703.0]), min_samples=1) == 1.0


def test_whole_metre_altitude_is_not_read_as_spikes():
    """Rounded to the metre, the neighbour residuals are mostly exactly 0,
    the MAD collapses and the 0.5 m floor threw out 23 % of the readings
    (7 hills and 30 depressions reported instead of 2 and 3)."""
    ds, _ = synthetic_terrain()
    ds.df[sch.ELEVATION] = np.round(ds.df[sch.ELEVATION])
    _, report = grid_from_points(ds)
    assert report["value_step_m"] == 1.0
    assert report["outliers_removed"] < 0.01 * report["points_total"]
    assert report["smooth_m"] >= 10.0
    assert any("steps of 1 m" in n for n in report["notes"])

    got = _brief(ds)
    assert got["hills"] == 2, got
    assert got["depressions"] <= 3, got
    assert got["source"]["value_step_m"] == 1.0
    assert got["source"]["kind"] == "points"

    # Decimetres: a quiet receiver's residuals collapse the same way.
    ds, _ = synthetic_terrain(noise_m=0.05)
    ds.df[sch.ELEVATION] = np.round(ds.df[sch.ELEVATION], 1)
    _, report = grid_from_points(ds)
    assert report["value_step_m"] == 0.1 and report["smooth_m"] >= 1.0
    assert report["outliers_removed"] < 0.01 * report["points_total"]
