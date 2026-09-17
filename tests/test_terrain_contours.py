"""Tests for the contour lines and the elevation profile.

A contour of a known surface has a known shape: on a tilted plane the lines
are straight, parallel, perpendicular to the fall and ``interval / gradient``
apart; on a Gaussian hill they are rings of a computable area. Those are the
checks here, plus the two contracts every consumer relies on — no vertex
ever lands on a masked cell, and the GeoJSON is plain enough to dump, load
and hand to geopandas — and the profile, which on a plane must come back as
a straight line of exactly the plane's slope.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.terrain import ElevationGrid, grid_from_points, synthetic_dem, synthetic_terrain
from agrosuite.terrain.contours import (
    INTERVALS,
    MAX_LINES,
    MIN_LINES,
    choose_interval,
    contours,
    contours_to_geodataframe,
    profile,
)

CELL = 5.0
N = 100
X0, Y0 = 500_000.0, 5_700_000.0
CRS = "EPSG:32612"

GRADIENT, FALL_DEG = 0.05, 135.0
HILL_H, HILL_S = 10.0, 50.0


def _grid(z: np.ndarray) -> ElevationGrid:
    return ElevationGrid(z, X0, Y0, CELL, CRS)


def _xy() -> tuple[np.ndarray, np.ndarray]:
    return _grid(np.zeros((N, N))).xy()


def _fall_unit(direction_deg: float = FALL_DEG) -> tuple[float, float]:
    return math.sin(math.radians(direction_deg)), math.cos(math.radians(direction_deg))


def _plane(gradient: float = GRADIENT, direction_deg: float = FALL_DEG) -> ElevationGrid:
    """A plane falling ``gradient`` (m/m) toward ``direction_deg``, about 700 m."""
    xx, yy = _xy()
    ux, uy = _fall_unit(direction_deg)
    return _grid(700.0 - gradient * (ux * (xx - X0) + uy * (yy - Y0)))


def _hill() -> ElevationGrid:
    """A Gaussian hill centred on cell (50, 50), no tilt."""
    xx, yy = _xy()
    dist = np.hypot(xx - xx[50, 50], yy - yy[50, 50])
    return _grid(HILL_H * np.exp(-dist ** 2 / (2.0 * HILL_S ** 2)))


def _hill_radius(level: float) -> float:
    return HILL_S * math.sqrt(2.0 * math.log(HILL_H / level))


def _metric(feature: dict, grid: ElevationGrid) -> np.ndarray:
    """A feature's vertices back in the grid's metric CRS, as (x, y) columns."""
    coords = np.asarray(feature["geometry"]["coordinates"], dtype="float64")
    x, y = grid.from_lonlat(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def _all_vertices_valid(features: list[dict], grid: ElevationGrid) -> bool:
    xy = np.concatenate([_metric(f, grid) for f in features])
    hit = grid.sample(xy[:, 0], xy[:, 1], values=grid.mask.astype("float64"), order=0)
    return bool(np.all(hit == 1.0))


def _is_closed(feature: dict) -> bool:
    coords = feature["geometry"]["coordinates"]
    return coords[0] == coords[-1]


# ==========================================================================
# Interval
# ==========================================================================

def test_choose_interval_at_the_boundaries():
    assert choose_interval(0.3) == 0.1      # just enough relief for 3 lines
    assert choose_interval(3.0) == 0.25     # exactly 12 lines
    assert choose_interval(40.0) == 2.5     # 16 lines beats 8 on a tie, finer wins
    assert choose_interval(400.0) == 20.0   # 20 lines; 10 m would give 40
    # Beyond the range no step fits; the nearest end of the ladder is used.
    assert choose_interval(0.05) == 0.1
    assert choose_interval(5000.0) == 20.0
    assert choose_interval(0.0) == 0.1
    assert choose_interval(float("nan")) == 0.1
    # The target moves the choice within the same bounds.
    assert choose_interval(40.0, target_count=40) == 1.0
    assert choose_interval(3.0, target_count=3) == 1.0


def test_choose_interval_keeps_count_within_bounds():
    for relief in np.geomspace(0.3, 800.0, 60):
        interval = choose_interval(float(relief))
        assert interval in INTERVALS
        count = relief / interval
        assert MIN_LINES - 1e-6 <= count <= MAX_LINES + 1e-6


# ==========================================================================
# Contours on known surfaces
# ==========================================================================

def test_plane_contours_are_straight_parallel_and_evenly_spaced():
    grid = _plane()
    features, interval = contours(grid, interval_m=1.0)
    assert interval == 1.0 and len(features) > 20
    ux, uy = _fall_unit()

    by_level: dict[float, list[np.ndarray]] = {}
    for f in features:
        xy = _metric(f, grid)
        level = f["properties"]["level_m"]
        by_level.setdefault(level, []).append(xy)
        # Every vertex sits on the plane at its own level.
        np.testing.assert_allclose(grid.sample(xy[:, 0], xy[:, 1]), level, atol=0.01)
        # Straight: no vertex leaves the chord by more than a hair.
        chord = xy[-1] - xy[0]
        chord /= np.hypot(*chord)
        rel = xy - xy[0]
        deviation = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0])
        assert deviation.max() < 0.05
        # Perpendicular to the fall.
        assert abs(chord[0] * ux + chord[1] * uy) < math.sin(math.radians(0.5))

    # Levels are integers (multiples of the interval), indexed in order.
    levels = sorted(by_level)
    assert all(float(lvl).is_integer() for lvl in levels)
    indexes = {f["properties"]["level_m"]: f["properties"]["index"] for f in features}
    assert [indexes[lvl] for lvl in levels] == list(range(len(levels)))
    majors = {f["properties"]["level_m"]: f["properties"]["major"] for f in features}
    assert all(majors[lvl] == (lvl % 5 == 0) for lvl in levels)

    # Evenly spaced: interval / gradient apart along the fall, higher
    # levels uphill (smaller projection on the fall direction).
    along = {lvl: np.mean([np.mean(xy[:, 0] * ux + xy[:, 1] * uy) for xy in xys])
             for lvl, xys in by_level.items()}
    expected = 1.0 / GRADIENT
    for lo, hi in zip(levels[:-1], levels[1:]):
        assert along[lo] - along[hi] == pytest.approx(expected, rel=0.05)


def test_hill_contours_are_closed_rings_shrinking_with_level():
    grid = _hill()
    features, interval = contours(grid)
    assert interval == 1.0
    levels = [f["properties"]["level_m"] for f in features]
    assert levels == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert [f["properties"]["index"] for f in features] == list(range(9))
    assert all(_is_closed(f) for f in features)

    areas = [Polygon(_metric(f, grid)).area for f in features]
    assert all(a > b for a, b in zip(areas[:-1], areas[1:]))
    for level, area in zip(levels, areas):
        if level <= 8.0:  # the 9 m ring is 23 m across, five cells: too coarse to score
            assert area == pytest.approx(math.pi * _hill_radius(level) ** 2, rel=0.06)


def test_masked_hole_splits_lines_and_gets_no_vertices():
    grid = _plane()
    z = grid.z.copy()
    z[40:50, 40:50] = np.nan
    grid = _grid(z)
    features, _ = contours(grid, interval_m=1.0)
    assert _all_vertices_valid(features, grid)
    pieces_per_level: dict[float, int] = {}
    for f in features:
        pieces_per_level[f["properties"]["level_m"]] = (
            pieces_per_level.get(f["properties"]["level_m"], 0) + 1
        )
    # Lines that crossed the hole now come in two pieces.
    assert max(pieces_per_level.values()) >= 2


def test_ring_cut_once_comes_back_as_one_open_line():
    grid = _hill()
    z = grid.z.copy()
    z[48:53, 60:] = np.nan  # a strip from east of the summit to the edge
    grid = _grid(z)
    features, _ = contours(grid, interval_m=1.0)
    assert _all_vertices_valid(features, grid)
    by_level: dict[float, list[dict]] = {}
    for f in features:
        by_level.setdefault(f["properties"]["level_m"], []).append(f)
    for level in (1.0, 2.0, 3.0, 4.0, 5.0):  # rings wider than the strip's start
        assert len(by_level[level]) == 1
        assert not _is_closed(by_level[level][0])
    for level in (7.0, 8.0, 9.0):  # rings that never reach the strip
        assert len(by_level[level]) == 1
        assert _is_closed(by_level[level][0])


def test_notched_dem_has_no_vertex_in_the_notch():
    grid, truth = synthetic_dem()
    features, interval = contours(grid)
    assert features and interval == 1.0
    assert _all_vertices_valid(features, grid)
    xy = np.concatenate([_metric(f, grid) for f in features])
    xmin, ymin, xmax, ymax = truth["notch"]["bounds_m"]
    # A vertex may sit up to half a cell past its cell's centre, never deeper.
    half = 0.5 * grid.cell
    inside = (
        (xy[:, 0] > xmin + half) & (xy[:, 0] < xmax - half)
        & (xy[:, 1] > ymin + half) & (xy[:, 1] < ymax - half)
    )
    assert not inside.any()
    # But the lines do reach the field: some vertex is within a cell of the notch.
    near = (xy[:, 0] < xmax + grid.cell) & (xy[:, 1] > ymin - grid.cell)
    assert near.any()


def test_contours_of_a_gridded_yield_file():
    ds, _truth = synthetic_terrain()
    grid, _report = grid_from_points(ds)
    features, interval = contours(grid)
    assert features and interval in INTERVALS
    assert _all_vertices_valid(features, grid)
    for f in features:
        assert (f["properties"]["level_m"] / interval) == pytest.approx(
            round(f["properties"]["level_m"] / interval), abs=1e-6
        )
    json.dumps(features)


# ==========================================================================
# Options and errors
# ==========================================================================

def test_explicit_interval_values_layer_and_errors():
    grid = _hill()
    features, interval = contours(grid, interval_m=2.0)
    assert interval == 2.0
    assert [f["properties"]["level_m"] for f in features] == [2.0, 4.0, 6.0, 8.0]
    assert [f["properties"]["major"] for f in features] == [False, False, False, False]
    features, _ = contours(grid, interval_m=2.5)
    assert [f["properties"]["level_m"] for f in features] == [2.5, 5.0, 7.5]
    # The 0 level is a multiple of 5 intervals; nothing else in a 10 m hill is.
    features, _ = contours(grid.with_values(grid.z - 5.0), interval_m=1.0)
    majors = [f["properties"]["level_m"] for f in features if f["properties"]["major"]]
    assert majors == [0.0]

    # Another layer on the same geometry, with its own NaN treated as outside.
    layer = grid.z.copy()
    layer[:, :50] = np.nan
    features, _ = contours(grid, interval_m=1.0, values=layer)
    xy = np.concatenate([_metric(f, grid) for f in features])
    assert xy[:, 0].min() > X0 + 50 * CELL - 0.5 * CELL
    assert not any(_is_closed(f) for f in features)

    # Nothing to draw on a flat field, and no crash.
    assert contours(_grid(np.full((N, N), 700.0)))[0] == []
    assert contours(_grid(np.full((N, N), np.nan)))[0] == []

    with pytest.raises(ValueError, match="positive"):
        contours(grid, interval_m=0.0)
    with pytest.raises(ValueError, match="coarser"):
        contours(grid, interval_m=0.001)
    with pytest.raises(ValueError, match="same grid"):
        contours(grid, values=np.zeros((3, 3)))


def test_no_level_within_rounding_noise_of_an_extreme():
    """A plateau at exactly a level, a float's width above the minimum or
    below the maximum, must not become hundreds of loops around cells that
    differ by 1e-13 m: no level is drawn within 1e-6 of the relief of
    either extreme."""
    xx, _ = _xy()
    plane = 690.0 + 0.02 * (xx - X0)          # 690 m at the west edge, 700 m near col 100
    z = np.minimum(plane, 700.0)              # capped: a plateau at exactly 700 m
    rng = np.random.default_rng(1)
    plateau = plane >= 700.0
    z[plateau] += rng.uniform(-1e-13, 1e-13, int(plateau.sum()))
    grid = _grid(z)
    features, interval = contours(grid, interval_m=1.0)
    levels = sorted({f["properties"]["level_m"] for f in features})
    assert 700.0 not in levels
    assert levels == [float(k) for k in range(691, 700)]
    assert all(not _is_closed(f) for f in features)  # straight lines, no noise rings
    # The same at the bottom: a floor a float's width under the minimum level.
    z = np.maximum(plane, 691.0)
    floor = plane <= 691.0
    z[floor] += rng.uniform(-1e-13, 1e-13, int(floor.sum()))
    levels = sorted({f["properties"]["level_m"] for f in contours(_grid(z), interval_m=1.0)[0]})
    assert 691.0 not in levels and 692.0 in levels


# ==========================================================================
# Profile
# ==========================================================================

def test_profile_along_the_fall_line_of_a_plane():
    grid = _plane()
    xx, yy = grid.xy()
    start = grid.to_lonlat(xx[10, 10], yy[10, 10])  # north-west, uphill
    end = grid.to_lonlat(xx[90, 90], yy[90, 90])    # south-east, downhill
    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))

    p = profile(grid, [start, end], n=81)
    assert set(p) == {"distance_m", "elev_m", "slope_pct", "points", "length_m"}
    assert len(p["distance_m"]) == len(p["elev_m"]) == len(p["slope_pct"]) == 81
    assert p["length_m"] == pytest.approx(80 * CELL * math.sqrt(2.0))
    assert p["distance_m"][0] == 0.0 and p["distance_m"][-1] == pytest.approx(p["length_m"])
    assert p["points"] == [list(np.round(start, 7)), list(np.round(end, 7))]
    elev = np.array(p["elev_m"], dtype="float64")
    assert np.isfinite(elev).all()
    # A straight line of the plane's gradient, walking downhill.
    np.testing.assert_allclose(elev, elev[0] - GRADIENT * np.array(p["distance_m"]), atol=0.01)
    assert p["slope_pct"][0] is None
    np.testing.assert_allclose(p["slope_pct"][1:], 100.0 * GRADIENT, atol=0.05)
    # Walking the other way the slope changes sign.
    back = profile(grid, [end, start], n=81)
    np.testing.assert_allclose(back["slope_pct"][1:], -100.0 * GRADIENT, atol=0.05)
    json.dumps(p)


def test_profile_outside_the_field_is_null():
    grid = _plane()
    xx, yy = grid.xy()
    outside = grid.to_lonlat(X0 - 200.0, yy[10, 10])
    inside = grid.to_lonlat(xx[90, 10], yy[10, 10])
    p = profile(grid, [(float(outside[0]), float(outside[1])), (float(inside[0]), float(inside[1]))], n=50)
    nulls = [v is None for v in p["elev_m"]]
    assert nulls[0] and not nulls[-1]
    assert sum(nulls) == pytest.approx(50 * 200.0 / p["length_m"], abs=2)
    # The step into the field has no slope; the first station may sit on the
    # outer half-cell rim, which samples the edge cell, so the plane's slope
    # (its east component, this line runs east) shows from the step after.
    first = nulls.index(False)
    assert p["slope_pct"][first] is None
    assert p["slope_pct"][first + 2] == pytest.approx(
        100.0 * GRADIENT * math.sin(math.radians(FALL_DEG)), abs=0.05
    )
    json.dumps(p)


def test_profile_entirely_outside_the_field_is_refused():
    """A line with under two stations on the field is a missed click, not a
    profile: 200 nulls would chart as a flat field."""
    grid = _plane()
    _, yy = grid.xy()
    a = grid.to_lonlat(X0 - 500.0, yy[10, 10])
    b = grid.to_lonlat(X0 - 300.0, yy[40, 10])
    with pytest.raises(ValueError, match="does not cross the field"):
        profile(grid, [(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))], n=50)
    # One station on the rim is still no profile; two are.
    edge = grid.to_lonlat(X0 + 0.25 * CELL, yy[10, 10])
    with pytest.raises(ValueError, match="does not cross the field"):
        profile(grid, [(float(a[0]), float(a[1])), (float(edge[0]), float(edge[1]))], n=200)
    inside = grid.to_lonlat(X0 + 1.5 * CELL, yy[10, 10])
    p = profile(grid, [(float(a[0]), float(a[1])), (float(inside[0]), float(inside[1]))], n=400)
    assert sum(v is not None for v in p["elev_m"]) >= 2


def test_profile_polyline_and_errors():
    grid = _plane()
    xx, yy = grid.xy()
    pts = [grid.to_lonlat(xx[r, c], yy[r, c]) for r, c in ((10, 10), (10, 60), (10, 60), (60, 60))]
    pts = [(float(lon), float(lat)) for lon, lat in pts]
    p = profile(grid, pts, n=101)
    assert p["length_m"] == pytest.approx(100 * CELL, rel=1e-6)
    assert len(p["points"]) == 4
    # Another layer can be profiled on the same grid.
    flat = profile(grid, pts, n=11, values=np.full(grid.shape, 3.0))
    assert flat["elev_m"] == [3.0] * 11 and flat["slope_pct"][1:] == [0.0] * 10

    with pytest.raises(ValueError, match="at least two points"):
        profile(grid, [pts[0]])
    with pytest.raises(ValueError, match="same place"):
        profile(grid, [pts[0], pts[0]])
    with pytest.raises(ValueError, match="stations"):
        profile(grid, pts, n=1)
    with pytest.raises(ValueError, match="no coordinates"):
        profile(grid, [pts[0], (float("nan"), 51.0)])


def test_profile_rejects_coordinates_pyproj_would_map_to_infinity():
    """Latitude 95 or longitude 1e300 are not refused by pyproj: they come
    back as inf, and the profile used to answer with an infinite length and
    every station sampling the one finite end. A third coordinate on a point
    was reported as 'fewer than two points'. Each case now says what is wrong."""
    grid = _plane()
    lon, lat = (float(v) for v in grid.to_lonlat(grid.x0 + 50 * CELL, grid.y0 - 50 * CELL))

    with pytest.raises(ValueError, match=r"latitude \(-90 to 90\)"):
        profile(grid, [(lon, 95.0), (lon, lat)])
    with pytest.raises(ValueError, match="metric x, y"):
        profile(grid, [(1e300, lat), (lon, lat)])
    with pytest.raises(ValueError, match="metric x, y"):
        profile(grid, [(grid.x0, grid.y0), (grid.x0 + 100.0, grid.y0)])
    with pytest.raises(ValueError, match="look swapped"):
        profile(grid, [(lat, lon), (lat + 0.001, lon)])
    with pytest.raises(ValueError, match="third"):
        profile(grid, [(lon, lat, 700.0), (lon + 0.001, lat, 700.0)])
    with pytest.raises(ValueError, match="third"):
        profile(grid, [(lon, lat), (lon + 0.001,)])
    with pytest.raises(ValueError, match="other than numbers"):
        profile(grid, [(lon, lat), ("east", lat)])

    # Every number the good case returns is finite, so a chart can draw it.
    p = profile(grid, [(lon, lat), (lon + 0.001, lat)], n=20)
    assert np.isfinite(p["length_m"]) and np.isfinite(p["distance_m"]).all()


# ==========================================================================
# GeoJSON and export
# ==========================================================================

def test_geojson_round_trips_through_json_and_geopandas(tmp_path):
    import geopandas as gpd

    grid, _ = synthetic_dem()
    features, _ = contours(grid, interval_m=1.0)
    text = json.dumps({"type": "FeatureCollection", "features": features})
    loaded = json.loads(text)
    assert all(f["geometry"]["type"] == "LineString" for f in loaded["features"])
    assert all(len(str(abs(c[0])).split(".")[1]) <= 7 for f in loaded["features"]
               for c in f["geometry"]["coordinates"])
    frame = gpd.GeoDataFrame.from_features(loaded["features"], crs="EPSG:4326")
    assert len(frame) == len(features)

    gdf = contours_to_geodataframe(features)
    assert list(gdf.columns) == ["level_m", "index", "major", "geometry"]
    assert gdf.crs.to_epsg() == 4326
    assert gdf["level_m"].dtype == "float64" and gdf["major"].dtype == "bool"
    assert set(gdf.geom_type) == {"LineString"}

    metric = contours_to_geodataframe(features, crs=grid.crs)
    assert metric.crs.to_epsg() == 32612
    xmin, ymin, xmax, ymax = grid.bounds_metric()
    assert metric.total_bounds[0] >= xmin and metric.total_bounds[2] <= xmax
    assert metric.total_bounds[1] >= ymin and metric.total_bounds[3] <= ymax

    out = tmp_path / "contours.geojson"
    gdf.to_file(out, driver="GeoJSON")
    back = gpd.read_file(out)
    assert len(back) == len(gdf)
    assert sorted(back["level_m"].unique()) == sorted(gdf["level_m"].unique())
    shp = tmp_path / "contours.shp"
    gdf.to_file(shp)
    assert len(gpd.read_file(shp)) == len(gdf)

    empty = contours_to_geodataframe([])
    assert len(empty) == 0 and list(empty.columns) == ["level_m", "index", "major", "geometry"]



# ==========================================================================
# The package namespace and what importing it costs
# ==========================================================================

def test_contours_submodule_is_not_shadowed_by_the_function():
    """``import agrosuite.terrain.contours as m`` must hand back the module;
    the function is re-exported under its own name, ``contour_lines``."""
    import importlib
    import types

    import agrosuite.terrain as terrain
    import agrosuite.terrain.contours as contours_module

    assert isinstance(contours_module, types.ModuleType)
    assert contours_module is importlib.import_module("agrosuite.terrain.contours")
    assert terrain.contours is contours_module
    assert terrain.contour_lines is contours_module.contours is contours
    assert "contour_lines" in terrain.__all__ and "contours" not in terrain.__all__


def test_importing_the_package_does_not_load_scipy_signal_or_skimage():
    """The server imports the terrain package at start-up; scipy.signal (and
    the scipy.stats it drags in) and skimage.measure cost over half a second
    and are only needed once a convolution or a contour is actually run."""
    import subprocess

    code = (
        "import sys; import agrosuite.terrain; "
        "print(sorted(m for m in sys.modules if m.startswith(('scipy.signal', 'skimage'))))"
    )
    root = str(Path(__file__).resolve().parents[1])
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "[]", out
    # And the lazy imports still work: a contour and a smoothing on demand.
    from agrosuite.terrain.grid import gaussian_smooth

    assert len(contours(_hill(), interval_m=2.0)[0]) == 4
    assert np.isfinite(gaussian_smooth(np.ones((5, 5)), 1.0)).all()
