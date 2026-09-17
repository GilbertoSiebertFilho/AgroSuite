"""Tests for the terrain hydrology: sink filling, routing, wetness.

Every test runs on a surface with a known answer — a plane, a crater
whose ponded volume is a closed formula, a V-shaped valley with one axis —
because "the water goes where the algorithm says" cannot be checked any
other way. The last tests use the synthetic DEM, where the truth dict
gives the bowl's spill level and volume, and pin the NaN contract every
layer must keep.
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

from agrosuite.terrain.grid import ElevationGrid
from agrosuite.terrain.hydrology import (
    D8_NAMES,
    D8_OFFSETS,
    FILL_EPSILON_M,
    Hydrology,
    analyse_hydrology,
    default_twi_threshold,
    depression_depth,
    depression_labels,
    depressions,
    drainage_lines,
    fill_sinks,
    flow_accumulation,
    flow_direction_d8,
    twi,
    wet_cells,
)

CRS = "EPSG:32612"
X0, Y0 = 500_000.0, 5_700_000.0


def _grid(n: int, cell: float) -> tuple[ElevationGrid, np.ndarray, np.ndarray]:
    """An empty n x n grid and its cell-centre coordinates."""
    grid = ElevationGrid(np.zeros((n, n)), X0, Y0, cell, CRS)
    X, Y = grid.xy()
    return grid, X, Y


def _plane(n: int = 120, cell: float = 5.0, gx: float = 0.01, gy: float = 0.01) -> ElevationGrid:
    """A plane rising ``gx`` per metre to the east and ``gy`` to the north,
    so with the defaults it falls to the south-west."""
    grid, X, Y = _grid(n, cell)
    grid.z = 700.0 + gx * (X - X0) + gy * (Y - (Y0 - n * cell))
    return grid


def _crater(n: int = 200, cell: float = 2.0, radius: float = 120.0, depth: float = 2.0):
    """A paraboloid bowl inside ``radius`` with a cone falling outward beyond
    it, so the rim is at exactly ``depth`` above the bottom and the ponded
    volume is pi k R^4 / 2."""
    grid, X, Y = _grid(n, cell)
    cx, cy = X0 + n * cell / 2, Y0 - n * cell / 2
    r = np.hypot(X - cx, Y - cy)
    k = depth / radius ** 2
    grid.z = 700.0 + np.where(r < radius, k * r ** 2, k * radius ** 2 - 0.02 * (r - radius))
    truth = {
        "centre": (cx, cy),
        "volume_m3": math.pi * k * radius ** 4 / 2,
        "area_ha": math.pi * radius ** 2 / 10_000,
        "rim": 700.0 + depth,
        "bottom": 700.0,
        "lake": r < radius,
    }
    return grid, truth


def _v_valley(n: int = 200, cell: float = 5.0):
    """A valley falling south whose floor is exactly one column of cells."""
    grid, X, Y = _grid(n, cell)
    axis_col = n // 2
    xa = X0 + (axis_col + 0.5) * cell
    grid.z = 700.0 + 0.01 * (Y - (Y0 - n * cell)) + 0.01 * np.abs(X - xa)
    return grid, xa, axis_col


def _receiver_elevation(fdir: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Elevation of the cell each cell drains to (NaN where it has none)."""
    out = np.full(z.shape, np.nan)
    rows, cols = z.shape
    for code, (dr, dc) in enumerate(D8_OFFSETS):
        r, c = np.nonzero(fdir == code)
        out[r, c] = z[r + dr, c + dc]
    return out


def _on_edge(grid: ElevationGrid) -> np.ndarray:
    """Valid cells with a NaN or off-array 8-neighbour."""
    from scipy import ndimage

    padded = np.pad(grid.mask, 1, constant_values=False)
    outside = ndimage.binary_dilation(~padded, structure=np.ones((3, 3), bool))
    return grid.mask & outside[1:-1, 1:-1]


# ==========================================================================
# Planes: nothing to fill, everything drains downhill
# ==========================================================================

def test_plane_has_no_depressions_and_drains_downhill():
    grid = _plane()
    filled = fill_sinks(grid)
    assert np.array_equal(filled, grid.z)
    assert depressions(grid, filled) == []

    fdir = flow_direction_d8(filled, grid)
    below = _receiver_elevation(fdir, filled)
    flows = fdir >= 0
    assert flows.any()
    assert np.all(below[flows] < filled[flows])
    # The plane falls to the south-west, so every interior cell goes SW.
    assert np.all(fdir[1:-1, 1:-1] == D8_NAMES.index("SW"))
    # Outlets only on the edge, and the downhill edges are outlets throughout.
    outlets = (fdir == -1) & grid.mask
    assert np.all(_on_edge(grid)[outlets])
    # (The north-west corner has no opposite cell to continue its SW
    # descent from, so it runs south along the edge instead.)
    assert np.all(outlets[-1, :]) and np.all(outlets[1:, 0])


def test_flat_ground_ramps_from_every_edge_and_ponds_nothing():
    """Dead-flat ground has no depression and drains to its nearest edge.

    The filling increment must climb from every edge at once, not from
    whichever edge cell was queued first: a single-corner ramp once made
    a flat 1 000 ha field report a 200 000 m3 'depression' in its far
    corner and drain the whole field to the other.
    """
    n = 120
    grid, _X, _Y = _grid(n, 5.0)
    grid.z = np.full((n, n), 700.0)
    grid.z[40:60, 70:90] = np.nan  # a slough in the middle is an edge too
    filled = fill_sinks(grid)
    raise_m = filled - grid.z
    # Exactly epsilon per cell of chessboard distance to the nearest edge
    # (the array border is an edge too, hence the padding).
    padded = np.pad(grid.mask, 1, constant_values=False)
    ring = ndimage.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1] - 1
    expected = FILL_EPSILON_M * np.where(grid.mask, ring, np.nan)
    assert np.allclose(raise_m, expected, atol=1e-10, equal_nan=True)
    assert np.nanmax(raise_m) < 1e-5

    # Without the slough (which, being an edge, rightly collects the ground
    # around it), water leaves straight across the nearest outer edge: the
    # top rows go north, the bottom rows south, nothing gathers in a corner
    # and no cell drains enough ground to count as a channel.
    grid.z = np.full((n, n), 700.0)
    result = analyse_hydrology(grid, np.zeros_like(grid.z))
    assert result.depressions == []
    assert result.to_dict()["ponded_cells"] == 0
    assert result.drainage_lines == [] and result.drainage_length_m == 0.0
    mid = n // 2 - 30
    assert np.all(result.fdir[1:20, mid] == D8_NAMES.index("N"))
    assert np.all(result.fdir[-20:-1, mid] == D8_NAMES.index("S"))
    assert np.nanmax(result.acc) <= n // 2 + 1
    rows_i, cols_i = np.nonzero(result.wet)
    assert abs(rows_i.mean() - (n - 1) / 2) < 1.0 and abs(cols_i.mean() - (n - 1) / 2) < 1.0


def test_bowl_in_flat_ground_spills_at_the_plain():
    """A bowl in level ground spills at the plain's own level: the filling
    increment across the flat must not lift the rim it reads."""
    n, cell, radius, depth = 400, 5.0, 50.0, 0.30
    grid, X, Y = _grid(n, cell)
    cx, cy = X0 + n * cell / 2, Y0 - n * cell / 2
    r = np.hypot(X - cx, Y - cy)
    k = depth / radius ** 2
    grid.z = 700.0 + np.where(r < radius, k * r ** 2 - depth, 0.0)
    filled = fill_sinks(grid)
    assert np.nanmax((filled - grid.z)[r >= radius]) < 1e-5
    found = depressions(grid, filled)
    assert len(found) == 1
    basin = found[0]
    assert basin["spill_m"] == pytest.approx(700.0, abs=1e-6)
    assert basin["max_depth_m"] == pytest.approx(depth, rel=0.01)
    # The basin is found above the 5 cm threshold but measured over the
    # whole lake, shallow rim included: only the cell quadrature remains.
    exact = math.pi * k * radius ** 4 / 2
    assert basin["volume_m3"] == pytest.approx(exact, rel=0.005)
    assert basin["area_ha"] == pytest.approx(math.pi * radius ** 2 / 10_000, rel=0.03)


@pytest.mark.parametrize("gx, gy, name", [
    (-0.01, 0.0, "E"), (0.01, 0.0, "W"), (0.0, 0.01, "S"), (0.0, -0.01, "N"),
    (-0.01, 0.01, "SE"), (0.01, -0.01, "NW"),
])
def test_flow_direction_codes_follow_the_compass(gx, gy, name):
    """Row 0 is north: a plane falling south flows to a higher row index."""
    grid = _plane(n=20, gx=gx, gy=gy)
    fdir = flow_direction_d8(fill_sinks(grid), grid)
    assert np.all(fdir[1:-1, 1:-1] == D8_NAMES.index(name))


def test_flow_accumulation_on_a_plane():
    grid = _plane()
    fdir = flow_direction_d8(fill_sinks(grid), grid)
    acc = flow_accumulation(fdir, grid)
    assert np.nanmin(acc) == 1.0
    # Along the fall line (a SW diagonal) the count grows by one per cell.
    diagonal = np.array([acc[i, grid.cols - 1 - i] for i in range(grid.rows)])
    assert np.all(np.diff(diagonal) == 1)
    outlets = (fdir == -1) & grid.mask
    assert np.nansum(acc[outlets]) == grid.valid_count


# ==========================================================================
# Bowls: filling, volume, spill level
# ==========================================================================

def test_bowl_depression_matches_the_analytic_volume():
    grid, truth = _crater()
    filled = fill_sinks(grid)
    found = depressions(grid, filled)
    assert len(found) == 1
    basin = found[0]
    assert basin["id"] == 1
    assert basin["volume_m3"] == pytest.approx(truth["volume_m3"], rel=0.10)
    assert basin["area_ha"] == pytest.approx(truth["area_ha"], rel=0.10)
    depth = truth["rim"] - truth["bottom"]
    assert abs(basin["spill_m"] - truth["rim"]) < 0.03 * depth
    assert basin["max_depth_m"] == pytest.approx(depth, rel=0.03)
    assert basin["bottom_m"] == pytest.approx(truth["bottom"], abs=0.01)
    assert basin["mean_depth_m"] == pytest.approx(depth / 2, rel=0.05)
    assert math.hypot(basin["x"] - truth["centre"][0], basin["y"] - truth["centre"][1]) <= grid.cell * 1.5
    lon, lat = grid.to_lonlat(basin["x"], basin["y"])
    assert basin["lon"] == pytest.approx(float(lon)) and basin["lat"] == pytest.approx(float(lat))
    json.dumps(found)


def test_fill_sinks_flattens_a_bowl_to_within_epsilon_per_cell():
    grid, truth = _crater()
    epsilon = 1e-4
    filled = fill_sinks(grid, epsilon=epsilon)
    assert np.all(filled >= grid.z)
    raised = filled > grid.z
    assert raised.sum() > 0.8 * truth["lake"].sum()
    # The lake rises by epsilon per cell away from its spill point, so it is
    # flat to within epsilon times the cells across it.
    across = 2 * 120.0 / grid.cell
    assert np.ptp(filled[raised]) <= epsilon * (across + 8)
    # Without the increment it is exactly flat.
    flat = fill_sinks(grid, epsilon=0.0)
    assert np.ptp(flat[flat > grid.z]) == 0.0
    # And with it, every interior cell has somewhere lower to go.
    fdir = flow_direction_d8(filled, grid)
    assert not np.any((fdir == -1) & ~_on_edge(grid))


def test_two_bowls_come_out_sorted_by_volume():
    grid, X, Y = _grid(200, 5.0)
    plane = 700.0 - 0.002 * (X - X0)

    def dip(cx, cy, radius, depth):
        r = np.hypot(X - cx, Y - cy)
        return np.where(r < radius, -depth * (1 - (r / radius) ** 2), 0.0)

    big = (X0 + 300.0, Y0 - 300.0)
    small = (X0 + 700.0, Y0 - 700.0)
    grid.z = plane + dip(*big, 150.0, 3.0) + dip(*small, 80.0, 1.5)
    found = depressions(grid, fill_sinks(grid))
    assert [b["id"] for b in found] == [1, 2]
    assert found[0]["volume_m3"] > found[1]["volume_m3"]
    assert found[0]["max_depth_m"] > found[1]["max_depth_m"]
    assert math.hypot(found[0]["x"] - big[0], found[0]["y"] - big[1]) <= 2 * grid.cell
    assert math.hypot(found[1]["x"] - small[0], found[1]["y"] - small[1]) <= 2 * grid.cell
    # A tiny or shallow threshold change does not invent a third basin.
    assert len(depressions(grid, fill_sinks(grid), min_depth_m=0.5)) == 2
    assert depressions(grid, fill_sinks(grid), min_area_ha=10.0) == []


def test_depression_depth_is_zero_on_a_plane_and_positive_in_a_bowl():
    grid, truth = _crater()
    depth = depression_depth(grid, fill_sinks(grid))
    assert np.nanmin(depth) == 0.0
    assert np.nanmax(depth) == pytest.approx(2.0, rel=0.03)
    plane = _plane()
    assert np.all(depression_depth(plane, fill_sinks(plane)) == 0.0)


# ==========================================================================
# Valleys: drainage lines and wetness
# ==========================================================================

def test_v_valley_gives_one_drainage_line_on_its_axis():
    grid, xa, axis_col = _v_valley()
    filled = fill_sinks(grid)
    fdir = flow_direction_d8(filled, grid)
    acc = flow_accumulation(fdir, grid)
    lines, total = drainage_lines(fdir, acc, grid, min_upstream_ha=1.0)
    assert len(lines) == 1
    line = lines[0]
    xs = np.asarray(line.coords)[:, 0]
    assert np.mean(np.abs(xs - xa)) < grid.cell
    assert len(line.coords) > 0.7 * grid.rows
    assert total == pytest.approx(line.length)
    # It runs downhill: south, to the outlet row.
    ys = np.asarray(line.coords)[:, 1]
    assert np.all(np.diff(ys) < 0)
    assert ys[-1] == pytest.approx(Y0 - (grid.rows - 0.5) * grid.cell)


def test_twi_is_higher_on_the_valley_axis_than_on_its_sides():
    grid, xa, axis_col = _v_valley()
    filled = fill_sinks(grid)
    acc = flow_accumulation(flow_direction_d8(filled, grid), grid)
    gy, gx = np.gradient(grid.z, grid.cell)
    slope_pct = 100.0 * np.hypot(gx, gy)
    index = twi(acc, slope_pct, grid)
    axis = np.nanmean(index[:, axis_col])
    sides = np.nanmean(index[:, [axis_col - 15, axis_col + 15]])
    assert axis > sides + 2.0
    # Formula check on one interior side cell: ln(acc * cell / tan beta).
    r, c = 50, axis_col + 15
    expected = math.log(acc[r, c] * grid.cell / max(slope_pct[r, c] / 100.0, 0.001))
    assert index[r, c] == pytest.approx(expected)


def test_wet_cells_threshold_and_ponding():
    grid, truth = _crater()
    filled = fill_sinks(grid)
    depth = depression_depth(grid, filled)
    acc = flow_accumulation(flow_direction_d8(filled, grid), grid)
    gy, gx = np.gradient(grid.z, grid.cell)
    index = twi(acc, 100.0 * np.hypot(gx, gy), grid)
    threshold = default_twi_threshold(index)
    assert threshold >= 8.0
    assert threshold == pytest.approx(max(np.nanpercentile(index, 90), 8.0))
    wet = wet_cells(index, depth)
    assert wet.dtype == bool
    assert np.all(wet[depth > 0.05])
    # An explicit threshold nobody reaches leaves only the ponded cells.
    only_ponded = wet_cells(index, depth, twi_threshold=1e9)
    assert np.array_equal(only_ponded, depth > 0.05)
    assert default_twi_threshold(np.full((3, 3), np.nan)) == 8.0


# ==========================================================================
# The synthetic DEM: truth and the NaN contract
# ==========================================================================

def test_depression_matches_the_synthetic_truth():
    from agrosuite.terrain.synthetic import synthetic_dem

    # A bowl alone on the tilted plane: the truth's bisection window holds
    # the whole basin, so its numbers are exact.
    grid, truth = synthetic_dem(
        cell_m=2.0, hills=(), valley=None, depression=(0.55, 0.20, 3.0, 60.0), tilt=(0.004, 225.0)
    )
    found = depressions(grid, fill_sinks(grid))
    assert len(found) == 1
    basin, expected = found[0], truth["depression"]
    assert basin["spill_m"] == pytest.approx(expected["spill_z_m"], abs=0.02)
    assert basin["max_depth_m"] == pytest.approx(expected["closed_depth_m"], abs=0.03)
    assert basin["volume_m3"] == pytest.approx(expected["volume_m3"], rel=0.05)
    assert basin["bottom_m"] == pytest.approx(expected["bottom_z_m"], abs=0.02)
    bx, by = expected["bottom_m"]
    assert math.hypot(basin["x"] - bx, basin["y"] - by) <= 1.5 * grid.cell


def test_nan_outside_the_field_stays_nan_in_every_layer():
    from agrosuite.terrain.synthetic import synthetic_dem

    grid, _truth = synthetic_dem(cell_m=5.0)
    outside = ~grid.mask
    assert outside.any()
    gy, gx = np.gradient(grid.fill_nearest(), grid.cell)
    slope_pct = 100.0 * np.hypot(gx, gy)
    slope_pct[outside] = np.nan

    result = analyse_hydrology(grid, slope_pct)
    assert isinstance(result, Hydrology)
    for name in ("filled", "depth", "acc", "twi"):
        layer = getattr(result, name)
        assert layer.shape == grid.shape
        assert np.array_equal(np.isnan(layer), outside), name
    assert result.fdir.dtype == np.int8
    assert np.all(result.fdir[outside] == -1)
    assert np.all(result.fdir[grid.mask] >= -1) and np.all(result.fdir[grid.mask] <= 7)
    assert not result.wet[outside].any()
    # The notch edge is an outlet like any other edge: everything drains.
    outlets = (result.fdir == -1) & grid.mask
    assert np.nansum(result.acc[outlets]) == grid.valid_count
    assert not np.any((result.fdir == -1) & grid.mask & ~_on_edge(grid))
    # The default field ponds in its bowl and in the valley above the hills.
    assert len(result.depressions) >= 1
    assert result.drainage_lines and result.drainage_length_m > 0
    for line in result.drainage_lines:
        xs, ys = np.asarray(line.coords).T
        assert np.all(np.isfinite(grid.sample(xs, ys)))
    summary = result.to_dict()
    json.dumps(summary)
    assert summary["drainage_lines"] == len(result.drainage_lines)
    assert summary["twi_threshold"] == result.twi_threshold


def test_drainage_network_shares_no_cell_between_lines():
    from agrosuite.terrain.synthetic import synthetic_dem

    grid, _truth = synthetic_dem(cell_m=5.0)
    filled = fill_sinks(grid)
    fdir = flow_direction_d8(filled, grid)
    acc = flow_accumulation(fdir, grid)
    lines, _total = drainage_lines(fdir, acc, grid, min_upstream_ha=0.5)
    assert len(lines) > 1
    # A cell is interior to at most one line; tributaries end where they join.
    seen: set[tuple[float, float]] = set()
    for line in lines:
        coords = list(line.coords)[:-1]
        for point in coords:
            assert point not in seen
            seen.add(point)
    # Longest line first, so the main stem is whole.
    assert lines[0].length == max(line.length for line in lines)


def test_shape_mismatch_is_refused():
    grid = _plane(n=10)
    with pytest.raises(ValueError, match="same grid"):
        flow_direction_d8(np.zeros((5, 5)), grid)
    with pytest.raises(ValueError, match="flow_direction_d8"):
        flow_accumulation(np.zeros((5, 5), dtype="int8"), grid)
    with pytest.raises(ValueError):
        fill_sinks(grid, epsilon=-1.0)


def test_rolling_surface_fills_and_routes_fast():
    n, cell = 500, 4.0
    grid, X, Y = _grid(n, cell)
    u, v = X - X0, Y - (Y0 - n * cell)
    grid.z = 700.0 + 3 * np.sin(u / 90) + 2 * np.cos(v / 70) + 1.5 * np.sin((u + v) / 120) - 0.002 * u
    start = time.perf_counter()
    filled = fill_sinks(grid)
    fdir = flow_direction_d8(filled, grid)
    acc = flow_accumulation(fdir, grid)
    elapsed = time.perf_counter() - start
    assert elapsed < 8.0, f"fill + route took {elapsed:.1f} s"
    assert np.all(filled >= grid.z)
    assert not np.any((fdir == -1) & ~_on_edge(grid))
    assert np.nansum(acc[fdir == -1]) == grid.valid_count
    assert len(depressions(grid, filled)) > 0


def test_gaussian_bowl_volume_counts_its_shallow_rim():
    """A Gaussian bowl holds 2 pi sigma^2 h; cut at the 5 cm ponding
    threshold it read 5 % under. The basin is found above the threshold
    and measured over the whole lake it sits in."""
    n, cell, sigma, depth = 200, 5.0, 40.0, 1.0
    grid, X, Y = _grid(n, cell)
    cx, cy = X0 + n * cell / 2, Y0 - n * cell / 2
    grid.z = 700.0 - depth * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    filled = fill_sinks(grid)
    found, labels = depression_labels(grid, filled)
    assert len(found) == 1
    basin = found[0]
    analytic = 2.0 * math.pi * sigma ** 2 * depth
    assert abs(basin["volume_m3"] - analytic) <= 0.02 * analytic
    assert basin["max_depth_m"] == pytest.approx(depth, abs=0.01)  # the deepest cell centre
    # The lake reaches every cell more than a tenth of a millimetre deep.
    deep = depression_depth(grid, filled) > 1e-4
    assert (labels == 1).sum() == deep.sum() and basin["area_ha"] == pytest.approx(deep.sum() * 25 / 10_000)
    assert depressions(grid, filled) == found
    result = analyse_hydrology(grid, np.zeros(grid.shape))
    assert result.depression_labels is not None and (result.depression_labels == labels).all()
    # Two pockets under one spill are one lake: one basin, once.
    z = grid.z.copy()
    z[100, 60:140] += 0.5  # a sill across the bowl, still under the rim
    grid.z = np.minimum(z, 700.0)
    found = depressions(grid, fill_sinks(grid))
    assert len(found) == 1
