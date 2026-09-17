"""Tests for the terrain derivatives: slope, aspect, curvature, hillshade,
TPI, roughness and the plane trend.

Every test runs on an analytic surface built straight into an
:class:`ElevationGrid` — a tilted plane or a Gaussian hill — because those
have closed-form derivatives, so the checks are exact numbers rather than
"looks plausible". The plane pins slope, aspect and the trend; the hill pins
the sign conventions (positive = convex, aspect = downhill bearing) and the
curvature formulas against the radial derivatives of the Gaussian; a notched
and clipped grid pins the no-NaN-bleed contract every later module relies on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.terrain import ElevationGrid
from agrosuite.terrain.derivatives import (
    SECTOR_LABELS,
    SECTOR_NAMES,
    aspect,
    aspect_sectors,
    curvature,
    hillshade,
    plane_trend,
    roughness,
    slope,
    tpi,
    tpi_standardized,
    tri,
)

CELL = 5.0
N = 100
X0, Y0 = 500_000.0, 5_700_000.0
CRS = "EPSG:32612"

#: Everything but the outer ring, where replicated borders bias a 3x3 kernel.
INNER = (slice(1, -1), slice(1, -1))


def _grid(z: np.ndarray) -> ElevationGrid:
    return ElevationGrid(z, X0, Y0, CELL, CRS)


def _xy() -> tuple[np.ndarray, np.ndarray]:
    return _grid(np.zeros((N, N))).xy()


def _plane(gradient: float = 0.05, direction_deg: float = 135.0) -> ElevationGrid:
    """A plane falling with ``gradient`` (m/m) toward the bearing ``direction_deg``."""
    xx, yy = _xy()
    ux, uy = math.sin(math.radians(direction_deg)), math.cos(math.radians(direction_deg))
    return _grid(700.0 - gradient * (ux * xx + uy * yy))


HILL_H, HILL_S = 10.0, 50.0  # metres high, Gaussian sigma in metres


def _hill() -> tuple[ElevationGrid, np.ndarray, tuple[int, int]]:
    """A Gaussian hill centred on cell (50, 50); returns the grid, the
    distance of each cell from the summit and the summit index."""
    xx, yy = _xy()
    summit = (50, 50)
    dist = np.hypot(xx - xx[summit], yy - yy[summit])
    return _grid(HILL_H * np.exp(-dist ** 2 / (2.0 * HILL_S ** 2))), dist, summit


def _hill_profile(dist: float) -> tuple[float, float]:
    """First and second radial derivatives of the Gaussian hill."""
    f = HILL_H * math.exp(-dist ** 2 / (2.0 * HILL_S ** 2))
    return -(dist / HILL_S ** 2) * f, (dist ** 2 / HILL_S ** 4 - 1.0 / HILL_S ** 2) * f


def _angle_diff(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, wrap-safe."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def _notched(grid: ElevationGrid) -> ElevationGrid:
    """The same surface with a north-west notch cut out and a circular field
    boundary — holes both inside and around the array."""
    z = grid.z.copy()
    z[:30, :25] = np.nan
    xx, yy = grid.xy()
    dist = np.hypot(xx - xx[50, 50], yy - yy[50, 50])
    z[dist > 45 * CELL] = np.nan
    return grid.with_values(z)


# ==========================================================================
# Plane: slope, aspect, trend
# ==========================================================================

def test_plane_slope_is_uniform():
    pct, deg = slope(_plane())
    assert np.allclose(pct[INNER], 5.0, atol=0.01)
    assert np.allclose(deg[INNER], math.degrees(math.atan(0.05)), atol=0.01)
    assert pct.shape == deg.shape == (N, N)


def test_plane_edge_cells_are_finite_and_use_replicated_borders():
    """The outer ring is finite (no NaN) but leans on copied values, so its
    slope is below the true 5 % — never above it."""
    pct, _ = slope(_plane())
    ring = np.ones((N, N), dtype=bool)
    ring[INNER] = False
    assert np.isfinite(pct[ring]).all()
    assert (pct[ring] > 0).all() and (pct[ring] <= 5.0 + 1e-9).all()


@pytest.mark.parametrize("direction", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 300.0])
def test_plane_aspect_is_the_downhill_bearing(direction):
    asp = aspect(_plane(direction_deg=direction))
    assert np.isfinite(asp[INNER]).all()
    assert (_angle_diff(asp[INNER], direction) < 0.5).all()
    assert (asp[INNER] >= 0).all() and (asp[INNER] < 360).all()


def test_aspect_flat_threshold():
    assert np.isnan(aspect(_plane(gradient=0.003))).all()
    assert np.isfinite(aspect(_plane(gradient=0.01))[INNER]).all()
    assert np.isnan(aspect(_plane(gradient=0.003), flat_threshold_pct=0.1)).sum() == 0


def test_aspect_sectors_codes():
    a = np.array([0.0, 22.4, 22.5, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 337.4, 337.5, 359.9, np.nan])
    codes = aspect_sectors(a)
    assert codes.tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 0, 0, -1]
    assert codes.dtype.kind == "i"
    assert len(SECTOR_LABELS) == len(SECTOR_NAMES) == 8
    assert SECTOR_LABELS[codes[3]] == "E" and SECTOR_NAMES[codes[8]] == "north-west"
    assert aspect_sectors(np.full((3, 3), np.nan)).tolist() == [[-1] * 3] * 3


def test_plane_trend_recovers_the_plane():
    trend = plane_trend(_plane())
    assert trend["gradient_pct"] == pytest.approx(5.0, abs=1e-6)
    assert trend["direction_deg"] == pytest.approx(135.0, abs=1e-6)
    assert trend["r2"] == pytest.approx(1.0, abs=1e-9)
    assert trend["a"] == pytest.approx(-0.05 / math.sqrt(2), abs=1e-9)
    assert trend["b"] == pytest.approx(0.05 / math.sqrt(2), abs=1e-9)
    assert trend["c"] == pytest.approx(700.0, abs=1e-3)
    # The extent along the diagonal of a 500 m square is its full diagonal,
    # outer edge to outer edge, so the drop is 5 % of that.
    assert trend["extent_m"] == pytest.approx(N * CELL * math.sqrt(2), rel=1e-9)
    assert trend["drop_m"] == pytest.approx(0.05 * N * CELL * math.sqrt(2), rel=1e-9)
    for value in trend.values():
        assert isinstance(value, float)


def test_plane_trend_with_holes_and_along_an_axis():
    trend = plane_trend(_notched(_plane(gradient=0.02, direction_deg=90.0)))
    assert trend["gradient_pct"] == pytest.approx(2.0, abs=1e-6)
    assert trend["direction_deg"] == pytest.approx(90.0, abs=1e-6)
    assert trend["r2"] == pytest.approx(1.0, abs=1e-9)
    # A circle of radius 45 cells spans 91 cells edge to edge along the x axis.
    assert trend["extent_m"] == pytest.approx(91 * CELL, rel=1e-9)


def test_plane_trend_flat_and_too_small():
    trend = plane_trend(_grid(np.full((N, N), 700.0)))
    # Least squares leaves a 1e-12 residue, not an exact zero.
    assert trend["gradient_pct"] == pytest.approx(0.0, abs=1e-9)
    assert trend["drop_m"] == pytest.approx(0.0, abs=1e-6)
    assert trend["c"] == pytest.approx(700.0, abs=1e-6)
    assert trend["r2"] == 1.0
    z = np.full((N, N), np.nan)
    z[10, 10] = z[10, 11] = 700.0
    with pytest.raises(ValueError, match="at least three cells"):
        plane_trend(_grid(z))


# ==========================================================================
# Gaussian hill: sign conventions and curvature formulas
# ==========================================================================

def test_hill_tpi_positive_at_summit_negative_at_foot():
    grid, dist, summit = _hill()
    t = tpi(grid, radius_m=50.0)
    assert t[summit] > 1.0
    # Beyond sqrt(2) sigma the Gaussian is locally below its surroundings.
    foot = (summit[0], summit[1] + int(2.5 * HILL_S / CELL))
    assert dist[foot] == pytest.approx(2.5 * HILL_S)
    assert t[foot] < -0.05
    zs = tpi_standardized(t)
    assert np.nanmean(zs) == pytest.approx(0.0, abs=1e-9)
    assert np.nanstd(zs) == pytest.approx(1.0, abs=1e-9)
    assert zs[summit] > zs[foot]


def test_tpi_small_radius_uses_the_3x3_window():
    grid, _, _ = _hill()
    t = tpi(grid, radius_m=1.0)  # under one 5 m cell
    z = grid.z
    r, c = 40, 60
    expected = z[r, c] - z[r - 1:r + 2, c - 1:c + 2].mean()
    assert t[r, c] == pytest.approx(expected, abs=1e-12)


def test_tpi_standardized_degenerate_inputs():
    flat = np.full((4, 4), 2.0)
    flat[0, 0] = np.nan
    out = tpi_standardized(flat)
    assert np.isnan(out[0, 0]) and (out[np.isfinite(out)] == 0.0).all()
    assert np.isnan(tpi_standardized(np.full((2, 2), np.nan))).all()


def test_hill_curvature_matches_the_radial_derivatives():
    """Positive = convex: near the summit profile and plan are both positive,
    at the foot profile turns negative, and each equals the closed form of
    the Gaussian (profile = -f'', plan = -f'/d, total = their sum) to a few
    tenths of a percent."""
    grid, dist, summit = _hill()
    profile, plan, total = curvature(grid)
    r0, c0 = summit
    # Points on the axes (no cross term) and on the diagonals (cross term F
    # carries the whole answer): all must agree with the radial formulas.
    for r, c in [(r0 - 4, c0), (r0, c0 + 4), (r0 - 4, c0 + 4), (r0 + 4, c0 - 4), (r0 - 20, c0), (r0 + 14, c0 + 14)]:
        d = dist[r, c]
        fp, fpp = _hill_profile(d)
        assert profile[r, c] == pytest.approx(-fpp, rel=0.01)
        assert plan[r, c] == pytest.approx(-fp / d, rel=0.01)
        assert total[r, c] == pytest.approx(-fpp - fp / d, rel=0.01)
    assert profile[r0 - 4, c0] > 0 and plan[r0 - 4, c0] > 0 and total[r0 - 4, c0] > 0
    foot = (r0 - 20, c0)  # 100 m out, twice sigma: concave along the slope
    assert profile[foot] < 0
    assert plan[foot] > 0  # still divergent: the contours are circles
    # At the summit there is no slope line, so profile and plan are 0 by
    # convention and the total is the Laplacian, 2h/s^2.
    assert profile[summit] == 0.0 and plan[summit] == 0.0
    assert total[summit] == pytest.approx(2.0 * HILL_H / HILL_S ** 2, rel=0.01)
    # total == profile + plan wherever there is a slope line; the far tail
    # of the Gaussian (and the summit) is flagged flat and split as 0 + 0.
    sloping = (dist > 0) & (dist < 4 * HILL_S)
    assert np.allclose(total[sloping], (profile + plan)[sloping], atol=1e-12)


def test_trough_plan_curvature_is_convergent():
    """A tilted trough gathers water: plan curvature on its axis is negative."""
    xx, yy = _xy()
    k, m = 0.002, 0.03
    z = k * (xx - xx[50, 50]) ** 2 - m * yy  # falls to the south along the axis
    _, plan, _ = curvature(_grid(z))
    axis = plan[INNER[0], 50]
    assert (axis < 0).all()
    assert axis[10] == pytest.approx(-2.0 * k, rel=1e-6)


def test_hill_aspect_points_downhill_radially():
    grid, _, (r0, c0) = _hill()
    asp = aspect(grid)
    for (r, c), bearing in [((r0 - 10, c0), 0.0), ((r0, c0 + 10), 90.0), ((r0 + 10, c0), 180.0), ((r0, c0 - 10), 270.0)]:
        assert _angle_diff(asp[r, c], bearing) < 0.5
    assert _angle_diff(asp[r0 - 7, c0 + 7], 45.0) < 0.5
    assert np.isnan(asp[r0, c0])  # the summit itself is flat


def test_hillshade_range_and_lit_side():
    grid, _, (r0, c0) = _hill()
    shade = hillshade(grid)  # sun from the north-west
    assert (shade >= 0).all() and (shade <= 1).all()
    assert shade[r0 - 10, c0 - 10] > shade[r0 + 10, c0 + 10] + 0.1
    # Turning the sun to the south-east swaps the lit flank.
    shade_se = hillshade(grid, azimuth_deg=135.0)
    assert shade_se[r0 + 10, c0 + 10] > shade_se[r0 - 10, c0 - 10] + 0.1
    # Flat ground is lit by the sine of the sun's altitude.
    flat = hillshade(_grid(np.full((N, N), 700.0)), altitude_deg=30.0)
    assert np.allclose(flat, 0.5, atol=1e-12)


def test_hillshade_z_factor_scales_the_gradient():
    """Doubling the vertical exaggeration is the same as doubling the slope."""
    exaggerated = hillshade(_plane(gradient=0.05), z_factor=2.0)
    steeper = hillshade(_plane(gradient=0.10))
    assert np.allclose(exaggerated[INNER], steeper[INNER], atol=1e-12)
    assert not np.allclose(exaggerated[INNER], hillshade(_plane(gradient=0.05))[INNER])


# ==========================================================================
# Roughness measures on a plane
# ==========================================================================

@pytest.mark.parametrize("direction", [90.0, 135.0, 20.0])
def test_roughness_and_tri_on_a_plane_equal_the_per_cell_drops(direction):
    grid = _plane(gradient=0.05, direction_deg=direction)
    gx = -0.05 * math.sin(math.radians(direction)) * CELL  # rise per cell eastward
    gy = -0.05 * math.cos(math.radians(direction)) * CELL  # rise per cell northward
    rough = roughness(grid)
    ruggedness = tri(grid)
    # Range over the 3x3: opposite corners along the fall line.
    assert np.allclose(rough[INNER], 2.0 * (abs(gx) + abs(gy)), atol=1e-9)
    # Mean absolute difference: two of each edge and diagonal neighbour.
    expected = (abs(gx) + abs(gy) + abs(gx + gy) + abs(gx - gy)) / 4.0
    assert np.allclose(ruggedness[INNER], expected, atol=1e-9)
    assert expected > 0


# ==========================================================================
# Masks: NaN exactly on masked cells, nowhere else
# ==========================================================================

def test_no_nan_bleed_on_a_notched_grid():
    grid = _notched(_plane())
    assert 0 < grid.valid_count < N * N
    before = grid.z.copy()
    expected_nan = ~grid.mask
    pct, deg = slope(grid)
    profile, plan, total = curvature(grid)
    outputs = {
        "slope_pct": pct, "slope_deg": deg, "aspect": aspect(grid),
        "profile": profile, "plan": plan, "total": total,
        "hillshade": hillshade(grid), "tpi": tpi(grid, 30.0),
        "tpi_std": tpi_standardized(tpi(grid, 30.0)),
        "roughness": roughness(grid), "tri": tri(grid),
    }
    for name, out in outputs.items():
        assert out.shape == grid.shape, name
        assert np.array_equal(np.isnan(out), expected_nan), f"{name}: NaN pattern differs from the mask"
    # Every cell whose whole 3x3 window is valid is exact: the holes do not
    # reach past the ring of cells touching them.
    from scipy import ndimage

    core = ndimage.binary_erosion(grid.mask, np.ones((3, 3), dtype=bool))
    ring = grid.mask & ~core
    assert core.sum() > 0.8 * grid.valid_count and ring.sum() > 100
    assert np.allclose(pct[core], 5.0, atol=0.01)
    assert (_angle_diff(outputs["aspect"][core], 135.0) < 0.5).all()
    assert np.allclose(outputs["hillshade"][core], outputs["hillshade"][50, 50], atol=1e-9)
    # The ring leans on copied values: finite and of the right order, no more.
    assert (pct[ring] > 0).all() and (pct[ring] < 1.5 * 5.0).all()
    assert (_angle_diff(outputs["aspect"][ring], 135.0) < 45.0).all()
    # Nothing was modified in place.
    assert np.array_equal(grid.z, before, equal_nan=True)


def test_hill_with_holes_keeps_its_summit_and_shape():
    grid, dist, summit = _hill()
    holed = _notched(grid)
    t_full = tpi(grid, 50.0)
    t_holed = tpi(holed, 50.0)
    # Far from any hole the two agree exactly; the mean is over valid cells.
    assert t_holed[summit] == pytest.approx(t_full[summit], abs=1e-9)
    assert np.isnan(t_holed[10, 10])
    asp = aspect(holed)
    assert np.isnan(asp[10, 10]) and np.isfinite(asp[summit[0] + 10, summit[1]])


def test_outputs_are_new_arrays():
    grid = _plane()
    pct, _ = slope(grid)
    assert not np.shares_memory(pct, grid.z)
    t = tpi(grid, 25.0)
    assert not np.shares_memory(t, grid.z)


def test_large_grid_is_fast_enough():
    """400 000 cells through every derivative in well under a second: the
    interface calls these on each request."""
    import time

    rng = np.random.default_rng(3)
    z = rng.normal(size=(632, 632)).cumsum(axis=0).cumsum(axis=1) / 100.0
    grid = _grid(z)
    t0 = time.perf_counter()
    slope(grid)
    aspect(grid)
    curvature(grid)
    hillshade(grid)
    tpi(grid, 100.0)
    roughness(grid)
    tri(grid)
    plane_trend(grid)
    assert time.perf_counter() - t0 < 5.0


def test_a_bearing_a_hair_west_of_north_is_zero_not_360():
    """-1e-15 degrees modulo 360 is 359.99999999999994, which prints as
    360.0: bearings are rounded before they are wrapped."""
    from agrosuite.terrain.derivatives import _bearing_deg

    bearing = _bearing_deg(np.array([-1e-15, 1e-15, -1e-9, 0.0]), np.array([1.0, 1.0, 1.0, -1.0]))
    assert bearing.tolist() == [0.0, 0.0, 0.0, 180.0]
    # A plane falling a hair west of north.
    trend = plane_trend(_plane(gradient=0.01, direction_deg=-1e-13))
    assert trend["direction_deg"] == 0.0
    asp = aspect(_plane(gradient=0.01, direction_deg=-1e-13))
    assert np.nanmax(asp) == 0.0 and np.nanmin(asp) == 0.0
    assert plane_trend(_plane(gradient=0.01, direction_deg=359.99996))["direction_deg"] == 0.0
    assert plane_trend(_plane(gradient=0.01, direction_deg=359.9994))["direction_deg"] == pytest.approx(359.9994)
