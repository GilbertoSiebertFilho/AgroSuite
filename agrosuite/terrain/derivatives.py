"""Slope, aspect, curvature and the other local derivatives of the relief.

Everything a farmer asks about relief — how steep, which way it faces,
where the ground bends from ridge to hollow — is a derivative of the
elevation surface, and every one of them is read off the 3x3 neighbourhood
of a cell or a small disc around it. This module holds those neighbourhood
operators; the landform classifier, the hydrology and the findings are
written on top of them.

Three decisions run through every function here.

**No NaN bleed.** A kernel that touches a masked cell would return NaN and
the field would lose a ring of cells at every edge and around every hole. So
each function fills the holes with the nearest valid elevation first
(:meth:`ElevationGrid.fill_nearest`), computes over the full array with
replicated borders, and re-masks the result to the grid's own footprint. The
cost is that the outermost ring of valid cells leans on copied values: on a
straight edge its slope component perpendicular to the edge is halved, on a
jagged edge the copied diagonal neighbour can push the slope either way (a
5 % plane reads 1.5-5.7 % on that ring) and its aspect by a few tens of
degrees. Every cell whose whole 3x3 window is valid is exact. That ring is
the headland, so findings should not be built on it alone.

**Sign convention: positive is convex.** Profile, plan and total curvature
all come out positive where the ground bulges upward and negative where it
hollows. A hilltop and the crest of a ridge are positive; a valley floor and
the bottom of a bowl are negative. In flow terms: positive profile curvature
means water accelerates (convex slope break), negative means it slows and
drops sediment; positive plan curvature means flow spreads out (a ridge or
nose), negative means it gathers (a draw or hollow). The Zevenbergen & Thorne
(1987) expressions are used with the sign of their profile term kept and the
sign of their plan term flipped, so that both agree with this one rule; the
total is ``profile + plan`` exactly. ArcGIS uses the opposite sign for
profile curvature — anyone comparing must negate it.

**Directions are compass bearings.** Aspect, the trend direction and the
hillshade azimuth are all degrees clockwise from north, 0 = north, 90 = east,
and "aspect" is the direction the slope faces, i.e. the downhill direction.
Row 0 of the grid is its north edge, so "up a row" is north.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import ElevationGrid, disc_kernel, nan_convolve

#: Short and long names of the eight compass sectors, in the order of the
#: codes :func:`aspect_sectors` returns (0 = N, clockwise).
SECTOR_LABELS: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SECTOR_NAMES: tuple[str, ...] = (
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
)

#: Squared gradient (m/m)^2 below which a cell has no downhill direction and
#: the curvature split into profile and plan is undefined; both are set to
#: zero there, as ArcGIS does, rather than dividing by nothing.
_FLAT_GRADIENT_SQ = 1e-12


# ==========================================================================
# Neighbourhood plumbing
# ==========================================================================

def _window(filled: np.ndarray) -> np.ndarray:
    """The 3x3 neighbourhood of every cell as a ``(rows, cols, 3, 3)`` view.

    ``w[r, c, i, j]`` is the value at row ``r + i - 1``, column ``c + j - 1``,
    so ``w[:, :, 0, 1]`` is the northern neighbour of every cell. Borders are
    replicated, which is what keeps the outermost ring finite. A strided
    view, not a copy: nine copies of a 400 000-cell grid would be wasteful
    for what is a handful of additions.
    """
    padded = np.pad(filled, 1, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, (3, 3))


def _horn_gradient(filled: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """``(dz/dx, dz/dy)`` in m/m by Horn's (1981) weighted differences.

    ``dz/dx`` is positive where the ground rises to the east, ``dz/dy`` where
    it rises to the north. Horn's kernel weights the four edge neighbours
    twice the corners, which is a 3x3 least-squares plane fit in disguise
    and is why it is the GIS default: it uses all eight neighbours, so the
    metre-level GPS noise that survives gridding averages down instead of
    landing whole on the slope.
    """
    w = _window(filled)
    nw, n, ne = w[:, :, 0, 0], w[:, :, 0, 1], w[:, :, 0, 2]
    we, ea = w[:, :, 1, 0], w[:, :, 1, 2]
    sw, s, se = w[:, :, 2, 0], w[:, :, 2, 1], w[:, :, 2, 2]
    dzdx = ((ne + 2.0 * ea + se) - (nw + 2.0 * we + sw)) / (8.0 * cell)
    dzdy = ((nw + 2.0 * n + ne) - (sw + 2.0 * s + se)) / (8.0 * cell)
    return dzdx, dzdy


def _masked(values: np.ndarray, grid: ElevationGrid) -> np.ndarray:
    """A fresh float64 copy of ``values`` with NaN wherever the grid is masked."""
    out = np.array(values, dtype="float64", copy=True)
    out[~grid.mask] = np.nan
    return out


#: Bearings are rounded to this many decimals before being wrapped: a
#: vector a hair west of north is -1e-15 degrees, which modulo 360 is
#: 359.99999999999994 and prints as 360.0; rounded first it is -0.0, and
#: wraps to 0. Four decimals is a third of a millimetre across a kilometre,
#: finer than any grid here can measure a direction.
BEARING_DECIMALS = 4


def _bearing_deg(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Compass bearing in ``[0, 360)`` of the vector ``(east, north)``."""
    bearing = np.round(np.degrees(np.arctan2(east, north)), BEARING_DECIMALS) % 360.0
    # Belt and braces: whatever the rounding, a bearing must never be 360.
    bearing[bearing >= 360.0] = 0.0
    return bearing


# ==========================================================================
# Slope and aspect
# ==========================================================================

def slope(grid: ElevationGrid) -> tuple[np.ndarray, np.ndarray]:
    """Steepness of every cell as ``(slope_pct, slope_deg)``.

    Percent is ``100 * tan``: rise over run, the figure agronomists and
    erosion tables use. Degrees are kept alongside because hillshade and the
    landform rules want the angle.
    """
    dzdx, dzdy = _horn_gradient(grid.fill_nearest(), grid.cell)
    gradient = np.hypot(dzdx, dzdy)
    return _masked(100.0 * gradient, grid), _masked(np.degrees(np.arctan(gradient)), grid)


def aspect(grid: ElevationGrid, flat_threshold_pct: float = 0.5) -> np.ndarray:
    """Direction each cell faces, in degrees clockwise from north.

    0 faces north, 90 east, 180 south, 270 west: the direction water would
    leave the cell. NaN where the slope is under ``flat_threshold_pct``,
    because on flat ground the direction is decided by GPS noise and a map
    of it would be confetti; the threshold is a percent slope so the same
    setting means the same thing on every grid.
    """
    dzdx, dzdy = _horn_gradient(grid.fill_nearest(), grid.cell)
    bearing = _bearing_deg(-dzdx, -dzdy)
    bearing[100.0 * np.hypot(dzdx, dzdy) < float(flat_threshold_pct)] = np.nan
    return _masked(bearing, grid)


def aspect_sectors(aspect_deg: np.ndarray) -> np.ndarray:
    """Aspect binned into the eight compass sectors.

    Codes 0..7 for N, NE, E, SE, S, SW, W, NW (see :data:`SECTOR_LABELS`),
    each 45 degrees wide and centred on its bearing so that north is
    337.5-22.5. -1 where the aspect is NaN (flat or outside the field).
    """
    a = np.asarray(aspect_deg, dtype="float64")
    codes = np.full(a.shape, -1, dtype="int32")
    finite = np.isfinite(a)
    codes[finite] = (np.floor(((a[finite] + 22.5) % 360.0) / 45.0).astype("int32")) % 8
    return codes


# ==========================================================================
# Curvature
# ==========================================================================

def curvature(grid: ElevationGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(profile, plan, total)`` curvature in 1/m, positive where convex.

    Zevenbergen & Thorne (1987) fit a quadratic to the 3x3 window; its second
    derivatives along the slope line (profile) and along the contour (plan)
    are the two quantities that separate a crest from a hollow. The
    convention here, unlike theirs and unlike ArcGIS, is one rule for all
    three: positive = convex (hilltop, ridge crest, nose), negative = concave
    (valley floor, bowl, draw). So profile > 0 is where runoff accelerates,
    plan > 0 is where it spreads out, and ``total == profile + plan``. Values
    are per metre — multiply by 100 to compare with the ArcGIS 1/100 m
    figures (and negate its profile).

    Cells with no gradient have no slope line, so profile and plan are set
    to 0 there; the total, which needs no direction, is still meaningful.
    """
    cell = grid.cell
    w = _window(grid.fill_nearest())
    z1, z2, z3 = w[:, :, 0, 0], w[:, :, 0, 1], w[:, :, 0, 2]
    z4, z5, z6 = w[:, :, 1, 0], w[:, :, 1, 1], w[:, :, 1, 2]
    z7, z8, z9 = w[:, :, 2, 0], w[:, :, 2, 1], w[:, :, 2, 2]
    cell2 = cell * cell
    d = ((z4 + z6) / 2.0 - z5) / cell2
    e = ((z2 + z8) / 2.0 - z5) / cell2
    f = (-z1 + z3 + z7 - z9) / (4.0 * cell2)
    g = (-z4 + z6) / (2.0 * cell)
    h = (z2 - z8) / (2.0 * cell)

    gradient_sq = g * g + h * h
    flat = gradient_sq < _FLAT_GRADIENT_SQ
    safe = np.where(flat, 1.0, gradient_sq)
    profile = -2.0 * (d * g * g + e * h * h + f * g * h) / safe
    plan = -2.0 * (d * h * h + e * g * g - f * g * h) / safe
    profile[flat] = 0.0
    plan[flat] = 0.0
    total = -2.0 * (d + e)
    return _masked(profile, grid), _masked(plan, grid), _masked(total, grid)


# ==========================================================================
# Hillshade
# ==========================================================================

def hillshade(
    grid: ElevationGrid,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Shaded relief in ``[0, 1]`` for a sun at ``azimuth_deg`` / ``altitude_deg``.

    The standard hillshade — the cosine of the angle between the surface
    normal and the direction of the sun — written as a dot product rather
    than the usual slope/aspect trigonometry; the two are the same number.
    A north-west sun at 45 degrees is the cartographic default because the
    eye reads relief lit from the top left as hills rather than hollows.
    ``z_factor`` exaggerates the vertical, which is needed to see anything on
    a prairie field whose relief is a few metres across a kilometre.
    """
    dzdx, dzdy = _horn_gradient(grid.fill_nearest(), grid.cell)
    az = math.radians(float(azimuth_deg))
    alt = math.radians(float(altitude_deg))
    sun_e, sun_n, sun_up = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    normal_e = -float(z_factor) * dzdx
    normal_n = -float(z_factor) * dzdy
    norm = np.sqrt(normal_e * normal_e + normal_n * normal_n + 1.0)
    shade = (normal_e * sun_e + normal_n * sun_n + sun_up) / norm
    return _masked(np.clip(shade, 0.0, 1.0), grid)


# ==========================================================================
# Position and roughness
# ==========================================================================

def tpi(grid: ElevationGrid, radius_m: float) -> np.ndarray:
    """Topographic position index: elevation minus the mean within ``radius_m``.

    Positive where the cell stands above its surroundings, negative where it
    sits below them, in metres. The radius is the size of the feature being
    asked about — a 30 m radius tells knolls from hollows, a 300 m one tells
    the hill from the valley — so the caller picks it and the landform
    classifier compares two. The mean is taken over the valid cells of the
    disc only (normalized convolution), so the edge of the field is compared
    with the field, not with copies of its own rim. A radius under one cell
    falls back to the 3x3 neighbourhood; the centre cell is included in the
    mean either way, which scales nothing but the magnitude.
    """
    radius_cells = float(radius_m) / grid.cell
    if radius_cells < 1.0:
        kernel = np.full((3, 3), 1.0 / 9.0)
    else:
        kernel = disc_kernel(radius_cells)
    neighbourhood_mean = nan_convolve(grid.z, kernel)
    return _masked(grid.z - neighbourhood_mean, grid)


def tpi_standardized(tpi_values: np.ndarray) -> np.ndarray:
    """TPI as z-scores over the valid cells, ``(tpi - mean) / sd``.

    Landform thresholds in the literature (Weiss 2001) are stated in
    standard deviations because the metre value of a "ridge" depends on how
    rugged the field is. A surface with no variation has no scale to
    standardize by and comes back as zeros rather than NaN.
    """
    t = np.asarray(tpi_values, dtype="float64")
    finite = np.isfinite(t)
    if not finite.any():
        return t.copy()
    sd = float(t[finite].std())
    if not sd > 0.0:
        return np.where(finite, 0.0, np.nan)
    return (t - float(t[finite].mean())) / sd


def roughness(grid: ElevationGrid) -> np.ndarray:
    """Elevation range (max - min) within the 3x3 window, in metres.

    Not detrended: a smooth 5 % slope has a roughness of the drop across
    three cells. That is intended — the number answers "how much does the
    ground change under the implement here", which on a steep field it does.
    """
    w = _window(grid.fill_nearest())
    return _masked(w.max(axis=(2, 3)) - w.min(axis=(2, 3)), grid)


def tri(grid: ElevationGrid) -> np.ndarray:
    """Terrain ruggedness index: mean absolute difference to the 8 neighbours.

    Wilson et al.'s (2007) form, in metres, the one GDAL reports; smoother
    than :func:`roughness` because it averages the eight differences instead
    of taking the two extremes, so a single noisy neighbour counts one
    eighth rather than everything.
    """
    filled = grid.fill_nearest()
    w = _window(filled)
    mean_abs_diff = np.abs(w - filled[:, :, None, None]).sum(axis=(2, 3)) / 8.0
    return _masked(mean_abs_diff, grid)


# ==========================================================================
# General fall
# ==========================================================================

def plane_trend(grid: ElevationGrid) -> dict[str, Any]:
    """The plane ``z = a*x + b*y + c`` fitted to the valid cells: the field's general fall.

    Returns ``gradient_pct`` (steepness of the plane), ``direction_deg``
    (downhill bearing, clockwise from north), ``drop_m`` (that gradient times
    the field's extent along that bearing — the "falls 9 m from corner to
    corner" figure), ``r2`` (how much of the relief the plane explains: near
    1 on a uniformly tilted field, low where hills and hollows dominate),
    the coefficients ``a``, ``b``, ``c`` in the grid's metric CRS, and
    ``extent_m``. ``direction_deg`` is 0 when the gradient is 0 — check the
    gradient before quoting a direction. A flat field is explained perfectly
    by its own mean, so its ``r2`` is 1.

    The fit is centred on the mean cell coordinate before solving, because
    UTM eastings around 500 000 m would otherwise make the normal equations
    ill-conditioned; ``c`` is then moved back to the original origin.
    """
    mask = grid.mask
    n = int(np.count_nonzero(mask))
    if n < 3:
        raise ValueError(
            f"A plane needs at least three cells with an elevation; this grid has {n}. "
            "Grid a file with more elevation readings, or a larger DEM extent."
        )
    xx, yy = grid.xy()
    x = xx[mask]
    y = yy[mask]
    z = grid.z[mask]
    xm, ym = float(x.mean()), float(y.mean())
    design = np.column_stack([x - xm, y - ym, np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    a, b, c_centred = (float(v) for v in coef)
    c = c_centred - a * xm - b * ym

    residual = z - design @ coef
    ss_res = float(residual @ residual)
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    gradient = math.hypot(a, b)
    direction = 0.0
    if gradient > 0.0:
        direction = round(math.degrees(math.atan2(-a, -b)), BEARING_DECIMALS) % 360.0
    if direction >= 360.0:
        direction = 0.0
    # Extent of the footprint along the downhill bearing, measured between
    # the outer edges of the extreme cells: a square cell projects onto the
    # bearing with half-width (|ux| + |uy|) * cell / 2 on either side.
    ux, uy = math.sin(math.radians(direction)), math.cos(math.radians(direction))
    projection = x * ux + y * uy
    extent = float(projection.max() - projection.min()) + (abs(ux) + abs(uy)) * grid.cell
    return {
        "gradient_pct": 100.0 * gradient,
        "direction_deg": direction,
        "drop_m": gradient * extent,
        "r2": max(min(r2, 1.0), 0.0),
        "a": a,
        "b": b,
        "c": c,
        "extent_m": extent,
    }
