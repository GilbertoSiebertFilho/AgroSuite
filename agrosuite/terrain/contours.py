"""Contour lines and elevation profiles.

A contour map is the one picture of relief people already read without
training: the lines crowd where the ground is steep and spread where it is
flat, close into rings around a hilltop or a hollow, and bend up-valley where
water gathers. The slope, curvature and landform layers say the same things
in colour, but a farmer checking the analysis against the field trusts the
lines first, and a profile drawn between two clicks — "how far does it drop
from the gate to the slough?" — is the most direct answer to how the field
falls.

Two details decide whether the lines can be trusted. The interval has to
suit the field: 0.25 m on a prairie quarter with 3 m of relief gives a dozen
lines, on a hilly field it gives two hundred and the map goes black, so the
interval is chosen from the relief unless the user sets it. And the lines
have to stop at the field edge: contouring runs on a nearest-filled copy of
the grid so lines reach the last valid cell instead of stopping a cell short,
and every vertex whose nearest cell is outside the field is then dropped, so
nothing is drawn over the notch or the neighbour's land that the fill
invented.

Everything comes back as GeoJSON in WGS84, which the map draws directly and
:func:`contours_to_geodataframe` turns into the shapefile or GeoPackage the
export step writes.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from shapely.geometry import LineString

from ..core import crs as crs_mod
from ..core.units import Phrase
from .grid import ElevationGrid

# skimage.measure is imported inside :func:`contours`: it costs about half a
# second to load, and the server (and the MCP server) import this package at
# start-up while most sessions never draw a contour.

#: The intervals a map reader expects to see: each is a round number, and
#: every relief between 0.3 m and 800 m has one of them giving 3 to 40 lines.
INTERVALS: tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0)

#: Fewer than 3 lines show no shape; more than 40 hide the map under them.
MIN_LINES = 3
MAX_LINES = 40

#: An explicit interval that would draw more lines than this is refused:
#: each level is a full pass of marching squares over the grid, and no
#: reader can separate 250 lines on one field.
MAX_LEVELS = 250

#: A piece shorter than this many cells is a noise ring around a single
#: bump in the grid, or the stub left when a line is clipped at the edge.
_MIN_PIECE_CELLS = 2.0

#: Douglas-Peucker tolerance as a fraction of the cell: marching squares
#: puts a vertex on every cell edge the line crosses, four times what a
#: smooth line needs, and a quarter cell is under the grid's own precision.
_SIMPLIFY_CELLS = 0.25

#: 7 decimals of a degree is a centimetre — well under GPS precision, and
#: it keeps the GeoJSON at a third of the size of full floats.
_LONLAT_DECIMALS = 7

#: Slack on the line-count bounds so that 0.3 / 0.1, which floating point
#: evaluates just under 3, still counts as 3 lines.
_COUNT_TOLERANCE = 1e-9

#: A level closer than this fraction of the relief to the lowest or highest
#: cell is not drawn: it would trace the rounding noise on a plateau that
#: sits a float's width above the minimum, hundreds of loops around cells
#: that differ by 1e-13 m, rather than anything on the ground.
_LEVEL_MARGIN = 1e-6

#: Stations on the field a profile needs before it is a profile at all.
_MIN_PROFILE_STATIONS = 2


# ==========================================================================
# Interval
# ==========================================================================

def choose_interval(relief_m: float, target_count: int = 12) -> float:
    """The contour interval that draws about ``target_count`` lines.

    Picks from :data:`INTERVALS` the step whose expected line count
    (``relief / interval``) is closest to the target while staying between
    :data:`MIN_LINES` and :data:`MAX_LINES`; a tie goes to the finer step,
    since the extra lines still read. Below 0.3 m of relief no step gives
    three lines and the finest is returned; above 800 m none gives forty or
    fewer and the coarsest is.
    """
    relief = float(relief_m)
    if not math.isfinite(relief) or relief <= 0.0:
        return INTERVALS[0]
    counts = [relief / step for step in INTERVALS]
    allowed = [
        i for i, count in enumerate(counts)
        if MIN_LINES * (1.0 - _COUNT_TOLERANCE) <= count <= MAX_LINES * (1.0 + _COUNT_TOLERANCE)
    ]
    if not allowed:
        return INTERVALS[0] if counts[0] < MIN_LINES else INTERVALS[-1]
    best = min(allowed, key=lambda i: (abs(counts[i] - float(target_count)), INTERVALS[i]))
    return INTERVALS[best]


# ==========================================================================
# Contours
# ==========================================================================

def _clip_to_mask(line: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    """Split a marching-squares line into the runs whose vertices sit on
    valid cells.

    A vertex belongs to the cell whose centre is nearest, the same rule
    :meth:`ElevationGrid.sample` applies, so a line reaches half a cell past
    the last valid centre and no further. A closed ring that was cut open
    only because its arbitrary start vertex fell inside a masked stretch is
    rejoined across that start, so the map does not show a break where the
    ground has none.
    """
    rows, cols = valid.shape
    rn = np.clip(np.rint(line[:, 0]).astype("int64"), 0, rows - 1)
    cn = np.clip(np.rint(line[:, 1]).astype("int64"), 0, cols - 1)
    keep = valid[rn, cn]
    if keep.all():
        return [line]
    if not keep.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype("int8"), [0]))))
    runs = [line[start:end] for start, end in zip(edges[0::2], edges[1::2])]
    closed = bool(np.array_equal(line[0], line[-1]))
    if closed and keep[0] and keep[-1] and len(runs) >= 2:
        runs = [np.vstack([runs[-1][:-1], runs[0]])] + runs[1:-1]
    return [run for run in runs if len(run) >= 2]


def _to_metric(rowcol: np.ndarray, grid: ElevationGrid) -> np.ndarray:
    """Marching-squares (row, col) vertices to metric (x, y) columns."""
    x = grid.x0 + (rowcol[:, 1] + 0.5) * grid.cell
    y = grid.y0 - (rowcol[:, 0] + 0.5) * grid.cell
    return np.column_stack([x, y])


def _polyline_length(xy: np.ndarray) -> float:
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def _simplify(xy: np.ndarray, tolerance: float, min_length: float) -> np.ndarray | None:
    """Douglas-Peucker on one piece; None when nothing worth drawing is left."""
    simplified = LineString(xy).simplify(tolerance, preserve_topology=False)
    if simplified.is_empty or len(simplified.coords) < 2 or simplified.length < min_length:
        return None
    return np.asarray(simplified.coords, dtype="float64")


def _levels(zmin: float, zmax: float, interval: float) -> range:
    """The integer multiples ``k`` with ``zmin < k * interval < zmax``.

    A level at exactly the lowest or highest cell is a point, not a line,
    so the bounds are strict, and they are pulled in by
    :data:`_LEVEL_MARGIN` of the relief on each side so a level within
    floating-point noise of an extreme is not drawn either.
    """
    margin = _LEVEL_MARGIN * (zmax - zmin)
    return range(
        math.floor((zmin + margin) / interval) + 1,
        math.ceil((zmax - margin) / interval),
    )


def contours(
    grid: ElevationGrid,
    interval_m: float | None = None,
    values: np.ndarray | None = None,
    units: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Contour lines of the grid as GeoJSON LineString features in WGS84.

    Levels are the multiples of the interval inside the valid range, so two
    grids of the same field always draw the same lines whatever their own
    minimum happens to be. Each feature carries ``level_m``, ``index`` (the
    0-based rank of its level among the levels drawn) and ``major`` (true on
    every fifth line, counted from the levels that are multiples of five
    intervals — the ones a map labels).

    ``values`` contours another layer on the same geometry (a smoothed
    surface, a wetness index); NaN there is treated as outside, as is
    anything outside the grid's own mask. ``interval_m`` defaults to
    :func:`choose_interval` on the layer's relief.

    ``units`` is the reader's unit set, and it reaches the two refusals
    only. They name an interval and suggest another, and the reader typed
    theirs in the unit on screen: told to "choose 2 m or coarser" after
    typing 1.5 in feet, they would type 2 feet and be refused again.

    Returns ``(features, interval_m)``.
    """
    values = grid.z if values is None else np.asarray(values, dtype="float64")
    if values.shape != grid.shape:
        raise ValueError(
            f"Values shaped {values.shape} do not fit a grid of {grid.shape}; "
            "contours can only be drawn on a layer computed on this same grid."
        )
    valid = grid.mask & np.isfinite(values)

    if interval_m is None:
        interval = None
    else:
        interval = float(interval_m)
        if not (math.isfinite(interval) and interval > 0.0):
            say = Phrase(units)
            raise ValueError(
                "The contour interval must be a positive height — "
                f"{say.length(0.25, None)} suits a flat field, "
                f"{say.length(1.0, None)} or {say.length(2.0, None)} a rolling one."
            )

    if not valid.any():
        return [], (interval if interval is not None else INTERVALS[0])

    zmin = float(values[valid].min())
    zmax = float(values[valid].max())
    if interval is None:
        interval = choose_interval(zmax - zmin)
    levels = _levels(zmin, zmax, interval)
    if len(levels) > MAX_LEVELS:
        suggested = choose_interval(zmax - zmin, target_count=MAX_LINES)
        say = Phrase(units)
        raise ValueError(
            f"A {say.length(interval, None)} interval over "
            f"{say.length(zmax - zmin)} of relief would draw {len(levels)} contour "
            f"lines; choose {say.length(suggested, None)} or coarser."
        )
    if len(levels) == 0:
        return [], interval

    from skimage import measure

    filled = grid.fill_nearest(values)
    min_length = _MIN_PIECE_CELLS * grid.cell
    tolerance = _SIMPLIFY_CELLS * grid.cell

    pieces: list[tuple[int, np.ndarray]] = []
    for k in levels:
        for line in measure.find_contours(filled, k * interval):
            for run in _clip_to_mask(line, valid):
                xy = _to_metric(run, grid)
                if _polyline_length(xy) < min_length:
                    continue
                simplified = _simplify(xy, tolerance, min_length)
                if simplified is not None:
                    pieces.append((k, simplified))
    if not pieces:
        return [], interval

    # One transformer call for every vertex of every line: pyproj's
    # per-call cost dwarfs the arithmetic when there are hundreds of pieces.
    counts = np.array([len(xy) for _, xy in pieces])
    all_xy = np.concatenate([xy for _, xy in pieces])
    lon, lat = grid.to_lonlat(all_xy[:, 0], all_xy[:, 1])
    coords = np.column_stack([np.round(lon, _LONLAT_DECIMALS), np.round(lat, _LONLAT_DECIMALS)])
    splits = np.cumsum(counts)[:-1]

    drawn = sorted({k for k, _ in pieces})
    index_of = {k: i for i, k in enumerate(drawn)}
    features = []
    for (k, _), ring in zip(pieces, np.split(coords, splits)):
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": ring.tolist()},
            "properties": {
                "level_m": round(k * interval, 6),
                "index": index_of[k],
                "major": bool(k % 5 == 0),
            },
        })
    return features, interval


# ==========================================================================
# Profile
# ==========================================================================

def _profile_points(points_lonlat: Sequence[Sequence[float]]) -> np.ndarray:
    """The profile vertices as an ``(n, 2)`` array of valid WGS84 positions.

    The points arrive from a map click, from a hand-typed request or from
    Claude driving the MCP server, so each mistake gets its own message:
    the vertex count, a third coordinate (a GeoJSON position may carry an
    altitude; a metric ``(x, y, z)`` triple looks the same), a missing
    number, and a longitude or latitude outside its range. The range check
    matters because pyproj does not refuse latitude 95: it returns inf, and
    the profile would come back as a line of infinite length with every
    station sampling the one finite end.
    """
    points = [tuple(p) for p in points_lonlat]
    if len(points) < 2:
        raise ValueError(
            "A profile needs at least two points as (longitude, latitude) pairs: "
            "click where the line starts and where it ends."
        )
    if any(len(p) != 2 for p in points):
        raise ValueError(
            "Each profile point must be a (longitude, latitude) pair: a point with "
            "one value or with a third (an elevation, or metric x, y, z) cannot be "
            "placed on the map. Send the two WGS84 coordinates of each point."
        )
    try:
        pts = np.asarray(points, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A profile point holds something other than numbers; draw the line "
            "again on the map."
        ) from exc
    if not np.isfinite(pts).all():
        raise ValueError("A profile point has no coordinates; draw the line again on the map.")
    lon, lat = pts[:, 0], pts[:, 1]
    if (np.abs(lon) > 180.0).any() or (np.abs(lat) > 90.0).any():
        if (np.abs(lat) <= 180.0).all() and (np.abs(lon) <= 90.0).all():
            hint = " The latitude and longitude look swapped."
        elif (np.abs(pts) > 360.0).any():
            hint = " These look like metric x, y rather than degrees."
        else:
            hint = ""
        raise ValueError(
            "A profile point is outside the range of a longitude (-180 to 180) and a "
            f"latitude (-90 to 90).{hint} Send each point as (longitude, latitude) in "
            "WGS84 degrees."
        )
    return pts


def _nan_to_none(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def profile(
    grid: ElevationGrid,
    points_lonlat: Sequence[Sequence[float]],
    n: int = 200,
    values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Elevation along a polyline drawn on the map.

    The line is sampled at ``n`` stations evenly spaced along its length
    in the metric CRS (bilinear interpolation of the grid), so a chart of
    ``elev_m`` against ``distance_m`` is a true cross-section. ``slope_pct``
    is aligned with the stations: entry ``i`` is the gradient of the step
    from station ``i - 1`` to station ``i``, signed so that walking downhill
    is positive, and entry 0 is None because nothing precedes the start.
    Stations outside the field are None in both lists; a line with fewer
    than two stations on the field is refused, because a chart of nothing
    looks like a flat field rather than a missed click.

    ``points_lonlat`` is any number of ``(lon, lat)`` vertices from two up;
    ``values`` samples another layer on the same grid instead of elevation.
    """
    pts = _profile_points(points_lonlat)
    n = int(n)
    if n < 2:
        raise ValueError("A profile needs at least two stations along the line.")

    x, y = grid.from_lonlat(pts[:, 0], pts[:, 1])
    # pyproj answers an impossible or far-off position with inf rather than
    # an error; the range check above catches the common cases, this one
    # whatever else the projection cannot place (a point on the far side of
    # the globe from a UTM zone, for instance).
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(
            "A profile point cannot be placed on this field's map projection "
            f"({grid.crs}); draw the line over the field itself."
        )
    step_m = np.hypot(np.diff(x), np.diff(y))
    # A repeated vertex is a double click; np.interp needs strictly
    # increasing distances, so it is dropped rather than left as a zero step.
    keep = np.concatenate(([True], step_m > 0.0))
    x, y = x[keep], y[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(step_m[step_m > 0.0])))
    length = float(cumulative[-1])
    if length <= 0.0:
        raise ValueError(
            "The profile points are all at the same place; pick points some distance apart."
        )

    distance = np.linspace(0.0, length, n)
    xs = np.interp(distance, cumulative, x)
    ys = np.interp(distance, cumulative, y)
    z = grid.sample(xs, ys, values=values, order=1)
    if int(np.count_nonzero(np.isfinite(z))) < _MIN_PROFILE_STATIONS:
        raise ValueError(
            "The line does not cross the field: fewer than two of its stations fall "
            "on a cell with an elevation. Draw the line over the field itself, "
            "between two points inside its outline."
        )

    step = length / (n - 1)
    slope = np.full(n, np.nan)
    slope[1:] = 100.0 * (z[:-1] - z[1:]) / step

    return {
        "distance_m": distance.tolist(),
        "elev_m": _nan_to_none(z),
        "slope_pct": _nan_to_none(slope),
        "points": np.round(pts, _LONLAT_DECIMALS).tolist(),
        "length_m": length,
    }


# ==========================================================================
# Export
# ==========================================================================

def contours_to_geodataframe(features: Sequence[dict[str, Any]], crs: str = crs_mod.WGS84):
    """The contour features as a GeoDataFrame for the export writers.

    The features are WGS84, as GeoJSON always is; ``crs`` is the CRS the
    frame is returned in, so passing the grid's metric CRS reprojects. The
    column names are under the 10-character DBF limit, so the frame goes
    straight to a shapefile.
    """
    import geopandas as gpd
    import pandas as pd

    rows = [
        {
            "level_m": float(f["properties"]["level_m"]),
            "index": int(f["properties"]["index"]),
            "major": bool(f["properties"]["major"]),
        }
        for f in features
    ]
    geoms = [LineString(f["geometry"]["coordinates"]) for f in features]
    df = pd.DataFrame(rows, columns=["level_m", "index", "major"]).astype(
        {"level_m": "float64", "index": "int64", "major": "bool"}
    )
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(geoms, crs=crs_mod.WGS84))
    if crs and not gdf.crs.equals(crs):
        gdf = gdf.to_crs(crs)
    return gdf
