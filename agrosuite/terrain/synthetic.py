"""Synthetic fields with known relief, for the tests and the terrain demo.

A relief analyser can only be checked against ground truth, and no real
yield file comes with one. These generators build a field whose surface is
an explicit function — a tilted plane, Gaussian hills, a valley trough and a
closed bowl — and return that function alongside the data, so a test can
ask "did the summit come out where it was put?" and "is the ponded volume
right?" with exact answers.

The point dataset mimics a harvest export: serpentine passes one swath
apart, a reading every couple of metres, GPS altitude with per-point noise
and a constant per-pass offset — the two defects a yield monitor's altitude
really has. The DEM variant is the same surface, noiseless, on a grid.

Every position is generated directly in the field's UTM zone, so the truth
surface takes the same metric coordinates ``Dataset.project()`` produces.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import ndimage

from ..core import crs as crs_mod
from ..core.dataset import Dataset, DatasetMeta
from .grid import ElevationGrid, _transformer

_LN2 = math.log(2.0)

#: 8-connectivity, as in :mod:`grid`.
_EIGHT = np.ones((3, 3), dtype=bool)

Surface = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ==========================================================================
# The surface
# ==========================================================================

def _bump(dist2: np.ndarray, radius: float) -> np.ndarray:
    """Radial profile at half height at ``radius``: exp(-ln2 (d/r)^2).

    Half-height radius is the number a person would measure on the map, which
    makes the truth easier to reason about than a Gaussian sigma.
    """
    return np.exp(-_LN2 * dist2 / (radius * radius))


def _segment_distance2(x, y, ax, ay, bx, by) -> np.ndarray:
    """Squared distance from points to the segment A-B."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = ((x - ax) * dx + (y - ay) * dy) / length2 if length2 > 0 else np.zeros_like(x)
    t = np.clip(t, 0.0, 1.0)
    px, py = ax + t * dx, ay + t * dy
    return (x - px) ** 2 + (y - py) ** 2


def _build_surface(
    origin: tuple[float, float],
    size_m: float,
    hills,
    valley,
    depression,
    tilt,
    base_m: float,
) -> tuple[Surface, Surface, Surface]:
    """Return ``(surface, plane, relief)`` callables over metric x, y.

    ``surface = plane + relief``; the plane is the general fall of the field
    and the relief holds the hills, valley and bowl.
    """
    ox, oy = origin
    gradient, direction = tilt
    # Downhill bearing: degrees clockwise from north. The plane loses
    # ``gradient`` metres per metre travelled that way, through the centre.
    ex, ny = math.sin(math.radians(direction)), math.cos(math.radians(direction))
    half = 0.5 * size_m

    def plane(x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        u, v = x - ox - half, y - oy - half
        return base_m - gradient * (u * ex + v * ny)

    def relief(x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        z = np.zeros(np.broadcast(x, y).shape, dtype="float64")
        for fx, fy, height, radius in hills or ():
            cx, cy = ox + fx * size_m, oy + fy * size_m
            z += height * _bump((x - cx) ** 2 + (y - cy) ** 2, radius)
        if valley is not None:
            fx0, fy0, fx1, fy1, depth, half_width = valley
            d2 = _segment_distance2(
                x, y, ox + fx0 * size_m, oy + fy0 * size_m, ox + fx1 * size_m, oy + fy1 * size_m
            )
            z -= depth * _bump(d2, half_width)
        if depression is not None:
            fx, fy, depth, radius = depression
            cx, cy = ox + fx * size_m, oy + fy * size_m
            z -= depth * _bump((x - cx) ** 2 + (y - cy) ** 2, radius)
        return z

    def surface(x, y):
        return plane(x, y) + relief(x, y)

    return surface, plane, relief


def _inside_factory(origin, size_m: float, notch) -> Surface:
    """Footprint test: the square minus a rectangle out of its NW corner."""
    ox, oy = origin

    def inside(x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        u, v = x - ox, y - oy
        ok = (u >= 0) & (u <= size_m) & (v >= 0) & (v <= size_m)
        if notch is not None:
            fx, fy = notch
            ok &= ~((u < fx * size_m) & (v > (1.0 - fy) * size_m))
        return ok

    return inside


# ==========================================================================
# Numerical truth of the discrete features
# ==========================================================================

def _local_extreme(surface, inside, centre, radius, sign: float, cell: float = 1.0):
    """Position and height of the surface's extreme near ``centre``.

    The tilt shifts a hill's true summit a few metres uphill of its Gaussian
    centre; a test that checks summit detection needs the real one.
    """
    cx, cy = centre
    half = 1.5 * radius
    n = int(math.ceil(2 * half / cell))
    xs = cx - half + (np.arange(n) + 0.5) * cell
    ys = cy - half + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    Z = surface(X, Y)
    Z = np.where(inside(X, Y), Z, np.nan)
    Z = np.where(np.isfinite(Z), sign * Z, -np.inf)
    r, c = np.unravel_index(int(np.argmax(Z)), Z.shape)
    return (float(X[r, c]), float(Y[r, c])), float(sign * Z[r, c])


def _closed_depression(surface, inside, centre, radius, cell: float = 1.0) -> dict[str, Any]:
    """Spill level, closed depth, area and volume of the bowl around ``centre``.

    The bowl is superimposed on a sloping plane, so the closed part is
    shallower than the bowl's nominal depth: water fills it only up to the
    saddle on the downhill side. The spill level is found by bisection — the
    highest level at which the flooded region around the bottom is still
    enclosed — on a 1 m grid, which is what a depression detector must
    reproduce.
    """
    cx, cy = centre
    half = max(4.0 * radius, 100.0)
    n = int(math.ceil(2 * half / cell))
    xs = cx - half + (np.arange(n) + 0.5) * cell
    ys = cy - half + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    Z = surface(X, Y)
    ok = inside(X, Y)
    Z = np.where(ok, Z, np.nan)

    near = (X - cx) ** 2 + (Y - cy) ** 2 <= (1.5 * radius) ** 2
    search = np.where(near & ok, Z, np.inf)
    r0, c0 = np.unravel_index(int(np.argmin(search)), Z.shape)
    z_bottom = float(Z[r0, c0])

    def enclosed(level: float) -> tuple[bool, np.ndarray]:
        wet = ok & (Z < level)
        labels, _ = ndimage.label(wet, structure=_EIGHT)
        comp = labels == labels[r0, c0]
        if not comp[r0, c0]:
            return True, comp
        touches_edge = comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
        touches_outside = bool((ndimage.binary_dilation(comp, structure=_EIGHT) & ~ok).any())
        return not (touches_edge or touches_outside), comp

    lo, hi = z_bottom, float(np.nanmax(Z))
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        closed, _ = enclosed(mid)
        if closed:
            lo = mid
        else:
            hi = mid
    spill = lo
    _, comp = enclosed(spill)
    depth = spill - z_bottom
    volume = float(np.sum(spill - Z[comp])) * cell * cell if depth > 0 else 0.0
    area = float(comp.sum()) * cell * cell if depth > 0 else 0.0
    return {
        "bottom_m": (float(X[r0, c0]), float(Y[r0, c0])),
        "bottom_z_m": z_bottom,
        "spill_z_m": float(spill),
        "closed_depth_m": float(depth),
        "area_m2": area,
        "volume_m3": volume,
    }


def _truth(
    crs: str,
    origin: tuple[float, float],
    size_m: float,
    centre_lonlat: tuple[float, float],
    hills,
    valley,
    depression,
    tilt,
    base_m: float,
    notch,
) -> dict[str, Any]:
    surface, plane, relief = _build_surface(origin, size_m, hills, valley, depression, tilt, base_m)
    inside = _inside_factory(origin, size_m, notch)
    to_ll = _transformer(crs, crs_mod.WGS84)
    from_ll = _transformer(crs_mod.WGS84, crs)

    def to_lonlat(x, y):
        lon, lat = to_ll.transform(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))
        return np.asarray(lon), np.asarray(lat)

    def from_lonlat(lon, lat):
        x, y = from_ll.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
        return np.asarray(x), np.asarray(y)

    def ll(point):
        lon, lat = to_lonlat(point[0], point[1])
        return (float(lon), float(lat))

    ox, oy = origin
    footprint_m2 = size_m * size_m
    notch_info = None
    if notch is not None:
        fx, fy = notch
        footprint_m2 -= fx * size_m * fy * size_m
        notch_info = {
            "corner": "north-west",
            "fraction": (float(fx), float(fy)),
            "bounds_m": (ox, oy + (1.0 - fy) * size_m, ox + fx * size_m, oy + size_m),
        }

    # The general fall as the findings would state it: highest to lowest
    # point of the plane over the footprint.
    corners = np.array([[ox, oy], [ox + size_m, oy], [ox, oy + size_m], [ox + size_m, oy + size_m]])
    plane_z = plane(corners[:, 0], corners[:, 1])
    gradient, direction = tilt

    hills_truth = []
    for fx, fy, height, radius in hills or ():
        centre = (ox + fx * size_m, oy + fy * size_m)
        summit, summit_z = _local_extreme(surface, inside, centre, radius, +1.0)
        hills_truth.append({
            "centre_m": centre,
            "centre_lonlat": ll(centre),
            "height_m": float(height),
            "radius_m": float(radius),
            "summit_m": summit,
            "summit_lonlat": ll(summit),
            "summit_z_m": summit_z,
        })

    valley_truth = None
    if valley is not None:
        fx0, fy0, fx1, fy1, depth, half_width = valley
        start = (ox + fx0 * size_m, oy + fy0 * size_m)
        end = (ox + fx1 * size_m, oy + fy1 * size_m)
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(int(length // 10.0), 1)
        t = np.linspace(0.0, 1.0, steps + 1)
        axis = np.column_stack([start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])])
        axis = axis[inside(axis[:, 0], axis[:, 1])]
        lon, lat = to_lonlat(axis[:, 0], axis[:, 1])
        valley_truth = {
            "start_m": start,
            "end_m": end,
            "start_lonlat": ll(start),
            "end_lonlat": ll(end),
            "depth_m": float(depth),
            "half_width_m": float(half_width),
            "length_m": float(length),
            "axis_m": axis,
            "axis_lonlat": np.column_stack([lon, lat]),
        }

    depression_truth = None
    if depression is not None:
        fx, fy, depth, radius = depression
        centre = (ox + fx * size_m, oy + fy * size_m)
        closed = _closed_depression(surface, inside, centre, radius)
        depression_truth = {
            "centre_m": centre,
            "centre_lonlat": ll(centre),
            "depth_param_m": float(depth),
            "radius_m": float(radius),
            "bottom_lonlat": ll(closed["bottom_m"]),
            **closed,
        }

    return {
        "crs": crs,
        "origin_m": (float(ox), float(oy)),
        "size_m": float(size_m),
        "centre_lonlat": (float(centre_lonlat[0]), float(centre_lonlat[1])),
        "centre_m": (ox + 0.5 * size_m, oy + 0.5 * size_m),
        "base_m": float(base_m),
        "area_ha": footprint_m2 / 10_000.0,
        "notch": notch_info,
        "tilt": {
            "gradient": float(gradient),
            "direction_deg": float(direction),
            "fall_m": float(plane_z.max() - plane_z.min()),
        },
        "hills": hills_truth,
        "valley": valley_truth,
        "depression": depression_truth,
        "surface": surface,
        "plane": plane,
        "relief": relief,
        "inside": inside,
        "to_lonlat": to_lonlat,
        "from_lonlat": from_lonlat,
    }


def _frame(centre_lonlat, size_m: float) -> tuple[str, tuple[float, float]]:
    """UTM zone of the field and the metric position of its SW corner."""
    lon, lat = centre_lonlat
    crs = crs_mod.utm_epsg(lon, lat)
    cx, cy = _transformer(crs_mod.WGS84, crs).transform(lon, lat)
    return crs, (float(cx) - 0.5 * size_m, float(cy) - 0.5 * size_m)


# ==========================================================================
# Generators
# ==========================================================================

def synthetic_terrain(
    size_m: float = 800,
    spacing_m: float = 2.0,
    swath_m: float = 9.0,
    seed: int = 1,
    noise_m: float = 0.3,
    pass_offset_m: float = 0.4,
    hills=((0.25, 0.30, 4.0, 90.0), (0.70, 0.65, 3.0, 70.0)),
    valley=(0.20, 0.0, 0.85, 1.0, 2.5, 60.0),
    depression=(0.55, 0.20, 0.9, 40.0),
    tilt=(0.008, 225.0),
    base_m: float = 700.0,
    centre_lonlat: tuple[float, float] = (-113.55, 51.75),
    notch: tuple[float, float] | None = (0.25, 0.25),
) -> tuple[Dataset, dict[str, Any]]:
    """A harvest-style point dataset over a field of known relief.

    Parameters
    ----------
    size_m:
        Side of the square field.
    spacing_m, swath_m:
        Distance between readings along a pass, and between passes.
    noise_m, pass_offset_m:
        Standard deviation of the per-point altitude noise and of the
        constant offset each pass carries.
    hills:
        ``(fx, fy, height_m, radius_m)`` per hill, positions as fractions of
        the field from its south-west corner; ``radius_m`` is where the hill
        is at half its height.
    valley:
        ``(fx0, fy0, fx1, fy1, depth_m, half_width_m)`` — a trough along the
        segment between the two fractional points.
    depression:
        ``(fx, fy, depth_m, radius_m)`` — a closed bowl; the part that
        actually ponds is shallower because of the tilt, see
        ``truth['depression']``.
    tilt:
        ``(gradient m/m, downhill bearing deg from north)`` of the plane.
    notch:
        Fraction of the width and height cut out of the north-west corner,
        so the field is not its own convex hull; ``None`` for a full square.

    Returns
    -------
    (Dataset, truth)
        ``truth`` carries the noiseless ``surface(x_m, y_m)`` in the
        dataset's metric CRS, the footprint test ``inside(x_m, y_m)``, and
        every feature's position in metres and lon/lat.
    """
    rng = np.random.default_rng(seed)
    crs, origin = _frame(centre_lonlat, size_m)
    truth = _truth(crs, origin, size_m, centre_lonlat, hills, valley, depression, tilt, base_m, notch)
    surface, inside = truth["surface"], truth["inside"]
    ox, oy = origin

    n_passes = max(int(round(size_m / swath_m)), 1)
    offsets = rng.normal(0.0, pass_offset_m, n_passes) if pass_offset_m > 0 else np.zeros(n_passes)
    along = np.arange(0.5 * spacing_m, size_m, spacing_m)

    xs, ys, ps = [], [], []
    for p in range(n_passes):
        px = ox + (p + 0.5) * swath_m
        track = along if p % 2 == 0 else along[::-1]
        y = oy + track
        x = np.full(track.size, px) + rng.normal(0.0, 0.10, track.size)
        keep = inside(x, y)
        xs.append(x[keep])
        ys.append(y[keep])
        ps.append(np.full(int(keep.sum()), p, dtype="int64"))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    pass_id = np.concatenate(ps)
    n = x.size

    z_true = surface(x, y)
    elev = z_true + rng.normal(0.0, noise_m, n) + offsets[pass_id]

    speed = np.clip(6.5 + rng.normal(0.0, 0.25, n), 3.0, 9.0)
    dt = spacing_m / (speed / 3.6)
    # A headland turn between passes keeps the timeline realistic.
    turn = np.where(np.diff(pass_id, prepend=pass_id[0]) != 0, 15.0, 0.0)
    seconds = np.cumsum(dt + turn)
    timestamp = pd.Timestamp("2025-08-20 09:00:00") + pd.to_timedelta(seconds, unit="s")

    # A plausible canola yield for a dry year, following the whole height of
    # the ground rather than only the bumps in it: water runs down the
    # field's general fall as well as off its hilltops, so the yield is set
    # against the height above the field's own mean — thinnest on the
    # hilltops, best on the low ground — and the closed bowl is drowned out
    # on top of that. Against the local relief alone the yield came out
    # uncorrelated with the elevation the same file carries (r = -0.1), and
    # a demo whose yield-against-relief chart is a flat line teaches the
    # user nothing about a tool built to find that relation.
    height = surface(x, y)
    height = height - height.mean()
    yield_kg = 2_600.0 - 110.0 * height + rng.normal(0.0, 220.0, n)
    if depression is not None:
        fx, fy, _depth, radius = depression
        d2 = (x - (ox + fx * size_m)) ** 2 + (y - (oy + fy * size_m)) ** 2
        yield_kg -= 900.0 * _bump(d2, 0.5 * radius)
    yield_kg = np.clip(yield_kg, 300.0, None)

    lon, lat = truth["to_lonlat"](x, y)
    df = pd.DataFrame({
        "lon": lon,
        "lat": lat,
        "timestamp": timestamp,
        "value": yield_kg,
        "swath_m": float(swath_m),
        "speed_kmh": speed,
        "pass_id": pass_id,
        "elev_m": elev,
    })
    meta = DatasetMeta(
        name="Terrain demo",
        source_path="<synthetic>",
        source_format="demo",
        brand="generic",
        brand_label="Synthetic data",
        operation="harvest",
        crop="canola",
        field_name="Terrain demo",
        value_label="Yield",
        value_unit="kg/ha",
        notes=["Synthetic field with known relief, for checking the terrain analysis."],
    )
    ds = Dataset(df, meta)
    ds.ensure_derived(default_swath_m=swath_m)

    truth["n_points"] = int(n)
    truth["n_passes"] = int(n_passes)
    truth["swath_m"] = float(swath_m)
    truth["spacing_m"] = float(spacing_m)
    truth["noise_m"] = float(noise_m)
    truth["pass_offsets_m"] = offsets
    truth["elev_true_m"] = z_true if len(ds.df) == n else surface(
        ds.df["x"].to_numpy(), ds.df["y"].to_numpy()
    )
    return ds, truth


def synthetic_dem(
    size_m: float = 800,
    cell_m: float = 5.0,
    hills=((0.25, 0.30, 4.0, 90.0), (0.70, 0.65, 3.0, 70.0)),
    valley=(0.20, 0.0, 0.85, 1.0, 2.5, 60.0),
    depression=(0.55, 0.20, 0.9, 40.0),
    tilt=(0.008, 225.0),
    base_m: float = 700.0,
    centre_lonlat: tuple[float, float] = (-113.55, 51.75),
    notch: tuple[float, float] | None = (0.25, 0.25),
) -> tuple[ElevationGrid, dict[str, Any]]:
    """The same noiseless surface as :func:`synthetic_terrain`, on a grid.

    Cells outside the footprint (the notch) are NaN, as a DEM clipped to a
    field boundary would be; pass ``notch=None`` for a full rectangle. The
    truth dict has the same keys as the point generator's.
    """
    crs, origin = _frame(centre_lonlat, size_m)
    truth = _truth(crs, origin, size_m, centre_lonlat, hills, valley, depression, tilt, base_m, notch)
    ox, oy = origin
    n = max(int(math.ceil(size_m / cell_m)), 1)
    x0, y0 = ox, oy + size_m
    grid = ElevationGrid(np.zeros((n, n)), x0, y0, cell_m, crs)
    X, Y = grid.xy()
    z = truth["surface"](X, Y)
    z[~truth["inside"](X, Y)] = np.nan
    grid.z = z
    return grid, truth
