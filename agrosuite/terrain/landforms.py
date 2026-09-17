"""Landform classes, discrete hills and lows, and the share tables.

A slope map answers "how steep"; this module answers "what is where". Two
tools do that work. The first is the two-scale topographic position index
of Weiss (2001): a cell that sits above its surroundings at both a 30 m and
a 150 m scale is a hilltop, one below at both scales is a valley floor, and
the mixed cases are the slopes between. The second is plain connected
components: the patches where the large-scale index stands well above (or
below) the field's own spread are the hills (or lows) a person would point
at on the map, and each gets a summit, an area and a height above its
foot — the ground where the flank stops falling, or the saddle where the
hill meets the next listed hill or low, whichever is higher.

Every area here is counted on the grid (valid cells times the cell area),
never on a rendered image, so the class shares add up to the grid's own
area exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from ..core.units import Phrase
from .derivatives import SECTOR_LABELS, SECTOR_NAMES, aspect_sectors
from .grid import ElevationGrid, connected_components
from .render import LANDFORM_COLORS

#: 8-connectivity, the same structuring element the component labelling uses,
#: so a component's ring is the ring of the same neighbourhood rule.
_EIGHT = np.ones((3, 3), dtype=bool)

#: The ring statistic :func:`foot_level` walks on: the upper quartile of the
#: ring for a hill (lower for a low). See the function for why not the median.
FOOT_PERCENTILE = 75.0

#: How closely :func:`saddle_level` brackets a saddle, in metres: a
#: millimetre is far under any height the analysis quotes (one decimal).
SADDLE_TOLERANCE_M = 1e-3

#: The smallest patch reported as a hill or a low, in hectares.
MIN_FEATURE_AREA_HA = 0.05

#: The six landform classes in code order. The Weiss ten-class scheme is
#: collapsed here because a farmer's decisions (headland, drainage, erosion)
#: split the field six ways at most; the finer classes only add noise.
CLASSES: list[dict[str, Any]] = [
    {"key": "hilltop", "code": 1, "label": "Hilltop / ridge", "color": LANDFORM_COLORS[1]},
    {"key": "upper_slope", "code": 2, "label": "Upper slope", "color": LANDFORM_COLORS[2]},
    {"key": "mid_slope", "code": 3, "label": "Mid slope", "color": LANDFORM_COLORS[3]},
    {"key": "flat", "code": 4, "label": "Flat", "color": LANDFORM_COLORS[4]},
    {"key": "lower_slope", "code": 5, "label": "Lower slope", "color": LANDFORM_COLORS[5]},
    {"key": "valley", "code": 6, "label": "Valley / hollow", "color": LANDFORM_COLORS[6]},
]
CLASS_BY_CODE: dict[int, dict[str, Any]] = {c["code"]: c for c in CLASSES}

#: Slope classes, in percent. The breaks follow the usual agronomic bands:
#: under 2 % drains slowly, 2-5 % is where sheet erosion starts on bare
#: soil, above 10 % machinery and erosion both become a concern.
SLOPE_CLASSES: list[dict[str, Any]] = [
    {"key": "flat", "label": "Flat (< 2 %)", "from_pct": 0.0, "to_pct": 2.0},
    {"key": "gentle", "label": "Gentle (2-5 %)", "from_pct": 2.0, "to_pct": 5.0},
    {"key": "moderate", "label": "Moderate (5-10 %)", "from_pct": 5.0, "to_pct": 10.0},
    {"key": "strong", "label": "Strong (10-15 %)", "from_pct": 10.0, "to_pct": 15.0},
    {"key": "steep", "label": "Steep (> 15 %)", "from_pct": 15.0, "to_pct": None},
]

#: Field character: the one word a farmer would use for the whole field.
CHARACTERS: dict[str, str] = {
    "flat": "flat",
    "gently_undulating": "gently undulating",
    "rolling": "rolling",
    "hilly": "hilly",
}


# ==========================================================================
# Classification
# ==========================================================================

def classify(
    tpi_small_std: np.ndarray,
    tpi_large_std: np.ndarray,
    slope_pct: np.ndarray,
    flat_slope_pct: float = 2.0,
) -> np.ndarray:
    """Six landform classes from the two standardized TPIs and the slope.

    The inputs are the standardized indices (units of their own standard
    deviation) at the small and the large neighbourhood; "high" means above
    +1, "low" below -1, "level" in between. The rule table, after Weiss
    (2001) collapsed to six classes:

    ==========  ==========  ==================  ====================
    large       small       slope               class (code)
    ==========  ==========  ==================  ====================
    high        high        any                 hilltop (1)
    high        level       any                 upper_slope (2)
    high        low         any                 upper_slope (2) — a hollow high on a hill
    level       high        any                 upper_slope (2) — a local ridge on a slope
    level       level       < flat_slope_pct    flat (4)
    level       level       >= flat_slope_pct   mid_slope (3)
    level       low         any                 lower_slope (5) — a local draw
    low         high        any                 lower_slope (5) — a knoll in a valley
    low         level       any                 lower_slope (5)
    low         low         any                 valley (6)
    ==========  ==========  ==================  ====================

    Returns an int8 array of codes 1..6, and 0 where any input is NaN
    (outside the field).
    """
    small = np.asarray(tpi_small_std, dtype="float64")
    large = np.asarray(tpi_large_std, dtype="float64")
    slope = np.asarray(slope_pct, dtype="float64")
    if small.shape != large.shape or small.shape != slope.shape:
        raise ValueError(
            "The two TPI layers and the slope must be shaped alike; compute them "
            "on this same grid."
        )
    valid = np.isfinite(small) & np.isfinite(large) & np.isfinite(slope)
    codes = np.zeros(small.shape, dtype="int8")
    with np.errstate(invalid="ignore"):
        l_hi, l_lo = large > 1.0, large < -1.0
        s_hi, s_lo = small > 1.0, small < -1.0
        l_mid = ~l_hi & ~l_lo
        s_mid = ~s_hi & ~s_lo
        codes[l_hi & s_hi] = 1
        codes[l_hi & ~s_hi] = 2
        codes[l_mid & s_hi] = 2
        codes[l_mid & s_mid & (slope < flat_slope_pct)] = 4
        codes[l_mid & s_mid & (slope >= flat_slope_pct)] = 3
        codes[l_mid & s_lo] = 5
        codes[l_lo & ~s_lo] = 5
        codes[l_lo & s_lo] = 6
    codes[~valid] = 0
    return codes


def class_areas(codes: np.ndarray, cell: float) -> list[dict[str, Any]]:
    """Area and share of each landform class, in code order (all six listed)."""
    counts = np.bincount(codes[codes > 0].astype("int64").ravel(), minlength=7)
    total = int(counts[1:].sum())
    out = []
    for cls in CLASSES:
        n = int(counts[cls["code"]])
        out.append({
            "key": cls["key"],
            "code": cls["code"],
            "label": cls["label"],
            "area_ha": n * cell ** 2 / 10_000.0,
            "pct": 100.0 * n / total if total else 0.0,
            "color": cls["color"],
        })
    return out


# ==========================================================================
# Shares of slope and aspect
# ==========================================================================

def slope_classes(slope_pct: np.ndarray, cell: float) -> list[dict[str, Any]]:
    """Area and share of the field in each slope band (see SLOPE_CLASSES)."""
    slope = np.asarray(slope_pct, dtype="float64")
    finite = slope[np.isfinite(slope)]
    total = int(finite.size)
    out = []
    for cls in SLOPE_CLASSES:
        lo, hi = cls["from_pct"], cls["to_pct"]
        n = int(np.count_nonzero((finite >= lo) & ((finite < hi) if hi is not None else True)))
        out.append({
            "key": cls["key"],
            "label": cls["label"],
            "from_pct": lo,
            "to_pct": hi,
            "area_ha": n * cell ** 2 / 10_000.0,
            "pct": 100.0 * n / total if total else 0.0,
        })
    return out


def aspect_distribution(
    aspect_deg: np.ndarray,
    slope_pct: np.ndarray,
    cell: float,
    flat_pct: float = 0.5,
) -> list[dict[str, Any]]:
    """Area facing each compass sector, plus the ground too flat to face anywhere.

    A cell counts as ``flat`` when its slope is under ``flat_pct`` or its
    aspect is undefined; the eight sectors and ``flat`` together cover every
    valid cell exactly once.
    """
    aspect = np.asarray(aspect_deg, dtype="float64")
    slope = np.asarray(slope_pct, dtype="float64")
    valid = np.isfinite(slope)
    total = int(np.count_nonzero(valid))
    with np.errstate(invalid="ignore"):
        flat = valid & (~np.isfinite(aspect) | (slope < flat_pct))
    sectors = aspect_sectors(np.where(flat, np.nan, aspect))
    sectors[~valid] = -1
    counts = np.bincount(sectors[sectors >= 0].astype("int64").ravel(), minlength=8)
    out = []
    for code in range(8):
        n = int(counts[code])
        out.append({
            "key": SECTOR_LABELS[code],
            "label": f"Facing {SECTOR_NAMES[code]}",
            "area_ha": n * cell ** 2 / 10_000.0,
            "pct": 100.0 * n / total if total else 0.0,
        })
    n_flat = int(np.count_nonzero(flat))
    out.append({
        "key": "flat",
        "label": "Flat (no aspect)",
        "area_ha": n_flat * cell ** 2 / 10_000.0,
        "pct": 100.0 * n_flat / total if total else 0.0,
    })
    return out


# ==========================================================================
# Field character and position words
# ==========================================================================

def character(
    relief_m: float,
    mean_slope_pct: float,
    p95_slope_pct: float,
    units: dict | None = None,
) -> dict[str, str]:
    """The one-word character of the field, with the reason in a sentence.

    Mean slope is the deciding number because it is what the whole field
    feels like under a machine; the relief only separates "flat" from
    "gently undulating" (a field can average under 1 % and still fall 5 m
    end to end).

    ``why`` is prose with a height written into it, so it takes the
    reader's unit set like every other sentence in the app; ``None`` is
    the metric store, which is what the stored summary holds until
    :func:`agrosuite.terrain.analysis.restate` writes it again for a
    reader.
    """
    say = Phrase(units)
    relief = float(relief_m) if np.isfinite(relief_m) else 0.0
    mean = float(mean_slope_pct) if np.isfinite(mean_slope_pct) else 0.0
    p95 = float(p95_slope_pct) if np.isfinite(p95_slope_pct) else 0.0
    if mean < 1.0 and relief < 2.0:
        key = "flat"
        why = (
            f"The slope averages {mean:.1f} % and the whole field lies within "
            f"{say.length(relief)} of height: flat ground."
        )
    elif mean < 3.0:
        key = "gently_undulating"
        why = (
            f"The slope averages {mean:.1f} % (95 % of the field is under {p95:.1f} %) "
            f"over {say.length(relief)} of relief: gentle undulations rather than "
            "distinct hills."
        )
    elif mean < 8.0:
        key = "rolling"
        why = (
            f"The slope averages {mean:.1f} % (95 % of the field is under {p95:.1f} %) "
            f"over {say.length(relief)} of relief: rolling ground with real hills and "
            "hollows."
        )
    else:
        key = "hilly"
        why = (
            f"The slope averages {mean:.1f} % and reaches {p95:.1f} % on a twentieth of "
            f"the field, over {say.length(relief)} of relief: hilly ground."
        )
    return {"key": key, "label": CHARACTERS[key], "why": why}


def _valid_extent(grid: ElevationGrid) -> tuple[float, float, float, float]:
    """``(xmin, xmax, ymin, ymax)`` of the valid cell centres."""
    rows = np.flatnonzero(grid.mask.any(axis=1))
    cols = np.flatnonzero(grid.mask.any(axis=0))
    if rows.size == 0:
        xmin, ymin, xmax, ymax = grid.bounds_metric()
        return xmin, xmax, ymin, ymax
    xmin = grid.x0 + (cols[0] + 0.5) * grid.cell
    xmax = grid.x0 + (cols[-1] + 0.5) * grid.cell
    ymax = grid.y0 - (rows[0] + 0.5) * grid.cell
    ymin = grid.y0 - (rows[-1] + 0.5) * grid.cell
    return xmin, xmax, ymin, ymax


def position_label(grid: ElevationGrid, x: float, y: float) -> str:
    """Where a point lies in the field, in words: thirds of the valid extent.

    ``'centre'`` is the middle third both ways; ``'northern part'`` the
    middle third east-west but the northern third north-south; the corners
    are ``'north-west part'`` and so on. Points beyond the extent take the
    nearest third.
    """
    xmin, xmax, ymin, ymax = _valid_extent(grid)
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)
    fx = min(max((float(x) - xmin) / width, 0.0), 1.0)
    fy = min(max((float(y) - ymin) / height, 0.0), 1.0)
    ew = "west" if fx < 1.0 / 3.0 else ("east" if fx > 2.0 / 3.0 else "")
    ns = "south" if fy < 1.0 / 3.0 else ("north" if fy > 2.0 / 3.0 else "")
    if ns and ew:
        return f"{ns}-{ew} part"
    if ns:
        return f"{ns}ern part"
    if ew:
        return f"{ew}ern part"
    return "centre"


# ==========================================================================
# Discrete hills and lows
# ==========================================================================

def _component_stats(
    grid: ElevationGrid,
    labels: np.ndarray,
    count: int,
    sign: float,
    slope_pct: np.ndarray | None,
    measure: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per component: extreme cell, ring median, area, centroid, elongation.

    ``sign`` is +1 for hills (extreme = highest cell, height = extreme
    minus foot) and -1 for lows (extreme = lowest, depth = rim minus
    extreme). The extreme cell is found on the grid's own elevation; the
    height is measured on ``measure`` (default the elevation), which lets
    the caller pass the plane-detrended surface. Each component is examined
    inside its own bounding box so a field with hundreds of knolls does not
    cost hundreds of full-array dilations.
    """
    out: list[dict[str, Any]] = []
    if count == 0:
        return out
    objects = ndimage.find_objects(labels)
    z = grid.z
    measure = z if measure is None else np.asarray(measure, dtype="float64")
    if measure.shape != grid.shape:
        raise ValueError("The surface to measure heights on must be shaped like the grid.")
    for k, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        r0 = max(sl[0].start - 1, 0)
        r1 = min(sl[0].stop + 1, grid.rows)
        c0 = max(sl[1].start - 1, 0)
        c1 = min(sl[1].stop + 1, grid.cols)
        window = (slice(r0, r1), slice(c0, c1))
        comp = labels[window] == k
        zw = z[window]
        mw = measure[window]
        n_cells = int(comp.sum())
        # The extreme cell: highest for a hill, lowest for a low.
        score = np.where(comp, sign * zw, -np.inf)
        rr, cc = np.unravel_index(int(np.argmax(score)), score.shape)
        extreme_z = float(zw[rr, cc])
        # Height above the foot of the feature, on the measuring surface.
        top = float(np.max(sign * mw[comp]) * sign)
        full = np.zeros(grid.shape, dtype=bool)
        full[window] = comp
        ring_level = foot_level(measure, full, sign)
        if not np.isfinite(ring_level):
            # The component fills the whole field: measure against its own
            # far end instead, which is the best the data can say.
            ring_level = float(np.min(sign * mw[comp]) * sign)
        height = sign * (top - ring_level)
        rows_i, cols_i = np.nonzero(comp)
        cy = grid.y0 - (rows_i.mean() + r0 + 0.5) * grid.cell
        cx = grid.x0 + (cols_i.mean() + c0 + 0.5) * grid.cell
        # Elongation from the second moments of the cell positions: the ratio
        # of the principal axes tells a valley trough (long) from a bowl.
        if n_cells > 2:
            cov = np.cov(np.vstack([cols_i, rows_i]).astype("float64"))
            eig = np.linalg.eigvalsh(cov)
            elongation = float(math.sqrt(max(eig[1], 1e-12) / max(eig[0], 1e-12)))
        else:
            elongation = 1.0
        mean_slope = None
        if slope_pct is not None:
            sw = slope_pct[window][comp]
            sw = sw[np.isfinite(sw)]
            mean_slope = float(sw.mean()) if sw.size else 0.0
        out.append({
            "label_id": k,
            "row": int(rr + r0),
            "col": int(cc + c0),
            "extreme_m": extreme_z,
            "top_m": float(top),
            "ring_m": ring_level,
            "height_m": float(height),
            "cells": n_cells,
            "area_ha": n_cells * grid.cell ** 2 / 10_000.0,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "elongation": elongation,
            "mean_slope_pct": mean_slope,
        })
    return out


def foot_level(measure: np.ndarray, patch: np.ndarray, sign: float = 1.0) -> float:
    """The level of the ground at the foot of a hill (``sign`` +1) or the rim
    of a low (``sign`` -1): the median of the ring around the patch, after
    the patch has been grown outward as far as the ground keeps falling
    (rising).

    The patch a caller has is a threshold on the topographic position
    index, a level line somewhere on the flank, so the ring right around it
    is still on the slope and would understate the height by whatever the
    threshold happened to cut off (and the number would move with the TPI
    radius). Walking the ring outward one cell at a time until its median
    stops falling reaches the foot regardless of where the flank was cut:
    on a lone hill that is where the ground goes flat. Only the ring's
    median moves the walk, so a neighbouring hill that the ring runs into
    on one side does not stop it early, and the median of a ring of many
    cells is steady against GPS noise. The walk knows nothing of the other
    features, though: on a rolling field it descends past the saddle to
    the next hill and into the neighbouring hollows, and over-reads the
    height by the depth of those; :func:`saddle_level` is the correction,
    and :func:`hills` takes the higher of the two.

    Returns NaN when the patch has no valid ring at all (it fills the whole
    field). ``measure`` is the surface the height is read on, NaN outside
    the field.
    """
    measure = np.asarray(measure, dtype="float64")
    region = np.asarray(patch, dtype=bool)
    if region.shape != measure.shape:
        raise ValueError("The patch must be shaped like the surface it is measured on.")
    valid = np.isfinite(measure)
    # Work inside a window that grows with the region: a field with
    # hundreds of knolls must not cost hundreds of full-array dilations.
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    if rows.size == 0:
        return float("nan")
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    level = float("nan")
    for _ in range(max(measure.shape)):
        r0, r1 = max(r0 - 1, 0), min(r1 + 1, measure.shape[0])
        c0, c1 = max(c0 - 1, 0), min(c1 + 1, measure.shape[1])
        window = (slice(r0, r1), slice(c0, c1))
        inside = region[window]
        grown = ndimage.binary_dilation(inside, structure=_EIGHT)
        ring = grown & ~inside & valid[window]
        if not ring.any():
            break
        ring_level = float(sign * np.percentile(sign * measure[window][ring], FOOT_PERCENTILE))
        if np.isfinite(level) and not sign * (level - ring_level) > 0.0:
            break
        level = ring_level
        region[window] = grown
        if grown.all() and r0 == 0 and c0 == 0 and (r1, c1) == measure.shape:
            break
    return level


def saddle_level(
    measure: np.ndarray,
    patch: np.ndarray,
    stop: np.ndarray,
    sign: float = 1.0,
    tolerance_m: float = SADDLE_TOLERANCE_M,
) -> float:
    """The level of the saddle between a hill (``sign`` +1) and the ground
    in ``stop``: the highest level at which the connected patch of cells
    above it that holds the hill's summit reaches a ``stop`` cell.

    This is the classic prominence construction — lower a water level
    from the summit and watch the island grow until it joins another
    feature — and it is exact where the ring walk of :func:`foot_level` is
    not: the ring grows geometrically, so on a rolling field it has passed
    the saddle and descended into the neighbouring hollows before it
    stops, and the hill reads 8 % taller than it is. ``stop`` is the cells
    of the other listed features (the other hills and the lows), whose
    contact defines the saddle; a cell 8-adjacent to one counts as
    contact. For a low (``sign`` -1) everything is mirrored: the lowest
    level at which the pool rising from the bottom reaches a hill or
    another low. The level is found by bisection to ``tolerance_m``, one
    component labelling per step. Returns NaN when ``stop`` holds no valid
    cell: a lone hill has no saddle, and its foot is the ring walk's.
    """
    measure = np.asarray(measure, dtype="float64")
    patch = np.asarray(patch, dtype=bool)
    stop = np.asarray(stop, dtype=bool)
    if patch.shape != measure.shape or stop.shape != measure.shape:
        raise ValueError("The patch and the stop cells must be shaped like the surface they lie on.")
    valid = np.isfinite(measure)
    stop = stop & valid & ~patch
    if not stop.any() or not (patch & valid).any():
        return float("nan")
    # Everything is done "downhill from the summit" in units of sign * z.
    up = np.where(valid, sign * measure, -np.inf)
    seed = np.unravel_index(int(np.argmax(np.where(patch, up, -np.inf))), up.shape)
    near = ndimage.binary_dilation(stop, structure=_EIGHT)
    lo = float(up[valid].min()) - tolerance_m  # the whole field: certainly touches
    hi = float(up[seed])  # nothing above the summit: certainly not
    while hi - lo > tolerance_m:
        mid = 0.5 * (lo + hi)
        labels, _ = ndimage.label(up > mid, structure=_EIGHT)
        island = labels == labels[seed]
        if bool((island & near).any()):
            lo = mid
        else:
            hi = mid
    return float(sign * lo)


def _apply_saddles(
    measure: np.ndarray,
    labels: np.ndarray,
    stats: list[dict[str, Any]],
    sign: float,
    min_height_m: float,
    stop: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Re-read the height of every feature that passed the ring walk against
    the saddles to the other listed features: the other survivors of the
    same kind, and ``stop`` (the listed features of the other kind). The
    foot is the higher of the ring walk's level and the saddle (mirrored
    for a low), and a feature that drops under ``min_height_m`` on the
    saddle is dropped: what looked like a hill was the shoulder of another.
    """
    survivors = [s for s in stats if s["height_m"] >= float(min_height_m)]
    if not survivors or (len(survivors) < 2 and stop is None):
        return survivors
    own = np.isin(labels, [s["label_id"] for s in survivors])
    others = own if stop is None else (own | np.asarray(stop, dtype=bool))
    for s in survivors:
        comp = labels == s["label_id"]
        saddle = saddle_level(measure, comp, others & ~comp, sign)
        if np.isfinite(saddle) and sign * (saddle - s["ring_m"]) > 0.0:
            s["ring_m"] = float(saddle)
            s["height_m"] = float(sign * (s["top_m"] - saddle))
    return [s for s in survivors if s["height_m"] >= float(min_height_m)]


def _finish(grid: ElevationGrid, items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Sort by size, number from 1, add coordinates and the position words."""
    items.sort(key=lambda d: (-d["height_m"] * d["area_ha"], -d["area_ha"]))
    out = []
    for rank, d in enumerate(items, start=1):
        x = grid.x0 + (d["col"] + 0.5) * grid.cell
        y = grid.y0 - (d["row"] + 0.5) * grid.cell
        lon, lat = grid.to_lonlat(x, y)
        item = {
            "id": rank,
            "label": f"{'Hill' if kind == 'hill' else 'Low'} {rank}",
            "row": d["row"],
            "col": d["col"],
            "x": float(x),
            "y": float(y),
            "lon": float(lon),
            "lat": float(lat),
            "area_ha": float(d["area_ha"]),
            "position": position_label(grid, d["centroid_x"], d["centroid_y"]),
            "elongation": float(d["elongation"]),
            "label_id": int(d["label_id"]),
        }
        out.append(item)
    return out


def hills(
    grid: ElevationGrid,
    tpi_std: np.ndarray,
    slope_pct: np.ndarray | None = None,
    tpi_std_threshold: float = 1.0,
    min_area_ha: float = MIN_FEATURE_AREA_HA,
    min_height_m: float = 0.5,
    measure: np.ndarray | None = None,
    stop: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """The discrete hills: patches where the standardized large-scale TPI
    exceeds ``tpi_std_threshold``.

    Each hill's ``height_m`` is the top of the patch minus the level of
    its foot — the height a person sees from the foot of the hill, not the
    height above the field's lowest point, and not a fraction of it that
    depends on where the TPI threshold cut the flank. The foot is the
    higher of two readings: the ring around the patch walked outward until
    the ground stops falling (:func:`foot_level`), and the saddle at which
    the hill joins another listed hill or a listed low
    (:func:`saddle_level`), because past the saddle the ring is measuring
    the next feature, not this one. ``stop`` is the cells of the listed
    lows, when the caller has them (:mod:`analysis` runs the lows first
    without a stop to find out which are real); the other hills that pass
    the height test stop the walk whether or not it is given. ``measure``
    is the surface heights are read on (default the elevation itself);
    :mod:`analysis` passes the elevation minus the field's fitted plane,
    which changes nothing for a hill in the interior but stops the highest
    corner of a tilted field, cut off by the edge, from counting as one.
    Patches under ``min_area_ha`` or ``min_height_m`` are dropped: on GPS
    altitude a 0.3 m bump is noise, not a hill.

    Returns ``(hills, labels)``: the hill dicts, largest first, and an int32
    label array whose value at a cell is the ``id`` of the hill it belongs
    to (0 elsewhere), for drawing outlines.
    """
    with np.errstate(invalid="ignore"):
        high = np.asarray(tpi_std, dtype="float64") > float(tpi_std_threshold)
    high &= grid.mask
    min_cells = max(int(math.ceil(float(min_area_ha) * 10_000.0 / grid.cell ** 2)), 1)
    labels, count = connected_components(high, min_cells=min_cells)
    surface = grid.z if measure is None else np.asarray(measure, dtype="float64")
    stats = _apply_saddles(
        surface, labels, _component_stats(grid, labels, count, +1.0, slope_pct, measure),
        +1.0, min_height_m, stop,
    )
    items = _finish(grid, stats, "hill")
    by_id = {s["label_id"]: s for s in stats}
    out = []
    relabel = np.zeros(count + 1, dtype="int32")
    for item in items:
        s = by_id[item["label_id"]]
        relabel[item["label_id"]] = item["id"]
        out.append({
            "id": item["id"],
            "label": item["label"],
            "row": item["row"],
            "col": item["col"],
            "x": item["x"],
            "y": item["y"],
            "lon": item["lon"],
            "lat": item["lat"],
            "area_ha": item["area_ha"],
            "summit_m": float(s["extreme_m"]),
            "height_m": float(s["height_m"]),
            "mean_slope_pct": float(s["mean_slope_pct"]) if s["mean_slope_pct"] is not None else None,
            "position": item["position"],
            "elongation": item["elongation"],
        })
    return out, relabel[labels]


def lows(
    grid: ElevationGrid,
    tpi_std: np.ndarray,
    depression_mask: np.ndarray | None = None,
    slope_pct: np.ndarray | None = None,
    tpi_std_threshold: float = 1.0,
    min_area_ha: float = MIN_FEATURE_AREA_HA,
    min_depth_m: float = 0.5,
    measure: np.ndarray | None = None,
    stop: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """The discrete lows — valley troughs and hollows — where the standardized
    large-scale TPI is under ``-tpi_std_threshold``.

    ``depth_m`` is the level of the rim minus the lowest cell, read on
    ``measure`` (see :func:`hills`); the rim is the lower of the ring walked
    upward (:func:`foot_level`) and the saddle to the listed hills
    (``stop``) and the other listed lows (:func:`saddle_level`). ``closed``
    is True when the patch overlaps a cell that the sink filling found to
    pond (``depression_mask``): somewhere in it water cannot get out; a
    valley that drains off the field is open. ``elongation`` (long axis
    over short axis of the patch) separates a trough from a bowl. Returns
    ``(lows, labels)`` like :func:`hills`.
    """
    with np.errstate(invalid="ignore"):
        low = np.asarray(tpi_std, dtype="float64") < -float(tpi_std_threshold)
    low &= grid.mask
    min_cells = max(int(math.ceil(float(min_area_ha) * 10_000.0 / grid.cell ** 2)), 1)
    labels, count = connected_components(low, min_cells=min_cells)
    surface = grid.z if measure is None else np.asarray(measure, dtype="float64")
    stats = _apply_saddles(
        surface, labels, _component_stats(grid, labels, count, -1.0, slope_pct, measure),
        -1.0, min_depth_m, stop,
    )
    items = _finish(grid, stats, "low")
    by_id = {s["label_id"]: s for s in stats}
    if depression_mask is not None:
        depression_mask = np.asarray(depression_mask, dtype=bool)
        if depression_mask.shape != grid.shape:
            raise ValueError("The depression mask must be shaped like the grid.")
    out = []
    relabel = np.zeros(count + 1, dtype="int32")
    for item in items:
        s = by_id[item["label_id"]]
        relabel[item["label_id"]] = item["id"]
        closed = False
        if depression_mask is not None:
            closed = bool(np.any(depression_mask & (labels == item["label_id"])))
        out.append({
            "id": item["id"],
            "label": item["label"],
            "row": item["row"],
            "col": item["col"],
            "x": item["x"],
            "y": item["y"],
            "lon": item["lon"],
            "lat": item["lat"],
            "area_ha": item["area_ha"],
            "bottom_m": float(s["extreme_m"]),
            "depth_m": float(s["height_m"]),
            "mean_slope_pct": float(s["mean_slope_pct"]) if s["mean_slope_pct"] is not None else None,
            "closed": closed,
            "position": item["position"],
            "elongation": item["elongation"],
        })
    return out, relabel[labels]
