"""Where the water goes: sink filling, flow routing, wetness and ponding.

A farmer asking about relief is really asking two things about water:
where it collects and where it runs. Neither can be read off a slope map.
Water collects in closed depressions — cells with no downhill path to the
field edge — and how much a depression holds depends on the height of the
lowest point on its rim, not on the depth of its bottom. Water runs along
the lines every upslope cell drains to, and a cell is wet when a lot of
ground drains through it and it is too flat to shed the water on.

All of that comes from one operation: filling the sinks. Raising every
depression to its spill level (Barnes et al. 2014, *Priority-Flood*) gives
a surface on which every cell has a downhill path to the edge; the amount
each cell was raised is the depth of water it would pond, and the drainage
directions on the filled surface route every cell to an outlet. The field
edge is the outlet everywhere: NaN cells are the outside world and water
leaving the field there is nobody's problem.

Everything here works on an :class:`~.grid.ElevationGrid`, keeps NaN
outside the field, and returns metric numbers; the interface converts.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString

from .grid import ElevationGrid, connected_components

#: 8-connectivity: water crosses a cell corner as readily as an edge.
_EIGHT = np.ones((3, 3), dtype=bool)

#: D8 direction codes, clockwise from east, as (row step, column step).
#: Row index grows southward, so code 2 (south) is one row down.
D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
D8_NAMES: tuple[str, ...] = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")

#: The step by which sink filling raises each cell above the one the flood
#: reached it from, so that a filled lake drains instead of sitting flat.
#: A nanometre: ten thousand times the rounding step of a float64 at 700 m
#: (a tenth of a picometre), so it survives at any elevation on Earth; the
#: D8 routing sees it as a slope; and across the widest grid the app allows
#: the ramp adds under a hundredth of a millimetre to a depth, so ponded
#: depths and volumes are the real ones.
FILL_EPSILON_M = 1e-9

#: Depth of ponded water below which a cell is not called wet: the vertical
#: precision a GPS altitude can honestly claim after smoothing.
WET_DEPTH_M = 0.05

#: A cell raised by less than this by the sink filling was not in a basin:
#: it is on the epsilon ramp the filling lays across level ground (a
#: nanometre per cell, under a hundredth of a millimetre across the widest
#: grid the app allows). Above it, a raised cell is genuinely below the
#: spill level of the lake it belongs to, however thinly it is covered.
RAISED_DEPTH_M = 1e-4

#: A topographic wetness index below this never marks a cell wet, whatever
#: the field's distribution: 8 is roughly a hectare of catchment on a 1 %
#: slope, the point where ground stays soft after rain on most soils.
MIN_TWI_THRESHOLD = 8.0

_SQRT2 = math.sqrt(2.0)


def _check_shape(values: np.ndarray, grid: ElevationGrid, what: str) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.shape != grid.z.shape:
        raise ValueError(
            f"The {what} is shaped {values.shape} but the grid is {grid.shape}; "
            "compute every hydrology layer on the same grid, starting from "
            "fill_sinks."
        )
    return values


# ==========================================================================
# Sink filling
# ==========================================================================

def fill_sinks(grid: ElevationGrid, epsilon: float = FILL_EPSILON_M) -> np.ndarray:
    """Raise every closed depression to the level at which it spills.

    Priority-flood: the flood starts from every cell on the field edge
    (a valid cell next to NaN or the array border) and advances always
    through the lowest unvisited cell, the way rising water would enter
    from outside. A cell lower than the one the flood reaches it from is
    part of a depression and is raised to that level plus ``epsilon``, so
    the filled surface has no flats and every cell drains strictly downhill
    to the edge. The tiny gradient across a filled basin adds at most
    ``epsilon`` per cell to the reported depth.

    Every cell, raised or not, waits its turn in the priority queue
    (Barnes et al. 2014, *Priority-Flood+ε*). Barnes' faster variant
    hands raised cells to a plain queue that is drained first, and that
    is wrong once an increment is added: on level ground every cell is
    "raised", so whichever edge cell the queue popped first floods the
    whole flat by itself and the ramp climbs from that one corner instead
    of from the nearest edge. A dead-flat 1 000 ha field then carried a
    6 cm 'depression' of 200 000 m³ in its far corner and drained to the
    other. Ordered by level, all the edge cells advance together and a
    flat cell ends up ``epsilon`` per cell above its nearest edge, which
    is where its water goes. The cost is a heap push per cell instead of
    a queue append: 400 000 cells still fill in 0.4 s. The loop runs on
    Python lists rather than arrays because element access is what
    dominates here, and a list is several times faster per element than
    a NumPy scalar.

    Returns the filled elevation, NaN where the grid is masked.
    """
    if not (epsilon >= 0):
        raise ValueError("The sink-filling epsilon must be zero or a positive number of metres.")
    z = grid.z
    rows, cols = z.shape
    width = cols + 2

    # A one-cell NaN pad removes every bounds check from the inner loop and
    # makes the array border look like the field edge.
    padded = np.full((rows + 2, cols + 2), np.nan)
    padded[1:-1, 1:-1] = z
    valid = np.isfinite(padded)
    if not valid.any():
        return z.copy()
    seed = valid & ndimage.binary_dilation(~valid, structure=_EIGHT)

    flat = padded.ravel()
    values = flat.tolist()
    closed = bytearray((~valid | seed).ravel().astype(np.uint8).tobytes())
    seed_idx = np.flatnonzero(seed.ravel())
    heap = list(zip(flat[seed_idx].tolist(), seed_idx.tolist()))
    heapq.heapify(heap)

    offsets = (1, width + 1, width, width - 1, -1, -width - 1, -width, -width + 1)
    eps = float(epsilon)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        zc, c = pop(heap)
        for off in offsets:
            n = c + off
            if closed[n]:
                continue
            closed[n] = 1
            zn = values[n]
            if zn <= zc:
                zn = zc + eps
                values[n] = zn
            push(heap, (zn, n))

    return np.asarray(values, dtype="float64").reshape(rows + 2, cols + 2)[1:-1, 1:-1].copy()


def depression_depth(grid: ElevationGrid, filled: np.ndarray) -> np.ndarray:
    """Depth of water each cell would hold: ``filled - z``, never negative,
    NaN outside the field."""
    filled = _check_shape(filled, grid, "filled surface")
    depth = filled - grid.z
    with np.errstate(invalid="ignore"):
        depth[depth < 0] = 0.0
    depth[~grid.mask] = np.nan
    return depth


def depressions(
    grid: ElevationGrid,
    filled: np.ndarray,
    min_depth_m: float = 0.05,
    min_area_ha: float = 0.02,
) -> list[dict[str, Any]]:
    """The closed basins that pond water, largest volume first.

    A basin is found where an 8-connected patch of cells holds more than
    ``min_depth_m`` of water when full; patches smaller than
    ``min_area_ha`` are ignored. The depth threshold keeps GPS-level noise
    out of the list: a 3 cm dip in a smoothed altitude surface is not a
    place water stands. Once found, a basin is measured over the whole
    lake the sink filling raised around that patch — every connected cell
    under its spill level, however shallow the water on it — because the
    rim of a bowl holds water too: cut at the 5 cm threshold, a gently
    sloping bowl lost a twentieth of its volume to its own edge. Nested
    basins share one lake and are reported as one, filled to the
    outermost spill, because that is the ponded area a wet spring
    produces; so are two pockets separated by a sill under the spill.

    Each dict carries, in plain Python types: ``id``, the ``row``/``col``,
    ``x``/``y`` (metric) and ``lon``/``lat`` of the deepest cell,
    ``area_ha``, ``max_depth_m``, ``mean_depth_m``, ``volume_m3``,
    ``spill_m`` (the elevation the water reaches before it runs out) and
    ``bottom_m`` (the lowest ground in the basin).
    """
    return _basins(grid, filled, min_depth_m, min_area_ha)[0]


def depression_labels(
    grid: ElevationGrid,
    filled: np.ndarray,
    min_depth_m: float = 0.05,
    min_area_ha: float = 0.02,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """:func:`depressions` together with an int32 label array whose value
    at a cell is the ``id`` of the basin it belongs to (0 elsewhere)."""
    return _basins(grid, filled, min_depth_m, min_area_ha)


def _basins(
    grid: ElevationGrid,
    filled: np.ndarray,
    min_depth_m: float,
    min_area_ha: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    depth = depression_depth(grid, filled)
    with np.errstate(invalid="ignore"):
        wet = depth > float(min_depth_m)
        raised = depth > RAISED_DEPTH_M
    min_cells = max(int(math.ceil(float(min_area_ha) * 10_000.0 / grid.cell ** 2)), 1)
    found, n_found = connected_components(wet, min_cells=min_cells)
    if n_found == 0:
        return [], np.zeros(grid.shape, dtype="int32")
    # Grow each patch to its lake. Two raised cells that touch were raised
    # to the same level (the flood reaches a cell from its lowest closed
    # neighbour, so neighbouring raised cells differ by at most epsilon),
    # which makes every 8-connected patch of raised cells one lake at one
    # spill level; a patch found above the threshold lies wholly in one.
    lakes, n_lakes = connected_components(raised)
    lake_of_patch = ndimage.maximum(lakes, found, np.arange(1, n_found + 1)).astype("int64")
    kept = np.unique(lake_of_patch[lake_of_patch > 0])
    remap = np.zeros(n_lakes + 1, dtype="int32")
    remap[kept] = np.arange(1, kept.size + 1, dtype="int32")
    labels = remap[lakes]
    count = int(kept.size)

    index = np.arange(1, count + 1)
    cells = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    depth0 = np.where(np.isfinite(depth), depth, 0.0)
    volume = np.bincount(labels.ravel(), weights=depth0.ravel(), minlength=count + 1)[1:]
    deepest = ndimage.maximum_position(depth0, labels, index)
    max_depth = ndimage.maximum(depth0, labels, index)
    bottom = ndimage.minimum(grid.z, labels, index)
    spill = ndimage.minimum(filled, labels, index)

    rows_i = np.array([r for r, _ in deepest], dtype="int64")
    cols_i = np.array([c for _, c in deepest], dtype="int64")
    x = grid.x0 + (cols_i + 0.5) * grid.cell
    y = grid.y0 - (rows_i + 0.5) * grid.cell
    lon, lat = grid.to_lonlat(x, y)
    area_m2 = cells * grid.cell ** 2
    volume_m3 = volume * grid.cell ** 2

    order = np.argsort(-volume_m3, kind="stable")
    by_rank = np.zeros(count + 1, dtype="int32")
    by_rank[order + 1] = np.arange(1, count + 1, dtype="int32")
    out: list[dict[str, Any]] = []
    for rank, k in enumerate(order, start=1):
        out.append({
            "id": rank,
            "row": int(rows_i[k]),
            "col": int(cols_i[k]),
            "x": float(x[k]),
            "y": float(y[k]),
            "lon": float(lon[k]),
            "lat": float(lat[k]),
            "area_ha": float(area_m2[k] / 10_000.0),
            "max_depth_m": float(max_depth[k]),
            "mean_depth_m": float(volume[k] / cells[k]),
            "volume_m3": float(volume_m3[k]),
            "spill_m": float(spill[k]),
            "bottom_m": float(bottom[k]),
        })
    return out, by_rank[labels]


# ==========================================================================
# Flow routing
# ==========================================================================

def flow_direction_d8(filled: np.ndarray, grid: ElevationGrid) -> np.ndarray:
    """Steepest-descent direction of every cell, as an int8 code.

    Codes 0..7 run clockwise from east (E, SE, S, SW, W, NW, N, NE — see
    ``D8_NAMES``); -1 marks a masked cell or an outlet, a cell whose water
    leaves the field. Diagonal neighbours are ``sqrt(2)`` cells away, so a
    diagonal drop must be 41 % bigger to win.

    At the field edge the ground does not stop where the data does. A
    NaN neighbour is given the elevation the slope would continue to
    (``2 z - z_opposite``), and when that outward direction is the
    steepest the cell is an outlet. Ignoring NaN neighbours instead would
    deflect the water along the downhill edge and gather it into a channel
    there that exists on no map; with the slope continued, a field that
    falls to its south boundary sheds water across the whole of it. On a
    filled surface every non-edge cell has a lower neighbour, so -1 only
    occurs on the edge.
    """
    filled = _check_shape(filled, grid, "filled surface")
    rows, cols = filled.shape
    valid = np.isfinite(filled) & grid.mask
    z = np.where(valid, filled, np.nan)
    padded = np.full((rows + 2, cols + 2), np.nan)
    padded[1:-1, 1:-1] = z

    def window(dr: int, dc: int) -> np.ndarray:
        return padded[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]

    best = np.zeros(z.shape)
    fdir = np.full(z.shape, -1, dtype="int8")
    with np.errstate(invalid="ignore"):
        for code, (dr, dc) in enumerate(D8_OFFSETS):
            neighbour = window(dr, dc)
            opposite = window(-dr, -dc)
            outward = np.isnan(neighbour) & np.isfinite(opposite)
            candidate = np.where(outward, 2.0 * z - opposite, neighbour)
            distance = grid.cell * (_SQRT2 if dr and dc else 1.0)
            slope = (z - candidate) / distance
            better = slope > best
            best[better] = slope[better]
            fdir[better] = np.where(outward[better], -1, code).astype("int8")
    fdir[~valid] = -1
    return fdir


def _receivers(fdir: np.ndarray, grid: ElevationGrid) -> tuple[np.ndarray, np.ndarray]:
    """Flat index of the cell each cell drains to (-1 for none), and the
    flat validity mask. A direction pointing off the array or into a
    masked cell counts as an outlet: that water has left the field."""
    fdir = np.asarray(fdir)
    if fdir.shape != grid.z.shape:
        raise ValueError(
            f"The flow-direction array is shaped {fdir.shape} but the grid is "
            f"{grid.shape}; compute it with flow_direction_d8 on this grid."
        )
    rows, cols = fdir.shape
    valid = grid.mask.ravel()
    d = fdir.ravel().astype("int64")
    has = (d >= 0) & (d < 8) & valid
    idx = np.flatnonzero(has)
    dr = np.array([o[0] for o in D8_OFFSETS], dtype="int64")
    dc = np.array([o[1] for o in D8_OFFSETS], dtype="int64")
    r = idx // cols + dr[d[idx]]
    c = idx % cols + dc[d[idx]]
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    target = np.where(inside, r * cols + c, -1)
    target[inside] = np.where(valid[target[inside]], target[inside], -1)
    receivers = np.full(rows * cols, -1, dtype="int64")
    receivers[idx] = target
    return receivers, valid


def flow_accumulation(fdir: np.ndarray, grid: ElevationGrid) -> np.ndarray:
    """Number of cells draining through each cell, itself included.

    Multiply by the cell area for the catchment. Cells are settled from the
    heads down: a cell passes its total to its receiver once every cell
    that drains into it has passed its own — a topological pass driven by
    in-degree, one vectorised step per level, so the Python loop runs as
    many times as the longest flow path is long, not once per cell. NaN
    outside the field. Cells caught in a cycle (impossible on a filled
    surface, possible on a raw one with flats) keep what had reached them.
    """
    receivers, valid = _receivers(fdir, grid)
    n = receivers.size
    acc = np.where(valid, 1.0, np.nan)
    has = receivers >= 0
    indegree = np.bincount(receivers[has], minlength=n)
    frontier = np.flatnonzero(valid & (indegree == 0))
    while frontier.size:
        target = receivers[frontier]
        flowing = target >= 0
        if not flowing.any():
            break
        target = target[flowing]
        np.add.at(acc, target, acc[frontier[flowing]])
        unique, counts = np.unique(target, return_counts=True)
        indegree[unique] -= counts
        frontier = unique[indegree[unique] == 0]
    return acc.reshape(grid.shape)


def twi(acc: np.ndarray, slope_pct: np.ndarray, grid: ElevationGrid) -> np.ndarray:
    """Topographic wetness index, ``ln(a / tan beta)``.

    ``a`` is the specific catchment area — upstream area per metre of
    contour, i.e. ``acc * cell`` — and ``beta`` the local slope. Flat
    ground with a big catchment scores high (water arrives and does not
    leave); a steep cell with nothing above it scores low. ``tan beta`` is
    floored at 0.001 (0.1 %) so a dead-flat cell scores high rather than
    infinite. NaN outside the field.
    """
    acc = _check_shape(acc, grid, "flow accumulation")
    slope_pct = _check_shape(slope_pct, grid, "slope")
    with np.errstate(invalid="ignore", divide="ignore"):
        tan_beta = np.maximum(slope_pct / 100.0, 0.001)
        out = np.log(acc * grid.cell / tan_beta)
    out[~grid.mask | ~np.isfinite(acc) | ~np.isfinite(slope_pct)] = np.nan
    return out


def drainage_lines(
    fdir: np.ndarray,
    acc: np.ndarray,
    grid: ElevationGrid,
    min_upstream_ha: float = 1.0,
) -> tuple[list[LineString], float]:
    """The lines water runs along, as LineStrings of cell centres (metric CRS).

    A channel cell drains at least ``min_upstream_ha`` of ground. Each line
    starts at a head — a channel cell no other channel cell drains into —
    and follows the D8 directions to the outlet or to the first cell
    already on another line, so the network shares no segment twice. Heads
    are traced longest path first, which keeps the main line whole from
    head to outlet and lets tributaries end where they join it.

    Returns ``(lines, total_length_m)``.
    """
    acc = _check_shape(acc, grid, "flow accumulation")
    receivers, _valid = _receivers(fdir, grid)
    with np.errstate(invalid="ignore"):
        channel = (acc.ravel() * grid.cell ** 2 / 10_000.0) >= float(min_upstream_ha)
    channel_idx = np.flatnonzero(channel)
    if channel_idx.size == 0:
        return [], 0.0

    fed = np.zeros(receivers.size, dtype=bool)
    downstream = receivers[channel_idx]
    fed[downstream[downstream >= 0]] = True
    heads = channel_idx[~fed[channel_idx]]

    rec = receivers.tolist()

    def path_length(start: int) -> int:
        steps, c = 0, start
        while True:
            c = rec[c]
            if c < 0:
                return steps
            steps += 1

    lengths = np.array([path_length(int(h)) for h in heads], dtype="int64")
    order = np.argsort(-lengths, kind="stable")

    X, Y = grid.xy()
    xs, ys = X.ravel(), Y.ravel()
    visited = bytearray(receivers.size)
    lines: list[LineString] = []
    total = 0.0
    for k in order:
        c = int(heads[k])
        path = [c]
        while True:
            c = rec[c]
            if c < 0:
                break
            path.append(c)
            if visited[c]:
                break
        for p in path:
            visited[p] = 1
        if len(path) < 2:
            continue
        idx = np.asarray(path, dtype="int64")
        line = LineString(np.column_stack([xs[idx], ys[idx]]))
        lines.append(line)
        total += line.length
    return lines, float(total)


# ==========================================================================
# Wetness
# ==========================================================================

def default_twi_threshold(twi_values: np.ndarray) -> float:
    """The wetness index above which a cell counts as likely wet: the
    field's own 90th percentile, but never below ``MIN_TWI_THRESHOLD``.

    Relative because fields differ — a flat one would otherwise be wet
    everywhere, a rolling one nowhere; floored because a field that is
    simply well drained everywhere should not be told a tenth of it is wet.
    """
    finite = np.asarray(twi_values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MIN_TWI_THRESHOLD
    return float(max(np.percentile(finite, 90.0), MIN_TWI_THRESHOLD))


def wet_cells(
    twi_values: np.ndarray,
    depth: np.ndarray,
    twi_threshold: float | None = None,
) -> np.ndarray:
    """Cells that are likely wet: ponding more than ``WET_DEPTH_M`` of
    water, or with a wetness index at or above the threshold (default
    :func:`default_twi_threshold`). False outside the field."""
    twi_values = np.asarray(twi_values, dtype="float64")
    depth = np.asarray(depth, dtype="float64")
    if twi_values.shape != depth.shape:
        raise ValueError("The wetness index and depression depth must be shaped alike.")
    threshold = default_twi_threshold(twi_values) if twi_threshold is None else float(twi_threshold)
    with np.errstate(invalid="ignore"):
        return (depth > WET_DEPTH_M) | (twi_values >= threshold)


# ==========================================================================
# The whole chain
# ==========================================================================

@dataclass
class Hydrology:
    """Every hydrology layer of one grid, computed in the right order."""

    filled: np.ndarray
    depth: np.ndarray
    fdir: np.ndarray
    acc: np.ndarray
    twi: np.ndarray
    wet: np.ndarray
    twi_threshold: float
    depressions: list[dict[str, Any]] = field(default_factory=list)
    drainage_lines: list[LineString] = field(default_factory=list)
    drainage_length_m: float = 0.0
    #: Basin ``id`` at each cell of a listed basin's lake, 0 elsewhere.
    depression_labels: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (no arrays): the numbers a report quotes."""
        finite = self.depth[np.isfinite(self.depth)]
        wet_count = int(np.count_nonzero(self.wet))
        return {
            "depressions": self.depressions,
            "ponded_cells": int(np.count_nonzero(finite > WET_DEPTH_M)),
            "ponded_volume_m3": float(sum(d["volume_m3"] for d in self.depressions)),
            "wet_cells": wet_count,
            "twi_threshold": float(self.twi_threshold),
            "drainage_lines": len(self.drainage_lines),
            "drainage_length_m": float(self.drainage_length_m),
        }


def analyse_hydrology(
    grid: ElevationGrid,
    slope_pct: np.ndarray,
    epsilon: float = FILL_EPSILON_M,
    min_depth_m: float = 0.05,
    min_area_ha: float = 0.02,
    min_upstream_ha: float = 1.0,
    twi_threshold: float | None = None,
) -> Hydrology:
    """Fill, route, index and trace in one call.

    The steps depend on each other in a fixed order (directions need the
    filled surface, accumulation the directions, wetness both) and this
    is the one place that order is written down.
    """
    filled = fill_sinks(grid, epsilon=epsilon)
    depth = depression_depth(grid, filled)
    fdir = flow_direction_d8(filled, grid)
    acc = flow_accumulation(fdir, grid)
    index = twi(acc, slope_pct, grid)
    threshold = default_twi_threshold(index) if twi_threshold is None else float(twi_threshold)
    wet = wet_cells(index, depth, threshold)
    basins, basin_labels = depression_labels(
        grid, filled, min_depth_m=min_depth_m, min_area_ha=min_area_ha
    )
    lines, length = drainage_lines(fdir, acc, grid, min_upstream_ha=min_upstream_ha)
    return Hydrology(
        filled=filled,
        depth=depth,
        fdir=fdir,
        acc=acc,
        twi=index,
        wet=wet,
        twi_threshold=threshold,
        depressions=basins,
        drainage_lines=lines,
        drainage_length_m=length,
        depression_labels=basin_labels,
    )
