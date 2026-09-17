"""The elevation grid every terrain derivative is computed on.

A yield or as-applied file carries one GPS altitude per logged point, along
the track the machine drove. Nothing about relief — slope, curvature, where
water collects — can be read off scattered points directly, so the first step
is always the same: put the elevation on a regular square grid. Every later
module (slope, landforms, depressions, contours, zones) works on the
:class:`ElevationGrid` built here, which is why its geometry is spelled out so
carefully: row 0 is the north edge, cell centres are at half-cell offsets, and
``rowcol`` maps a cell centre to an integer index, exactly the convention a
GeoTIFF uses. A DEM read from disk and a grid built from points therefore look
identical to everything downstream.

Turning GPS altitudes into a grid is not just interpolation. Monitor altitude
is the least reliable number in the file: it spikes when the receiver loses
satellites, and it shifts by a constant between passes when the receiver
re-converges after a headland turn. :func:`grid_from_points` removes both
before gridding, and reports how much of each it found — the pass-offset
spread and the residual vertical noise are the honest quality indicators the
findings quote. Two more failures are caught by value rather than by
neighbourhood, because every neighbour agrees with them: a run logged at 0 m
or a no-data value (the receiver had no fix), and a whole pass sitting tens
of metres off the rest of the field; and an altitude quantised to whole
metres is recognised so its risers are not taken for spikes or slopes.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from ..core import schema as sch
from . import notes as notes_mod

#: Weight below which a normalized convolution has no valid support.
_MIN_WEIGHT = 1e-9

#: The pass-offset system is one dense equation per pass; beyond this many
#: passes (a pass detector gone wrong on a curvy field) it is not worth it.
_MAX_PASSES = 2000

#: Fewer readings than this cannot describe a relief.
_MIN_POINTS = 30

#: Swath widths outside this range are not implements but mis-mapped columns.
_MIN_SWATH_M = 1.0
_MAX_SWATH_M = 60.0

#: The pass-offset search always looks at least this far around a point.
_MIN_PAIR_RADIUS_M = 15.0

#: Point pairs the pass-offset search may materialise at once (two int64
#: each, plus the per-pair working arrays: about 250 MB at this size).
_MAX_PAIRS = 4_000_000

#: The default smoothing leaves about this much altitude noise on the
#: surface (well under the 5 cm that counts as ponding), or, on a coarse
#: grid, whatever keeps the slope noise across two cells under
#: :data:`_TARGET_SLOPE_NOISE` — a large cell already spans enough relief
#: for a few centimetres not to matter.
_TARGET_SURFACE_NOISE_M = 0.03
_TARGET_SLOPE_NOISE = 0.004

#: Value steps an elevation is commonly quantised to, coarsest first: whole
#: metres (SRTM, every integer raster, a monitor that logs the altitude as
#: an integer), half metres, decimetres.
VALUE_STEPS = (1.0, 0.5, 0.1)

#: Readings needed before a step is declared: three values on whole metres
#: can be chance, fifty cannot.
_MIN_STEP_SAMPLES = 50

#: Default smoothing scale for a quantised elevation, in multiples of its
#: value step. Whole metres turn a gentle slope into terraces with 1 m
#: risers; a Gaussian of sigma s spreads a riser of height q into a ramp
#: whose steepest point is q / (s * sqrt(2 pi)), so ten times the step
#: brings a 1 m riser under 4 %, the slope of the ground it is cut into,
#: while a 90 m hill loses a tenth of its height at most.
QUANTISED_SMOOTH_STEPS = 10.0

#: Altitudes a receiver or an export writes when there is no height: a
#: plain zero when the fix is lost, and the usual raster no-data markers.
_FILL_VALUES = (0.0, -9999.0, -32768.0)

#: A pass whose median altitude is farther than this from the other passes
#: is not on the same ground: no field falls 30 m between two adjacent
#: passes. Relative to the spread of the pass medians, so a long field on
#: a hillside, whose passes genuinely climb, is left alone.
_MIN_PASS_JUMP_M = 30.0
_PASS_JUMP_MADS = 10.0

#: 8-connectivity for component labelling: a diagonal neighbour is still a
#: neighbour on a square grid, otherwise a ridge crossed at 45 degrees splits
#: one feature in two.
_EIGHT = np.ones((3, 3), dtype=bool)


@functools.lru_cache(maxsize=64)
def _transformer(src: str, dst: str):
    """Cached pyproj transformer; building one costs milliseconds and the
    grid asks for the same pair thousands of times while rendering."""
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


# ==========================================================================
# The grid
# ==========================================================================

@dataclass
class ElevationGrid:
    """A square-celled elevation raster in a metric CRS.

    Attributes
    ----------
    z:
        2-D float64 array ``(rows, cols)``; NaN outside the field. Row 0 is
        the NORTH edge, so ``y`` decreases as the row index grows, like a
        GeoTIFF read through rasterio.
    x0, y0:
        Metric coordinates of the WEST edge of column 0 and the NORTH edge of
        row 0 — the outer corner of the array, not a cell centre.
    cell:
        Cell size in metres (cells are square).
    crs:
        Metric CRS accepted by pyproj, e.g. ``"EPSG:32612"``.
    """

    z: np.ndarray
    x0: float
    y0: float
    cell: float
    crs: str

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype="float64")
        if z.ndim != 2 or z.size == 0:
            raise ValueError("An elevation grid needs a 2-D array with at least one cell.")
        if not (self.cell > 0):
            raise ValueError("The grid cell size must be a positive number of metres.")
        self.z = z
        self.x0 = float(self.x0)
        self.y0 = float(self.y0)
        self.cell = float(self.cell)
        self.crs = str(self.crs)

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return int(self.z.shape[0])

    @property
    def cols(self) -> int:
        return int(self.z.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def mask(self) -> np.ndarray:
        """True where the cell holds an elevation (inside the field)."""
        return ~np.isnan(self.z)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell CENTRE coordinates as two 2-D arrays shaped like ``z``."""
        xs = self.x0 + (np.arange(self.cols) + 0.5) * self.cell
        ys = self.y0 - (np.arange(self.rows) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def rowcol(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) of metric coordinates.

        A cell centre maps to an integer index, which is what
        ``scipy.ndimage.map_coordinates`` expects.
        """
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        row = (self.y0 - y) / self.cell - 0.5
        col = (x - self.x0) / self.cell - 0.5
        return row, col

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Metric coordinates to WGS84 longitude / latitude."""
        from ..core import crs as crs_mod

        lon, lat = _transformer(self.crs, crs_mod.WGS84).transform(
            np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
        )
        return np.asarray(lon), np.asarray(lat)

    def from_lonlat(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """WGS84 longitude / latitude to this grid's metric CRS."""
        from ..core import crs as crs_mod

        x, y = _transformer(crs_mod.WGS84, self.crs).transform(
            np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
        )
        return np.asarray(x), np.asarray(y)

    def bounds_metric(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the OUTER edges of the array."""
        return (
            self.x0,
            self.y0 - self.rows * self.cell,
            self.x0 + self.cols * self.cell,
            self.y0,
        )

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in WGS84.

        Corners alone are not enough: a projected rectangle has curved edges
        in longitude/latitude, so the edge midpoints are transformed too and
        the extremes taken over all eight points.
        """
        xmin, ymin, xmax, ymax = self.bounds_metric()
        xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        xs = np.array([xmin, xmax, xmax, xmin, xmid, xmax, xmid, xmin])
        ys = np.array([ymax, ymax, ymin, ymin, ymax, ymid, ymin, ymid])
        lon, lat = self.to_lonlat(xs, ys)
        return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    def area_ha(self) -> float:
        """Area of the valid cells, in hectares."""
        return self.valid_count * self.cell ** 2 / 10_000.0

    def transform(self):
        """Affine transform in the rasterio convention (pixel -> world)."""
        from affine import Affine

        return Affine(self.cell, 0.0, self.x0, 0.0, -self.cell, self.y0)

    # ------------------------------------------------------------------
    # Derived grids
    # ------------------------------------------------------------------
    def with_values(self, values: np.ndarray) -> "ElevationGrid":
        """A new grid sharing this geometry and carrying ``values``."""
        values = np.asarray(values, dtype="float64")
        if values.shape != self.z.shape:
            raise ValueError(
                f"Values shaped {values.shape} do not fit a grid of {self.shape}; "
                "a derived layer must be computed on this same grid."
            )
        return ElevationGrid(values, self.x0, self.y0, self.cell, self.crs)

    def fill_nearest(self, values: np.ndarray | None = None) -> np.ndarray:
        """``values`` (default ``z``) with every NaN replaced by the nearest
        valid value — the standard trick before convolving, so a kernel near
        the field edge sees plausible ground instead of a hole. The result is
        re-masked by the caller."""
        return fill_nearest(self.z if values is None else values)

    def sample(self, x, y, values: np.ndarray | None = None, order: int = 1):
        """Values of the grid at metric coordinates.

        NaN outside the array and wherever the nearest cell is masked. The
        interpolation runs on a nearest-filled copy and is then re-masked by
        the nearest cell, so a point just inside the field edge gets the
        edge value instead of NaN bleeding in from the outside.
        """
        values = self.z if values is None else np.asarray(values, dtype="float64")
        if values.shape != self.z.shape:
            raise ValueError("The values to sample must be shaped like the grid.")
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        row, col = self.rowcol(x, y)
        row = np.atleast_1d(row)
        col = np.atleast_1d(col)

        # mode='nearest' clamps the outer half-cell rim to the edge cell; the
        # rim is inside the raster and must not come back NaN. Everything
        # beyond the edge is caught by the nearest-cell test below.
        filled = fill_nearest(values)
        out = ndimage.map_coordinates(
            filled, [row, col], order=order, mode="nearest", prefilter=order > 1
        )

        rn = np.rint(row)
        cn = np.rint(col)
        inside = (rn >= 0) & (rn < self.rows) & (cn >= 0) & (cn < self.cols)
        inside &= np.isfinite(row) & np.isfinite(col)
        valid = np.zeros(row.shape, dtype=bool)
        valid[inside] = np.isfinite(values[rn[inside].astype(int), cn[inside].astype(int)])
        out = np.where(valid, out, np.nan)
        return float(out[0]) if scalar else out

    def coarsen(self, factor: int) -> "ElevationGrid":
        """A grid with ``factor`` x ``factor`` blocks averaged into one cell.

        Used to cap the cell count: a 1 m DEM of a section is 6 million
        cells, far more than any relief question needs. Blocks that are
        entirely NaN stay NaN; partial blocks take the mean of what they hold.
        """
        factor = int(factor)
        if factor < 1:
            raise ValueError("The coarsening factor must be a positive integer.")
        if factor == 1:
            return ElevationGrid(self.z.copy(), self.x0, self.y0, self.cell, self.crs)
        rows = -(-self.rows // factor) * factor
        cols = -(-self.cols // factor) * factor
        padded = np.full((rows, cols), np.nan)
        padded[: self.rows, : self.cols] = self.z
        blocks = padded.reshape(rows // factor, factor, cols // factor, factor)
        with warnings.catch_warnings():
            # An all-NaN block is a legitimate empty cell, not a warning.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = np.nanmean(blocks, axis=(1, 3))
        return ElevationGrid(z, self.x0, self.y0, self.cell * factor, self.crs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe geometry summary for the interface."""
        west, south, east, north = self.bounds_lonlat()
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cell_m": self.cell,
            "crs": self.crs,
            "bounds_lonlat": [west, south, east, north],
            "area_ha": self.area_ha(),
        }


# ==========================================================================
# Raster helpers shared by the derivative modules
# ==========================================================================

def fill_nearest(values: np.ndarray) -> np.ndarray:
    """``values`` with each NaN replaced by the nearest finite value."""
    values = np.asarray(values, dtype="float64")
    missing = np.isnan(values)
    if not missing.any():
        return values.copy()
    if missing.all():
        return values.copy()
    _, (ri, ci) = ndimage.distance_transform_edt(missing, return_indices=True)
    return values[ri, ci]


def nan_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Normalized convolution: the kernel-weighted mean of the valid cells.

    Convolving a field with holes directly would spread NaN over every cell a
    kernel touches, and treating the holes as zero would drag the field edge
    down. Weighting by validity does neither: each output is the mean of the
    valid cells under the kernel, and NaN only where there are none. The
    result extends beyond the original mask — callers re-mask when the
    original footprint is what they want.

    ``kernel`` is any 2-D float array; odd sizes keep it centred.
    """
    # scipy.signal pulls in scipy.stats and takes about half a second to
    # load; the server imports this package at start-up, so it is fetched
    # here, the first time a convolution is actually run.
    from scipy import signal

    values = np.asarray(values, dtype="float64")
    kernel = np.asarray(kernel, dtype="float64")
    if kernel.ndim != 2:
        raise ValueError("The convolution kernel must be a 2-D array.")
    valid = np.isfinite(values).astype("float64")
    filled = np.where(valid > 0, values, 0.0)
    # 'auto' switches to FFT for the large kernels a 50 m smoothing window
    # on a 2 m grid produces; direct convolution would take minutes there.
    num = signal.convolve(filled, kernel, mode="same", method="auto")
    den = signal.convolve(valid, kernel, mode="same", method="auto")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < _MIN_WEIGHT] = np.nan
    return out


def disc_kernel(radius_cells: float) -> np.ndarray:
    """A filled circle of radius ``radius_cells``, normalized to sum 1.

    A circular window is the natural neighbourhood on the ground: a square
    one reaches 41 % farther along the diagonals, which biases every
    neighbourhood statistic toward the diagonal directions.
    """
    radius = float(radius_cells)
    if radius < 0:
        raise ValueError("The disc radius cannot be negative.")
    n = max(int(math.ceil(radius)), 0)
    offsets = np.arange(-n, n + 1)
    ii, jj = np.meshgrid(offsets, offsets, indexing="ij")
    disc = (ii ** 2 + jj ** 2 <= radius ** 2).astype("float64")
    disc[n, n] = 1.0  # the centre always belongs, whatever the radius
    return disc / disc.sum()


def gaussian_kernel(sigma_cells: float) -> np.ndarray:
    """A 2-D Gaussian truncated at three sigma, normalized to sum 1."""
    sigma = float(sigma_cells)
    half = max(int(math.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-half, half + 1, dtype="float64")
    line = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = np.outer(line, line)
    return kernel / kernel.sum()


def gaussian_smooth(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing with the kernel truncated at 3 sigma."""
    values = np.asarray(values, dtype="float64")
    if not (sigma_cells > 0):
        return values.copy()
    return nan_convolve(values, gaussian_kernel(sigma_cells))


def connected_components(mask: np.ndarray, min_cells: int = 1) -> tuple[np.ndarray, int]:
    """Label the 8-connected regions of a boolean mask.

    Returns ``(labels, count)`` with labels numbered 1..count and 0 outside;
    regions smaller than ``min_cells`` are dropped and the rest renumbered,
    so a caller can iterate ``range(1, count + 1)`` without gaps.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=_EIGHT)
    labels = labels.astype("int32")
    if count == 0:
        return labels, 0
    if min_cells > 1:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = sizes >= int(min_cells)
        keep[0] = False
        remap = np.zeros(count + 1, dtype="int32")
        remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype="int32")
        labels = remap[labels]
        count = int(keep.sum())
    return labels, int(count)


def value_step(values: np.ndarray, min_samples: int = _MIN_STEP_SAMPLES) -> float:
    """The step the elevations are quantised to, or 0 when they are not.

    Shared by the DEM reader and the point gridder because both suffer
    the same way. An SRTM tile, and any DEM QGIS converted from an
    integer raster, holds whole metres: a 0.8 % slope then reads as a
    staircase of 1 m risers 125 m apart, each a 20 % slope over one 5 m
    cell. A monitor that logs the altitude to the metre does worse: the
    neighbour residuals the spike filter and the noise estimate rest on
    are mostly exactly zero, their MAD collapses, and every reading on a
    riser is thrown out as a spike. Knowing the step lets both callers
    widen the spike threshold, smooth the risers away and refuse to
    report features smaller than a step.

    The test is that every finite value lies on the lattice, tried from
    the coarsest step down. The count of DISTINCT values is deliberately
    not a criterion: a flat field in whole metres holds a dozen distinct
    values and is exactly the case that needs the step known. What guards
    against chance is the number of readings, ``min_samples``.
    """
    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size < max(int(min_samples), 1):
        return 0.0
    for step in VALUE_STEPS:
        scaled = finite / step
        if np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-3):
            return float(step)
    return 0.0


# ==========================================================================
# From points to grid
# ==========================================================================

def _group_median(keys: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of ``values`` per integer key in ``0..n_groups-1``; NaN for an
    empty group. One sort instead of a Python loop over groups."""
    finite = np.isfinite(values)
    keys = keys[finite]
    values = values[finite]
    out = np.full(n_groups, np.nan)
    if keys.size == 0:
        return out
    order = np.lexsort((values, keys))
    k = keys[order]
    v = values[order]
    ids = np.arange(n_groups)
    starts = np.searchsorted(k, ids, side="left")
    ends = np.searchsorted(k, ids, side="right")
    counts = ends - starts
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    out[has] = 0.5 * (v[lo] + v[hi])
    return out


def _median_swath(df) -> tuple[float, dict | None]:
    """Median swath width, and the note about it as the fact it states.

    The width is capped at :data:`_MAX_SWATH_M`: a column that reads
    thousands of metres is a mis-mapped field (a total width, centimetres),
    and every tolerance below scales with the swath, so an absurd value
    would turn the whole grid into one 'field' and the pass search into
    every pair of points.
    """
    if sch.SWATH in df.columns:
        swath = df[sch.SWATH].to_numpy(dtype="float64", na_value=np.nan)
        swath = swath[np.isfinite(swath) & (swath > 0)]
        if swath.size:
            median = float(np.median(swath))
            if median > _MAX_SWATH_M:
                return _MAX_SWATH_M, notes_mod.fact(
                    "swath_capped", declared_m=median, used_m=_MAX_SWATH_M)
            return max(median, _MIN_SWATH_M), None
    return 10.0, notes_mod.fact("swath_assumed", swath_m=10.0)


def _coordinate_problem(df) -> str:
    """Why the coordinates cannot be used, for the error message.

    The two ways a file with latitude and longitude columns ends up with no
    usable position are empty cells (a wrong column mapped, comma decimals
    read as text) and values out of range (latitude and longitude swapped,
    or a projected coordinate in a lon/lat column). They call for different
    fixes, so the message tells them apart.
    """
    if sch.LON not in df.columns or sch.LAT not in df.columns:
        return (
            "This file carries no coordinates, so its elevation cannot be placed "
            "on the ground. Open a file with latitude and longitude."
        )
    lon = df[sch.LON].to_numpy(dtype="float64", na_value=np.nan)
    lat = df[sch.LAT].to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not finite.any():
        return (
            "This file's latitude and longitude are empty in every row, so its "
            "elevation cannot be placed on the ground. Check that the right "
            "columns were mapped and that the numbers use a decimal point."
        )
    out = (np.abs(lat[finite]) > 90.0) | (np.abs(lon[finite]) > 180.0)
    if out.mean() > 0.5:
        i = int(np.nonzero(finite)[0][np.nonzero(out)[0][0]])
        return (
            f"The coordinates are out of range (latitude {lat[i]:g}, longitude "
            f"{lon[i]:g}): latitude and longitude look swapped, or the columns "
            "hold projected metres rather than degrees. Fix the column mapping."
        )
    return (
        f"Only {int(finite.sum())} rows carry both a position and an altitude; "
        "at least 30 are needed to describe a relief. Open a file covering "
        "the whole field."
    )


def _neighbour_residual(tree: cKDTree, z: np.ndarray, k: int = 12) -> np.ndarray:
    """Each altitude minus the median of its ``k`` nearest neighbours.

    The residual is the file's own measure of altitude noise: over a few
    metres the ground is smooth, so whatever a reading differs from its
    neighbours by is the receiver, not the relief.
    """
    n = z.size
    kk = min(k + 1, n)
    _, idx = tree.query(tree.data, k=kk, workers=-1)
    neighbours = z[idx[:, 1:]] if kk > 1 else z[:, None]
    return z - np.median(neighbours, axis=1)


def _remove_outliers(tree: cKDTree, z: np.ndarray, k: int = 12, step: float = 0.0) -> np.ndarray:
    """Mask of points whose altitude disagrees with their neighbourhood.

    A GPS altitude spike is a single reading metres off the ground around
    it. Comparing each point with the median of its nearest neighbours picks
    that out; the threshold scales with the file's own scatter (MAD), with a
    floor so a very quiet file does not lose genuine points.

    ``step`` is the value step of a quantised altitude (see
    :func:`value_step`). There the residuals are mostly exactly zero, the
    MAD collapses to nothing and the floor alone would discard every
    reading on a riser — a quarter of a whole-metre file — so the threshold
    is at least three steps.
    """
    residual = _neighbour_residual(tree, z, k)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    threshold = max(0.5, 3.5 * 1.4826 * mad, 3.0 * float(step))
    return np.abs(residual) > threshold


def _fill_value_mask(z: np.ndarray) -> np.ndarray:
    """Readings that are a fill value rather than a height.

    A receiver that has lost its fix logs an altitude of exactly 0, and
    some exports write -9999 or -32768 for the same thing. Such a reading
    is not a spike the neighbour filter can see when a whole run of them
    agrees with each other, so they are picked out by value — but only
    when the field's own level is far from that value, because a polder
    really does lie at 0 m. 'Far' is the same tolerance a stray pass gets.
    """
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    tolerance = max(_MIN_PASS_JUMP_M, _PASS_JUMP_MADS * mad)
    bad = np.zeros(z.size, dtype=bool)
    for value in _FILL_VALUES:
        if abs(median - value) > tolerance:
            bad |= z == value
    return bad


def _stray_passes(z: np.ndarray, pass_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Passes whose altitude is not on the same ground as the rest.

    When the receiver loses its fix for one run and logs whatever it has —
    zero, or the last height it knew from another field — every neighbour
    on that pass agrees, so the spike filter keeps it, and the pass
    detrending then blends a hundred-metre offset into the adjacent passes
    as if it were a hillside. The pass has to be judged as a whole: its
    median is compared with the median of the other passes' medians, and
    it is dropped when the two differ by more than
    :data:`_MIN_PASS_JUMP_M` or :data:`_PASS_JUMP_MADS` times the MAD of
    the pass medians, whichever is larger. Fewer than three passes cannot
    say which one is wrong.

    Returns ``(passes, medians, stray)``: the sorted pass ids, their
    median altitude and a flag per pass.
    """
    passes, index = np.unique(pass_id, return_inverse=True)
    medians = _group_median(index, z, passes.size)
    stray = np.zeros(passes.size, dtype=bool)
    if passes.size < 3:
        return passes, medians, stray
    centre = float(np.nanmedian(medians))
    mad = float(np.nanmedian(np.abs(medians - centre)))
    tolerance = max(_MIN_PASS_JUMP_M, _PASS_JUMP_MADS * mad)
    for k in range(passes.size):
        others = np.delete(medians, k)
        stray[k] = abs(medians[k] - float(np.nanmedian(others))) > tolerance
    return passes, medians, stray


def _point_noise(tree: cKDTree, z: np.ndarray, k: int = 12) -> float:
    """Robust standard deviation of the altitude noise, in metres.

    Each reading is compared with the plane fitted through its ``k``
    nearest neighbours: over a few metres the ground is a plane, so the
    residual is the receiver's noise and nothing else (a median of the
    neighbours would read the slope itself as noise wherever they lie
    mostly on one side, as on an outer pass). The residual carries the
    reading's noise plus a little of the fit's, about 1.05 sigma; MAD
    rather than the standard deviation so a few spikes that survived the
    filter do not inflate it.
    """
    n = z.size
    if n < 5:
        return 0.0
    kk = min(k + 1, n)
    _, idx = tree.query(tree.data, k=kk, workers=-1)
    nb = idx[:, 1:]
    dx = tree.data[nb, 0] - tree.data[:, None, 0]
    dy = tree.data[nb, 1] - tree.data[:, None, 1]
    zn = z[nb]
    ones = np.ones_like(dx)
    A = np.stack([dx, dy, ones], axis=-1)                # (n, k, 3)
    ata = np.einsum("nki,nkj->nij", A, A)
    atz = np.einsum("nki,nk->ni", A, zn)
    # A whisker of ridge keeps a single pass (neighbours on one line) solvable.
    ridge = (1e-6 * (ata[:, 0, 0] + ata[:, 1, 1]) + 1e-9)[:, None, None] * np.eye(3)[None]
    beta = np.linalg.solve(ata + ridge, atz[..., None])[..., 0]
    residual = z - beta[:, 2]
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return 1.4826 * mad / 1.05


def _ridge_lambda(A: np.ndarray, b: np.ndarray, n_iter: int = 30) -> float:
    """Ridge parameter for the pass-offset system, learned from the data.

    The offsets are a random effect (white from pass to pass) and the
    equations carry noise; the right shrinkage is the ratio of the two
    variances, and neither is known in advance — a receiver that never lost
    lock has offsets of a centimetre, a bad day gives half a metre. EM on
    the random-effects model estimates both from the equations themselves,
    so a quiet file gets almost no correction and a noisy one gets a firm
    one. (Generalized cross-validation was tried first and under-shrinks
    badly whenever the offsets are large.)
    """
    n, p = A.shape
    if n < 2:
        return 1.0
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    ub = u.T @ b
    scale = max(float(np.var(b)), 1e-8)
    var_o, var_e = scale, 0.1 * scale
    null = p - s.size
    for _ in range(n_iter):
        lam = var_e / var_o
        coef = s * ub / (s ** 2 + lam)
        posterior = var_e / (s ** 2 + lam)
        var_o = (float(np.sum(coef ** 2)) + float(np.sum(posterior)) + null * var_e / lam) / p
        fitted = u @ (s * coef)
        var_e = (float(np.sum((b - fitted) ** 2)) + float(np.sum(s ** 2 * posterior))) / n
        var_o = max(var_o, 1e-12)
        var_e = max(var_e, 1e-12)
    return float(np.clip(var_e / var_o, 1e-4, 10.0))


def _estimate_pairs(tree: cKDTree, radius: float, rng: np.random.Generator) -> float:
    """Roughly how many point pairs lie within ``radius`` of each other.

    Counted on a sample before the pairs are materialised, because the
    pair list is what decides the memory of the pass search: a dense
    sprayer log within a 50 m radius is hundreds of neighbours per point.
    """
    n = tree.n
    sample = tree.data if n <= 2000 else tree.data[rng.choice(n, 2000, replace=False)]
    counts = tree.query_ball_point(sample, radius, return_length=True, workers=-1)
    return 0.5 * n * max(float(np.mean(counts)) - 1.0, 0.0)


def _detrend_passes(
    tree: cKDTree,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pass_id: np.ndarray,
    radius: float,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate the constant vertical offset of each pass.

    Each point is compared with the mean altitude of the points from OTHER
    passes around it. With neighbours on both sides the ground slope cancels
    to first order, so the median of that difference over a pass measures
    the pass's offset relative to its neighbours' — a second difference
    across passes. Those relations form one small linear system per field,
    solved with ridge shrinkage: an offset pattern that varies slowly from
    pass to pass is indistinguishable from a tilt of the ground and is left
    alone, while the pass-to-pass jumps a receiver actually produces are
    removed. Iterating re-estimates the medians on the corrected data.

    Points whose other-pass neighbours lie mostly on one side — the outer
    passes, a pass next to a skipped strip — are not used as evidence,
    because the slope does not cancel there and would be read as an offset.

    ``radius`` only has to reach the adjacent passes; the number of pairs
    grows with its square, so the caller keeps it at about one swath and a
    half whatever the grid cell.

    Returns ``(pass values, offset to subtract per pass)`` or None when no
    pass has a neighbour from another pass.
    """
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return None
    i, j = pairs[:, 0], pairs[:, 1]
    cross = pass_id[i] != pass_id[j]
    i, j = i[cross], j[cross]
    if i.size == 0:
        return None
    src = np.concatenate([i, j])
    nb = np.concatenate([j, i])
    n = z.size

    passes, pass_index = np.unique(pass_id, return_inverse=True)
    n_passes = passes.size
    counts = np.bincount(pass_index, minlength=n_passes).astype("float64")

    cnt = np.bincount(src, minlength=n).astype("float64")
    has = cnt > 0
    safe = np.where(has, cnt, 1.0)
    cx = np.bincount(src, weights=x[nb], minlength=n) / safe
    cy = np.bincount(src, weights=y[nb], minlength=n) / safe
    balanced = has & (np.hypot(cx - x, cy - y) < 0.3 * radius)
    if not balanced.any():
        return None

    # Share of each pass among the neighbours of each pass's balanced points.
    key = pass_index[src] * n_passes + pass_index[nb]
    weight = balanced[src] / cnt[src]
    share = np.bincount(key, weights=weight, minlength=n_passes * n_passes)
    share = share.reshape(n_passes, n_passes)
    n_balanced = np.bincount(pass_index[balanced], minlength=n_passes).astype("float64")
    rows = n_balanced > 0
    W = np.zeros((n_passes, n_passes))
    W[rows] = share[rows] / n_balanced[rows, None]
    A = (np.eye(n_passes) - W)[rows]

    corrected = z.copy()
    offsets = np.zeros(n_passes)
    lam = None
    for _ in range(n_iter):
        neighbour_mean = np.bincount(src, weights=corrected[nb], minlength=n) / safe
        q = np.where(balanced, corrected - neighbour_mean, np.nan)
        b = _group_median(pass_index, q, n_passes)[rows]
        if lam is None:
            lam = _ridge_lambda(A, b)
        step = np.linalg.solve(A.T @ A + lam * np.eye(n_passes), A.T @ b)
        # Offsets are relative; anchoring their weighted mean at zero keeps
        # the field's mean elevation where the receiver put it.
        step -= np.average(step, weights=counts)
        corrected -= step[pass_index]
        offsets += step
    return passes, offsets


def _apply_offsets(z: np.ndarray, pass_id: np.ndarray, passes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``z`` with each pass's offset subtracted; a pass the estimate did not
    see (possible when it ran on a sample) is left as it is."""
    idx = np.clip(np.searchsorted(passes, pass_id), 0, passes.size - 1)
    hit = passes[idx] == pass_id
    return z - np.where(hit, offsets[idx], 0.0)


def _auto_cell(swath: float) -> float:
    cell = float(np.clip(max(5.0, 0.5 * swath), 2.0, 25.0))
    return round(cell * 2.0) / 2.0


def _geometry(xmin, xmax, ymin, ymax, cell: float, margin: float):
    """Array origin and shape covering the points plus ``margin`` metres.

    The origin snaps to a multiple of the cell so two grids of the same
    field at the same cell size line up cell for cell.
    """
    x0 = math.floor((xmin - margin) / cell) * cell
    y0 = math.ceil((ymax + margin) / cell) * cell
    cols = int(math.ceil((xmax + margin - x0) / cell))
    rows = int(math.ceil((y0 - (ymin - margin)) / cell))
    return x0, y0, max(rows, 1), max(cols, 1)


def _axis_gradient(z: np.ndarray, axis: int) -> np.ndarray:
    """Rise per index step along ``axis``, on a grid that is NaN outside.

    Central difference where both neighbours exist; at the edge of the
    valid area — which is where it is needed — the one-sided difference
    over two cells, then over one; zero where the cell stands alone.
    """
    nan = np.full(z.shape, np.nan)

    def shifted(k: int) -> np.ndarray:
        """``z`` moved so that ``out[i] = z[i + k]`` along the axis."""
        out = nan.copy()
        src = [slice(None)] * z.ndim
        dst = [slice(None)] * z.ndim
        if k > 0:
            src[axis], dst[axis] = slice(k, None), slice(None, -k)
        else:
            src[axis], dst[axis] = slice(None, k), slice(-k, None)
        out[tuple(dst)] = z[tuple(src)]
        return out

    ahead, behind = shifted(1), shifted(-1)
    ahead2, behind2 = shifted(2), shifted(-2)
    candidates = (
        0.5 * (ahead - behind),
        0.5 * (z - behind2),
        0.5 * (ahead2 - z),
        z - behind,
        ahead - z,
    )
    g = np.zeros(z.shape)
    done = np.zeros(z.shape, dtype=bool)
    for cand in candidates:
        use = ~done & np.isfinite(cand)
        g[use] = cand[use]
        done |= use
    return g


def _extrapolate_band(
    z: np.ndarray, core: np.ndarray, band: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Values for the ``band`` cells just outside the triangulated ``core``.

    Those cells are under the implement but beyond the outermost track, so
    no triangle reaches them. Copying the nearest reading would put a flat
    step there — zero slope, a false rim around every field — so the local
    slope of the core is continued instead: exact on a plane, and honest
    about a hillside running off the edge. A band cell with no core within
    two cells (a single pass makes triangles too thin to hold any cell
    centre) takes its ``fallback`` value, the nearest reading.
    """
    r, c = np.nonzero(band)
    if not core.any():
        return fallback[r, c]
    gr = _axis_gradient(z, 0)
    gc = _axis_gradient(z, 1)
    _, (ri, ci) = ndimage.distance_transform_edt(~core, return_indices=True)
    r1, c1 = ri[r, c], ci[r, c]
    continued = z[r1, c1] + gr[r1, c1] * (r - r1) + gc[r1, c1] * (c - c1)
    near = np.maximum(np.abs(r - r1), np.abs(c - c1)) <= 2
    return np.where(near, continued, fallback[r, c])


def _block_note(mask: np.ndarray, cell: float, x0: float, y0: float) -> dict | None:
    """The fact behind the remark that the readings form separate blocks.

    One export often holds two fields worked in the same job. Gridded as
    one, the fall, relief and features of both are mixed and the cell is
    widened to span the gap between them, so the user is told rather than
    left to read a 'field' 35 km across.
    """
    labels, count = connected_components(mask)
    if count < 2:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    order = np.argsort(sizes)[::-1]
    area = sizes * cell ** 2 / 10_000.0
    if area[order[1]] < max(1.0, 0.05 * float(area.sum())):
        return None
    centres = ndimage.center_of_mass(mask, labels, [int(order[0]) + 1, int(order[1]) + 1])
    (r_a, c_a), (r_b, c_b) = centres
    apart = math.hypot((c_a - c_b) * cell, (r_a - r_b) * cell)
    blocks = int((area >= max(1.0, 0.05 * float(area.sum()))).sum())
    return notes_mod.fact(
        "blocks",
        blocks=blocks,
        largest_ha=float(area[order[0]]),
        second_ha=float(area[order[1]]),
        apart_m=float(apart),
    )


def _strip_width(x: np.ndarray, y: np.ndarray) -> float:
    """Width of the point cloud across its long axis, in metres.

    The spread along the smaller principal axis, scaled as a uniform strip
    (width = sqrt(12) times the standard deviation).
    """
    if x.size < 3:
        return 0.0
    cov = np.cov(np.vstack([x - x.mean(), y - y.mean()]))
    smallest = float(max(np.linalg.eigvalsh(cov)[0], 0.0))
    return math.sqrt(12.0 * smallest)


def grid_from_points(
    dataset,
    cell_m: float | None = None,
    max_gap_m: float | None = None,
    smooth_m: float | None = None,
    detrend_passes: bool = True,
    remove_outliers: bool = True,
    max_cells: int = 400_000,
    units: dict[str, Any] | None = None,
) -> tuple[ElevationGrid, dict[str, Any]]:
    """Grid the GPS elevation of a point dataset.

    Parameters
    ----------
    dataset:
        A :class:`~agrosuite.core.dataset.Dataset` carrying ``elev_m``.
    cell_m:
        Cell size; by default half the swath width, at least 5 m, so a cell
        holds readings from more than one pass and the pass offsets average
        out rather than print through.
    max_gap_m:
        How far from a reading a cell may be. By default cells reached by
        the triangulation of the points are kept within 1.5 swaths (or 3
        cells) of a reading, and no triangle may span more than twice that,
        so a skipped strip is bridged but a notch or a slough driven around
        stays empty; beyond the outermost track only the half swath the
        implement actually covered is kept, with the local slope continued
        into it. That is what keeps the grid to the field's true outline
        and area instead of its convex hull. An explicit value sets both
        distances at once.
    smooth_m:
        Gaussian smoothing scale. By default just enough to bring the
        altitude noise measured in the file down to about 3 cm on the
        surface — none at all for a noiseless file — because slope is a
        derivative and turns decimetre noise into nonsense, while every
        metre of smoothing shaves the depth of small hollows.
    detrend_passes:
        Remove the constant altitude offset between passes.
    remove_outliers:
        Drop altitude spikes before gridding.
    max_cells:
        Upper bound on ``rows * cols``; the cell grows until it fits.
    units:
        The unit set the notes are written in for *this* answer; ``None``,
        the default, is the metric store. It decides nothing else: every
        number in the report is metric, and ``report['note_facts']`` lets
        the notes be written again for a reader who works in another set,
        which is why nothing here depends on getting this right.

    Returns
    -------
    (ElevationGrid, report)
        The smoothed grid and a JSON-safe report of what was done and how
        trustworthy the result is. ``report['notes']`` carries the remarks
        the user should read alongside the numbers — an assumed swath, a
        widened cell, smoothing applied, readings forming a strip or
        several blocks — as finished sentences, and
        ``report['note_facts']`` carries the same remarks as the facts
        they state, for :func:`agrosuite.terrain.notes.render` to say
        again when the unit picker moves.
    """
    df = dataset.df
    non_numeric = 0
    non_numeric_sample = ""
    if sch.ELEVATION not in df.columns:
        elev = np.full(len(df), np.nan)
    else:
        column = df[sch.ELEVATION]
        # An export can carry 'n/a' or '---' in an altitude column, which
        # makes the column text; those readings are missing, not an error.
        numeric = pd.to_numeric(column, errors="coerce")
        elev = numeric.to_numpy(dtype="float64", na_value=np.nan)
        bad = numeric.isna() & column.notna()
        non_numeric = int(bad.sum())
        if non_numeric:
            non_numeric_sample = str(column[bad].iloc[0])[:20]
    n_elev = int(np.isfinite(elev).sum())
    missing = int(len(df) - n_elev)
    if n_elev == 0:
        raise ValueError(
            "This file carries no usable elevation, so its relief cannot be "
            "analysed. Open a yield or as-applied export that includes the GPS "
            "altitude, or a DEM GeoTIFF of the field."
        )
    if n_elev < _MIN_POINTS:
        raise ValueError(
            f"Only {n_elev} of the {len(df)} records carry an altitude; at least "
            f"{_MIN_POINTS} are needed to describe a relief. Open a yield or "
            "as-applied export that includes the GPS altitude on every point, "
            "or a DEM GeoTIFF of the field."
        )

    if sch.X not in df.columns or sch.Y not in df.columns or not dataset.metric_crs:
        if sch.LON not in df.columns or sch.LAT not in df.columns:
            raise ValueError(_coordinate_problem(df))
        dataset.project()
        df = dataset.df
        # Dataset.project leaves x/y out when no longitude is finite.
        if sch.X not in df.columns or sch.Y not in df.columns or not dataset.metric_crs:
            raise ValueError(_coordinate_problem(df))
    crs = str(dataset.metric_crs)
    x = df[sch.X].to_numpy(dtype="float64", na_value=np.nan)
    y = df[sch.Y].to_numpy(dtype="float64", na_value=np.nan)

    # The notes are collected as the facts they state and written at the
    # end, so the same list can be said again in another unit set without
    # reading the file again. A note whose numbers are counts and a sample
    # of the column's own text has nothing in it that moves with the
    # reader, so it is kept as written.
    facts: list[dict[str, Any]] = []
    if non_numeric:
        facts.append(notes_mod.plain(
            f"{non_numeric} of the {len(df)} records carry an altitude that is not a "
            f"number ({non_numeric_sample!r}); they were counted as missing."
        ))
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(elev)
    if int(finite.sum()) < _MIN_POINTS:
        raise ValueError(_coordinate_problem(df))
    points_total = int(len(df))
    x, y, z = x[finite], y[finite], elev[finite]
    pass_id = None
    if sch.PASS in df.columns:
        raw = df[sch.PASS].to_numpy()[finite]
        try:
            pass_id = np.asarray(raw, dtype="int64")
        except (TypeError, ValueError):
            pass_id = np.unique(raw.astype(str), return_inverse=True)[1]

    swath, swath_fact = _median_swath(df)
    if swath_fact:
        facts.append(swath_fact)

    # ---- readings that are no height at all: a lost fix, a fill value
    passes_before = int(np.unique(pass_id).size) if pass_id is not None else 0
    fill_removed = 0
    bad = _fill_value_mask(z)
    if bad.any():
        fill_removed = int(bad.sum())
        keep = ~bad
        x, y, z = x[keep], y[keep], z[keep]
        if pass_id is not None:
            pass_id = pass_id[keep]
        if z.size < _MIN_POINTS:
            raise ValueError(
                f"{fill_removed} of the {n_elev} altitudes are a fill value (0, -9999 or "
                "-32768) rather than a height, and too few real readings remain to "
                "describe the relief. The receiver had no fix; open another export, "
                "or a DEM GeoTIFF of the field."
            )
        facts.append(notes_mod.fact("fill_values", readings=fill_removed))
    stray_removed = 0
    if pass_id is not None:
        passes_ids, medians, stray = _stray_passes(z, pass_id)
        if stray.any():
            keep = ~np.isin(pass_id, passes_ids[stray])
            stray_removed = int((~keep).sum())
            x, y, z = x[keep], y[keep], z[keep]
            pass_id = pass_id[keep]
            if z.size < _MIN_POINTS:
                raise ValueError(
                    f"{int(stray.sum())} of the {passes_ids.size} passes sit at an altitude "
                    "far from the others, and too few readings remain to describe the "
                    "relief. The receiver had no steady fix; open another export, or a "
                    "DEM GeoTIFF of the field."
                )
            n_stray = int(stray.sum())
            facts.append(notes_mod.fact(
                "stray_passes",
                passes=n_stray,
                of=int(passes_ids.size),
                shown=[[int(p), float(m)]
                       for p, m in zip(passes_ids[stray][:5], medians[stray][:5])],
                more=max(n_stray - 5, 0),
                elsewhere_m=float(np.nanmedian(medians[~stray])),
                readings=stray_removed,
            ))
    passes_dropped = passes_before - (int(np.unique(pass_id).size) if pass_id is not None else 0)

    # ---- altitude spikes
    step = value_step(z)
    tree = cKDTree(np.column_stack([x, y]))
    outliers_removed = 0
    if remove_outliers and z.size > 13:
        bad = _remove_outliers(tree, z, step=step)
        outliers_removed = int(bad.sum())
        if outliers_removed:
            keep = ~bad
            x, y, z = x[keep], y[keep], z[keep]
            if pass_id is not None:
                pass_id = pass_id[keep]
            tree = cKDTree(np.column_stack([x, y]))
    points_used = int(z.size)
    if points_used < _MIN_POINTS:
        raise ValueError(
            f"Only {points_used} elevation readings survive the spike filter; the "
            "file's altitude is too erratic to describe the relief."
        )

    # ---- cell size, then the gap tolerances that depend on it
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    requested = cell_m
    auto_cell = _auto_cell(swath)
    cell = float(cell_m) if cell_m else auto_cell
    if not (cell > 0):
        raise ValueError("The cell size must be a positive number of metres.")
    explicit_gap = float(max_gap_m) if max_gap_m else None
    while True:
        gap = explicit_gap if explicit_gap else max(1.5 * swath, 3.0 * cell)
        edge = explicit_gap if explicit_gap else 0.5 * swath
        x0, y0, rows, cols = _geometry(xmin, xmax, ymin, ymax, cell, edge)
        if rows * cols <= max_cells:
            break
        cell = round((cell + 0.5) * 2.0) / 2.0
    if requested and cell != float(requested):
        facts.append(notes_mod.fact(
            "cell_capped", requested_m=float(requested), cell_m=cell, max_cells=max_cells))
    elif not requested and cell != auto_cell:
        facts.append(notes_mod.fact(
            "cell_widened",
            span_x_m=xmax - xmin,
            span_y_m=ymax - ymin,
            auto_cell_m=auto_cell,
            cell_m=cell,
            max_cells=max_cells,
        ))

    # ---- too sparse to be a survey of the ground
    spacing = float(np.median(tree.query(tree.data, k=2, workers=-1)[0][:, 1])) if points_used > 1 else 0.0
    if spacing > 2.0 * gap:
        raise ValueError(
            f"The {points_used} readings lie about {spacing:.0f} m apart, too far "
            "for the ground between them to be interpolated. A relief analysis "
            "needs the passes of a whole field, with a reading every few metres "
            "along the track."
        )

    # ---- pass-to-pass offsets
    passes = int(np.unique(pass_id).size) if pass_id is not None else 0
    pass_offset_sd = None
    detrended = False
    if detrend_passes and pass_id is not None and passes > _MAX_PASSES:
        facts.append(notes_mod.plain(
            f"The file has {passes:,} passes, too many to estimate an altitude "
            "offset for each; the pass offsets were left in.".replace(",", " ")
        ))
    elif detrend_passes and pass_id is not None and passes >= 2:
        # The radius only has to reach the adjacent passes (with a floor so a
        # narrow implement still gathers enough neighbours per point). It
        # must not grow with the cell: the pair list is quadratic in it, and
        # a coarse 'overview' cell would otherwise ask for every pair in the
        # file.
        radius = max(1.5 * swath, _MIN_PAIR_RADIUS_M)
        rng = np.random.default_rng(0)
        expected_pairs = _estimate_pairs(tree, radius, rng)
        if expected_pairs > _MAX_PAIRS:
            # Pairs go with the square of the point density, so a random
            # sample bounds the memory; the offsets are medians over whole
            # passes and lose nothing to it.
            keep = rng.random(points_used) < math.sqrt(_MAX_PAIRS / expected_pairs)
            sample_tree = cKDTree(np.column_stack([x[keep], y[keep]]))
            result = _detrend_passes(sample_tree, x[keep], y[keep], z[keep], pass_id[keep], radius)
        else:
            result = _detrend_passes(tree, x, y, z, pass_id, radius)
        if result is not None:
            pass_values, offsets = result
            z = _apply_offsets(z, pass_id, pass_values, offsets)
            pass_offset_sd = float(np.std(offsets))
            detrended = True

    # ---- gridding at the cell centres near a point
    xs = x0 + (np.arange(cols) + 0.5) * cell
    ys = y0 - (np.arange(rows) + 0.5) * cell
    cx, cy = np.meshgrid(xs, ys)
    centres = np.column_stack([cx.ravel(), cy.ravel()])
    dist, nearest = tree.query(centres, k=1, workers=-1)
    near = dist <= gap
    z_grid = np.full(rows * cols, np.nan)
    if near.any():
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay, QhullError

        try:
            tri = Delaunay(tree.data)
        except (QhullError, ValueError) as exc:
            raise ValueError(
                "The points lie along a line rather than over an area, so no "
                "surface can be fitted through them. A relief analysis needs "
                "the passes of a whole field."
            ) from exc
        # A triangle longer than the gap tolerance on both sides spans ground
        # nobody drove over — a notch in the outline, a slough — and its
        # interpolated value is a guess across the hole, not a measurement.
        corners = tri.points[tri.simplices]
        longest = np.max(
            [np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))],
            axis=0,
        )
        spans_hole = longest > 2.0 * gap
        simplex = tri.find_simplex(centres[near])
        covered = (simplex >= 0) & ~spans_hole[np.maximum(simplex, 0)]
        idx = np.nonzero(near)[0][covered]
        z_grid[idx] = LinearNDInterpolator(tri, z)(centres[idx])
    z_grid = z_grid.reshape(rows, cols)
    core = np.isfinite(z_grid)
    # The half swath beyond the outermost track is under the implement but
    # outside every triangle; the core's slope is continued into it.
    band = ~core & (dist.reshape(rows, cols) <= edge)
    if band.any():
        z_grid[band] = _extrapolate_band(z_grid, core, band, z[nearest].reshape(rows, cols))
    mask = np.isfinite(z_grid)

    # ---- smoothing, then back to the footprint
    noise = _point_noise(tree, z)
    if smooth_m is not None:
        smooth = float(smooth_m)
    else:
        # A cell interpolated inside a triangle averages three readings and
        # carries about 0.7 of a reading's noise; a Gaussian of s cells then
        # averages 4 pi s^2 cells' worth. Solve for the s that leaves the
        # target noise on the surface, and round to half a metre.
        target = max(_TARGET_SURFACE_NOISE_M, _TARGET_SLOPE_NOISE * 2.0 * cell / math.sqrt(2.0))
        sigma_cells = 0.7 * noise / (2.0 * math.sqrt(math.pi) * target)
        smooth = round(sigma_cells * cell * 2.0) / 2.0
        # A quantised altitude hides its noise (the residuals are mostly
        # exactly zero) and adds risers of its own, so the smoothing is at
        # least what takes a riser down to the slope of the ground.
        if step > 0:
            smooth = max(smooth, QUANTISED_SMOOTH_STEPS * step)
    z_smooth = gaussian_smooth(z_grid, smooth / cell) if smooth > 0 else z_grid.copy()
    z_smooth[~mask] = np.nan
    grid = ElevationGrid(z_smooth, x0, y0, cell, crs)
    # The noise left on the surface, by the same model that chose the
    # smoothing: 0.7 of a reading's noise per interpolated cell, averaged
    # over 4 pi s^2 cells by the Gaussian. It is what a slope read off the
    # surface has to beat, where ``vertical_noise`` below is what a single
    # reading scatters around that surface.
    surface_noise = 0.7 * noise
    if smooth > 0:
        surface_noise = min(surface_noise, 0.7 * noise / (2.0 * math.sqrt(math.pi) * smooth / cell))
    if step > 0:
        facts.append(notes_mod.fact("quantised", step_m=step, smooth_m=smooth))
    if smooth > 0 and not (step > 0 and smooth <= QUANTISED_SMOOTH_STEPS * step):
        facts.append(notes_mod.fact("smoothing", smooth_m=smooth, noise_m=noise))

    sampled = grid.sample(x, y)
    residual = z - sampled
    residual = residual[np.isfinite(residual)]
    vertical_noise = float(np.std(residual)) if residual.size > 1 else 0.0

    # ---- how much ground the answer rests on
    if spacing > cell:
        extent_ha = max((xmax - xmin) * (ymax - ymin) / 10_000.0, 1e-9)
        facts.append(notes_mod.fact(
            "spacing", spacing_m=spacing, cell_m=cell, per_ha=points_used / extent_ha))
    width = _strip_width(x, y)
    if width < 1.5 * swath:
        facts.append(notes_mod.fact("strip", width_m=width + swath))
    block_fact = _block_note(mask, cell, x0, y0)
    if block_fact:
        facts.append(block_fact)

    report: dict[str, Any] = {
        "points_total": points_total,
        "points_used": points_used,
        "outliers_removed": outliers_removed,
        "fill_values_removed": fill_removed,
        "passes": passes,
        "passes_dropped": passes_dropped,
        "value_step_m": step,
        "pass_offset_sd_m": pass_offset_sd,
        "detrended": detrended,
        "cell_m": cell,
        "rows": rows,
        "cols": cols,
        "area_ha": grid.area_ha(),
        "max_gap_m": gap,
        "edge_gap_m": float(edge),
        "smooth_m": smooth,
        "point_noise_m": noise,
        "vertical_noise_m": vertical_noise,
        "surface_noise_m": float(surface_noise),
        "missing_elevations": missing,
        "non_numeric_elevations": non_numeric,
        "spacing_m": spacing,
        "crs": crs,
        # The sentences, in the unit set this call was given, for whoever
        # is reading now; the facts beside them, so a later reader in
        # another unit set is not stuck with this one's.
        "notes": notes_mod.render(facts, units),
        "note_facts": facts,
    }
    return grid, report
