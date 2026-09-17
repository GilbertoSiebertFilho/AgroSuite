"""Elevation rasters: a DEM GeoTIFF in, terrain layers out.

A yield file carries one GPS altitude per logged point, along the track. A
DEM — from LiDAR, SRTM, a drone survey or the provincial elevation service —
carries a height for every cell of a grid, and when the user has one it is
the better source for a relief question. GeoTIFF is the format every DEM
comes in, and the one QGIS reads back, so this module speaks only that.

Reading is more than opening the file. DEMs arrive in geographic coordinates
(a cell that is 30 m tall and 19 m wide at 51 °N), in US survey feet, in
Web Mercator (whose "metres" are 1.6 ground metres at that latitude), or at a
resolution far finer than any relief question needs. Every terrain
derivative assumes square cells in ground metres, so :func:`read_dem`
normalizes all of that into one :class:`~agrosuite.terrain.grid.ElevationGrid`
and says what it changed. Writing is the way back: slope, landform classes
and the elevation grid itself go to QGIS, or into the folder next to the
yield map, as GeoTIFFs carrying the CRS and nodata so nothing has to be
guessed on the other side.

rasterio is imported lazily: the registry imports this module at start-up,
and the interface must not pay for GDAL until someone actually opens a DEM.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..core import crs as crs_mod
from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta
from . import brands as brands_mod

if TYPE_CHECKING:  # pragma: no cover - the grid is imported lazily at runtime
    from ..terrain.grid import ElevationGrid

#: Cells above which a DEM is coarsened before analysis. A 1 m LiDAR tile of a
#: quarter section is 6 million cells; the relief of a field reads the same at
#: 2 m and the derivatives run in seconds instead of minutes.
MAX_CELLS = 4_000_000

#: Metres per degree of latitude, the same constant the preflight uses.
_M_PER_DEG = 111_320.0

#: Projected CRSs whose "metre" is not a ground metre: Web Mercator stretches
#: distances by 1 / cos(latitude), which at 51 °N turns a 0.8 % slope into
#: a 0.5 % one. They are reprojected like a geographic CRS.
_PSEUDO_MERCATOR = {3857, 3785, 900913}

#: The nodata the app writes: the value most GIS tools already expect, and
#: one no elevation on Earth reaches.
DEFAULT_NODATA = -9999.0

#: Fill values a raster commonly carries WITHOUT declaring them: -9999 is the
#: GIS convention (and the app's own), -9998 a variant of it, -32768 the
#: int16 minimum that SRTM voids and QGIS conversions leave behind. A float
#: raster also arrives filled with the float32 extreme (±3.4028235e38) or
#: with 1e38; anything of that magnitude is caught by :data:`_HUGE_FILL`
#: rather than listed, because the extreme does not compare exactly once
#: it has been widened to float64.
_COMMON_FILLS = (-9999.0, -9998.0, -32768.0)
_HUGE_FILL = 1e38

#: An unlisted fill gives itself away the same way: one exact value, far
#: below every real height, repeated over a share of the cells that no
#: measured surface repeats. 1 000 m below the median is deeper than any
#: field's relief; half a percent of the cells is more than chance.
_FILL_BELOW_MEDIAN_M = 1000.0
_FILL_MIN_SHARE = 0.005

#: Distinct values below which a raster is a flat surface, not a quantised
#: one. A constant DEM (one value) lies on every lattice; declaring it
#: "whole metres" would force a 10 m smoothing and a 1 m minimum feature
#: height on a surface with no terraces to smooth. Two values with a riser
#: between them are already a terrace: the project's own whole-metre test
#: DEM, a prairie field with 10 m of relief, holds eleven distinct values
#: over 25 600 cells, so the threshold has to stay at "not constant".
MIN_STEP_DISTINCT = 2


# ==========================================================================
# Helpers
# ==========================================================================

def _open(path: Path):
    """Open a raster, turning GDAL's error into one that names the file."""
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        return rasterio.open(path)
    except RasterioIOError as exc:
        raise ValueError(
            f"'{path.name}' could not be opened as a raster: {exc}. "
            "The elevation layer has to be a GeoTIFF (.tif) — export it from QGIS "
            "with Raster > Conversion > Translate if it is in another format."
        ) from exc


def _crs_string(crs) -> str:
    """``"EPSG:32612"`` when the CRS has a code, its WKT otherwise."""
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()


def _corner_bounds(src) -> tuple[float, float, float, float]:
    """``(xmin, ymin, xmax, ymax)`` from the four corners of the array.

    rasterio's ``bounds`` reads only two corners, which is wrong for a
    rotated raster; the four-corner box is right for every transform.
    """
    t = src.transform
    corners = [t @ (0, 0), t @ (src.width, 0), t @ (0, src.height), t @ (src.width, src.height)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _centre_lonlat(src) -> tuple[float, float]:
    """WGS84 longitude / latitude of the raster's centre."""
    from rasterio.warp import transform as warp_transform

    xmin, ymin, xmax, ymax = _corner_bounds(src)
    lon, lat = warp_transform(src.crs, crs_mod.WGS84, [0.5 * (xmin + xmax)], [0.5 * (ymin + ymax)])
    return float(lon[0]), float(lat[0])


def _ground_metre_factor(crs, lat: float) -> float | None:
    """Metres of ground per unit of the CRS, or ``None`` when it is not a
    projected CRS whose unit is a fixed length (i.e. a geographic one)."""
    if not crs.is_projected:
        return None
    epsg = crs.to_epsg()
    if epsg in _PSEUDO_MERCATOR:
        return math.cos(math.radians(lat))
    try:
        _, factor = crs.linear_units_factor
    except Exception:  # rasterio raises its own CRSError for odd units
        return 1.0
    return float(factor)


def _native_cell_m(src, lat: float) -> tuple[float, float]:
    """The raster's cell size along x and y, in ground metres at the centre."""
    t = src.transform
    # Column and row vectors of the transform: their lengths are the cell
    # sizes even when the raster is rotated.
    cx = math.hypot(t.a, t.d)
    cy = math.hypot(t.b, t.e)
    factor = _ground_metre_factor(src.crs, lat)
    if factor is None:
        return cx * _M_PER_DEG * math.cos(math.radians(lat)), cy * _M_PER_DEG
    return cx * factor, cy * factor


def _in_ground_metres(crs) -> tuple[bool, str]:
    """Whether the CRS's unit is a ground metre, with the reason when not."""
    if not crs.is_projected:
        return False, "geographic coordinates in degrees"
    if crs.to_epsg() in _PSEUDO_MERCATOR:
        return False, "Web Mercator, whose metres are not ground metres"
    try:
        name, factor = crs.linear_units_factor
    except Exception:
        return True, ""
    if abs(factor - 1.0) > 1e-9:
        return False, f"units of {name}"
    return True, ""


def _north_up_square(t) -> bool:
    """True for the ordinary GeoTIFF layout: no rotation, square cells, row
    0 at the north edge — exactly the :class:`ElevationGrid` convention."""
    return (
        t.b == 0.0 and t.d == 0.0 and t.a > 0.0 and t.e < 0.0
        and math.isclose(t.a, -t.e, rel_tol=1e-9)
    )


def _nice_cell(cell_m: float) -> float:
    """Round a derived cell size to a figure a person would choose.

    A 1 arc-second DEM at 51 °N is 19.4 m by 30.9 m; a 24.5 m cell is the
    honest equivalent but "25 m" is what anyone would write down, and a
    grid built at 25 m lines up with one built at 5 m.
    """
    if cell_m < 2.0:
        step = 0.1
    elif cell_m < 10.0:
        step = 0.5
    elif cell_m < 50.0:
        step = 1.0
    else:
        step = 5.0
    return max(step, math.floor(cell_m / step + 0.5) * step)


def _cap_cell(cell_m: float, width_m: float, height_m: float) -> tuple[float, bool]:
    """Grow the cell by an integer factor until the grid fits ``MAX_CELLS``."""
    cells = math.ceil(width_m / cell_m + 1) * math.ceil(height_m / cell_m + 1)
    if cells <= MAX_CELLS:
        return cell_m, False
    factor = max(2, math.ceil(math.sqrt(cells / MAX_CELLS)))
    return cell_m * factor, True


def _thousands(value: float) -> str:
    """Thousands separator that leaves the sentence's own commas alone."""
    return f"{value:,.0f}".replace(",", "\u202f")


def _ordinal(n: int) -> str:
    """2nd, 3rd, 4th... for the subsampling note."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _value_step(values: np.ndarray, integer: bool = False) -> float | None:
    """The step the elevations are quantised to: 0 when they are not, and
    ``None`` when the surface is flat and the question does not arise.

    An SRTM tile, and any DEM QGIS converted from an integer raster, holds
    whole metres: a 0.8 % slope then reads as a staircase of 1 m risers
    125 m apart, and each riser is a 20 % slope over one 5 m cell. Knowing
    the step lets the analyser smooth the risers away and refuse to report
    features smaller than a step, so the caller measures it here on the
    raw values, before any resampling blurs the plateaus. The test itself
    lives with the grid (:func:`agrosuite.terrain.grid.value_step`) because
    a monitor that logs whole metres needs exactly the same one.

    One guard is the raster's own: fewer than :data:`MIN_STEP_DISTINCT`
    distinct values is a constant surface, which lies on every lattice and
    has no risers, so nothing is reported for it (``None``). ``integer``
    says the band's dtype already proves the step is 1 m.
    """
    from ..terrain.grid import value_step

    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    if np.unique(finite).size < MIN_STEP_DISTINCT:
        return None
    return 1.0 if integer else value_step(values)


def _undeclared_nodata(values: np.ndarray) -> tuple[float | None, float]:
    """The fill value a raster uses without declaring it, with the share of
    the finite cells it covers; ``(None, 0.0)`` when there is none.

    A DEM exported without its nodata tag keeps the fill in the pixels:
    -9999 outside the clip, -32768 in the voids, the float32 extreme where
    a tool wrote "no value". Read as heights, they make a 700 m field span
    10 700 m of relief, and everything downstream — the contour interval,
    the slope classes, the hills — is computed on that. The candidates are
    the listed fills, any value of magnitude :data:`_HUGE_FILL` or more,
    and any exact value more than :data:`_FILL_BELOW_MEDIAN_M` below the
    median that repeats over :data:`_FILL_MIN_SHARE` of the cells. When
    several are present the most frequent one is the fill; the warper can
    mask only one value, and a raster uses one.
    """
    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, 0.0
    counts: dict[float, int] = {}
    for fill in _COMMON_FILLS:
        n = int(np.count_nonzero(finite == fill))
        if n:
            counts[fill] = n
    huge = finite[np.abs(finite) >= _HUGE_FILL]
    if huge.size:
        for value, n in zip(*np.unique(huge, return_counts=True)):
            counts[float(value)] = int(n)
    low = finite[finite < float(np.median(finite)) - _FILL_BELOW_MEDIAN_M]
    if low.size:
        uniques, n = np.unique(low, return_counts=True)
        top = int(np.argmax(n))
        if n[top] > _FILL_MIN_SHARE * finite.size:
            counts.setdefault(float(uniques[top]), int(n[top]))
    if not counts:
        return None, 0.0
    fill = max(counts, key=counts.get)
    return fill, counts[fill] / finite.size


def _nodata_note(name: str, fill: float, share: float) -> str:
    return (
        f"'{name}' declares no nodata value, but {fill:g} fills {share * 100:.1f} % of "
        "its cells — the usual marker for 'no elevation' — so those cells were treated "
        "as nodata. Declare it in the file (QGIS: Layer Properties > Transparency, "
        "or gdal_edit -a_nodata) so every tool reads the raster the same way."
    )


def _looks_like_image(src) -> bool:
    """Three or more 8-bit bands is a photo (RGB), never an elevation model."""
    return src.count >= 3 and all(np.dtype(d) == np.uint8 for d in src.dtypes)


def _band_notes(src, name: str) -> list[str]:
    """Remarks for a raster that may not be a DEM at all; refusing is for
    the clear case (an RGB image), a note for the doubtful ones."""
    notes = []
    dtype = np.dtype(src.dtypes[0])
    if src.count > 1:
        notes.append(
            f"'{name}' has {src.count} bands; only the first was read as elevation. "
            "A DEM has one band — if this file is something else, open the elevation "
            "layer instead."
        )
    if dtype == np.uint8:
        notes.append(
            f"'{name}' stores 8-bit values (0–255), the range of an image, not of "
            "heights above sea level. If it is a photo or a classified map rather "
            "than an elevation model, the relief read from it means nothing."
        )
    return notes


def _nodata_for_info(nodata) -> float | None:
    """A JSON-safe nodata: NaN cannot go through the API, and a NaN nodata
    says nothing a NaN cell does not already say."""
    if nodata is None or not math.isfinite(nodata):
        return None
    return float(nodata)


# ==========================================================================
# Reading
# ==========================================================================

def describe(path: str | Path) -> dict[str, Any]:
    """What a raster is, from its header alone — no pixels are read.

    Used by the file inspector so the interface can say "1200 × 900 cells
    of 5 m, EPSG:32612" before the user commits to importing it.
    """
    path = Path(path)
    with _open(path) as src:
        cells = src.width * src.height
        info: dict[str, Any] = {
            "rows": int(src.height),
            "cols": int(src.width),
            "cells": int(cells),
            "bands": int(src.count),
            "dtype": str(src.dtypes[0]) if src.count else None,
            "nodata": _nodata_for_info(src.nodata),
            "crs": None,
            "cell_m": None,
            "cell_xy_m": None,
            "needs_reprojection": False,
            "too_large": cells > MAX_CELLS,
            "looks_like_image": _looks_like_image(src),
        }
        if src.crs is None:
            info["summary"] = (
                f"{src.width} × {src.height} cells with no coordinate reference system"
            )
            return info

        lon, lat = _centre_lonlat(src)
        cx, cy = _native_cell_m(src, lat)
        metric, reason = _in_ground_metres(src.crs)
        info.update({
            "crs": _crs_string(src.crs),
            "cell_m": float(math.sqrt(cx * cy)),
            "cell_xy_m": [float(cx), float(cy)],
            "needs_reprojection": not metric,
            "centre_lonlat": [lon, lat],
        })
        cell_text = f"{cx:.4g} m" if math.isclose(cx, cy, rel_tol=0.02) else f"{cx:.3g} × {cy:.3g} m"
        info["summary"] = (
            f"{src.width} × {src.height} cells of about {cell_text} ({info['crs']}"
            + (f", {reason}" if reason else "")
            + f"), {info['dtype']}"
            + (f", nodata {src.nodata:g}" if info["nodata"] is not None else "")
            + (f" — {src.count} bands of 8-bit values: an image, not an elevation model"
               if info["looks_like_image"] else "")
        )
        return info


def read_dem(
    path: str | Path,
    target_cell_m: float | None = None,
    units: dict[str, Any] | None = None,
) -> tuple["ElevationGrid", dict[str, Any]]:
    """Read band 1 of a GeoTIFF as an :class:`ElevationGrid` in ground metres.

    The grid is the raster itself when the file is already what the terrain
    modules expect — a projected CRS in metres, square cells, north up, no
    more than ``MAX_CELLS`` cells — so a GeoTIFF written by
    :func:`write_geotiff` reads back cell for cell. Otherwise the raster is
    resampled with ``rasterio.warp.reproject`` onto a grid that is: in the
    UTM zone of the raster's centre when the source is geographic, in feet
    or in Web Mercator; at ``target_cell_m`` when given, else at the native
    resolution converted to metres and rounded to a sensible figure; and
    coarsened by an integer factor when it would exceed ``MAX_CELLS``.
    Resampling is bilinear at similar resolution and a block average when
    the cell grows, so coarsening does not alias a rough surface.

    Parameters
    ----------
    path:
        A GeoTIFF (any GDAL raster opens, but GeoTIFF is what the app writes
        and what QGIS exchanges).
    target_cell_m:
        Cell size to resample to. ``None`` keeps the native resolution.
    units:
        The unit set ``info['notes']`` is written in for this answer;
        ``None`` is metric, and every number in ``info`` is metric
        whatever is passed. It decides nothing that lasts:
        ``info['note_facts']`` holds the same remarks as the facts they
        state, so a reader who moves the unit picker gets them in the set
        on screen rather than the one the file was opened in.

    Returns
    -------
    (grid, info)
        ``info`` is JSON-safe and records what was done: ``path``,
        ``crs_in``, ``crs``, ``cell_in_m`` (geometric mean of the native
        x/y cell in ground metres), ``cell_in_xy_m``, ``cell_m``,
        ``rows_in``, ``cols_in``, ``rows``, ``cols``, ``dtype_in``,
        ``nodata`` (``None`` when unset or NaN), ``nodata_detected`` (the
        fill value treated as nodata when the file declares none — see
        :func:`_undeclared_nodata` — else ``None``), ``reprojected``,
        ``resampled``, ``coarsened``, ``value_step_m`` (1.0 for whole-metre
        elevations — every integer raster, and SRTM converted to float —
        0.5 for half metres, 0.0 when the values are not quantised,
        ``None`` for a constant surface), ``notes`` (remarks for the
        user, as sentences) and ``note_facts`` (the same remarks as facts,
        for :func:`agrosuite.terrain.notes.render`).

    Raises
    ------
    ValueError
        When the file is not a raster, is an image (three or more 8-bit
        bands), has no CRS, holds no band or no valid cell, or
        ``target_cell_m`` is not positive.
    """
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds

    # Imported here, like the grid: the terrain package reads rasters
    # through this module, so importing it at the top would close the ring.
    from ..terrain.grid import ElevationGrid
    from ..terrain.notes import fact, plain, render

    path = Path(path)
    if target_cell_m is not None and not (float(target_cell_m) > 0):
        raise ValueError("The target cell size must be a positive number of metres.")

    with _open(path) as src:
        if src.count < 1:
            raise ValueError(f"'{path.name}' holds no raster band.")
        if _looks_like_image(src):
            raise ValueError(
                f"'{path.name}' is an image ({src.count} bands of 8-bit values), not an "
                "elevation model: its pixels are colours, and the relief read from them "
                "would be nonsense. Open the DEM of the field instead — a single-band "
                "GeoTIFF of heights in metres."
            )
        if src.crs is None:
            raise ValueError(
                f"'{path.name}' carries no coordinate reference system, so its cells "
                "cannot be placed on the ground. Open it in QGIS, assign the CRS it "
                "was made in (Raster > Projections > Assign Projection), save it as a "
                "new GeoTIFF and open that one."
            )

        t = src.transform
        nodata = src.nodata
        is_float = np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
        crs_in = _crs_string(src.crs)
        lon_c, lat_c = _centre_lonlat(src)
        cell_x, cell_y = _native_cell_m(src, lat_c)
        cell_in = math.sqrt(cell_x * cell_y)
        metric, reason = _in_ground_metres(src.crs)
        native_cells = src.width * src.height
        # Every remark is collected as the fact it states and written at
        # the end (see :mod:`agrosuite.terrain.notes`), so a reader who
        # moves the unit picker gets them in the set on screen instead of
        # the one this raster happened to be opened in. The ones whose
        # numbers are counts, shares or a CRS name are kept as written:
        # nothing in them moves with the reader.
        facts: list[dict[str, Any]] = [plain(n) for n in _band_notes(src, path.name)]

        info: dict[str, Any] = {
            "path": str(path),
            "crs_in": crs_in,
            "cell_in_m": float(cell_in),
            "cell_in_xy_m": [float(cell_x), float(cell_y)],
            "rows_in": int(src.height),
            "cols_in": int(src.width),
            "dtype_in": str(src.dtypes[0]),
            "nodata": _nodata_for_info(nodata),
            "nodata_detected": None,
            "reprojected": False,
            "resampled": False,
            "coarsened": False,
        }
        if nodata is not None and not math.isfinite(nodata):
            facts.append(plain("NaN marks the cells without elevation in this raster."))

        keep_native = (
            metric
            and _north_up_square(t)
            and native_cells <= MAX_CELLS
            and (target_cell_m is None or math.isclose(float(target_cell_m), t.a, rel_tol=1e-9))
        )
        if keep_native:
            z = src.read(1).astype("float64")
            if nodata is not None and math.isfinite(nodata):
                z[z == nodata] = np.nan
            z[~np.isfinite(z)] = np.nan
            if nodata is None:
                fill, share = _undeclared_nodata(z)
                if fill is not None:
                    z[z == fill] = np.nan
                    info["nodata_detected"] = fill
                    facts.append(plain(_nodata_note(path.name, fill, share)))
                    nodata = fill
            _require_elevation(z, path.name, nodata)
            grid = ElevationGrid(z, t.c, t.f, t.a, crs_in)
            info.update({
                "crs": crs_in, "cell_m": grid.cell, "rows": grid.rows, "cols": grid.cols,
                "value_step_m": _value_step(z, integer=not is_float),
                "notes": render(facts, units),
                "note_facts": facts,
            })
            return grid, info

        # The step and an undeclared fill both have to be read off the raw
        # values: the warper interpolates between plateaus and hides the
        # step, and it smears a fill it was not told about into the cells
        # next to it. A decimated read (nearest, no averaging) keeps the
        # original values without loading a huge tile.
        sample_shape = (min(src.height, 512), min(src.width, 512))
        sample = src.read(1, out_shape=sample_shape, resampling=Resampling.nearest).astype("float64")
        if nodata is not None and math.isfinite(nodata):
            sample[sample == nodata] = np.nan
        if nodata is None:
            fill, share = _undeclared_nodata(sample)
            if fill is not None:
                sample[sample == fill] = np.nan
                info["nodata_detected"] = fill
                facts.append(plain(_nodata_note(path.name, fill, share)))
                nodata = fill
        value_step = _value_step(sample, integer=not is_float)

        # ---- anything else goes through the warper onto a grid of our own
        if metric:
            dst_crs = crs_in
            bounds = _corner_bounds(src)
        else:
            dst_crs = crs_mod.utm_epsg(lon_c, lat_c)
            bounds = transform_bounds(src.crs, dst_crs, *_corner_bounds(src), densify_pts=21)
            info["reprojected"] = True
            facts.append(plain(
                f"Reprojected from {crs_in} ({reason}) to {dst_crs}, so that the cells "
                "are square and in metres and slopes come out right."
            ))
        west, south, east, north = bounds

        if target_cell_m is not None:
            cell = float(target_cell_m)
        elif metric and _north_up_square(t):
            cell = float(t.a)
        else:
            cell = _nice_cell(cell_in)
            if not _north_up_square(t) and metric:
                facts.append(fact("raster_square_cells", cell_m=cell))

        cell, coarsened = _cap_cell(cell, east - west, north - south)
        if coarsened:
            info["coarsened"] = True
            out_cells = math.ceil((east - west) / cell) * math.ceil((north - south) / cell)
            facts.append(fact("raster_coarsened", native_cells=native_cells,
                              cell_m=cell, out_cells=out_cells))
        elif target_cell_m is not None:
            facts.append(fact("raster_resampled", cell_in_m=cell_in, cell_m=cell))

        # Snap the origin to a multiple of the cell, as grid_from_points does,
        # so two rasters of the same field at the same cell line up exactly.
        x0 = math.floor(west / cell) * cell
        y0 = math.ceil(north / cell) * cell
        cols = max(1, math.ceil((east - x0) / cell))
        rows = max(1, math.ceil((y0 - south) / cell))

        # Bilinear keeps a plane a plane; once the cell grows past the native
        # one, a block average is what keeps a rough surface from aliasing.
        resampling = Resampling.average if cell >= 1.5 * cell_in else Resampling.bilinear
        if nodata is None and is_float:
            # An undeclared nodata in a float raster is NaN by convention, and
            # saying so keeps the warper from smearing it into its neighbours.
            # (A fill found above is already in ``nodata`` and is masked the
            # same way, before the interpolation can reach it.)
            src_nodata = np.nan
        else:
            src_nodata = nodata

        z = np.full((rows, cols), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1),
            destination=z,
            src_transform=t,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=Affine(cell, 0.0, x0, 0.0, -cell, y0),
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        z[~np.isfinite(z)] = np.nan
        if nodata is not None and math.isfinite(nodata):
            z[z == nodata] = np.nan
        _require_elevation(z, path.name, nodata)

        grid = ElevationGrid(z, x0, y0, cell, dst_crs)
        info.update({
            "crs": dst_crs, "cell_m": grid.cell, "rows": grid.rows, "cols": grid.cols,
            "resampled": True, "value_step_m": value_step,
            "notes": render(facts, units), "note_facts": facts,
        })
        return grid, info


def _require_elevation(z: np.ndarray, name: str, nodata) -> None:
    """A raster with no valid cell is refused here, where the reason is
    known; downstream it would import as an empty layer and the analyser
    would suggest a finer cell, which cannot conjure values that are not
    there."""
    if np.isfinite(z).any():
        return
    marker = f" ({nodata:g})" if nodata is not None and math.isfinite(nodata) else ""
    raise ValueError(
        f"'{name}' holds no elevation: every cell is nodata{marker}. Check that the "
        "raster was clipped to an area that actually covers the field, and that its "
        "nodata value is the one the file declares (QGIS: Layer Properties > "
        "Transparency)."
    )


# ==========================================================================
# Writing
# ==========================================================================

def _profile(grid: "ElevationGrid", dtype: str, nodata, predictor: int) -> dict[str, Any]:
    """rasterio creation profile shared by the two writers: LZW-compressed,
    tiled, so QGIS opens a large layer without reading all of it."""
    return {
        "driver": "GTiff",
        "height": grid.rows,
        "width": grid.cols,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform(),
        "nodata": nodata,
        "compress": "lzw",
        "predictor": predictor,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def _write(path: Path, profile: dict[str, Any], data: np.ndarray, tags: dict | None) -> Path:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if tags:
            # GeoTIFF tags are strings; numbers and booleans are written as
            # their text so they read back the way they were printed.
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    return path


def write_geotiff(
    grid: "ElevationGrid",
    path: str | Path,
    values: np.ndarray | None = None,
    nodata: float | None = DEFAULT_NODATA,
    dtype: str = "float32",
    tags: dict[str, Any] | None = None,
) -> Path:
    """Write a continuous layer (elevation, slope, wetness...) as a GeoTIFF.

    ``values`` defaults to the grid's own elevation; any array shaped like
    the grid can be written on its geometry. NaN cells are written as
    ``nodata`` (``None`` writes NaN itself, which only a float ``dtype``
    can hold). ``float32`` keeps elevation to the millimetre and halves the
    file; pass ``"float64"`` for a lossless round trip.
    """
    data = grid.z if values is None else np.asarray(values, dtype="float64")
    if data.shape != grid.shape:
        raise ValueError(
            f"Values shaped {data.shape} do not fit a grid of {grid.shape}; "
            "a layer is written on the geometry of the grid it was computed on."
        )
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError(
            f"write_geotiff writes floating-point layers; for a categorical layer "
            f"in {dtype} use write_geotiff_int."
        )
    out = data.astype(dtype)
    if nodata is not None:
        nodata = float(nodata)
        out[~np.isfinite(data)] = nodata
    return _write(Path(path), _profile(grid, dtype, nodata, predictor=3), out, tags)


def write_geotiff_int(
    grid: "ElevationGrid",
    path: str | Path,
    values: np.ndarray,
    nodata: int = 255,
    dtype: str = "uint8",
    tags: dict[str, Any] | None = None,
    labels: dict[int, str] | None = None,
) -> Path:
    """Write a categorical layer (landform class, zone id) as a GeoTIFF.

    Cells that are NaN, or not finite, become ``nodata``. ``labels`` maps a
    class value to its name and is written as ``class_<value>`` tags, which
    is how the names survive the trip to QGIS and back.
    """
    data = np.asarray(values, dtype="float64")
    if data.shape != grid.shape:
        raise ValueError(
            f"Values shaped {data.shape} do not fit a grid of {grid.shape}; "
            "a layer is written on the geometry of the grid it was computed on."
        )
    kind = np.dtype(dtype)
    if not np.issubdtype(kind, np.integer):
        raise ValueError(f"write_geotiff_int needs an integer dtype, not {dtype}.")
    lo, hi = np.iinfo(kind).min, np.iinfo(kind).max
    if not (lo <= nodata <= hi):
        raise ValueError(f"nodata {nodata} does not fit in {dtype} ({lo}..{hi}).")

    valid = np.isfinite(data)
    if valid.any():
        vmin, vmax = data[valid].min(), data[valid].max()
        if vmin < lo or vmax > hi:
            raise ValueError(
                f"Class values run from {vmin:g} to {vmax:g}, outside {dtype} "
                f"({lo}..{hi}); pick a wider dtype."
            )
        if (data[valid] == nodata).any():
            raise ValueError(
                f"A class uses the value {nodata}, which is the nodata marker; "
                "pick a different nodata."
            )
    out = np.full(grid.shape, nodata, dtype=kind)
    out[valid] = np.rint(data[valid]).astype(kind)

    all_tags = dict(tags or {})
    for value, label in (labels or {}).items():
        all_tags[f"class_{int(value)}"] = label
    return _write(Path(path), _profile(grid, dtype, int(nodata), predictor=2), out, all_tags)


# ==========================================================================
# Into the app's data model
# ==========================================================================

def dataset_from_dem(
    grid: "ElevationGrid",
    info: dict[str, Any] | None = None,
    max_points: int = 150_000,
    meta_name: str | None = None,
) -> Dataset:
    """A point :class:`Dataset` of the DEM's cell centres.

    The rest of the app — the map, the preflight, the project panel —
    understands points, not rasters, and this is the bridge: one row per
    valid cell with ``lon``/``lat``, ``elev_m`` and ``value`` set to the
    elevation so the map colours it the moment it opens. A large DEM is
    subsampled on a regular stride so the browser never receives more than
    ``max_points`` rows; the terrain analyser does not use these points, it
    reopens the raster named in ``meta.extra["dem_path"]`` at full resolution.

    ``x``/``y`` are projected into the grid's own CRS, so they coincide with
    the raster's coordinates rather than sitting a datum shift away.
    """
    info = dict(info or {})
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")

    mask = grid.mask
    total_valid = int(mask.sum())
    stride = max(1, math.ceil(math.sqrt(total_valid / max_points))) if total_valid else 1
    while True:
        sub = np.zeros(grid.shape, dtype=bool)
        sub[::stride, ::stride] = True
        keep = mask & sub
        if int(keep.sum()) <= max_points:
            break
        stride += 1

    X, Y = grid.xy()
    x = X[keep]
    y = Y[keep]
    lon, lat = grid.to_lonlat(x, y)
    elev = grid.z[keep]
    df = pd.DataFrame({
        sch.LON: lon,
        sch.LAT: lat,
        sch.ELEVATION: elev,
        sch.VALUE: elev.copy(),
    })

    dem_path = str(info.get("path") or "")
    brand = brands_mod.get_brand("generic")

    # The remarks travel as the facts they state, not as finished sentences:
    # a layer opened while the screen was in acres is still on screen when
    # the picker moves to hectares, and its notes have to move with it. The
    # sentences below are the metric rendering, which is what a caller
    # outside the app (a test, the CLI) gets; the session writes them again
    # in the unit set on screen every time it hands the layer over.
    from agrosuite.terrain import notes as notes_mod

    note_facts = list(info.get("note_facts") or [])
    if not note_facts and info.get("notes"):
        # A raster read by an older path gave sentences and no facts; keep
        # them rather than dropping what the reader found.
        note_facts = [notes_mod.plain(str(n)) for n in info["notes"]]
    note_facts.append(notes_mod.fact(
        "dem_points", stride=int(stride), shown=len(df), cells=total_valid,
    ))
    notes = notes_mod.render(note_facts)

    meta = DatasetMeta(
        name=meta_name or (Path(dem_path).stem if dem_path else "elevation"),
        source_path=dem_path,
        source_format="geotiff",
        brand=brand.key,
        brand_label=brand.label,
        operation="elevation",
        value_label="Elevation",
        value_unit="m",
        source_value_unit="m",
        geometry_type="point",
        notes=notes,
        extra={
            "dem_path": dem_path,
            "dem_cell_m": float(grid.cell),
            "dem_full_cells": total_valid,
            "dem_rows": grid.rows,
            "dem_cols": grid.cols,
            "dem_crs": grid.crs,
            "dem_area_ha": float(grid.area_ha()),
            "dem_stride": int(stride),
            # The facts behind ``meta.notes``, so the layer's own remarks
            # can be written again in whatever units the reader is in.
            "note_facts": note_facts,
            "dem_info": {k: v for k, v in info.items()
                         if k not in ("notes", "note_facts")},
        },
    )
    dataset = Dataset(df, meta)
    if len(df):
        dataset.project(grid.crs)
    return dataset


def read_raster(path: str | Path, brand_hint: str | None = None) -> Dataset:
    """The registry's reader: a GeoTIFF straight into a :class:`Dataset`.

    ``brand_hint`` is accepted for the reader signature's sake; a DEM has no
    monitor behind it.
    """
    grid, info = read_dem(path)
    return dataset_from_dem(grid, info)
