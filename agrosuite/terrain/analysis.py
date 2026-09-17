"""The terrain analysis a user actually sees: one call, one result, one report.

:func:`analyze` takes a dataset — a yield or as-applied file with GPS
altitude, or a DEM read through :mod:`agrosuite.formats.raster` — and runs
the whole chain in order: grid, derivatives, hydrology, landform classes,
discrete hills and lows, contours. :class:`TerrainResult` holds the layers
for the map and turns them into the JSON summary the interface shows, the
plain-English :func:`findings` a farmer reads, GeoJSON outlines of the
features, and the zone datasets the rest of the app already knows how to
export to a monitor and join with yield.

Two decisions shape the numbers. The topographic position indices are
computed on the elevation minus the field's fitted plane: in the interior a
plane contributes nothing to TPI, but at the field edge the neighbourhood
disc is cut off on the outside, and on a tilted field the uphill edge would
otherwise read as a ridge and the downhill edge as a valley. And a hill (or
low) is only reported when it stands at least ``min_feature_height_m`` above
(below) the foot of the ground around it, a threshold that defaults to twice
the measured GPS noise — a bump smaller than the noise is the noise. The
same threshold floors the landform classes, the closed depressions and
the wetness index: a cell whose position index is within a fraction of it
reads as level ground; a basin shallower than it is counted but not
listed, because on an unsmoothed altitude the noise alone digs dozens of
"potholes" holding real-looking volumes; and a field whose whole residual
relief is under it gets no "likely wet" ground beyond what actually
ponds, because standardized indices always have tails and a dead-flat
field must not come back a tenth hilltop and a tenth wet. The wetness
index is ranked only over cells whose slope the surface can vouch for
(over twice its noise per cell), because where the slope is floored the
index is a constant, not a ranking, and a flat plain next to one hill
came back a tenth "likely wet". And every share of the field — slope
classes, aspect sectors, landform classes, wet ground — is taken inside
the field's outermost ring of cells, whose derivatives lean on copied
values (see :mod:`derivatives`); the layers keep the ring, the shares do
not, or a uniform 3 % plane reports 2 % of itself flat.
A field with no relief the data can vouch for at all — neither beyond its
general fall nor in it — is *level*: it gets no contour lines, drainage
lines, wet cells or features, and one finding says so, because every one
of those would otherwise be drawn from the rounding noise of the gridding
(a constant altitude column comes back as 953 contour loops around
differences of 1e-13 m).

Everything stored here is metric, the sentences included: :func:`summary`
writes them in metres, hectares and cubic metres, and the structured
numbers beside them are the same store an interface converts at display
time. A sentence cannot be converted at display time — its numbers are
written into it, and restating it in another unit means deciding again
what it says — so the reader's unit set travels into the writing of it
instead, through :class:`~agrosuite.core.units.Phrase`. :func:`findings`
takes that unit set, and :func:`restate` writes a stored summary's
sentences again for a reader who works in another one, from the numbers
the summary already carries and without touching a single one of them.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from ..core import preflight as preflight_mod
from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta
from ..core.units import Phrase
from . import contours as contours_mod
from . import derivatives as deriv
from . import hydrology as hydro_mod
from . import landforms as lf
from . import notes as notes_mod
from .grid import QUANTISED_SMOOTH_STEPS as _QUANTISED_SMOOTH_STEPS
from .grid import ElevationGrid, connected_components, gaussian_smooth, grid_from_points

#: Ground steeper than this is quoted as "steep" in the findings: the point
#: where erosion on bare soil and machine stability both become a concern.
STEEP_PCT = 10.0

#: A ponded volume above this earns a warning rather than a note.
WARN_VOLUME_M3 = 500.0

#: Pass-to-pass altitude offsets above this spread mean the receiver was
#: not RTK-corrected, so the absolute heights are approximate.
RTK_OFFSET_SD_M = 0.3

#: Aspect ratio of a low's footprint above which it is called a valley
#: rather than a hollow.
VALLEY_ELONGATION = 2.5

#: A cell whose TPI (metres) is within this fraction of the minimum feature
#: height reads as level ground in the landform classes, whatever its
#: z-score: the standardization divides by the field's own spread, which on
#: a flat field is the noise. A tenth, not a half, because the small-scale
#: index of a real hill is only a fraction of the hill (it measures the
#: summit's curvature over 30 m, not its height): a 4 m hill's summit stands
#: 0.15 m above its 30 m disc, a flat field's cells a centimetre or two.
LEVEL_TPI_FRACTION = 0.1

#: A plane fall shorter than this is not quoted: the plane fit of a level
#: field carries a direction that is an artefact of the arithmetic.
MIN_TREND_DROP_M = 0.1

#: The least relief the analysis will draw anything from: a field whose
#: residual relief (5th to 95th percentile of the detrended surface) AND
#: plane fall are both under ``max(LEVEL_RELIEF_M, 2 x vertical noise, the
#: value step)`` is level, and its contours, drainage lines, wet cells and
#: features are all left empty. 5 cm is the ponding depth threshold, the
#: vertical precision a smoothed GPS altitude can claim; twice the measured
#: noise covers a noisy file, and the value step a raster in whole metres.
#: The plane fall is part of the test because a tilted field with no
#: relief beyond its tilt still has real, parallel contours.
LEVEL_RELIEF_M = 0.05

#: A grid narrower than this many valid rows or columns cannot be
#: analysed: every derivative reads a 3x3 window, and marching squares
#: needs two rows and two columns to place a line at all.
MIN_GRID_SPAN = 3

#: The smallest slope the wetness index may rank a cell on, in percent:
#: twice the slope noise of the surface (see :func:`_noise_slope_pct`),
#: and never under this. A DEM reports no noise; even there a slope under
#: a tenth of a percent is a rounding artefact, and the index floors
#: tan(beta) at that value anyway.
MIN_NOISE_SLOPE_PCT = 0.1

#: Below this share of cells with a usable slope, the wetness ranking is
#: not made at all: the 90th percentile of a sliver of the field says
#: nothing about the rest, and only the listed depressions are marked.
MIN_WET_CANDIDATE_SHARE = 0.02

#: The smallest closed depression listed, in hectares: the same floor as
#: the hills and lows (:data:`landforms.MIN_FEATURE_AREA_HA`).
MIN_DEPRESSION_AREA_HA = lf.MIN_FEATURE_AREA_HA

#: 8-connectivity, as in :mod:`grid`: the interior is the mask eroded by
#: one cell in every direction, diagonals included.
_EIGHT = np.ones((3, 3), dtype=bool)

#: Default smoothing scale for a quantised elevation, in multiples of its
#: value step; defined with the grid, which applies it to points too.
QUANTISED_SMOOTH_STEPS = _QUANTISED_SMOOTH_STEPS

#: The map layers in the order the interface lists them: key, label, unit,
#: palette. The landform layer is categorical; every other one continuous.
LAYER_SPECS: list[tuple[str, str, str, str]] = [
    ("elevation", "Elevation", "m", "elevation"),
    ("slope_pct", "Slope", "%", "slope"),
    ("aspect_deg", "Aspect", "deg", "aspect"),
    ("hillshade", "Hillshade", "", "hillshade"),
    ("tpi_small", "Position (small scale)", "m", "diverging"),
    ("tpi_large", "Position (large scale)", "m", "diverging"),
    ("twi", "Wetness index", "index", "wetness"),
    ("landform", "Landform", "", "landform"),
    ("curvature_profile", "Profile curvature", "index", "diverging"),
    ("curvature_plan", "Plan curvature", "index", "diverging"),
    ("depression_depth", "Ponding depth", "m", "wetness"),
    ("flow_acc", "Flow accumulation", "", "accumulation"),
]

ZONE_KINDS = ("landform", "slope_class", "elevation_bands", "wetness")

#: The narrowest elevation band worth a zone: the precision its label shows.
BAND_MIN_M = 0.1


# ==========================================================================
# Options
# ==========================================================================

@dataclass
class TerrainOptions:
    """What the user may change; ``None`` means "choose from the data".

    ``cell_m``, ``smooth_m``, ``detrend_passes`` and ``remove_outliers``
    reach :func:`grid_from_points` (``cell_m`` also picks the DEM cell);
    the TPI radii and the feature height are resolved by :func:`analyze`
    and the resolved values are echoed in ``summary()['options']``.
    """

    cell_m: float | None = None
    smooth_m: float | None = None
    tpi_small_m: float | None = None
    tpi_large_m: float | None = None
    contour_interval_m: float | None = None
    detrend_passes: bool = True
    remove_outliers: bool = True
    min_feature_height_m: float | None = None
    max_cells: int = 400_000
    hillshade_azimuth: float = 315.0
    hillshade_altitude: float = 45.0
    min_upstream_ha: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TerrainOptions":
        """Build from a JSON body; unknown keys are an error so a typo in the
        interface does not silently fall back to the default."""
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown terrain option(s) {', '.join(unknown)}; the options are "
                f"{', '.join(sorted(known))}."
            )
        return cls(**data)


#: What a unit is called in a sentence about typing one in. The picker's
#: own labels ("foot (ft)") are for a dropdown, not for the middle of a
#: refusal, and "a positive number of ft" is not English.
_LENGTH_WORDS = {"m": "metres", "ft": "feet", "cm": "centimetres",
                 "in": "inches", "yd": "yards"}
_AREA_WORDS = {"ha": "hectares", "ac": "acres", "alq": "alqueires",
               "m2": "square metres", "km2": "square kilometres"}


def _length_word(say: Phrase) -> str:
    return _LENGTH_WORDS.get(say.length_unit, say.length_unit)


def _area_word(say: Phrase) -> str:
    return _AREA_WORDS.get(say.area_unit, say.area_unit)


def _compact(value_m: float, say: Phrase) -> str:
    """A length in the reader's unit, kept as short as ``%g`` would.

    These refusals quote numbers nobody meant — a smoothing scale of a
    million metres, typed or sent by a slipped decimal point. Spelled out
    in full ("1 000 000 m") it reads like a plausible figure with too many
    digits; the exponent says at a glance that it is not one. A sane value
    still reads as itself, and in the reader's unit, because a user who
    typed feet cannot act on a refusal quoted in metres.
    """
    return f"{say.to_length(value_m):g} {say.length_unit}"


def _positive_or_none(value: Any, name: str, units: dict[str, Any] | None = None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be a positive number of {_length_word(Phrase(units))}, "
            "or left empty."
        )
    return value


def _within_field(
    value: float | None,
    name: str,
    across_m: float,
    units: dict[str, Any] | None = None,
) -> None:
    """Refuse a scale wider than the field: the kernel it would build is
    (2 r / cell)² cells, so a stray 1 000 000 m is a terabyte, and no radius
    beyond the field's own span means anything anyway.

    Said in the reader's unit set: the box this number was typed into is
    in feet when the picker is, so a refusal in metres is a refusal the
    user cannot act on — they would have to convert it back to find out
    what to type instead.
    """
    if value is not None and value > across_m:
        say = Phrase(units)
        raise ValueError(
            f"{name} of {_compact(value, say)} is wider than the field, which is about "
            f"{say.length(across_m, 0)} across. Use a smaller value, or leave it "
            "empty to have it chosen from the grid."
        )


def _across_m(grid: ElevationGrid) -> float:
    """The field's span, edge to edge along its diagonal."""
    xmin, xmax, ymin, ymax = lf._valid_extent(grid)
    return float(math.hypot(xmax - xmin + grid.cell, ymax - ymin + grid.cell))


def _points_across_m(dataset: Dataset) -> float:
    """The span of a point file from its lon/lat box, before any gridding:
    good to a few percent, which is all a sanity bound needs."""
    df = dataset.df
    if sch.LON not in df.columns or sch.LAT not in df.columns:
        return float("inf")
    lon = df[sch.LON].to_numpy(dtype="float64")
    lat = df[sch.LAT].to_numpy(dtype="float64")
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        return float("inf")
    lon, lat = lon[ok], lat[ok]
    mid = math.radians(float(lat.mean()))
    dx = (lon.max() - lon.min()) * 111_320.0 * math.cos(mid)
    dy = (lat.max() - lat.min()) * 110_574.0
    return float(math.hypot(dx, dy))


# ==========================================================================
# Result
# ==========================================================================

@dataclass
class TerrainResult:
    """Everything :func:`analyze` produced, on one grid."""

    grid: ElevationGrid
    layers: dict[str, np.ndarray]
    filled: np.ndarray
    fdir: np.ndarray
    acc: np.ndarray
    features: dict[str, list[dict[str, Any]]]
    contours: dict[str, Any]
    drainage: dict[str, Any]
    source_report: dict[str, Any]
    options: TerrainOptions
    trend: dict[str, Any] = field(default_factory=dict)
    wet: np.ndarray | None = None
    twi_threshold: float = float("nan")
    wet_by_index: bool = True
    noise_slope_pct: float = MIN_NOISE_SLOPE_PCT
    wet_candidate_share: float = 1.0
    depressions_unlisted: int = 0
    depression_floor_m: float = hydro_mod.WET_DEPTH_M
    drainage_length_m: float = 0.0
    contour_interval_m: float | None = float("nan")
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    level: bool = False
    residual_relief_m: float = float("nan")
    relief_floor_m: float = float("nan")
    source_name: str = ""
    timing_s: dict[str, float] = field(default_factory=dict)
    _summary: dict[str, Any] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def layer_meta(self) -> list[dict[str, Any]]:
        """The ``layers`` list of the summary: key, label, kind, unit, range, palette."""
        out = []
        for key, label, unit, palette in LAYER_SPECS:
            values = self.layers[key]
            finite = values[np.isfinite(values)]
            out.append({
                "key": key,
                "label": label,
                "kind": "categorical" if key == "landform" else "continuous",
                "unit": unit,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "palette": palette,
            })
        return out

    # ------------------------------------------------------------------
    def _steep(self) -> dict[str, Any]:
        """The ground over :data:`STEEP_PCT`: how much, and where it lies."""
        grid = self.grid
        slope = self.layers["slope_pct"]
        with np.errstate(invalid="ignore"):
            steep = slope > STEEP_PCT
        n_steep = int(np.count_nonzero(steep))
        pct = 100.0 * n_steep / grid.valid_count if grid.valid_count else 0.0
        area_ha = n_steep * grid.cell ** 2 / 10_000.0
        # A patch too small to matter is reported as none: a single steep
        # cell on a 60 ha field is the gridding, not a hillside.
        worth_saying = pct >= 1.0 or area_ha >= 0.2
        return {
            "threshold_pct": STEEP_PCT,
            "pct": pct,
            "area_ha": area_ha,
            "where": _where_steep(self, steep) if worth_saying else None,
        }

    def _wet_position(self) -> str | None:
        """Which part of the field the likely-wet cells sit in, or ``None``."""
        if self.wet is None or not self.wet.any():
            return None
        grid = self.grid
        rows_i, cols_i = np.nonzero(self.wet)
        cx = grid.x0 + (cols_i.mean() + 0.5) * grid.cell
        cy = grid.y0 - (rows_i.mean() + 0.5) * grid.cell
        return lf.position_label(grid, cx, cy)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """The JSON summary (cached): every number a plain Python type.

        The sentences it carries — the findings, the character's reason,
        the note under the share tables, the notes the grid builder and the
        raster reader wrote while the file was being read — are written in
        metric, like every number stored in this app, and :func:`restate`
        writes them all again for a reader working in another unit set.
        The reading's notes manage it because ``source['note_facts']``
        keeps what each one states, not only how it was said.
        """
        if self._summary is not None:
            return self._summary
        grid = self.grid
        cell = grid.cell
        z = grid.z[grid.mask]
        slope = self.layers["slope_pct"]
        slope_f = slope[np.isfinite(slope)]
        twi = self.layers["twi"]
        twi_f = twi[np.isfinite(twi)]

        z_hist = _histogram(z, 20)
        s_hist = _histogram(slope_f, 20)
        relief = float(z.max() - z.min()) if z.size else 0.0
        mean_slope = float(slope_f.mean()) if slope_f.size else 0.0
        p95_slope = float(np.percentile(slope_f, 95)) if slope_f.size else 0.0
        trend = self.trend
        direction = float(trend.get("direction_deg", 0.0))

        # The shares are read inside the outermost ring of cells, where
        # every 3x3 window is real ground; the areas are those shares of
        # the whole field, so they still add up to the field.
        area_ha = grid.area_ha()
        interior = _interior(grid.mask)
        n_interior = int(np.count_nonzero(interior))
        ring_ha = (grid.valid_count - n_interior) * cell ** 2 / 10_000.0
        slope_in = np.where(interior, slope, np.nan)
        aspect_in = np.where(interior, self.layers["aspect_deg"], np.nan)
        codes_in = np.where(interior, _codes(self.layers["landform"]), 0)
        wet_in = int(np.count_nonzero(self.wet & interior)) if self.wet is not None else 0
        wet_pct = 100.0 * wet_in / n_interior if n_interior else 0.0
        grid_info = grid.to_dict()
        grid_info["interior_ha"] = n_interior * cell ** 2 / 10_000.0
        grid_info["edge_ring_ha"] = ring_ha
        grid_info["shares_note"] = shares_note(grid_info)

        summary = {
            "source": dict(self.source_report),
            "grid": grid_info,
            "elevation": {
                "min_m": float(z.min()) if z.size else None,
                "max_m": float(z.max()) if z.size else None,
                "mean_m": float(z.mean()) if z.size else None,
                "median_m": float(np.median(z)) if z.size else None,
                "relief_m": relief,
                "p05_m": float(np.percentile(z, 5)) if z.size else None,
                "p95_m": float(np.percentile(z, 95)) if z.size else None,
                "histogram": z_hist,
                "residual_relief_m": float(self.residual_relief_m),
                "relief_floor_m": float(self.relief_floor_m),
                "level": bool(self.level),
            },
            "slope": {
                "mean_pct": mean_slope,
                "median_pct": float(np.median(slope_f)) if slope_f.size else 0.0,
                "p95_pct": p95_slope,
                "max_pct": float(slope_f.max()) if slope_f.size else 0.0,
                "histogram": s_hist,
                "classes": _field_areas(lf.slope_classes(slope_in, cell), area_ha),
                # The steep ground, measured here rather than in the sentence
                # that reports it: the layers live in memory and are evicted,
                # the summary is saved with the project, and the sentence has
                # to be writable again from the summary alone.
                "steep": self._steep(),
            },
            "aspect": {
                "sectors": _field_areas(lf.aspect_distribution(aspect_in, slope_in, cell), area_ha),
            },
            "trend": {
                "gradient_pct": float(trend.get("gradient_pct", 0.0)),
                "direction_deg": direction,
                "direction_label": _sector_name(direction),
                "drop_m": float(trend.get("drop_m", 0.0)),
                "r2": float(trend.get("r2", 0.0)),
            },
            "character": lf.character(relief, mean_slope, p95_slope),
            "landforms": {
                "tpi_small_m": self.options.tpi_small_m,
                "tpi_large_m": self.options.tpi_large_m,
                "classes": _field_areas(lf.class_areas(codes_in, cell), area_ha),
            },
            "features": {
                "hills": [_public_hill(h) for h in self.features.get("hills", [])],
                "lows": [_public_low(l) for l in self.features.get("lows", [])],
                "depressions": [_public_depression(d) for d in self.features.get("depressions", [])],
                "depressions_unlisted": int(self.depressions_unlisted),
                "depression_floor_m": float(self.depression_floor_m),
            },
            "wetness": {
                "twi_mean": float(twi_f.mean()) if twi_f.size else None,
                "twi_p90": float(np.percentile(twi_f, 90)) if twi_f.size else None,
                "twi_threshold": float(self.twi_threshold),
                "noise_slope_pct": float(self.noise_slope_pct),
                "ranked_share_pct": 100.0 * float(self.wet_candidate_share),
                "wet_area_ha": wet_pct / 100.0 * area_ha,
                "wet_pct": wet_pct,
                "drainage_length_m": float(self.drainage_length_m),
                # Whether the index could rank anything, and where the wet
                # cells sit: both read off the arrays, so that the sentences
                # about them can be written again from the summary alone.
                "by_index": bool(self.wet_by_index),
                "where": self._wet_position(),
            },
            "contours": {
                # None on a level field: no interval draws anything there.
                "interval_m": None if self.contour_interval_m is None else float(self.contour_interval_m),
                "count": len(self.contours.get("features", [])),
            },
            "layers": self.layer_meta(),
            "findings": [],
            "options": self.options.to_dict(),
        }
        summary = _jsonable(summary)
        summary["findings"] = findings(self, summary)
        self._summary = summary
        return summary

    # ------------------------------------------------------------------
    def feature_collection(self) -> dict[str, Any]:
        """Hills, lows and depressions as WGS84 polygons (GeoJSON).

        Each feature's properties are the feature dict plus ``kind`` in
        ``{'hill', 'low', 'depression'}``.
        """
        features: list[dict[str, Any]] = []
        for kind, key, public in (
            ("hill", "hills", _public_hill),
            ("low", "lows", _public_low),
            ("depression", "depressions", _public_depression),
        ):
            labels = self.labels.get(key)
            if labels is None:
                continue
            polygons = _polygons_from_labels(self.grid, labels, simplify_m=self.grid.cell / 2.0)
            for item in self.features.get(key, []):
                geom = polygons.get(int(item["id"]))
                if geom is None:
                    continue
                props = dict(public(item))
                props["kind"] = kind
                features.append({
                    "type": "Feature",
                    "geometry": _geojson_geometry(_to_wgs84(self.grid, geom)),
                    "properties": _jsonable(props),
                })
        return {"type": "FeatureCollection", "features": features}


# ==========================================================================
# The analysis
# ==========================================================================

def analyze(
    dataset: Dataset,
    options: TerrainOptions | None = None,
    units: dict[str, Any] | None = None,
) -> TerrainResult:
    """Run the whole terrain chain on a dataset.

    A dataset that came from a DEM (``meta.extra['dem_path']``) is analysed
    from the raster itself at full resolution, because the dataset's points
    are only a sample of its cells; if that file has gone the analysis
    stops and says so rather than quietly gridding the sample as if it were
    a GPS survey. Anything else is gridded from its GPS altitude.

    ``units`` is the reader's unit set, and it decides how the answer is
    said, never what it holds: every number the analysis produces is
    metric, the findings and the reading's own notes are written from
    those numbers, and :func:`restate` writes them again for whoever
    reads next. Passing nothing here costs a caller nothing but a metric
    first reading.
    """
    options = options or TerrainOptions()
    if isinstance(options, dict):
        options = TerrainOptions.from_dict(options)
    options = _coerced(options)
    cell_m = _positive_or_none(options.cell_m, "The cell size", units)
    smooth_m = None if options.smooth_m is None else float(options.smooth_m)
    if smooth_m is not None and (not np.isfinite(smooth_m) or smooth_m < 0):
        raise ValueError(
            "The smoothing scale must be zero or a positive number of "
            f"{_length_word(Phrase(units))}."
        )
    max_cells = int(options.max_cells) if options.max_cells else 400_000
    if max_cells < 100:
        raise ValueError("max_cells must allow at least 100 cells.")
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    # ---- the grid
    extra = dataset.meta.extra or {}
    if extra.get("zones_by"):
        # The zones this analyser made: their points are cell centres and
        # their polygons carry one row each. Gridding either as a GPS track
        # would give back a coarser copy of the relief they were cut from.
        source = extra.get("terrain_source") or "its source"
        raise ValueError(
            f"These are terrain zones derived from {source!r}; analyse {source!r} "
            "instead."
        )
    dem_path = extra.get("dem_path")
    if dem_path:
        if not Path(str(dem_path)).is_file():
            raise ValueError(
                f"The elevation raster this layer was read from, {dem_path}, can no "
                "longer be found (moved or deleted). The points on the map are only a "
                "sample of its cells, so put the file back or open it again before "
                "analysing the relief."
            )
        grid, source = _grid_from_dem(
            str(dem_path), cell_m, smooth_m, max_cells,
            elevation_factor=float(extra.get("elevation_factor") or 1.0),
            elevation_unit_in=extra.get("elevation_unit_in"),
            units=units,
        )
    else:
        _within_field(smooth_m, "The smoothing scale", _points_across_m(dataset), units)
        grid, report = grid_from_points(
            dataset,
            cell_m=cell_m,
            smooth_m=smooth_m,
            detrend_passes=bool(options.detrend_passes),
            remove_outliers=bool(options.remove_outliers),
            max_cells=max_cells,
            units=units,
        )
        source = {
            "kind": "points",
            "points_total": int(report["points_total"]),
            "points_used": int(report["points_used"]),
            "outliers_removed": int(report["outliers_removed"]),
            "passes": int(report["passes"]),
            "pass_offset_sd_m": report["pass_offset_sd_m"],
            "vertical_noise_m": float(report["vertical_noise_m"]),
            "detrended": bool(report["detrended"]),
            "dem_path": None,
            "notes": list(report.get("notes", [])),
            "note_facts": list(report.get("note_facts", [])),
            "smooth_m": float(report["smooth_m"]),
            "value_step_m": float(report.get("value_step_m") or 0.0),
            "passes_dropped": int(report.get("passes_dropped") or 0),
            "surface_noise_m": float(report.get("surface_noise_m") or 0.0),
            "missing_elevations": int(report.get("missing_elevations") or 0),
            "non_numeric_elevations": int(report.get("non_numeric_elevations") or 0),
            "unit_doubt": _elevation_unit_doubt(dataset),
        }
    span_rows = int(np.count_nonzero(grid.mask.any(axis=1)))
    span_cols = int(np.count_nonzero(grid.mask.any(axis=0)))
    if span_rows < MIN_GRID_SPAN or span_cols < MIN_GRID_SPAN:
        # Caught here rather than left to skimage's "Input array must be at
        # least 2x2": a single row of cells is a line, not a surface.
        raise ValueError(
            f"The elevation covers only {span_rows} row{'s' if span_rows != 1 else ''} by "
            f"{span_cols} column{'s' if span_cols != 1 else ''} of grid cells, too small "
            f"to analyse: a relief needs at least {MIN_GRID_SPAN} by {MIN_GRID_SPAN}. Use a "
            "finer cell size or a file covering the whole field."
        )
    if grid.valid_count < 9:
        raise ValueError(
            "The elevation covers fewer than nine grid cells, too little ground to "
            "describe a relief. Use a finer cell size or a file covering the whole field."
        )
    timing["grid"] = time.perf_counter() - t0

    # ---- resolved options
    resolved = _resolve_options(options, grid, source, units)
    resolved.cell_m = grid.cell
    resolved.smooth_m = source.get("smooth_m", smooth_m)
    resolved.max_cells = max_cells

    # ---- derivatives
    t1 = time.perf_counter()
    slope_pct, _slope_deg = deriv.slope(grid)
    aspect_deg = deriv.aspect(grid)
    profile, plan, _total = deriv.curvature(grid)
    shade = deriv.hillshade(
        grid, azimuth_deg=float(resolved.hillshade_azimuth),
        altitude_deg=float(resolved.hillshade_altitude),
    )
    trend = deriv.plane_trend(grid)
    xx, yy = grid.xy()
    detrended = grid.with_values(grid.z - (trend["a"] * xx + trend["b"] * yy))
    tpi_small = deriv.tpi(detrended, float(resolved.tpi_small_m))
    tpi_large = deriv.tpi(detrended, float(resolved.tpi_large_m))
    tpi_small_std = deriv.tpi_standardized(tpi_small)
    tpi_large_std = deriv.tpi_standardized(tpi_large)
    min_height = float(resolved.min_feature_height_m)
    tpi_floor = LEVEL_TPI_FRACTION * min_height
    with np.errstate(invalid="ignore"):
        tpi_small_std[np.abs(tpi_small) < tpi_floor] = 0.0
        tpi_large_std[np.abs(tpi_large) < tpi_floor] = 0.0
    residual = detrended.z[grid.mask]
    residual_relief = float(np.percentile(residual, 95) - np.percentile(residual, 5))
    relief_floor = max(
        LEVEL_RELIEF_M,
        2.0 * float(source.get("vertical_noise_m") or 0.0),
        float(source.get("value_step_m") or 0.0),
    )
    level = residual_relief < relief_floor and float(trend["drop_m"]) < relief_floor
    timing["derivatives"] = time.perf_counter() - t1

    # ---- hydrology
    t2 = time.perf_counter()
    hydro = hydro_mod.analyse_hydrology(
        grid, slope_pct, min_upstream_ha=float(resolved.min_upstream_ha)
    )
    with np.errstate(invalid="ignore"):
        ponded = hydro.depth > hydro_mod.WET_DEPTH_M
    # A closed depression is listed on the same terms as a hill: deeper
    # than the feature floor and at least the feature area. The rest are
    # counted, so the reader knows they exist, and left to the ponding
    # layer, so the noise does not come back as a list of potholes.
    depression_floor = max(hydro_mod.WET_DEPTH_M, min_height)
    if level:
        dep_list, dep_labels, unlisted = [], np.zeros(grid.shape, dtype="int32"), 0
    else:
        dep_list, dep_labels, unlisted = _depression_features(
            grid, hydro, depression_floor, MIN_DEPRESSION_AREA_HA
        )
    # The wetness index ranks cells against each other, so its top decile
    # exists on every field; where the residual relief is within the noise
    # that ranking is noise too, and only the listed ponding is marked.
    wet_by_index = residual_relief >= min_height and not level
    noise_slope = _noise_slope_pct(
        float(source.get("surface_noise_m") or 0.0), float(source.get("smooth_m") or 0.0), grid.cell
    )
    wet, twi_threshold, candidate_share = _wet_cells(
        grid, hydro.twi, slope_pct, dep_labels > 0, noise_slope, wet_by_index
    )
    timing["hydrology"] = time.perf_counter() - t2

    # ---- landforms and discrete features
    t3 = time.perf_counter()
    landform = lf.classify(tpi_small_std, tpi_large_std, slope_pct)
    if level:
        # Nothing on a level field stands out of the noise, whatever the
        # component labelling finds in it.
        no_labels = np.zeros(grid.shape, dtype="int32")
        hill_list, hill_labels = [], no_labels
        low_list, low_labels = [], no_labels
    else:
        # Each kind is found once on its own, so that the other kind knows
        # which of its features are real, then measured against them: the
        # saddle to a listed hill or low is where a feature's foot is.
        _, hills_alone = lf.hills(
            grid, tpi_large_std, slope_pct, min_height_m=min_height, measure=detrended.z
        )
        _, lows_alone = lf.lows(
            grid, tpi_large_std, ponded, slope_pct, min_depth_m=min_height, measure=detrended.z
        )
        hill_list, hill_labels = lf.hills(
            grid, tpi_large_std, slope_pct, min_height_m=min_height, measure=detrended.z,
            stop=lows_alone > 0,
        )
        low_list, low_labels = lf.lows(
            grid, tpi_large_std, ponded, slope_pct, min_depth_m=min_height, measure=detrended.z,
            stop=hills_alone > 0,
        )
    timing["landforms"] = time.perf_counter() - t3

    # ---- contours and drainage lines
    t4 = time.perf_counter()
    interval = _positive_or_none(resolved.contour_interval_m, "The contour interval", units)
    if level:
        contour_features: list[dict[str, Any]] = []
        interval_used = None
        drainage_lines = []
        drainage_length = 0.0
    else:
        contour_features, interval_used = contours_mod.contours(grid, interval_m=interval)
        resolved.contour_interval_m = float(interval_used)
        drainage_lines = hydro.drainage_lines
        drainage_length = float(hydro.drainage_length_m)
    drainage = _drainage_collection(grid, drainage_lines)
    timing["contours"] = time.perf_counter() - t4

    layers = {
        "elevation": grid.z.copy(),
        "slope_pct": slope_pct,
        "aspect_deg": aspect_deg,
        "hillshade": shade,
        "tpi_small": tpi_small,
        "tpi_large": tpi_large,
        "twi": hydro.twi,
        "landform": np.where(landform > 0, landform, np.nan).astype("float64"),
        "curvature_profile": profile,
        "curvature_plan": plan,
        "depression_depth": hydro.depth,
        "flow_acc": hydro.acc,
    }
    timing["total"] = time.perf_counter() - t0
    return TerrainResult(
        grid=grid,
        layers=layers,
        filled=hydro.filled,
        fdir=hydro.fdir,
        acc=hydro.acc,
        features={"hills": hill_list, "lows": low_list, "depressions": dep_list},
        contours={"type": "FeatureCollection", "features": contour_features},
        drainage=drainage,
        source_report=source,
        options=resolved,
        trend=trend,
        wet=wet,
        twi_threshold=float(twi_threshold),
        wet_by_index=bool(wet_by_index),
        noise_slope_pct=float(noise_slope),
        wet_candidate_share=float(candidate_share),
        depressions_unlisted=int(unlisted),
        depression_floor_m=float(depression_floor),
        drainage_length_m=drainage_length,
        contour_interval_m=None if interval_used is None else float(interval_used),
        labels={"hills": hill_labels, "lows": low_labels, "depressions": dep_labels},
        source_name=dataset.meta.name,
        timing_s=timing,
        level=bool(level),
        residual_relief_m=residual_relief,
        relief_floor_m=float(relief_floor),
    )


def _grid_from_dem(
    path: str,
    cell_m: float | None,
    smooth_m: float | None,
    max_cells: int,
    elevation_factor: float = 1.0,
    elevation_unit_in: str | None = None,
    units: dict[str, Any] | None = None,
) -> tuple[ElevationGrid, dict[str, Any]]:
    """The DEM re-read at full resolution, coarsened to the cell budget.

    ``elevation_factor`` is the unit conversion declared when the file was
    opened (feet to metres): the dataset's points already carry it, the
    raster on disk does not. A quantised raster (whole metres) is smoothed
    by default, because its terraces would otherwise be read as slopes and
    hills; the smoothed values are rounded to the micrometre so that the
    plateaus a Gaussian leaves are exactly level and a contour at their
    height is one line, not hundreds of loops around rounding noise.
    """
    from ..formats import raster as raster_mod

    grid, info = raster_mod.read_dem(path, target_cell_m=cell_m, units=units)
    facts = list(info.get("note_facts") or [notes_mod.plain(n) for n in info.get("notes", [])])
    if elevation_factor != 1.0:
        grid = grid.with_values(grid.z * elevation_factor)
        unit = elevation_unit_in or "the declared unit"
        # The unit named here is the one the file declared, not the one the
        # reader works in, so the sentence is the same for every reader.
        facts.append(notes_mod.plain(
            f"Elevations were converted from {unit} to metres, as declared when the "
            "file was opened."
        ))
    step = float(info.get("value_step_m") or 0.0) * abs(elevation_factor)
    total_cells = grid.rows * grid.cols
    if total_cells > max_cells:
        factor = int(math.ceil(math.sqrt(total_cells / max_cells)))
        grid = grid.coarsen(factor)
        facts.append(notes_mod.fact(
            "dem_coarsened", cell_in_m=float(info["cell_m"]), factor=factor,
            cell_m=float(grid.cell), max_cells=max_cells))
    if smooth_m is None and step > 0:
        # At least one cell: a sigma under half a cell changes nothing.
        smooth_m = max(QUANTISED_SMOOTH_STEPS * step, grid.cell)
        smooth_m = min(smooth_m, _across_m(grid) / 4.0)
    if smooth_m:
        _within_field(smooth_m, "The smoothing scale", _across_m(grid), units)
        smoothed = gaussian_smooth(grid.z, smooth_m / grid.cell)
        if step > 0:
            smoothed = np.round(smoothed, 6)
        smoothed[~grid.mask] = np.nan
        grid = grid.with_values(smoothed)
    return grid, {
        "kind": "dem",
        "points_total": int(total_cells),
        "points_used": int(grid.valid_count),
        "outliers_removed": 0,
        "passes": 0,
        "pass_offset_sd_m": None,
        "vertical_noise_m": None,
        "detrended": False,
        "dem_path": str(path),
        "notes": notes_mod.render(facts, units),
        "note_facts": facts,
        "smooth_m": float(smooth_m or 0.0),
        "value_step_m": step,
        # A raster carries no reading noise the analysis can measure.
        "surface_noise_m": None,
        "missing_elevations": int(total_cells - grid.valid_count),
        "non_numeric_elevations": 0,
        # A raster declares its unit through meta.extra, so no doubt applies.
        "unit_doubt": None,
    }


def _resolve_options(
    options: TerrainOptions,
    grid: ElevationGrid,
    source: dict,
    units: dict[str, Any] | None = None,
) -> TerrainOptions:
    """Fill the ``None`` options from the grid.

    ``units`` reaches the refusals only — every option it resolves is
    metric — so that a scale the user typed is refused in the unit they
    typed it in.
    """
    say = Phrase(units)
    resolved = TerrainOptions(**options.to_dict())
    cell = grid.cell
    xmin, xmax, ymin, ymax = lf._valid_extent(grid)
    shorter = max(min(xmax - xmin, ymax - ymin) + cell, 3.0 * cell)
    across = _across_m(grid)
    large = _positive_or_none(options.tpi_large_m, "The large TPI radius", units)
    small = _positive_or_none(options.tpi_small_m, "The small TPI radius", units)
    _within_field(large, "The large TPI radius", across, units)
    _within_field(small, "The small TPI radius", across, units)
    if large is None:
        large = max(10.0 * cell, 150.0)
        large = max(min(large, shorter / 4.0), 3.0 * cell)
        if small is not None:
            # A chosen small radius must stay the smaller scale.
            large = max(large, 2.0 * small)
    if small is None:
        small = max(3.0 * cell, 30.0)
        small = max(min(small, large / 2.0), cell)
    if not small < large:
        raise ValueError(
            f"The small TPI radius ({_compact(small, say)}) must be smaller than the "
            f"large one ({_compact(large, say)}): the small scale tells knolls from "
            "hollows, the large one the hill from the valley. Set both, or leave them "
            "empty to have them chosen from the grid."
        )
    resolved.tpi_small_m = float(small)
    resolved.tpi_large_m = float(large)
    if options.min_feature_height_m is None:
        noise = source.get("vertical_noise_m") or 0.0
        # A DEM of whole metres cannot vouch for anything under a metre: a
        # terrace riser is exactly one step tall and would pass for a hill.
        step = source.get("value_step_m") or 0.0
        resolved.min_feature_height_m = float(max(0.5, 2.0 * float(noise), float(step)))
    else:
        resolved.min_feature_height_m = float(options.min_feature_height_m)
        if not np.isfinite(resolved.min_feature_height_m) or resolved.min_feature_height_m < 0:
            raise ValueError(
                "The minimum feature height must be zero or a positive number of "
                f"{_length_word(say)}."
            )
    upstream = float(options.min_upstream_ha)
    if not np.isfinite(upstream) or upstream <= 0:
        raise ValueError(
            "The upstream area for a drainage line must be a positive number of "
            f"{_area_word(say)}: with no threshold every cell is a channel."
        )
    resolved.min_upstream_ha = upstream
    altitude = float(options.hillshade_altitude)
    if not np.isfinite(altitude) or not 0.0 < altitude <= 90.0:
        raise ValueError(
            "The hillshade sun altitude must be above 0 and at most 90 degrees "
            "(45 is the usual choice)."
        )
    azimuth = float(options.hillshade_azimuth)
    if not np.isfinite(azimuth):
        raise ValueError("The hillshade sun azimuth must be a bearing in degrees (315 is the usual choice).")
    resolved.hillshade_altitude = altitude
    resolved.hillshade_azimuth = azimuth % 360.0
    return resolved


def _depression_features(
    grid: ElevationGrid,
    hydro: hydro_mod.Hydrology,
    min_depth_m: float,
    min_area_ha: float,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """The closed basins that clear the feature floor, with their labels
    and position words, and the count of those that did not.

    The hydrology finds every basin over 5 cm; a basin shallower than
    ``min_depth_m`` (the feature floor, twice the reading noise) or
    smaller than ``min_area_ha`` is a dip in the noise as far as the data
    can tell, so it is counted, not listed. The survivors keep the
    hydrology's order (largest volume first) and are renumbered from 1.
    """
    labels = hydro.depression_labels
    if labels is None:
        labels = np.zeros(grid.shape, dtype="int32")
    relabel = np.zeros(int(labels.max()) + 1, dtype="int32")
    out = []
    unlisted = 0
    for d in hydro.depressions:
        if d["max_depth_m"] < float(min_depth_m) or d["area_ha"] < float(min_area_ha):
            unlisted += 1
            continue
        item = dict(d)
        item["id"] = len(out) + 1
        item["label"] = f"Depression {item['id']}"
        rows_i, cols_i = np.nonzero(labels == int(d["id"]))
        if rows_i.size == 0:
            rows_i, cols_i = np.array([d["row"]]), np.array([d["col"]])
        cx = grid.x0 + (cols_i.mean() + 0.5) * grid.cell
        cy = grid.y0 - (rows_i.mean() + 0.5) * grid.cell
        item["position"] = lf.position_label(grid, cx, cy)
        relabel[int(d["id"])] = item["id"]
        out.append(item)
    return out, relabel[labels], unlisted


def _wet_cells(
    grid: ElevationGrid,
    twi: np.ndarray,
    slope_pct: np.ndarray,
    listed: np.ndarray,
    noise_slope_pct: float,
    by_index: bool,
) -> tuple[np.ndarray, float, float]:
    """The likely-wet cells: the listed depressions, plus, when the index
    is trusted, the cells above the wetness threshold among those that
    have a slope to be ranked on.

    A cell flatter than ``noise_slope_pct`` carries no slope information:
    its tan(beta) is floored and its index is the same constant as every
    other floored cell's, so a dead-flat plain next to one hill came back
    a tenth "likely wet" — the tenth being whichever floored cells the
    accumulation happened to favour. The threshold is the 90th percentile
    of the index over the cells that do have a slope (never under
    :data:`hydrology.MIN_TWI_THRESHOLD`); a listed depression is in
    whatever its slope. When under :data:`MIN_WET_CANDIDATE_SHARE` of the
    field can be ranked, no ranking is made. Returns ``(wet, threshold,
    candidate_share)``.
    """
    listed = np.asarray(listed, dtype=bool) & grid.mask
    if not by_index:
        return listed.copy(), float("nan"), 0.0
    with np.errstate(invalid="ignore"):
        candidates = grid.mask & ((slope_pct >= float(noise_slope_pct)) | listed)
    share = float(np.count_nonzero(candidates)) / grid.valid_count if grid.valid_count else 0.0
    if share < MIN_WET_CANDIDATE_SHARE:
        return listed.copy(), float("nan"), share
    threshold = hydro_mod.default_twi_threshold(twi[candidates])
    with np.errstate(invalid="ignore"):
        wet = (candidates & (twi >= threshold)) | listed
    return wet, float(threshold), share


def _noise_slope_pct(surface_noise_m: float, smooth_m: float, cell_m: float) -> float:
    """The slope, in percent, under which a cell's slope is noise: twice
    the slope noise of the surface, and at least :data:`MIN_NOISE_SLOPE_PCT`.

    The slope noise is the surface noise over the length the slope is read
    across: one cell on a raw surface (a difference of two cells each
    carrying the noise), the smoothing length on a smoothed one, because
    the derivative of a Gaussian-smoothed field spreads its noise over
    the Gaussian's width — on the demo field the slope read off the
    smoothed surface differs from the true surface's by 0.32 % (standard
    deviation), and 0.029 m of surface noise over 10 m of smoothing says
    0.29 %. Over one cell instead it would say 0.59 % and the threshold
    1.2 %, which excludes the axis of a 2.5 m valley falling at 0.8 %:
    the wettest line on the field, marked dry between two wet flanks.
    """
    length = max(float(smooth_m), float(cell_m))
    return max(100.0 * 2.0 * float(surface_noise_m) / length, MIN_NOISE_SLOPE_PCT)


def _interior(mask: np.ndarray) -> np.ndarray:
    """The valid cells whose whole 3x3 window is valid: the mask eroded by
    one cell, the array border counting as outside like a NaN does. Falls
    back to the mask itself when nothing is left (a 3-cell-wide strip),
    because a share of nothing is no share at all."""
    interior = ndimage.binary_erosion(mask, structure=_EIGHT, border_value=0)
    return interior if interior.any() else mask


def _field_areas(rows: list[dict[str, Any]], area_ha: float) -> list[dict[str, Any]]:
    """The share table with each area as that share of the whole field, so
    the table still adds up to the field although the shares were read on
    its interior."""
    for row in rows:
        row["area_ha"] = float(row["pct"]) / 100.0 * float(area_ha)
    return rows


def _coerced(options: TerrainOptions) -> TerrainOptions:
    """The options with every number a number, or a sentence saying which
    one is not: an interface or an MCP call can send ``'abc'`` or ``''``
    for a box left empty, and ``float('abc')`` explains nothing."""
    out = TerrainOptions(**options.to_dict())
    for name in _FLOAT_OPTIONS:
        setattr(out, name, _option_number(getattr(options, name), name, float, True))
    out.max_cells = _option_number(options.max_cells, "max_cells", int, False)
    for name in ("detrend_passes", "remove_outliers"):
        value = getattr(options, name)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered not in ("true", "false", "1", "0", "yes", "no", ""):
                raise ValueError(
                    f"The terrain option {name} must be true or false, not {value!r}."
                )
            value = lowered in ("true", "1", "yes")
        setattr(out, name, bool(value))
    return out


#: The options that are a number of metres, degrees or hectares, or empty.
_FLOAT_OPTIONS = (
    "cell_m", "smooth_m", "tpi_small_m", "tpi_large_m", "contour_interval_m",
    "min_feature_height_m", "hillshade_azimuth", "hillshade_altitude", "min_upstream_ha",
)


def _option_number(value: Any, name: str, kind, optional: bool):
    """``value`` as ``kind`` (float or int); ``None`` or blank means "not
    set" where that is allowed."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        if optional:
            return None
        return None
    if isinstance(value, bool):
        raise ValueError(f"The terrain option {name} must be a number, not {value!r}.")
    try:
        number = kind(float(value)) if kind is int else float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"The terrain option {name} must be a number, not {value!r}; leave it "
            "empty to have it chosen from the data."
        ) from None
    if kind is int and float(number) != float(value):
        raise ValueError(f"The terrain option {name} must be a whole number, not {value!r}.")
    return number


def _drainage_collection(grid: ElevationGrid, lines) -> dict[str, Any]:
    """Drainage lines as a WGS84 GeoJSON FeatureCollection with lengths."""
    features = []
    for index, line in enumerate(lines):
        xy = np.asarray(line.coords, dtype="float64")
        lon, lat = grid.to_lonlat(xy[:, 0], xy[:, 1])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)],
            },
            "properties": {"index": index, "length_m": float(line.length)},
        })
    return {"type": "FeatureCollection", "features": features}


# ==========================================================================
# Findings
# ==========================================================================

def findings(
    result: "TerrainResult | dict[str, Any]",
    summary: dict[str, Any] | None = None,
    units: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """The sentences a farmer reads, in the order the relief is described.

    Each entry is ``{'level': 'ok' | 'info' | 'warning', 'text': ...}``.

    Every quantity goes through a :class:`~agrosuite.core.units.Phrase`
    built on ``units`` — the reader's unit set, the shape of
    ``UNIT_PRESETS['canada']`` — so the same analysis says "total relief
    is 34.6 ft over 149 ac" to someone working in acres and "10.5 m over
    60.5 ha" to someone working in hectares. ``None`` is the metric store,
    which is what a caller that names no units gets, and what
    :meth:`TerrainResult.summary` writes.

    The thresholds convert with the measurements. "Features smaller than
    0.6 m are not reported" is as much a claim about the ground as the
    heights above it, and a reader who works in feet cannot check a claim
    made in metres against a relief quoted in feet.

    ``result`` is the analysed :class:`TerrainResult`, or — once the
    analysis is nothing but the summary it left behind, saved with the
    project or evicted from memory — that summary itself. The sentences
    are written from the summary alone either way, which is what lets a
    stored report be said again in another unit set.
    """
    if isinstance(result, dict):
        summary = result
    elif summary is None:
        # summary() calls findings() itself; pass its dict in to avoid the loop.
        return result.summary()["findings"]
    say = Phrase(units)
    out: list[dict[str, str]] = []
    elev = summary["elevation"]
    area_ha = float(summary["grid"]["area_ha"])

    # (0) a unit that would make every number below wrong comes first
    doubt = summary["source"].get("unit_doubt")
    if doubt:
        out.append(_warning(str(doubt)))

    # (1) character, relief, area
    char = _character(summary, units)
    out.append(_info(
        f"Total relief is {say.length(elev['relief_m'])} over {say.area(area_ha)}: "
        f"a {char['label']} field. {char['why']}"
    ))

    # (2) the general fall
    trend = summary["trend"]
    if trend["drop_m"] >= MIN_TREND_DROP_M and (trend["r2"] > 0.3 or trend["gradient_pct"] > 0.5):
        from_word = _sector_name(trend["direction_deg"] + 180.0)
        out.append(_info(
            f"The field falls about {say.length(trend['drop_m'])} from the {from_word} to the "
            f"{trend['direction_label']}, an average gradient of "
            f"{trend['gradient_pct']:.1f} %."
        ))

    # (3) hills
    hills = summary["features"]["hills"]
    lows = summary["features"]["lows"]
    if hills:
        out.append(_info(_count_sentence(len(hills), "distinct hill", "rises", "rise")
                         + " above the surrounding ground."))
        for h in hills[:5]:
            text = (
                f"{h['label']} in the {h['position']} rises {say.length(h['height_m'])} above "
                f"its surroundings over {say.area(h['area_ha'])}"
            )
            if h.get("mean_slope_pct") is not None:
                text += f"; its top averages {h['mean_slope_pct']:.1f} % slope"
            out.append(_info(text + "."))
        if len(hills) > 5:
            out.append(_info(f"{len(hills) - 5} smaller hills are drawn on the map but not listed here."))

    # (4) lows and valleys
    if lows:
        for l in lows[:5]:
            shape = "valley" if l.get("elongation", 1.0) >= VALLEY_ELONGATION else "hollow"
            leave = (
                "part of it is closed, so water ponds there rather than running off"
                if l["closed"] else "water drains out of it"
            )
            out.append(_info(
                f"{l['label']}, a {shape} in the {l['position']}, lies "
                f"{say.length(l['depth_m'])} below the ground around it over "
                f"{say.area(l['area_ha'])}; {leave}."
            ))
        if len(lows) > 5:
            out.append(_info(f"{len(lows) - 5} smaller lows are drawn on the map but not listed here."))
    if not hills and not lows:
        out.append(_info("No distinct hills or hollows stand out from the general fall of the field."))

    # (5) closed depressions
    deps = summary["features"]["depressions"]
    for d in deps[:5]:
        text = (
            f"A closed depression of {say.area(d['area_ha'])} in the {d['position']}, up to "
            f"{say.length(d['max_depth_m'])} deep, holds roughly {say.volume(d['volume_m3'])} "
            "before it spills; expect ponding after heavy rain or snowmelt."
        )
        out.append(_warning(text) if d["volume_m3"] > WARN_VOLUME_M3 else _info(text))
    if len(deps) > 5:
        rest = deps[5:]
        out.append(_info(
            f"{len(rest)} smaller depressions hold another "
            f"{say.volume(sum(d['volume_m3'] for d in rest))} together."
        ))
    unlisted = int(summary["features"].get("depressions_unlisted") or 0)
    if unlisted:
        floor = float(summary["features"]["depression_floor_m"])
        out.append(_info(
            f"{unlisted} shallow hollow{'s' if unlisted != 1 else ''} within the elevation "
            f"noise (under {say.length(floor)} deep, or under "
            f"{say.area(MIN_DEPRESSION_AREA_HA, 2)}) "
            f"{'were' if unlisted != 1 else 'was'} not listed as "
            f"{'closed depressions' if unlisted != 1 else 'a closed depression'}; the "
            f"ponding-depth layer still shows {'them' if unlisted != 1 else 'it'}."
        ))

    # (6) steep ground
    steep = summary["slope"].get("steep") or {}
    has_steep = bool(steep.get("where"))
    if has_steep:
        out.append(_warning(
            f"{say.percent(steep['pct'])} of the field ({say.area(steep['area_ha'])}) is "
            f"steeper than {float(steep.get('threshold_pct') or STEEP_PCT):.0f} %, "
            f"{steep['where']}; erosion risk on bare soil."
        ))

    # (7) wetness — or, on a level field, the one sentence that explains
    # why the map has no contours, drainage lines, wet ground or features
    wet = summary["wetness"]
    if elev.get("level"):
        out.append(_info(
            f"The field is level within the precision of its elevation: beyond its "
            f"general fall the ground varies by {say.length(elev['residual_relief_m'], 2)} "
            f"(5th to 95th percentile), under the {say.length(elev['relief_floor_m'], 2)} "
            "the data can vouch for, so no contour lines, drainage lines, likely-wet "
            "ground or hills and hollows are drawn; they would trace the noise, not "
            "the ground."
        ))
    elif not wet.get("by_index"):
        out.append(_info(
            "The relief beyond the general fall is within the elevation noise, so the "
            "wetness index cannot tell wet ground from dry here; only the listed "
            "depressions are marked as likely wet."
        ))
    elif float(wet["ranked_share_pct"]) / 100.0 < MIN_WET_CANDIDATE_SHARE:
        out.append(_info(
            f"Only {say.percent(wet['ranked_share_pct'])} of the field has a slope "
            f"the surface can vouch for (over {float(wet['noise_slope_pct']):.1f} %), too "
            "little to rank wet ground against dry; only the listed depressions are "
            "marked as likely wet."
        ))
    if wet.get("where") and wet["wet_pct"] >= 1.0:
        along = ", along the drainage lines" if wet["drainage_length_m"] > 0 else ""
        out.append(_info(
            f"Likely wet ground covers {say.area(wet['wet_area_ha'])} "
            f"({say.percent(wet['wet_pct'])}), mostly in the {wet['where']}{along}."
        ))

    # (8) data quality
    source = summary["source"]
    if source["kind"] == "points":
        noise = source.get("vertical_noise_m")
        min_h = float(summary["options"]["min_feature_height_m"])
        if noise is not None:
            out.append(_info(
                f"GPS elevation noise is about {say.length(noise)} after smoothing; "
                f"features smaller than {say.length(min_h)} are not reported."
            ))
        sd = source.get("pass_offset_sd_m")
        if sd is not None and sd > RTK_OFFSET_SD_M:
            out.append(_warning(
                f"The altitude shifted by about {say.length(sd)} from pass to pass, which "
                "means the receiver had no RTK correction. The offsets were corrected before "
                "gridding, so the shape of the relief is sound, but absolute heights are "
                "approximate."
            ))
    else:
        cell = float(summary["grid"]["cell_m"])
        step = float(source.get("value_step_m") or 0.0)
        if step > 0:
            min_h = float(summary["options"]["min_feature_height_m"])
            smooth = float(source.get("smooth_m") or 0.0)
            out.append(_info(
                f"The elevation raster ({say.length(cell, None)} cells) stores heights in "
                f"steps of {say.length(step, None)}, which turns a gentle slope into "
                f"terraces; the surface was smoothed over {say.length(smooth, None)} before "
                f"reading slopes, and features smaller than {say.length(min_h)} are not "
                "reported because a single step would pass for one."
            ))
        else:
            out.append(_info(
                f"The relief was read from the elevation raster at {say.length(cell, None)} "
                "cells; no GPS noise or pass offsets apply."
            ))
    for note in source_notes(source, units):
        out.append(_info(note))

    # (9) closing line
    if not deps and not has_steep:
        out.append(_ok(
            "No closed depressions or steep ground: the relief should not limit "
            "drainage or machinery."
        ))
    return out


def _character(summary: dict[str, Any], units: dict[str, Any] | None) -> dict[str, str]:
    """The field's character with its reason written in ``units``.

    The reason carries a height, so it is prose and follows the reader
    like every other sentence; the class it names is decided by the same
    numbers whatever the reader works in.
    """
    return lf.character(
        float(summary["elevation"].get("relief_m") or 0.0),
        float(summary["slope"].get("mean_pct") or 0.0),
        float(summary["slope"].get("p95_pct") or 0.0),
        units,
    )


def source_notes(source: dict[str, Any], units: dict[str, Any] | None = None) -> list[str]:
    """What the grid builder and the raster reader had to say, in ``units``.

    These come from the reading of the file rather than from the analysis
    of the ground, but they are sentences about ground all the same — a
    smoothing scale, the spacing between readings, two blocks so far
    apart they cannot be one field — and a reader working in feet cannot
    check them against a relief quoted in feet. So the reading keeps the
    *fact* each one states (``source['note_facts']``, see
    :mod:`agrosuite.terrain.notes`) and the sentence is written here,
    every time, from the same metric numbers. Nothing is read again.

    A summary written before the facts existed, or by a version that knows
    a note this one does not, still has its sentences: they are given back
    as they were written rather than lost.
    """
    facts = source.get("note_facts")
    if facts:
        try:
            return notes_mod.render(facts, units)
        except ValueError:
            pass
    return [str(note) for note in source.get("notes", [])]


def shares_note(grid_info: dict[str, Any], units: dict[str, Any] | None = None) -> str:
    """Why the share tables do not cover the whole field, in ``units``."""
    say = Phrase(units)
    return (
        f"The slope, aspect, landform and wet shares are read on the "
        f"{say.area(grid_info.get('interior_ha'))} inside the field's outermost ring of "
        f"cells; the ring ({say.area(grid_info.get('edge_ring_ha'))}) is drawn on every "
        "layer but left out of the shares, because its slope and aspect lean on copied "
        "values."
    )


def restate(summary: dict[str, Any], units: dict[str, Any] | None = None) -> dict[str, Any]:
    """The same analysis with its sentences written in another unit set.

    Every number in the summary stays exactly as it was stored — metric,
    to the last decimal — and only the prose is written again: the
    findings, the reason under the field's character, the note under the
    share tables, and the notes the file's own reading left behind. That
    is what lets the unit picker change every sentence on screen without
    re-running an analysis, and a project saved in one unit set open in
    another.

    The copy is shallow where nothing changed and fresh where it did, so
    the stored summary is never edited underneath its owner.
    """
    if not summary:
        return summary
    out = dict(summary)
    out["grid"] = {**summary.get("grid", {})}
    out["grid"]["shares_note"] = shares_note(out["grid"], units)
    # The notes the file's reading left behind travel as the facts they
    # state, so they move with the picker like everything else on screen.
    out["source"] = {**summary.get("source", {}),
                     "notes": source_notes(summary.get("source", {}), units)}
    character = _character(summary, units)
    out["character"] = {**summary.get("character", {}), "why": character["why"]}
    out["findings"] = findings(out, units=units)
    return out


def _where_steep(result: TerrainResult, steep: np.ndarray) -> str:
    """Where the steep cells lie: on the flank of a named hill when most of
    them sit within reach of one, otherwise the part of the field."""
    grid = result.grid
    rows_i, cols_i = np.nonzero(steep)
    x = grid.x0 + (cols_i + 0.5) * grid.cell
    y = grid.y0 - (rows_i + 0.5) * grid.cell
    hills = result.features.get("hills", [])
    best = None
    for h in hills:
        reach = 2.0 * math.sqrt(h["area_ha"] * 10_000.0 / math.pi) + float(result.options.tpi_large_m)
        near = np.hypot(x - h["x"], y - h["y"]) <= reach
        share = float(near.mean()) if near.size else 0.0
        if share >= 0.4 and (best is None or share > best[0]):
            bearing = math.degrees(math.atan2(x[near].mean() - h["x"], y[near].mean() - h["y"])) % 360.0
            best = (share, h["label"], _sector_name(bearing))
    if best is not None:
        return f"mostly on the {best[2]} side of {best[1]}"
    return f"mostly in the {lf.position_label(grid, float(x.mean()), float(y.mean()))}"


# ==========================================================================
# Zones for the rest of the app
# ==========================================================================

def _zone_codes(result: TerrainResult, by: str, bands: int = 4) -> tuple[np.ndarray, dict[int, str]]:
    """Integer zone codes (0 outside the field) and their labels."""
    grid = result.grid
    if by == "landform":
        codes = _codes(result.layers["landform"])
        return codes, {c["code"]: c["label"] for c in lf.CLASSES}
    if by == "slope_class":
        slope = result.layers["slope_pct"]
        codes = np.zeros(grid.shape, dtype="int32")
        labels = {}
        for k, cls in enumerate(lf.SLOPE_CLASSES, start=1):
            lo, hi = cls["from_pct"], cls["to_pct"]
            with np.errstate(invalid="ignore"):
                sel = (slope >= lo) & ((slope < hi) if hi is not None else np.isfinite(slope))
            codes[sel] = k
            labels[k] = cls["label"]
        return codes, labels
    if by == "elevation_bands":
        bands = int(bands)
        if bands < 2 or bands > 20:
            raise ValueError("Elevation bands must be between 2 and 20.")
        z = grid.z
        finite = z[grid.mask]
        edges = np.quantile(finite, np.linspace(0.0, 1.0, bands + 1))
        # Each band must be at least as wide as its label's precision, or the
        # zones read "700.0–700.0 m" and mean nothing on the monitor.
        relief = float(edges[-1] - edges[0])
        if relief < BAND_MIN_M * bands:
            raise ValueError(
                f"The field has only {relief:.2f} m of relief, too little to split into "
                f"{bands} elevation bands of at least {BAND_MIN_M:g} m each. Ask for fewer "
                "bands, or zone the field by landform, slope class or wetness instead."
            )
        # A quantized DEM can repeat an edge (many cells at exactly the same
        # height); those bands would be empty, so they are merged away.
        edges = np.unique(np.round(edges, 6))
        bands = int(edges.size - 1)
        codes = np.zeros(grid.shape, dtype="int32")
        codes[grid.mask] = np.clip(np.searchsorted(edges[1:-1], finite, side="right") + 1, 1, bands)
        names = _band_names(bands)
        labels = {
            k: f"{names[k - 1]} ({edges[k - 1]:.1f}–{edges[k]:.1f} m)" for k in range(1, bands + 1)
        }
        return codes, labels
    if by == "wetness":
        codes = np.zeros(grid.shape, dtype="int32")
        codes[grid.mask] = 1
        if result.wet is not None:
            codes[result.wet & grid.mask] = 2
        return codes, {1: "Well drained", 2: "Likely wet"}
    raise ValueError(
        f"Zones can be made by {', '.join(ZONE_KINDS)}; '{by}' is not one of them."
    )


def _label_lookup(labels: dict[int, str]) -> np.ndarray:
    """Object array indexed by zone code, for labelling 400 000 cells at once."""
    lut = np.empty(max(labels) + 1, dtype=object)
    lut[:] = ""
    for k, v in labels.items():
        lut[k] = v
    return lut


def _band_names(bands: int) -> list[str]:
    if bands == 2:
        return ["Low", "High"]
    if bands == 3:
        return ["Low", "Middle", "High"]
    if bands == 4:
        return ["Low", "Lower middle", "Upper middle", "High"]
    names = [f"Band {k}" for k in range(1, bands + 1)]
    names[0], names[-1] = "Low", "High"
    return names


def _zone_meta(result: TerrainResult, by: str, labels: dict[int, str], geometry_type: str) -> DatasetMeta:
    return DatasetMeta(
        name=f"{result.source_name or 'terrain'} – {by.replace('_', ' ')} zones",
        source_path=str(result.source_report.get("dem_path") or ""),
        source_format="terrain",
        brand="generic",
        brand_label="Generic / not identified",
        operation="elevation",
        value_label="Terrain zone",
        value_unit="",
        source_value_unit="",
        geometry_type=geometry_type,
        notes=[f"Zones derived from the relief ({by.replace('_', ' ')}) of {result.source_name!r}."],
        extra={
            "terrain_source": result.source_name,
            "zones_by": by,
            "zone_labels": {int(k): v for k, v in labels.items()},
            # The zone points are one per grid cell: with the cell size the
            # dataset knows its area (rows x cell x cell) without the arrays.
            "zones_cell_m": float(result.grid.cell),
        },
    )


def zones_points(result: TerrainResult, by: str, bands: int = 4) -> Dataset:
    """One point per valid cell centre, carrying the zone code and the
    terrain layers a join or an export may want."""
    grid = result.grid
    codes, labels = _zone_codes(result, by, bands)
    mask = grid.mask & (codes > 0)
    xx, yy = grid.xy()
    x = xx[mask]
    y = yy[mask]
    lon, lat = grid.to_lonlat(x, y)
    code = codes[mask].astype("int64")
    df = pd.DataFrame({
        sch.LON: lon,
        sch.LAT: lat,
        sch.VALUE: code.astype("float64"),
        "zone": code,
        "zone_label": _label_lookup(labels)[code],
        sch.ELEVATION: grid.z[mask],
        "slope_pct": result.layers["slope_pct"][mask],
        "twi": result.layers["twi"][mask],
        "landform": _codes(result.layers["landform"])[mask].astype("int64"),
        "aspect_deg": result.layers["aspect_deg"][mask],
    })
    ds = Dataset(df, _zone_meta(result, by, labels, "point"))
    ds.project(grid.crs)
    return ds


def zones_polygons(result: TerrainResult, by: str, bands: int = 4) -> Dataset:
    """The zones dissolved into one (multi)polygon each, in WGS84."""
    grid = result.grid
    codes, labels = _zone_codes(result, by, bands)
    polygons = _polygons_from_labels(grid, codes, simplify_m=grid.cell / 2.0)
    rows = []
    geometry = []
    slope = result.layers["slope_pct"]
    for code in sorted(labels):
        geom = polygons.get(code)
        if geom is None or geom.is_empty:
            continue
        sel = codes == code
        n = int(np.count_nonzero(sel))
        wgs = _to_wgs84(grid, geom)
        rep = wgs.representative_point()
        s = slope[sel]
        s = s[np.isfinite(s)]
        rows.append({
            sch.LON: float(rep.x),
            sch.LAT: float(rep.y),
            "zone": int(code),
            "zone_label": labels[code],
            sch.VALUE: float(code),
            "area_ha": n * grid.cell ** 2 / 10_000.0,
            "mean_elev_m": float(np.nanmean(grid.z[sel])),
            "mean_slope_pct": float(s.mean()) if s.size else 0.0,
        })
        geometry.append(wgs)
    columns = [sch.LON, sch.LAT, "zone", "zone_label", sch.VALUE, "area_ha", "mean_elev_m", "mean_slope_pct"]
    df = pd.DataFrame(rows, columns=columns)
    ds = Dataset(df, _zone_meta(result, by, labels, "polygon"), geometry=geometry)
    if len(df):
        ds.project(grid.crs)
    return ds


# ==========================================================================
# Helpers
# ==========================================================================

def _codes(landform_layer: np.ndarray) -> np.ndarray:
    """The landform layer (float with NaN outside) back to int codes, 0 outside."""
    codes = np.where(np.isfinite(landform_layer), landform_layer, 0.0)
    return np.rint(codes).astype("int32")


def _histogram(values: np.ndarray, bins: int) -> dict[str, list[float]]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = float(values.min()), float(values.max())
    # A constant layer (or one whose spread is below the float spacing at
    # 700 m, which is what smoothing leaves of a constant) gets a 1 m box.
    if hi - lo <= 1e-9 * max(1.0, abs(lo)):
        hi = lo + 1.0
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _sector_name(bearing_deg: float) -> str:
    """Compass word for a bearing clockwise from north."""
    code = int(((float(bearing_deg) + 22.5) % 360.0) // 45.0) % 8
    return deriv.SECTOR_NAMES[code]


def _public_hill(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": h["id"], "label": h["label"], "lon": h["lon"], "lat": h["lat"],
        "area_ha": h["area_ha"], "summit_m": h["summit_m"], "height_m": h["height_m"],
        "mean_slope_pct": h.get("mean_slope_pct"), "position": h["position"],
        "elongation": h.get("elongation", 1.0),
    }


def _public_low(l: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": l["id"], "label": l["label"], "lon": l["lon"], "lat": l["lat"],
        "area_ha": l["area_ha"], "bottom_m": l["bottom_m"], "depth_m": l["depth_m"],
        "closed": bool(l["closed"]), "position": l["position"],
        "elongation": l.get("elongation", 1.0),
    }


def _public_depression(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": d["id"], "label": d["label"], "lon": d["lon"], "lat": d["lat"],
        "area_ha": d["area_ha"], "max_depth_m": d["max_depth_m"],
        "volume_m3": d["volume_m3"], "spill_m": d["spill_m"], "position": d["position"],
    }


def _polygons_from_labels(grid: ElevationGrid, labels: np.ndarray, simplify_m: float) -> dict[int, Any]:
    """Dissolve each label's cells into one (multi)polygon in the grid CRS.

    ``rasterio.features.shapes`` traces the cell squares of each label
    directly (8-connected, so a diagonal run stays one piece), which is the
    union of the squares without building them one by one; the outline is
    then simplified by ``simplify_m`` so the stair-step edge does not
    become a thousand vertices in the GeoJSON.
    """
    from rasterio import features as rio_features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    labels = np.asarray(labels).astype("int32")
    if not np.any(labels > 0):
        return {}
    pieces: dict[int, list] = {}
    for geom, value in rio_features.shapes(
        labels, mask=labels > 0, connectivity=8, transform=grid.transform()
    ):
        pieces.setdefault(int(value), []).append(shape(geom))
    out = {}
    for code, polys in pieces.items():
        merged = unary_union(polys) if len(polys) > 1 else polys[0]
        if simplify_m > 0:
            simplified = merged.simplify(simplify_m, preserve_topology=True)
            if not simplified.is_empty:
                merged = simplified
        out[code] = merged
    return out


def _to_wgs84(grid: ElevationGrid, geom):
    from shapely.ops import transform as shp_transform

    return shp_transform(lambda x, y, z=None: grid.to_lonlat(x, y), geom)


def _geojson_geometry(geom) -> dict[str, Any]:
    from shapely.geometry import mapping

    data = mapping(geom)
    return _jsonable(data)


def _jsonable(value: Any) -> Any:
    """Plain Python types only: numpy scalars unwrapped, NaN to None, tuples to lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _elevation_unit_doubt(dataset: Dataset) -> str | None:
    """A sentence when the file's other lengths say the altitude is in feet.

    Nothing in an altitude column tells its unit: 1 900 is a fair height in
    metres in the foothills and 585 m written in feet on the Prairies. The
    swath width beside it does tell, because a monitor writes every length
    in the one system it was set to, and the same width bands the
    preliminary check reads are used here. The doubt is reported, not
    corrected: the analysis has no business changing a dataset, and
    declaring the units makes a converted copy that analyses right.
    """
    if sch.SWATH not in dataset.df.columns:
        return None
    width = pd.to_numeric(dataset.df[sch.SWATH], errors="coerce").dropna()
    width = width[width > 0]
    if len(width) < 20:
        return None
    median = float(width.median())
    fits = [u for u, (lo, hi) in preflight_mod.LENGTH_HINTS.items() if lo <= median <= hi]
    if fits != ["ft"]:
        return None
    return (
        f"The swath width in this file reads {median:.0f}, which is feet, so the GPS "
        "altitude is almost certainly in feet as well and every height, depth and "
        "volume below is about 3.3 times too large. Declare the file's units as feet "
        "(the preliminary check offers it in one click) and analyse the converted copy."
    )


def _info(text: str) -> dict[str, str]:
    return {"level": "info", "text": text}


def _ok(text: str) -> dict[str, str]:
    return {"level": "ok", "text": text}


def _warning(text: str) -> dict[str, str]:
    return {"level": "warning", "text": text}


# ``_m``, ``_ha``, ``_pct`` and ``_m3`` lived here and wrote a metre, a
# hectare, a percentage and a cubic metre into a sentence. They are gone
# rather than kept as wrappers: each one hard-coded the unit it printed,
# which is the whole of what was wrong, and a wrapper of the same name
# would invite the next sentence to use it. Every quantity now goes
# through :class:`~agrosuite.core.units.Phrase`, which carries the
# reader's unit set and knows what a hectare is called on their screen.


def _count_sentence(n: int, noun: str, singular_verb: str, plural_verb: str) -> str:
    if n == 1:
        return f"One {noun} {singular_verb}"
    return f"{n} {noun}s {plural_verb}"
