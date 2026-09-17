"""Terrain (relief) routes: analyse a dataset's elevation and serve the result.

The analysis itself lives in :mod:`agrosuite.terrain.analysis`; this module
is the HTTP face of it — the summary for the report panel, one PNG per map
layer, the contours and feature outlines as GeoJSON, a profile along a line
drawn on the map, zone datasets the rest of the app can export or join, and
a folder of GeoTIFFs and vector files for QGIS.

A :class:`~agrosuite.terrain.analysis.TerrainResult` holds a dozen
grid-sized arrays — a 400 000-cell grid is some 40 MB — which is too much to
keep for every dataset of a long session, while the interface asks for one
layer at a time and the zones and the export read the arrays again. So the
four most recent results stay in memory, and the JSON summary, which is
small and is what a saved report needs, lives in the session entry's
``reports['terrain']``, where it outlives the arrays. Everything here is
metric in and metric out; the interface converts for display.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ...core import crs as crs_mod
from ...core.units import Phrase
from ...formats import raster as raster_mod
from ...formats import writers
from ...terrain import analysis as analysis_mod
from ...terrain import landforms as lf
from ...terrain import render as render_mod
from ...terrain import yieldrelief
from ...terrain.analysis import LAYER_SPECS, ZONE_KINDS, TerrainOptions, TerrainResult
from ...terrain.contours import contours as contour_lines
from ...terrain.contours import contours_to_geodataframe
from ...terrain.contours import profile as profile_along
from ...terrain.synthetic import synthetic_terrain

router = APIRouter(prefix="/api/terrain", tags=["terrain"])

#: Results kept in memory. Four covers the usual case — the field being
#: looked at, a DEM of the same field, and one or two comparisons — without
#: letting a session that analyses twenty files hold twenty sets of arrays.
MAX_RESULTS = 4

#: How much of the hillshade shows through a coloured layer. The landform
#: map is categorical and its colours carry the meaning, so it takes less.
HILLSHADE_STRENGTH = 0.6
HILLSHADE_STRENGTH_CATEGORICAL = 0.35

#: Stations a profile may be sampled at. A chart a screen wide has a few
#: hundred pixels to draw, and the grid has nothing new to say between two
#: stations closer than half a cell; a request for millions — one typed
#: zero too many, from the interface or from Claude driving the MCP server
#: — would otherwise tie the server up building a body of hundreds of MB.
MAX_PROFILE_STATIONS = 5000

LAYER_KEYS = [spec[0] for spec in LAYER_SPECS]
_PALETTE_BY_KEY = {spec[0]: spec[3] for spec in LAYER_SPECS}
_UNIT_BY_KEY = {spec[0]: spec[2] for spec in LAYER_SPECS}
_LABEL_BY_KEY = {spec[0]: spec[1] for spec in LAYER_SPECS}

#: The GeoTIFFs the export writes, with the sentence the README gives each.
RASTER_EXPORTS: list[tuple[str, str]] = [
    ("elevation", "Elevation, metres, on the datum of the source"),
    ("slope_pct", "Slope, percent (100 x rise / run)"),
    ("aspect_deg", "Aspect, degrees clockwise from north, the direction the ground faces "
                   "downhill; nodata on ground flatter than 0.5 %"),
    ("twi", "Topographic wetness index, dimensionless; higher is wetter"),
    ("hillshade", "Hillshade, 0 (shadow) to 1 (fully lit)"),
    ("landform", "Landform class, integer codes listed below"),
    ("depression_depth", "Depth of water that would pond after the surface is sink-filled, "
                         "metres (0 where none)"),
]

EXPORT_PARTS = ("geotiff", "contours", "features", "drainage")
VECTOR_FORMATS = ("shapefile", "geojson")

#: The origin of a dataset that came out of the cleaning as the kept half.
#: Only that one: ``clean_removed`` is the discards, which is the opposite
#: of a cleaned layer, and every other origin is a file as it arrived.
CLEANED_ORIGINS = ("clean",)

ZONE_LABELS = {
    "landform": "landform",
    "slope_class": "slope class",
    "elevation_bands": "elevation bands",
    "wetness": "wetness",
}


# ==========================================================================
# Request models
# ==========================================================================

class AnalyzeRequest(BaseModel):
    """The terrain options as the interface sends them; ``None`` means
    "choose from the data". Unknown keys are refused so a typo in the
    interface cannot silently fall back to a default."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    cell_m: float | None = None
    smooth_m: float | None = None
    tpi_small_m: float | None = None
    tpi_large_m: float | None = None
    contour_interval_m: float | None = None
    detrend_passes: bool = True
    remove_outliers: bool = True
    min_feature_height_m: float | None = None
    hillshade_azimuth: float = 315.0
    hillshade_altitude: float = 45.0
    min_upstream_ha: float = 1.0


class ProfileRequest(BaseModel):
    """A line across the field, and optionally a layer to read along it.

    ``values_dataset_id`` is left out by every caller that only wants the
    ground, and the response then has exactly the shape it always had.

    ``units`` is the reader's unit set, and it reaches only the one
    sentence the answer carries (``values_meta.note``); every number stays
    metric.
    """

    points: list[list[float]]
    n: int = 200
    values_dataset_id: str | None = None
    value_column: str | None = None
    units: dict[str, Any] | None = None


class YieldReliefRequest(BaseModel):
    """The value layer to read against an analysed relief.

    ``yield_dataset_id`` may be the analysed dataset itself — a yield map
    whose own GPS altitude was gridded, which is the common case — or a
    separate layer, a DEM having been analysed with the yield map loaded
    beside it. Unknown keys are refused, as on the analysis itself, so a
    typo cannot silently fall back to a default.
    """

    model_config = ConfigDict(extra="forbid")

    yield_dataset_id: str
    bands: int = yieldrelief.DEFAULT_BANDS
    value_column: str | None = None
    #: The unit set on screen, the shape of ``UNIT_PRESETS['canada']``. It
    #: reaches the findings only — every number in the answer stays metric
    #: — because a sentence with its numbers written into it cannot be
    #: restated in another unit at display time without re-deciding what it
    #: says. The interface posts again when the reader switches preset,
    #: which is cheaper than it sounds: the grid is already in memory.
    units: dict[str, Any] | None = None


class ZonesRequest(BaseModel):
    by: str
    bands: int = 4
    kind: str = "points"


class TerrainExportRequest(BaseModel):
    include: list[str] = Field(default_factory=lambda: list(EXPORT_PARTS))
    vector_format: str = "shapefile"


# ==========================================================================
# The result cache
# ==========================================================================

@dataclass
class _Cached:
    """One analysis kept in memory, with the map pieces derived from it once."""

    result: TerrainResult
    #: Leaflet ``[[south, west], [north, east]]`` of the web-mercator image box.
    bounds: list[list[float]]
    layers: list[dict[str, Any]] | None = None
    features: dict[str, Any] | None = None


_cache: "OrderedDict[str, _Cached]" = OrderedDict()
_lock = threading.Lock()


def _server():
    # Imported inside the handlers: the server imports this package at
    # start-up, so a top-level import here would be circular.
    from agrosuite.app import server as server_mod

    return server_mod


def _missing(exc: KeyError) -> str:
    """The sentence the session put in its KeyError.

    ``str()`` of a KeyError wraps the argument in quotation marks, and the
    interface would show the 404 as ``"Dataset 'x' is not loaded..."``,
    quotes included.
    """
    return str(exc.args[0]) if exc.args else str(exc)


def _entry(dataset_id: str):
    server_mod = _server()
    try:
        return server_mod.state.get(dataset_id)
    except KeyError as exc:
        raise server_mod._fail(_missing(exc), 404)


def _remember(dataset_id: str, result: TerrainResult) -> _Cached:
    cached = _Cached(result=result, bounds=_leaflet_bounds(result.grid))
    state = _server().state
    with _lock:
        _cache.pop(dataset_id, None)
        _cache[dataset_id] = cached
        # A dataset removed from the session takes its arrays with it.
        for key in list(_cache):
            try:
                state.get(key)
            except KeyError:
                _cache.pop(key, None)
        while len(_cache) > MAX_RESULTS:
            _cache.popitem(last=False)
    return cached


def _cached(dataset_id: str) -> _Cached:
    """The analysis of a dataset, or a 404 that says what to do."""
    server_mod = _server()
    try:
        entry = server_mod.state.get(dataset_id)
    except KeyError as exc:
        # Deleted from the session: its arrays go now, not at the next sweep.
        with _lock:
            _cache.pop(dataset_id, None)
        raise server_mod._fail(_missing(exc), 404)
    with _lock:
        cached = _cache.get(dataset_id)
        if cached is not None:
            _cache.move_to_end(dataset_id)
    if cached is not None:
        return cached
    if "terrain" in entry.reports:
        raise server_mod._fail(
            f"The terrain analysis of '{entry.label}' is no longer in memory: the app "
            f"keeps the {MAX_RESULTS} most recent to hold the map layers. Run the "
            "analysis again to bring it back.",
            404,
        )
    raise server_mod._fail(
        f"Run the terrain analysis first: '{entry.label}' has not been analysed. "
        "Post its id to /api/terrain/analyze.",
        404,
    )


def _leaflet_bounds(grid) -> list[list[float]]:
    """The corners of the layer PNGs, without rendering one.

    Every PNG of a grid shares the box :func:`render.to_web_mercator`
    builds, so the corners are computed once here the same way it does.
    """
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = render_mod.mercator_bounds(grid)
    to_ll = Transformer.from_crs(render_mod.WEB_MERCATOR, crs_mod.WGS84, always_xy=True)
    west, south = to_ll.transform(xmin, ymin)
    east, north = to_ll.transform(xmax, ymax)
    return [[float(south), float(west)], [float(north), float(east)]]


# ==========================================================================
# Colour stretch shared by the legend and the PNG
# ==========================================================================

def _stretch(key: str, values: np.ndarray, palette: str) -> tuple[float, float]:
    """The ``(vmin, vmax)`` a layer is drawn with when the caller sets none.

    The percentile stretch of :func:`render.default_range` suits elevation
    and slope, but not a layer that is mostly one value: ponding depth is 0
    on most of a field, so its 2nd and 98th percentiles are both 0 and the
    stretch would widen to -0.5..0.5 and colour the whole field mid-ramp.
    Hillshade is a fraction with fixed meaning, and flow accumulation runs
    on a log ramp where the full range is the point.
    """
    finite = values[np.isfinite(values)]
    if key == "hillshade":
        return (0.0, 1.0)
    if key == "depression_depth":
        positive = finite[finite > 0]
        hi = float(np.percentile(positive, 98)) if positive.size else 0.1
        return (0.0, max(hi, 0.01))
    if key == "flow_acc":
        return (1.0, float(finite.max()) if finite.size and finite.max() > 1 else 2.0)
    return render_mod.default_range(values, palette)


def _legend(key: str, values: np.ndarray) -> dict[str, Any] | None:
    if key == "landform":
        legend = render_mod.legend_for(categorical=render_mod.LANDFORM_COLORS)
        for cls in legend["classes"]:
            spec = lf.CLASS_BY_CODE.get(cls["code"])
            if spec:
                cls["key"] = spec["key"]
                cls["label"] = spec["label"]
        return legend
    palette = _PALETTE_BY_KEY[key]
    try:
        vmin, vmax = _stretch(key, values, palette)
    except ValueError:
        # Nothing finite inside the field: the layer draws transparent and
        # has no range to label.
        return None
    return render_mod.legend_for(palette, vmin, vmax)


def _shade_for_blend(result: TerrainResult) -> np.ndarray:
    """The hillshade scaled so flat ground keeps the palette's true colour.

    The raw hillshade of flat ground under a 45-degree sun is 0.71, which
    would darken a flat field by a fifth at the default strength and make
    every legend colour a lie. Dividing by the flat value puts flat ground
    at 1 and only the sides turned away from the sun darken; the sunlit
    sides are clipped to 1 rather than brightened, because a colour lighter
    than the legend's is as misleading as one darker.
    """
    altitude = float(result.options.hillshade_altitude or 45.0)
    flat = math.sin(math.radians(altitude))
    if not flat > 0:
        flat = 1.0
    with np.errstate(invalid="ignore"):
        return np.clip(result.layers["hillshade"] / flat, 0.0, 1.0)


def _optional_float(raw: str | None, name: str) -> float | None:
    """A query number that may be absent or an empty string (the interface
    sends ``vmin=&vmax=`` when the boxes are blank)."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        raise _server()._fail(f"'{name}' must be a number, not '{raw}'.")
    if not math.isfinite(value):
        raise _server()._fail(f"'{name}' must be a finite number.")
    return value


# ==========================================================================
# Analysis and summary
# ==========================================================================

@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run the terrain analysis on a loaded dataset and return its summary.

    What is stored is the summary as the analyser wrote it: metric
    numbers and metric sentences, like everything else this app keeps.
    What is sent back is the same summary said in the units on screen —
    :func:`agrosuite.terrain.analysis.restate` writes the sentences again
    and touches no number — so the reader never sees a sentence in one
    unit beside a table in another, and a later change of units needs
    only another read, not another analysis.
    """
    server_mod = _server()
    entry = _entry(request.dataset_id)
    _refuse_zones(entry)
    options = TerrainOptions(**request.model_dump(exclude={"dataset_id"}))
    units = server_mod.state.display_units
    try:
        # The units reach the analysis itself for one thing only: the notes
        # the grid builder and the raster reader write while the file is
        # being read, which cannot be written again without reading it again.
        result = analysis_mod.analyze(entry.dataset, options, units)
        summary = result.summary()
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)
    _remember(request.dataset_id, result)
    # Replaces any earlier report: an analysis re-run with other options is
    # the one the user now wants to read.
    entry.reports["terrain"] = summary
    server_mod.state.touch()
    return {
        "dataset_id": request.dataset_id,
        "summary": analysis_mod.restate(summary, units),
    }


def _refuse_zones(entry) -> None:
    """A zone dataset is an output of the analysis, not an input to it.

    Its points are grid-cell centres and its polygons carry one row each;
    re-gridded as a GPS track they would give back a coarser copy of the
    relief they were cut from, under the label of a new analysis. The
    analyser itself refuses too; here the message can also name the source
    dataset's id, which is what the caller has to post.
    """
    extra = entry.dataset.meta.extra or {}
    if not extra.get("zones_by"):
        return
    source_id = extra.get("terrain_source_id")
    source = extra.get("terrain_source") or "its source"
    hint = ""
    if source_id:
        try:
            source_entry = _server().state.get(str(source_id))
        except KeyError:
            hint = " (that dataset is no longer loaded: open it again)"
        else:
            source = source_entry.label
            hint = f" (dataset {source_id}: post it to /api/terrain/analyze)"
    raise _server()._fail(
        f"These are terrain zones derived from {source!r}; analyse {source!r} instead{hint}."
    )


@router.get("/{dataset_id}")
def summary(dataset_id: str) -> dict[str, Any]:
    """The summary of the last analysis of a dataset, said in today's units.

    The stored summary is metric, sentences included; what comes back here
    is the same summary with its prose written again in the unit set the
    reader is working in. No number differs between the two.
    """
    entry = _entry(dataset_id)
    units = _server().state.display_units
    with _lock:
        cached = _cache.get(dataset_id)
    if cached is not None:
        return {
            "dataset_id": dataset_id,
            "summary": analysis_mod.restate(cached.result.summary(), units),
        }
    # The summary outlives the arrays: it is what a saved report shows.
    report = entry.reports.get("terrain")
    if report is not None:
        return {"dataset_id": dataset_id, "summary": analysis_mod.restate(report, units)}
    raise _server()._fail(
        f"Run the terrain analysis first: '{entry.label}' has not been analysed. "
        "Post its id to /api/terrain/analyze.",
        404,
    )


# ==========================================================================
# Map layers
# ==========================================================================

@router.get("/{dataset_id}/layers")
def layers(dataset_id: str) -> dict[str, Any]:
    """Every map layer with its image URL, corners and legend."""
    cached = _cached(dataset_id)
    if cached.layers is None:
        result = cached.result
        out = []
        for meta in result.layer_meta():
            key = meta["key"]
            out.append({
                **meta,
                "url": f"/api/terrain/{dataset_id}/layer/{key}.png",
                "bounds": cached.bounds,
                "legend": _legend(key, result.layers[key]),
            })
        cached.layers = out
    return {"layers": cached.layers}


@router.get("/{dataset_id}/layer/{key}.png")
def layer_png(
    dataset_id: str,
    key: str,
    hillshade: bool = False,
    vmin: str | None = None,
    vmax: str | None = None,
) -> Response:
    """One layer drawn as a web-mercator PNG for the map.

    ``hillshade=1`` blends the relief into the colours of every layer but
    the hillshade itself — at :data:`HILLSHADE_STRENGTH` for continuous
    layers and more lightly for the landform classes, whose flat colours
    are the legend. ``vmin`` / ``vmax`` override the stretch the legend of
    ``/layers`` was built with; either may be blank.
    """
    server_mod = _server()
    cached = _cached(dataset_id)
    result = cached.result
    if key not in result.layers:
        raise server_mod._fail(
            f"There is no '{key}' layer. The layers are: {', '.join(LAYER_KEYS)}.", 404
        )
    values = result.layers[key]
    lo = _optional_float(vmin, "vmin")
    hi = _optional_float(vmax, "vmax")
    shade = _shade_for_blend(result) if hillshade and key != "hillshade" else None
    try:
        if key == "landform":
            png, _bounds, _legend_used = render_mod.render_png(
                result.grid, values, categorical=render_mod.LANDFORM_COLORS,
                hillshade=shade, hillshade_strength=HILLSHADE_STRENGTH_CATEGORICAL,
            )
        else:
            palette = _PALETTE_BY_KEY[key]
            if lo is None or hi is None:
                default_lo, default_hi = _stretch(key, values, palette)
                lo = default_lo if lo is None else lo
                hi = default_hi if hi is None else hi
            png, _bounds, _legend_used = render_mod.render_png(
                result.grid, values, palette=palette, vmin=lo, vmax=hi,
                hillshade=shade, hillshade_strength=HILLSHADE_STRENGTH,
            )
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)
    # The image changes whenever the analysis is re-run with other options,
    # so the browser must not keep an old one under the same URL.
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


# ==========================================================================
# Contours, features, profile
# ==========================================================================

@router.get("/{dataset_id}/contours")
def contours(dataset_id: str, interval_m: str | None = None) -> dict[str, Any]:
    """The contour lines as GeoJSON; an ``interval_m`` redraws them at that step."""
    server_mod = _server()
    result = _cached(dataset_id).result
    interval = _optional_float(interval_m, "interval_m")
    if interval is None or result.level:
        # A level field has no contours at any interval: redrawing it at
        # 0.1 m would trace the rounding noise the analysis refused to draw.
        features = result.contours.get("features", [])
        used = result.contour_interval_m
    else:
        try:
            features, used = contour_lines(
                result.grid, interval_m=interval,
                units=_server().state.display_units,
            )
        except ValueError as exc:
            raise server_mod._fail(str(exc), 400)
    return {
        "type": "FeatureCollection",
        "features": features,
        "interval_m": None if used is None or not math.isfinite(used) else float(used),
        "count": len(features),
    }


@router.get("/{dataset_id}/features")
def features(dataset_id: str) -> dict[str, Any]:
    """Hills, lows and closed depressions as polygons, plus the drainage lines."""
    cached = _cached(dataset_id)
    if cached.features is None:
        cached.features = cached.result.feature_collection()
    return {**cached.features, "drainage": cached.result.drainage}


@router.post("/{dataset_id}/profile")
def profile(dataset_id: str, request: ProfileRequest) -> dict[str, Any]:
    """Elevation along a line drawn on the map, and optionally a value with it.

    With no ``values_dataset_id`` the answer is exactly what it has always
    been. With one, two keys are added — ``values``, aligned station for
    station with ``distance_m``, and ``values_meta`` — and nothing else
    changes, so a caller that ignores them reads the same response.
    """
    server_mod = _server()
    result = _cached(dataset_id).result
    # The request may name its own unit set; otherwise the one the reader is
    # working in, which the interface has already told the server about.
    units = request.units or server_mod.state.display_units
    say = Phrase(units)
    if request.n > MAX_PROFILE_STATIONS:
        raise server_mod._fail(
            f"A profile of {request.n:,} stations is more than a chart can show: ask "
            f"for at most {MAX_PROFILE_STATIONS:,}. The line is sampled evenly, so "
            f"one station per {say.length(result.grid.cell / 2, None)} (half a grid "
            "cell) already reads every cell it crosses."
        )
    try:
        line = profile_along(result.grid, request.points, n=request.n)
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)
    if not request.values_dataset_id:
        return line

    values_entry = _entry(request.values_dataset_id)
    try:
        series, meta = yieldrelief.profile_values(
            result.grid, line, values_entry.dataset,
            value_column=request.value_column, values_label=values_entry.label,
            units=units,
        )
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)
    line["values"] = series
    line["values_meta"] = meta
    return line


# ==========================================================================
# Yield against the relief
# ==========================================================================

@router.post("/{dataset_id}/yield")
def yield_against_relief(dataset_id: str, request: YieldReliefRequest) -> dict[str, Any]:
    """Read a value layer against the relief of ``dataset_id``.

    The relief has to be in memory, not merely reported: the comparison
    samples the grid itself under every point, which the saved summary
    cannot do. So an evicted analysis is refused with the sentence that
    says to run it again, exactly as the map layers are.

    The result is kept on the analysed dataset's entry, beside its terrain
    report, so the printed page carries it and a reload brings it back.
    What is stored carries metric sentences, like every other report in
    this app, and what is sent back carries them in the unit set this call
    was given — the one the request named, or the one the reader is
    working in. The printed page writes them again in whatever units it is
    asked for, and so does the next read after the picker changes.
    """
    server_mod = _server()
    entry = _entry(dataset_id)
    result = _cached(dataset_id).result
    values_entry = _entry(request.yield_dataset_id)
    # Cleaning produces a new dataset rather than changing one, so a layer
    # that has been cleaned is a layer whose origin is the cleaning.
    cleaned = values_entry.origin in CLEANED_ORIGINS
    units = request.units or server_mod.state.display_units
    try:
        summary = yieldrelief.analyse(
            result, values_entry.dataset,
            terrain_id=dataset_id, terrain_label=entry.label,
            values_id=request.yield_dataset_id, values_label=values_entry.label,
            bands=request.bands, value_column=request.value_column,
            cleaned=cleaned,
        )
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)
    entry.reports["terrain_yield"] = summary
    server_mod.state.touch()
    return {**summary, "findings": yieldrelief.findings(summary, units)}


# ==========================================================================
# Zones
# ==========================================================================

@router.post("/{dataset_id}/zones")
def zones(dataset_id: str, request: ZonesRequest) -> dict[str, Any]:
    """Turn the relief into a zone dataset the rest of the app can use."""
    server_mod = _server()
    entry = _entry(dataset_id)
    result = _cached(dataset_id).result
    if request.by not in ZONE_KINDS:
        raise server_mod._fail(
            f"Zones can be made by {', '.join(ZONE_KINDS)}; '{request.by}' is not one of them."
        )
    if request.kind not in ("points", "polygons"):
        raise server_mod._fail(
            f"kind must be 'points' (one per grid cell) or 'polygons' (one per zone), "
            f"not '{request.kind}'."
        )
    try:
        if request.kind == "points":
            dataset = analysis_mod.zones_points(result, request.by, bands=request.bands)
        else:
            dataset = analysis_mod.zones_polygons(result, request.by, bands=request.bands)
    except ValueError as exc:
        raise server_mod._fail(str(exc), 400)

    label = f"Terrain zones ({ZONE_LABELS[request.by]}) — {entry.label}"
    # The id of the source, so a later request to analyse the zones can
    # point back at the dataset that should be analysed instead.
    dataset.meta.extra["terrain_source_id"] = dataset_id
    # No parent: the zones are a new layer beside the source, not a version
    # of it. Passing the source as parent would move its project role onto
    # the zones and leave the yield map with none.
    summary = server_mod._register(dataset, label, "terrain_zones")
    # The preliminary check files the zones under 'terrain' because they carry
    # elevation, but they are an output of the analysis, not a layer of the
    # field: like the joined table, they start with no role, and the user
    # gives them one (a plan, say) when they are to go to the monitor.
    server_mod.state.project["roles"].pop(summary["id"], None)
    server_mod.state.get(summary["id"]).role = None
    summary["role"] = None
    server_mod.state.touch()

    zone_labels = dataset.meta.extra.get("zone_labels", {})
    return {
        "dataset": summary,
        "zone_labels": {str(code): text for code, text in zone_labels.items()},
        "by": request.by,
        "kind": request.kind,
        "bands": request.bands if request.by == "elevation_bands" else None,
        "source_id": dataset_id,
    }


# ==========================================================================
# Export
# ==========================================================================

@router.post("/{dataset_id}/export")
def export(dataset_id: str, request: TerrainExportRequest) -> dict[str, Any]:
    """Write the analysis as GeoTIFFs and vector files for QGIS, zipped."""
    server_mod = _server()
    entry = _entry(dataset_id)
    result = _cached(dataset_id).result

    include = [part.strip().lower() for part in request.include]
    unknown = sorted(set(include) - set(EXPORT_PARTS))
    if unknown:
        raise server_mod._fail(
            f"Unknown export part(s) {', '.join(unknown)}; choose from "
            f"{', '.join(EXPORT_PARTS)}."
        )
    if not include:
        raise server_mod._fail(
            f"Nothing to export: include at least one of {', '.join(EXPORT_PARTS)}."
        )
    vector_format = request.vector_format.strip().lower()
    if vector_format not in VECTOR_FORMATS:
        raise server_mod._fail(
            f"vector_format must be one of {', '.join(VECTOR_FORMATS)}, "
            f"not '{request.vector_format}'."
        )

    state = server_mod.state
    out_dir = _fresh_dir(state.exports, f"{_slug(entry.label)}_terrain")
    written: list[dict[str, str]] = []
    notes: list[str] = []
    try:
        if "geotiff" in include:
            written.extend(_write_rasters(result, out_dir))
        if "contours" in include:
            written.extend(_write_contours(result, out_dir, vector_format, notes))
        if "features" in include:
            written.extend(_write_collection(
                result.feature_collection(), out_dir / "features", vector_format,
                "Hills, lows and closed depressions as polygons; 'kind' says which "
                "and the other columns are the summary's numbers", notes,
                empty_note="No hill, low or closed depression was found, so there is "
                           "no features file.",
            ))
        if "drainage" in include:
            written.extend(_write_collection(
                result.drainage, out_dir / "drainage", vector_format,
                "Drainage lines where the upstream area exceeds the threshold; "
                "'index' and 'length_m'", notes,
                empty_note="No drainage line reaches the upstream-area threshold, so "
                           "there is no drainage file.",
            ))
    except Exception as exc:
        raise server_mod._fail(f"The terrain export failed: {exc}")

    readme = out_dir / "README.txt"
    readme.write_text(
        _readme(result, entry.label, written, notes, _server().state.display_units),
        encoding="utf-8",
    )
    written.append({"name": readme.name, "what": "This description"})

    zip_path = state.exports / f"{out_dir.name}.zip"
    bundle_info = writers.bundle([out_dir], zip_path)
    token = state.register_file(zip_path)
    return {
        "path": str(out_dir),
        "zip_path": str(zip_path),
        "filename": zip_path.name,
        "download_url": f"/api/download/{token}",
        "files": [item["name"] for item in written],
        "entries": bundle_info["entries"],
        "notes": notes,
    }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] or "terrain"


def _fresh_dir(base: Path, name: str) -> Path:
    """A folder that does not exist yet: a second export of the same field
    sits beside the first rather than over it."""
    candidate = base / name
    n = 2
    while candidate.exists():
        candidate = base / f"{name}_{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def _write_rasters(result: TerrainResult, out_dir: Path) -> list[dict[str, str]]:
    grid = result.grid
    written = []
    common = {
        "source": result.source_name,
        "cell_m": f"{grid.cell:g}",
        "written_by": "AgroSuite terrain analysis",
    }
    for key, what in RASTER_EXPORTS:
        path = out_dir / f"{key}.tif"
        values = result.layers[key]
        tags = {**common, "layer": _LABEL_BY_KEY[key], "unit": _UNIT_BY_KEY[key] or "none"}
        if key == "landform":
            raster_mod.write_geotiff_int(
                grid, path, values, nodata=255, dtype="uint8", tags=tags,
                labels={c["code"]: c["label"] for c in lf.CLASSES},
            )
        else:
            raster_mod.write_geotiff(grid, path, values, tags=tags)
        written.append({"name": path.name, "what": what})
    return written


def _write_contours(
    result: TerrainResult, out_dir: Path, vector_format: str, notes: list[str]
) -> list[dict[str, str]]:
    features = result.contours.get("features", [])
    if not features:
        # A level field has no interval at all (None); a flat one may have
        # an interval that no level of falls between its extremes.
        notes.append("The relief is too flat for a contour line at the chosen interval, "
                     "so there is no contours file.")
        return []
    what = (
        f"Contour lines every {result.contour_interval_m:g} m; 'level_m', 'index' "
        "(rank of the level) and 'major' (1 on every fifth level)"
    )
    if vector_format == "geojson":
        path = out_dir / "contours.geojson"
        writers.write_geojson(result.contours, path)
    else:
        path = out_dir / "contours.shp"
        gdf = contours_to_geodataframe(features)
        # DBF has no boolean: 'major' travels as 0/1.
        gdf["major"] = gdf["major"].astype("int32")
        gdf.to_file(path, driver="ESRI Shapefile")
    return [{"name": path.name, "what": what}]


def _write_collection(
    collection: dict[str, Any],
    stem: Path,
    vector_format: str,
    what: str,
    notes: list[str],
    empty_note: str,
) -> list[dict[str, str]]:
    """A GeoJSON FeatureCollection written as the chosen vector format."""
    features = collection.get("features", [])
    if not features:
        notes.append(empty_note)
        return []
    if vector_format == "geojson":
        path = stem.with_suffix(".geojson")
        writers.write_geojson(collection, path)
        return [{"name": path.name, "what": what}]

    import geopandas as gpd

    path = stem.with_suffix(".shp")
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs_mod.WGS84)
    columns = [c for c in gdf.columns if c != gdf.geometry.name]
    for column in columns:
        series = gdf[column]
        if series.dtype != object:
            continue
        present = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if present and all(isinstance(v, (bool, np.bool_)) for v in present):
            # DBF has no boolean and a column mixing True and None will not
            # write: 1/0 with the gaps left empty.
            gdf[column] = series.map(
                lambda v: np.nan if v is None or (isinstance(v, float) and math.isnan(v))
                else float(bool(v))
            ).astype("float64")
        else:
            gdf[column] = series.map(lambda v: "" if v is None else str(v))
    mapping, warnings = writers.safe_field_names(columns)
    gdf = gdf.rename(columns=mapping)
    for warning in warnings:
        notes.append(f"{path.name}: {warning}")
    gdf.to_file(path, driver="ESRI Shapefile")
    return [{"name": path.name, "what": what}]


def _readme(
    result: TerrainResult,
    label: str,
    written: list[dict[str, str]],
    notes: list[str],
    units: dict[str, Any] | None = None,
) -> str:
    """The README beside the exported layers.

    The findings are the ones the reader saw on screen, written in the
    units they work in — the same sentences, not a metric copy of them.
    The lines above them describe the files themselves: a GeoTIFF's cell
    size and extent are in the units of its own CRS, which is metric, and
    they say so by naming the CRS.
    """
    summary = analysis_mod.restate(result.summary(), units)
    grid = summary["grid"]
    source = summary["source"]
    lines = [
        f"Terrain analysis of {label}",
        f"Written by AgroSuite on {_dt.date.today().isoformat()}",
        "",
        f"Source: {'elevation raster ' + str(source.get('dem_path')) if source['kind'] == 'dem' else 'GPS altitude of ' + str(source['points_used']) + ' points'}",
        f"Grid: {grid['rows']} x {grid['cols']} cells of {grid['cell_m']:g} m in {grid['crs']}, "
        f"{grid['area_ha']:.1f} ha",
        f"Character: {summary['character']['label']} — {summary['character']['why']}",
        "",
        "Files",
        "-----",
    ]
    width = max((len(item["name"]) for item in written), default=10) + 2
    for item in written:
        lines.append(f"  {item['name']:<{width}}{item['what']}")
    lines += [
        "",
        "Every GeoTIFF is on the grid above, one band, float32 with nodata -9999 "
        "outside the field, except landform.tif, which is uint8 with nodata 255. "
        f"Vector files are in WGS84 (EPSG:4326).",
        "",
        "Landform codes (landform.tif)",
        "-----------------------------",
    ]
    for cls in lf.CLASSES:
        lines.append(f"  {cls['code']}  {cls['label']}")
    if notes:
        lines += ["", "Notes", "-----"]
        lines += [f"  - {note}" for note in notes]
    lines += ["", "Findings", "--------"]
    for finding in summary["findings"]:
        lines.append(f"  [{finding['level']}] {finding['text']}")
    lines.append("")
    return "\n".join(lines)


# ==========================================================================
# Demo
# ==========================================================================

@router.post("/demo")
def demo() -> dict[str, Any]:
    """Load the synthetic field of known relief, so the analyser can be tried
    without a file. The truth about what was put there travels in the
    dataset's metadata for anyone who wants to compare."""
    server_mod = _server()
    dataset, truth = synthetic_terrain()
    dataset.meta.extra["terrain_truth"] = _public_truth(truth)
    return server_mod._register(dataset, "Terrain demo", "demo")


def _public_truth(truth: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe part of the synthetic truth: where each feature is."""

    def pair(value) -> list[float]:
        return [float(value[0]), float(value[1])]

    hills = [
        {
            "summit_lonlat": pair(h["summit_lonlat"]),
            "height_m": float(h["height_m"]),
            "radius_m": float(h["radius_m"]),
            "summit_z_m": float(h["summit_z_m"]),
        }
        for h in truth.get("hills") or []
    ]
    valley = truth.get("valley")
    depression = truth.get("depression")
    tilt = truth.get("tilt") or {}
    return {
        "area_ha": float(truth.get("area_ha", 0.0)),
        "tilt": {
            "gradient_pct": 100.0 * float(tilt.get("gradient", 0.0)),
            "direction_deg": float(tilt.get("direction_deg", 0.0)),
            "fall_m": float(tilt.get("fall_m", 0.0)),
        },
        "hills": hills,
        "valley": None if valley is None else {
            "start_lonlat": pair(valley["start_lonlat"]),
            "end_lonlat": pair(valley["end_lonlat"]),
            "depth_m": float(valley["depth_m"]),
            "half_width_m": float(valley["half_width_m"]),
        },
        "depression": None if depression is None else {
            "bottom_lonlat": pair(depression["bottom_lonlat"]),
            "centre_lonlat": pair(depression["centre_lonlat"]),
            "closed_depth_m": float(depression["closed_depth_m"]),
            "spill_z_m": float(depression["spill_z_m"]),
            "volume_m3": float(depression["volume_m3"]),
            "area_m2": float(depression["area_m2"]),
        },
        "noise_m": float(truth.get("noise_m", 0.0)),
    }
