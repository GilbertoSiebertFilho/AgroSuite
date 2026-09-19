"""The Machine tab's API: a telemetry log's day, and the crop under its tyres.

``POST /api/machine/analyze`` reads one telemetry dataset — a SoMat log —
with the machine's settings, and optionally a boundary and a yield map from
the same project. What is stored is the summary as the analysis wrote it,
metric; what is sent back is that summary with its sentences written in the
units on screen, and the map layers: the path by activity, the stops, and
the strips the tyres crushed.

The layers and the per-channel statistics are kept in memory for the few
most recent analyses, like the relief's arrays; the summary is saved with
the project and outlives them.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from ... import machine
from ...core import schema as sch
from ...formats import writers
from ...machine import analysis as analysis_mod
from ...machine import synthetic

router = APIRouter(prefix="/api/machine", tags=["machine"])

#: Analyses whose map layers are kept in memory.
MAX_RESULTS = 4

#: Metres the drawn path and strips may move when simplified for the map:
#: a browser has no use for a vertex every half second, and a strip a tyre
#: wide drawn to a tenth of a metre is below what anyone sees on a field. The
#: areas are measured on the full geometry; the export simplifies far less,
#: because in QGIS someone may measure the strips again.
PATH_SIMPLIFY_M = 1.0
STRIPS_SIMPLIFY_M = 0.10
EXPORT_STRIPS_SIMPLIFY_M = 0.01


class AnalyzeRequest(BaseModel):
    """The machine's settings as the interface sends them, metric. ``None``
    leaves the analysis's default; unknown keys are refused."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    boundary_id: str | None = None
    yield_id: str | None = None
    boom_width_m: float | None = None
    tyre_size: str | None = None
    tyre_width_m: float | None = None
    track_width_m: float | None = None
    rear_follows_front: bool = True
    def_tank_l: float | None = None
    fuel_price: float | None = None
    def_price: float | None = None
    crop_price_per_kg: float | None = None
    loss_fraction: float | None = None
    yield_kg_ha: float | None = None
    idle_speed_kmh: float | None = None
    road_speed_kmh: float | None = None


@dataclass
class _Cached:
    result: machine.MachineResult
    layers: dict[str, Any]


_cache: "OrderedDict[str, _Cached]" = OrderedDict()
_lock = threading.Lock()


def _server():
    from agrosuite.app import server as server_mod

    return server_mod


def _entry(dataset_id: str):
    server_mod = _server()
    try:
        return server_mod.state.get(dataset_id)
    except KeyError as exc:
        raise server_mod._fail(str(exc.args[0]) if exc.args else str(exc), 404)


def _remember(dataset_id: str, cached: _Cached) -> None:
    state = _server().state
    with _lock:
        _cache.pop(dataset_id, None)
        _cache[dataset_id] = cached
        for key in list(_cache):
            try:
                state.get(key)
            except KeyError:
                _cache.pop(key, None)
        while len(_cache) > MAX_RESULTS:
            _cache.popitem(last=False)


def _cached(dataset_id: str) -> _Cached:
    entry = _entry(dataset_id)
    with _lock:
        cached = _cache.get(dataset_id)
        if cached is not None:
            _cache.move_to_end(dataset_id)
    if cached is not None:
        return cached
    if "machine" in entry.reports:
        raise _server()._fail(
            f"The machine analysis of '{entry.label}' is no longer in memory: the app keeps the "
            f"{MAX_RESULTS} most recent for their map layers. Run it again to bring them back.", 404)
    raise _server()._fail(
        f"'{entry.label}' has not been analysed. Post its id to /api/machine/analyze.", 404)


# ==========================================================================
# Map layers
# ==========================================================================

def _to_lonlat(crs: str):
    import shapely
    from pyproj import Transformer

    project = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def convert(geom):
        return shapely.transform(geom, lambda xy: np.column_stack(project.transform(xy[:, 0], xy[:, 1])))

    return convert


def _layers(dataset, result: machine.MachineResult,
            strips_simplify_m: float = STRIPS_SIMPLIFY_M) -> dict[str, Any]:
    """The path by activity, the stops and the tyre strips, as GeoJSON."""
    from shapely.geometry import LineString, Point, mapping

    df = dataset.df
    crs = dataset.metric_crs
    if crs is None or not {sch.X, sch.Y}.issubset(df.columns):
        return {"track": _collection([]), "stops": _collection([]), "strips": _collection([])}
    convert = _to_lonlat(crs)
    dt = result.summary["span"]["sample_s"]
    xy = df[[sch.X, sch.Y]].to_numpy(dtype=float)
    times = df[sch.TIMESTAMP].astype(str).to_numpy() if sch.TIMESTAMP in df else None
    fuel = df["fuel_l"].to_numpy(dtype=float) if "fuel_l" in df else np.zeros(len(df))
    distance = df[sch.DISTANCE].to_numpy(dtype=float) if sch.DISTANCE in df else np.zeros(len(df))

    track, stops = [], []
    for a, b, activity in analysis_mod._runs(result.activity):
        props = {
            "activity": activity, "label": analysis_mod.ACTIVITY_LABELS[activity],
            "start": times[a][:19] if times is not None else None,
            "end": times[b - 1][:19] if times is not None else None,
            "duration_s": (b - a) * dt,
            "distance_m": float(np.nansum(distance[a:b])),
            "fuel_l": float(np.nansum(fuel[a:b])),
        }
        points = xy[a:b]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        if activity in ("idle", "off"):
            centre = Point(np.median(points, axis=0))
            stops.append({"type": "Feature", "properties": props, "geometry": mapping(convert(centre))})
        elif len(points) >= 2:
            line = LineString(points).simplify(PATH_SIMPLIFY_M)
            track.append({"type": "Feature", "properties": props, "geometry": mapping(convert(line))})

    strips = []
    if result.footprint is not None and not result.footprint.is_empty:
        shape = result.footprint.simplify(strips_simplify_m)
        t = result.summary.get("trampling", {})
        strips.append({"type": "Feature", "geometry": mapping(convert(shape)),
                       "properties": {"area_ha": t.get("area_ha"), "lost_kg": t.get("lost_kg")}})
    return {"track": _collection(track), "stops": _collection(stops), "strips": _collection(strips)}


def _collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


# ==========================================================================
# Analysis
# ==========================================================================

def _other(dataset_id: str | None, what: str):
    if not dataset_id:
        return None
    try:
        return _server().state.get(dataset_id)
    except KeyError:
        raise _server()._fail(f"The {what} picked is no longer loaded; pick it again.", 404)


@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Analyse a telemetry log and return its summary, in today's units, and
    its map layers."""
    server_mod = _server()
    state = server_mod.state
    entry = _entry(request.dataset_id)
    dataset = entry.dataset
    if dataset.meta.operation != "telemetry":
        raise server_mod._fail(
            f"'{entry.label}' is not a machine log: the Machine tab reads telemetry — a SoMat "
            ".sie file — and this is a {dataset.meta.operation} dataset.")

    settings = request.model_dump(exclude={"dataset_id", "boundary_id", "yield_id"}, exclude_none=True)
    if "crop_price_per_kg" not in settings:
        price = (state.project.get("prices") or {}).get("crop_price")
        if price:
            settings["crop_price_per_kg"] = float(price)

    boundary = boundary_area = lonlat = None
    boundary_entry = _other(request.boundary_id, "boundary")
    if boundary_entry is not None:
        try:
            lonlat, boundary, boundary_area = machine.boundary_polygon(
                boundary_entry.dataset, dataset.metric_crs)
        except ValueError as exc:
            raise server_mod._fail(str(exc))

    yield_kg_ha = yield_source = None
    yield_entry = _other(request.yield_id, "yield map")
    if yield_entry is not None:
        yield_kg_ha, records = machine.mean_yield(yield_entry.dataset, lonlat)
        if yield_kg_ha is None:
            raise server_mod._fail(f"'{yield_entry.label}' holds no yield to average.")
        yield_source = f"yield map '{yield_entry.label}', mean of {records:,} records"

    try:
        result = machine.analyse(dataset, settings, boundary=boundary, boundary_area_ha=boundary_area,
                                 yield_kg_ha=yield_kg_ha, yield_source=yield_source)
    except ValueError as exc:
        raise server_mod._fail(str(exc))

    summary = result.summary
    summary["boundary"] = ({"id": request.boundary_id, "label": boundary_entry.label}
                           if boundary_entry is not None else None)
    summary["yield_map"] = ({"id": request.yield_id, "label": yield_entry.label}
                            if yield_entry is not None else None)
    layers = _layers(dataset, result)
    _remember(request.dataset_id, _Cached(result=result, layers=layers))
    entry.reports["machine"] = summary
    state.touch()
    return {"dataset_id": request.dataset_id,
            "summary": machine.restate(summary, state.display_units),
            "layers": layers}


@router.get("/{dataset_id}")
def summary(dataset_id: str) -> dict[str, Any]:
    """The last analysis of a dataset, said in today's units."""
    entry = _entry(dataset_id)
    report = entry.reports.get("machine")
    if report is None:
        raise _server()._fail(
            f"'{entry.label}' has not been analysed. Post its id to /api/machine/analyze.", 404)
    with _lock:
        cached = _cache.get(dataset_id)
    return {"dataset_id": dataset_id,
            "summary": machine.restate(report, _server().state.display_units),
            "layers": cached.layers if cached is not None else None}


@router.get("/{dataset_id}/layers")
def layers(dataset_id: str) -> dict[str, Any]:
    return _cached(dataset_id).layers


STAT_FIELDS = ["selection", "channel", "unit", "n", "duration_s", "min", "t_min", "max", "t_max",
               "peak_to_peak", "mean", "median", "rms", "std", "variance", "skewness", "kurtosis",
               "crest_factor"]


def _statistics_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=STAT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    return buffer.getvalue()


@router.get("/{dataset_id}/statistics")
def statistics(dataset_id: str) -> dict[str, Any]:
    """The per-channel statistics, InField style, for the whole log and each activity."""
    return {"dataset_id": dataset_id, "rows": _cached(dataset_id).result.statistics}


@router.get("/{dataset_id}/statistics.csv")
def statistics_csv(dataset_id: str) -> Response:
    entry = _entry(dataset_id)
    body = _statistics_csv(_cached(dataset_id).result.statistics or [])
    from .terrain import _slug

    return Response(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{_slug(entry.label)}_statistics.csv"'})


# ==========================================================================
# Export
# ==========================================================================

README = """Machine telemetry — {name}

Written by AgroSuite from the machine log '{source}'.

Files
  activity_track.{ext}  the path, one line per stretch of one activity
                        (field, road), with its start, end, duration,
                        distance (m) and fuel (L)
  stops.{ext}           one point per stop — engine running (idle) or off —
                        with its duration and fuel
  tyre_strips.{ext}     the ground the tyres crossed inside the field, one
                        (multi)polygon; area_ha and lost_kg as attributes
  samples.csv           every sample of the log with its activity and every
                        channel, metric: km/h, m, L, L/h, %
  statistics.csv        per-channel statistics for the whole log and each
                        activity: n, duration, min and max with their times,
                        mean, median, RMS, standard deviation, variance,
                        skewness, kurtosis (3 for a Gaussian), crest factor
  summary.json          the analysis as the app stores it, metric

Coordinates are WGS84 (EPSG:4326).

What the analysis found
{findings}
"""


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector_format: str = "shapefile"


@router.post("/{dataset_id}/export")
def export(dataset_id: str, request: ExportRequest | None = None) -> dict[str, Any]:
    """Write the analysis as vector files, a sample table, the statistics and
    a README, zipped for the download."""
    import geopandas as gpd

    from .terrain import _fresh_dir, _slug

    request = request or ExportRequest()
    server_mod = _server()
    state = server_mod.state
    entry = _entry(dataset_id)
    cached = _cached(dataset_id)
    fmt = request.vector_format.strip().lower()
    if fmt not in ("shapefile", "geojson"):
        raise server_mod._fail("vector_format must be shapefile or geojson.")
    ext = "shp" if fmt == "shapefile" else "geojson"
    driver = "ESRI Shapefile" if fmt == "shapefile" else "GeoJSON"

    out_dir = _fresh_dir(state.exports, f"{_slug(entry.label)}_machine")
    files = []
    layers = dict(cached.layers)
    layers["strips"] = _layers(entry.dataset, cached.result,
                               strips_simplify_m=EXPORT_STRIPS_SIMPLIFY_M)["strips"]
    for key, name in (("track", "activity_track"), ("stops", "stops"), ("strips", "tyre_strips")):
        features = layers[key]["features"]
        if not features:
            continue
        frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        if fmt == "shapefile":
            # Ten characters a field name, in a shapefile.
            frame = frame.rename(columns={"duration_s": "duration", "distance_m": "dist_m"})
        frame.to_file(out_dir / f"{name}.{ext}", driver=driver)
        files.append(f"{name}.{ext}")

    df = entry.dataset.df.copy()
    df.insert(0, "activity", cached.result.activity)
    df.drop(columns=[c for c in (sch.X, sch.Y, sch.VALUE) if c in df], errors="ignore") \
      .to_csv(out_dir / "samples.csv", index=False)
    (out_dir / "statistics.csv").write_text(_statistics_csv(cached.result.statistics or []),
                                            encoding="utf-8")
    report = entry.reports.get("machine") or cached.result.summary
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    said = machine.findings(report, state.display_units)
    (out_dir / "README.txt").write_text(README.format(
        name=entry.label, source=entry.dataset.meta.source_path or entry.label, ext=ext,
        findings="\n".join(f"  - {f['text']}" for f in said)), encoding="utf-8")
    files += ["samples.csv", "statistics.csv", "summary.json", "README.txt"]

    zip_path = state.exports / f"{out_dir.name}.zip"
    bundle_info = writers.bundle([out_dir], zip_path)
    token = state.register_file(zip_path)
    return {"path": str(out_dir), "files": files, "download_url": f"/api/download/{token}",
            "entries": bundle_info["entries"]}


# ==========================================================================
# Demo
# ==========================================================================

@router.post("/demo")
def demo() -> dict[str, Any]:
    """A made-up sprayer day and its field's boundary, so the tab can be
    tried without a logger. The day is written as a real .sie file and read
    back through the same reader a real log goes through."""
    import pandas as pd
    from shapely.geometry import Polygon

    from ...core.dataset import Dataset, DatasetMeta
    from ...formats import registry

    server_mod = _server()
    state = server_mod.state
    path, day = synthetic.demo_log(state.uploads / "sprayer_day_demo.sie")
    log = registry.read_any(path)
    log.meta.name = "Sprayer day (demo)"
    log.meta.extra["machine_truth"] = {k: v for k, v in day.truth.items() if not isinstance(v, np.ndarray)}
    logged = server_mod._register(log, "Sprayer day (demo)", "demo")

    ring = day.boundary_lonlat()
    polygon = Polygon(ring)
    boundary = Dataset(
        pd.DataFrame({sch.LON: [polygon.centroid.x], sch.LAT: [polygon.centroid.y], sch.VALUE: [np.nan]}),
        DatasetMeta(name="Demo field boundary", source_format="demo", operation="boundary",
                    geometry_type="polygon", value_label="Boundary"),
        geometry=[polygon],
    )
    bounded = server_mod._register(boundary, "Demo field boundary", "demo")
    return {"log": logged, "boundary": bounded,
            "suggested": {"boom_width_m": day.truth["boom_m"], "tyre_size": "380/90R46",
                          "track_width_m": 3.048, "def_tank_l": 60.0}}
