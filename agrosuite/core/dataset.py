"""AgroSuite's canonical data model.

A :class:`Dataset` is the single representation every input format converges
to — shapefile, monitor CSV, ISOXML, Augmenta GeoJSON. Internally it holds a
``pandas.DataFrame`` rather than a ``GeoDataFrame``, because the app's heavy
operations (cleaning filters, neighbourhoods, grid joins) are vectorised over
projected ``x``/``y`` columns; shapely geometry is materialised only when it
is genuinely needed — clipping to a boundary, exporting, computing overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from . import crs as crs_mod
from . import schema as sch
from . import units as units_mod

#: Operations the app recognizes.
OPERATIONS = (
    "harvest",       # harvest / yield map
    "application",   # as-applied fertilizer or crop protection
    "planting",      # seeding / planting
    "prescription",  # variable rate prescription (Rx)
    "boundary",      # field boundary
    "guidance",      # guidance lines
    "vigor",         # vegetation index / Augmenta
    "soil",          # soil sampling
    "elevation",     # elevation raster (DEM), for the terrain analyser
    "unknown",
)

OPERATION_LABELS = {
    "harvest": "Harvest (yield)",
    "application": "Application (as-applied)",
    "planting": "Seeding / planting",
    "prescription": "Prescription (Rx)",
    "boundary": "Field boundary",
    "guidance": "Guidance lines",
    "vigor": "Vigour / vegetation index",
    "soil": "Soil sampling",
    "elevation": "Elevation (DEM)",
    "unknown": "Not identified",
}


@dataclass
class DatasetMeta:
    """Provenance and interpretation metadata for a dataset."""

    name: str = "dataset"
    source_path: str = ""
    source_format: str = "unknown"
    brand: str = "unknown"
    brand_label: str = "Unknown"
    operation: str = "unknown"
    crop: str | None = None
    field_name: str | None = None
    value_label: str = "Value"
    value_unit: str = ""
    source_value_unit: str = ""
    geometry_type: str = "point"
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operation_label"] = OPERATION_LABELS.get(self.operation, self.operation)
        return data


class Dataset:
    """A normalized set of georeferenced points (or polygons).

    Parameters
    ----------
    df:
        Table with at least ``lon`` and ``lat`` columns in WGS84.
    meta:
        Provenance metadata.
    geometry:
        Optional list of shapely geometries aligned with ``df`` — used when
        the source is polygonal (boundary, prescription grid).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        meta: DatasetMeta | None = None,
        geometry: list | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.meta = meta or DatasetMeta()
        self.geometry = list(geometry) if geometry is not None else None
        self.metric_crs: str | None = None
        self._prepare()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        """Ensure types, projected coordinates and derived fields."""
        for col in sch.NUMERIC_COLUMNS:
            if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if sch.TIMESTAMP in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[sch.TIMESTAMP]):
                self.df[sch.TIMESTAMP] = pd.to_datetime(
                    self.df[sch.TIMESTAMP], errors="coerce", format="mixed"
                )

        if sch.LON in self.df.columns and sch.LAT in self.df.columns:
            self.project()

    def project(self, target: str | None = None) -> None:
        """Fill ``x``/``y`` in a metric CRS (automatic UTM by default)."""
        from pyproj import Transformer

        lon = self.df[sch.LON].to_numpy(dtype="float64", na_value=np.nan)
        lat = self.df[sch.LAT].to_numpy(dtype="float64", na_value=np.nan)
        if not np.isfinite(lon).any():
            return
        self.metric_crs = target or crs_mod.pick_metric_crs(lon, lat)
        transformer = Transformer.from_crs(crs_mod.WGS84, self.metric_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        self.df[sch.X] = x
        self.df[sch.Y] = y

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    def ensure_derived(self, default_swath_m: float | None = None) -> None:
        """Compute distance, speed, heading and passes when they are missing.

        Many monitor CSVs carry only position and value. Without speed and
        swath width no serious cleaning filter works, so those quantities are
        reconstructed from the track itself.

        An elevation layer — the cells of a DEM, or the zones the terrain
        analyser derives from one — is not a track. A distance, heading and
        pass number worked out from the order of its rows would be numbers
        about nothing, and they would travel into every export and monitor
        package made from it, so none is added.
        """
        if self.meta.operation == "elevation":
            return
        self.sort_by_time()
        n = len(self.df)
        if n == 0:
            return

        has_xy = sch.X in self.df.columns and sch.Y in self.df.columns
        if has_xy:
            x = self.df[sch.X].to_numpy(dtype="float64", na_value=np.nan)
            y = self.df[sch.Y].to_numpy(dtype="float64", na_value=np.nan)
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            step = np.hypot(dx, dy)
            step[0] = step[1] if n > 1 else 0.0

            if sch.DISTANCE not in self.df.columns:
                self.df[sch.DISTANCE] = step

            if sch.HEADING not in self.df.columns:
                heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
                if n > 1:
                    heading[0] = heading[1]
                self.df[sch.HEADING] = heading

            dt = self._time_delta_seconds()
            if sch.SPEED not in self.df.columns and dt is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    speed = np.where(dt > 0, step / dt * 3.6, np.nan)
                self.df[sch.SPEED] = pd.Series(speed).ffill().bfill().to_numpy()
                self.meta.notes.append(
                    "Speed reconstructed from the track and the timestamps."
                )

        if sch.SPEED in self.df.columns:
            unit = units_mod.guess_speed_unit(self.df[sch.SPEED])
            if unit != "km/h":
                self.df[sch.SPEED] = units_mod.speed_to_kmh(self.df[sch.SPEED], unit)
                self.meta.notes.append(f"Speed converted from {unit} to km/h.")

        if sch.SWATH not in self.df.columns and default_swath_m:
            self.df[sch.SWATH] = float(default_swath_m)
            self.meta.notes.append(
                f"Swath width missing from the file; {default_swath_m:g} m assumed."
            )

        if sch.PASS not in self.df.columns and sch.HEADING in self.df.columns:
            self.df[sch.PASS] = self._detect_passes()

    def _time_delta_seconds(self) -> np.ndarray | None:
        """Interval between consecutive records, in seconds."""
        if sch.TIMESTAMP in self.df.columns:
            ts = self.df[sch.TIMESTAMP]
            if pd.api.types.is_datetime64_any_dtype(ts) and ts.notna().any():
                # The column's resolution depends on where it came from: text
                # stamps out of a file parse to microseconds under pandas 3,
                # the demo's arithmetic gives nanoseconds. Differencing the
                # stamps themselves keeps the seconds honest either way.
                # Reading the integer ticks as nanoseconds made every real
                # file's interval a thousand times too short, and a 12 s flow
                # delay then shifted thousands of records instead of six.
                dt = ts.diff().dt.total_seconds().to_numpy(dtype="float64", copy=True)
                if len(dt) > 1:
                    dt[0] = dt[1]
                # Absurd gaps mean a jump between separate operations.
                dt[(dt <= 0) | (dt > 60)] = np.nan
                return dt
        if sch.ELAPSED in self.df.columns:
            seconds = self.df[sch.ELAPSED].to_numpy(dtype="float64", na_value=np.nan)
            dt = np.diff(seconds, prepend=seconds[0])
            if len(dt) > 1:
                dt[0] = dt[1]
            dt[(dt <= 0) | (dt > 60)] = np.nan
            return dt
        return None

    def _detect_passes(self, angle_tol_deg: float = 35.0) -> np.ndarray:
        """Number the passes by grouping records with a stable heading.

        A pass ends when the machine turns more than ``angle_tol_deg`` away
        from the running mean heading — which is the headland turn.
        """
        heading = self.df[sch.HEADING].to_numpy(dtype="float64", na_value=np.nan)
        n = len(heading)
        pass_id = np.zeros(n, dtype="int32")
        if n == 0:
            return pass_id

        current = 0
        reference = heading[0] if math.isfinite(heading[0]) else 0.0
        for i in range(1, n):
            h = heading[i]
            if not math.isfinite(h):
                pass_id[i] = current
                continue
            diff = abs((h - reference + 180.0) % 360.0 - 180.0)
            if diff > angle_tol_deg:
                current += 1
                reference = h
            else:
                # A damped circular mean keeps the reference stable on gentle curves.
                reference = reference + 0.15 * ((h - reference + 180.0) % 360.0 - 180.0)
                reference %= 360.0
            pass_id[i] = current
        return pass_id

    def sort_by_time(self) -> None:
        """Sort chronologically — a prerequisite for every sequential filter."""
        if sch.TIMESTAMP in self.df.columns and self.df[sch.TIMESTAMP].notna().any():
            self.df = self.df.sort_values(sch.TIMESTAMP, kind="stable").reset_index(drop=True)
        elif sch.ELAPSED in self.df.columns and self.df[sch.ELAPSED].notna().any():
            self.df = self.df.sort_values(sch.ELAPSED, kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def numeric_columns(self) -> list[str]:
        """Numeric columns that could serve as the analysed variable."""
        skip = {sch.LON, sch.LAT, sch.X, sch.Y, sch.ELAPSED}
        return [
            c for c in self.df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(self.df[c])
        ]

    def bounds(self) -> list[float] | None:
        """Bounding box in WGS84: ``[west, south, east, north]``."""
        # Half a pair — a table with a longitude column and no latitude — is
        # no more placeable than none, and must not fail on the bare name.
        if sch.LON not in self.df.columns or sch.LAT not in self.df.columns or self.df.empty:
            return None
        lon = self.df[sch.LON].to_numpy(dtype="float64", na_value=np.nan)
        lat = self.df[sch.LAT].to_numpy(dtype="float64", na_value=np.nan)
        mask = np.isfinite(lon) & np.isfinite(lat)
        if not mask.any():
            return None
        return [
            float(np.min(lon[mask])), float(np.min(lat[mask])),
            float(np.max(lon[mask])), float(np.max(lat[mask])),
        ]

    def stats(self, column: str = sch.VALUE) -> dict[str, float | int]:
        """Descriptive statistics for the given column."""
        if column not in self.df.columns:
            return {}
        s = pd.to_numeric(self.df[column], errors="coerce").dropna()
        if s.empty:
            return {"n": 0}
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        return {
            "n": int(s.size),
            "mean": mean,
            "std": std,
            "cv": float(std / mean * 100.0) if mean else 0.0,
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q3": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "skew": float(s.skew()) if len(s) > 2 else 0.0,
        }

    def area_ha(self) -> float:
        """Worked area estimated by summing the swaths (width x advance).

        A polygon layer — a boundary, a prescription, the zones of the
        terrain analyser — has no swath either: its area is the ground its
        polygons cover, measured in the metric CRS. An elevation raster's
        area is the ground its cells cover, which the reader recorded in
        ``meta.extra['dem_area_ha']``, and the zone points of the terrain
        analyser are one per grid cell, so their count times the cell area
        (``meta.extra['zones_cell_m']``) is the field. The dataset summary,
        the project panel and the preflight all have to quote the one
        figure rather than a 0 next to a 60.
        """
        if self.geometry is not None and self.meta.geometry_type == "polygon":
            return self._polygon_area_ha()
        if self.meta.operation == "elevation":
            extra = self.meta.extra or {}
            if extra.get("dem_area_ha"):
                return float(extra["dem_area_ha"])
            cell = float(extra.get("zones_cell_m") or 0.0)
            return len(self.df) * cell * cell / 10_000.0
        if sch.SWATH not in self.df.columns or sch.DISTANCE not in self.df.columns:
            return 0.0
        swath = self.df[sch.SWATH].to_numpy(dtype="float64", na_value=np.nan)
        dist = self.df[sch.DISTANCE].to_numpy(dtype="float64", na_value=np.nan)
        area = np.nansum(swath * dist)
        return float(area / 10_000.0)

    def _polygon_area_ha(self) -> float:
        """The polygons' area in hectares, measured in the metric CRS.

        Degrees are not a unit of area, so the geometries go through the
        dataset's own metric CRS — or, on a dataset never projected, the
        UTM zone of their extent, the same choice :meth:`project` makes.
        """
        import geopandas as gpd

        geoms = [g for g in self.geometry if g is not None and not g.is_empty]
        if not geoms:
            return 0.0
        series = gpd.GeoSeries(geoms, crs=crs_mod.WGS84)
        target = self.metric_crs
        if not target:
            west, south, east, north = series.total_bounds
            target = crs_mod.pick_metric_crs([west, east], [south, north])
        return float(series.to_crs(target).area.sum() / 10_000.0)

    # ------------------------------------------------------------------
    # Interoperability
    # ------------------------------------------------------------------
    def to_geodataframe(self, metric: bool = False):
        """Convert to a ``GeoDataFrame`` (WGS84 or the metric CRS)."""
        import geopandas as gpd
        from shapely.geometry import Point

        if self.geometry is not None:
            gdf = gpd.GeoDataFrame(self.df.copy(), geometry=self.geometry, crs=crs_mod.WGS84)
        else:
            geom = gpd.points_from_xy(self.df[sch.LON], self.df[sch.LAT])
            gdf = gpd.GeoDataFrame(self.df.copy(), geometry=geom, crs=crs_mod.WGS84)
        if metric and self.metric_crs:
            gdf = gdf.to_crs(self.metric_crs)
        return gdf

    def copy(self) -> "Dataset":
        """Independent copy, preserving metadata and metric CRS."""
        import copy as _copy

        clone = Dataset.__new__(Dataset)
        clone.df = self.df.copy()
        clone.meta = _copy.deepcopy(self.meta)
        clone.geometry = list(self.geometry) if self.geometry is not None else None
        clone.metric_crs = self.metric_crs
        return clone

    def subset(self, mask) -> "Dataset":
        """New dataset holding only the rows where ``mask`` is true."""
        mask = np.asarray(mask, dtype=bool)
        clone = self.copy()
        clone.df = self.df.loc[mask].reset_index(drop=True)
        if self.geometry is not None:
            clone.geometry = [g for g, keep in zip(self.geometry, mask) if keep]
        return clone

    def summary(self) -> dict[str, Any]:
        """Serializable summary used by the interface."""
        return {
            "meta": self.meta.to_dict(),
            "rows": len(self.df),
            "columns": self.columns,
            "numeric_columns": self.numeric_columns(),
            "bounds": self.bounds(),
            "metric_crs": self.metric_crs,
            "area_ha": round(self.area_ha(), 2),
            "stats": self.stats(),
        }

#: Columns that accept a source unit declaration, and each one's group.
SOURCE_UNIT_COLUMNS = {
    sch.VALUE: "rate_mass",
    sch.TARGET_RATE: "rate_mass",
    sch.APPLIED_RATE: "rate_mass",
    sch.SPEED: "speed",
    sch.SWATH: "length",
    sch.DISTANCE: "length",
    sch.ELEVATION: "length",
}


def _declare_dem_elevation_unit(dataset: "Dataset", unit: str, factor: float) -> None:
    """Carry a DEM's declared elevation unit to everything that reads it.

    The points of an elevation layer are a sample of the raster's cells,
    shown on the map through ``value``; the terrain analyser reopens the
    raster itself. Converting only the ``elev_m`` column would leave the map
    colours in feet and the analysis reporting feet as metres — 2 296 m
    for a 700 m field — so the same factor goes to the mirrored value and
    into ``meta.extra`` for the raster reader. The factor accumulates,
    because a second declaration on a copy converts the column a second
    time as well.
    """
    extra = dataset.meta.extra
    if sch.VALUE in dataset.df.columns and dataset.meta.value_label == "Elevation":
        dataset.df[sch.VALUE] = pd.to_numeric(dataset.df[sch.VALUE], errors="coerce") * factor
        dataset.meta.source_value_unit = unit
    extra["elevation_factor"] = float(extra.get("elevation_factor", 1.0)) * float(factor)
    extra["elevation_unit_in"] = unit


def apply_source_units(
    dataset: "Dataset",
    declared: dict[str, str],
    crop: str | None = None,
) -> list[str]:
    """Convert columns from the unit declared in the file to the internal one.

    A monitor writes in whatever system it was configured for: a North
    American John Deere delivers bu/ac, mph and feet; a European Väderstad
    delivers kg/ha, km/h and metres. Without declaring that, a map in bu/ac
    would be read as if it were kg/ha and every number would come out far too
    small.

    Parameters
    ----------
    declared:
        Map of ``{column: unit}``, e.g. ``{"value": "bu/ac", "speed": "mph"}``.
    crop:
        Crop, needed for the bushel-based units.

    Returns
    -------
    list[str]
        Description of the conversions applied, for the notes.
    """
    applied: list[str] = []
    for column, unit in (declared or {}).items():
        if not unit or column not in dataset.df.columns:
            continue
        group = SOURCE_UNIT_COLUMNS.get(column)
        if group is None:
            continue
        internal = units_mod.UNIT_GROUPS[group]["internal"]
        if unit == internal:
            continue
        try:
            factor = units_mod.unit_factor(group, unit, crop)
        except ValueError:
            continue
        dataset.df[column] = pd.to_numeric(dataset.df[column], errors="coerce") * factor
        label = sch.LABELS.get(column, column)
        applied.append(f"{label}: converted from {unit} to {internal}.")
        if column == sch.ELEVATION and dataset.meta.operation == "elevation":
            _declare_dem_elevation_unit(dataset, unit, factor)

    if applied:
        dataset.meta.notes.extend(applied)
        if crop and not dataset.meta.crop:
            dataset.meta.crop = crop
    return applied
