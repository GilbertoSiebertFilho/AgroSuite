"""Machine telemetry: a logger's day, and the crop under its tyres.

:mod:`.analysis` splits a telemetry log (a SoMat ``.sie``, read by
:mod:`agrosuite.formats.somat`) into field work, standing, road and engine
off, and says where the time, distance, fuel and DEF went.
:mod:`.trampling` draws the tyre strips inside the field and puts an area —
and, given a yield, a crop loss — on them.

The two helpers here are what a caller needs to hand the analysis the field
and the yield from other files in the project.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import schema as sch
from .analysis import (ACTIVITIES, ACTIVITY_LABELS, MachineResult, MachineSettings,
                       analyse, findings, restate)
from .trampling import TyreSetup, parse_tyre

__all__ = [
    "ACTIVITIES", "ACTIVITY_LABELS", "MachineResult", "MachineSettings", "TyreSetup",
    "analyse", "boundary_polygon", "findings", "mean_yield", "parse_tyre", "restate",
]


def boundary_polygon(dataset, metric_crs: str) -> tuple[Any, Any, float]:
    """A loaded boundary as ``(lonlat_polygon, metric_polygon, area_ha)``.

    The polygons a boundary file declares, merged; a file with none —
    points only — is refused, because the envelope of where a machine drove
    is not the field and would call the headland road.
    """
    from pyproj import Transformer
    from shapely.ops import transform, unary_union

    polygons = [g for g in (dataset.geometry or [])
                if g is not None and not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
    if not polygons:
        raise ValueError(
            f"'{dataset.meta.name}' holds no polygon, so it cannot say where the field is. "
            "Pick the field's boundary file."
        )
    lonlat = unary_union(polygons)
    if not lonlat.is_valid:
        lonlat = lonlat.buffer(0)
    project = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    metric = transform(project, lonlat)
    return lonlat, metric, float(metric.area) / 10_000.0


def mean_yield(dataset, lonlat_boundary=None) -> tuple[float | None, int]:
    """The mean of a yield map's main variable (kg/ha), inside the boundary
    when there is one. Returns ``(kg_ha, records)``."""
    import shapely

    df = dataset.df
    if sch.VALUE not in df:
        return None, 0
    values = df[sch.VALUE].to_numpy(dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if lonlat_boundary is not None and {sch.LON, sch.LAT}.issubset(df.columns):
        lon = df[sch.LON].to_numpy(dtype=float)
        lat = df[sch.LAT].to_numpy(dtype=float)
        located = np.isfinite(lon) & np.isfinite(lat)
        inside = np.zeros(len(df), dtype=bool)
        inside[located] = shapely.contains_xy(lonlat_boundary, lon[located], lat[located])
        if (mask & inside).sum() >= 10:
            mask &= inside
    if not mask.any():
        return None, 0
    return float(values[mask].mean()), int(mask.sum())
