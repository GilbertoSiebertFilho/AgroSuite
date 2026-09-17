"""Guidance (AB) lines.

An AB line is the pair of points that defines the working direction; the
monitor generates the parallel passes from it, spaced by the implement
width. When a strip trial is laid out along a direction, the AB line has to
follow exactly that direction — otherwise the operator's passes cut across
the strips and the experiment is lost.
"""

from __future__ import annotations

import math
from typing import Any


def ab_line_from_direction(
    boundary_lonlat: list[tuple[float, float]],
    angle_deg: float,
    name: str = "AB",
    extend_m: float = 100.0,
) -> dict[str, Any]:
    """Create an AB line crossing the field along a given direction.

    Parameters
    ----------
    angle_deg:
        Direction in **mathematical** degrees — 0 points east and increases
        counter-clockwise. It is the same angle the trial layout returns, so
        that the line and the strips end up aligned.
    extend_m:
        How far to extend the line beyond the field, in metres. Extra length
        helps: the monitor needs the reference before the machine enters.
    """
    from pyproj import Transformer
    from shapely.geometry import Polygon

    from .crs import WGS84, pick_metric_crs

    if len(boundary_lonlat) < 3:
        raise ValueError("Boundary too small to build an AB line from.")

    lons = [p[0] for p in boundary_lonlat]
    lats = [p[1] for p in boundary_lonlat]
    metric_crs = pick_metric_crs(lons, lats)
    to_metric = Transformer.from_crs(WGS84, metric_crs, always_xy=True)
    to_wgs = Transformer.from_crs(metric_crs, WGS84, always_xy=True)

    xs, ys = to_metric.transform(lons, lats)
    field = Polygon(zip(xs, ys))
    if not field.is_valid:
        field = field.buffer(0)
    centroid = field.centroid

    # Half the bounding box diagonal guarantees the line crosses the whole
    # field in any direction.
    min_x, min_y, max_x, max_y = field.bounds
    half = math.hypot(max_x - min_x, max_y - min_y) / 2.0 + extend_m

    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad) * half, math.sin(rad) * half
    a = to_wgs.transform(centroid.x - dx, centroid.y - dy)
    b = to_wgs.transform(centroid.x + dx, centroid.y + dy)

    # Compass bearing: 0 = north, increasing clockwise.
    heading = (90.0 - angle_deg) % 360.0
    return {
        "name": name,
        "type": 1,  # AB line
        "a": (float(a[0]), float(a[1])),
        "b": (float(b[0]), float(b[1])),
        "heading": round(heading, 2),
    }


def ab_line_from_points(
    a_lonlat: tuple[float, float],
    b_lonlat: tuple[float, float],
    name: str = "AB",
) -> dict[str, Any]:
    """Build an AB line from two points picked on the map."""
    from pyproj import Transformer

    from .crs import WGS84, pick_metric_crs

    metric_crs = pick_metric_crs([a_lonlat[0], b_lonlat[0]], [a_lonlat[1], b_lonlat[1]])
    to_metric = Transformer.from_crs(WGS84, metric_crs, always_xy=True)
    ax, ay = to_metric.transform(*a_lonlat)
    bx, by = to_metric.transform(*b_lonlat)

    length = math.hypot(bx - ax, by - ay)
    if length < 1.0:
        raise ValueError(
            "Points A and B are less than a metre apart, so the resulting "
            "direction would be unreliable. Move them apart along the pass."
        )
    heading = (math.degrees(math.atan2(bx - ax, by - ay)) + 360.0) % 360.0
    return {
        "name": name,
        "type": 1,
        "a": (float(a_lonlat[0]), float(a_lonlat[1])),
        "b": (float(b_lonlat[0]), float(b_lonlat[1])),
        "heading": round(heading, 2),
        "length_m": round(length, 1),
    }


def ab_lines_to_geojson(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert AB lines to GeoJSON, for drawing on the map and exporting."""
    features = []
    for line in lines:
        if line.get("a") and line.get("b"):
            coordinates = [list(line["a"]), list(line["b"])]
        elif line.get("points"):
            coordinates = [list(p) for p in line["points"]]
        else:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "name": line.get("name"),
                "type": line.get("type", 1),
                "heading": line.get("heading"),
            },
        })
    return {"type": "FeatureCollection", "features": features}
