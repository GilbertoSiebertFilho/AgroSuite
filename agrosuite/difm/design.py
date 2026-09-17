"""Laying out on-farm strip trials.

A useful trial needs three properties this layout guarantees:

* **Randomization in blocks.** Rates are drawn within each block of
  consecutive strips, so a fertility gradient running across the field does
  not get confounded with the rate effect.
* **Operable strips.** Strip width is a multiple of the implement width;
  otherwise the operator cannot drive the trial.
* **Replication.** Each rate appears once per block; the number of blocks is
  the number of replicates.

The output is ready to become a prescription in any of the app's export
formats.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.units import Phrase


def _principal_direction(polygon) -> float:
    """Angle (degrees) of the longest side of the field's minimum bounding box.

    Strips parallel to the longest side come out longer, which cuts the number
    of turns and increases the usable area of each treatment.
    """
    rectangle = polygon.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)[:4]
    best_length, best_angle = -1.0, 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best_length:
            best_length = length
            best_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return best_angle


def design_strips(
    boundary_lonlat: list[tuple[float, float]],
    rates: list[float],
    implement_width_m: float = 12.0,
    passes_per_strip: int = 2,
    blocks: int = 4,
    angle_deg: float | None = None,
    buffer_m: float = 0.0,
    seed: int = 0,
    units: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate trial strips over a field boundary.

    Parameters
    ----------
    boundary_lonlat:
        Field boundary as a list of ``(lon, lat)`` in WGS84.
    rates:
        Rates to test. At least three — below that there is no curve to fit.
    implement_width_m:
        Working width of the implement that will apply the trial.
    passes_per_strip:
        How many implement passes make up each strip. Two or more leave the
        centre pass free of the neighbouring strips' influence.
    blocks:
        Number of replicates (randomized blocks).
    angle_deg:
        Strip direction. When absent, the field's longest side is used.
    buffer_m:
        Inward setback from the boundary, dropping the headland.
    units:
        The reader's unit set, which reaches the two refusals that quote a
        width: they name the setback and the strip width the user typed,
        and have to name them in the unit they were typed in.

    Returns
    -------
    dict
        With ``features`` (strip GeoJSON), ``summary`` and ``warnings``.
    """
    from pyproj import Transformer
    from shapely import affinity
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union

    if len(rates) < 3:
        raise ValueError("A response trial needs at least 3 distinct rates.")
    if len(boundary_lonlat) < 3:
        raise ValueError("Invalid field boundary: fewer than 3 vertices.")
    if blocks < 1:
        raise ValueError("The trial needs at least 1 block.")

    from ..core.crs import WGS84, pick_metric_crs

    lons = [p[0] for p in boundary_lonlat]
    lats = [p[1] for p in boundary_lonlat]
    metric_crs = pick_metric_crs(lons, lats)
    to_metric = Transformer.from_crs(WGS84, metric_crs, always_xy=True)
    to_wgs = Transformer.from_crs(metric_crs, WGS84, always_xy=True)

    xs, ys = to_metric.transform(lons, lats)
    field = Polygon(zip(xs, ys))
    if not field.is_valid:
        field = field.buffer(0)
    if field.is_empty:
        raise ValueError("The field boundary produced an empty polygon.")

    say = Phrase(units)
    warnings: list[str] = []
    working = field.buffer(-abs(buffer_m)) if buffer_m else field
    if working.is_empty:
        raise ValueError(
            f"The {say.length(buffer_m, None)} setback consumed the whole field. "
            "Reduce the headland."
        )
    if working.geom_type == "MultiPolygon":
        working = max(working.geoms, key=lambda g: g.area)
        warnings.append("The setback split the field; the largest continuous part was used.")

    angle = angle_deg if angle_deg is not None else _principal_direction(working)
    centroid = working.centroid
    # Rotate the field so the strips line up with the X axis.
    aligned = affinity.rotate(working, -angle, origin=centroid, use_radians=False)
    min_x, min_y, max_x, max_y = aligned.bounds

    strip_width = implement_width_m * max(1, int(passes_per_strip))
    total_strips = int(math.floor((max_y - min_y) / strip_width))
    if total_strips < len(rates):
        raise ValueError(
            f"The field only fits {total_strips} strip(s) of "
            f"{say.length(strip_width, None)}, and the trial needs at least "
            f"{len(rates)} — one per rate. Reduce the strip width or the number of "
            "rates."
        )

    usable_blocks = min(blocks, total_strips // len(rates))
    if usable_blocks < blocks:
        warnings.append(
            f"The field fits {usable_blocks} complete block(s) instead of the {blocks} "
            "requested; the trial was adjusted."
        )
    if usable_blocks < 2:
        warnings.append(
            "With a single block there is no replication: the analysis will not be "
            "able to separate the rate effect from the field's natural variation."
        )

    # Draw the rate order within each block, as in a randomized block design.
    rng = np.random.default_rng(seed)
    assignment: list[float] = []
    for _ in range(usable_blocks):
        order = rng.permutation(len(rates))
        assignment.extend(float(rates[i]) for i in order)

    features: list[dict[str, Any]] = []
    areas: list[float] = []
    for index, rate in enumerate(assignment):
        y0 = min_y + index * strip_width
        band = Polygon([
            (min_x - 10, y0), (max_x + 10, y0),
            (max_x + 10, y0 + strip_width), (min_x - 10, y0 + strip_width),
        ])
        piece = aligned.intersection(band)
        if piece.is_empty or piece.area < strip_width * implement_width_m:
            continue
        piece = affinity.rotate(piece, angle, origin=centroid, use_radians=False)

        geoms = piece.geoms if piece.geom_type == "MultiPolygon" else [piece]
        for part in geoms:
            if part.area < strip_width * implement_width_m:
                continue
            ring = [to_wgs.transform(x, y) for x, y in part.exterior.coords]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
                "properties": {
                    "strip": index + 1,
                    "block": index // len(rates) + 1,
                    "rate": round(float(rate), 2),
                    "area_ha": round(part.area / 10_000.0, 3),
                },
            })
            areas.append(part.area / 10_000.0)

    if not features:
        raise ValueError("No usable strip was generated with these parameters.")

    counts: dict[float, int] = {}
    for feature in features:
        rate = feature["properties"]["rate"]
        counts[rate] = counts.get(rate, 0) + 1

    return {
        "features": {"type": "FeatureCollection", "features": features},
        "summary": {
            "rates": sorted(counts),
            "reps_per_rate": {str(k): v for k, v in sorted(counts.items())},
            "strips": len(features),
            "blocks": usable_blocks,
            "strip_width_m": strip_width,
            "passes_per_strip": max(1, int(passes_per_strip)),
            "direction_deg": round(angle % 180.0, 1),
            "total_area_ha": round(sum(areas), 2),
            "mean_strip_area_ha": round(sum(areas) / len(areas), 3),
            "metric_crs": metric_crs,
        },
        "warnings": warnings,
    }
