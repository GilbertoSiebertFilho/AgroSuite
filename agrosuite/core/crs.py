"""Coordinate reference systems.

Every metric calculation in AgroSuite — distance between points, swath area,
overlap, neighbourhood — happens in a CRS projected in metres. Since monitors
deliver geographic coordinates in WGS84, AgroSuite automatically picks the
UTM zone containing the centre of the field.
"""

from __future__ import annotations

import math

WGS84 = "EPSG:4326"


def utm_epsg(lon: float, lat: float) -> str:
    """EPSG code of the UTM zone containing the given point.

    326xx in the northern hemisphere, 327xx in the southern one.
    """
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Invalid coordinates for choosing a UTM zone.")
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def pick_metric_crs(lon_series, lat_series) -> str:
    """Pick the metric CRS suited to a set of points."""
    import numpy as np

    lon = np.asarray(lon_series, dtype="float64")
    lat = np.asarray(lat_series, dtype="float64")
    mask = np.isfinite(lon) & np.isfinite(lat)
    if not mask.any():
        raise ValueError("No valid coordinates in the dataset.")
    return utm_epsg(float(np.median(lon[mask])), float(np.median(lat[mask])))


def looks_geographic(x_series, y_series) -> bool:
    """Heuristic for detecting coordinates in decimal degrees.

    Used when the file carries no ``.prj``: values within ±180 / ±90 occur
    almost exclusively in geographic coordinates, since a field in UTM has X
    in the hundreds of thousands.
    """
    import numpy as np

    x = np.asarray(x_series, dtype="float64")
    y = np.asarray(y_series, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return False
    return bool(
        np.nanmax(np.abs(x[mask])) <= 180.0 and np.nanmax(np.abs(y[mask])) <= 90.0
    )
