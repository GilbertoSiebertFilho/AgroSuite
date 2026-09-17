"""Reading data in the Augmenta format.

Augmenta is an on-board vision system that classifies the crop in real time
and drives variable rate application. Its exports arrive in three shapes, all
covered here:

1. **Session GeoJSON** — a ``FeatureCollection`` whose features are the
   points or cells of the pass, with vigour/biomass properties and the rate
   that was actually applied.
2. **Wrapped JSON** — the same GeoJSON inside an object carrying session
   metadata (``session``/``field``/``machine``) under some key.
3. **CSV/SHP** — handled by the generic readers, which recognize Augmenta's
   index columns through the alias dictionary.

What sets Augmenta apart is the semantics: besides the applied rate it
carries a vigour index per point, and it is that (vigour, rate) pair that
feeds the app's application-efficiency analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta

#: Property keys that characterize an Augmenta export.
AUGMENTA_KEYS = {
    "vigor", "biomass", "biomass_index", "canopy", "canopy_cover",
    "crop_coverage", "weed_coverage", "applied_rate", "vra_rate",
    "session_id", "augmenta_id", "ndvi", "ndre", "nozzle", "spray_rate",
    "green_index", "plant_count", "coverage_percent",
}

#: Preference order for choosing the dataset's main variable.
VALUE_PRIORITY = (
    "applied_rate", "vra_rate", "spray_rate", "rate",
    "vigor", "biomass_index", "biomass", "ndvi", "ndre",
    "canopy_cover", "crop_coverage", "green_index",
)

#: Recognized vigour indices, kept apart from the rate for the cross analysis.
VIGOR_KEYS = (
    "vigor", "biomass_index", "biomass", "ndvi", "ndre",
    "canopy_cover", "canopy", "crop_coverage", "green_index",
)


def _feature_collection(payload: Any) -> dict | None:
    """Locate the ``FeatureCollection`` inside a possibly wrapped payload."""
    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            return payload
        for value in payload.values():
            found = _feature_collection(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _feature_collection(value)
            if found is not None:
                return found
    return None


def is_augmenta_payload(payload: Any) -> bool:
    """Say whether an already-loaded JSON carries the Augmenta signature."""
    fc = _feature_collection(payload)
    if fc is None:
        return False
    text = json.dumps(payload, default=str)[:4000].lower()
    if "augmenta" in text:
        return True
    for feature in fc.get("features", [])[:20]:
        props = (feature or {}).get("properties") or {}
        keys = {sch.normalize_name(k) for k in props}
        if len(keys & AUGMENTA_KEYS) >= 2:
            return True
    return False


def _session_metadata(payload: Any) -> dict[str, Any]:
    """Extract session metadata from the wrapper, when present."""
    meta: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return meta
    interesting = (
        "session", "session_id", "field", "field_name", "farm", "machine",
        "machine_id", "operator", "crop", "product", "date", "start_time",
        "end_time", "application", "device", "firmware", "unit",
    )
    for key, value in payload.items():
        norm = sch.normalize_name(key)
        if norm in interesting and not isinstance(value, (list, dict)):
            meta[norm] = value
        elif norm in ("session", "field", "machine", "application") and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not isinstance(sub_value, (list, dict)):
                    meta[f"{norm}_{sch.normalize_name(sub_key)}"] = sub_value
    return meta


def read_augmenta(path: Path, payload: Any | None = None) -> Dataset:
    """Read an Augmenta export and return the normalized dataset."""
    path = Path(path)
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))

    fc = _feature_collection(payload)
    if fc is None:
        raise ValueError("Augmenta file with no recognizable FeatureCollection.")

    import geopandas as gpd
    from shapely.geometry import shape

    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    geometries: list[Any] = []
    polygonal = False

    for feature in fc.get("features", []):
        if not feature:
            continue
        geom_json = feature.get("geometry")
        if not geom_json:
            continue
        try:
            geom = shape(geom_json)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            polygonal = True
        props = dict(feature.get("properties") or {})
        point = geom.representative_point()
        props[sch.LON] = point.x
        props[sch.LAT] = point.y
        rows.append(props)
        geometries.append(geom)

    if not rows:
        raise ValueError("No valid feature found in the Augmenta export.")

    df = pd.DataFrame(rows)
    original_columns = [c for c in df.columns if c not in (sch.LON, sch.LAT)]
    df.columns = [str(c).strip() for c in df.columns]

    # Choose the main variable before the generic mapping, so the applied rate
    # takes priority over any other numeric column.
    norm_lookup = {sch.normalize_name(c): c for c in df.columns}
    value_source = next((norm_lookup[k] for k in VALUE_PRIORITY if k in norm_lookup), None)

    mapping = sch.map_columns([c for c in df.columns if c not in (sch.LON, sch.LAT)])
    mapping.pop(value_source, None)
    if mapping:
        df = df.rename(columns=mapping)

    if value_source:
        df[sch.VALUE] = pd.to_numeric(df[value_source], errors="coerce")
        notes.append(f"Main variable: '{value_source}'.")

    vigor_source = next((norm_lookup[k] for k in VIGOR_KEYS if k in norm_lookup), None)
    if vigor_source:
        df["vigor_index"] = pd.to_numeric(df[vigor_source], errors="coerce")
        notes.append(f"Vigour index read from '{vigor_source}'.")

    session = _session_metadata(payload)
    if polygonal:
        notes.append("Augmenta polygon cells: representative points used for analysis.")

    operation = "application" if value_source in (
        norm_lookup.get("applied_rate"), norm_lookup.get("vra_rate"), norm_lookup.get("spray_rate")
    ) else "vigor"

    meta = DatasetMeta(
        name=str(session.get("field_name") or session.get("session_id") or path.stem),
        source_path=str(path),
        source_format="augmenta",
        brand="augmenta",
        brand_label="Augmenta",
        operation=operation,
        crop=session.get("crop"),
        field_name=session.get("field_name") or session.get("field"),
        value_label="Applied rate" if operation == "application" else "Vigour index",
        geometry_type="polygon" if polygonal else "point",
        notes=notes,
        extra={
            "brand_confidence": 1.0,
            "augmenta_session": session,
            "original_columns": original_columns,
            "vigor_column": "vigor_index" if vigor_source else None,
        },
    )
    return Dataset(df, meta, geometry=geometries if polygonal else None)


def vigor_rate_summary(ds: Dataset, bins: int = 6) -> dict[str, Any]:
    """Cross the vigour index against the applied rate.

    This is the reading that makes sense of Augmenta data: if the system is
    modulating properly, the rate should vary monotonically across the vigour
    classes. A flat relationship means the application was effectively
    uniform — that is, variable rate did not engage.
    """
    import numpy as np

    if "vigor_index" not in ds.df.columns or sch.VALUE not in ds.df.columns:
        return {"available": False, "reason": "Dataset has no vigour/rate pair."}

    frame = ds.df[["vigor_index", sch.VALUE]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 10:
        return {"available": False, "reason": "Too few points to cross vigour and rate."}

    try:
        classes = pd.qcut(frame["vigor_index"], q=bins, duplicates="drop")
    except ValueError:
        return {"available": False, "reason": "Vigour index does not vary enough."}

    grouped = frame.groupby(classes, observed=True)[sch.VALUE].agg(["count", "mean", "std"])
    rows = [
        {
            "class": f"{interval.left:.3g} - {interval.right:.3g}",
            "n": int(row["count"]),
            "mean_rate": round(float(row["mean"]), 3),
            "sd": round(float(row["std"]), 3) if pd.notna(row["std"]) else 0.0,
        }
        for interval, row in grouped.iterrows()
    ]

    corr = float(frame["vigor_index"].corr(frame[sch.VALUE]))
    rates = grouped["mean"].to_numpy()
    spread = float(np.ptp(rates) / np.mean(rates) * 100.0) if np.mean(rates) else 0.0

    if spread < 5:
        reading = ("The rate barely changed across the vigour classes: the "
                   "application was, in practice, uniform.")
    elif corr < -0.3:
        reading = ("Higher rate where vigour is lower — compensatory behaviour, "
                   "typical of a corrective application.")
    elif corr > 0.3:
        reading = ("Higher rate where vigour is higher — proportional to biomass, "
                   "typical of fungicide and desiccation.")
    else:
        reading = ("Weak relationship between vigour and rate; check how the response "
                   "map was configured.")

    return {
        "available": True,
        "classes": rows,
        "correlation": round(corr, 3) if pd.notna(corr) else None,
        "relative_spread_pct": round(spread, 1),
        "reading": reading,
    }
