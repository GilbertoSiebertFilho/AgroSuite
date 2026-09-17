"""Readers for field files.

Covers the formats that actually move between monitors and the office:

* **Shapefile** (.shp + .dbf + .shx + .prj) — the common currency between
  platforms; every manufacturer on the list can read or write it.
* **GeoJSON / JSON** — Augmenta's standard output and that of web services.
* **CSV / TXT** — monitor log exports, with every separator, decimal mark and
  encoding variation that turns up in the real world.
* **Excel** (.xlsx) — sampling and trial spreadsheets.
* **KML/KMZ** — field boundaries.

ISOXML has its own module (:mod:`agrosuite.formats.isoxml`), being a folder
format with an XML header and binary logs.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.crs import WGS84, looks_geographic
from ..core.dataset import Dataset, DatasetMeta
from . import brands as brands_mod

#: Encodings tried in order — older DBF and non-US CSV often come in latin-1,
#: and European monitors sometimes in cp1252.
ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")

#: Candidate separators for tabular files.
DELIMITERS = (",", ";", "\t", "|")


# ==========================================================================
# Helpers
# ==========================================================================

def _read_text(path: Path, limit: int | None = None) -> str:
    """Read a text file, trying the usual encodings."""
    raw = path.read_bytes() if limit is None else path.open("rb").read(limit)
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _sniff_delimiter(sample: str) -> str:
    """Work out a CSV's column separator."""
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        pass
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {d: first_line.count(d) for d in DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _looks_like_unit_row(row: pd.Series) -> bool:
    """Detect the unit row some monitors insert under the header.

    For example ``Yield, Moisture, Speed`` followed by ``bu/ac, %, mph``. That
    row is text in columns that should be numeric.
    """
    values = [str(v).strip() for v in row.tolist() if str(v).strip() not in ("", "nan")]
    if not values:
        return False
    unit_like = {
        "bu/ac", "kg/ha", "t/ha", "lb/ac", "%", "mph", "km/h", "m/s", "ft", "m",
        "deg", "sec", "s", "ac", "ha", "gal/ac", "l/ha", "sc/ha", "seeds/ac",
    }
    hits = sum(1 for v in values if v.lower() in unit_like)
    numeric = 0
    for v in values:
        try:
            float(v.replace(",", "."))
            numeric += 1
        except ValueError:
            pass
    return hits >= 1 and numeric <= len(values) * 0.3


def _decimal_comma_ratio(sample: str, delimiter: str) -> float:
    """Fraction of fields that look like numbers with a decimal comma."""
    lines = [ln for ln in sample.splitlines()[1:40] if ln.strip()]
    if not lines:
        return 0.0
    total = comma_numbers = 0
    for line in lines:
        for cell in line.split(delimiter):
            cell = cell.strip().strip('"')
            if not cell:
                continue
            total += 1
            if "," in cell and cell.replace(",", "").replace("-", "").replace(".", "").isdigit():
                comma_numbers += 1
    return comma_numbers / total if total else 0.0


def _normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename recognized columns to the canonical schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = sch.map_columns(df.columns)
    if mapping:
        df = df.rename(columns=mapping)
    return df, mapping


def _resolve_coordinates(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Ensure ``lon``/``lat`` columns from whatever the table provides."""
    if sch.LON in df.columns and sch.LAT in df.columns:
        return df
    # Some exports name the columns simply X and Y.
    candidates = [(c, sch.normalize_name(c)) for c in df.columns]
    x_col = next((c for c, n in candidates if n in ("x", "coord_x", "este", "easting")), None)
    y_col = next((c for c, n in candidates if n in ("y", "coord_y", "norte", "northing")), None)
    if x_col and y_col:
        if looks_geographic(df[x_col], df[y_col]):
            df = df.rename(columns={x_col: sch.LON, y_col: sch.LAT})
            notes.append("X/Y columns read as longitude/latitude (WGS84).")
        else:
            notes.append(
                "X/Y columns are in projected units with no CRS declared — give the "
                "source EPSG so they can be reprojected correctly."
            )
    return df


def _pick_value_column(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Choose the main variable when the mapping found no ``value`` column."""
    if sch.VALUE in df.columns:
        return df
    priority = (sch.APPLIED_RATE, sch.TARGET_RATE, sch.FLOW)
    for col in priority:
        if col in df.columns:
            df[sch.VALUE] = pd.to_numeric(df[col], errors="coerce")
            notes.append(f"Main variable taken from '{sch.LABELS[col]}'.")
            return df
    ignore = {sch.LON, sch.LAT, sch.X, sch.Y, sch.ELEVATION, sch.SPEED, sch.SWATH,
              sch.HEADING, sch.DISTANCE, sch.PASS, sch.SECTION, sch.ELAPSED}
    numeric = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    if numeric:
        # The numeric column with the highest relative variability is usually the measurement.
        best, best_cv = None, -1.0
        for c in numeric:
            s = df[c].dropna()
            if s.empty or s.mean() == 0:
                continue
            cv = abs(float(s.std()) / float(s.mean()))
            if 0 < cv < 5 and cv > best_cv:
                best, best_cv = c, cv
        chosen = best or numeric[0]
        df[sch.VALUE] = pd.to_numeric(df[chosen], errors="coerce")
        notes.append(f"Main variable taken from column '{chosen}'.")
    return df


def _guess_operation(df: pd.DataFrame, path: Path, brand: str) -> str:
    """Infer the operation type from column names and from the path."""
    cols = {sch.normalize_name(c) for c in df.columns}
    text = sch.normalize_name(str(path))
    if any(k in text for k in ("boundary", "contorno", "limite", "field_border")):
        return "boundary"
    if any(k in text for k in ("prescription", "prescricao", "_rx", "rx_", "target")):
        return "prescription"
    if sch.TARGET_RATE in cols and sch.VALUE not in cols:
        return "prescription"
    if any(k in cols for k in ("yield", "yld", "dry_yield", "moisture_pct", "flow_kgs")):
        return "harvest"
    if sch.TARGET_RATE in cols and sch.VALUE in cols:
        # A target rate and a measurement that are the *same numbers* means
        # there was never a measurement: the main variable was filled in from
        # the target because nothing else was there. That is a plan, not a log
        # of what the machine did.
        target = pd.to_numeric(df[sch.TARGET_RATE], errors="coerce")
        value = pd.to_numeric(df[sch.VALUE], errors="coerce")
        if target.notna().any() and np.allclose(
            target.fillna(0).to_numpy(), value.fillna(0).to_numpy(), equal_nan=True
        ):
            return "prescription"
        # Otherwise the target came from the map and the value is what actually
        # went out — an as-applied record.
        return "application"
    if any(k in text for k in ("harvest", "colheita", "yield", "rendimento")):
        return "harvest"
    if any(k in cols for k in ("vigor", "ndvi", "ndre", "biomass_index", "canopy")):
        return "vigor"
    if brand == "augmenta":
        return "vigor"
    if any(k in text for k in ("planting", "plantio", "seeding", "semeadura")):
        return "planting"
    if any(k in text for k in ("applied", "aplicacao", "application", "spray", "fert")):
        return "application"
    if sch.APPLIED_RATE in cols:
        return "application"
    return "unknown"


def _build_dataset(
    df: pd.DataFrame,
    path: Path,
    source_format: str,
    notes: list[str],
    geometry: list | None = None,
    brand_hint: str | None = None,
    extra_text: str = "",
) -> Dataset:
    """Build the final :class:`Dataset` from an already-read table."""
    if not len(df):
        # A header and nothing under it parses fine and would become a dataset
        # with no records: selectable, saved into the project, useless. A
        # readme dropped with a field folder is the usual way to get one.
        raise ValueError(
            f"'{path.name}' holds no records: a header line at most, and nothing "
            "under it. It is not a data file, or the export came out empty."
        )
    original_columns = list(df.columns)
    df, mapping = _normalize_frame(df)
    df = _resolve_coordinates(df, notes)
    df = _pick_value_column(df, notes)

    brand_key, confidence = brands_mod.detect_brand(
        original_columns, str(path), extra_text
    )
    if brand_hint:
        brand_key, confidence = brand_hint, 1.0
    brand = brands_mod.get_brand(brand_key)

    meta = DatasetMeta(
        name=path.stem or "dataset",
        source_path=str(path),
        source_format=source_format,
        brand=brand.key,
        brand_label=brand.label,
        operation=_guess_operation(df, path, brand.key),
        geometry_type="polygon" if geometry is not None else "point",
        notes=notes,
        extra={
            "brand_confidence": confidence,
            "column_mapping": mapping,
            "original_columns": original_columns,
            # The manufacturer's usual export unit is the strongest single hint
            # when the file's magnitude does not match the assumed unit.
            "brand_default_rate_unit": brand.default_units.get("yield")
            or brand.default_units.get("rate"),
        },
    )
    return Dataset(df, meta, geometry=geometry)


# ==========================================================================
# Readers, by format
# ==========================================================================

def read_shapefile(path: Path, brand_hint: str | None = None) -> Dataset:
    """Read a shapefile, or any vector source GDAL supports."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    notes: list[str] = []

    if gdf.crs is None:
        notes.append(
            "Shapefile with no .prj — coordinates assumed to be WGS84. If the field "
            "shows up in the wrong place on the map, give the correct EPSG."
        )
        gdf = gdf.set_crs(WGS84, allow_override=True)
    elif gdf.crs.to_string() != WGS84:
        notes.append(f"Reprojected from {gdf.crs.to_string()} to WGS84.")
        gdf = gdf.to_crs(WGS84)

    geom_types = set(gdf.geom_type.dropna().unique())
    polygonal = bool(geom_types & {"Polygon", "MultiPolygon"})
    linear = bool(geom_types & {"LineString", "MultiLineString"})

    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    centroids = gdf.geometry.representative_point()
    df[sch.LON] = centroids.x.to_numpy()
    df[sch.LAT] = centroids.y.to_numpy()

    geometry = list(gdf.geometry) if (polygonal or linear) else None
    if polygonal:
        notes.append("Polygon layer: representative points used for placement and analysis.")

    ds = _build_dataset(df, Path(path), "shapefile", notes, geometry, brand_hint)
    if polygonal and ds.meta.operation == "unknown":
        ds.meta.operation = "prescription" if sch.TARGET_RATE in ds.df.columns else "boundary"
    if linear:
        ds.meta.geometry_type = "line"
        if ds.meta.operation == "unknown":
            ds.meta.operation = "guidance"

    # A boundary or AB-line shapefile is setup material, not analysis material:
    # keeping the geometry in WGS84 lets it be re-exported to another monitor
    # without any lossy conversion in between.
    if ds.meta.operation in ("boundary", "guidance"):
        ds.meta.extra["field_setup"] = _setup_from_geometry(gdf, ds.meta.operation)
    return ds


def read_geojson(path: Path, brand_hint: str | None = None) -> Dataset:
    """Read GeoJSON/JSON, including Augmenta's session format."""
    from .augmenta import is_augmenta_payload, read_augmenta

    payload = json.loads(_read_text(Path(path)))
    if is_augmenta_payload(payload):
        return read_augmenta(Path(path), payload)

    import geopandas as gpd

    gdf = gpd.read_file(path)
    notes: list[str] = []
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    elif gdf.crs.to_string() != WGS84:
        gdf = gdf.to_crs(WGS84)

    polygonal = bool(set(gdf.geom_type.dropna().unique()) & {"Polygon", "MultiPolygon"})
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    pts = gdf.geometry.representative_point()
    df[sch.LON] = pts.x.to_numpy()
    df[sch.LAT] = pts.y.to_numpy()
    geometry = list(gdf.geometry) if polygonal else None
    return _build_dataset(df, Path(path), "geojson", notes, geometry, brand_hint)


def read_tabular(path: Path, brand_hint: str | None = None) -> Dataset:
    """Read a monitor CSV/TXT, handling separator, decimal mark and encoding."""
    path = Path(path)
    sample = _read_text(path, limit=64_000)
    if not sample.strip():
        raise ValueError(f"Empty file: {path.name}")

    notes: list[str] = []
    delimiter = _sniff_delimiter(sample)
    decimal = "." 
    if delimiter != "," and _decimal_comma_ratio(sample, delimiter) > 0.25:
        decimal = ","
        notes.append("Comma read as the decimal separator.")

    last_error: Exception | None = None
    df = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(
                path, sep=delimiter, encoding=enc, decimal=decimal,
                engine="python", skip_blank_lines=True, on_bad_lines="skip",
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    if df is None:
        raise ValueError(f"Could not parse {path.name}: {last_error}")

    if len(df) and _looks_like_unit_row(df.iloc[0]):
        units_row = {c: str(v).strip() for c, v in df.iloc[0].items()}
        df = df.iloc[1:].reset_index(drop=True)
        notes.append("Unit row under the header discarded.")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        ds = _build_dataset(df, path, "csv", notes, brand_hint=brand_hint, extra_text=sample[:2000])
        ds.meta.extra["source_units"] = units_row
        return ds

    return _build_dataset(df, path, "csv", notes, brand_hint=brand_hint, extra_text=sample[:2000])


def read_excel(path: Path, brand_hint: str | None = None) -> Dataset:
    """Read the first sheet of an Excel file."""
    df = pd.read_excel(path)
    notes = ["Excel workbook: first sheet imported."]
    return _build_dataset(df, Path(path), "excel", notes, brand_hint=brand_hint)


def read_kml(path: Path, brand_hint: str | None = None) -> Dataset:
    """Read KML/KMZ — typically field boundaries."""
    import geopandas as gpd

    path = Path(path)
    notes: list[str] = []
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as zf:
            inner = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if inner is None:
                raise ValueError("The KMZ holds no .kml file.")
            data = zf.read(inner)
        gdf = gpd.read_file(io.BytesIO(data))
        notes.append(f"KMZ unpacked ({inner}).")
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    elif gdf.crs.to_string() != WGS84:
        gdf = gdf.to_crs(WGS84)

    polygonal = bool(set(gdf.geom_type.dropna().unique()) & {"Polygon", "MultiPolygon"})
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    pts = gdf.geometry.representative_point()
    df[sch.LON] = pts.x.to_numpy()
    df[sch.LAT] = pts.y.to_numpy()
    ds = _build_dataset(df, path, "kml", notes, list(gdf.geometry) if polygonal else None, brand_hint)
    if polygonal:
        ds.meta.operation = "boundary"
    return ds


def _setup_from_geometry(gdf, operation: str) -> dict:
    """Extract a boundary or guidance lines from a vector layer."""
    rings: list[list[tuple[float, float]]] = []
    lines: list[dict] = []

    for _, row in gdf.iterrows():
        geometry = row[gdf.geometry.name]
        if geometry is None or geometry.is_empty:
            continue
        parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
        for part in parts:
            if part.geom_type == "Polygon":
                rings.append([(float(x), float(y)) for x, y in part.exterior.coords])
            elif part.geom_type == "LineString":
                points = [(float(x), float(y)) for x, y in part.coords]
                if len(points) < 2:
                    continue
                name = next(
                    (str(row[c]) for c in ("name", "NAME", "Name", "label", "LABEL")
                     if c in gdf.columns and row.get(c) is not None),
                    f"Line {len(lines) + 1}",
                )
                lines.append({
                    "name": name,
                    "type": 1 if len(points) == 2 else 3,  # AB line or recorded curve
                    "a": points[0],
                    "b": points[-1],
                    "points": points,
                })

    field = {
        "name": None,
        "boundaries": [[{"type": 1, "points": r}] for r in rings],
        "headlands": [],
        "obstacles": [],
        "guidance_lines": lines,
    }
    return {"fields": [field]}
