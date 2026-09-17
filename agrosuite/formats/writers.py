"""Writing files to take back to the monitor.

Every platform has its quirks, and most of them are small — but enough for
the monitor to refuse the file in the middle of the field:

* A shapefile's DBF limits field names to **10 characters**. Longer names get
  truncated silently by the library, and two fields can collide; here the
  truncation is done deliberately and reported.
* Each monitor looks for the rate under a different field name. The exporter
  writes the field under the name the chosen platform expects and, for
  safety, repeats the value in a generic ``RATE`` field.
* A prescription must be **polygons**, not points. When the source is a set
  of points, the cells are generated as a grid.
* ISOXML demands a complete rectangular grid, with zeros where nothing is
  applied.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.crs import WGS84
from ..core.dataset import Dataset
from . import isoxml as isoxml_mod

#: Rate field name each platform expects in an Rx shapefile.
RATE_FIELD_BY_BRAND = {
    "john_deere": "RATE",
    "ag_leader": "RATE",
    "raven": "RATE",
    "trimble": "TGT_RATE",
    "case_ih": "RATE",
    "new_holland": "RATE",
    "bourgault": "RATE",
    "vaderstad": "RATE",
    "augmenta": "RATE",
    "climate_fieldview": "RATE",
    "topcon": "RATE",
    "generic": "RATE",
}

#: Character limit for a field name in the DBF format.
DBF_FIELD_LIMIT = 10


def safe_field_names(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    """Truncate names to the DBF limit, resolving collisions.

    Returns
    -------
    (mapping, warnings)
        ``mapping`` goes from the original name to the written one.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    warnings: list[str] = []

    for column in columns:
        name = str(column)[:DBF_FIELD_LIMIT]
        if name in used:
            for suffix in range(1, 100):
                candidate = f"{str(column)[:DBF_FIELD_LIMIT - len(str(suffix))]}{suffix}"
                if candidate not in used:
                    name = candidate
                    break
        if name != str(column):
            warnings.append(f"Field '{column}' written as '{name}' (DBF limit).")
        mapping[column] = name
        used.add(name)
    return mapping, warnings


# ==========================================================================
# Generic outputs
# ==========================================================================

def write_vector(
    dataset: Dataset,
    path: Path,
    driver: str | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Write a dataset as a shapefile, GeoPackage or GeoJSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = dataset.to_geodataframe()

    if columns:
        keep = [c for c in columns if c in gdf.columns]
        gdf = gdf[keep + [gdf.geometry.name]]

    # Dates become text: DBF has no datetime type.
    for column in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[column]):
            gdf[column] = gdf[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    warnings: list[str] = []
    suffix = path.suffix.lower()
    if driver is None:
        driver = {".shp": "ESRI Shapefile", ".gpkg": "GPKG", ".geojson": "GeoJSON",
                  ".json": "GeoJSON"}.get(suffix, "ESRI Shapefile")

    if driver == "ESRI Shapefile":
        mapping, warnings = safe_field_names(
            [c for c in gdf.columns if c != gdf.geometry.name]
        )
        gdf = gdf.rename(columns=mapping)

    gdf.to_file(path, driver=driver)
    return {"path": str(path), "driver": driver, "features": len(gdf), "warnings": warnings}


def write_csv(dataset: Dataset, path: Path, columns: list[str] | None = None) -> dict[str, Any]:
    """Write the table as CSV, with latitude and longitude first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.df
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    lead = [c for c in (sch.LON, sch.LAT) if c in df.columns]
    ordered = lead + [c for c in df.columns if c not in lead]
    df[ordered].to_csv(path, index=False, encoding="utf-8-sig")
    return {"path": str(path), "rows": len(df), "columns": list(ordered)}


def write_geojson(features: dict[str, Any], path: Path) -> dict[str, Any]:
    """Write an already-assembled ``FeatureCollection``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "features": len(features.get("features", []))}


# ==========================================================================
# Prescription
# ==========================================================================

def features_to_geodataframe(features: dict[str, Any], rate_property: str = "rate"):
    """Convert the prescription ``FeatureCollection`` into a ``GeoDataFrame``."""
    import geopandas as gpd
    from shapely.geometry import shape

    rows, geometries = [], []
    for feature in features.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        properties = dict(feature.get("properties") or {})
        if rate_property not in properties:
            continue
        geometries.append(shape(geometry))
        rows.append(properties)

    if not rows:
        raise ValueError(
            f"No feature carrying the rate property '{rate_property}' was found."
        )
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geometries, crs=WGS84)


def write_prescription_shapefile(
    features: dict[str, Any],
    path: Path,
    brand: str = "generic",
    rate_property: str = "rate",
    rate_unit: str = "kg/ha",
    product: str | None = None,
) -> dict[str, Any]:
    """Write the prescription as a polygon shapefile, ready for the monitor."""
    gdf = features_to_geodataframe(features, rate_property)
    field_name = RATE_FIELD_BY_BRAND.get(brand, "RATE")

    rate_values = pd.to_numeric(gdf[rate_property], errors="coerce").round(3)

    # DBF field names are case-insensitive, so a source property called "rate"
    # would collide with the monitor's "RATE" field and the driver would
    # silently rename one of them to RATE_1 — which is exactly the field the
    # monitor would then fail to find. Drop the source column and write the
    # value only under the names the monitor expects.
    written = {field_name.upper(), "RATE", "UNIT"}
    if product:
        written.add("PRODUCT")
    gdf = gdf.drop(columns=[c for c in gdf.columns
                            if c != gdf.geometry.name and c.upper() in written])

    gdf[field_name] = rate_values
    if field_name.upper() != "RATE":
        gdf["RATE"] = rate_values
    gdf["UNIT"] = rate_unit[:10]
    if product:
        gdf["PRODUCT"] = str(product)[:32]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping, warnings = safe_field_names([c for c in gdf.columns if c != gdf.geometry.name])
    gdf.rename(columns=mapping).to_file(path, driver="ESRI Shapefile")

    return {
        "path": str(path),
        "features": len(gdf),
        "rate_field": field_name,
        "unit": rate_unit,
        "warnings": warnings,
        "sidecars": [str(path.with_suffix(ext)) for ext in (".shp", ".shx", ".dbf", ".prj")],
    }


def rasterize_prescription(
    features: dict[str, Any],
    rate_property: str = "rate",
    cell_m: float = 10.0,
) -> dict[str, Any]:
    """Convert prescription polygons into a regular grid in degrees.

    ISOXML only accepts an axis-aligned rectangular grid in geographic
    coordinates. Cells outside every polygon get zero, which is how the
    terminal understands "apply nothing here".
    """
    from shapely.geometry import box
    from shapely.strtree import STRtree

    gdf = features_to_geodataframe(features, rate_property)
    rates = pd.to_numeric(gdf[rate_property], errors="coerce").to_numpy(dtype="float64")

    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    mid_lat = (min_lat + max_lat) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(mid_lat))

    cell_lat = cell_m / m_per_deg_lat
    cell_lon = cell_m / m_per_deg_lon

    cols = max(1, int(np.ceil((max_lon - min_lon) / cell_lon)))
    rows = max(1, int(np.ceil((max_lat - min_lat) / cell_lat)))

    # An absurd grid usually means the cell is far too small for the field.
    if rows * cols > 4_000_000:
        raise ValueError(
            f"The grid would come to {rows * cols:,} cells. Increase the cell size to "
            "produce a file the terminal can actually load."
        )

    grid = np.zeros((rows, cols), dtype="float64")
    geometries = list(gdf.geometry)
    tree = STRtree(geometries)

    for row in range(rows):
        cell_min_lat = min_lat + row * cell_lat
        cell_max_lat = cell_min_lat + cell_lat
        for col in range(cols):
            cell_min_lon = min_lon + col * cell_lon
            cell = box(cell_min_lon, cell_min_lat, cell_min_lon + cell_lon, cell_max_lat)
            candidates = tree.query(cell)
            best_rate, best_area = 0.0, 0.0
            for index in candidates:
                intersection = geometries[index].intersection(cell)
                if intersection.is_empty:
                    continue
                area = intersection.area
                if area > best_area:
                    best_area, best_rate = area, float(rates[index])
            # Only apply where the polygon covers most of the cell.
            grid[row, col] = best_rate if best_area > cell.area * 0.5 else 0.0

    return {
        "grid": grid,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "cell_lon": float(cell_lon),
        "cell_lat": float(cell_lat),
        "rows": rows,
        "cols": cols,
        "cell_m": cell_m,
        "cells_with_rate": int((grid > 0).sum()),
    }


def write_prescription_isoxml(
    features: dict[str, Any],
    out_dir: Path,
    rate_property: str = "rate",
    rate_kind: str = "mass",
    cell_m: float = 10.0,
    task_name: str = "Prescription",
    field_name: str = "Field",
    product_name: str = "Product",
) -> dict[str, Any]:
    """Write the prescription as a TASKDATA folder for an ISOBUS terminal."""
    raster = rasterize_prescription(features, rate_property, cell_m)
    boundary = _outer_ring(features)

    taskdata_dir = isoxml_mod.write_prescription(
        out_dir,
        grid=raster["grid"],
        min_lon=raster["min_lon"],
        min_lat=raster["min_lat"],
        cell_lon=raster["cell_lon"],
        cell_lat=raster["cell_lat"],
        rate_kind=rate_kind,
        task_name=task_name,
        field_name=field_name,
        product_name=product_name,
        boundary=boundary,
    )
    return {
        "path": str(taskdata_dir),
        "rows": raster["rows"],
        "cols": raster["cols"],
        "cells_with_rate": raster["cells_with_rate"],
        "cell_m": cell_m,
        "ddi": f"{isoxml_mod.RX_DDI[rate_kind]:04X}",
        "unit": isoxml_mod.RX_DDI_LABELS[rate_kind],
    }


def _outer_ring(features: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Outer boundary of the polygon set, for the field registry."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    geometries = [
        shape(f["geometry"]) for f in features.get("features", []) if f.get("geometry")
    ]
    if not geometries:
        return None
    merged = unary_union(geometries)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    if merged.geom_type != "Polygon":
        return None
    return [(float(x), float(y)) for x, y in merged.exterior.coords]


def write_prescription_csv(
    features: dict[str, Any],
    path: Path,
    rate_property: str = "rate",
) -> dict[str, Any]:
    """Write the prescription as a CSV of centroids — the fallback format.

    Some older controllers read neither shapefile nor ISOXML, but will import
    a table of points carrying a rate.
    """
    gdf = features_to_geodataframe(features, rate_property)
    centroids = gdf.geometry.representative_point()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["longitude", "latitude", "rate", "area_ha"])
        for point, (_, row) in zip(centroids, gdf.iterrows()):
            writer.writerow([
                f"{point.x:.8f}", f"{point.y:.8f}",
                row.get(rate_property, ""), row.get("area_ha", ""),
            ])
    return {"path": str(path), "rows": len(gdf)}


def bundle(paths: list[Path], zip_path: Path) -> dict[str, Any]:
    """Bundle the generated files into a ZIP to carry on the stick."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    added: list[str] = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in paths:
            item = Path(item)
            if item.is_dir():
                for child in item.rglob("*"):
                    if child.is_file():
                        # A ZIP separates with '/' on every platform, and
                        # zipfile rewrites a backslash it is handed. Naming
                        # the entry in POSIX form keeps what is reported
                        # equal to what the archive holds, on Windows too.
                        entry = child.relative_to(item.parent).as_posix()
                        archive.write(child, entry)
                        added.append(entry)
            elif item.exists():
                archive.write(item, item.name)
                added.append(item.name)
                # A shapefile is a set: without its companions it will not open.
                if item.suffix.lower() == ".shp":
                    for ext in (".shx", ".dbf", ".prj", ".cpg"):
                        sidecar = item.with_suffix(ext)
                        if sidecar.exists():
                            archive.write(sidecar, sidecar.name)
                            added.append(sidecar.name)

    return {"path": str(zip_path), "entries": sorted(set(added))}


# ==========================================================================
# Ready-to-load package for the monitor
# ==========================================================================

def build_package(
    out_dir: Path,
    monitor: str = "generic",
    prescription: dict[str, Any] | None = None,
    boundary: list[tuple[float, float]] | None = None,
    guidance_lines: list[dict[str, Any]] | None = None,
    dataset: Dataset | None = None,
    rate_property: str = "rate",
    rate_kind: str = "mass",
    rate_unit: str = "kg/ha",
    cell_m: float = 10.0,
    field_name: str = "Field",
    task_name: str = "Prescription",
    product_name: str = "Product",
    customer_name: str = "AgroSuite",
    farm_name: str = "Farm",
) -> dict[str, Any]:
    """Assemble the folder that goes on the stick, laid out as the monitor expects.

    It gathers in one place everything the machine needs to run the job: field
    boundary, guidance lines and prescription. Whatever the chosen platform
    does not accept is simply left out, and that is recorded in the result —
    better for the user to know Ag Leader did not get the AB lines than to
    find out in the cab.

    Returns
    -------
    dict
        ``{"folder", "contents", "skipped", "readme", "monitor"}``.
    """
    from . import packages as packages_mod

    profile = packages_mod.get_profile(monitor)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contents: list[dict[str, Any]] = []
    skipped: list[str] = []
    rate_field: str | None = None

    def subfolder(fmt: str) -> Path:
        name = profile.layout.get(fmt, ".")
        target = out_dir if name == "." else out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    # ---------------------------------------------------------------- ISOXML
    wants_isoxml = any(
        profile.preferred.get(artifact) == "isoxml"
        for artifact in ("prescription", "boundary", "guidance")
        if artifact in profile.accepts
    )
    iso_boundary = boundary if "boundary" in profile.accepts else None
    iso_guidance = guidance_lines if "guidance" in profile.accepts else None
    iso_prescription = None

    if prescription and "prescription" in profile.accepts and wants_isoxml:
        raster = rasterize_prescription(prescription, rate_property, cell_m)
        iso_prescription = {
            "grid": raster["grid"],
            "min_lon": raster["min_lon"], "min_lat": raster["min_lat"],
            "cell_lon": raster["cell_lon"], "cell_lat": raster["cell_lat"],
            "rate_kind": rate_kind,
        }
        if iso_boundary is None:
            iso_boundary = _outer_ring(prescription)

    if wants_isoxml and (iso_boundary or iso_guidance or iso_prescription):
        taskdata = isoxml_mod.write_field_setup(
            subfolder("isoxml"),
            field_name=field_name,
            boundary=iso_boundary,
            guidance_lines=iso_guidance,
            customer_name=customer_name,
            farm_name=farm_name,
            prescription=iso_prescription,
            task_name=task_name,
            product_name=product_name,
        )
        parts = []
        if iso_boundary:
            parts.append("boundary")
        if iso_guidance:
            parts.append(f"{len(iso_guidance)} AB line(s)")
        if iso_prescription:
            parts.append("grid prescription")
        contents.append({
            "artifact": "isoxml",
            "path": str(Path(taskdata).relative_to(out_dir)),
            "detail": ", ".join(parts),
            "absolute": str(taskdata),
        })

    # ------------------------------------------------------------ shapefile
    if prescription and "prescription" in profile.accepts:
        target = subfolder("shapefile")
        info = write_prescription_shapefile(
            prescription, target / f"{_slug(task_name)}.shp",
            brand=profile.key, rate_property=rate_property,
            rate_unit=rate_unit, product=product_name,
        )
        rate_field = info["rate_field"]
        contents.append({
            "artifact": "prescription",
            "path": str(Path(info["path"]).relative_to(out_dir)),
            "detail": f"{info['features']} polygons, field {rate_field}",
            "absolute": info["path"],
        })

    if boundary and "boundary" in profile.accepts and \
            profile.preferred.get("boundary") in ("shapefile", "geojson"):
        target = subfolder(profile.preferred["boundary"])
        collection = _boundary_collection(boundary, field_name)
        if profile.preferred["boundary"] == "geojson":
            path = target / f"{_slug(field_name)}_boundary.geojson"
            write_geojson(collection, path)
        else:
            path = target / f"{_slug(field_name)}_boundary.shp"
            _write_boundary_shapefile(collection, path)
        contents.append({
            "artifact": "boundary",
            "path": str(path.relative_to(out_dir)),
            "detail": "field polygon",
            "absolute": str(path),
        })

    if prescription and profile.preferred.get("prescription") == "geojson":
        target = subfolder("geojson")
        path = target / f"{_slug(task_name)}.geojson"
        write_geojson(prescription, path)
        contents.append({
            "artifact": "prescription",
            "path": str(path.relative_to(out_dir)),
            "detail": f"{len(prescription.get('features', []))} features",
            "absolute": str(path),
        })

    # ----------------------------------------------------------------- dados
    if dataset is not None and "data" in profile.accepts:
        target = subfolder("shapefile")
        path = target / f"{_slug(dataset.meta.name)}.shp"
        info = write_vector(dataset, path)
        contents.append({
            "artifact": "data",
            "path": str(path.relative_to(out_dir)),
            "detail": f"{info['features']} points",
            "absolute": info["path"],
        })

    # ---------------------------------------------------- what was left out
    for artifact, value in (("prescription", prescription), ("boundary", boundary),
                            ("guidance", guidance_lines), ("data", dataset)):
        if value is not None and artifact not in profile.accepts:
            skipped.append(
                f"{packages_mod.ARTIFACT_LABELS[artifact]}: the {profile.label} does "
                "not take this kind of file through this route."
            )
    if guidance_lines and "guidance" in profile.accepts and not wants_isoxml:
        skipped.append(
            "Guidance lines: this platform receives AB lines through ISOXML, which is "
            "not part of its preferred layout. Build the generic ISOBUS package if you "
            "need to carry the lines."
        )

    readme = packages_mod.write_readme(out_dir, profile, contents, rate_field, rate_unit)
    return {
        "folder": str(out_dir),
        "monitor": profile.key,
        "monitor_label": profile.label,
        "contents": contents,
        "skipped": skipped,
        "readme": str(readme),
        "instructions": list(profile.instructions),
    }


def _slug(text: str) -> str:
    """Safe filename: no accents, spaces or special characters."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_only = "".join(c for c in normalized if not unicodedata.combining(c))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ascii_only)
    return safe.strip("_") or "file"


def _boundary_collection(ring: list[tuple[float, float]], name: str) -> dict[str, Any]:
    """Wrap a boundary ring as a ``FeatureCollection``."""
    closed = [list(p) for p in ring]
    if closed and closed[0] != closed[-1]:
        closed.append(closed[0])
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [closed]},
            "properties": {"NAME": name[:32], "TYPE": "boundary"},
        }],
    }


def _write_boundary_shapefile(collection: dict[str, Any], path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import shape

    rows, geometries = [], []
    for feature in collection["features"]:
        geometries.append(shape(feature["geometry"]))
        rows.append(feature.get("properties") or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geometries, crs=WGS84).to_file(
        path, driver="ESRI Shapefile"
    )
