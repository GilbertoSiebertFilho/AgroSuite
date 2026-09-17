"""Working with QGIS.

Two directions, both of which come up in practice.

**Out of AgroSuite.** After cleaning and analysing, the layers are worth
looking at in a proper GIS — to compare against soil maps, to sketch zones by
hand, to lay out a prescription over imagery. Everything goes into one
GeoPackage, because a GeoPackage is a single file that holds many layers and
QGIS opens it natively. A ``.qgs`` project file is written alongside it, which
saves the step of adding each layer by hand.

**Into AgroSuite.** When the zones or the prescription were drawn in QGIS, the
result has to reach the monitor. Reading the project's layer list and pulling
the ones that matter is quicker than exporting each by hand, and it keeps the
names the user gave them.

On what is guaranteed and what is convenience: the GeoPackage is the reliable
half. It is an OGC standard, GDAL writes it, and QGIS reads it the same way on
every version. The ``.qgs`` file is a convenience — its schema is QGIS's own
and changes between releases — so the project file is written as a plain layer
list and nothing depends on it being honoured.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.crs import WGS84
from ..core.dataset import Dataset

#: QGIS project schema version this writer targets. Newer QGIS reads older
#: projects; the version is declared so it says so rather than guessing.
QGIS_VERSION = "3.28.0"


# ==========================================================================
# Out of AgroSuite
# ==========================================================================

def write_geopackage(
    layers: dict[str, Dataset],
    path: Path,
    metric: bool = False,
) -> dict[str, Any]:
    """Write every layer into one GeoPackage.

    Parameters
    ----------
    layers:
        Map of ``layer name -> Dataset``. Names become the GeoPackage layer
        names, so they are what shows up in the QGIS layer panel.
    metric:
        Write in the projected CRS instead of WGS84. Useful when the next step
        in QGIS involves measuring or buffering, which want metres.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    written: list[dict[str, Any]] = []
    for name, dataset in layers.items():
        gdf = dataset.to_geodataframe(metric=metric)

        # A GeoPackage column cannot be a pandas datetime in every GDAL build;
        # text is read the same way everywhere.
        for column in gdf.columns:
            if str(gdf[column].dtype).startswith("datetime"):
                gdf[column] = gdf[column].dt.strftime("%Y-%m-%d %H:%M:%S")

        renamed = _distinct_field_names([c for c in gdf.columns if c != gdf.geometry.name])
        if renamed:
            gdf = gdf.rename(columns=renamed)

        safe = _layer_name(name)
        gdf.to_file(path, layer=safe, driver="GPKG")
        record: dict[str, Any] = {
            "layer": safe,
            "features": len(gdf),
            "geometry": str(gdf.geom_type.dropna().iloc[0]) if len(gdf) else "None",
            "crs": gdf.crs.to_string() if gdf.crs else None,
        }
        if renamed:
            record["renamed"] = renamed
        written.append(record)

    return {"path": str(path), "layers": written}


def write_project(
    gpkg_path: Path,
    layers: list[dict[str, Any]],
    project_path: Path,
    title: str = "AgroSuite",
    value_field: str = "value",
) -> dict[str, Any]:
    """Write a ``.qgs`` project that opens the GeoPackage's layers.

    The project file lists the layers and nothing more: no symbology, no
    print layouts, no processing history. Anything richer would depend on
    QGIS's own schema, which changes between releases, and a project that
    fails to open is worse than one the user styles themselves.
    """
    gpkg_path = Path(gpkg_path)
    project_path = Path(project_path)
    project_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("qgis", {
        "projectname": title,
        "version": QGIS_VERSION,
        "saveUser": "agrosuite",
        "saveDateTime": datetime.now().isoformat(timespec="seconds"),
    })
    ET.SubElement(root, "homePath", {"path": str(gpkg_path.parent)})
    ET.SubElement(root, "title").text = title

    # Project CRS. Everything AgroSuite writes to a GeoPackage is WGS84 unless
    # asked otherwise, and QGIS reprojects on the fly from there.
    srs = ET.SubElement(root, "projectCrs")
    crs = ET.SubElement(srs, "spatialrefsys")
    ET.SubElement(crs, "authid").text = WGS84
    ET.SubElement(crs, "description").text = "WGS 84"
    ET.SubElement(crs, "geographicflag").text = "true"

    project_layers = ET.SubElement(root, "projectlayers")
    tree_group = ET.SubElement(root, "layer-tree-group")
    order = ET.SubElement(root, "layerorder")

    for index, layer in enumerate(layers):
        layer_id = f"{layer['layer']}_{index}"
        source = f"{gpkg_path}|layername={layer['layer']}"

        maplayer = ET.SubElement(project_layers, "maplayer", {
            "type": "vector",
            "geometry": layer.get("geometry", "Point"),
            "hasScaleBasedVisibilityFlag": "0",
        })
        ET.SubElement(maplayer, "id").text = layer_id
        ET.SubElement(maplayer, "datasource").text = source
        ET.SubElement(maplayer, "layername").text = layer["layer"]
        ET.SubElement(maplayer, "provider", {"encoding": "UTF-8"}).text = "ogr"
        layer_srs = ET.SubElement(maplayer, "srs")
        layer_crs = ET.SubElement(layer_srs, "spatialrefsys")
        ET.SubElement(layer_crs, "authid").text = layer.get("crs") or WGS84

        ET.SubElement(tree_group, "layer-tree-layer", {
            "id": layer_id, "name": layer["layer"],
            "source": source, "providerKey": "ogr",
            "checked": "Qt::Checked", "expanded": "1",
        })
        ET.SubElement(order, "layer", {"id": layer_id})

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(project_path, encoding="UTF-8", xml_declaration=True)
    return {"path": str(project_path), "layers": len(layers)}


def export_for_qgis(
    layers: dict[str, Dataset],
    out_dir: Path,
    name: str = "agrosuite",
    metric: bool = False,
) -> dict[str, Any]:
    """Write the GeoPackage and the project file together."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg = out_dir / f"{_layer_name(name)}.gpkg"
    package = write_geopackage(layers, gpkg, metric=metric)
    project = write_project(
        gpkg, package["layers"], out_dir / f"{_layer_name(name)}.qgs", title=name
    )
    return {
        "folder": str(out_dir),
        "geopackage": package,
        "project": project,
        "note": (
            "Open the .qgs to get every layer at once. If your QGIS version does "
            "not take the project file, open the .gpkg directly — it holds the "
            "same layers and is a standard format."
        ),
    }


def _distinct_field_names(columns: list[str]) -> dict[str, str]:
    """Renames that keep every column when GeoPackage folds their case.

    A reader keeps the source column beside the canonical one when both
    exist — a John Deere export has ``Product`` and the schema's
    ``product`` — and a DataFrame holds both. A GeoPackage is SQLite, whose
    column names are case-insensitive, so the second one fails to be added
    and the whole export stops. The later column takes a numbered suffix,
    the way a second export folder does, rather than being dropped.
    """
    renamed: dict[str, str] = {}
    taken: set[str] = set()
    for column in columns:
        name = str(column)
        if name.lower() in taken:
            n = 2
            while f"{name}_{n}".lower() in taken:
                n += 1
            renamed[column] = f"{name}_{n}"
            name = f"{name}_{n}"
        taken.add(name.lower())
    return renamed


def _layer_name(raw: str) -> str:
    """A layer name a GeoPackage and QGIS both accept."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(raw))
    ascii_only = "".join(c for c in normalized if not unicodedata.combining(c))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ascii_only)
    safe = "_".join(filter(None, safe.split("_")))
    return safe[:60] or "layer"


# ==========================================================================
# Into AgroSuite
# ==========================================================================

def read_project(path: Path) -> dict[str, Any]:
    """List the layers of a QGIS project, without loading their data.

    Handles both ``.qgs`` (plain XML) and ``.qgz`` (a zip holding one). Layer
    sources are resolved against the project's folder, because QGIS writes
    them relative and an absolute path from another machine would not exist
    here anyway.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {path}")

    if path.suffix.lower() == ".qgz":
        with zipfile.ZipFile(path) as archive:
            inner = next((n for n in archive.namelist() if n.lower().endswith(".qgs")), None)
            if inner is None:
                raise ValueError("The .qgz holds no .qgs project file.")
            root = ET.fromstring(archive.read(inner))
    else:
        root = ET.parse(path).getroot()

    base = path.parent
    layers: list[dict[str, Any]] = []
    for maplayer in root.iter("maplayer"):
        datasource = (maplayer.findtext("datasource") or "").strip()
        if not datasource:
            continue
        name = maplayer.findtext("layername") or "layer"
        provider = maplayer.findtext("provider") or ""

        # "path|layername=x" points at one layer inside a container.
        file_part, _, options = datasource.partition("|")
        sub_layer = None
        for option in options.split("|"):
            if option.startswith("layername="):
                sub_layer = option.split("=", 1)[1]

        resolved = Path(file_part)
        if not resolved.is_absolute():
            resolved = (base / file_part).resolve()

        layers.append({
            "name": name,
            "provider": provider,
            "path": str(resolved),
            "sub_layer": sub_layer,
            "exists": resolved.exists(),
            "geometry": maplayer.get("geometry"),
            "importable": provider == "ogr" and resolved.exists(),
        })

    return {
        "path": str(path),
        "title": root.findtext("title") or path.stem,
        "version": root.get("version"),
        "layers": layers,
        "importable": [layer for layer in layers if layer["importable"]],
    }


def read_project_layer(layer: dict[str, Any]) -> Dataset:
    """Import one layer listed by :func:`read_project`."""
    from . import readers

    source = Path(layer["path"])
    if not source.exists():
        raise FileNotFoundError(
            f"The project points at {source}, which is not on this machine. "
            "Layers stored elsewhere have to be copied over first."
        )

    if layer.get("sub_layer"):
        import geopandas as gpd
        import pandas as pd

        from ..core import schema as sch
        from ..core.dataset import DatasetMeta

        gdf = gpd.read_file(source, layer=layer["sub_layer"])
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84, allow_override=True)
        elif gdf.crs.to_string() != WGS84:
            gdf = gdf.to_crs(WGS84)

        polygonal = bool(set(gdf.geom_type.dropna().unique()) & {"Polygon", "MultiPolygon"})
        df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
        points = gdf.geometry.representative_point()
        df[sch.LON] = points.x.to_numpy()
        df[sch.LAT] = points.y.to_numpy()

        dataset = readers._build_dataset(
            df, source, "qgis",
            [f"Imported from the QGIS project layer '{layer['name']}'."],
            geometry=list(gdf.geometry) if polygonal else None,
        )
        dataset.meta.name = layer["name"]
        return dataset

    dataset = readers.read_shapefile(source)
    dataset.meta.name = layer["name"]
    dataset.meta.notes.append(f"Imported from the QGIS project layer '{layer['name']}'.")
    return dataset
