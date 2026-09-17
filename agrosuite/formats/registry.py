"""Format detection and dispatch to the right reader.

The user throws whatever they have at the app: a loose ``.shp``, the whole
ZIP the monitor wrote to the stick, a ``TASKDATA`` folder, a log CSV or an
Augmenta GeoJSON. This module decides what each of those is and calls the
appropriate reader, without requiring the user to know the format's technical
name.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..core.dataset import Dataset
from . import isoxml as isoxml_mod
from . import johndeere as jd_mod
from . import raster as raster_mod
from . import readers

#: Extensions accepted on import, grouped by family.
VECTOR_EXT = {".shp", ".gpkg", ".geojson", ".json", ".kml", ".kmz", ".gml"}
TABULAR_EXT = {".csv", ".txt", ".dat", ".log", ".tsv"}
EXCEL_EXT = {".xlsx", ".xls", ".xlsm"}
ARCHIVE_EXT = {".zip"}
#: Elevation rasters (DEM). The only raster the app reads: a GeoTIFF is what
#: every elevation source exports and what QGIS exchanges.
RASTER_EXT = {".tif", ".tiff"}

#: Extensions that make up a shapefile, used when extracting from a ZIP.
SHAPEFILE_SIDECARS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"}

#: QGIS projects are listed so they show up in the file browser, but they are
#: handled by :mod:`agrosuite.formats.qgis` rather than by a reader here: a
#: project names layers, it does not hold them.
QGIS_EXT = {".qgs", ".qgz"}

#: Saved AgroSuite projects. Listed for the same reason as QGIS projects — the
#: user should find them in the file browser — but opened by
#: :mod:`agrosuite.app.persist`, because a project is a whole session, not a
#: dataset to add to one.
PROJECT_EXT = {".agrosuite"}

ALL_IMPORT_EXT = (
    VECTOR_EXT | TABULAR_EXT | EXCEL_EXT | ARCHIVE_EXT | RASTER_EXT
    | QGIS_EXT | PROJECT_EXT
    | {".xml", ".iso"}
)


@dataclass
class DetectedSource:
    """Result of inspecting a path."""

    kind: str          # 'shapefile' | 'geojson' | 'csv' | 'excel' | 'kml' | 'isoxml' | 'archive' | 'raster'
    path: Path
    label: str
    detail: str = ""


def _system_entry(path: Path) -> bool:
    """A dot-file, or anything under ``__MACOSX``: what macOS writes beside
    the user's files on a stick and into a zip. ``._Boundary.shp`` is the
    resource fork of ``Boundary.shp``, not a shapefile, and sorted by name
    it comes first."""
    return path.name.startswith(".") or "__MACOSX" in path.parts


def detect(path: str | Path) -> DetectedSource:
    """Classify a path — file, folder or ZIP — without reading all of it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        if isoxml_mod.find_taskdata(path):
            return DetectedSource("isoxml", path, "ISOXML folder (TASKDATA)")

        # The John Deere card comes before the loose-shapefile search: the
        # GS2/GS3/JD-Data tree says which layer is a boundary and which is a
        # prescription, which a bare .shp in a folder would not.
        card_root, generation = jd_mod.find_card_root(path)
        if card_root is not None:
            return DetectedSource(
                "jd_card", path, f"John Deere card — {generation}",
                detail=f"root at {card_root.name}",
            )

        shapefiles = sorted(p for p in path.glob("*.shp") if not _system_entry(p))
        if shapefiles:
            return DetectedSource(
                "shapefile", shapefiles[0], "Folder holding a shapefile",
                detail=f"{len(shapefiles)} shapefile(s); using {shapefiles[0].name}",
            )
        # A zipped DEM lands here: the ZIP was unpacked into a folder.
        rasters = sorted(p for p in path.iterdir() if p.suffix.lower() in RASTER_EXT)
        if rasters:
            return DetectedSource(
                "raster", rasters[0], "Folder holding a GeoTIFF",
                detail=f"{len(rasters)} raster(s); using {rasters[0].name}",
            )
        raise ValueError(
            f"Folder '{path.name}' holds neither a TASKDATA.XML, a shapefile nor a GeoTIFF."
        )

    suffix = path.suffix.lower()
    if suffix in ARCHIVE_EXT:
        return DetectedSource("archive", path, "Compressed archive")
    if suffix == ".shp":
        return DetectedSource("shapefile", path, "Shapefile")
    if suffix in (".geojson", ".json"):
        return DetectedSource("geojson", path, "GeoJSON / JSON")
    if suffix in (".kml", ".kmz"):
        return DetectedSource("kml", path, "KML / KMZ")
    if suffix == ".gpkg":
        return DetectedSource("shapefile", path, "GeoPackage")
    if suffix in TABULAR_EXT:
        return DetectedSource("csv", path, "Text table (CSV/TXT)")
    if suffix in EXCEL_EXT:
        return DetectedSource("excel", path, "Excel workbook")
    if suffix in RASTER_EXT:
        return DetectedSource("raster", path, "Elevation raster (GeoTIFF)")
    if suffix in (".xml", ".iso"):
        if path.name.upper() == "TASKDATA.XML":
            return DetectedSource("isoxml", path, "TASKDATA.XML (ISOXML)")
        found = isoxml_mod.find_taskdata(path.parent)
        if found:
            return DetectedSource("isoxml", found, "ISOXML in the same folder")
        raise ValueError(f"XML not recognized as ISOXML: {path.name}")

    if suffix in QGIS_EXT:
        raise ValueError(
            "That is a QGIS project. Use 'Open a QGIS project' on the Export tab, "
            "which lists its layers and lets you pick which ones to bring in."
        )

    if suffix in PROJECT_EXT:
        raise ValueError(
            "That is a saved AgroSuite project. Use 'Open project', which restores "
            "the whole session — datasets, cleaning results, roles and prices — "
            "rather than importing it as one more file."
        )

    raise ValueError(
        f"Extension '{suffix or path.name}' is not supported on import. "
        f"Accepted formats: {', '.join(sorted(ALL_IMPORT_EXT))}."
    )


def extract_archive(path: Path, workdir: Path | None = None) -> Path:
    """Unpack a ZIP into a temporary folder and return the useful root.

    If the ZIP holds a single folder at its root — the usual shape of monitor
    exports — that folder is returned directly, so the next detection step
    sees the contents rather than the wrapper.
    """
    path = Path(path)
    target = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="agrosuite_zip_"))
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Guard against absolute paths or ".." inside the ZIP.
            name = Path(member.filename)
            parts = [p for p in name.parts if p not in ("", ".", "..") and not p.startswith("/")]
            if not parts:
                continue
            destination = target.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(destination, "wb") as out:
                shutil.copyfileobj(source, out)

    entries = [p for p in target.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return target


def read_any(path: str | Path, brand_hint: str | None = None) -> Dataset:
    """Import any supported source, resolving ZIPs and folders."""
    source = detect(path)

    if source.kind == "archive":
        extracted = extract_archive(source.path)
        inner = detect(extracted)
        if inner.kind == "archive":
            raise ValueError("Nested ZIP not supported; unpack it before importing.")
        dataset = _dispatch(inner, brand_hint)
        # The ZIP's name says more than the inner file's.
        dataset.meta.notes.append(f"Extracted from {Path(path).name}.")
        return dataset

    return _dispatch(source, brand_hint)


def _dispatch(source: DetectedSource, brand_hint: str | None) -> Dataset:
    readers_by_kind = {
        "shapefile": readers.read_shapefile,
        "geojson": readers.read_geojson,
        "csv": readers.read_tabular,
        "excel": readers.read_excel,
        "kml": readers.read_kml,
        "isoxml": isoxml_mod.read_isoxml,
        "jd_card": read_jd_card,
        "raster": raster_mod.read_raster,
    }
    reader = readers_by_kind.get(source.kind)
    if reader is None:
        raise ValueError(f"No reader for type '{source.kind}'.")
    return reader(source.path, brand_hint)


def read_jd_card(path: Path, brand_hint: str | None = None) -> Dataset:
    """Import the most useful layer from a John Deere card.

    A card can hold several layers; operation data and prescriptions matter
    more than the boundary, so the choice follows that order. The full
    inventory stays in the metadata for the interface to list the rest.
    """
    inv = jd_mod.inventory(Path(path))
    layers = jd_mod.readable_layers(inv)

    if not layers:
        proprietary = ", ".join(sorted({e["kind"] for e in inv.proprietary}))
        raise ValueError(
            f"The card was recognized ({inv.generation}), but no file in an open "
            "format was found inside it."
            + (f" What is there is proprietary: {proprietary}." if proprietary else "")
            + " In SMS, export again choosing shapefile instead of GreenStar."
        )

    order = {"data": 0, "prescription": 1, "boundary": 2, "guidance": 3}
    layers.sort(key=lambda layer: order.get(layer["role"], 9))
    chosen = layers[0]

    dataset = _dispatch(detect(Path(chosen["path"])), brand_hint or "john_deere")
    dataset.meta.brand = "john_deere"
    dataset.meta.brand_label = "John Deere"
    dataset.meta.notes.append(
        f"Imported from a {inv.generation} card: {chosen['relative']}."
    )
    if len(layers) > 1:
        dataset.meta.notes.append(
            "Other layers on the card: "
            + ", ".join(f"{layer['relative']} ({layer['role']})" for layer in layers[1:])
            + "."
        )
    dataset.meta.extra["jd_card"] = inv.to_dict()
    dataset.meta.extra["jd_card_layers"] = layers
    return dataset


def inspect(path: str | Path) -> dict:
    """Light inspection for the interface to show before actually importing."""
    source = detect(path)
    info = {
        "kind": source.kind,
        "label": source.label,
        "detail": source.detail,
        "path": str(source.path),
        "name": Path(path).name,
    }
    if source.kind == "jd_card":
        inv = jd_mod.inventory(source.path)
        info["john_deere_card"] = inv.to_dict()
        info["layers"] = jd_mod.readable_layers(inv)
        info["detail"] = inv.summary()

    if source.kind == "raster":
        try:
            info["raster"] = raster_mod.describe(source.path)
            info["detail"] = info["raster"]["summary"]
        except Exception as exc:
            info["detail"] = f"Raster unreadable: {exc}"

    if source.kind == "isoxml":
        try:
            # detect() hands back the folder for an ISOXML tree; the parser
            # wants the TASKDATA.XML inside it, wherever it sits.
            taskdata = isoxml_mod.find_taskdata(source.path)
            if taskdata is None:
                raise ValueError(f"no TASKDATA.XML under '{source.path.name}'")
            catalog = isoxml_mod.parse_taskdata(taskdata)
            info["isoxml"] = {
                "version": catalog["version"],
                "software": catalog["software"],
                "fields": [f["name"] for f in catalog["fields"]],
                "tasks": [t["name"] for t in catalog["tasks"]],
                "products": list(catalog["products"].values()),
            }
        except Exception as exc:
            info["detail"] = f"TASKDATA.XML unreadable: {exc}"
    return info
