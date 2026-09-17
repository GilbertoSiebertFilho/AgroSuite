"""Checking the files before taking them to the monitor.

Finding out the map will not open happens in the worst possible place: engine
running, operator waiting, USB stick in hand. This module checks what can be
checked on the computer — the known causes of rejection, one by one — and
returns a report at three levels:

``ok``
    The item was checked and conforms.
``warning``
    Works in most cases, but depends on firmware or display configuration;
    worth confirming on screen.
``fail``
    The monitor will refuse this file. Do not take it as it is.

What **cannot** be guaranteed from here: firmware version, import menus and
proprietary formats. That is why no check claims "this will work" — it claims
that a known cause of failure has been ruled out.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

#: Extensions that must travel with a .shp for the monitor to open it.
REQUIRED_SIDECARS = (".shx", ".dbf", ".prj")

#: Character limit for a field name in DBF.
DBF_FIELD_LIMIT = 10

#: Above this, many terminals take far too long or refuse to load.
MAX_RX_FEATURES = 20_000
MAX_GRID_CELLS = 2_000_000
MAX_FILE_MB = 32


@dataclass
class Check:
    """A single checked item."""

    item: str
    status: str          # 'ok' | 'warning' | 'fail'
    message: str
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ok(item, message):
    return Check(item, "ok", message)


def _warn(item, message, fix=""):
    return Check(item, "warning", message, fix)


def _fail(item, message, fix=""):
    return Check(item, "fail", message, fix)


# ==========================================================================
# Shapefile
# ==========================================================================

def validate_shapefile(
    shp_path: Path,
    rate_field: str | None = None,
    expect_polygons: bool = True,
) -> list[Check]:
    """Check a shapefile against what monitors require."""
    import geopandas as gpd

    shp_path = Path(shp_path)
    checks: list[Check] = []

    if not shp_path.exists():
        return [_fail("File", f"{shp_path.name} does not exist.")]

    # --- companion files
    missing = [ext for ext in REQUIRED_SIDECARS if not shp_path.with_suffix(ext).exists()]
    if missing:
        checks.append(_fail(
            "File set",
            f"Missing {', '.join(missing)} alongside {shp_path.name}.",
            "Always copy .shp, .shx, .dbf and .prj together — the monitor will not "
            "open the map without all four.",
        ))
    else:
        checks.append(_ok("File set", ".shp, .shx, .dbf and .prj all present."))

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        return checks + [_fail("Readback", f"The file could not be read back: {exc}")]

    # --- geometry
    if gdf.empty:
        checks.append(_fail("Features", "The shapefile holds no features at all.",
                            "Check the parameters that produced the map."))
        return checks

    geom_types = set(gdf.geom_type.dropna().unique())
    if expect_polygons:
        if geom_types <= {"Polygon", "MultiPolygon"}:
            checks.append(_ok("Geometry type",
                              f"{len(gdf)} polygon(s) — what a monitor expects in a prescription."))
        else:
            checks.append(_fail(
                "Geometry type",
                f"Geometry is {', '.join(sorted(geom_types))}; a prescription must be polygons.",
                "Build the prescription from the trial layout or a grid, not from points.",
            ))

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        checks.append(_warn(
            "Polygon validity",
            f"{invalid} polygon(s) with invalid geometry (self-intersection or open ring).",
            "Some monitors skip the invalid feature; others refuse the whole file.",
        ))
    else:
        checks.append(_ok("Polygon validity", "Every geometry is valid."))

    empty = int(gdf.geometry.is_empty.sum())
    if empty:
        checks.append(_fail("Empty geometries", f"{empty} feature(s) with no geometry.",
                            "Remove them before exporting."))

    if len(gdf) > MAX_RX_FEATURES:
        checks.append(_warn(
            "Feature count",
            f"{len(gdf):,} polygons. Past roughly {MAX_RX_FEATURES:,} many terminals "
            "take minutes to load, or give up.",
            "Increase the cell size or simplify the zones.",
        ))
    else:
        checks.append(_ok("Feature count", f"{len(gdf)} polygon(s)."))

    # --- projection
    if gdf.crs is None:
        checks.append(_fail("Projection", "No .prj — the monitor cannot tell where the map is.",
                            "Export again from AgroSuite, which always writes the .prj."))
    elif gdf.crs.to_epsg() == 4326:
        checks.append(_ok("Projection", "Geographic WGS84 (EPSG:4326), accepted by every monitor."))
    else:
        checks.append(_warn(
            "Projection",
            f"CRS is {gdf.crs.to_string()} rather than WGS84. Older displays tend to "
            "assume WGS84 and shift the map.",
            "Re-export in EPSG:4326 if the map shows up in the wrong place.",
        ))

    # --- plausible coordinates
    bounds = gdf.total_bounds
    if gdf.crs is not None and gdf.crs.to_epsg() == 4326:
        if not (-180 <= bounds[0] <= 180 and -90 <= bounds[1] <= 90
                and -180 <= bounds[2] <= 180 and -90 <= bounds[3] <= 90):
            checks.append(_fail("Coordinates", "Values outside the latitude/longitude range."))
        else:
            checks.append(_ok(
                "Location",
                f"Between {bounds[1]:.4f}, {bounds[0]:.4f} and {bounds[3]:.4f}, {bounds[2]:.4f}.",
            ))

    # --- field names
    attributes = [c for c in gdf.columns if c != gdf.geometry.name]
    too_long = [c for c in attributes if len(c) > DBF_FIELD_LIMIT]
    if too_long:
        checks.append(_fail(
            "Field names",
            f"Fields longer than {DBF_FIELD_LIMIT} characters: {', '.join(too_long)}.",
            "DBF truncates silently and can collapse two fields into one.",
        ))
    else:
        checks.append(_ok("Field names", f"{len(attributes)} field(s) within the DBF limit."))

    # --- rate field
    if rate_field:
        if rate_field not in gdf.columns:
            checks.append(_fail(
                "Rate field",
                f"Field '{rate_field}' does not exist in the file. "
                f"Available fields: {', '.join(attributes)}.",
                "Without it the monitor cannot find the rate at import time.",
            ))
        else:
            series = gdf[rate_field]
            if not np.issubdtype(series.dtype, np.number):
                checks.append(_fail(
                    "Rate field",
                    f"'{rate_field}' has type {series.dtype} and must be numeric.",
                ))
            else:
                values = series.to_numpy(dtype="float64")
                nulls = int(np.isnan(values).sum())
                negatives = int((values < 0).sum())
                zeros = int((values == 0).sum())
                if nulls:
                    checks.append(_fail(
                        "Null rates", f"{nulls} polygon(s) with no rate value.",
                        "Write an explicit zero where nothing should be applied.",
                    ))
                if negatives:
                    checks.append(_fail("Negative rates", f"{negatives} polygon(s) with a negative rate."))
                if not nulls and not negatives:
                    detail = (f"{np.nanmin(values):.4g} to {np.nanmax(values):.4g}"
                              f" across {len(values)} polygons")
                    if zeros:
                        detail += f"; {zeros} at zero (no-application area)"
                    checks.append(_ok("Rate field", f"'{rate_field}': {detail}."))
                if np.nanmax(values) > 1e6:
                    checks.append(_warn(
                        "Rate magnitude",
                        f"Maximum rate of {np.nanmax(values):.4g} — too high for most "
                        "inputs. Check that the unit is right.",
                    ))

    # --- non-ASCII text in the DBF
    text_columns = [c for c in attributes if gdf[c].dtype == object]
    has_accents = any(
        isinstance(v, str) and any(ord(ch) > 127 for ch in v)
        for c in text_columns for v in gdf[c].dropna().head(200)
    )
    cpg = shp_path.with_suffix(".cpg")
    if has_accents and not cpg.exists():
        checks.append(_warn(
            "Character encoding",
            "There is non-ASCII text and no .cpg declaring the encoding.",
            "Older displays show garbled characters; avoid accents in names.",
        ))
    elif has_accents:
        checks.append(_ok("Character encoding",
                          f"Non-ASCII text with the encoding declared in {cpg.name}."))

    # --- size
    total_mb = sum(
        shp_path.with_suffix(ext).stat().st_size
        for ext in (".shp", ".shx", ".dbf", ".prj")
        if shp_path.with_suffix(ext).exists()
    ) / 1e6
    if total_mb > MAX_FILE_MB:
        checks.append(_warn(
            "Size", f"{total_mb:.1f} MB across the file set. Older terminals stall past "
            f"about {MAX_FILE_MB} MB.",
            "Cut the number of polygons or increase the cell size.",
        ))
    else:
        checks.append(_ok("Size", f"{total_mb:.2f} MB across the file set."))

    return checks


# ==========================================================================
# ISOXML
# ==========================================================================

def validate_taskdata(taskdata_dir: Path) -> list[Check]:
    """Check a TASKDATA folder against ISO 11783-10."""
    taskdata_dir = Path(taskdata_dir)
    checks: list[Check] = []

    if taskdata_dir.name.upper() != "TASKDATA":
        checks.append(_fail(
            "Folder name",
            f"The folder is called '{taskdata_dir.name}' and must be called TASKDATA.",
            "The terminal looks for exactly that name at the root of the stick.",
        ))
    else:
        checks.append(_ok("Folder name", "TASKDATA, as the standard requires."))

    xml_path = next(
        (p for p in taskdata_dir.iterdir() if p.name.upper() == "TASKDATA.XML"), None
    )
    if xml_path is None:
        return checks + [_fail("TASKDATA.XML", "The main file is missing.")]
    if xml_path.name != "TASKDATA.XML":
        checks.append(_warn(
            "TASKDATA.XML",
            f"The file is named '{xml_path.name}'. Some terminals only recognize "
            "the name in full upper case.",
            "Rename it to TASKDATA.XML.",
        ))

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        return checks + [_fail("TASKDATA.XML", f"Invalid XML: {exc}")]

    if root.tag != "ISO11783_TaskData":
        checks.append(_fail("Root element",
                            f"Root is '{root.tag}' instead of ISO11783_TaskData."))
    else:
        version = f"{root.get('VersionMajor', '?')}.{root.get('VersionMinor', '?')}"
        checks.append(_ok("Structure", f"ISO 11783-10 version {version}."))

    # --- field, boundary, lines
    fields = list(root.iter("PFD"))
    if not fields:
        checks.append(_warn("Field", "No PFD declared; the terminal will have no field to attach to."))
    for pfd in fields:
        name = pfd.get("C") or pfd.get("A")
        rings = [lsg for pln in pfd.findall("PLN") for lsg in pln.findall("LSG")]
        closed_ok = True
        for lsg in rings:
            points = [(p.get("D"), p.get("C")) for p in lsg.findall("PNT")]
            if len(points) < 3:
                closed_ok = False
            elif points[0] != points[-1]:
                closed_ok = False
        if rings:
            if closed_ok:
                checks.append(_ok("Boundary", f"Field '{name}': {len(rings)} closed ring(s)."))
            else:
                checks.append(_warn(
                    "Boundary",
                    f"Field '{name}': one of the rings does not close on its first point.",
                    "Most terminals close it themselves, but some refuse the file.",
                ))

        for ggp in pfd.findall("GGP"):
            for gpn in ggp.findall("GPN"):
                types = {
                    p.get("A") for lsg in gpn.findall("LSG") for p in lsg.findall("PNT")
                }
                if gpn.get("C") == "1" and not {"6", "7"} <= types:
                    checks.append(_fail(
                        "AB line",
                        f"'{gpn.get('B')}' is typed as an AB line but carries no pair of "
                        "reference points (types 6 and 7).",
                        "Without both points the terminal cannot generate the passes.",
                    ))
                else:
                    checks.append(_ok("AB line", f"'{gpn.get('B')}' declares points A and B."))

    # --- prescription grid
    for grd in root.iter("GRD"):
        name = grd.get("G")
        declared_length = int(grd.get("H") or 0)
        rows = int(grd.get("F") or 0)
        cols = int(grd.get("E") or 0)
        grid_type = grd.get("I")

        bin_path = next(
            (p for p in taskdata_dir.iterdir()
             if p.stem.upper() == (name or "").upper() and p.suffix.upper() == ".BIN"),
            None,
        )
        if bin_path is None:
            checks.append(_fail("Grid", f"The binary {name}.BIN is not in the folder."))
            continue

        actual = bin_path.stat().st_size
        expected = rows * cols * (4 if grid_type == "2" else 1)
        if actual != expected:
            checks.append(_fail(
                "Grid",
                f"{bin_path.name} holds {actual} bytes, but {cols}x{rows} cells of type "
                f"{grid_type} need {expected}.",
                "The terminal reads the grid shifted and applies the rate in the wrong place.",
            ))
        elif declared_length and declared_length != actual:
            checks.append(_warn(
                "Grid",
                f"The length attribute declares {declared_length} bytes and the file holds {actual}.",
                "Some terminals trust the declared value.",
            ))
        else:
            values = np.frombuffer(bin_path.read_bytes(),
                                   dtype="<u4" if grid_type == "2" else "<u1")
            with_rate = int((values > 0).sum())
            checks.append(_ok(
                "Grid",
                f"{cols}x{rows} cells, {actual} bytes match; "
                f"{with_rate} cell(s) carry a rate.",
            ))
            if with_rate == 0:
                checks.append(_fail(
                    "Grid with no rate",
                    "Every cell is zero — the machine would apply nothing.",
                    "Check the cell size: cells that are too small may never reach half "
                    "coverage inside any polygon.",
                ))
            if rows * cols > MAX_GRID_CELLS:
                checks.append(_warn(
                    "Grid size",
                    f"{rows * cols:,} cells. Older terminals take far too long to load.",
                    "Increase the cell size.",
                ))

        # The rate needs a declared DDI, or the terminal cannot tell what it is.
        task = next((t for t in root.iter("TSK") if grd in list(t.iter("GRD"))), None)
        pdv = list(task.iter("PDV")) if task is not None else []
        if not pdv:
            checks.append(_fail(
                "Rate unit",
                "The task declares no PDV carrying the DDI of the applied quantity.",
                "Without the DDI, the terminal cannot tell whether the number is kg/ha, "
                "L/ha or seeds/ha.",
            ))
        else:
            checks.append(_ok("Rate unit",
                              f"DDI {pdv[0].get('A')} declared in the treatment zone."))

    return checks


# ==========================================================================
# Whole package
# ==========================================================================

def validate_package(folder: Path, monitor: str = "generic") -> dict[str, Any]:
    """Check every file in a package and summarize the result."""
    from . import packages as packages_mod

    folder = Path(folder)
    profile = packages_mod.get_profile(monitor)
    groups: list[dict[str, Any]] = []

    for taskdata in sorted(folder.rglob("TASKDATA")):
        if taskdata.is_dir():
            groups.append({
                "file": str(taskdata.relative_to(folder)),
                "kind": "ISOXML",
                "checks": [c.to_dict() for c in validate_taskdata(taskdata)],
            })

    for shp in sorted(folder.rglob("*.shp")):
        import geopandas as gpd

        # A boundary has no rate field; a prescription does.
        is_boundary = "boundary" in shp.stem.lower() or "contorno" in shp.stem.lower()
        groups.append({
            "file": str(shp.relative_to(folder)),
            "kind": "Boundary (shapefile)" if is_boundary else "Prescription (shapefile)",
            "checks": [
                c.to_dict() for c in validate_shapefile(
                    shp,
                    rate_field=None if is_boundary else profile.rate_field,
                    expect_polygons=True,
                )
            ],
        })

    all_checks = [c for g in groups for c in g["checks"]]
    failures = sum(1 for c in all_checks if c["status"] == "fail")
    warnings = sum(1 for c in all_checks if c["status"] == "warning")

    if failures:
        verdict = "fail"
        summary = (
            f"{failures} problem(s) the {profile.label} would refuse. "
            "Fix them before taking the stick out."
        )
    elif warnings:
        verdict = "warning"
        summary = (
            f"Nothing blocking, but {warnings} point(s) depend on firmware or display "
            "configuration. Confirm them on screen before starting."
        )
    elif all_checks:
        verdict = "ok"
        summary = (
            f"All {len(all_checks)} checks passed. The known causes of rejection on the "
            f"{profile.label} have been ruled out."
        )
    else:
        verdict = "warning"
        summary = "No checkable file was found in the package."

    return {
        "monitor": profile.key,
        "monitor_label": profile.label,
        "verdict": verdict,
        "summary": summary,
        "totals": {
            "ok": sum(1 for c in all_checks if c["status"] == "ok"),
            "warning": warnings,
            "fail": failures,
        },
        "groups": groups,
        "caveat": (
            "This check compares the file contents against the standard and against the "
            "known causes of rejection. It cannot test your display's firmware version "
            "or proprietary formats — the final confirmation is loading the package on "
            "the monitor before heading to the field."
        ),
    }
