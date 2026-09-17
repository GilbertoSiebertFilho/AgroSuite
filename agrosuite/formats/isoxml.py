"""ISOXML — ISO 11783-10 (TASKDATA).

The only genuinely open format common to Väderstad, Bourgault (X30/X35),
Case IH and New Holland ISOBUS, Topcon/Müller and much of the Trimble
lineup. An ISOXML folder looks like this::

    TASKDATA/
        TASKDATA.XML      registry (customer, farm, field, product, task)
        TLG00001.XML      log header: which fields exist in the binary
        TLG00001.BIN      log records, little-endian binary
        GRD00001.BIN      prescription grid, binary

This module handles both ends:

* **Reading** the registry, the field boundaries and the TLG logs — the TLG
  header says which fields are present, and the binary is unpacked according
  to that declaration.
* **Writing** a prescription as a task with a type-2 grid, which is what
  terminals consume for variable rate.

Sign convention from the standard: latitude = *north*, longitude = *east*,
both in decimal degrees; in the binary they arrive as 32-bit integers in
units of 1e-7 degree.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta

#: The standard's date epoch: days counted from 1980-01-01.
ISO_EPOCH = date(1980, 1, 1)

#: Scale of the integer coordinates in the binary log.
COORD_SCALE = 1e-7


@dataclass(frozen=True)
class DDI:
    """Entry in the ISOBUS data dictionary."""

    code: int
    label: str
    iso_unit: str
    #: Factor taking the standard's integer to AgroSuite's internal unit.
    to_internal: float
    internal_unit: str
    canonical: str | None = None


#: Partial DDI dictionary — covers the quantities used in variable rate,
#: harvest and travel telemetry. DDIs outside this table are kept as
#: ``ddi_<code>`` columns holding the raw value, so no log data is lost
#: silently.
DDI_TABLE: dict[int, DDI] = {
    0x0001: DDI(0x0001, "Setpoint volume per area rate", "mm3/m2", 0.01, "L/ha", sch.TARGET_RATE),
    0x0002: DDI(0x0002, "Actual volume per area rate", "mm3/m2", 0.01, "L/ha", sch.APPLIED_RATE),
    0x0003: DDI(0x0003, "Setpoint mass per time rate", "mg/s", 1e-6, "kg/s", None),
    0x0004: DDI(0x0004, "Actual mass per time rate", "mg/s", 1e-6, "kg/s", sch.FLOW),
    0x0005: DDI(0x0005, "Setpoint mass per area rate", "mg/m2", 0.01, "kg/ha", sch.TARGET_RATE),
    0x0006: DDI(0x0006, "Setpoint mass per area rate", "mg/m2", 0.01, "kg/ha", sch.TARGET_RATE),
    0x0007: DDI(0x0007, "Actual mass per area rate", "mg/m2", 0.01, "kg/ha", sch.APPLIED_RATE),
    0x000A: DDI(0x000A, "Setpoint count per area rate", "1/m2", 10.0, "seeds/ha", sch.TARGET_RATE),
    0x000B: DDI(0x000B, "Actual count per area rate", "1/m2", 10.0, "seeds/ha", sch.APPLIED_RATE),
    0x0043: DDI(0x0043, "Actual ground speed", "mm/s", 0.0036, "km/h", sch.SPEED),
    0x0046: DDI(0x0046, "GNSS speed", "mm/s", 0.0036, "km/h", sch.SPEED),
    0x0048: DDI(0x0048, "Actual work state", "-", 1.0, "-", None),
    0x0049: DDI(0x0049, "Effective working width", "mm", 0.001, "m", sch.SWATH),
    0x0074: DDI(0x0074, "Distance travelled (working)", "mm", 0.001, "m", sch.DISTANCE),
    0x0075: DDI(0x0075, "Distance travelled (total)", "mm", 0.001, "m", None),
    0x0077: DDI(0x0077, "Area worked", "mm2", 1e-10, "ha", None),
    0x0082: DDI(0x0082, "Yield (mass per area)", "mg/m2", 0.01, "kg/ha", sch.VALUE),
    0x0083: DDI(0x0083, "Grain moisture", "ppm", 1e-4, "%", sch.MOISTURE),
    0x0084: DDI(0x0084, "Accumulated harvested mass", "g", 0.001, "kg", None),
    0x0090: DDI(0x0090, "Section width", "mm", 0.001, "m", None),
}

#: Preferred DDIs when writing a prescription, by input type.
RX_DDI = {
    "mass": 0x0006,    # kg/ha  -> mg/m²
    "volume": 0x0001,  # L/ha   -> mm³/m²
    "count": 0x000A,   # sem/ha -> 1/m²
}

RX_DDI_LABELS = {
    "mass": "Mass per area (kg/ha) — dry fertilizer, lime",
    "volume": "Volume per area (L/ha) — spray solution, liquid fertilizer",
    "count": "Count per area (seeds/ha) — seeding",
}


def ddi_scale_to_iso(kind: str) -> float:
    """Factor taking the internal unit to the standard's integer."""
    return {"mass": 100.0, "volume": 100.0, "count": 0.1}[kind]


# ==========================================================================
# Reading
# ==========================================================================

def find_taskdata(root: Path) -> Path | None:
    """Locate TASKDATA.XML inside a folder, case-insensitively."""
    root = Path(root)
    if root.is_file() and root.name.upper() == "TASKDATA.XML":
        return root
    if root.is_dir():
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.upper() == "TASKDATA.XML":
                return candidate
    return None


def _text(element: ET.Element | None, attr: str) -> str | None:
    if element is None:
        return None
    value = element.get(attr)
    return value if value not in (None, "") else None


def parse_taskdata(taskdata_path: Path) -> dict[str, Any]:
    """Read the TASKDATA.XML registry: customers, farms, fields, tasks."""
    tree = ET.parse(taskdata_path)
    root = tree.getroot()

    customers = {c.get("A"): c.get("B") for c in root.iter("CTR")}
    farms = {f.get("A"): f.get("B") for f in root.iter("FRM")}
    products = {p.get("A"): p.get("B") for p in root.iter("PDT")}

    fields: list[dict[str, Any]] = []
    for pfd in root.iter("PFD"):
        polygons = []
        for pln in pfd.iter("PLN"):
            for lsg in pln.iter("LSG"):
                ring = [
                    (float(pnt.get("D")), float(pnt.get("C")))  # (lon, lat)
                    for pnt in lsg.iter("PNT")
                    if pnt.get("C") and pnt.get("D")
                ]
                if len(ring) >= 3:
                    polygons.append(ring)
        fields.append({
            "id": pfd.get("A"),
            "name": pfd.get("C") or pfd.get("B"),
            "area_m2": float(pfd.get("D")) if pfd.get("D") else None,
            "customer": customers.get(pfd.get("E")),
            "farm": farms.get(pfd.get("F")),
            "polygons": polygons,
        })

    tasks: list[dict[str, Any]] = []
    for tsk in root.iter("TSK"):
        logs = [tlg.get("A") for tlg in tsk.iter("TLG") if tlg.get("A")]
        grids = [
            {
                "min_north": float(grd.get("A")),
                "min_east": float(grd.get("B")),
                "cell_north": float(grd.get("C")),
                "cell_east": float(grd.get("D")),
                "max_col": int(grd.get("E")),
                "max_row": int(grd.get("F")),
                "filename": grd.get("G"),
                "grid_type": int(grd.get("I") or 2),
            }
            for grd in tsk.iter("GRD")
            if grd.get("A") and grd.get("G")
        ]
        tasks.append({
            "id": tsk.get("A"),
            "name": tsk.get("B"),
            "customer": customers.get(tsk.get("C")),
            "farm": farms.get(tsk.get("D")),
            "field": tsk.get("E"),
            "logs": logs,
            "grids": grids,
        })

    return {
        "version": f"{root.get('VersionMajor', '?')}.{root.get('VersionMinor', '?')}",
        "software": root.get("ManagementSoftwareManufacturer"),
        "customers": customers,
        "farms": farms,
        "products": products,
        "fields": fields,
        "tasks": tasks,
    }


def _tlg_layout(header_path: Path) -> dict[str, Any]:
    """Interpret the TLGxxxxx.XML header.

    In the standard, an attribute that is **present and empty** means "this
    field is written on every record of the binary"; a missing attribute means
    the field is not logged at all. That distinction defines the byte layout.
    """
    root = ET.parse(header_path).getroot()
    tim = root if root.tag == "TIM" else root.find("TIM")
    if tim is None:
        raise ValueError(f"{header_path.name}: TIM element missing.")

    fields: list[tuple[str, str, int]] = []  # (name, struct format, bytes)
    if "A" in tim.attrib and tim.get("A") == "":
        fields.append(("time_ms", "<I", 4))
        fields.append(("date_days", "<H", 2))

    ptn = tim.find("PTN")
    if ptn is not None:
        optional = [
            ("A", "north", "<i", 4), ("B", "east", "<i", 4), ("C", "up", "<i", 4),
            ("D", "pos_status", "<B", 1), ("E", "pdop", "<H", 2), ("F", "hdop", "<H", 2),
            ("G", "n_sats", "<B", 1), ("H", "gps_utc_time", "<I", 4),
            ("I", "gps_utc_date", "<H", 2),
        ]
        for attr, name, fmt, size in optional:
            if attr in ptn.attrib and ptn.get(attr) == "":
                fields.append((name, fmt, size))

    dlvs = []
    for index, dlv in enumerate(tim.findall("DLV")):
        raw_ddi = dlv.get("A") or "0"
        try:
            code = int(raw_ddi, 16)
        except ValueError:
            code = 0
        dlvs.append({
            "index": index,
            "ddi": code,
            "device_element": dlv.get("C"),
        })

    return {"fields": fields, "dlvs": dlvs}


def read_tlg(header_path: Path) -> pd.DataFrame:
    """Read a TLGxxxxx.XML + .BIN pair and return the records as a table."""
    header_path = Path(header_path)
    bin_path = header_path.with_suffix(".BIN")
    if not bin_path.exists():
        bin_path = header_path.with_suffix(".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Log binary missing for {header_path.name}.")

    layout = _tlg_layout(header_path)
    fields = layout["fields"]
    dlv_meta = {d["index"]: d for d in layout["dlvs"]}
    data = bin_path.read_bytes()

    records: list[dict[str, Any]] = []
    offset = 0
    total = len(data)
    fixed_size = sum(size for _, _, size in fields)

    while offset + fixed_size + 1 <= total:
        record: dict[str, Any] = {}
        for name, fmt, size in fields:
            record[name] = struct.unpack_from(fmt, data, offset)[0]
            offset += size

        count = data[offset]
        offset += 1
        # An absurd counter means the stream has lost sync: better to stop than
        # to produce thousands of junk rows.
        if count > 64 or offset + count * 5 > total:
            break
        for _ in range(count):
            index = data[offset]
            value = struct.unpack_from("<i", data, offset + 1)[0]
            offset += 5
            meta = dlv_meta.get(index)
            ddi_code = meta["ddi"] if meta else index
            record[f"__dlv_{ddi_code}"] = value
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "north" in df.columns and "east" in df.columns:
        df[sch.LAT] = df["north"] * COORD_SCALE
        df[sch.LON] = df["east"] * COORD_SCALE
        df = df.drop(columns=["north", "east"])
    if "up" in df.columns:
        df[sch.ELEVATION] = df["up"] / 1000.0
        df = df.drop(columns=["up"])

    if "date_days" in df.columns and "time_ms" in df.columns:
        base = pd.to_datetime(ISO_EPOCH) + pd.to_timedelta(df["date_days"], unit="D")
        df[sch.TIMESTAMP] = base + pd.to_timedelta(df["time_ms"], unit="ms")
        df = df.drop(columns=["date_days", "time_ms"])

    # Convert each DLV to the internal unit, or keep the raw value.
    for column in [c for c in df.columns if c.startswith("__dlv_")]:
        code = int(column.removeprefix("__dlv_"))
        entry = DDI_TABLE.get(code)
        if entry is None:
            df = df.rename(columns={column: f"ddi_{code:04X}"})
            continue
        converted = df[column] * entry.to_internal
        target = entry.canonical or f"ddi_{code:04X}"
        if target in df.columns:
            target = f"ddi_{code:04X}"
        df[target] = converted
        df = df.drop(columns=[column])

    return df


def read_grid(base_dir: Path, grid_info: dict, rate_kind: str = "mass") -> pd.DataFrame:
    """Read a type-2 prescription grid as a table of cells.

    The binary holds ``max_col * max_row`` unsigned 32-bit integers, scanned
    row by row from the south-west corner. Each cell becomes a row with its
    centre in geographic coordinates and the rate already converted to the
    internal unit.
    """
    base_dir = Path(base_dir)
    name = grid_info["filename"]
    candidates = [
        base_dir / f"{name}.BIN", base_dir / f"{name}.bin", base_dir / name,
    ]
    grid_path = next((c for c in candidates if c.exists()), None)
    if grid_path is None:
        raise FileNotFoundError(f"Grid binary {name} not found.")

    rows, cols = grid_info["max_row"], grid_info["max_col"]
    if grid_info.get("grid_type", 2) == 1:
        raw = np.frombuffer(grid_path.read_bytes(), dtype="<u1")
        scale = 1.0  # treatment zone code, not a rate
    else:
        raw = np.frombuffer(grid_path.read_bytes(), dtype="<u4")
        scale = 1.0 / ddi_scale_to_iso(rate_kind)

    expected = rows * cols
    if raw.size < expected:
        raise ValueError(
            f"Grid {name}: {raw.size} cells in the binary against {expected} declared."
        )
    values = raw[:expected].reshape(rows, cols).astype("float64") * scale

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    lat = grid_info["min_north"] + (row_idx + 0.5) * grid_info["cell_north"]
    lon = grid_info["min_east"] + (col_idx + 0.5) * grid_info["cell_east"]

    return pd.DataFrame({
        sch.LON: lon.ravel(),
        sch.LAT: lat.ravel(),
        sch.TARGET_RATE: values.ravel(),
        sch.VALUE: values.ravel(),
        "grid_row": row_idx.ravel(),
        "grid_col": col_idx.ravel(),
    })


def read_isoxml(path: Path, brand_hint: str | None = None) -> Dataset:
    """Import an ISOXML folder (or TASKDATA.XML) as a dataset."""
    path = Path(path)
    taskdata = find_taskdata(path)
    if taskdata is None:
        raise ValueError("No TASKDATA.XML found in the given folder.")

    catalog = parse_taskdata(taskdata)
    base = taskdata.parent
    notes = [f"ISOXML version {catalog['version']}."]
    if catalog.get("software"):
        notes.append(f"Generated by {catalog['software']}.")

    frames: list[pd.DataFrame] = []
    for task in catalog["tasks"]:
        for log_name in task["logs"]:
            header = next(
                (p for p in base.iterdir()
                 if p.is_file() and p.stem.upper() == log_name.upper()
                 and p.suffix.upper() == ".XML"),
                None,
            )
            if header is None:
                continue
            try:
                frame = read_tlg(header)
            except Exception as exc:  # a corrupt log must not invalidate the rest
                notes.append(f"Log {log_name} skipped: {exc}")
                continue
            if frame.empty:
                continue
            frame["task"] = task["name"] or task["id"]
            frames.append(frame)

    grid_frames: list[pd.DataFrame] = []
    if not frames:
        for task in catalog["tasks"]:
            for grid_info in task["grids"]:
                try:
                    grid_frame = read_grid(base, grid_info)
                except Exception as exc:
                    notes.append(f"Grid {grid_info['filename']} skipped: {exc}")
                    continue
                grid_frame["task"] = task["name"] or task["id"]
                grid_frames.append(grid_frame)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        geometry = None
        operation = "harvest" if sch.VALUE in df.columns else "application"
    elif grid_frames:
        df = pd.concat(grid_frames, ignore_index=True)
        geometry = None
        operation = "prescription"
        notes.append("No TLG log; the task's prescription grids were imported instead.")
    else:
        # No logs: import the field boundaries from the registry instead.
        from shapely.geometry import Polygon

        rows, geoms = [], []
        for fld in catalog["fields"]:
            for ring in fld["polygons"]:
                poly = Polygon(ring)
                if not poly.is_valid or poly.is_empty:
                    continue
                point = poly.representative_point()
                rows.append({
                    "field_name": fld["name"],
                    "area_ha": (fld["area_m2"] or 0) / 10_000.0,
                    "farm": fld["farm"],
                    "customer": fld["customer"],
                    sch.LON: point.x,
                    sch.LAT: point.y,
                })
                geoms.append(poly)
        if not rows:
            raise ValueError("ISOXML with no logs and no usable field boundaries.")
        df = pd.DataFrame(rows)
        geometry = geoms
        operation = "boundary"
        notes.append("No readable TLG log; the registry's boundaries were imported.")

    if sch.VALUE not in df.columns:
        for candidate in (sch.APPLIED_RATE, sch.TARGET_RATE):
            if candidate in df.columns:
                df[sch.VALUE] = df[candidate]
                notes.append(f"Main variable: {sch.LABELS[candidate]}.")
                break

    from . import brands as brands_mod

    # TASKDATA declares what generated it, which identifies the platform better
    # than any column heuristic could.
    detected = brand_hint
    if not detected:
        software = catalog.get("software") or ""
        guessed, confidence = brands_mod.detect_brand(path=software)
        detected = guessed if confidence >= 0.4 else "isoxml"
    brand = brands_mod.get_brand(detected)
    first_field = catalog["fields"][0]["name"] if catalog["fields"] else None
    meta = DatasetMeta(
        name=first_field or taskdata.parent.name,
        source_path=str(taskdata),
        source_format="isoxml",
        brand=brand.key,
        brand_label=brand.label,
        operation=operation,
        field_name=first_field,
        geometry_type="polygon" if geometry else "point",
        notes=notes,
        extra={
            "isoxml_catalog": {
                "version": catalog["version"],
                "software": catalog["software"],
                "fields": [f["name"] for f in catalog["fields"]],
                "tasks": [t["name"] for t in catalog["tasks"]],
                "products": list(catalog["products"].values()),
            },
            # Boundary and AB lines travel with the data: that is what allows
            # importing from one monitor and re-exporting to another without
            # redrawing anything.
            "field_setup": read_field_setup(taskdata),
        },
    )
    return Dataset(df, meta, geometry=geometry)


# ==========================================================================
# Writing a prescription
# ==========================================================================

def write_prescription(
    out_dir: Path,
    grid: np.ndarray,
    min_lon: float,
    min_lat: float,
    cell_lon: float,
    cell_lat: float,
    rate_kind: str = "mass",
    task_name: str = "Prescription",
    field_name: str = "Field",
    product_name: str = "Product",
    customer_name: str = "AgroSuite",
    farm_name: str = "Farm",
    boundary: list[tuple[float, float]] | None = None,
) -> Path:
    """Write a TASKDATA folder holding a type-2 grid prescription.

    Parameters
    ----------
    grid:
        A ``(rows, cols)`` matrix of rates in the internal unit — kg/ha, L/ha
        or seeds/ha depending on ``rate_kind``. Row 0 is the **southernmost**
        and column 0 the westernmost, which is the order the standard expects
        in the binary. Cells with no rate should be ``NaN`` or 0.
    min_lon, min_lat:
        South-west corner of the grid, in decimal degrees.
    cell_lon, cell_lat:
        Cell size in degrees.
    rate_kind:
        ``"mass"``, ``"volume"`` or ``"count"``.

    Returns
    -------
    Path
        The ``TASKDATA`` folder that was created.
    """
    out_dir = Path(out_dir)
    taskdata_dir = out_dir / "TASKDATA"
    taskdata_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = grid.shape
    scale = ddi_scale_to_iso(rate_kind)
    values = np.nan_to_num(np.asarray(grid, dtype="float64"), nan=0.0)
    iso_values = np.rint(np.clip(values * scale, 0, 2**31 - 1)).astype("<u4")

    grid_name = "GRD00001"
    grid_path = taskdata_dir / f"{grid_name}.BIN"
    grid_path.write_bytes(iso_values.tobytes(order="C"))

    ddi = RX_DDI[rate_kind]
    root = ET.Element("ISO11783_TaskData", {
        "VersionMajor": "4",
        "VersionMinor": "0",
        "ManagementSoftwareManufacturer": "AgroSuite",
        "ManagementSoftwareVersion": "1.0",
        "DataTransferOrigin": "1",
    })
    ET.SubElement(root, "CTR", {"A": "CTR1", "B": customer_name})
    ET.SubElement(root, "FRM", {"A": "FRM1", "B": farm_name, "I": "CTR1"})

    pfd = ET.SubElement(root, "PFD", {
        "A": "PFD1", "C": field_name, "E": "CTR1", "F": "FRM1",
    })
    if boundary and len(boundary) >= 3:
        pln = ET.SubElement(pfd, "PLN", {"A": "1"})
        lsg = ET.SubElement(pln, "LSG", {"A": "1"})
        for order, (lon, lat) in enumerate(boundary, start=1):
            ET.SubElement(lsg, "PNT", {
                "A": "2", "C": f"{lat:.9f}", "D": f"{lon:.9f}", "I": str(order),
            })

    ET.SubElement(root, "PDT", {"A": "PDT1", "B": product_name})

    tsk = ET.SubElement(root, "TSK", {
        "A": "TSK1", "B": task_name, "C": "CTR1", "D": "FRM1", "E": "PFD1", "G": "1",
    })
    tzn = ET.SubElement(tsk, "TZN", {"A": "0"})
    ET.SubElement(tzn, "PDV", {"A": f"{ddi:04X}", "B": "0", "C": "PDT1"})
    ET.SubElement(tsk, "GRD", {
        "A": f"{min_lat:.9f}",
        "B": f"{min_lon:.9f}",
        "C": f"{cell_lat:.9f}",
        "D": f"{cell_lon:.9f}",
        "E": str(cols),
        "F": str(rows),
        "G": grid_name,
        "H": str(grid_path.stat().st_size),
        "I": "2",
    })

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        taskdata_dir / "TASKDATA.XML", encoding="UTF-8", xml_declaration=True
    )
    return taskdata_dir


# ==========================================================================
# Field setup: boundary, guidance lines and prescription
# ==========================================================================
# ISO 11783-10 classifies each geometry by a numeric code. The ones that
# matter here:
#
#   PLN@A  polygon type      1 = field boundary, 7 = headland,
#                            6 = obstacle, 11 = enclave
#   LSG@A  line type         1 = polygon exterior, 2 = polygon interior,
#                            5 = guidance pattern, 3 = tramline
#   PNT@A  point type        2 = generic, 6 = guidance reference A,
#                            7 = guidance reference B, 10 = field reference
#   GPN@C  guidance type     1 = AB line, 2 = A plus, 3 = curve,
#                            4 = pivot, 5 = spiral

PLN_BOUNDARY = 1
PLN_OBSTACLE = 6
PLN_HEADLAND = 7
PLN_ENCLAVE = 11

LSG_EXTERIOR = 1
LSG_INTERIOR = 2
LSG_GUIDANCE = 5

PNT_GENERIC = 2
PNT_GUIDANCE_A = 6
PNT_GUIDANCE_B = 7

GPN_AB_LINE = 1
GPN_A_PLUS = 2
GPN_CURVE = 3
GPN_PIVOT = 4

GUIDANCE_TYPE_LABELS = {
    GPN_AB_LINE: "AB line (two points)",
    GPN_A_PLUS: "A plus (point and heading)",
    GPN_CURVE: "Recorded curve",
    GPN_PIVOT: "Centre pivot",
}


def _ring_element(parent: ET.Element, ring, line_type: int, point_type: int) -> None:
    """Write an LSG with its points, in decimal degrees."""
    lsg = ET.SubElement(parent, "LSG", {"A": str(line_type)})
    for order, (lon, lat) in enumerate(ring, start=1):
        ET.SubElement(lsg, "PNT", {
            "A": str(point_type),
            "C": f"{lat:.9f}",
            "D": f"{lon:.9f}",
            "I": str(order),
        })


def write_field_setup(
    out_dir: Path,
    field_name: str = "Field",
    boundary: list[tuple[float, float]] | None = None,
    inner_rings: list[list[tuple[float, float]]] | None = None,
    headland: list[tuple[float, float]] | None = None,
    guidance_lines: list[dict] | None = None,
    customer_name: str = "AgroSuite",
    farm_name: str = "Farm",
    area_m2: float | None = None,
    prescription: dict | None = None,
    task_name: str | None = None,
    product_name: str = "Product",
) -> Path:
    """Write a TASKDATA with boundary, guidance lines and, if given, an Rx.

    This is the setup file you carry on the stick: the ISOBUS terminal loads
    the field, the AB lines and the prescription in one go, with nothing to
    redraw in the cab.

    Parameters
    ----------
    boundary:
        The field's outer ring, in ``(lon, lat)``. It is closed automatically.
    inner_rings:
        Enclaves — inner areas that are not part of the field.
    headland:
        Headland polygon, when there is one distinct from the boundary.
    guidance_lines:
        List of lines, each ``{"name": str, "type": int, "a": (lon, lat),
        "b": (lon, lat)}`` for an AB line, or ``{"name", "type": 3, "points":
        [(lon, lat), ...]}`` for a recorded curve.
    prescription:
        ``{"grid": ndarray, "min_lon", "min_lat", "cell_lon", "cell_lat",
        "rate_kind"}`` — the same structure rasterization returns.

    Returns
    -------
    Path
        The ``TASKDATA`` folder that was created.
    """
    out_dir = Path(out_dir)
    taskdata_dir = out_dir / "TASKDATA"
    taskdata_dir.mkdir(parents=True, exist_ok=True)

    root = ET.Element("ISO11783_TaskData", {
        "VersionMajor": "4",
        "VersionMinor": "0",
        "ManagementSoftwareManufacturer": "AgroSuite",
        "ManagementSoftwareVersion": "1.0",
        "DataTransferOrigin": "1",
    })
    ET.SubElement(root, "CTR", {"A": "CTR1", "B": customer_name})
    ET.SubElement(root, "FRM", {"A": "FRM1", "B": farm_name, "I": "CTR1"})

    pfd_attrs = {"A": "PFD1", "C": field_name, "E": "CTR1", "F": "FRM1"}
    if area_m2:
        pfd_attrs["D"] = str(int(round(area_m2)))
    pfd = ET.SubElement(root, "PFD", pfd_attrs)

    if boundary and len(boundary) >= 3:
        closed = list(boundary)
        if closed[0] != closed[-1]:
            closed.append(closed[0])
        pln = ET.SubElement(pfd, "PLN", {"A": str(PLN_BOUNDARY), "B": f"{field_name} boundary"})
        _ring_element(pln, closed, LSG_EXTERIOR, PNT_GENERIC)
        # Enclaves go in as interior rings of the same polygon.
        for ring in (inner_rings or []):
            if len(ring) >= 3:
                inner = list(ring)
                if inner[0] != inner[-1]:
                    inner.append(inner[0])
                _ring_element(pln, inner, LSG_INTERIOR, PNT_GENERIC)

    if headland and len(headland) >= 3:
        closed = list(headland)
        if closed[0] != closed[-1]:
            closed.append(closed[0])
        pln = ET.SubElement(pfd, "PLN", {"A": str(PLN_HEADLAND), "B": "Headland"})
        _ring_element(pln, closed, LSG_EXTERIOR, PNT_GENERIC)

    if guidance_lines:
        ggp = ET.SubElement(pfd, "GGP", {"A": "GGP1", "B": f"{field_name} guidance"})
        for index, line in enumerate(guidance_lines, start=1):
            pattern_type = int(line.get("type", GPN_AB_LINE))
            attrs = {
                "A": f"GPN{index}",
                "B": line.get("name") or f"Line {index}",
                "C": str(pattern_type),
            }
            if line.get("heading") is not None:
                attrs["G"] = f"{float(line['heading']):.2f}"
            gpn = ET.SubElement(ggp, "GPN", attrs)

            if pattern_type in (GPN_AB_LINE, GPN_A_PLUS) and line.get("a"):
                lsg = ET.SubElement(gpn, "LSG", {"A": str(LSG_GUIDANCE)})
                a_lon, a_lat = line["a"]
                ET.SubElement(lsg, "PNT", {
                    "A": str(PNT_GUIDANCE_A), "C": f"{a_lat:.9f}", "D": f"{a_lon:.9f}", "I": "1",
                })
                if line.get("b"):
                    b_lon, b_lat = line["b"]
                    ET.SubElement(lsg, "PNT", {
                        "A": str(PNT_GUIDANCE_B), "C": f"{b_lat:.9f}", "D": f"{b_lon:.9f}", "I": "2",
                    })
            elif line.get("points"):
                _ring_element(gpn, line["points"], LSG_GUIDANCE, PNT_GENERIC)

    if prescription is not None:
        grid = np.asarray(prescription["grid"], dtype="float64")
        rows, cols = grid.shape
        rate_kind = prescription.get("rate_kind", "mass")
        scale = ddi_scale_to_iso(rate_kind)
        iso_values = np.rint(
            np.clip(np.nan_to_num(grid, nan=0.0) * scale, 0, 2**31 - 1)
        ).astype("<u4")

        grid_name = "GRD00001"
        grid_path = taskdata_dir / f"{grid_name}.BIN"
        grid_path.write_bytes(iso_values.tobytes(order="C"))

        ET.SubElement(root, "PDT", {"A": "PDT1", "B": product_name})
        tsk = ET.SubElement(root, "TSK", {
            "A": "TSK1", "B": task_name or f"Rx {field_name}",
            "C": "CTR1", "D": "FRM1", "E": "PFD1", "G": "1",
        })
        tzn = ET.SubElement(tsk, "TZN", {"A": "0"})
        ET.SubElement(tzn, "PDV", {
            "A": f"{RX_DDI[rate_kind]:04X}", "B": "0", "C": "PDT1",
        })
        ET.SubElement(tsk, "GRD", {
            "A": f"{prescription['min_lat']:.9f}",
            "B": f"{prescription['min_lon']:.9f}",
            "C": f"{prescription['cell_lat']:.9f}",
            "D": f"{prescription['cell_lon']:.9f}",
            "E": str(cols), "F": str(rows),
            "G": grid_name, "H": str(grid_path.stat().st_size), "I": "2",
        })
    else:
        # An empty task tied to the field makes the terminal list it even when
        # the stick carries only a boundary and AB lines.
        ET.SubElement(root, "TSK", {
            "A": "TSK1", "B": task_name or f"Setup {field_name}",
            "C": "CTR1", "D": "FRM1", "E": "PFD1", "G": "1",
        })

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        taskdata_dir / "TASKDATA.XML", encoding="UTF-8", xml_declaration=True
    )
    return taskdata_dir


def read_field_setup(taskdata_path: Path) -> dict:
    """Read boundary, headland and guidance lines from a TASKDATA."""
    root = ET.parse(Path(taskdata_path)).getroot()
    fields: list[dict] = []

    for pfd in root.iter("PFD"):
        boundaries, headlands, obstacles = [], [], []
        for pln in pfd.findall("PLN"):
            poly_type = int(pln.get("A") or PLN_BOUNDARY)
            rings = []
            for lsg in pln.findall("LSG"):
                ring = [
                    (float(p.get("D")), float(p.get("C")))
                    for p in lsg.findall("PNT")
                    if p.get("C") and p.get("D")
                ]
                if len(ring) >= 3:
                    rings.append({"type": int(lsg.get("A") or LSG_EXTERIOR), "points": ring})
            if not rings:
                continue
            target = (headlands if poly_type == PLN_HEADLAND
                      else obstacles if poly_type in (PLN_OBSTACLE, PLN_ENCLAVE)
                      else boundaries)
            target.append(rings)

        lines: list[dict] = []
        for ggp in pfd.findall("GGP"):
            for gpn in ggp.findall("GPN"):
                pattern_type = int(gpn.get("C") or GPN_AB_LINE)
                points = [
                    {
                        "point_type": int(p.get("A") or PNT_GENERIC),
                        "lon": float(p.get("D")),
                        "lat": float(p.get("C")),
                    }
                    for lsg in gpn.findall("LSG")
                    for p in lsg.findall("PNT")
                    if p.get("C") and p.get("D")
                ]
                a = next((p for p in points if p["point_type"] == PNT_GUIDANCE_A), None)
                b = next((p for p in points if p["point_type"] == PNT_GUIDANCE_B), None)
                lines.append({
                    "name": gpn.get("B") or gpn.get("A"),
                    "type": pattern_type,
                    "type_label": GUIDANCE_TYPE_LABELS.get(pattern_type, "Other"),
                    "heading": float(gpn.get("G")) if gpn.get("G") else None,
                    "a": (a["lon"], a["lat"]) if a else None,
                    "b": (b["lon"], b["lat"]) if b else None,
                    "points": [(p["lon"], p["lat"]) for p in points],
                })

        fields.append({
            "name": pfd.get("C") or pfd.get("B") or pfd.get("A"),
            "area_m2": float(pfd.get("D")) if pfd.get("D") else None,
            "boundaries": boundaries,
            "headlands": headlands,
            "obstacles": obstacles,
            "guidance_lines": lines,
        })

    return {"fields": fields}
