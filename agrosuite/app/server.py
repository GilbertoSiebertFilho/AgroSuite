"""AgroSuite's local server.

The app runs as a server on the machine itself and the interface opens in the
browser. That is deliberate: agricultural data is geospatial, and a real map —
with zoom, satellite imagery and thousands of coloured points — works far
better in a browser than in a widget window.

Nothing leaves the machine. The server listens only on ``127.0.0.1`` and the
files live in a temporary session folder.
"""

from __future__ import annotations

import shutil
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import autosave as autosave_mod
from .. import demo as demo_mod
from ..clean import pipeline as clean_pipeline
from ..clean import steps as clean_steps
from ..core import schema as sch
from ..core import units as units_mod
from ..core.dataset import OPERATION_LABELS, apply_source_units
from ..difm import analysis as difm_analysis
from ..difm import design as difm_design
from ..difm import response as difm_response
from ..formats import augmenta as augmenta_mod
from ..formats import brands as brands_mod
from ..formats import isoxml as isoxml_mod
from ..core import guidance as guidance_mod
from ..core import preflight as preflight_mod
from ..core import workflow as workflow_mod
from ..difm import join as join_mod
from ..formats import qgis as qgis_mod
from ..formats import usb as usb_mod
from ..formats import johndeere as jd_mod
from ..formats import packages as packages_mod
from ..formats import validate as validate_mod
from ..formats import registry, writers
from ..terrain import analysis as terrain_analysis
from ..terrain import yieldrelief
from . import session as session_mod

STATIC_DIR = Path(__file__).parent / "static"

state = session_mod.Session()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """What the app does before it answers anything, and on the way out.

    Before: it picks the last project back up, so that starting the app is
    the same gesture as coming back to the work. After: it writes the
    session one last time, because the final couple of seconds are the ones
    the auto-saver's idle delay has not reached yet.

    Both belong to the process that serves — the launcher says so through
    the environment. An app object imported by a test or a script starts no
    thread, reopens nothing and writes nothing.
    """
    if autosave_mod.enabled_for_process():
        from .routes import persist as persist_routes

        try:
            persist_routes.resume_latest()
        except Exception as exc:  # pragma: no cover - the app has to start
            state.resumed = {"ok": False, "name": None, "reason": (
                f"The last project could not be reopened: {exc}. The app has started "
                "empty; nothing on disk was changed."
            )}
    autosave_mod.start(lambda: state)
    try:
        yield
    finally:
        autosave_mod.stop()


app = FastAPI(title="AgroSuite", version="1.0.0", lifespan=lifespan)


# ==========================================================================
# Request models
# ==========================================================================

class PathRequest(BaseModel):
    path: str
    brand: str | None = None
    #: Units the source file is in, keyed by canonical column.
    source_units: dict[str, str] = Field(default_factory=dict)
    crop: str | None = None


class DemoRequest(BaseModel):
    kind: str = "harvest"


class UnitsRequest(BaseModel):
    """Re-declaring the units of an already-loaded dataset."""

    source_units: dict[str, str] = Field(default_factory=dict)
    crop: str | None = None


class DisplayUnitsRequest(BaseModel):
    """The unit set the reader is working in, as the pickers hold it.

    It is the shape of ``UNIT_PRESETS['canada']`` plus whatever else the
    interface keeps beside it (the preset's key, its label); only the
    quantities are read. Nothing stored changes — this decides which units
    the app's own sentences are written in.
    """

    units: dict[str, Any] = Field(default_factory=dict)


class CleanRequest(BaseModel):
    preset: str | None = None
    value_column: str = sch.VALUE
    corrections: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, Any] = Field(default_factory=dict)


class DifmRequest(BaseModel):
    rate_column: str = sch.APPLIED_RATE
    value_column: str = sch.VALUE
    crop_price: float = 1.0
    input_cost: float = 0.0
    cell_m: float = 20.0
    edge_margin_m: float = 6.0
    zone_column: str | None = None
    models: list[str] | None = None
    rate_max: float | None = None


class DesignRequest(BaseModel):
    boundary: list[list[float]] | None = None
    boundary_dataset_id: str | None = None
    rates: list[float]
    implement_width_m: float = 12.0
    passes_per_strip: int = 2
    blocks: int = 4
    angle_deg: float | None = None
    buffer_m: float = 0.0
    seed: int = 0


class RoleRequest(BaseModel):
    """Assigning a dataset its role inside the project."""

    dataset_id: str
    role: str


class PricesRequest(BaseModel):
    """Prices, in the internal units: currency per kg of crop and of input."""

    crop_price: float = 0.0
    input_cost: float = 0.0
    currency: str = "CAD"
    crop: str | None = None


class ProjectRequest(BaseModel):
    name: str | None = None
    goal: str | None = None


class JoinRequest(BaseModel):
    """Joining the project's layers onto a shared grid."""

    cell_m: float | None = None
    min_points_per_cell: int = 1
    min_purity: float = 0.8
    carry: list[str] = Field(default_factory=list)


class QgisExportRequest(BaseModel):
    """Writing the session's layers out for QGIS."""

    dataset_ids: list[str] = Field(default_factory=list)
    name: str = "agrosuite"
    metric: bool = False


class QgisImportRequest(BaseModel):
    """Pulling layers out of a QGIS project."""

    path: str
    layers: list[str] = Field(default_factory=list)


class UsbRequest(BaseModel):
    folder: str
    drive: str
    replace: list[str] = Field(default_factory=list)


class GuidanceRequest(BaseModel):
    """Building an AB line, from a direction or from two points."""

    boundary: list[list[float]] | None = None
    boundary_dataset_id: str | None = None
    angle_deg: float | None = None
    point_a: list[float] | None = None
    point_b: list[float] | None = None
    name: str = "AB"


class PackageRequest(BaseModel):
    """A complete package to take to the monitor."""

    monitor: str = "generic"
    features: dict[str, Any] | None = None
    boundary: list[list[float]] | None = None
    boundary_dataset_id: str | None = None
    guidance_lines: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    rate_property: str = "rate"
    rate_kind: str = "mass"
    rate_unit: str = "kg/ha"
    crop: str | None = None
    cell_m: float = 10.0
    field_name: str = "Field"
    task_name: str = "Prescription"
    product_name: str = "Product"
    customer_name: str = "AgroSuite"
    farm_name: str = "Farm"


class ExportRequest(BaseModel):
    dataset_id: str | None = None
    features: dict[str, Any] | None = None
    formats: list[str] = Field(default_factory=lambda: ["shapefile"])
    brand: str = "generic"
    rate_property: str = "rate"
    rate_kind: str = "mass"
    #: Unit the rate will be **written** in, in the output file.
    rate_unit: str = "kg/ha"
    crop: str | None = None
    cell_m: float = 10.0
    task_name: str = "Prescription"
    field_name: str = "Field"
    product_name: str = "Product"
    columns: list[str] | None = None


# ==========================================================================
# Helpers
# ==========================================================================

def _fail(message: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _refuse_elevation(entry, step: str) -> None:
    """Steps built for machine tracks stop at an elevation layer.

    The cleaning filters judge speed, swath overlap and pass ends, and the
    economic fit wants a rate against a yield. The cells of a DEM, or the zones
    the terrain analyser derives from one, carry none of that; left alone,
    the default cleaning preset would run over the raster's scan order and
    register a 'clean' copy of a relief with a few cells taken out as
    outliers. Refusing names the step that does apply.
    """
    if entry.dataset.meta.operation == "elevation":
        raise _fail(
            f"'{entry.label}' is an elevation layer, and {step} is for machine tracks "
            "(yield, as-applied, planting). Analyse its relief instead: the Terrain "
            "step, or POST /api/terrain/analyze."
        )


@app.exception_handler(Exception)
async def unhandled(request, exc: Exception):  # pragma: no cover - safety net
    """Turn unexpected failures into a readable message rather than a silent 500."""
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return JSONResponse(
        {
            "detail": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=6),
        },
        status_code=500,
    )


def _register(
    dataset,
    label: str,
    origin: str,
    parent_id: str | None = None,
    source_units: dict[str, str] | None = None,
    crop: str | None = None,
) -> dict[str, Any]:
    # Units are applied before the derived fields: speed reconstructed from the
    # track already comes out in km/h, and converting it afterwards would break it.
    if source_units:
        apply_source_units(dataset, source_units, crop)
    dataset.ensure_derived()
    entry = state.add(dataset, label=label, origin=origin, parent_id=parent_id)

    # The preliminary pass runs on import, not on request: the user should not
    # have to ask whether the file they just opened is usable.
    try:
        report = preflight_mod.run(dataset, state.display_units)
    except Exception as exc:  # a failed check must not block the import
        report = {
            "verdict": "warning",
            "summary": f"The preliminary check could not run: {exc}",
            "findings": [], "info": {}, "suggested_role": "other",
            "role_label": "Other", "next_step": {"step": "review", "label": "Review the file",
                                                 "why": "The automatic check failed."},
        }
    entry.reports["preflight"] = report

    # A derived dataset — converted units, a cleaned copy, a join — takes over
    # its parent's role in the project. Leaving the role on the original would
    # put two versions of the same layer into the analysis, and the stale one
    # is exactly the version the conversion was meant to replace.
    inherited = state.project["roles"].pop(parent_id, None) if parent_id else None
    if inherited:
        try:
            state.get(parent_id).role = None
        except KeyError:
            pass
        if parent_id in state.project["reviewed"]:
            state.project["reviewed"].discard(parent_id)
            state.project["reviewed"].add(entry.id)
    entry.role = inherited or report.get("suggested_role")
    state.project["roles"][entry.id] = entry.role

    # Every import comes through here — a file, a folder, a demo, a QGIS
    # layer, a cleaned copy, a join, a set of zones — which makes this the
    # one place that has to say the session changed for all of them.
    state.touch()

    summary = entry.summary(state.display_units)
    summary["preflight"] = report
    return summary


# ==========================================================================
# Catalogue
# ==========================================================================

@app.get("/api/units")
def units_catalog() -> dict[str, Any]:
    """Catalogue of units and crops for the interface pickers."""
    catalog = units_mod.unit_catalog()
    catalog["display"] = dict(state.display_units)
    return catalog


@app.put("/api/units/display")
def set_display_units(request: DisplayUnitsRequest) -> dict[str, Any]:
    """Tell the server which units the reader is working in.

    Everything the app *says* — the first look, the cleaning report, the
    relief, the notes on a joined layer — is prose with its numbers written
    into it, and prose cannot be converted after the fact without deciding
    again what it says. So the interface names the unit set here whenever
    the preset or one picker changes, and from then on the sentences are
    written in it. No stored number moves: the answer is the resolved set,
    and the interface re-reads the reports it is showing.

    An unknown unit is refused by name rather than quietly ignored, because
    a picker the server did not understand would leave the sentences in one
    unit and the tables in another — the very thing this fixes.
    """
    try:
        units_mod.Phrase(request.units)
    except ValueError as exc:
        raise _fail(str(exc), 400)
    state.display_units = dict(request.units)
    return {"units": dict(state.display_units)}


@app.post("/api/datasets/{dataset_id}/units")
def redeclare_units(dataset_id: str, request: UnitsRequest) -> dict[str, Any]:
    """Re-apply source units to an already-loaded dataset.

    This covers the case where the unit only becomes obvious after seeing the
    numbers: an average yield of 55 on a canola map is bu/ac, not kg/ha. The
    conversion produces a new dataset, leaving the original untouched.
    """
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)

    converted = entry.dataset.copy()
    applied = apply_source_units(converted, request.source_units, request.crop)
    if not applied:
        raise _fail(
            "No conversion applies: the chosen units are already the internal ones, "
            "or those columns do not exist in this dataset."
        )
    summary = _register(converted, f"{entry.label} · converted", "units", entry.id)
    # Declaring a file's units *is* reviewing it: the user looked at what came
    # in and stated what it means. Making them tick a second box would be
    # asking the same question twice.
    state.project["reviewed"].add(summary["id"])
    state.touch()
    return {"dataset": summary, "conversions": applied}


@app.get("/api/workflow")
def workflow_catalog() -> dict[str, Any]:
    """The stages and goals, for the interface to draw the flow."""
    return workflow_mod.describe()


def _project_state() -> dict[str, Any]:
    """Where the project stands, what it is missing and what to do next."""
    project = state.project
    entries = {e["id"]: e for e in state.list()}

    layers = []
    roles: set[str] = set()
    for dataset_id, role in project["roles"].items():
        entry = entries.get(dataset_id)
        if entry is None:
            continue
        layers.append({
            "dataset_id": dataset_id,
            "label": entry["label"],
            "role": role,
            "role_label": preflight_mod.ROLES.get(role, role),
            "rows": entry["rows"],
            "origin": entry["origin"],
            "reviewed": dataset_id in project["reviewed"],
        })
        if role:
            roles.add(role)

    prices = project["prices"]
    has_prices = bool(prices.get("crop_price"))

    # A zone column is anything categorical the yield layer carries beyond the
    # canonical ones — that is what lets the analysis split the field.
    has_zone = False
    for dataset_id, role in project["roles"].items():
        if role != "yield":
            continue
        try:
            dataset = state.get(dataset_id).dataset
        except KeyError:
            continue
        for column in dataset.df.columns:
            if column in sch.LABELS or column in ("x", "y"):
                continue
            if dataset.df[column].nunique(dropna=True) <= 12:
                has_zone = True
                break

    # Read off the snapshot taken above rather than listing the session again
    # per role: it is the same list, and rebuilding it inside the loop costs a
    # summary of every dataset per role. A dataset removed between the
    # snapshot and here is simply one this answer does not describe.
    has_geometry = False
    has_rate_column = False
    for dataset_id in entries:
        try:
            dataset = state.get(dataset_id).dataset
        except KeyError:
            continue
        if project["roles"].get(dataset_id) and dataset.geometry is not None:
            has_geometry = True
        # One file can carry both sides of the response: the layers joined onto
        # a shared grid, or a trial exported with its rate column beside the
        # yield. The project then holds the rate without a second file marked
        # as-applied, and asking for one would ask for what it already has.
        columns = set(dataset.df.columns)
        if sch.VALUE in columns and {sch.APPLIED_RATE, sch.TARGET_RATE} & columns:
            has_rate_column = True

    goal = project["goal"]
    evaluation = workflow_mod.evaluate(
        goal, roles, has_prices=has_prices,
        has_geometry=has_geometry, has_zone=has_zone,
        has_layers=bool(entries), has_rate_column=has_rate_column,
    )

    cleaned = any(e["origin"] == "clean" for e in entries.values())
    analysed = any(
        # 'difm' is the stored report key; the analysis is called the economic
        # analysis on screen. See the note in agrosuite/core/workflow.py.
        "difm" in state.get(e["id"]).reports or "augmenta" in state.get(e["id"]).reports
        for e in entries.values()
    )
    designed = bool(project.get("design"))
    reviewed = bool(layers) and all(layer["reviewed"] for layer in layers)
    stage = workflow_mod.stage_of(
        len(layers), reviewed, cleaned, analysed, project["exported"],
        goal=goal, ready=evaluation["ready"], designed=designed,
    )
    track = workflow_mod.track_of(goal)

    return {
        "name": project["name"],
        "goal": goal,
        "stage": stage,
        "stage_label": workflow_mod.STAGE_LABELS[stage],
        "layers": layers,
        "roles_present": sorted(roles),
        "prices": prices,
        "evaluation": evaluation,
        "next_action": workflow_mod.next_action(stage, goal, evaluation),
        "role_options": preflight_mod.ROLES,
        # The track this goal belongs to, and how far its stages have got.
        # Every stage comes back reachable: the strip guides, it does not gate.
        "track": track,
        "track_label": workflow_mod.TRACKS[track]["label"],
        "stages": workflow_mod.stage_states(
            goal, stage, reviewed=reviewed, cleaned=cleaned, analysed=analysed,
            designed=designed, exported=project["exported"],
        ),
        "has_design": designed,
    }


@app.get("/api/project")
def get_project() -> dict[str, Any]:
    return _project_state()


@app.post("/api/project")
def update_project(request: ProjectRequest) -> dict[str, Any]:
    if request.name is not None:
        state.project["name"] = request.name
    if request.goal is not None:
        if request.goal not in workflow_mod.GOALS:
            raise _fail(f"Unknown goal: '{request.goal}'.")
        state.project["goal"] = request.goal
    # The project's name is also the auto-saved file's name: renaming it
    # here renames the file, rather than leaving the old one behind.
    state.touch()
    return _project_state()


@app.post("/api/project/role")
def set_role(request: RoleRequest) -> dict[str, Any]:
    """Set what a dataset is for, and mark it reviewed."""
    try:
        entry = state.get(request.dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    if request.role not in preflight_mod.ROLES:
        raise _fail(
            f"Unknown role: '{request.role}'. "
            f"Valid roles: {', '.join(preflight_mod.ROLES)}."
        )
    entry.role = request.role
    state.project["roles"][request.dataset_id] = request.role
    state.project["reviewed"].add(request.dataset_id)
    state.touch()
    return _project_state()


@app.post("/api/project/prices")
def set_prices(request: PricesRequest) -> dict[str, Any]:
    state.project["prices"] = {
        "crop_price": request.crop_price,
        "input_cost": request.input_cost,
        "currency": request.currency,
        "crop": request.crop,
    }
    state.touch()
    return _project_state()


@app.post("/api/project/join")
def join_project(request: JoinRequest) -> dict[str, Any]:
    """Join the project's layers onto a shared grid, ready for analysis."""
    layers: dict[str, Any] = {}
    for dataset_id, role in state.project["roles"].items():
        if role not in join_mod.ROLE_COLUMNS:
            continue
        try:
            layers[role] = state.get(dataset_id).dataset
        except KeyError:
            continue

    if "yield" not in layers:
        raise _fail(
            "No layer is marked as Yield. Set the harvest file's role before joining."
        )

    cell_m = request.cell_m or join_mod.suggest_cell_size(layers)
    try:
        joined, report = join_mod.join_layers(
            layers, cell_m=cell_m,
            min_points_per_cell=request.min_points_per_cell,
            min_purity=request.min_purity,
            carry=request.carry or None,
            units=state.display_units,
        )
    except ValueError as exc:
        raise _fail(str(exc))

    summary = _register(joined, f"{state.project['name']} · joined", "join")
    # The joined table is the analysis input, not one of the layers.
    state.project["roles"].pop(summary["id"], None)
    state.get(summary["id"]).role = None
    state.get(summary["id"]).reports["join"] = report
    state.touch()
    return {"dataset": summary, "report": report, "cell_m": cell_m}


@app.post("/api/qgis/export")
def qgis_export(request: QgisExportRequest) -> dict[str, Any]:
    """Write the chosen datasets as one GeoPackage plus a QGIS project."""
    ids = request.dataset_ids or [e["id"] for e in state.list()]
    layers: dict[str, Any] = {}
    for dataset_id in ids:
        try:
            entry = state.get(dataset_id)
        except KeyError:
            continue
        layers[entry.label] = entry.dataset
    if not layers:
        raise _fail("No dataset to export.")

    index = len(list(state.exports.iterdir())) + 1
    out_dir = state.exports / f"qgis_{index}"
    try:
        result = qgis_mod.export_for_qgis(
            layers, out_dir, name=request.name, metric=request.metric
        )
    except Exception as exc:
        raise _fail(f"Could not write for QGIS: {exc}")

    token = state.register_file(Path(result["geopackage"]["path"]))
    result["download_url"] = f"/api/download/{token}"
    return result


@app.get("/api/qgis/project")
def qgis_project(path: str) -> dict[str, Any]:
    """List the layers of a QGIS project, without loading their data."""
    try:
        return qgis_mod.read_project(Path(path).expanduser())
    except Exception as exc:
        raise _fail(str(exc))


@app.post("/api/qgis/import")
def qgis_import(request: QgisImportRequest) -> dict[str, Any]:
    """Import the chosen layers of a QGIS project."""
    try:
        project = qgis_mod.read_project(Path(request.path).expanduser())
    except Exception as exc:
        raise _fail(str(exc))

    wanted = set(request.layers) if request.layers else None
    imported, skipped = [], []
    for layer in project["layers"]:
        if wanted is not None and layer["name"] not in wanted:
            continue
        if not layer["importable"]:
            skipped.append({
                "name": layer["name"],
                "reason": "not on this machine" if not layer["exists"]
                          else f"provider '{layer['provider']}' is not a file layer",
            })
            continue
        try:
            dataset = qgis_mod.read_project_layer(layer)
        except Exception as exc:
            skipped.append({"name": layer["name"], "reason": str(exc)})
            continue
        imported.append(_register(dataset, layer["name"], "qgis"))

    if not imported:
        raise _fail(
            "No layer could be imported. "
            + (f"Reasons: {'; '.join(s['reason'] for s in skipped)}." if skipped else "")
        )
    return {"imported": imported, "skipped": skipped, "project": project["title"]}


@app.get("/api/usb")
def usb_drives() -> dict[str, Any]:
    """Drives the app could copy a package to."""
    return {"drives": usb_mod.list_drives()}


@app.post("/api/usb/plan")
def usb_plan(request: UsbRequest) -> dict[str, Any]:
    """Say what copying would replace, before anything is written."""
    try:
        return usb_mod.plan_write(Path(request.folder), request.drive)
    except ValueError as exc:
        raise _fail(str(exc))


@app.post("/api/usb/write")
def usb_write(request: UsbRequest) -> dict[str, Any]:
    """Copy the package to the drive root."""
    try:
        result = usb_mod.write_to_drive(
            Path(request.folder), request.drive, replace=request.replace
        )
    except ValueError as exc:
        raise _fail(str(exc))
    state.project["exported"] = True
    state.touch()
    return result


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    """Everything the interface needs to build its forms."""
    return {
        "brands": brands_mod.brand_catalog(),
        "operations": OPERATION_LABELS,
        "presets": {
            key: {
                "label": value["label"],
                "description": value["description"],
                "corrections": value["corrections"],
                "steps": value["steps"],
            }
            for key, value in clean_pipeline.PRESETS.items()
        },
        "steps": [
            {
                "key": cls.key,
                "label": cls.label,
                "description": cls.description,
                "defaults": cls.defaults,
            }
            for cls in clean_steps.STEP_CLASSES
        ],
        "response_models": difm_response.MODEL_LABELS,
        "rate_kinds": isoxml_mod.RX_DDI_LABELS,
        "import_extensions": sorted(registry.ALL_IMPORT_EXT),
        "columns": sch.LABELS,
        "monitors": packages_mod.profile_catalog(),
        "roles": preflight_mod.ROLES,
        "workflow": workflow_mod.describe(),
        "artifact_labels": packages_mod.ARTIFACT_LABELS,
        "units": units_mod.unit_catalog(),
        "unit_columns": {k: v for k, v in __import__(
            "agrosuite.core.dataset", fromlist=["SOURCE_UNIT_COLUMNS"]
        ).SOURCE_UNIT_COLUMNS.items()},
    }


# ==========================================================================
# Import
# ==========================================================================

@app.post("/api/import/upload")
async def import_upload(
    file: UploadFile = File(...),
    brand: str | None = None,
    rate_unit: str | None = None,
    speed_unit: str | None = None,
    length_unit: str | None = None,
    crop: str | None = None,
) -> dict[str, Any]:
    """Import a file uploaded from the browser."""
    if not file.filename:
        raise _fail("File has no name.")

    target = state.uploads / file.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    if target.suffix.lower() == ".shp":
        raise _fail(
            "A shapefile is several files (.shp, .shx, .dbf, .prj), and uploading the "
            ".shp on its own will not open. Zip the folder and upload the zip, or "
            "import by local path instead."
        )

    try:
        dataset = registry.read_any(target, brand)
    except Exception as exc:
        raise _fail(f"Could not read '{file.filename}': {exc}")

    declared = {
        sch.VALUE: rate_unit, sch.TARGET_RATE: rate_unit, sch.APPLIED_RATE: rate_unit,
        sch.SPEED: speed_unit, sch.SWATH: length_unit,
    }
    declared = {k: v for k, v in declared.items() if v}
    return _register(dataset, Path(file.filename).stem, "upload", source_units=declared, crop=crop)


@app.post("/api/import/path")
def import_path(request: PathRequest) -> dict[str, Any]:
    """Import from a path on this machine's disk.

    This is the preferred route for shapefiles and ISOXML folders, which only
    make sense with all of their companion files present.
    """
    path = Path(request.path.strip().strip('"'))
    if not path.exists():
        raise _fail(f"Path not found: {path}")
    try:
        dataset = registry.read_any(path, request.brand)
    except Exception as exc:
        raise _fail(f"Could not read '{path.name}': {exc}")
    return _register(
        dataset, path.stem or path.name, "path",
        source_units=request.source_units, crop=request.crop,
    )


@app.post("/api/import/demo")
def import_demo(request: DemoRequest) -> dict[str, Any]:
    """Load a synthetic dataset for trying the app out."""
    if request.kind == "trial":
        dataset = demo_mod.synthetic_trial()
        label = "Strip trial (demo)"
    else:
        dataset = demo_mod.synthetic_harvest()
        label = "Harvest with defects (demo)"
    return _register(dataset, label, "demo")


@app.get("/api/browse")
def browse(path: str = "") -> dict[str, Any]:
    """List a folder on disk, so files can be picked without typing a path."""
    target = Path(path).expanduser() if path else Path.home()
    if not target.exists():
        raise _fail(f"Folder not found: {target}")
    if target.is_file():
        target = target.parent

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            is_dir = child.is_dir()
            importable = is_dir or child.suffix.lower() in registry.ALL_IMPORT_EXT
            if not importable:
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_dir": is_dir,
                "size": child.stat().st_size if not is_dir else None,
            })
    except PermissionError:
        raise _fail(f"No permission to list: {target}")

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "entries": entries[:500],
    }


@app.get("/api/inspect")
def inspect(path: str) -> dict[str, Any]:
    """Quick inspection of a path, before importing."""
    try:
        return registry.inspect(path)
    except Exception as exc:
        raise _fail(str(exc))


# ==========================================================================
# Datasets
# ==========================================================================

@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    return {"datasets": state.list()}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    data = entry.summary(state.display_units)
    data["preview"] = session_mod.preview_table(entry.dataset)
    data["reports"] = {k: True for k in entry.reports}
    # The preliminary report travels with the dataset so the interface can show
    # it without a second round trip; the heavier reports stay behind their own
    # endpoints.
    data["reports_data"] = {"preflight": _restated(entry, "preflight")}
    return data


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    state.remove(dataset_id)
    state.touch()
    return {"ok": True}


@app.get("/api/datasets/{dataset_id}/map")
def dataset_map(
    dataset_id: str,
    column: str = sch.VALUE,
    group_column: str | None = None,
) -> dict[str, Any]:
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    return session_mod.map_payload(entry.dataset, column, group_column=group_column)


@app.get("/api/datasets/{dataset_id}/stats")
def dataset_stats(dataset_id: str, column: str = sch.VALUE) -> dict[str, Any]:
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    return {"column": column, "stats": entry.dataset.stats(column)}


@app.get("/api/datasets/{dataset_id}/report/{kind}")
def dataset_report(dataset_id: str, kind: str) -> dict[str, Any]:
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    if entry.reports.get(kind) is None:
        raise _fail(f"This dataset has no '{kind}' report.", 404)
    return _restated(entry, kind)


def _restated(entry, kind: str) -> dict[str, Any] | None:
    """A stored report with its sentences in the units now on screen.

    A report is stored as it was made: numbers always metric, sentences in
    whatever unit set was in force when it was made — which is right for
    the answer that call returned, and stale the moment the picker moves.
    So every read comes through here, and the sentences are written again
    from those same numbers in ``state.display_units``. A report made an
    hour ago in acres reads in hectares the moment the picker changes, and
    a project saved last season opens in whatever units this session is
    in. Not one stored number is touched.

    The first look is re-read rather than rewritten: it is a few passes
    over the columns, it costs milliseconds, and reading the data again
    is the one way its sentences cannot drift from what they describe.
    An economic report says nothing with a unit in it — its notes are
    written free of one on purpose, so the interface can put them beside
    figures it converts itself — and comes back as it is.
    """
    report = entry.reports.get(kind)
    if report is None:
        return None
    units = state.display_units
    try:
        if kind == "preflight":
            return preflight_mod.run(entry.dataset, units)
        if kind == "terrain":
            return terrain_analysis.restate(report, units)
        if kind == "terrain_yield":
            return {**report, "findings": yieldrelief.findings(report, units)}
        if kind == "clean":
            return clean_pipeline.restate(report, units, entry.dataset.meta.operation)
    except Exception:
        # A report that cannot be said again is still worth showing as it
        # was written: the numbers in it are right either way.
        return report
    return report


# ==========================================================================
# Cleaning
# ==========================================================================

@app.post("/api/datasets/{dataset_id}/clean")
def clean_dataset(dataset_id: str, request: CleanRequest) -> dict[str, Any]:
    """Run the cleaning and return the report with the resulting datasets."""
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    _refuse_elevation(entry, "cleaning")

    if request.steps:
        config = {"corrections": request.corrections, "steps": request.steps}
    else:
        preset_key = request.preset or clean_pipeline.preset_for(entry.dataset.meta.operation)
        preset = clean_pipeline.PRESETS.get(preset_key)
        if preset is None:
            raise _fail(f"Preset '{preset_key}' does not exist.")
        config = {
            "corrections": {**preset["corrections"], **request.corrections},
            "steps": preset["steps"],
        }

    try:
        result = clean_pipeline.run(
            entry.dataset, config, request.value_column, units=state.display_units)
    except Exception as exc:
        raise _fail(f"Cleaning failed: {exc}")

    clean_summary = _register(result.clean, f"{entry.label} · clean", "clean", entry.id)
    removed_summary = None
    if len(result.removed):
        removed_summary = _register(
            result.removed, f"{entry.label} · removed", "clean_removed", entry.id
        )

    state.get(clean_summary["id"]).reports["clean"] = result.report
    entry.reports["clean"] = result.report
    state.touch()

    return {
        "report": result.report,
        "clean": clean_summary,
        "removed": removed_summary,
        "config": config,
    }


# ==========================================================================
# Economic analysis
#
# The path and the report key stay 'difm': they are the wire names, and a
# project file, a stored report or a script written against the old name must
# keep working. Only the label the user reads changed.
# ==========================================================================

@app.post("/api/datasets/{dataset_id}/difm")
def difm(dataset_id: str, request: DifmRequest) -> dict[str, Any]:
    """Fit the yield response and find the economic optimum rate."""
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    _refuse_elevation(entry, "the economic analysis")
    try:
        report = difm_analysis.analyze(
            entry.dataset,
            rate_column=request.rate_column,
            value_column=request.value_column,
            crop_price=request.crop_price,
            input_cost=request.input_cost,
            cell_m=request.cell_m,
            edge_margin_m=request.edge_margin_m,
            zone_column=request.zone_column,
            models=request.models,
            rate_max=request.rate_max,
            units=state.display_units,
        )
    except Exception as exc:
        raise _fail(f"The economic analysis failed: {exc}")
    entry.reports["difm"] = report
    state.touch()
    return report


@app.post("/api/datasets/{dataset_id}/augmenta")
def augmenta_report(dataset_id: str) -> dict[str, Any]:
    """Vigour against rate, specific to Augmenta data."""
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    report = augmenta_mod.vigor_rate_summary(entry.dataset)
    entry.reports["augmenta"] = report
    state.touch()
    return report


# ==========================================================================
# Trial design
# ==========================================================================

@app.post("/api/design")
def design(request: DesignRequest) -> dict[str, Any]:
    """Generate the strip layout for an on-farm trial."""
    boundary: list[tuple[float, float]] | None = None

    if request.boundary_dataset_id:
        try:
            entry = state.get(request.boundary_dataset_id)
        except KeyError as exc:
            raise _fail(str(exc), 404)
        boundary = _boundary_from_dataset(entry.dataset)
        if boundary is None:
            raise _fail(
                "The chosen dataset has no usable field boundary. Import a boundary "
                "shapefile, or draw the field on the map."
            )
    elif request.boundary:
        boundary = [(float(p[0]), float(p[1])) for p in request.boundary]

    if not boundary:
        raise _fail("Give the field boundary to lay the trial out on.")

    try:
        result = difm_design.design_strips(
            boundary_lonlat=boundary,
            rates=request.rates,
            implement_width_m=request.implement_width_m,
            passes_per_strip=request.passes_per_strip,
            blocks=request.blocks,
            angle_deg=request.angle_deg,
            buffer_m=request.buffer_m,
            seed=request.seed,
            units=state.display_units,
        )
    except Exception as exc:
        raise _fail(str(exc))

    # Kept on the project, not only in the browser: the layout is what the
    # export offers as the prescription, and a project file that lost it
    # would hand back a session with the trial to lay out again. The request
    # travels with it so the settings can be read off a reopened file.
    state.project["design"] = {
        "result": result,
        "request": request.model_dump(),
        "dataset_id": request.boundary_dataset_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    state.touch()
    return result


def _boundary_from_dataset(dataset) -> list[tuple[float, float]] | None:
    """Extract a boundary from the dataset.

    The order of preference matters: a boundary declared in the file is the
    field's real limit, while the convex hull of the points is only the
    envelope of where the machine drove — and it inflates headlands and detours.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    setup = dataset.meta.extra.get("field_setup")
    for fld in (setup or {}).get("fields", []):
        for rings in fld.get("boundaries", []):
            exterior = next((r for r in rings if r.get("type") == 1), None)
            if exterior and len(exterior["points"]) >= 3:
                return [(float(x), float(y)) for x, y in exterior["points"]]

    if dataset.geometry:
        polygons = [g for g in dataset.geometry if g is not None and not g.is_empty]
        if polygons:
            merged = unary_union(polygons)
            if merged.geom_type == "MultiPolygon":
                merged = max(merged.geoms, key=lambda g: g.area)
            if merged.geom_type == "Polygon":
                return [(float(x), float(y)) for x, y in merged.exterior.coords]

    if sch.LON in dataset.df.columns:
        frame = dataset.df[[sch.LON, sch.LAT]].dropna()
        if len(frame) >= 3:
            hull = MultiPoint(list(zip(frame[sch.LON], frame[sch.LAT]))).convex_hull
            if hull.geom_type == "Polygon":
                return [(float(x), float(y)) for x, y in hull.exterior.coords]
    return None


# ==========================================================================
# Export
# ==========================================================================


@app.get("/api/datasets/{dataset_id}/setup")
def dataset_setup(dataset_id: str) -> dict[str, Any]:
    """A dataset's boundary and guidance lines, as GeoJSON.

    This is what allows importing setup from one monitor and sending it to
    another: the boundary and the AB lines cross the app without any
    reconstruction.
    """
    try:
        entry = state.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc), 404)

    setup = entry.dataset.meta.extra.get("field_setup")
    if not setup:
        boundary = _boundary_from_dataset(entry.dataset)
        if not boundary:
            return {"available": False, "reason": "This dataset carries no boundary and no AB lines."}
        setup = {"fields": [{
            "name": entry.dataset.meta.field_name or entry.label,
            "boundaries": [[{"type": 1, "points": boundary}]],
            "headlands": [], "obstacles": [], "guidance_lines": [],
        }]}
        derived = True
    else:
        derived = False

    fields = []
    for fld in setup.get("fields", []):
        outer = None
        for rings in fld.get("boundaries", []):
            exterior = next((r for r in rings if r.get("type") == 1), None)
            if exterior:
                outer = exterior["points"]
                break
        fields.append({
            "name": fld.get("name"),
            "area_m2": fld.get("area_m2"),
            "boundary": outer,
            "guidance_lines": fld.get("guidance_lines", []),
            "guidance_geojson": guidance_mod.ab_lines_to_geojson(
                fld.get("guidance_lines", [])
            ),
        })

    return {
        "available": bool(fields),
        "derived_from_points": derived,
        "fields": fields,
    }


@app.post("/api/guidance")
def make_guidance(request: GuidanceRequest) -> dict[str, Any]:
    """Create an AB line from a direction, or from two points."""
    if request.point_a and request.point_b:
        try:
            line = guidance_mod.ab_line_from_points(
                (request.point_a[0], request.point_a[1]),
                (request.point_b[0], request.point_b[1]),
                request.name,
            )
        except ValueError as exc:
            raise _fail(str(exc))
        return {"line": line, "geojson": guidance_mod.ab_lines_to_geojson([line])}

    boundary = _resolve_boundary(request.boundary, request.boundary_dataset_id)
    if request.angle_deg is None:
        raise _fail("Give the line's direction, or points A and B.")
    try:
        line = guidance_mod.ab_line_from_direction(boundary, request.angle_deg, request.name)
    except ValueError as exc:
        raise _fail(str(exc))
    return {"line": line, "geojson": guidance_mod.ab_lines_to_geojson([line])}


@app.post("/api/export/package")
def export_package(request: PackageRequest) -> dict[str, Any]:
    """Assemble the full package for the chosen monitor and return the ZIP."""
    profile = packages_mod.get_profile(request.monitor)

    boundary = None
    if request.boundary or request.boundary_dataset_id:
        boundary = _resolve_boundary(request.boundary, request.boundary_dataset_id)

    dataset = None
    if request.dataset_id:
        try:
            dataset = state.get(request.dataset_id).dataset
        except KeyError as exc:
            raise _fail(str(exc), 404)

    features = request.features
    notes: list[str] = []
    if features:
        features, note = _convert_features_rate(
            features, request.rate_property, request.rate_kind,
            request.rate_unit, request.crop,
        )
        if note:
            notes.append(note)

    index = len(list(state.exports.iterdir())) + 1
    out_dir = state.exports / f"package_{profile.key}_{index}"

    try:
        result = writers.build_package(
            out_dir,
            monitor=request.monitor,
            prescription=features,
            boundary=boundary,
            guidance_lines=request.guidance_lines or None,
            dataset=dataset,
            rate_property=request.rate_property,
            rate_kind=request.rate_kind,
            rate_unit=request.rate_unit,
            cell_m=request.cell_m,
            field_name=request.field_name,
            task_name=request.task_name,
            product_name=request.product_name,
            customer_name=request.customer_name,
            farm_name=request.farm_name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _fail(f"Failed to assemble the package: {exc}")

    if not result["contents"]:
        raise _fail(
            f"Nothing was generated for the {profile.label}. Check that a prescription, "
            "a boundary or AB lines are selected."
        )

    # Verification runs over the files just written, not over what we intended
    # to write: that is the difference between trusting the code and checking
    # the result. The ZIP is only assembled afterwards.
    verification = validate_mod.validate_package(out_dir, request.monitor)

    zip_path = state.exports / f"{out_dir.name}.zip"
    bundle_info = writers.bundle([out_dir], zip_path)
    token = state.register_file(zip_path)

    return {
        **result,
        "verification": verification,
        "notes": notes,
        "bundle": {
            "entries": bundle_info["entries"],
            "download_url": f"/api/download/{token}",
            "filename": zip_path.name,
        },
    }


def _resolve_boundary(
    explicit: list[list[float]] | None,
    dataset_id: str | None,
) -> list[tuple[float, float]]:
    """Resolve the boundary coming from the map or from a loaded dataset."""
    if explicit:
        return [(float(p[0]), float(p[1])) for p in explicit]
    if dataset_id:
        try:
            entry = state.get(dataset_id)
        except KeyError as exc:
            raise _fail(str(exc), 404)
        boundary = _boundary_from_dataset(entry.dataset)
        if boundary:
            return boundary
    raise _fail("Give the field boundary, or pick a dataset that carries one.")


#: Unit group matching each ISOXML rate kind.
RATE_KIND_GROUP = {"mass": "rate_mass", "volume": "rate_volume", "count": "rate_count"}

#: Every unit a dataset's value may carry and still be a rate the export
#: converts: the keys of the mass, volume and count rate groups.
RATE_UNIT_KEYS = frozenset(
    entry["key"]
    for group in ("rate_mass", "rate_volume", "rate_count")
    for entry in units_mod.UNIT_GROUPS[group]["units"]
)


def _value_is_rate(dataset) -> bool:
    """Whether the dataset's value columns hold a rate the export may scale.

    The unit picker of the Export tab is a rate unit, and dividing by its
    factor only makes sense on a yield or an applied rate. A monitor log's
    reader leaves ``meta.value_unit`` blank, and its value is the internal
    kg/ha; a reader that names another unit — a DEM's metres, or the zone
    codes of the terrain analyser, whose 1, 2, 3 were coming out as 0.9,
    1.8, 2.7 lb/ac — is saying the value is not a rate, and so is the
    elevation operation.
    """
    meta = dataset.meta
    if meta.operation == "elevation":
        return False
    unit = (meta.value_unit or "").strip()
    return not unit or unit in RATE_UNIT_KEYS


def _convert_features_rate(
    features: dict[str, Any],
    rate_property: str,
    rate_kind: str,
    rate_unit: str,
    crop: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Convert the features' rate to the chosen output unit.

    ISOXML is the exception: the standard fixes the unit of each DDI, so the
    grid is always written from the internal value. Only shapefile, CSV and
    GeoJSON carry the unit the user asked for.
    """
    group = RATE_KIND_GROUP.get(rate_kind, "rate_mass")
    internal = units_mod.UNIT_GROUPS[group]["internal"]
    if not rate_unit or rate_unit == internal:
        return features, None

    try:
        factor = units_mod.unit_factor(group, rate_unit, crop)
    except ValueError as exc:
        raise _fail(str(exc))
    if not factor:
        return features, None

    converted = {
        "type": "FeatureCollection",
        "features": [
            {
                **feature,
                "properties": {
                    **(feature.get("properties") or {}),
                    rate_property: (
                        round(float((feature.get("properties") or {})[rate_property]) / factor, 4)
                        if (feature.get("properties") or {}).get(rate_property) is not None
                        else None
                    ),
                },
            }
            for feature in features.get("features", [])
        ],
    }
    return converted, f"Rate converted from {internal} to {rate_unit} when written."


@app.post("/api/export")
def export(request: ExportRequest) -> dict[str, Any]:
    """Generate the requested files and return the download links."""
    outputs: list[dict[str, Any]] = []
    generated: list[Path] = []
    out_dir = state.exports / f"export_{len(list(state.exports.iterdir())) + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "".join(c for c in request.task_name if c.isalnum() or c in "_- ").strip() or "export"
    base = base.replace(" ", "_")

    notes: list[str] = []
    features_internal = request.features
    features_out = request.features
    if request.features:
        features_out, note = _convert_features_rate(
            request.features, request.rate_property, request.rate_kind,
            request.rate_unit, request.crop,
        )
        if note:
            notes.append(note)

    try:
        if request.features:
            for fmt in request.formats:
                if fmt == "shapefile":
                    info = writers.write_prescription_shapefile(
                        features_out, out_dir / f"{base}.shp",
                        brand=request.brand, rate_property=request.rate_property,
                        rate_unit=request.rate_unit, product=request.product_name,
                    )
                    generated.append(Path(info["path"]))
                elif fmt == "isoxml":
                    # Always from the internal value: the standard's DDI already
                    # fixes the unit written into the binary.
                    info = writers.write_prescription_isoxml(
                        features_internal, out_dir,
                        rate_property=request.rate_property, rate_kind=request.rate_kind,
                        cell_m=request.cell_m, task_name=request.task_name,
                        field_name=request.field_name, product_name=request.product_name,
                    )
                    generated.append(Path(info["path"]))
                elif fmt == "csv":
                    info = writers.write_prescription_csv(
                        features_out, out_dir / f"{base}.csv",
                        rate_property=request.rate_property,
                    )
                    generated.append(Path(info["path"]))
                elif fmt == "geojson":
                    info = writers.write_geojson(features_out, out_dir / f"{base}.geojson")
                    generated.append(Path(info["path"]))
                else:
                    raise _fail(f"Unsupported prescription format: '{fmt}'.")
                outputs.append({"format": fmt, **info})

        elif request.dataset_id:
            entry = state.get(request.dataset_id)
            export_dataset = entry.dataset
            group = RATE_KIND_GROUP.get(request.rate_kind, "rate_mass")
            internal = units_mod.UNIT_GROUPS[group]["internal"]
            if (
                request.rate_unit and request.rate_unit != internal
                and _value_is_rate(entry.dataset)
            ):
                factor = units_mod.unit_factor(group, request.rate_unit, request.crop)
                export_dataset = entry.dataset.copy()
                converted = [
                    column for column in (sch.VALUE, sch.TARGET_RATE, sch.APPLIED_RATE)
                    if column in export_dataset.df.columns
                ]
                for column in converted:
                    export_dataset.df[column] = export_dataset.df[column] / factor
                # The note reports what was done, not what was asked: a
                # dataset with no rate column is written as it is.
                if converted:
                    notes.append(
                        f"Rate columns converted from {internal} to {request.rate_unit}."
                    )
            for fmt in request.formats:
                if fmt == "shapefile":
                    info = writers.write_vector(
                        export_dataset, out_dir / f"{base}.shp", columns=request.columns
                    )
                elif fmt == "geojson":
                    info = writers.write_vector(
                        export_dataset, out_dir / f"{base}.geojson",
                        driver="GeoJSON", columns=request.columns,
                    )
                elif fmt == "gpkg":
                    info = writers.write_vector(
                        export_dataset, out_dir / f"{base}.gpkg",
                        driver="GPKG", columns=request.columns,
                    )
                elif fmt == "csv":
                    info = writers.write_csv(
                        export_dataset, out_dir / f"{base}.csv", columns=request.columns
                    )
                else:
                    raise _fail(f"Unsupported data format: '{fmt}'.")
                generated.append(Path(info["path"]))
                outputs.append({"format": fmt, **info})
        else:
            raise _fail("Give a dataset or a prescription to export.")

    except HTTPException:
        raise
    except KeyError as exc:
        raise _fail(str(exc), 404)
    except Exception as exc:
        raise _fail(f"Export failed: {exc}")

    zip_path = out_dir / f"{base}.zip"
    bundle_info = writers.bundle(generated, zip_path)
    token = state.register_file(zip_path)

    return {
        "outputs": outputs,
        "notes": notes,
        "bundle": {
            "entries": bundle_info["entries"],
            "download_url": f"/api/download/{token}",
            "filename": zip_path.name,
        },
        "folder": str(out_dir),
    }


@app.post("/api/validate")
def validate_folder(path: str) -> dict[str, Any]:
    """Check a package already written to disk, including from another session."""
    target = Path(path).expanduser()
    if not target.exists():
        raise _fail(f"Folder not found: {target}")
    return validate_mod.validate_package(target, "generic")


@app.get("/api/inspect/card")
def inspect_card(path: str) -> dict[str, Any]:
    """Inventory a John Deere card without importing anything."""
    target = Path(path).expanduser()
    if not target.exists():
        raise _fail(f"Path not found: {target}")
    inv = jd_mod.inventory(target)
    return {**inv.to_dict(), "layers": jd_mod.readable_layers(inv)}


@app.get("/api/download/{token}")
def download(token: str):
    try:
        path = state.file_for(token)
    except KeyError as exc:
        raise _fail(str(exc), 404)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


# ==========================================================================
# Interface
# ==========================================================================

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise _fail("Interface not found. Reinstall the app.", 500)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "datasets": len(state.list()), "workdir": str(state.workdir)}


def _include_feature_routers() -> None:
    """Include every ``router`` found in :mod:`agrosuite.app.routes`."""
    import importlib
    import pkgutil

    from . import routes as routes_package

    for module_info in pkgutil.iter_modules(routes_package.__path__):
        module = importlib.import_module(f"{routes_package.__name__}.{module_info.name}")
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)


_include_feature_routers()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
