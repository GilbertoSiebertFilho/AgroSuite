"""Saving a session to a project file and reopening it.

A session is hours of work: files opened, units declared, cleaning run and
judged, roles assigned, prices typed in, a trial laid out. Closing the app
used to lose all of it, and re-importing the raw files does not bring it
back — the cleaning report, the removal reasons, the roles, the reviewed
ticks and the strip layout exist only in memory. The project file captures
that whole state so the work can be picked up where it was left.

The file is one ZIP with the extension ``.agrosuite``: a ``manifest.json``
describing the session and one parquet file per dataset. Parquet rather than
CSV because CSV loses the very things that make a dataset usable again —
column types, datetime precision and the difference between an empty string
and a missing value. Geometry travels as WKB beside each dataset, which
round-trips coordinates bit for bit; GeoJSON would round them.

Nothing here decides *when* to save, and no function reads the clock: the
timestamp is passed in by the caller, so a saved file says what the caller
meant it to say and the library is reproducible under test. There are two
callers — the Save button in :mod:`agrosuite.app.routes.persist` and the
auto-saver in :mod:`agrosuite.app.autosave` — and they write the same file
through the same function, so a project saved by hand and one the app saved
by itself are the same thing on disk.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import fields as dataclass_fields
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__
from ..core import schema as sch
from ..core.dataset import Dataset, DatasetMeta
from . import session as session_mod

#: Bumped when a file written today would not open on the code that wrote
#: version N-1. Readers refuse files newer than they understand rather than
#: guessing at them.
FORMAT_VERSION = 1

#: Extension of a project file. Also known to the format registry, which lists
#: it in the file browser and refuses to *import* it.
EXTENSION = ".agrosuite"

#: Marker that tells a manifest apart from any other ZIP that happens to carry
#: a manifest.json.
APP_MARKER = "AgroSuite"

#: How many recently used project files to remember.
RECENT_LIMIT = 10

_MANIFEST = "manifest.json"
_DATA_DIR = "data"
_WKB_COLUMN = "wkb"


# ==========================================================================
# Home folder and recent files
# ==========================================================================

def home_dir() -> Path:
    """Where AgroSuite keeps its own small files, such as the recent list.

    Read from the environment on every call rather than at import: the tests
    point it at a temporary folder, and a packaged install may set it to a
    per-user location the app cannot know in advance.
    """
    configured = os.environ.get("AGROSUITE_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".agrosuite"


def _recent_file() -> Path:
    return home_dir() / "recent.json"


def _read_recent() -> list[dict[str, Any]]:
    """The stored recent list, or nothing if there is none.

    A damaged list is not user data — it is a convenience the app rebuilds as
    files are opened — so a corrupt file is treated as empty rather than as
    an error that would stop the app from opening anything.
    """
    path = _recent_file()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = payload.get("recent") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and item.get("path")]


def _write_recent(items: list[dict[str, Any]]) -> None:
    path = _recent_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps({"recent": items}, indent=2).encode("utf-8"))


def remember_recent(path: str | Path, saved_at: str | None = None) -> list[dict[str, Any]]:
    """Put ``path`` at the top of the recent list and return the list.

    Deduplicated by path, so reopening the same project every day keeps one
    entry rather than ten copies of it. ``saved_at`` is read from the file
    when the caller does not know it.
    """
    resolved = Path(path).expanduser().resolve()
    if saved_at is None:
        try:
            saved_at = read_manifest(resolved).get("saved_at")
        except (OSError, ValueError):
            saved_at = None

    key = str(resolved)
    items = [item for item in _read_recent() if item.get("path") != key]
    items.insert(0, {"path": key, "name": resolved.stem, "saved_at": saved_at})
    _write_recent(items[:RECENT_LIMIT])
    return recent_files()


def forget_recent(path: str | Path) -> list[dict[str, Any]]:
    """Take a file off the recent list, and return what is left.

    For a project that has moved: renamed, or followed a change of the
    projects folder. Its old name would otherwise sit in the list as an
    entry that is "not found" and leads nowhere — for the very project that
    is open, under its new name, one line further up.
    """
    key = str(Path(path).expanduser().resolve())
    items = [item for item in _read_recent() if item.get("path") != key]
    _write_recent(items)
    return recent_files()


def recent_files() -> list[dict[str, Any]]:
    """Recently used project files, most recent first.

    ``exists`` is checked now, not when the entry was written: a project on a
    USB stick that is no longer plugged in should be shown as unavailable
    rather than fail on click.
    """
    return [
        {
            "path": item["path"],
            "name": item.get("name") or Path(item["path"]).stem,
            "exists": Path(item["path"]).exists(),
            "saved_at": item.get("saved_at"),
        }
        for item in _read_recent()
    ]


def slugify(name: str) -> str:
    """Turn a project name into a file name that every filesystem accepts."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", str(name or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    return cleaned or "project"


# ==========================================================================
# Saving
# ==========================================================================

def save_session(state: session_mod.Session, path: str | Path, saved_at: str) -> dict[str, Any]:
    """Write the whole session to ``path`` as a project file.

    Parameters
    ----------
    state:
        The session to save.
    path:
        Destination. The caller decides whether an existing file may be
        replaced; this function only makes sure the replacement is complete
        — the file is assembled beside the target and moved into place in
        one step, so an interrupted save leaves the previous copy untouched.
    saved_at:
        ISO timestamp recorded in the manifest, supplied by the caller.
    """
    target = Path(path)
    manifest: dict[str, Any] = {
        "app": APP_MARKER,
        "app_version": __version__,
        "format": FORMAT_VERSION,
        # When the work was left. It is what the app names when it offers to
        # pick the project up again, so it is read straight off the manifest
        # rather than off the file's own timestamp, which a copy would move.
        "saved_at": saved_at,
        "project": _project_to_manifest(state.project),
        # Where the reader was: the dataset selected and the tab open. Coming
        # back to a project on the Data tab with nothing selected is not
        # coming back to where the work was left.
        #
        # This is an added key, not a new format version: files written
        # before it exist on the owner's machine, and they still open (see
        # _view_from_manifest). Bumping FORMAT_VERSION would have refused
        # files this code reads perfectly well, and a version before this one
        # ignores a manifest key it does not know.
        "view": _view_to_manifest(state.view),
        "datasets": [],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for entry in _entries(state):
            record, data, geometry = _pack_entry(entry)
            # Parquet is already compressed; deflating it again costs time
            # on a large map and saves almost nothing.
            zf.writestr(record["data"], data, compress_type=zipfile.ZIP_STORED)
            if geometry is not None:
                zf.writestr(record["geometry"], geometry, compress_type=zipfile.ZIP_STORED)
            manifest["datasets"].append(record)
        zf.writestr(
            _MANIFEST,
            json.dumps(manifest, indent=2, default=_json_default),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    write_atomically(target, buffer.getvalue())
    return {
        "path": str(target),
        "saved_at": saved_at,
        "datasets": len(manifest["datasets"]),
        "size_bytes": target.stat().st_size,
    }


def _entries(state: session_mod.Session) -> list[session_mod.Entry]:
    """Every entry, in the order the session holds them.

    Insertion order is the history of the session — parents before the
    datasets derived from them — and reloading in the same order keeps the
    list the user sees identical to the one they saved.
    """
    return [state.get(item["id"]) for item in state.list()]


def _pack_entry(entry: session_mod.Entry) -> tuple[dict[str, Any], bytes, bytes | None]:
    dataset = entry.dataset
    frame, coerced = _arrow_safe(dataset.df)
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(
            f"Dataset '{entry.label}' has duplicated column names "
            f"({', '.join(duplicates)}), which the project file cannot hold. "
            "Re-import the file with distinct column names."
        )

    data = io.BytesIO()
    frame.to_parquet(data, engine="pyarrow", index=False)

    geometry_bytes: bytes | None = None
    if dataset.geometry is not None:
        geometry_bytes = _geometry_to_parquet(dataset.geometry)

    record = {
        "id": entry.id,
        "label": entry.label,
        "origin": entry.origin,
        "parent_id": entry.parent_id,
        "role": entry.role,
        "reports": entry.reports,
        "meta": _meta_to_dict(dataset.meta),
        "metric_crs": dataset.metric_crs,
        "has_geometry": dataset.geometry is not None,
        "rows": int(len(frame)),
        # Recorded so a reader on another pandas can put the types back the
        # way they were, rather than trust what parquet infers.
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "coerced_columns": coerced,
        "data": f"{_DATA_DIR}/{entry.id}.parquet",
        "geometry": f"{_DATA_DIR}/{entry.id}.geometry.parquet" if geometry_bytes is not None else None,
    }
    return record, data.getvalue(), geometry_bytes


def _arrow_safe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A shallow copy that parquet will accept, and the columns it changed.

    A shapefile attribute table or an Excel sheet can carry a column that
    mixes numbers and text; Arrow refuses such a column outright. Rather than
    fail the whole save over one attribute, the column is written as text and
    the manifest says so — the user keeps their session and knows what moved.
    """
    frame = frame.copy(deep=False)
    frame.columns = [str(column) for column in frame.columns]
    coerced: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if series.dtype != object:
            continue
        try:
            pa.array(series, from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
            frame[column] = series.where(series.notna(), None).map(
                lambda value: value if value is None else str(value)
            )
            coerced.append(column)
    return frame, coerced


def _geometry_to_parquet(geometry: list) -> bytes:
    import shapely

    wkb = [None if geom is None else shapely.to_wkb(geom) for geom in geometry]
    table = pa.table({_WKB_COLUMN: pa.array(wkb, type=pa.binary())})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _meta_to_dict(meta: DatasetMeta) -> dict[str, Any]:
    # ``to_dict`` adds the display label for the operation; that is derived,
    # not stored, and ``DatasetMeta(**data)`` would reject it on the way back.
    return {item.name: getattr(meta, item.name) for item in dataclass_fields(DatasetMeta)}


def _project_to_manifest(project: dict[str, Any]) -> dict[str, Any]:
    data = dict(project)
    data["reviewed"] = sorted(project.get("reviewed") or [])
    return data


def _view_to_manifest(view: dict[str, Any] | None) -> dict[str, Any]:
    """The view state as it goes into the manifest: only what it is for.

    Text and nothing else, so that whatever the interface starts keeping
    beside the two fields cannot turn a project file into something only
    this version can read.
    """
    source = view or {}
    return {
        key: value if isinstance(value, str) and value else None
        for key, value in ((key, source.get(key)) for key in session_mod.default_view())
    }


def _json_default(value: Any) -> Any:
    """Make the odd non-JSON value in a report serializable.

    Reports are built from numpy results; most are converted at source, but a
    stray ``int64`` or an array is not worth failing a save over. Numbers stay
    numbers so the interface can still do arithmetic on them after a reload;
    anything unknown becomes text, which at least keeps the record readable.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_atomically(target: Path, payload: bytes) -> None:
    """Write to a sibling temporary file and move it over the target.

    The move is the only moment the target changes, so a crash or a full disk
    midway leaves the previous file intact instead of a truncated ZIP that
    will never open again.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# ==========================================================================
# Loading
# ==========================================================================

def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a project file's manifest without loading its data.

    Cheap enough to call on a file the user merely pointed at: it reads the
    ZIP directory and one small JSON member.
    """
    path = Path(path)
    name = path.name
    if path.is_dir():
        raise IsADirectoryError(
            f"That is a folder; choose the {EXTENSION} file inside it: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"'{name}' is not an AgroSuite project file: it is not a ZIP archive. "
            f"Project files are written by 'Save project' and end in {EXTENSION}."
        )
    with zipfile.ZipFile(path) as zf:
        if _MANIFEST not in zf.namelist():
            raise ValueError(
                f"'{name}' is not an AgroSuite project file: it has no manifest. "
                "If it is monitor data in a ZIP, open it from the Data tab instead."
            )
        try:
            manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"'{name}' has a damaged manifest and cannot be opened: {exc}")

    if not isinstance(manifest, dict) or manifest.get("app") != APP_MARKER:
        raise ValueError(
            f"'{name}' was not written by AgroSuite; only files saved with "
            "'Save project' can be opened here."
        )
    version = manifest.get("format")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"'{name}' does not declare a usable format version.")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"'{name}' was saved by a newer AgroSuite (project format {version}; this "
            f"version reads up to {FORMAT_VERSION}). Update the app to open it."
        )
    if not isinstance(manifest.get("datasets"), list):
        raise ValueError(f"'{name}' lists no datasets; the file is damaged.")
    return manifest


def load_session(state: session_mod.Session, path: str | Path) -> dict[str, Any]:
    """Replace the session's contents with the project saved at ``path``.

    Every dataset is read and rebuilt before the session is touched: a
    damaged file must fail with the current work still in place, never with
    half a project loaded and the other half gone.
    """
    path = Path(path)
    manifest = read_manifest(path)

    restored: list[tuple[dict[str, Any], Dataset]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        for position, record in enumerate(manifest["datasets"], start=1):
            # The ids are checked here, with the data, and not while adding
            # to the session: a record without one, or two records sharing
            # one, would otherwise surface after the session was cleared.
            dataset_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError(
                    f"'{path.name}' is damaged: dataset record {position} has no id. "
                    "Save the project again from the session it came from, or open "
                    "another file; the session that is open was left as it was."
                )
            if dataset_id in seen:
                raise ValueError(
                    f"'{path.name}' is damaged: two datasets share the id '{dataset_id}'. "
                    "Save the project again from the session it came from, or open "
                    "another file; the session that is open was left as it was."
                )
            seen.add(dataset_id)
            try:
                restored.append((record, _unpack_entry(zf, record)))
            except Exception as exc:
                raise ValueError(
                    f"Dataset '{record.get('label', record.get('id'))}' in "
                    f"'{path.name}' could not be read: {exc}"
                ) from exc

    state.clear()
    state.project = _project_from_manifest(manifest.get("project") or {})
    state.view = _view_from_manifest(manifest.get("view"), [r["id"] for r, _ in restored])

    entries: list[session_mod.Entry] = []
    for record, dataset in restored:
        entry = state.add(
            dataset,
            label=record.get("label"),
            origin=record.get("origin") or "import",
            parent_id=record.get("parent_id"),
            dataset_id=record["id"],
        )
        entry.reports = dict(record.get("reports") or {})
        entry.role = record.get("role")
        entries.append(entry)

    return {
        "path": str(path),
        "saved_at": manifest.get("saved_at"),
        "format": manifest["format"],
        "project": state.project["name"],
        "datasets": [entry.summary(state.display_units) for entry in entries],
        # Handed back with the datasets so the interface can put the strips
        # back on the map without a second request.
        "design": state.project.get("design"),
        # And so it can select the dataset and open the tab the session was
        # left on, in the same round trip.
        "view": dict(state.view),
    }


def _unpack_entry(zf: zipfile.ZipFile, record: dict[str, Any]) -> Dataset:
    frame = pd.read_parquet(io.BytesIO(zf.read(record["data"])), engine="pyarrow")
    _restore_dtypes(frame, record.get("dtypes") or {})

    geometry = None
    if record.get("geometry"):
        geometry = _geometry_from_parquet(zf.read(record["geometry"]))

    meta = _meta_from_dict(record.get("meta") or {})
    dataset = Dataset(frame, meta, geometry)

    # The constructor projects lon/lat afresh in whatever CRS it would pick
    # today. The saved session was analysed in the CRS it had then — its
    # x/y, its cell grid, its neighbourhoods — so that one wins, and the
    # stored x/y come back untouched rather than recomputed.
    saved_crs = record.get("metric_crs")
    if saved_crs and dataset.metric_crs != saved_crs:
        if sch.LON in dataset.df.columns and sch.LAT in dataset.df.columns:
            dataset.project(saved_crs)
        dataset.metric_crs = saved_crs
    for column in (sch.X, sch.Y):
        if column in frame.columns:
            dataset.df[column] = frame[column].to_numpy()
    return dataset


def _restore_dtypes(frame: pd.DataFrame, dtypes: dict[str, str]) -> None:
    """Cast columns back to the dtypes they were saved with.

    Parquet keeps the values; what it cannot promise across pandas versions
    is the dtype label — a string column may come back as object, a
    timestamp at another resolution. Where the recorded dtype cannot be
    applied the column is left as read: the values are right either way.
    """
    for column, dtype in dtypes.items():
        if column not in frame.columns or str(frame[column].dtype) == dtype:
            continue
        try:
            frame[column] = frame[column].astype(dtype)
        except (TypeError, ValueError):
            continue


def _geometry_from_parquet(payload: bytes) -> list:
    import shapely

    table = pq.read_table(io.BytesIO(payload))
    return [None if wkb is None else shapely.from_wkb(wkb) for wkb in table.column(_WKB_COLUMN).to_pylist()]


def _meta_from_dict(data: dict[str, Any]) -> DatasetMeta:
    # Only the fields the dataclass knows: a file written by a version with
    # one more field must still open here.
    known = {item.name for item in dataclass_fields(DatasetMeta)}
    return DatasetMeta(**{key: value for key, value in data.items() if key in known})


def _view_from_manifest(data: Any, dataset_ids: list[str]) -> dict[str, Any]:
    """The view state a file carries, or an empty one when it carries none.

    A project file written before the view existed is not a damaged file: it
    simply says nothing about where the reader was, and the interface falls
    back to what it has always done — the first dataset, the first tab.

    A dataset id the file no longer holds is dropped rather than restored:
    it would select nothing and leave the panel empty.
    """
    view = session_mod.default_view()
    if not isinstance(data, dict):
        return view
    for key in view:
        value = data.get(key)
        if isinstance(value, str) and value:
            view[key] = value
    if view["dataset_id"] not in dataset_ids:
        view["dataset_id"] = None
    return view


def _project_from_manifest(data: dict[str, Any]) -> dict[str, Any]:
    project = session_mod.default_project()
    for key in project:
        if key in data and data[key] is not None:
            project[key] = data[key]
    project["reviewed"] = set(data.get("reviewed") or [])
    project["roles"] = dict(data.get("roles") or {})
    project["prices"] = dict(data.get("prices") or {})
    return project
