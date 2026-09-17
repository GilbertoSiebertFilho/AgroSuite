"""App session state.

Since AgroSuite runs locally and for one person, state lives in memory: the
loaded datasets, the cleaning and analysis results, and the files generated
for export. What is on disk is a copy: the auto-saver writes the session to
the projects folder whenever it has changed and gone quiet (see
:mod:`agrosuite.app.autosave`), and it is this object that says when that
happened — :meth:`Session.touch`, called by every route that changes
anything.

A dataset is never overwritten. Cleaning produces two new datasets — clean
and removed — that live alongside the original. That is what makes it
possible to compare before and after, and to undo a bad cleaning without
re-importing the file.

**The unit set the reader works in lives here too**, in
:attr:`Session.display_units`. Every number the app stores is metric and
the interface converts it for the screen, but a *sentence* cannot be
converted for the screen: its numbers are written inside it, and restating
it in another unit means deciding again what it says. So the generators
write their prose in the reader's unit set, and they need to know it at
the moment they write.

It is kept on the session rather than sent with every request because it
is a property of the reader, not of the request — one person, one screen,
one set of units at a time — and because the prose is written in a dozen
places (the first look, the cleaning report, the relief, the economics,
the notes on a joined layer) that would otherwise each grow a parameter
the interface has to remember to fill. The interface sets it whenever the
preset or one picker changes; a request that names its own set (the
printed report, which is a document for someone else, and the terrain
endpoints, which were written that way) overrides it for that call.

Nothing stored depends on it. Reports are stored as they were generated,
with metric numbers, and every path that hands prose to a person writes
the sentences again in the unit set in force at that moment — which is
why changing the picker changes every sentence on screen without
re-importing or re-analysing anything, and why a project saved in acres
opens in hectares if that is what the reader is in now.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import schema as sch
from ..core import units as units_mod
from ..core.dataset import Dataset

#: Ceiling on points sent to the map in one response. Above it the sampling is
#: systematic, which preserves the spatial pattern without stalling the browser.
MAP_POINT_LIMIT = 60_000


@dataclass
class Entry:
    """One dataset in the session, with its history."""

    id: str
    dataset: Dataset
    label: str
    origin: str = "import"
    parent_id: str | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    role: str | None = None

    def summary(self, units: dict[str, Any] | None = None) -> dict[str, Any]:
        """The layer as the interface reads it, its remarks in ``units``.

        A reader's remarks about a file — what the raster was resampled to,
        how wide the passes were, what was assumed — are prose with numbers
        written into them, and they were written once, when the file was
        opened. ``units`` writes them again from the facts kept beside
        them, so moving the unit picker moves a layer's notes along with
        every other sentence on screen. ``None`` leaves them as the reader
        wrote them, which is what a caller outside the app wants.
        """
        data = self.dataset.summary()
        if units is not None:
            restated = self._notes_in(units)
            if restated is not None:
                data["meta"] = {**data["meta"], "notes": restated}
        data.update({
            "id": self.id,
            "label": self.label,
            "origin": self.origin,
            "parent_id": self.parent_id,
            "role": self.role,
            "has_clean_report": "clean" in self.reports,
            "has_difm_report": "difm" in self.reports,
        })
        return data

    def _notes_in(self, units: dict[str, Any]) -> list[str] | None:
        """The remarks written again, or ``None`` when there is nothing to
        write again: a layer whose reader kept no facts keeps its
        sentences, because half-restated notes would read worse than
        consistent stale ones."""
        facts = (self.dataset.meta.extra or {}).get("note_facts")
        if not facts:
            return None
        from agrosuite.terrain import notes as notes_mod

        try:
            return notes_mod.render(facts, units)
        except Exception:
            # A fact this version cannot write — a project from a later
            # build — is not worth failing a listing over.
            return None


def default_project() -> dict[str, Any]:
    """A fresh project, as a session starts and as 'New project' returns to.

    One project per session: the set of files describing one field and
    season, plus the prices that turn yield into money. Kept in one place so
    that resetting a session cannot drift from starting one.
    """
    return {
        "name": "Untitled project",
        "goal": "difm",
        "roles": {},        # dataset_id -> role
        "prices": {},       # crop_price, input_cost, currency, crop
        "reviewed": set(),  # dataset ids the user has confirmed
        "exported": False,
        # The trial layout last generated: {result, request, dataset_id,
        # created_at}, or None. It is project state, not a passing result —
        # the export tab offers it as the prescription — so it is saved with
        # the project and comes back when the file is reopened.
        "design": None,
    }


def default_view() -> dict[str, Any]:
    """Where the reader is, as a fresh session starts: nowhere in particular.

    The two things that decide what is on screen — which dataset is selected
    and which tab is open — and nothing else. They are saved with the project
    so that reopening it lands on the work rather than on the Data tab with
    an empty map, and they are kept here rather than in the browser because
    the browser is reloaded, replaced and closed, while the session is what
    gets written to the file.
    """
    return {"dataset_id": None, "tab": None}


class Session:
    """In-memory repository for the session's data."""

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._files: dict[str, Path] = {}
        self._lock = threading.Lock()
        self.project: dict[str, Any] = default_project()
        # The project file the session was opened from or last saved to:
        # ``{"path": Path, "saved_at": str | None, "token": str | None}``,
        # or None for a session that has never touched one. Kept here rather
        # than in the browser so that reloading the page does not forget
        # where the next save should go.
        self.project_file: dict[str, Any] | None = None
        # The file the auto-saver writes this session to, inside the projects
        # folder, or None until it has chosen one. Kept here because it
        # belongs to the session: a rename moves it, a new project drops it,
        # and a reopened project inherits the file it came out of.
        self.autosave_path: Path | None = None
        # What the app reopened by itself when it started, until the
        # interface has said so once. See routes.persist.resume_latest.
        self.resumed: dict[str, Any] | None = None
        # Which dataset is selected and which tab is open.
        self.view: dict[str, Any] = default_view()
        # How many times the session has changed, and when it last did.
        # Together they are the whole of what the auto-saver watches: a
        # number that has moved since the last write, and a clock that has
        # been still long enough to suggest the person has stopped typing.
        self.revision: int = 0
        self.touched_at: float = time.monotonic()
        # The unit set every sentence is written in, until the interface
        # says otherwise: the app's default preset, the same one the
        # pickers open on.
        self.display_units: dict[str, Any] = dict(
            units_mod.UNIT_PRESETS[units_mod.DEFAULT_PRESET]
        )
        self.workdir = Path(tempfile.mkdtemp(prefix="agrosuite_"))
        self.uploads = self.workdir / "uploads"
        self.exports = self.workdir / "exports"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.exports.mkdir(parents=True, exist_ok=True)

    # -- what has changed ------------------------------------------------
    def touch(self) -> None:
        """Say that something in the session changed.

        Every route that changes state calls this, and it does the least it
        can — one increment and one clock reading — because it is on the way
        out of an import of a million records as much as on the way out of a
        tick box. Deciding what to do about the change is somebody else's
        job: see :mod:`agrosuite.app.autosave`.
        """
        with self._lock:
            self.revision += 1
            self.touched_at = time.monotonic()

    # -- datasets --------------------------------------------------------
    def add(
        self,
        dataset: Dataset,
        label: str | None = None,
        origin: str = "import",
        parent_id: str | None = None,
        dataset_id: str | None = None,
    ) -> Entry:
        # A reopened project keeps its original ids: reports, roles and
        # parent links all refer to them, and re-numbering would cut every
        # clean copy loose from the file it came from.
        entry = Entry(
            id=dataset_id or uuid.uuid4().hex[:12],
            dataset=dataset,
            label=label or dataset.meta.name,
            origin=origin,
            parent_id=parent_id,
        )
        with self._lock:
            if entry.id in self._entries:
                raise ValueError(
                    f"Dataset id '{entry.id}' is already in use in this session; "
                    "clear the session before reloading it."
                )
            self._entries[entry.id] = entry
        return entry

    def get(self, dataset_id: str) -> Entry:
        entry = self._entries.get(dataset_id)
        if entry is None:
            raise KeyError(f"Dataset '{dataset_id}' is not loaded in this session.")
        return entry

    def remove(self, dataset_id: str) -> None:
        with self._lock:
            self._entries.pop(dataset_id, None)
            self.project["roles"].pop(dataset_id, None)
            self.project["reviewed"].discard(dataset_id)

    def list(self) -> list[dict[str, Any]]:
        # The snapshot is taken under the lock and summarized outside it: a
        # request that clears or adds a dataset while this one is building
        # its answer must not make the iteration itself fail.
        with self._lock:
            entries = list(self._entries.values())
        return [entry.summary(self.display_units) for entry in entries]

    def clear(self) -> None:
        # Whatever replaces the datasets — a new project, a reopened file —
        # is no longer what the remembered file holds, so the link goes too,
        # and so does the file the auto-saver was writing: the next project
        # gets a file of its own rather than being written over the last
        # one. The view goes with the datasets it pointed at.
        with self._lock:
            self._entries.clear()
            self.project_file = None
            self.autosave_path = None
            self.view = default_view()

    # -- files -----------------------------------------------------------
    def register_file(self, path: Path) -> str:
        """Register a generated file and return its download token."""
        token = uuid.uuid4().hex[:16]
        with self._lock:
            self._files[token] = Path(path)
        return token

    def file_for(self, token: str) -> Path:
        path = self._files.get(token)
        if path is None or not path.exists():
            raise KeyError("File not found, or already removed.")
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


# --------------------------------------------------------------------------
# Serialization for the map and the charts
# --------------------------------------------------------------------------

def map_payload(
    dataset: Dataset,
    column: str = sch.VALUE,
    limit: int = MAP_POINT_LIMIT,
    group_column: str | None = None,
) -> dict[str, Any]:
    """Prepare the points for drawing on the map.

    It returns parallel arrays rather than GeoJSON: for tens of thousands of
    points, GeoJSON multiplies the response size and the browser's parse time
    several times over for no gain — the drawing happens on a canvas, which
    only needs the coordinates and the value.
    """
    df = dataset.df
    if sch.LON not in df.columns or sch.LAT not in df.columns:
        return {"count": 0, "lon": [], "lat": [], "values": [], "column": column}

    columns = [sch.LON, sch.LAT] + ([column] if column in df.columns else [])
    if group_column and group_column in df.columns and group_column not in columns:
        columns.append(group_column)
    frame = df[columns].dropna(subset=[sch.LON, sch.LAT])

    total = len(frame)
    if total > limit:
        # Systematic sampling keeps the spatial coverage of the pass, unlike
        # simply taking the first N rows.
        step = int(np.ceil(total / limit))
        frame = frame.iloc[::step]

    values: list[float | None] = []
    if column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        values = [None if pd.isna(v) else round(float(v), 4) for v in series]

    payload: dict[str, Any] = {
        "count": int(len(frame)),
        "total": int(total),
        "sampled": bool(total > limit),
        "column": column,
        "lon": [round(float(v), 7) for v in frame[sch.LON]],
        "lat": [round(float(v), 7) for v in frame[sch.LAT]],
        "values": values,
        "bounds": dataset.bounds(),
    }

    # A categorical column travelling beside the values is what lets the map
    # show *why* each point was removed, rather than only that it was.
    if group_column and group_column in frame.columns:
        payload["groups"] = [
            None if pd.isna(v) else str(v) for v in frame[group_column]
        ]
        payload["group_column"] = group_column

    if values:
        finite = [v for v in values if v is not None]
        if finite:
            array = np.asarray(finite, dtype="float64")
            payload["scale"] = {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                # The colour scale uses percentiles: one extreme point should
                # not flatten the contrast of the whole map.
                "low": float(np.percentile(array, 2)),
                "high": float(np.percentile(array, 98)),
            }

    if dataset.geometry is not None:
        payload["polygons"] = _polygon_payload(dataset)

    return payload


def _polygon_payload(dataset: Dataset, limit: int = 4000) -> list[list[list[float]]]:
    """Outer rings of the polygons, for drawing boundaries and grids."""
    rings: list[list[list[float]]] = []
    for geometry in (dataset.geometry or [])[:limit]:
        if geometry is None or geometry.is_empty:
            continue
        parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
        for part in parts:
            if part.geom_type == "Polygon":
                rings.append([[round(x, 7), round(y, 7)] for x, y in part.exterior.coords])
            elif part.geom_type == "LineString":
                rings.append([[round(x, 7), round(y, 7)] for x, y in part.coords])
    return rings


def preview_table(dataset: Dataset, rows: int = 25) -> dict[str, Any]:
    """A sample of the table for visual inspection in the interface."""
    frame = dataset.df.head(rows).copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame = frame.replace({np.nan: None, np.inf: None, -np.inf: None})

    records = []
    for record in frame.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = None if not np.isfinite(value) else round(float(value), 4)
            elif isinstance(value, (np.bool_,)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        records.append(clean)

    return {"columns": list(frame.columns), "rows": records}
