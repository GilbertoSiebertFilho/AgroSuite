"""The printable report over HTTP.

The page itself is built by :mod:`agrosuite.report`; this module decides
where the file goes, stamps it with the time, and turns refusals into
messages the interface can show. The unit set travels in the request rather
than being read from anywhere on the server, because the server holds no
display preference: the numbers on paper must match the numbers on the
screen that asked for them.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ... import report as report_mod

router = APIRouter(prefix="/api/report", tags=["report"])


class ReportRequest(BaseModel):
    """What to print, and in which units.

    ``units`` carries the interface's current set — ``yield_unit``,
    ``input_rate_unit``, ``area_unit``, ``length_unit``, ``speed_unit``,
    ``currency``, ``crop``. A key left out falls back to the app's default
    preset, so an older client still gets a report rather than an error.
    """

    dataset_id: str
    units: dict[str, Any] = Field(default_factory=dict)
    #: Heading in place of the project name — for a report about one trial
    #: inside a project that covers several.
    title: str | None = None


def _reserve_report_path(folder: Path) -> Path:
    """``report_<n>.pdf`` for the first ``n`` not yet taken, created empty.

    Counting existing files would reuse a number after one is deleted and
    overwrite the report that took its place. Probing for a free name is not
    enough either: the PDF only reaches the disk once it is fully built, and
    two requests arriving inside that window would both see the same free
    number and the later one would silently replace the earlier file.
    Creating the file exclusively at the moment the name is chosen takes the
    name for everyone else in the same step, so the second request gets the
    next number. A caller that then fails to write must give the name back.
    """
    number = 1
    while True:
        path = folder / f"report_{number}.pdf"
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            number += 1
            continue
        return path


@router.post("")
def create_report(request: ReportRequest) -> dict[str, Any]:
    """Write the report for one dataset and return where to fetch it."""
    from agrosuite.app import server as server_mod

    state = server_mod.state
    try:
        entry = state.get(request.dataset_id)
    except KeyError as exc:
        # str() of a KeyError adds its own quotes around the message.
        raise server_mod._fail(str(exc.args[0] if exc.args else exc), 404)

    if not entry.reports:
        raise server_mod._fail(
            f"There is nothing to report on '{entry.label}' yet. Open it from the Data "
            "tab so the first look runs, then clean it or run the economic analysis."
        )

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    try:
        target = _reserve_report_path(state.exports)
    except OSError as exc:
        raise server_mod._fail(
            f"Could not create a file in '{state.exports}': {exc.strerror or exc}. "
            "Check that the session folder is writable and the disk has space.",
            500,
        )
    written = False
    try:
        result = report_mod.build_pdf(
            entry, target, request.units, generated_at, state.project, request.title,
        )
        written = True
    except ValueError as exc:
        raise server_mod._fail(str(exc))
    except OSError as exc:
        raise server_mod._fail(
            f"Could not write '{target}': {exc.strerror or exc}. "
            "Check that the session folder is writable and the disk has space.",
            500,
        )
    finally:
        # A refusal must not leave the reserved name behind: an empty
        # report_<n>.pdf would be skipped by every later probe and sit in the
        # folder as a report that does not open.
        if not written:
            target.unlink(missing_ok=True)

    token = state.register_file(target)
    return {
        "path": str(target),
        "filename": target.name,
        "download_url": f"/api/download/{token}",
        "sections": result["sections"],
        "pages": result["pages"],
        "size_bytes": result["size_bytes"],
        "generated_at": generated_at,
        "units": result["units"],
        "dataset_id": entry.id,
        "dataset_label": entry.label,
    }
