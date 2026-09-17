"""Project file endpoints: save the session, reopen it, start afresh, and
pick it up where it was left.

The library that writes and reads the file is :mod:`agrosuite.app.persist`;
this module only decides *where* a save goes, *whether* an existing file may
be replaced, and what to tell the user when either goes wrong. It also
remembers which file the session came from or last went to, so that
re-saving it asks nothing and reloading the page does not forget it, and it
looks at a file before the session is given up for it.

The app also saves by itself, into the projects folder
(:mod:`agrosuite.app.autosave`). Two ends of that live here: the start-up
reopen, which puts the last project back before anyone asks, and the view
state, which is what makes the reopened session look like the one that was
left rather than a list of layers on an empty map.

Every route here that hands a project file to the auto-saver claims it
first (:mod:`agrosuite.app.projectlock`), and every route that gives one up
lets it go. A second app — the launcher takes the next free port, so
double-clicking it twice gives two — therefore never reopens the project
the first one is writing: it starts empty and says so.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from .. import autosave as autosave_mod
from .. import persist as persist_mod
from .. import projectlock as lock_mod
from .. import session as session_mod
from .. import settings as settings_mod

router = APIRouter(prefix="/api/session", tags=["session"])


class SaveRequest(BaseModel):
    """Where to save, and what to call the project.

    ``name`` becomes the project's name as part of the save rather than
    through a separate request: a save that is refused must leave the
    project as it was, or the panel, the report title and the file would
    disagree about what the project is called.
    """

    name: str | None = None
    path: str | None = None
    #: Replacing an existing file is a decision, not a default: the interface
    #: asks first. The one exception is the file the session was opened from
    #: or last saved to, which is re-saved without asking.
    overwrite: bool = False


class OpenRequest(BaseModel):
    path: str


class ViewRequest(BaseModel):
    """Where the reader is: the dataset selected, the tab open.

    Both optional, because the interface sends whichever it knows: a tab is
    open before anything is loaded, and a dataset can be selected on any tab.
    """

    dataset_id: str | None = None
    tab: str | None = None


def _strip_quotes(raw: str) -> str:
    # Paths pasted from a file manager often arrive quoted.
    return raw.strip().strip('"').strip("'")


def _clean_path(raw: str) -> Path:
    return Path(_strip_quotes(raw)).expanduser()


def _save_target(chosen: Path, slug: str) -> Path:
    """The file a typed 'Save to' means, with its folder in place.

    One rule, the same in the dialog's hint: a path ending in ``.agrosuite``
    is the file; anything else is a folder, and the file inside it is named
    after the project. Whether the folder exists yet does not enter into it
    — the earlier rule looked, and a folder typed before it was created
    quietly became a file of that name in its parent. A missing folder is
    created instead: nothing can be overwritten by making one.

    Relative paths are refused. The server resolves them against wherever
    the app was started from, a place the person cannot see and did not
    choose; the file would land somewhere only a search would find.
    """
    from agrosuite.app import server as server_mod

    if not chosen.is_absolute():
        raise server_mod._fail(
            f"'{chosen}' is a relative path, and the app cannot tell where it is "
            "meant from. Give the full path of a folder, from the drive or the root "
            "(e.g. C:\\Data\\Projects or /home/you/projects)."
        )
    if chosen.suffix.lower() == persist_mod.EXTENSION:
        target, folder = chosen, chosen.parent
    else:
        target, folder = chosen / f"{slug}{persist_mod.EXTENSION}", chosen

    if folder.exists() and not folder.is_dir():
        raise server_mod._fail(
            f"'{folder}' is a file, not a folder. To save as a file, end the path in "
            f"{persist_mod.EXTENSION}; otherwise give a folder."
        )
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise server_mod._fail(
            f"Could not create the folder '{folder}': {exc.strerror or exc}. "
            "Check the path, and that its parent folder is writable."
        )
    return target


def _is_current(state: session_mod.Session, target: Path) -> bool:
    current = state.project_file
    return current is not None and current["path"] == target


def _add_to_recent(path: Path, saved_at: str | None) -> str | None:
    """Put ``path`` on the recent list; the warning to show if that failed.

    The list is a convenience kept under the home folder. A home folder that
    cannot be written — a read-only profile, a full disk, AGROSUITE_HOME
    pointing at a file — must not turn a save that succeeded, or an open
    that has already replaced the session, into an error the interface
    treats as a failure of the whole action.
    """
    try:
        persist_mod.remember_recent(path, saved_at)
    except OSError as exc:
        return (
            f"The file is fine, but the recent list could not be updated: {exc}. "
            "Point AGROSUITE_HOME at a writable folder to keep the list."
        )
    return None


def _joined(*parts: str | None) -> str | None:
    """The warnings that apply, in one string, or None when none do."""
    said = [part for part in parts if part]
    return "\n".join(said) if said else None


def _file_payload(state: session_mod.Session) -> dict[str, Any] | None:
    """The open file as the interface shows it, in one shape for every reply."""
    current = state.project_file
    if current is None:
        return None
    path: Path = current["path"]
    token = current.get("token")
    return {
        "path": str(path),
        "name": path.stem,
        "folder": str(path.parent),
        "saved_at": current.get("saved_at"),
        # A file in the session's own folder goes when the app closes; the
        # save dialog must not propose that folder as a place to keep work.
        "on_disk": not path.is_relative_to(state.workdir.resolve()),
        "download_url": f"/api/download/{token}" if token else None,
        # Decides what the next save proposes: a file named after the project
        # keeps following the name (the folder), one saved under a name of
        # its own is re-saved as it is (the full path).
        "named_after_project": path.stem == persist_mod.slugify(state.project["name"]),
    }


@router.post("/save")
def save_session(request: SaveRequest) -> dict[str, Any]:
    """Write the session to a ``.agrosuite`` file and remember it as recent."""
    from agrosuite.app import server as server_mod

    state = server_mod.state
    if not state.list():
        raise server_mod._fail(
            "Nothing to save yet: the session holds no datasets. Open a file first."
        )

    name = (request.name or "").strip() or state.project["name"]
    slug = persist_mod.slugify(name)
    chosen = _strip_quotes(request.path or "")
    if chosen:
        target = _save_target(Path(chosen).expanduser(), slug)
    else:
        target = state.exports / f"{slug}{persist_mod.EXTENSION}"
    # Resolved once, here: the status line, the recent list and the check
    # for "the file that is open" all compare paths, and '..', a trailing
    # separator or a symlinked folder would make one file look like two.
    target = target.resolve()

    # A file another window is auto-saving into is not refused here — a save
    # the person asked for by name is their decision — but it is never a
    # silent one: what lands there can be replaced by that window seconds
    # later, and the way out is a name of its own.
    taken = lock_mod.holder_of(target)
    contested = (
        f"'{target.stem}' is open in another AgroSuite window, which keeps writing that "
        "file: what is saved here can be replaced by it within seconds. Save under "
        "another name, or close the other window first."
    ) if taken is not None and taken.live and not taken.is_me() else None

    if target.exists() and not request.overwrite and not _is_current(state, target):
        raise server_mod._fail(
            f"'{target}' already exists. Choose another name, or confirm replacing it."
            + (f" {contested}" if contested else ""),
            409,
        )

    saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    previous_name = state.project["name"]
    state.project["name"] = name
    try:
        result = persist_mod.save_session(state, target, saved_at)
    except (OSError, ValueError) as exc:
        # Nothing was saved, so nothing changes — including the name.
        state.project["name"] = previous_name
        if isinstance(exc, OSError):
            raise server_mod._fail(
                f"Could not write '{target}': {exc.strerror or exc}. "
                "Check that the folder is writable and the disk has space."
            )
        raise server_mod._fail(str(exc))

    token = state.register_file(target)
    state.project_file = {"path": target, "saved_at": saved_at, "token": token}
    # The name given here is the project's, and the auto-saved file is named
    # after the project: a save under a new name renames that file too,
    # rather than leaving a copy of the work under the old one.
    if name != previous_name:
        state.touch()
    return {
        "path": str(target),
        "name": target.stem,
        "download_url": f"/api/download/{token}",
        "datasets": result["datasets"],
        "saved_at": saved_at,
        "size_bytes": result["size_bytes"],
        "file": _file_payload(state),
        "warning": _joined(_add_to_recent(target, saved_at), contested),
    }


def _peek(path: Path) -> dict[str, Any]:
    """What a project file holds, or why it cannot be opened — without
    loading it.

    Opening replaces the session, so the interface asks first; but a
    question about a path that will then fail ("closes the 3 datasets ...
    File not found") is worse than no question. The file browser puts the
    folder in the box when one is clicked, which made that the common case.
    The manifest is a small member of the ZIP, cheap to read on its own.
    """
    if path.is_dir():
        return {"ok": False, "reason": (
            f"That is a folder; choose the {persist_mod.EXTENSION} file inside it: {path}"
        )}
    if not path.exists():
        return {"ok": False, "reason": f"File not found: {path}"}
    try:
        manifest = persist_mod.read_manifest(path)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}
    except OSError as exc:
        return {"ok": False, "reason": (
            f"Could not read '{path.name}': {exc.strerror or exc}. Check that the file "
            "is readable and, on a removable drive, that the drive is still there."
        )}
    return {
        "ok": True,
        "path": str(path.resolve()),
        "name": path.stem,
        "project": (manifest.get("project") or {}).get("name"),
        "saved_at": manifest.get("saved_at"),
        "datasets": len(manifest["datasets"]),
    }


@router.get("/peek")
def peek_session(path: str) -> dict[str, Any]:
    """Describe a project file before opening it: ``{ok, name, project,
    saved_at, datasets}``, or ``{ok: false, reason}`` in words the person
    can act on. Never an error status: a refused path is an answer."""
    return _peek(_clean_path(path))


def _open(path: Path, remember: bool) -> dict[str, Any]:
    from agrosuite.app import server as server_mod

    state = server_mod.state
    if path.is_dir():
        raise server_mod._fail(
            f"That is a folder; choose the {persist_mod.EXTENSION} file inside it: {path}"
        )
    if not path.exists():
        raise server_mod._fail(f"File not found: {path}", 404)
    path = path.resolve()
    try:
        result = persist_mod.load_session(state, path)
    except ValueError as exc:
        raise server_mod._fail(str(exc))
    except Exception as exc:
        raise server_mod._fail(f"Could not open '{path.name}': {exc}")

    state.project_file = {"path": path, "saved_at": result["saved_at"], "token": None}
    locked = _adopt_for_autosave(state, path)
    # A project that has just been opened is a session the auto-saver has
    # never written. If it came from the projects folder that write goes
    # straight back to the same file; if it came from a USB stick or a
    # mailbox, this is what gives it a home on this machine.
    state.touch()
    return {
        "path": str(path),
        "name": path.stem,
        "saved_at": result["saved_at"],
        "project": result["project"],
        "datasets": result["datasets"],
        "design": result["design"],
        "view": result["view"],
        "file": _file_payload(state),
        "warning": _joined(_add_to_recent(path, result["saved_at"]) if remember else None,
                           locked),
        # Open, readable, editable — and not auto-saved, because the window
        # that has it is still writing it. Named on its own so the interface
        # can put it in the status line as well as in the toast.
        "locked": locked,
    }


def _adopt_for_autosave(state: session_mod.Session, path: Path) -> str | None:
    """Let the auto-saver carry on with the file that was just opened.

    A project opened out of the projects folder keeps writing to itself —
    the same file, one history, and the one the app offers to pick up next
    time. A project opened from anywhere else is left alone: writing an
    auto-save into somebody's Downloads folder, or onto a USB stick that
    will be pulled out, is not what the folder they chose is for, so the
    auto-saver gives it a file of its own on the next change.

    Adopting a file means writing to it, so it is claimed here. Returns the
    sentence to show when another window already has it: the project still
    opens — reading it harms nobody — but this window will not save into it,
    and the person has to be told that before they work in it for an hour.
    """
    try:
        folder = settings_mod.read().projects_dir
        inside = path.parent == folder.resolve()
    except OSError:
        inside = False

    if not inside:
        # Whatever was held belonged to the session that has just been
        # replaced; this one keeps its work somewhere else.
        lock_mod.release()
        state.autosave_path = None
        return None
    try:
        lock_mod.hold(path, state.project["name"])
    except lock_mod.Busy as busy:
        state.autosave_path = None
        return busy.holder.sentence()
    state.autosave_path = path
    return None


@router.post("/open")
def open_session(request: OpenRequest) -> dict[str, Any]:
    """Replace the session with a project file from this machine's disk."""
    return _open(_clean_path(request.path), remember=True)


@router.post("/upload")
async def upload_session(file: UploadFile = File(...)) -> dict[str, Any]:
    """Replace the session with a project file uploaded from the browser.

    The copy lands in the session's upload folder, which disappears with the
    session, so it is not added to the recent list: a recent entry that
    points nowhere after a restart would only mislead.
    """
    from agrosuite.app import server as server_mod

    if not file.filename:
        raise server_mod._fail("File has no name.")
    name = Path(file.filename).name
    if Path(name).suffix.lower() != persist_mod.EXTENSION:
        raise server_mod._fail(
            f"'{name}' is not a project file. Project files end in "
            f"{persist_mod.EXTENSION}; monitor data is opened from the Data tab."
        )

    state = server_mod.state
    target = state.uploads / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return _open(target, remember=False)


@router.get("/recent")
def recent_sessions() -> dict[str, Any]:
    """What a page needs the moment it loads.

    Recently saved or opened project files, most recent first; the file the
    session is on now, which is what lets a reloaded page know where the
    next save goes; where the reader was, so the page can come back to it;
    and whether the app reopened a project by itself when it started.

    That last one is handed over once and then forgotten: the toast it
    produces offers to start a new project instead, and offering that again
    every time the page is reloaded would be nagging about a decision
    already made.
    """
    from agrosuite.app import server as server_mod

    state = server_mod.state
    resumed, state.resumed = state.resumed, None
    return {
        "recent": persist_mod.recent_files(),
        "current": _file_payload(state),
        "view": dict(state.view),
        "resumed": resumed,
    }


@router.put("/view")
def set_view(request: ViewRequest) -> dict[str, Any]:
    """Record which dataset is selected and which tab is open.

    It is saved with the project, so a change here is a change to the
    session like any other. The interface sends it when it actually changes
    — not while the mouse moves — and identical values are not treated as a
    change at all, or clicking the same tab twice would write the file.
    """
    from agrosuite.app import server as server_mod

    state = server_mod.state
    view = dict(state.view)
    for key, value in (("dataset_id", request.dataset_id), ("tab", request.tab)):
        if value is not None:
            view[key] = value
    if view != state.view:
        state.view = view
        state.touch()
    return {"view": dict(state.view)}


@router.get("/autosave")
def autosave_status() -> dict[str, Any]:
    """Whether the session is saving itself, where, and how that is going.

    The status line in the panel reads this: "Saved 12:04", "Saving…", or
    the reason it could not and what to do about it.
    """
    from agrosuite.app import server as server_mod

    return autosave_mod.status(server_mod.state)


@router.get("/latest")
def latest_session() -> dict[str, Any]:
    """The project that would be picked up again, without loading it.

    ``available`` says there is one; ``resumable`` says the app would
    actually reopen it — which it only does into an empty session, and only
    while auto-save is on. The name, the layer count and the time it was
    saved come out of the manifest, so the interface can say which project
    it is offering before anything is read.
    """
    from agrosuite.app import server as server_mod

    state = server_mod.state
    config = settings_mod.read()
    candidate = autosave_mod.latest_project(config.projects_dir)
    holder = lock_mod.holder_of(candidate["path"]) if candidate else None
    locked = bool(holder and holder.live and not holder.is_me())
    return {
        "available": candidate is not None,
        # A project open in another window is not resumable here, and saying
        # so is the whole point of the key: the interface names it rather
        # than showing an empty app with no explanation.
        "resumable": bool(candidate and config.autosave and not state.list() and not locked),
        "folder": str(config.projects_dir),
        "project": candidate,
        "locked": locked,
        "lock": holder.payload() if locked and holder else None,
    }


def resume_latest() -> dict[str, Any] | None:
    """Reopen the last project, on start-up, into an empty session.

    This is what "pick it up where I left off" comes down to: the app is
    started by double-clicking, and the session it had is the one it should
    still have. It is done here rather than asked about in the interface,
    because a question at every start-up is a worse deal than a toast with
    'Start a new project instead' in it — one click to undo, and no click at
    all in the usual case.

    It never stops the app from starting. A folder that is not there, a file
    from a newer version, a damaged ZIP: each is reported on the session as
    something to say in the interface, and the app opens empty.
    """
    from agrosuite.app import server as server_mod

    state = server_mod.state
    config = settings_mod.read()
    if not config.autosave or state.list():
        return None

    found = autosave_mod.scan(config.projects_dir)
    candidate, skipped = found["project"], found["skipped"]
    if candidate is None:
        if not skipped:
            return None
        # Nothing in the folder opens. Starting empty and saying nothing
        # would look exactly like work that had gone.
        first = skipped[0]
        state.resumed = {"ok": False, "name": first["name"], "skipped": skipped, "reason": (
            f"The newest project in {config.projects_dir} could not be read: "
            f"{first['reason']} The app has started empty and the file was not "
            "changed — open it with 'Open project' to see the whole message."
        )}
        return state.resumed

    path = Path(candidate["path"])
    try:
        lock_mod.hold(path, candidate["project"])
    except lock_mod.Busy as busy:
        # The other window is writing this project right now. Reopening it
        # here would give two apps one file and the slower one's afternoon
        # to the rolling backup, so this one starts empty — usable, saving
        # under a name of its own as soon as there is anything to save.
        state.resumed = {
            "ok": False,
            "locked": True,
            "name": path.stem,
            "path": str(path),
            "project": candidate["project"],
            "lock": busy.holder.payload(),
            "reason": (
                f"{busy.holder.sentence()} This window has started empty, and that "
                "project was not touched."
            ),
        }
        return state.resumed

    try:
        result = persist_mod.load_session(state, path)
    except Exception as exc:
        # Claimed and not opened: the file is nobody's here, and holding it
        # would keep the other window out of a project this one never had.
        lock_mod.release()
        state.resumed = {"ok": False, "name": path.stem, "path": str(path), "reason": (
            f"'{path.name}' could not be reopened: {exc} The app has started empty; "
            "the file was not changed, and 'Open project' will say more about it."
        )}
        return state.resumed

    state.project_file = {"path": path, "saved_at": result["saved_at"], "token": None}
    # The file is its own auto-save from here on: the same file keeps the
    # project's history rather than a second one appearing beside it. It is
    # deliberately not marked as changed — what is on disk already says
    # exactly this, and writing it again would only move its clock.
    state.autosave_path = path
    state.resumed = {
        "ok": True,
        "path": str(path),
        "name": path.stem,
        "project": result["project"],
        "saved_at": result["saved_at"],
        "datasets": len(result["datasets"]),
        # Anything newer that could not be read. The project that came back
        # is the right one to offer, but a file the app walked past on the
        # way to it is the one the person will be looking for.
        "skipped": skipped,
    }
    return state.resumed


@router.post("/new")
def new_session() -> dict[str, Any]:
    """Drop every dataset and put the project back to its defaults.

    Files already generated for download stay reachable: they are on disk
    and the tokens cost nothing, whereas a link that stops working the moment
    a new project starts would surprise anyone who had just exported.

    The project that was open stays on disk exactly as it was. Starting a
    new one lets go of the file — ``Session.clear`` drops it — so the next
    auto-save writes a new file beside it rather than emptying that one:
    "new project" is not "delete the last one", and this is also the button
    the start-up toast offers, one click after the app reopened something.
    """
    from agrosuite.app import server as server_mod

    state = server_mod.state
    state.clear()
    state.project = session_mod.default_project()
    # The file that was being written is no longer this session's, so the
    # claim on it goes too: the other window — or the next launch — can pick
    # that project up straight away rather than waiting out a heartbeat.
    lock_mod.release()
    return {"ok": True}
